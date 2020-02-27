// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "src/servers/http_server_v2.h"

#include <event2/buffer.h>
#include <evhtp/evhtp.h>
#include <google/protobuf/text_format.h>
#include <re2/re2.h>
#include <algorithm>
#include <thread>
#include "src/core/api.pb.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/server_status.pb.h"
#include "src/core/trtserver.h"
#include "src/servers/common.h"

#ifdef TRTIS_ENABLE_TRACING
#include "src/servers/tracer.h"
#endif  // TRTIS_ENABLE_TRACING

namespace nvidia { namespace inferenceserver {

// Generic HTTP server using evhtp
class HTTPServerV2Impl : public HTTPServerV2 {
 public:
  explicit HTTPServerV2Impl(const int32_t port, const int thread_cnt)
      : port_(port), thread_cnt_(thread_cnt)
  {
  }

  virtual ~HTTPServerV2Impl() { Stop(); }

  static void Dispatch(evhtp_request_t* req, void* arg);

  TRTSERVER_Error* Start() override;
  TRTSERVER_Error* Stop() override;

 protected:
  virtual void Handle(evhtp_request_t* req) = 0;

  static void StopCallback(int sock, short events, void* arg);

  int32_t port_;
  int thread_cnt_;

  evhtp_t* htp_;
  struct event_base* evbase_;
  std::thread worker_;
  int fds_[2];
  event* break_ev_;
};

TRTSERVER_Error*
HTTPServerV2Impl::Start()
{
  if (!worker_.joinable()) {
    evbase_ = event_base_new();
    htp_ = evhtp_new(evbase_, NULL);
    evhtp_enable_flag(htp_, EVHTP_FLAG_ENABLE_NODELAY);
    evhtp_set_gencb(htp_, HTTPServerV2Impl::Dispatch, this);
    evhtp_use_threads_wexit(htp_, NULL, NULL, thread_cnt_, NULL);
    evhtp_bind_socket(htp_, "0.0.0.0", port_, 1024);
    // Set listening event for breaking event loop
    evutil_socketpair(AF_UNIX, SOCK_STREAM, 0, fds_);
    break_ev_ = event_new(evbase_, fds_[0], EV_READ, StopCallback, evbase_);
    event_add(break_ev_, NULL);
    worker_ = std::thread(event_base_loop, evbase_, 0);
    return nullptr;
  }

  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_ALREADY_EXISTS, "HTTP V2 server is already running.");
}

TRTSERVER_Error*
HTTPServerV2Impl::Stop()
{
  if (worker_.joinable()) {
    // Notify event loop to break via fd write
    send(fds_[1], &evbase_, sizeof(event_base*), 0);
    worker_.join();
    event_free(break_ev_);
    evutil_closesocket(fds_[0]);
    evutil_closesocket(fds_[1]);
    evhtp_unbind_socket(htp_);
    evhtp_free(htp_);
    event_base_free(evbase_);
    return nullptr;
  }

  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNAVAILABLE, "HTTP V2 server is not running.");
}

void
HTTPServerV2Impl::StopCallback(int sock, short events, void* arg)
{
  struct event_base* base = (struct event_base*)arg;
  event_base_loopbreak(base);
}

void
HTTPServerV2Impl::Dispatch(evhtp_request_t* req, void* arg)
{
  (static_cast<HTTPServerV2Impl*>(arg))->Handle(req);
}


#ifdef TRTIS_ENABLE_METRICS

// Handle HTTP requests to obtain prometheus metrics
class HTTPMetricsServer : public HTTPServerImpl {
 public:
  explicit HTTPMetricsServer(
      const std::shared_ptr<TRTSERVER_Server>& server, const int32_t port,
      const int thread_cnt)
      : HTTPServerImpl(port, thread_cnt), server_(server),
        api_regex_(R"(/metrics/?)")
  {
  }

  ~HTTPMetricsServer() = default;

 private:
  void Handle(evhtp_request_t* req) override;

  std::shared_ptr<TRTSERVER_Server> server_;
  re2::RE2 api_regex_;
};

void
HTTPMetricsServer::Handle(evhtp_request_t* req)
{
  LOG_VERBOSE(1) << "HTTP request: " << req->method << " "
                 << req->uri->path->full;

  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  evhtp_res res = EVHTP_RES_BADREQ;

  // Call to metric endpoint should not have any trailing string
  if (RE2::FullMatch(std::string(req->uri->path->full), api_regex_)) {
    TRTSERVER_Metrics* metrics = nullptr;
    TRTSERVER_Error* err = TRTSERVER_ServerMetrics(server_.get(), &metrics);
    if (err == nullptr) {
      const char* base;
      size_t byte_size;
      err = TRTSERVER_MetricsFormatted(
          metrics, TRTSERVER_METRIC_PROMETHEUS, &base, &byte_size);
      if (err == nullptr) {
        res = EVHTP_RES_OK;
        evbuffer_add(req->buffer_out, base, byte_size);
      }
    }

    TRTSERVER_MetricsDelete(metrics);
    TRTSERVER_ErrorDelete(err);
  }

  evhtp_send_reply(req, res);
}

#endif  // TRTIS_ENABLE_METRICS

// Handle HTTP requests to inference server APIs
class HTTPAPIServerV2 : public HTTPServerV2Impl {
 public:
  explicit HTTPAPIServerV2(
      const std::shared_ptr<TRTSERVER_Server>& server,
      const std::shared_ptr<nvidia::inferenceserver::TraceManager>&
          trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      const std::vector<std::string>& endpoints, const int32_t port,
      const int thread_cnt)
      : HTTPServerV2Impl(port, thread_cnt), server_(server),
        trace_manager_(trace_manager), smb_manager_(smb_manager),
        allocator_(nullptr), api_regex_(R"(/v2/(health|models|)(.*))"),
        health_regex_(R"(/(live|ready))"),
        model_regex_(R"(/([^/]+)(/version/(0-9)+)?(/infer|ready)?)")
  {
    TRTSERVER_Error* err = TRTSERVER_ServerId(server_.get(), &server_id_);
    if (err != nullptr) {
      server_id_ = "unknown:0";
      TRTSERVER_ErrorDelete(err);
    }

    FAIL_IF_ERR(
        TRTSERVER_ResponseAllocatorNew(
            &allocator_, InferResponseAlloc, ResponseRelease),
        "creating response allocator");
  }

  ~HTTPAPIServerV2()
  {
    LOG_TRTSERVER_ERROR(
        TRTSERVER_ResponseAllocatorDelete(allocator_),
        "deleting response allocator");
  }

  //
  // AllocPayload
  //
  // Simple structure that carries the userp payload needed for
  // allocation. These are just pointers into a HandlerState
  // object. HandlerState lifetime is always longer than what is
  // required for allocation callback so HandlerState manages the
  // lifetime of the actual objects referenced by those pointers.
  //
  struct ShmInfo {
    void* base_;
    size_t byte_size_;
    TRTSERVER_Memory_Type memory_type_;
    int64_t device_id_;
  };

  using TensorShmMap = std::unordered_map<std::string, ShmInfo>;

  struct AllocPayload {
    explicit AllocPayload() : shm_map_(nullptr) {}
    ~AllocPayload()
    {
      // Don't delete 'response_'.. it is owned by the HandlerState
      delete shm_map_;
    }

    std::vector<evbuffer*> response_buffer_;
    TensorShmMap* shm_map_;
  };

  // Class object associated to evhtp thread, requests received are bounded
  // with the thread that accepts it. Need to keep track of that and let the
  // corresponding thread send back the reply
  class InferRequestClass {
   public:
    InferRequestClass(
        evhtp_request_t* req, uint64_t request_id, const char* server_id,
        uint64_t unique_id);

    ~InferRequestClass()
    {
      for (auto buffer : response_meta_data_.response_buffer_) {
        if (buffer != nullptr) {
          evbuffer_free(buffer);
        }
      }
    }

    evhtp_request_t* EvHtpRequest() const { return req_; }

    static void InferComplete(
        TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
        TRTSERVER_InferenceResponse* response, void* userp);
    evhtp_res FinalizeResponse(TRTSERVER_InferenceResponse* response);

#ifdef TRTIS_ENABLE_TRACING
    std::unique_ptr<TraceMetaData> trace_meta_data_;
#endif  // TRTIS_ENABLE_TRACING

    AllocPayload response_meta_data_;

   private:
    evhtp_request_t* req_;
    evthr_t* thread_;
    const uint64_t request_id_;
    const char* const server_id_;
    const uint64_t unique_id_;
  };

 private:
  static TRTSERVER_Error* InferResponseAlloc(
      TRTSERVER_ResponseAllocator* allocator, const char* tensor_name,
      size_t byte_size, TRTSERVER_Memory_Type preferred_memory_type,
      int64_t preferred_memory_type_id, void* userp, void** buffer,
      void** buffer_userp, TRTSERVER_Memory_Type* actual_memory_type,
      int64_t* actual_memory_type_id);
  static TRTSERVER_Error* ResponseRelease(
      TRTSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
      size_t byte_size, TRTSERVER_Memory_Type memory_type,
      int64_t memory_type_id);

  void Handle(evhtp_request_t* req) override;
  void HandleHealth(evhtp_request_t* req, const std::string& kind);
  void HandleModelHealth(
      evhtp_request_t* req, const std::string& model_name,
      const std::string& model_version_str);
  void HandleMetadata(evhtp_request_t* req);
  void HandleModelMetadata(
      evhtp_request_t* req, const std::string& model_name,
      const std::string& model_version_str);
  void HandleInfer(
      evhtp_request_t* req, const std::string& model_name,
      const std::string& model_version_str);

#ifdef TRTIS_ENABLE_GPU
  TRTSERVER_Error* EVBufferToCudaHandle(
      evbuffer* handle_buffer, cudaIpcMemHandle_t** cuda_shm_handle);
#endif  // TRTIS_ENABLE_GPU
  TRTSERVER_Error* EVBufferToInput(
      const std::string& model_name, const InferRequestHeader& request_header,
      evbuffer* input_buffer,
      TRTSERVER_InferenceRequestProvider* request_provider,
      TensorShmMap* shm_map);

  static void OKReplyCallback(evthr_t* thr, void* arg, void* shared);
  static void BADReplyCallback(evthr_t* thr, void* arg, void* shared);

  std::shared_ptr<TRTSERVER_Server> server_;
  const char* server_id_;

  std::shared_ptr<TraceManager> trace_manager_;
  std::shared_ptr<SharedMemoryManager> shm_manager_;

  // The allocator that will be used to allocate buffers for the
  // inference result tensors.
  TRTSERVER_ResponseAllocator* allocator_;

  re2::RE2 api_regex_;
  re2::RE2 health_regex_;
  re2::RE2 model_regex_;
};

TRTSERVER_Error*
HTTPAPIServerV2::InferResponseAlloc(
    TRTSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRTSERVER_Memory_Type preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRTSERVER_Memory_Type* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  AllocPayload* payload = reinterpret_cast<AllocPayload*>(userp);
  evbuffer* evhttp_buffer = evbuffer_new();
  if (evhttp_buffer == nullptr) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INTERNAL,
        std::string("failed to create evbuffer for output tensor").c_str());
  } else {
    payload->response_buffer_.push_back(evhttp_buffer);
  }

  const TensorShmMap* shm_map = payload->shm_map_;

  *buffer = nullptr;
  *buffer_userp = nullptr;
  *actual_memory_type = preferred_memory_type;
  *actual_memory_type_id = preferred_memory_type_id;

  // Don't need to do anything if no memory was requested.
  if (byte_size > 0) {
    bool use_shm = false;

    if (shm_map != nullptr) {
      const auto& pr = shm_map->find(tensor_name);
      if (pr != shm_map->end()) {
        // If the output is in shared memory then check that the expected
        // requested buffer size is at least the byte size of the output.
        if (byte_size > pr->second.byte_size_) {
          return TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_INTERNAL,
              std::string(
                  "expected buffer size to be at least " +
                  std::to_string(pr->second.byte_size_) + " bytes but got " +
                  std::to_string(byte_size) + " bytes in output tensor")
                  .c_str());
        }

        *buffer = const_cast<void*>(pr->second.base_);
        *actual_memory_type = pr->second.memory_type_;
        *actual_memory_type_id = pr->second.device_id_;
        use_shm = true;

        LOG_VERBOSE(1) << "HTTP: using shared-memory for '" << tensor_name
                       << "', size: " << byte_size << ", addr: " << *buffer;
      }
    }

    if (!use_shm) {
      // Can't allocate for any memory type other than CPU. If asked to
      // allocate on GPU memory then force allocation on CPU instead.
      if (*actual_memory_type != TRTSERVER_MEMORY_CPU) {
        LOG_VERBOSE(1) << "HTTP: unable to provide '" << tensor_name << "' in "
                       << MemoryTypeString(*actual_memory_type) << ", will use "
                       << MemoryTypeString(TRTSERVER_MEMORY_CPU);
        *actual_memory_type = TRTSERVER_MEMORY_CPU;
        *actual_memory_type_id = 0;
      }

      // Reserve requested space in evbuffer...
      struct evbuffer_iovec output_iovec;
      if (evbuffer_reserve_space(evhttp_buffer, byte_size, &output_iovec, 1) !=
          1) {
        return TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INTERNAL,
            std::string(
                "failed to reserve " + std::to_string(byte_size) +
                " bytes in output tensor buffer")
                .c_str());
      }

      if (output_iovec.iov_len < byte_size) {
        return TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INTERNAL,
            std::string(
                "reserved " + std::to_string(output_iovec.iov_len) +
                " bytes in output tensor buffer, need " +
                std::to_string(byte_size))
                .c_str());
      }

      output_iovec.iov_len = byte_size;
      *buffer = output_iovec.iov_base;

      // Immediately commit the buffer space. We are relying on evbuffer
      // not to relocate this space. Because we request a contiguous
      // chunk every time (above by allowing only a single entry in
      // output_iovec), this seems to be a valid assumption.
      if (evbuffer_commit_space(evhttp_buffer, &output_iovec, 1) != 0) {
        *buffer = nullptr;
        return TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INTERNAL,
            "failed to commit output tensors to output buffer");
      }

      LOG_VERBOSE(1) << "HTTP using buffer for: '" << tensor_name
                     << "', size: " << byte_size << ", addr: " << *buffer;
    }
  }

  return nullptr;  // Success
}

TRTSERVER_Error*
HTTPAPIServerV2::ResponseRelease(
    TRTSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRTSERVER_Memory_Type memory_type, int64_t memory_type_id)
{
  LOG_VERBOSE(1) << "HTTP release: "
                 << "size " << byte_size << ", addr " << buffer;

  // Don't do anything when releasing a buffer since ResponseAlloc
  // wrote directly into the response ebvuffer.
  return nullptr;  // Success
}

void
HTTPAPIServerV2::Handle(evhtp_request_t* req)
{
  LOG_VERBOSE(1) << "HTTP V2 request: " << req->method << " "
                 << req->uri->path->full;

  std::string type, rest;
  if (RE2::FullMatch(
          std::string(req->uri->path->full), api_regex_, &type, &rest)) {
    // server health
    if (type == "health") {
      std::string kind;
      if (RE2::FullMatch(
              std::string(rest), health_regex_, &kind)) {
        HandleHealth(req, kind);
        return;
      }
    }
    // model health, status or infer
    else if (type == "models") {
      std::string model_name, model_version, kind;
      LOG_VERBOSE(1) << "type: " << type;
      LOG_VERBOSE(1) << "rest: " << rest;
      if (RE2::FullMatch(
              std::string(rest), model_regex_, &model_name,
              &model_version, &kind)) {
        if (kind == "ready") {
          HandleModelHealth(req, model_name, model_version);
          return;
        } else if (kind == "infer") {
          HandleInfer(req, model_name, model_version);
          return;
        } else {
          HandleModelMetadata(req, model_name, model_version);
          return;
        }
      }
      LOG_VERBOSE(1) << "model_name" << model_name;
    }
    // server metadata
    else {
      HandleMetadata(req);
      return;
    }
  }

  LOG_VERBOSE(1) << "HTTP V2 error: " << req->method << " "
                 << req->uri->path->full << " - "
                 << static_cast<int>(EVHTP_RES_BADREQ);
  evhtp_send_reply(req, EVHTP_RES_BADREQ);
}

void
HTTPAPIServerV2::HandleHealth(
    evhtp_request_t* req, const std::string& kind)
{
  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  TRTSERVER_Error* err = nullptr;
  bool health = false;

  if (kind == "live") {
    err = TRTSERVER_ServerIsLive(server_.get(), &health);
  }
  else {
    err = TRTSERVER_ServerIsReady(server_.get(), &health);
  }

  RequestStatus request_status;
  RequestStatusUtil::Create(
      &request_status, err, RequestStatusUtil::NextUniqueRequestId(),
      server_id_);

  evhtp_headers_add_header(
      req->headers_out,
      evhtp_header_new(
          kStatusHTTPHeader, request_status.ShortDebugString().c_str(), 1, 1));

  evhtp_send_reply(
      req, (health && (err == nullptr)) ? EVHTP_RES_OK : EVHTP_RES_BADREQ);

  TRTSERVER_ErrorDelete(err);
}

void
HTTPAPIServerV2::HandleModelHealth(
    evhtp_request_t* req, const std::string& model_name,
    const std::string& model_version_str)
{
  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  if (model_name.empty()) {
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    return;
  }

  bool ready = false;
  TRTSERVER_Protobuf* model_status_protobuf = nullptr;
  TRTSERVER_Error* err = TRTSERVER_ServerModelStatus(
      server_.get(), model_name.c_str(), &model_status_protobuf);
  if (err == nullptr) {
    const char* status_buffer;
    size_t status_byte_size;
    err = TRTSERVER_ProtobufSerialize(
        model_status_protobuf, &status_buffer, &status_byte_size);
    if (err == nullptr) {
      ServerStatus server_status;
      if (!server_status.ParseFromArray(status_buffer, status_byte_size)) {
        err = TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_UNKNOWN, "failed to parse server status");
      } else {
        const auto& itr = server_status.model_status().find(model_name);
        if (itr == server_status.model_status().end()) {
          err = TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_INTERNAL,
              std::string("unable to find health of \"" + model_name + "\"")
                  .c_str());
        } else {
          auto model_status = itr->second;

          // If requested version is -1 then find the highest valued
          // version.
          int64_t requested_version = -1;
          if (!model_version_str.empty()) {
            requested_version = std::atoll(model_version_str.c_str());
          }

          if (requested_version == -1) {
            for (const auto& pr : model_status.version_status()) {
              requested_version = std::max(requested_version, pr.first);
            }
          }

          const auto& vitr =
              model_status.version_status().find(requested_version);
          if (vitr == model_status.version_status().end()) {
            err = TRTSERVER_ErrorNew(
                TRTSERVER_ERROR_INVALID_ARG,
                std::string(
                    "no status available for model '" + model_name +
                    "', version " + model_version_str).c_str());
          } else {
            const ModelVersionStatus& version_status = vitr->second;
            ready = version_status.ready_state() == ModelReadyState::MODEL_READY;
          }
        }
      }
    }
  }

  TRTSERVER_ProtobufDelete(model_status_protobuf);
  RequestStatus request_status;
  RequestStatusUtil::Create(
      &request_status, err, RequestStatusUtil::NextUniqueRequestId(),
      server_id_);

  evhtp_headers_add_header(
      req->headers_out,
      evhtp_header_new(
          kStatusHTTPHeader, request_status.ShortDebugString().c_str(), 1, 1));

  evhtp_send_reply(
      req, (ready && (err == nullptr)) ? EVHTP_RES_OK : EVHTP_RES_BADREQ);

  TRTSERVER_ErrorDelete(err);
}

void
HTTPAPIServerV2::HandleModelMetadata(
    evhtp_request_t* req, const std::string& model_name,
    const std::string& model_version_str)
{
  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  if (model_name.empty()) {
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    return;
  }

  TRTSERVER_Protobuf* model_status_protobuf = nullptr;
  TRTSERVER_Error* err = TRTSERVER_ServerModelStatus(
                server_.get(), model_name.c_str(), &model_status_protobuf);
  if (err == nullptr) {
    const char* status_buffer;
    size_t status_byte_size;
    err = TRTSERVER_ProtobufSerialize(
        model_status_protobuf, &status_buffer, &status_byte_size);
    if (err == nullptr) {
      // Request text or binary format for status?
      std::string format;
      const char* format_c_str = evhtp_kv_find(req->uri->query, "format");
      if (format_c_str != NULL) {
        format = std::string(format_c_str);
      } else {
        format = "text";
      }

      if (format == "binary") {
        evbuffer_add(req->buffer_out, status_buffer, status_byte_size);
        evhtp_headers_add_header(
            req->headers_out,
            evhtp_header_new("Content-Type", "application/octet-stream", 1, 1));
      } else {
        ServerStatus server_status;
        if (!server_status.ParseFromArray(status_buffer, status_byte_size)) {
          err = TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_UNKNOWN, "failed to parse server status");
        } else {
          std::string server_status_str = server_status.DebugString();
          evbuffer_add(
              req->buffer_out, server_status_str.c_str(),
              server_status_str.size());
        }
      }
    }
  }

  TRTSERVER_ProtobufDelete(model_status_protobuf);

  RequestStatus request_status;
  RequestStatusUtil::Create(
      &request_status, err, RequestStatusUtil::NextUniqueRequestId(),
      server_id_);

  evhtp_headers_add_header(
      req->headers_out,
      evhtp_header_new(
          kStatusHTTPHeader, request_status.ShortDebugString().c_str(), 1, 1));

  evhtp_send_reply(
      req, (request_status.code() == RequestStatusCode::SUCCESS)
               ? EVHTP_RES_OK
               : EVHTP_RES_BADREQ);

  TRTSERVER_ErrorDelete(err);
}

void
HTTPAPIServerV2::HandleMetadata(evhtp_request_t* req)
{
  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  TRTSERVER_Protobuf* server_status_protobuf = nullptr;
  TRTSERVER_Error* err =
          TRTSERVER_ServerStatus(server_.get(), &server_status_protobuf);
  if (err == nullptr) {
    const char* status_buffer;
    size_t status_byte_size;
    err = TRTSERVER_ProtobufSerialize(
        server_status_protobuf, &status_buffer, &status_byte_size);
    if (err == nullptr) {
      // Request text or binary format for status?
      std::string format;
      const char* format_c_str = evhtp_kv_find(req->uri->query, "format");
      if (format_c_str != NULL) {
        format = std::string(format_c_str);
      } else {
        format = "text";
      }

      if (format == "binary") {
        evbuffer_add(req->buffer_out, status_buffer, status_byte_size);
        evhtp_headers_add_header(
            req->headers_out,
            evhtp_header_new("Content-Type", "application/octet-stream", 1, 1));
      } else {
        ServerStatus server_status;
        if (!server_status.ParseFromArray(status_buffer, status_byte_size)) {
          err = TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_UNKNOWN, "failed to parse server status");
        } else {
          std::string server_status_str = server_status.DebugString();
          evbuffer_add(
              req->buffer_out, server_status_str.c_str(),
              server_status_str.size());
        }
      }
    }
  }

  TRTSERVER_ProtobufDelete(server_status_protobuf);

  RequestStatus request_status;
  RequestStatusUtil::Create(
      &request_status, err, RequestStatusUtil::NextUniqueRequestId(),
      server_id_);

  evhtp_headers_add_header(
      req->headers_out,
      evhtp_header_new(
          kStatusHTTPHeader, request_status.ShortDebugString().c_str(), 1, 1));

  evhtp_send_reply(
      req, (request_status.code() == RequestStatusCode::SUCCESS)
               ? EVHTP_RES_OK
               : EVHTP_RES_BADREQ);

  TRTSERVER_ErrorDelete(err);
}

TRTSERVER_Error*
HTTPAPIServerV2::EVBufferToInput(
    const std::string& model_name, const InferRequestHeader& request_header,
    evbuffer* input_buffer,
    TRTSERVER_InferenceRequestProvider* request_provider, TensorShmMap* shm_map)
{
  // Extract individual input data from HTTP body and register in
  // 'request_provider'. The input data from HTTP body is not
  // necessarily contiguous so may need to register multiple input
  // "blocks" for a given input.
  //
  // Get the addr and size of each chunk of input data from the
  // evbuffer.
  struct evbuffer_iovec* v = nullptr;
  int v_idx = 0;

  int n = evbuffer_peek(input_buffer, -1, NULL, NULL, 0);
  if (n > 0) {
    v = static_cast<struct evbuffer_iovec*>(
        alloca(sizeof(struct evbuffer_iovec) * n));
    if (evbuffer_peek(input_buffer, -1, NULL, v, n) != n) {
      return TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_INTERNAL, "unexpected error getting input buffers ");
    }
  }

  // Get the byte-size for each input and from that get the blocks
  // holding the data for that input
  for (const auto& io : request_header.input()) {
    uint64_t byte_size = 0;
    RETURN_IF_ERR(TRTSERVER_InferenceRequestProviderInputBatchByteSize(
        request_provider, io.name().c_str(), &byte_size));

    // If 'byte_size' is zero then need to add an empty input data
    // block... the provider expects at least one data block for every
    // input.
    if (byte_size == 0) {
      RETURN_IF_ERR(TRTSERVER_InferenceRequestProviderSetInputData(
          request_provider, io.name().c_str(), nullptr, 0 /* byte_size */,
          TRTSERVER_MEMORY_CPU, 0 /* memory_type_id */));
    } else {
      // If input is in shared memory then verify that the size is
      // correct and set input from the shared memory.
#if 0
      if (io.has_shared_memory()) {
        if (byte_size != io.shared_memory().byte_size()) {
          return TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_INVALID_ARG,
              std::string(
                  "unexpected shared-memory size " +
                  std::to_string(io.shared_memory().byte_size()) +
                  " for input '" + io.name() + "', expecting " +
                  std::to_string(byte_size) + " for model '" + model_name + "'")
                  .c_str());
        }

        void* base;
        TRTSERVER_Memory_Type memory_type = TRTSERVER_MEMORY_CPU;
        int64_t memory_type_id;
        RETURN_IF_ERR(shm_manager_->GetMemoryInfo(
            io.shared_memory().name(), io.shared_memory().offset(), &base,
            &memory_type, &memory_type_id));
        RETURN_IF_ERR(TRTSERVER_InferenceRequestProviderSetInputData(
            request_provider, io.name().c_str(), base, byte_size, memory_type,
            memory_type_id));
      }
      else {
#endif
      // FIXMEV2 handle non-raw content types
      while ((byte_size > 0) && (v_idx < n)) {
        char* base = static_cast<char*>(v[v_idx].iov_base);
        size_t base_size;
        if (v[v_idx].iov_len > byte_size) {
          base_size = byte_size;
          v[v_idx].iov_base = static_cast<void*>(base + byte_size);
          v[v_idx].iov_len -= byte_size;
          byte_size = 0;
        } else {
          base_size = v[v_idx].iov_len;
          byte_size -= v[v_idx].iov_len;
          v_idx++;
        }

        RETURN_IF_ERR(TRTSERVER_InferenceRequestProviderSetInputData(
            request_provider, io.name().c_str(), base, base_size,
            TRTSERVER_MEMORY_CPU, 0 /* memory_type_id */));
      }

      if (byte_size != 0) {
        return TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INVALID_ARG,
            std::string(
                "unexpected size for input '" + io.name() + "', expecting " +
                std::to_string(byte_size) + " bytes for model '" + model_name +
                "'")
                .c_str());
      }
#if 0
      }
#endif
    }
  }

  if (v_idx != n) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected additional input data for model '" + model_name + "'")
            .c_str());
  }

#if 0
  // Initialize System Memory for Output if it uses shared memory
  for (const auto& io : request_header.output()) {
    if (io.has_shared_memory()) {
      void* base;
      TRTSERVER_Memory_Type memory_type;
      int64_t memory_type_id;
      RETURN_IF_ERR(shm_manager_->GetMemoryInfo(
          io.shared_memory().name(), io.shared_memory().offset(), &base,
          &memory_type, &memory_type_id));

      output_shm_map.emplace(
          io.name(),
          std::make_tuple(
              static_cast<const void*>(base), io.shared_memory().byte_size(),
              memory_type, memory_type_id));
    }
  }
#endif

  return nullptr;  // success
}

void
HTTPAPIServerV2::HandleInfer(
    evhtp_request_t* req, const std::string& model_name,
    const std::string& model_version_str)
{
  if (req->method != htp_method_POST) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  int64_t model_version = -1;
  if (!model_version_str.empty()) {
    model_version = std::atoll(model_version_str.c_str());
  }

#ifdef TRTIS_ENABLE_TRACING
  // Timestamps from evhtp are capture in 'req'. We record here since
  // this is the first place where we have a tracer.
  std::unique_ptr<TraceMetaData> trace_meta_data;
  if (trace_manager_ != nullptr) {
    trace_meta_data.reset(trace_manager_->SampleTrace());
    if (trace_meta_data != nullptr) {
      trace_meta_data->tracer_->SetModel(model_name, model_version);
      trace_meta_data->tracer_->CaptureTimestamp(
          TRTSERVER_TRACE_LEVEL_MIN, "http recv start",
          TIMESPEC_TO_NANOS(req->recv_start_ts));
      trace_meta_data->tracer_->CaptureTimestamp(
          TRTSERVER_TRACE_LEVEL_MIN, "http recv end",
          TIMESPEC_TO_NANOS(req->recv_end_ts));
    }
  }
#endif  // TRTIS_ENABLE_TRACING

  std::string infer_request_header(
      evhtp_kv_find(req->headers_in, kInferRequestHTTPHeader));

  InferRequestHeader request_header_protobuf;
  if (!google::protobuf::TextFormat::ParseFromString(
          infer_request_header, &request_header_protobuf)) {
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    return;
  }

  uint64_t unique_id = RequestStatusUtil::NextUniqueRequestId();

  // Create the inference request provider which provides all the
  // input information needed for an inference.
  TRTSERVER_InferenceRequestOptions* request_options = nullptr;
  TRTSERVER_Error* err = TRTSERVER_InferenceRequestOptionsNew(
      &request_options, model_name.c_str(), model_version);
  if (err == nullptr) {
    err = SetTRTSERVER_InferenceRequestOptions(
        request_options, request_header_protobuf);
  }
  TRTSERVER_InferenceRequestProvider* request_provider = nullptr;
  if (err == nullptr) {
    err = TRTSERVER_InferenceRequestProviderNewV2(
        &request_provider, server_.get(), request_options);
  }
  if (err == nullptr) {
    std::unique_ptr<InferRequestClass> infer_request(new InferRequestClass(
        req, request_header_protobuf.id(), server_id_, unique_id));
    err = EVBufferToInput(
        model_name, request_header_protobuf, req->buffer_in, request_provider,
        infer_request->response_meta_data_.shm_map_);
    if (err == nullptr) {
      // Provide the trace manager object to use for this request, if nullptr
      // then no tracing will be performed.
      TRTSERVER_TraceManager* trace_manager = nullptr;
#ifdef TRTIS_ENABLE_TRACING
      if (trace_meta_data != nullptr) {
        infer_request->trace_meta_data_ = std::move(trace_meta_data);
        TRTSERVER_TraceManagerNew(
            &trace_manager, TraceManager::CreateTrace,
            TraceManager::ReleaseTrace, infer_request->trace_meta_data_.get());
      }
#endif  // TRTIS_ENABLE_TRACING

      err = TRTSERVER_ServerInferAsync(
          server_.get(), trace_manager, request_provider, allocator_,
          reinterpret_cast<void*>(&infer_request->response_meta_data_),
          InferRequestClass::InferComplete,
          reinterpret_cast<void*>(infer_request.get()));
      if (err == nullptr) {
        infer_request.release();
      }
    }
  }

  // The request provider can be deleted before ServerInferAsync
  // callback completes.
  TRTSERVER_InferenceRequestProviderDelete(request_provider);
  TRTSERVER_InferenceRequestOptionsDelete(request_options);

  if (err != nullptr) {
    RequestStatus request_status;
    RequestStatusUtil::Create(&request_status, err, unique_id, server_id_);

    InferResponseHeader response_header;
    response_header.set_id(request_header_protobuf.id());
    evhtp_headers_add_header(
        req->headers_out,
        evhtp_header_new(
            kInferResponseHTTPHeader,
            response_header.ShortDebugString().c_str(), 1, 1));
    LOG_VERBOSE(1) << "Infer failed: " << request_status.msg();

    evhtp_headers_add_header(
        req->headers_out, evhtp_header_new(
                              kStatusHTTPHeader,
                              request_status.ShortDebugString().c_str(), 1, 1));
    evhtp_headers_add_header(
        req->headers_out,
        evhtp_header_new("Content-Type", "application/octet-stream", 1, 1));

    evhtp_send_reply(
        req, (request_status.code() == RequestStatusCode::SUCCESS)
                 ? EVHTP_RES_OK
                 : EVHTP_RES_BADREQ);
  }

  TRTSERVER_ErrorDelete(err);
}

void
HTTPAPIServerV2::OKReplyCallback(evthr_t* thr, void* arg, void* shared)
{
  HTTPAPIServerV2::InferRequestClass* infer_request =
      reinterpret_cast<HTTPAPIServerV2::InferRequestClass*>(arg);

  evhtp_request_t* request = infer_request->EvHtpRequest();
  evhtp_send_reply(request, EVHTP_RES_OK);
  evhtp_request_resume(request);

#ifdef TRTIS_ENABLE_TRACING
  if (infer_request->trace_meta_data_ != nullptr) {
    infer_request->trace_meta_data_->tracer_->CaptureTimestamp(
        TRTSERVER_TRACE_LEVEL_MIN, "http send start",
        TIMESPEC_TO_NANOS(request->send_start_ts));
    infer_request->trace_meta_data_->tracer_->CaptureTimestamp(
        TRTSERVER_TRACE_LEVEL_MIN, "http send end",
        TIMESPEC_TO_NANOS(request->send_end_ts));
  }
#endif  // TRTIS_ENABLE_TRACING

  delete infer_request;
}

void
HTTPAPIServerV2::BADReplyCallback(evthr_t* thr, void* arg, void* shared)
{
  HTTPAPIServerV2::InferRequestClass* infer_request =
      reinterpret_cast<HTTPAPIServerV2::InferRequestClass*>(arg);

  evhtp_request_t* request = infer_request->EvHtpRequest();
  evhtp_send_reply(request, EVHTP_RES_BADREQ);
  evhtp_request_resume(request);

#ifdef TRTIS_ENABLE_TRACING
  if (infer_request->trace_meta_data_ != nullptr) {
    infer_request->trace_meta_data_->tracer_->CaptureTimestamp(
        TRTSERVER_TRACE_LEVEL_MIN, "http send start",
        TIMESPEC_TO_NANOS(request->send_start_ts));
    infer_request->trace_meta_data_->tracer_->CaptureTimestamp(
        TRTSERVER_TRACE_LEVEL_MIN, "http send end",
        TIMESPEC_TO_NANOS(request->send_end_ts));
  }
#endif  // TRTIS_ENABLE_TRACING

  delete infer_request;
}

HTTPAPIServerV2::InferRequestClass::InferRequestClass(
    evhtp_request_t* req, uint64_t request_id, const char* server_id,
    uint64_t unique_id)
    : req_(req), request_id_(request_id), server_id_(server_id),
      unique_id_(unique_id)
{
  evhtp_connection_t* htpconn = evhtp_request_get_connection(req);
  thread_ = htpconn->thread;
  evhtp_request_pause(req);
}

void
HTTPAPIServerV2::InferRequestClass::InferComplete(
    TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
    TRTSERVER_InferenceResponse* response, void* userp)
{
  HTTPAPIServerV2::InferRequestClass* infer_request =
      reinterpret_cast<HTTPAPIServerV2::InferRequestClass*>(userp);
  for (auto buffer : infer_request->response_meta_data_.response_buffer_) {
    evbuffer_add_buffer(infer_request->req_->buffer_out, buffer);
  }
  if (infer_request->FinalizeResponse(response) == EVHTP_RES_OK) {
    evthr_defer(infer_request->thread_, OKReplyCallback, infer_request);
  } else {
    evthr_defer(infer_request->thread_, BADReplyCallback, infer_request);
  }

  // Don't need to explicitly delete 'trace_manager'. It will be deleted by
  // the TraceMetaData object in 'infer_request'.
  LOG_TRTSERVER_ERROR(
      TRTSERVER_InferenceResponseDelete(response), "deleting HTTP response");
}

evhtp_res
HTTPAPIServerV2::InferRequestClass::FinalizeResponse(
    TRTSERVER_InferenceResponse* response)
{
  InferResponseHeader response_header;

  TRTSERVER_Error* response_status =
      TRTSERVER_InferenceResponseStatus(response);
  if (response_status == nullptr) {
    TRTSERVER_Protobuf* response_protobuf = nullptr;
    response_status =
        TRTSERVER_InferenceResponseHeader(response, &response_protobuf);
    if (response_status == nullptr) {
      const char* buffer;
      size_t byte_size;
      response_status =
          TRTSERVER_ProtobufSerialize(response_protobuf, &buffer, &byte_size);
      if (response_status == nullptr) {
        if (!response_header.ParseFromArray(buffer, byte_size)) {
          response_status = TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_INTERNAL, "failed to parse response header");
        }
      }

      TRTSERVER_ProtobufDelete(response_protobuf);
    }
  }

  if (response_status == nullptr) {
    std::string format;
    const char* format_c_str = evhtp_kv_find(req_->uri->query, "format");
    if (format_c_str != NULL) {
      format = std::string(format_c_str);
    } else {
      format = "text";
    }

    // The description of the raw outputs needs to go in the
    // kInferResponseHTTPHeader since it is needed to interpret the
    // body. The entire response (including classifications) is
    // serialized at the end of the body.
    response_header.set_id(request_id_);

    std::string rstr;
    if (format == "binary") {
      response_header.SerializeToString(&rstr);
    } else {
      rstr = response_header.DebugString();
    }

    evbuffer_add(req_->buffer_out, rstr.c_str(), rstr.size());
  } else {
    evbuffer_drain(req_->buffer_out, -1);
    response_header.Clear();
    response_header.set_id(request_id_);
  }

  RequestStatus request_status;
  RequestStatusUtil::Create(
      &request_status, response_status, unique_id_, server_id_);

  evhtp_headers_add_header(
      req_->headers_out, evhtp_header_new(
                             kInferResponseHTTPHeader,
                             response_header.ShortDebugString().c_str(), 1, 1));
  evhtp_headers_add_header(
      req_->headers_out,
      evhtp_header_new(
          kStatusHTTPHeader, request_status.ShortDebugString().c_str(), 1, 1));
  evhtp_headers_add_header(
      req_->headers_out,
      evhtp_header_new("Content-Type", "application/octet-stream", 1, 1));

  TRTSERVER_ErrorDelete(response_status);

  return (request_status.code() == RequestStatusCode::SUCCESS)
             ? EVHTP_RES_OK
             : EVHTP_RES_BADREQ;
}

TRTSERVER_Error*
HTTPServerV2::CreateAPIServer(
    const std::shared_ptr<TRTSERVER_Server>& server,
    const std::shared_ptr<nvidia::inferenceserver::TraceManager>& trace_manager,
    const std::shared_ptr<SharedMemoryManager>& shm_manager,
    const std::map<int32_t, std::vector<std::string>>& port_map, int thread_cnt,
    std::vector<std::unique_ptr<HTTPServerV2>>* http_servers)
{
  if (port_map.empty()) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INVALID_ARG,
        "HTTP V2 is enabled but none of the service endpoints have a valid "
        "port assignment");
  }
  http_servers->clear();
  for (auto const& ep_map : port_map) {
    std::string addr = "0.0.0.0:" + std::to_string(ep_map.first);
    LOG_INFO << "Starting HTTPV2Service at " << addr;
    http_servers->emplace_back(new HTTPAPIServerV2(
        server, trace_manager, smb_manager, ep_map.second, ep_map.first,
        thread_cnt));
  }

  return nullptr;
}

TRTSERVER_Error*
HTTPServerV2::CreateMetricsServer(
    const std::shared_ptr<TRTSERVER_Server>& server, const int32_t port,
    const int thread_cnt, std::unique_ptr<HTTPServerV2>* metrics_server)
{
  std::string addr = "0.0.0.0:" + std::to_string(port);
  LOG_INFO << "Starting Metrics Service at " << addr;

#ifndef TRTIS_ENABLE_METRICS
  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNAVAILABLE, "Metrics support is disabled");
#endif  // !TRTIS_ENABLE_METRICS

#ifdef TRTIS_ENABLE_METRICS
  metrics_server->reset(new HTTPMetricsServer(server, port, thread_cnt));
  return nullptr;
#endif  // TRTIS_ENABLE_METRICS
}

}}  // namespace nvidia::inferenceserver
