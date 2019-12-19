// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "src/servers/http_server.h"

#include <event2/buffer.h>
#include <evhtp/evhtp.h>
#include <google/protobuf/text_format.h>
#include <re2/re2.h>
#include <algorithm>
#include <thread>
#include "src/core/api.pb.h"
#include "src/core/constants.h"
#include "src/core/grpc_service.grpc.pb.h"
#include "src/core/server_status.pb.h"
#include "src/core/trtserver.h"
#include "src/servers/common.h"

// The HTTP frontend logging is closely related to the server, thus keep it
// using the server logging utils
#include "src/core/logging.h"

#ifdef TRTIS_ENABLE_TRACING
#include "src/servers/tracer.h"
#endif  // TRTIS_ENABLE_TRACING

namespace nvidia { namespace inferenceserver {

// Generic HTTP server using evhtp
class HTTPServerImpl : public HTTPServer {
 public:
  explicit HTTPServerImpl(const int32_t port, const int thread_cnt)
      : port_(port), thread_cnt_(thread_cnt)
  {
  }

  virtual ~HTTPServerImpl() { Stop(); }

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
HTTPServerImpl::Start()
{
  if (!worker_.joinable()) {
    evbase_ = event_base_new();
    htp_ = evhtp_new(evbase_, NULL);
    evhtp_enable_flag(htp_, EVHTP_FLAG_ENABLE_NODELAY);
    evhtp_set_gencb(htp_, HTTPServerImpl::Dispatch, this);
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
HTTPServerImpl::Stop()
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
HTTPServerImpl::StopCallback(int sock, short events, void* arg)
{
  struct event_base* base = (struct event_base*)arg;
  event_base_loopbreak(base);
}

void
HTTPServerImpl::Dispatch(evhtp_request_t* req, void* arg)
{
  (static_cast<HTTPServerImpl*>(arg))->Handle(req);
}


// Handle HTTP requests to inference server APIs
class HTTPAPIServer : public HTTPServerImpl {
 public:
  explicit HTTPAPIServer(
      const std::shared_ptr<TRTSERVER_Server>& server,
      const std::shared_ptr<nvidia::inferenceserver::TraceManager>&
          trace_manager,
      const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
      const std::vector<std::string>& endpoints, const int32_t port,
      const int thread_cnt)
      : HTTPServerImpl(port, thread_cnt), server_(server),
        trace_manager_(trace_manager), smb_manager_(smb_manager),
        allocator_(nullptr),
        api_regex_(R"(/v1/models/([^(/|:)]+)(:predict|/metadata)?)")
  {
    TRTSERVER_Error* err = TRTSERVER_ServerId(server_.get(), &server_id_);
    if (err != nullptr) {
      server_id_ = "unknown:0";
      TRTSERVER_ErrorDelete(err);
    }

    FAIL_IF_ERR(
        TRTSERVER_ResponseAllocatorNew(
            &allocator_, ResponseAlloc, ResponseRelease),
        "creating response allocator");
  }

  ~HTTPAPIServer()
  {
    LOG_IF_ERR(
        TRTSERVER_ResponseAllocatorDelete(allocator_),
        "deleting response allocator");
  }

  using EVBufferTuple = std::tuple<
      evbuffer*,
      std::unordered_map<
          std::string,
          std::tuple<const void*, size_t, TRTSERVER_Memory_Type, int64_t>>,
      InferRequest>;

  // Class object associated to evhtp thread, requests received are bounded
  // with the thread that accepts it. Need to keep track of that and let the
  // corresponding thread send back the reply
  class InferRequestClass {
   public:
    InferRequestClass(
        evhtp_request_t* req, uint64_t request_id, const char* server_id,
        uint64_t unique_id);
    ~InferRequestClass() = default;

    evhtp_request_t* EvHtpRequest() const { return req_; }

    static void InferComplete(
        TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
        TRTSERVER_InferenceResponse* response, void* userp);
    evhtp_res FinalizeResponse(TRTSERVER_InferenceResponse* response);

#ifdef TRTIS_ENABLE_TRACING
    std::unique_ptr<TraceMetaData> trace_meta_data_;
#endif  // TRTIS_ENABLE_TRACING

    std::unique_ptr<EVBufferTuple> response_tuple_;

   private:
    evhtp_request_t* req_;
    evthr_t* thread_;
    const uint64_t request_id_;
    const char* const server_id_;
    const uint64_t unique_id_;
  };

 private:
  static TRTSERVER_Error* ResponseAlloc(
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
  void HandleHealth(evhtp_request_t* req, const std::string& model_name);
  void HandleStatus(evhtp_request_t* req, const std::string& model_name);
  void HandleInfer(evhtp_request_t* req, const std::string& model_name);

#ifdef TRTIS_ENABLE_GPU
  TRTSERVER_Error* EVBufferToCudaHandle(
      evbuffer* handle_buffer, cudaIpcMemHandle_t** cuda_shm_handle);
#endif  // TRTIS_ENABLE_GPU
  TRTSERVER_Error* EVBufferToInput(
      const std::string& model_name, const InferRequestHeader& request_header,
      const InferRequest& request,
      TRTSERVER_InferenceRequestProvider* request_provider,
      std::unordered_map<
          std::string,
          std::tuple<const void*, size_t, TRTSERVER_Memory_Type, int64_t>>&
          output_shm_map);

  static void OKReplyCallback(evthr_t* thr, void* arg, void* shared);
  static void BADReplyCallback(evthr_t* thr, void* arg, void* shared);

  std::shared_ptr<TRTSERVER_Server> server_;
  const char* server_id_;

  std::shared_ptr<TraceManager> trace_manager_;
  std::shared_ptr<SharedMemoryBlockManager> smb_manager_;

  // The allocator that will be used to allocate buffers for the
  // inference result tensors.
  TRTSERVER_ResponseAllocator* allocator_;

  re2::RE2 api_regex_;
};

TRTSERVER_Error*
HTTPAPIServer::ResponseAlloc(
    TRTSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRTSERVER_Memory_Type preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRTSERVER_Memory_Type* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  auto userp_tuple = reinterpret_cast<EVBufferTuple*>(userp);
  evbuffer* evhttp_buffer =
      reinterpret_cast<evbuffer*>(std::get<0>(*userp_tuple));
  const std::unordered_map<
      std::string,
      std::tuple<const void*, size_t, TRTSERVER_Memory_Type, int64_t>>&
      output_shm_map = std::get<1>(*userp_tuple);

  *buffer = nullptr;
  *buffer_userp = nullptr;
  *actual_memory_type = preferred_memory_type;
  *actual_memory_type_id = preferred_memory_type_id;

  // Don't need to do anything if no memory was requested.
  if (byte_size > 0) {
    auto pr = output_shm_map.find(tensor_name);
    if (pr != output_shm_map.end()) {
      // If the output is in shared memory then check that the expected buffer
      // size is at least the byte size of the output.
      if (byte_size > std::get<1>(pr->second)) {
        return TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INTERNAL,
            std::string(
                "expected buffer size to be at least " +
                std::to_string(std::get<1>(pr->second)) + " bytes but gets " +
                std::to_string(byte_size) + " bytes in output tensor")
                .c_str());
      }

      *buffer = const_cast<void*>(std::get<0>(pr->second));
      *actual_memory_type = std::get<2>(pr->second);
      *actual_memory_type_id = std::get<3>(pr->second);
    } else {
      // Can't allocate for any memory type other than CPU.
      if (preferred_memory_type != TRTSERVER_MEMORY_CPU) {
        LOG_VERBOSE(1)
            << "HTTP V2: unable to provide '" << tensor_name
            << "' in TRTSERVER_MEMORY_GPU, will use type TRTSERVER_MEMORY_CPU";
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
    }
  }

  LOG_VERBOSE(1) << "HTTP V2 allocation: '" << tensor_name
                 << "', size: " << byte_size << ", addr: " << *buffer;

  return nullptr;  // Success
}

TRTSERVER_Error*
HTTPAPIServer::ResponseRelease(
    TRTSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRTSERVER_Memory_Type memory_type, int64_t memory_type_id)
{
  LOG_VERBOSE(1) << "HTTP V2 release: "
                 << "size " << byte_size << ", addr " << buffer;

  // Don't do anything when releasing a buffer since ResponseAlloc
  // wrote directly into the response ebvuffer.
  return nullptr;  // Success
}

void
HTTPAPIServer::Handle(evhtp_request_t* req)
{
  LOG_VERBOSE(1) << "HTTP V2 request: " << req->method << " "
                 << req->uri->path->full;

  std::string model_name, rest;
  if (RE2::FullMatch(
          std::string(req->uri->path->full), api_regex_, &model_name, &rest)) {
    // status
    if (rest == "/metadata") {
      HandleStatus(req, model_name);
      return;
    }
    // health
    if (rest == "") {
      HandleHealth(req, model_name);
      return;
    }
    // infer
    if (rest == ":predict") {
      HandleInfer(req, model_name);
      return;
    }
  }

  LOG_VERBOSE(1) << "HTTP V2 error: " << req->method << " "
                 << req->uri->path->full << " - "
                 << static_cast<int>(EVHTP_RES_BADREQ);
  evhtp_send_reply(req, EVHTP_RES_BADREQ);
}

void
HTTPAPIServer::HandleHealth(evhtp_request_t* req, const std::string& model_name)
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
  TRTSERVER_Protobuf* server_status_protobuf = nullptr;
  TRTSERVER_Error* err = TRTSERVER_ServerModelStatus(
      server_.get(), model_name.c_str(), &server_status_protobuf);
  if (err == nullptr) {
    const char* status_buffer;
    size_t status_byte_size;
    err = TRTSERVER_ProtobufSerialize(
        server_status_protobuf, &status_buffer, &status_byte_size);
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
          auto model_versions = itr->second.version_status();
          for (auto mit = model_versions.begin(); mit != model_versions.end();
               ++mit) {
            ready = (mit->second.ready_state() == ModelReadyState::MODEL_READY)
                        ? true
                        : false;
          }
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
      req, (ready && (err == nullptr)) ? EVHTP_RES_OK : EVHTP_RES_BADREQ);

  TRTSERVER_ErrorDelete(err);
}
void
HTTPAPIServer::HandleStatus(evhtp_request_t* req, const std::string& model_name)
{
  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  if (model_name.empty()) {
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    return;
  }

  TRTSERVER_Protobuf* server_status_protobuf = nullptr;
  TRTSERVER_Error* err =
      (model_name.empty())
          ? TRTSERVER_ServerStatus(server_.get(), &server_status_protobuf)
          : TRTSERVER_ServerModelStatus(
                server_.get(), model_name.c_str(), &server_status_protobuf);
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
HTTPAPIServer::EVBufferToInput(
    const std::string& model_name, const InferRequestHeader& request_header,
    const InferRequest& request,
    TRTSERVER_InferenceRequestProvider* request_provider,
    std::unordered_map<
        std::string,
        std::tuple<const void*, size_t, TRTSERVER_Memory_Type, int64_t>>&
        output_shm_map)
{
  // Extract input data from HTTP body and register in
  // 'request_provider'.
  // Get the byte-size for each input and from that get the blocks
  // holding the data for that input
  size_t idx = 0;
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
        TRTSERVER_SharedMemoryBlock* smb = nullptr;
        RETURN_IF_ERR(smb_manager_->Get(&smb, io.shared_memory().name()));
        RETURN_IF_ERR(TRTSERVER_ServerSharedMemoryAddress(
            server_.get(), smb, io.shared_memory().offset(),
            io.shared_memory().byte_size(), &base));
        TRTSERVER_SharedMemoryBlockMemoryType(smb, &memory_type);
        TRTSERVER_SharedMemoryBlockMemoryTypeId(smb, &memory_type_id);
        RETURN_IF_ERR(TRTSERVER_InferenceRequestProviderSetInputData(
            request_provider, io.name().c_str(), base, byte_size, memory_type,
            memory_type_id));
      } else {
        const std::string& raw = request.raw_input(idx++);
        const void* base = raw.c_str();
        size_t request_byte_size = raw.size();

        if (byte_size != request_byte_size) {
          return TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_INVALID_ARG,
              std::string(
                  "unexpected size " + std::to_string(request_byte_size) +
                  " for input '" + io.name() + "', expecting " +
                  std::to_string(byte_size) + " for model '" + model_name + "'")
                  .c_str());
        }

        RETURN_IF_ERR(TRTSERVER_InferenceRequestProviderSetInputData(
            request_provider, io.name().c_str(), base, byte_size,
            TRTSERVER_MEMORY_CPU, 0 /* memory_type_id */));
      }
    }
  }

  // Initialize System Memory for Output if it uses shared memory
  for (const auto& io : request_header.output()) {
    if (io.has_shared_memory()) {
      void* base;
      TRTSERVER_SharedMemoryBlock* smb = nullptr;
      RETURN_IF_ERR(smb_manager_->Get(&smb, io.shared_memory().name()));
      RETURN_IF_ERR(TRTSERVER_ServerSharedMemoryAddress(
          server_.get(), smb, io.shared_memory().offset(),
          io.shared_memory().byte_size(), &base));

      TRTSERVER_Memory_Type memory_type;
      int64_t memory_type_id;
      TRTSERVER_SharedMemoryBlockMemoryType(smb, &memory_type);
      TRTSERVER_SharedMemoryBlockMemoryTypeId(smb, &memory_type_id);
      output_shm_map.emplace(
          io.name(),
          std::make_tuple(
              static_cast<const void*>(base), io.shared_memory().byte_size(),
              memory_type, memory_type_id));
    }
  }

  return nullptr;  // success
}

void
HTTPAPIServer::HandleInfer(evhtp_request_t* req, const std::string& model_name)
{
  if (req->method != htp_method_POST) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  // Assume -1 for now
  std::string model_version_str = "-1";
  if (model_name.empty()) {
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
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

  // Convert the json string to protobuf message
  EVBufferTuple* response_tuple(new EVBufferTuple());
  size_t buffer_length = evbuffer_get_length(req->buffer_in);
  char* request_buffer = (char*)malloc(sizeof(char) * buffer_length);
  evbuffer_copyout(req->buffer_in, request_buffer, buffer_length);
  std::string json_request_string = std::string(request_buffer, buffer_length);
  if (google::protobuf::util::JsonStringToMessage(
          json_request_string, &std::get<2>(*response_tuple)) !=
      google::protobuf::util::Status::OK) {
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    return;
  }
  free(request_buffer);

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
    err = EVBufferToInput(
        model_name, request_header_protobuf, std::get<2>(*response_tuple),
        request_provider, std::get<1>(*response_tuple));
    if (err == nullptr) {
      InferRequestClass* infer_request = new InferRequestClass(
          req, request_header_protobuf.id(), server_id_, unique_id);

      std::get<0>(*response_tuple) = req->buffer_out;
      infer_request->response_tuple_.reset(response_tuple);

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
          reinterpret_cast<void*>(response_tuple),
          InferRequestClass::InferComplete,
          reinterpret_cast<void*>(infer_request));
      if (err != nullptr) {
        delete infer_request;
        infer_request = nullptr;
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
HTTPAPIServer::OKReplyCallback(evthr_t* thr, void* arg, void* shared)
{
  HTTPAPIServer::InferRequestClass* infer_request =
      reinterpret_cast<HTTPAPIServer::InferRequestClass*>(arg);

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
HTTPAPIServer::BADReplyCallback(evthr_t* thr, void* arg, void* shared)
{
  HTTPAPIServer::InferRequestClass* infer_request =
      reinterpret_cast<HTTPAPIServer::InferRequestClass*>(arg);

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

HTTPAPIServer::InferRequestClass::InferRequestClass(
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
HTTPAPIServer::InferRequestClass::InferComplete(
    TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
    TRTSERVER_InferenceResponse* response, void* userp)
{
  HTTPAPIServer::InferRequestClass* infer_request =
      reinterpret_cast<HTTPAPIServer::InferRequestClass*>(userp);
  if (infer_request->FinalizeResponse(response) == EVHTP_RES_OK) {
    evthr_defer(infer_request->thread_, OKReplyCallback, infer_request);
  } else {
    evthr_defer(infer_request->thread_, BADReplyCallback, infer_request);
  }

  // Don't need to explicitly delete 'trace_manager'. It will be deleted by
  // the TraceMetaData object in 'infer_request'.
  LOG_IF_ERR(
      TRTSERVER_InferenceResponseDelete(response), "deleting HTTP response");
}

evhtp_res
HTTPAPIServer::InferRequestClass::FinalizeResponse(
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
HTTPServer::CreateAPIServer(
    const std::shared_ptr<TRTSERVER_Server>& server,
    const std::shared_ptr<nvidia::inferenceserver::TraceManager>& trace_manager,
    const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
    const std::map<int32_t, std::vector<std::string>>& port_map, int thread_cnt,
    std::vector<std::unique_ptr<HTTPServer>>* http_servers)
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
    http_servers->emplace_back(new HTTPAPIServer(
        server, trace_manager, smb_manager, ep_map.second, ep_map.first,
        thread_cnt));
  }

  return nullptr;
}

}}  // namespace nvidia::inferenceserver
