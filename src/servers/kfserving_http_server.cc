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

#include "src/servers/kfserving_http_server.h"

#include <event2/buffer.h>
#include <evhtp/evhtp.h>
#include <google/protobuf/text_format.h>
#include <re2/re2.h>
#include <algorithm>
#include <thread>
#include "src/core/api.pb.h"
#include "src/core/constants.h"
#include "src/core/server_status.pb.h"
#include "src/core/trtserver.h"
#include "src/servers/common.h"

#ifdef TRTIS_ENABLE_TRACING
#include "src/servers/tracer.h"
#endif  // TRTIS_ENABLE_TRACING

namespace nvidia { namespace inferenceserver {

// Generic HTTP server using evhtp
class KFServingHTTPServerImpl : public KFServingHTTPServer {
 public:
  explicit KFServingHTTPServerImpl(const int32_t port, const int thread_cnt)
      : port_(port), thread_cnt_(thread_cnt)
  {
  }

  virtual ~KFServingHTTPServerImpl() { Stop(); }

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
KFServingHTTPServerImpl::Start()
{
  if (!worker_.joinable()) {
    evbase_ = event_base_new();
    htp_ = evhtp_new(evbase_, NULL);
    evhtp_enable_flag(htp_, EVHTP_FLAG_ENABLE_NODELAY);
    evhtp_set_gencb(htp_, KFServingHTTPServerImpl::Dispatch, this);
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
      TRTSERVER_ERROR_ALREADY_EXISTS,
      "KFServing HTTP server is already running.");
}

TRTSERVER_Error*
KFServingHTTPServerImpl::Stop()
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
      TRTSERVER_ERROR_UNAVAILABLE, "HTTP server is not running.");
}

void
KFServingHTTPServerImpl::StopCallback(int sock, short events, void* arg)
{
  struct event_base* base = (struct event_base*)arg;
  event_base_loopbreak(base);
}

void
KFServingHTTPServerImpl::Dispatch(evhtp_request_t* req, void* arg)
{
  (static_cast<KFServingHTTPServerImpl*>(arg))->Handle(req);
}


// Handle HTTP requests to inference server APIs
class KFServingHTTPAPIServer : public KFServingHTTPServerImpl {
 public:
  explicit KFServingHTTPAPIServer(
      const std::shared_ptr<TRTSERVER_Server>& server,
      const std::shared_ptr<nvidia::inferenceserver::TraceManager>&
          trace_manager,
      const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
      const int32_t port, const int thread_cnt)
      : KFServingHTTPServerImpl(port, thread_cnt), server_(server),
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

  ~KFServingHTTPAPIServer()
  {
    LOG_IF_ERR(
        TRTSERVER_ResponseAllocatorDelete(allocator_),
        "deleting response allocator");
  }

  using EVBufferPair = std::pair<
      evbuffer*,
      std::unordered_map<
          std::string,
          std::tuple<const void*, size_t, TRTSERVER_Memory_Type, int64_t>>>;

  // Class object associated to evhtp thread, requests received are bounded
  // with the thread that accepts it. Need to keep track of that and let the
  // corresponding thread send back the reply
  class InferRequest {
   public:
    InferRequest(
        evhtp_request_t* req, uint64_t request_id, const char* server_id,
        uint64_t unique_id);
    ~InferRequest() = default;

    evhtp_request_t* EvHtpRequest() const { return req_; }

    static void InferComplete(
        TRTSERVER_Server* server, TRTSERVER_Trace* trace,
        TRTSERVER_InferenceResponse* response, void* userp);
    evhtp_res FinalizeResponse(TRTSERVER_InferenceResponse* response);

#ifdef TRTIS_ENABLE_TRACING
    std::unique_ptr<Tracer> tracer_;
#endif  // TRTIS_ENABLE_TRACING

    std::unique_ptr<EVBufferPair> response_pair_;

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
  void HandleHealth(evhtp_request_t* req, const std::string& health_uri);
  void HandleInfer(evhtp_request_t* req, const std::string& infer_uri);
  void HandleStatus(evhtp_request_t* req, const std::string& status_uri);

#if TRTIS_ENABLE_GPU
  TRTSERVER_Error* EVBufferToCudaHandle(
      evbuffer* handle_buffer, cudaIpcMemHandle_t** cuda_shm_handle);
#endif  // TRTIS_ENABLE_GPU
  TRTSERVER_Error* Base64BufferToInput(
      const std::string& model_name, const InferRequestHeader& request_header,
      evbuffer* input_buffer,
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
KFServingHTTPAPIServer::ResponseAlloc(
    TRTSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRTSERVER_Memory_Type preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRTSERVER_Memory_Type* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  auto userp_pair = reinterpret_cast<EVBufferPair*>(userp);
  evbuffer* evhttp_buffer = reinterpret_cast<evbuffer*>(userp_pair->first);
  const std::unordered_map<
      std::string,
      std::tuple<const void*, size_t, TRTSERVER_Memory_Type, int64_t>>&
      output_shm_map = userp_pair->second;

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
            << "HTTP: unable to provide '" << tensor_name
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

  LOG_VERBOSE(1) << "HTTP allocation: '" << tensor_name
                 << "', size: " << byte_size << ", addr: " << *buffer;

  return nullptr;  // Success
}

TRTSERVER_Error*
KFServingHTTPAPIServer::ResponseRelease(
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
KFServingHTTPAPIServer::Handle(evhtp_request_t* req)
{
  LOG_VERBOSE(1) << "HTTP request: " << req->method << " "
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
    if ((rest == "") && (model_name != "metadata")) {
      HandleHealth(req, model_name);
      return;
    }
    // infer
    if (rest == ":predict") {
      HandleInfer(req, model_name);
      return;
    }
  }

  LOG_VERBOSE(1) << "HTTP error: " << req->method << " " << req->uri->path->full
                 << " - " << static_cast<int>(EVHTP_RES_BADREQ);
  evhtp_send_reply(req, EVHTP_RES_BADREQ);
}

void
KFServingHTTPAPIServer::HandleHealth(
    evhtp_request_t* req, const std::string& model_name)
{
  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  if (model_name.empty()) {
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    return;
  }

  TRTSERVER_Error* err = nullptr;
  bool health = false;
  // ready mode
  // TODO implement on model name level
  err = TRTSERVER_ServerIsReady(server_.get(), &health);

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
KFServingHTTPAPIServer::HandleStatus(
    evhtp_request_t* req, const std::string& model_name)
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
  TRTSERVER_Error* err = TRTSERVER_ServerModelStatus(
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

#if TRTIS_ENABLE_GPU
TRTSERVER_Error*
KFServingHTTPAPIServer::EVBufferToCudaHandle(
    evbuffer* handle_buffer, cudaIpcMemHandle_t** cuda_shm_handle)
{
  // Extract serialzied cuda IPC handle from HTTP body and store in
  // 'cuda_shm_handle'.
  struct evbuffer_iovec* v = nullptr;
  *cuda_shm_handle = nullptr;
  size_t byte_size = sizeof(cudaIpcMemHandle_t);

  int n = evbuffer_peek(handle_buffer, -1, NULL, NULL, 0);
  if (n > 0) {
    v = static_cast<struct evbuffer_iovec*>(
        alloca(sizeof(struct evbuffer_iovec) * n));
    if (evbuffer_peek(handle_buffer, -1, NULL, v, n) != n) {
      return TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_INTERNAL, "unexpected error getting input buffers ");
    }
  }

  if (byte_size != v[0].iov_len) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected size for CUDA shared-memory handle, expecting " +
            std::to_string(byte_size) + " bytes")
            .c_str());
  }

  // Deserialize the cuda IPC handle
  *cuda_shm_handle = reinterpret_cast<cudaIpcMemHandle_t*>(v[0].iov_base);

  return nullptr;  // success
}
#endif  // TRTIS_ENABLE_GPU

TRTSERVER_Error*
KFServingHTTPAPIServer::Base64BufferToInput(
    const std::string& model_name, const InferRequestHeader& request_header,
    evbuffer* input_buffer,
    TRTSERVER_InferenceRequestProvider* request_provider,
    std::unordered_map<
        std::string,
        std::tuple<const void*, size_t, TRTSERVER_Memory_Type, int64_t>>&
        output_shm_map)
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
  // Use FromBase64 to convert a Base64 encoded string to ByteString
  // TODO encode serialized string in base64 (on Client side)
  // std::string base_decoded = base64_decode(std::string(base, base_size));
  // InferRequest infer_request;
  // infer_request.ParseFromString(base_decoded);

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
                  std::to_string(byte_size) + " bytes for model '" +
                  model_name + "'")
                  .c_str());
        }
      }
    }
  }

  if (v_idx != n) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected additional input data for model '" + model_name + "'")
            .c_str());
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
KFServingHTTPAPIServer::HandleInfer(
    evhtp_request_t* req, const std::string& model_name)
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
  std::unique_ptr<Tracer> tracer;
  if (trace_manager_ != nullptr) {
    tracer.reset(trace_manager_->SampleTrace());
    if (tracer != nullptr) {
      tracer->SetModel(model_name, model_version);
      tracer->CaptureTimestamp(
          TRTSERVER_TRACE_LEVEL_MIN, "http recv start",
          TIMESPEC_TO_NANOS(req->recv_start_ts));
      tracer->CaptureTimestamp(
          TRTSERVER_TRACE_LEVEL_MIN, "http recv end",
          TIMESPEC_TO_NANOS(req->recv_end_ts));
    }
  }
#endif  // TRTIS_ENABLE_TRACING

  std::string infer_request_header(
      evhtp_kv_find(req->headers_in, kInferRequestHTTPHeader));

  InferRequestHeader request_header;
  if (!google::protobuf::TextFormat::ParseFromString(
          infer_request_header, &request_header)) {
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    return;
  }

  std::string request_header_serialized;
  if (!request_header.SerializeToString(&request_header_serialized)) {
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    return;
  }

  uint64_t unique_id = RequestStatusUtil::NextUniqueRequestId();

  // Create the inference request provider which provides all the
  // input information needed for an inference.
  TRTSERVER_InferenceRequestProvider* request_provider = nullptr;
  TRTSERVER_Error* err = TRTSERVER_InferenceRequestProviderNew(
      &request_provider, server_.get(), model_name.c_str(), model_version,
      request_header_serialized.c_str(), request_header_serialized.size());
  if (err == nullptr) {
    EVBufferPair* response_pair(new EVBufferPair());
    err = Base64BufferToInput(
        model_name, request_header, req->buffer_in, request_provider,
        response_pair->second);
    if (err == nullptr) {
      InferRequest* infer_request =
          new InferRequest(req, request_header.id(), server_id_, unique_id);

      response_pair->first = req->buffer_out;
      infer_request->response_pair_.reset(response_pair);

      // Get the trace object to use for this request. If nullptr then
      // no tracing will be performed.
      TRTSERVER_Trace* trace = nullptr;
#ifdef TRTIS_ENABLE_TRACING
      if (tracer != nullptr) {
        infer_request->tracer_ = std::move(tracer);
        trace = infer_request->tracer_->ServerTrace();
      }
#endif  // TRTIS_ENABLE_TRACING

      err = TRTSERVER_ServerInferAsync(
          server_.get(), trace, request_provider, allocator_,
          reinterpret_cast<void*>(response_pair), InferRequest::InferComplete,
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

  if (err != nullptr) {
    RequestStatus request_status;
    RequestStatusUtil::Create(&request_status, err, unique_id, server_id_);

    InferResponseHeader response_header;
    response_header.set_id(request_header.id());
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
KFServingHTTPAPIServer::OKReplyCallback(evthr_t* thr, void* arg, void* shared)
{
  KFServingHTTPAPIServer::InferRequest* infer_request =
      reinterpret_cast<KFServingHTTPAPIServer::InferRequest*>(arg);

  evhtp_request_t* request = infer_request->EvHtpRequest();
  evhtp_send_reply(request, EVHTP_RES_OK);
  evhtp_request_resume(request);

#ifdef TRTIS_ENABLE_TRACING
  if (infer_request->tracer_ != nullptr) {
    infer_request->tracer_->CaptureTimestamp(
        TRTSERVER_TRACE_LEVEL_MIN, "http send start",
        TIMESPEC_TO_NANOS(request->send_start_ts));
    infer_request->tracer_->CaptureTimestamp(
        TRTSERVER_TRACE_LEVEL_MIN, "http send end",
        TIMESPEC_TO_NANOS(request->send_end_ts));
  }
#endif  // TRTIS_ENABLE_TRACING

  delete infer_request;
}

void
KFServingHTTPAPIServer::BADReplyCallback(evthr_t* thr, void* arg, void* shared)
{
  KFServingHTTPAPIServer::InferRequest* infer_request =
      reinterpret_cast<KFServingHTTPAPIServer::InferRequest*>(arg);

  evhtp_request_t* request = infer_request->EvHtpRequest();
  evhtp_send_reply(request, EVHTP_RES_BADREQ);
  evhtp_request_resume(request);

#ifdef TRTIS_ENABLE_TRACING
  if (infer_request->tracer_ != nullptr) {
    infer_request->tracer_->CaptureTimestamp(
        TRTSERVER_TRACE_LEVEL_MIN, "http send start",
        TIMESPEC_TO_NANOS(request->send_start_ts));
    infer_request->tracer_->CaptureTimestamp(
        TRTSERVER_TRACE_LEVEL_MIN, "http send end",
        TIMESPEC_TO_NANOS(request->send_end_ts));
  }
#endif  // TRTIS_ENABLE_TRACING

  delete infer_request;
}

KFServingHTTPAPIServer::InferRequest::InferRequest(
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
KFServingHTTPAPIServer::InferRequest::InferComplete(
    TRTSERVER_Server* server, TRTSERVER_Trace* trace,
    TRTSERVER_InferenceResponse* response, void* userp)
{
  KFServingHTTPAPIServer::InferRequest* infer_request =
      reinterpret_cast<KFServingHTTPAPIServer::InferRequest*>(userp);
  if (infer_request->FinalizeResponse(response) == EVHTP_RES_OK) {
    evthr_defer(infer_request->thread_, OKReplyCallback, infer_request);
  } else {
    evthr_defer(infer_request->thread_, BADReplyCallback, infer_request);
  }

  // Don't need to explicitly delete 'trace'. It will be deleted by
  // the Tracer object in 'infer_request'.
  LOG_IF_ERR(
      TRTSERVER_InferenceResponseDelete(response), "deleting HTTP response");
}

evhtp_res
KFServingHTTPAPIServer::InferRequest::FinalizeResponse(
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
KFServingHTTPServer::CreateAPIServer(
    const std::shared_ptr<TRTSERVER_Server>& server,
    const std::shared_ptr<nvidia::inferenceserver::TraceManager>& trace_manager,
    const std::shared_ptr<SharedMemoryBlockManager>& smb_manager, int32_t port_,
    int thread_cnt, std::unique_ptr<KFServingHTTPServer>* kfserving_http_server)
{
  std::string addr = "0.0.0.0:" + std::to_string(port_);
  LOG_INFO << "Starting KFServingHTTPService at " << addr;
  kfserving_http_server->reset(new KFServingHTTPAPIServer(
      server, trace_manager, smb_manager, port_, thread_cnt));

  return nullptr;
}

}}  // namespace nvidia::inferenceserver
