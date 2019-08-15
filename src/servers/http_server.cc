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
#include "src/core/server_status.pb.h"
#include "src/core/trtserver.h"
#include "src/servers/common.h"

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
      TRTSERVER_ERROR_ALREADY_EXISTS, "HTTP server is already running.");
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
      TRTSERVER_ERROR_UNAVAILABLE, "HTTP server is not running.");
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
class HTTPAPIServer : public HTTPServerImpl {
 public:
  explicit HTTPAPIServer(
      const std::shared_ptr<TRTSERVER_Server>& server,
      const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
      const std::vector<std::string>& endpoints, const int32_t port,
      const int thread_cnt)
      : HTTPServerImpl(port, thread_cnt), server_(server),
        smb_manager_(smb_manager), endpoint_names_(endpoints),
        allocator_(nullptr),
        api_regex_(
            R"(/api/(health|profile|infer|status|modelcontrol|sharedmemorycontrol)(.*))"),
        health_regex_(R"(/(live|ready))"),
        infer_regex_(R"(/([^/]+)(?:/(\d+))?)"), status_regex_(R"(/(.*))"),
        modelcontrol_regex_(R"(/(load|unload)/([^/]+))"),
        sharedmemorycontrol_regex_(
            R"(/(register|unregister|unregisterall|status)(.*))")
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

  using EVBufferPair = std::pair<
      evbuffer*,
      std::unordered_map<std::string, std::pair<const void*, size_t>>>;

  // Class object associated to evhtp thread, requests received are bounded
  // with the thread that accepts it. Need to keep track of that and let the
  // corresponding thread send back the reply
  class InferRequest {
   public:
    InferRequest(
        evhtp_request_t* req, uint64_t request_id, const char* server_id,
        uint64_t unique_id);
    ~InferRequest() = default;

    static void InferComplete(
        TRTSERVER_Server* server, TRTSERVER_InferenceResponse* response,
        void* userp);
    evhtp_res FinalizeResponse(TRTSERVER_InferenceResponse* response);

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
      TRTSERVER_ResponseAllocator* allocator, void** buffer,
      void** buffer_userp, const char* tensor_name, size_t byte_size,
      TRTSERVER_Memory_Type memory_type, int64_t memory_type_id, void* userp);
  static TRTSERVER_Error* ResponseRelease(
      TRTSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
      size_t byte_size, TRTSERVER_Memory_Type memory_type,
      int64_t memory_type_id);

  void Handle(evhtp_request_t* req) override;
  void HandleHealth(evhtp_request_t* req, const std::string& health_uri);
  void HandleProfile(evhtp_request_t* req, const std::string& profile_uri);
  void HandleInfer(evhtp_request_t* req, const std::string& infer_uri);
  void HandleStatus(evhtp_request_t* req, const std::string& status_uri);
  void HandleModelControl(
      evhtp_request_t* req, const std::string& modelcontrol_uri);
  void HandleSharedMemoryControl(
      evhtp_request_t* req, const std::string& sharedmemorycontrol_uri);

  TRTSERVER_Error* EVBufferToInput(
      const std::string& model_name, const InferRequestHeader& request_header,
      evbuffer* input_buffer,
      TRTSERVER_InferenceRequestProvider* request_provider,
      std::unordered_map<std::string, std::pair<const void*, size_t>>&
          output_shm_map);

  static void OKReplyCallback(evthr_t* thr, void* arg, void* shared);
  static void BADReplyCallback(evthr_t* thr, void* arg, void* shared);

  std::shared_ptr<TRTSERVER_Server> server_;
  const char* server_id_;

  std::shared_ptr<SharedMemoryBlockManager> smb_manager_;
  std::vector<std::string> endpoint_names_;

  // The allocator that will be used to allocate buffers for the
  // inference result tensors.
  TRTSERVER_ResponseAllocator* allocator_;

  re2::RE2 api_regex_;
  re2::RE2 health_regex_;
  re2::RE2 infer_regex_;
  re2::RE2 status_regex_;
  re2::RE2 modelcontrol_regex_;
  re2::RE2 sharedmemorycontrol_regex_;
};

TRTSERVER_Error*
HTTPAPIServer::ResponseAlloc(
    TRTSERVER_ResponseAllocator* allocator, void** buffer, void** buffer_userp,
    const char* tensor_name, size_t byte_size,
    TRTSERVER_Memory_Type memory_type, int64_t memory_type_id, void* userp)
{
  auto userp_pair = reinterpret_cast<EVBufferPair*>(userp);
  evbuffer* evhttp_buffer = reinterpret_cast<evbuffer*>(userp_pair->first);
  const std::unordered_map<std::string, std::pair<const void*, size_t>>&
      output_shm_map = userp_pair->second;

  *buffer = nullptr;
  *buffer_userp = nullptr;

  // Don't need to do anything if no memory was requested.
  if (byte_size > 0) {
    // Can't allocate for any memory type other than CPU.
    if (memory_type != TRTSERVER_MEMORY_CPU) {
      LOG_VERBOSE(1) << "HTTP allocation failed for type " << memory_type
                     << " for " << tensor_name;
      return nullptr;
    }

    auto pr = output_shm_map.find(tensor_name);
    if (pr != output_shm_map.end()) {
      // check for byte size mismatch
      if (byte_size != pr->second.second) {
        return TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INTERNAL,
            std::string(
                "expected buffer size to be " +
                std::to_string(pr->second.second) + "bytes but gets " +
                std::to_string(byte_size) + " bytes in output tensor")
                .c_str());
      }

      *buffer = const_cast<void*>(pr->second.first);
    } else {
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

  LOG_VERBOSE(1) << "HTTP allocation: " << tensor_name << ", size " << byte_size
                 << ", addr " << *buffer;

  return nullptr;  // Success
}

TRTSERVER_Error*
HTTPAPIServer::ResponseRelease(
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
HTTPAPIServer::Handle(evhtp_request_t* req)
{
  LOG_VERBOSE(1) << "HTTP request: " << req->method << " "
                 << req->uri->path->full;

  std::string endpoint, rest;
  if (RE2::FullMatch(
          std::string(req->uri->path->full), api_regex_, &endpoint, &rest)) {
    // status
    if (endpoint == "status" &&
        (std::find(endpoint_names_.begin(), endpoint_names_.end(), "status") !=
         endpoint_names_.end())) {
      HandleStatus(req, rest);
      return;
    }
    // health
    if (endpoint == "health" &&
        (std::find(endpoint_names_.begin(), endpoint_names_.end(), "health") !=
         endpoint_names_.end())) {
      HandleHealth(req, rest);
      return;
    }
    // profile
    if (endpoint == "profile" &&
        (std::find(endpoint_names_.begin(), endpoint_names_.end(), "profile") !=
         endpoint_names_.end())) {
      HandleProfile(req, rest);
      return;
    }
    // infer
    if (endpoint == "infer" &&
        (std::find(endpoint_names_.begin(), endpoint_names_.end(), "infer") !=
         endpoint_names_.end())) {
      HandleInfer(req, rest);
      return;
    }
    // modelcontrol
    if (endpoint == "modelcontrol" &&
        (std::find(
             endpoint_names_.begin(), endpoint_names_.end(), "modelcontrol") !=
         endpoint_names_.end())) {
      HandleModelControl(req, rest);
      return;
    }
    // sharedmemorycontrol
    if (endpoint == "sharedmemorycontrol" &&
        (std::find(
             endpoint_names_.begin(), endpoint_names_.end(),
             "sharedmemorycontrol") != endpoint_names_.end())) {
      HandleSharedMemoryControl(req, rest);
      return;
    }
  }

  LOG_VERBOSE(1) << "HTTP error: " << req->method << " " << req->uri->path->full
                 << " - " << static_cast<int>(EVHTP_RES_BADREQ);
  evhtp_send_reply(req, EVHTP_RES_BADREQ);
}

void
HTTPAPIServer::HandleHealth(evhtp_request_t* req, const std::string& health_uri)
{
  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  std::string mode;
  if ((health_uri.empty()) ||
      (!RE2::FullMatch(health_uri, health_regex_, &mode))) {
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    return;
  }

  TRTSERVER_Error* err = nullptr;
  bool health = false;

  if (mode == "live") {
    err = TRTSERVER_ServerIsLive(server_.get(), &health);
  } else if (mode == "ready") {
    err = TRTSERVER_ServerIsReady(server_.get(), &health);
  } else {
    err = TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_UNKNOWN,
        std::string("unknown health mode '" + mode + "'").c_str());
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
HTTPAPIServer::HandleProfile(
    evhtp_request_t* req, const std::string& profile_uri)
{
  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  if (!profile_uri.empty() && (profile_uri != "/")) {
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    return;
  }

  std::string cmd;
  const char* cmd_c_str = evhtp_kv_find(req->uri->query, "cmd");
  if (cmd_c_str != NULL) {
    cmd = std::string(cmd_c_str);
  }

  // For now profile is a nop...

  RequestStatus request_status;
  RequestStatusUtil::Create(
      &request_status, nullptr /* err */,
      RequestStatusUtil::NextUniqueRequestId(), server_id_);

  evhtp_headers_add_header(
      req->headers_out,
      evhtp_header_new(
          kStatusHTTPHeader, request_status.ShortDebugString().c_str(), 1, 1));

  evhtp_send_reply(
      req, (request_status.code() == RequestStatusCode::SUCCESS)
               ? EVHTP_RES_OK
               : EVHTP_RES_BADREQ);
}

void
HTTPAPIServer::HandleStatus(evhtp_request_t* req, const std::string& status_uri)
{
  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  std::string model_name;
  if (!status_uri.empty()) {
    if (!RE2::FullMatch(status_uri, status_regex_, &model_name)) {
      evhtp_send_reply(req, EVHTP_RES_BADREQ);
      return;
    }
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

void
HTTPAPIServer::HandleModelControl(
    evhtp_request_t* req, const std::string& modelcontrol_uri)
{
  if (req->method != htp_method_POST) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  std::string action_type_str, model_name;
  if ((modelcontrol_uri.empty()) || (!RE2::FullMatch(
                                        modelcontrol_uri, modelcontrol_regex_,
                                        &action_type_str, &model_name))) {
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    return;
  }

  TRTSERVER_Error* err = nullptr;
  if (action_type_str == "load") {
    err = TRTSERVER_ServerLoadModel(server_.get(), model_name.c_str());
  } else if (action_type_str == "unload") {
    err = TRTSERVER_ServerUnloadModel(server_.get(), model_name.c_str());
  } else {
    err = TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_UNKNOWN,
        std::string("unknown action type '" + action_type_str + "'").c_str());
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
      req, (request_status.code() == RequestStatusCode::SUCCESS)
               ? EVHTP_RES_OK
               : EVHTP_RES_BADREQ);

  TRTSERVER_ErrorDelete(err);
}

void
HTTPAPIServer::HandleSharedMemoryControl(
    evhtp_request_t* req, const std::string& sharedmemorycontrol_uri)
{
  if (req->method != htp_method_POST) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  re2::RE2 register_regex_(R"(/([^/]+)/(/[^/]+)/([0-9]+)/([0-9]+))");
  re2::RE2 unregister_regex_(R"(/([^/]+))");

  std::string action_type_str, remaining, name, shm_key;
  std::string offset_str, byte_size_str;
  if ((sharedmemorycontrol_uri.empty()) ||
      (!RE2::FullMatch(
          sharedmemorycontrol_uri, sharedmemorycontrol_regex_, &action_type_str,
          &remaining))) {
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    return;
  } else {
    if (remaining.empty()) {
      if ((action_type_str != "unregisterall") &&
          (action_type_str != "status")) {
        evhtp_send_reply(req, EVHTP_RES_BADREQ);
        return;
      }
    } else {
      if (action_type_str == "register" &&
          (!RE2::FullMatch(
              remaining, register_regex_, &name, &shm_key, &offset_str,
              &byte_size_str))) {
        evhtp_send_reply(req, EVHTP_RES_BADREQ);
        return;
      }
      if (action_type_str == "unregister" &&
          (!RE2::FullMatch(remaining, unregister_regex_, &name))) {
        evhtp_send_reply(req, EVHTP_RES_BADREQ);
        return;
      }
    }
  }

  size_t offset = std::atoll(offset_str.c_str());
  size_t byte_size = std::atoll(byte_size_str.c_str());

  TRTSERVER_Error* err = nullptr;
  TRTSERVER_SharedMemoryBlock* smb = nullptr;

  if (action_type_str == "register") {
    err = smb_manager_->Create(
        &smb, name.c_str(), shm_key.c_str(), offset, byte_size);
    if (err == nullptr) {
      err = TRTSERVER_ServerRegisterSharedMemory(server_.get(), smb);
    }
  } else if (action_type_str == "unregister") {
    err = smb_manager_->Remove(&smb, name.c_str());
    if ((err == nullptr) && (smb != nullptr)) {
      err = TRTSERVER_ServerUnregisterSharedMemory(server_.get(), smb);
      TRTSERVER_Error* del_err = TRTSERVER_SharedMemoryBlockDelete(smb);
      if (del_err != nullptr) {
        LOG_ERROR << "failed to delete shared memory block: "
                  << TRTSERVER_ErrorMessage(del_err);
      }
    }
  } else if (action_type_str == "unregisterall") {
    err = TRTSERVER_ServerUnregisterAllSharedMemory(server_.get());
  } else {
    err = TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_UNKNOWN,
        std::string("unknown action type '" + action_type_str + "'").c_str());
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
      req, (request_status.code() == RequestStatusCode::SUCCESS)
               ? EVHTP_RES_OK
               : EVHTP_RES_BADREQ);

  TRTSERVER_ErrorDelete(err);
}

TRTSERVER_Error*
HTTPAPIServer::EVBufferToInput(
    const std::string& model_name, const InferRequestHeader& request_header,
    evbuffer* input_buffer,
    TRTSERVER_InferenceRequestProvider* request_provider,
    std::unordered_map<std::string, std::pair<const void*, size_t>>&
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
          TRTSERVER_MEMORY_CPU));
    } else {
      // If input is in shared memory then verify that the size is
      // correct and set input from the shared memory.
      if (io.has_shared_memory()) {
        LOG_VERBOSE(1) << io.name() << " has shared memory";
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
        TRTSERVER_SharedMemoryBlock* smb = nullptr;
        RETURN_IF_ERR(smb_manager_->Get(&smb, io.shared_memory().name()));
        RETURN_IF_ERR(TRTSERVER_ServerSharedMemoryAddress(
            server_.get(), smb, io.shared_memory().offset(),
            io.shared_memory().byte_size(), &base));
        RETURN_IF_ERR(TRTSERVER_InferenceRequestProviderSetInputData(
            request_provider, io.name().c_str(), base, byte_size,
            TRTSERVER_MEMORY_CPU));
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
              TRTSERVER_MEMORY_CPU));
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
      LOG_VERBOSE(1) << io.name() << " has shared memory";
      void* base;
      TRTSERVER_SharedMemoryBlock* smb = nullptr;
      RETURN_IF_ERR(smb_manager_->Get(&smb, io.shared_memory().name()));
      RETURN_IF_ERR(TRTSERVER_ServerSharedMemoryAddress(
          server_.get(), smb, io.shared_memory().offset(),
          io.shared_memory().byte_size(), &base));
      output_shm_map.emplace(
          io.name(),
          std::make_pair(
              static_cast<const void*>(base), io.shared_memory().byte_size()));
    }
  }

  return nullptr;  // success
}

void
HTTPAPIServer::HandleInfer(evhtp_request_t* req, const std::string& infer_uri)
{
  if (req->method != htp_method_POST) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  std::string model_name, model_version_str;
  if ((infer_uri.empty()) ||
      (!RE2::FullMatch(
          infer_uri, infer_regex_, &model_name, &model_version_str))) {
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    return;
  }

  int64_t model_version = -1;
  if (!model_version_str.empty()) {
    model_version = std::atoll(model_version_str.c_str());
  }

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
    // std::map<std::string, std::pair<const void*, size_t>>* output_shm_map =
    //     new std::map<std::string, std::pair<const void*, size_t>>;
    err = EVBufferToInput(
        model_name, request_header, req->buffer_in, request_provider,
        response_pair->second);
    if (err == nullptr) {
      InferRequest* infer_request =
          new InferRequest(req, request_header.id(), server_id_, unique_id);

      response_pair->first = req->buffer_out;
      infer_request->response_pair_.reset(response_pair);
      // response_pair->op_shm_map_ = output_shm_map;

      err = TRTSERVER_ServerInferAsync(
          server_.get(), request_provider, allocator_,
          reinterpret_cast<void*>(response_pair), InferRequest::InferComplete,
          reinterpret_cast<void*>(infer_request));
      if (err != nullptr) {
        delete infer_request;
        infer_request = nullptr;
      }

      // The request provider can be deleted immediately after the
      // ServerInferAsync call returns.
      TRTSERVER_InferenceRequestProviderDelete(request_provider);
    }
  }

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
HTTPAPIServer::OKReplyCallback(evthr_t* thr, void* arg, void* shared)
{
  evhtp_request_t* request = (evhtp_request_t*)arg;
  evhtp_send_reply(request, EVHTP_RES_OK);
  evhtp_request_resume(request);
}

void
HTTPAPIServer::BADReplyCallback(evthr_t* thr, void* arg, void* shared)
{
  evhtp_request_t* request = (evhtp_request_t*)arg;
  evhtp_send_reply(request, EVHTP_RES_BADREQ);
  evhtp_request_resume(request);
}

HTTPAPIServer::InferRequest::InferRequest(
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
HTTPAPIServer::InferRequest::InferComplete(
    TRTSERVER_Server* server, TRTSERVER_InferenceResponse* response,
    void* userp)
{
  HTTPAPIServer::InferRequest* infer_request =
      reinterpret_cast<HTTPAPIServer::InferRequest*>(userp);
  if (infer_request->FinalizeResponse(response) == EVHTP_RES_OK) {
    evthr_defer(infer_request->thread_, OKReplyCallback, infer_request->req_);
  } else {
    evthr_defer(infer_request->thread_, BADReplyCallback, infer_request->req_);
  }

  LOG_IF_ERR(
      TRTSERVER_InferenceResponseDelete(response), "deleting HTTP response");
}

evhtp_res
HTTPAPIServer::InferRequest::FinalizeResponse(
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
    const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
    const std::map<int32_t, std::vector<std::string>>& port_map, int thread_cnt,
    std::vector<std::unique_ptr<HTTPServer>>* http_servers)
{
  if (port_map.empty()) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INVALID_ARG,
        "HTTP is enabled but none of the service endpoints have a valid port "
        "assignment");
  }
  http_servers->clear();
  for (auto const& ep_map : port_map) {
    std::string addr = "0.0.0.0:" + std::to_string(ep_map.first);
    LOG_INFO << "Starting HTTPService at " << addr;
    http_servers->emplace_back(new HTTPAPIServer(
        server, smb_manager, ep_map.second, ep_map.first, thread_cnt));
  }

  return nullptr;
}

TRTSERVER_Error*
HTTPServer::CreateMetricsServer(
    const std::shared_ptr<TRTSERVER_Server>& server, const int32_t port,
    const int thread_cnt, std::unique_ptr<HTTPServer>* metrics_server)
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
