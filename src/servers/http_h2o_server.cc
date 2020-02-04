// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <errno.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/json_util.h>
#include <h2o.h>
#include <h2o/cache.h>
#include <h2o/memcached.h>
#include <h2o/serverutil.h>
#include <openssl/ssl.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/signalfd.h>
#include <sys/time.h>
#include <unistd.h>

#include <re2/re2.h>
#include <algorithm>
#include <thread>
#include "src/core/api.pb.h"
#include "src/core/constants.h"
#include "src/core/server_status.pb.h"
#include "src/core/trtserver.h"
#include "src/servers/common.h"

// The HTTP frontend logging is closely related to the server, thus keep it
// using the server logging utils
#include "src/core/logging.h"

#ifdef TRTIS_ENABLE_TRACING
#include "src/servers/tracer.h"
#endif  // TRTIS_ENABLE_TRACING

#define USE_HTTPS 0
#define USE_MEMCACHED 0
#define DEFAULT_CACHE_LINE_SIZE 128

namespace nvidia { namespace inferenceserver {

// Handle HTTP requests to inference server APIs
class HTTPAPIServer : public HTTPServer {
 public:
  explicit HTTPAPIServer(
      const std::shared_ptr<TRTSERVER_Server>& server,
      const std::shared_ptr<nvidia::inferenceserver::TraceManager>&
          trace_manager,
      const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
      const std::vector<std::string>& endpoints, const int32_t port,
      const int thread_cnt)
      : server_(server), trace_manager_(trace_manager),
        smb_manager_(smb_manager), allocator_(nullptr), port_(port),
        thread_cnt_(thread_cnt)
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
    Stop();
    LOG_IF_ERR(
        TRTSERVER_ResponseAllocatorDelete(allocator_),
        "deleting response allocator");
  }

  using H2OBufferPair = std::pair<
      h2o_iovec_t*,
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
        TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
        TRTSERVER_InferenceResponse* response, void* userp);
    evhtp_res FinalizeResponse(TRTSERVER_InferenceResponse* response);

    std::unique_ptr<H2OBufferPair> response_pair_;

   private:
    evhtp_request_t* req_;
    const uint64_t request_id_;
    const char* const server_id_;
    const uint64_t unique_id_;
  };

 private:
  TRTSERVER_Error* Start() override;
  TRTSERVER_Error* Stop() override;

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

  int create_listener(int32_t port);
#if H2O_USE_LIBUV
  static void on_accept(uv_stream_t* listener, int status);
#else
  static void on_accept(h2o_socket_t* listener, const char* err);
#endif

  int setup_ssl(
      const char* cert_file, const char* key_file, const char* ciphers);
  h2o_pathconf_t* register_handler(
      h2o_hostconf_t* hostconf, const char* path,
      int (*on_req)(h2o_handler_t*, h2o_req_t*));

  static int health(h2o_handler_t* self, h2o_req_t* req);
  static int status(h2o_handler_t* self, h2o_req_t* req);
  static int Infer(h2o_handler_t* self, h2o_req_t* req);

  TRTSERVER_Error* H2OBufferToInput(
      const std::string& model_name, const InferRequestHeader& request_header,
      h2o_buffer_t* input_buffer,
      TRTSERVER_InferenceRequestProvider* request_provider,
      std::unordered_map<
          std::string,
          std::tuple<const void*, size_t, TRTSERVER_Memory_Type, int64_t>>&
          output_shm_map);

  h2o_globalconf_t config;
  h2o_context_t ctx;
  h2o_accept_ctx_t accept_ctx;

#if H2O_USE_LIBUV
  uv_tcp_t listener;
#else
  h2o_socket_t* listener_socket;
#endif

  h2o_multithread_receiver_t libmemcached_receiver;
  bool exit_loop = true;

  std::shared_ptr<TRTSERVER_Server> server_;
  const char* server_id_;

  std::shared_ptr<TraceManager> trace_manager_;
  std::shared_ptr<SharedMemoryBlockManager> smb_manager_;

  // The allocator that will be used to allocate buffers for the
  // inference result tensors.
  TRTSERVER_ResponseAllocator* allocator_;

  int32_t port_;
  int thread_cnt_;
};

struct h2o_custom_req_handler_t {
  h2o_handler_t super;
  HTTPAPIServer* http_server;
};

#if H2O_USE_LIBUV

void
HTTPAPIServer::on_accept(uv_stream_t* listener, int status)
{
  HTTPAPIServer* http_server = reinterpret_cast<HTTPAPIServer*>(listener->data);
  uv_tcp_t* conn;
  h2o_socket_t* sock;

  if (status != 0)
    return;

  conn = reinterpret_cast<uv_tcp_t*>(h2o_mem_alloc(sizeof(*conn)));
  uv_tcp_init(listener->loop, conn);

  if (uv_accept(listener, (uv_stream_t*)conn) != 0) {
    uv_close((uv_handle_t*)conn, (uv_close_cb)free);
    return;
  }

  sock = h2o_uv_socket_create((uv_stream_t*)conn, (uv_close_cb)free);
  h2o_accept(&http_server->accept_ctx, sock);
}

int
HTTPAPIServer::create_listener(int32_t port)
{
  struct sockaddr_in addr;
  int r;

  uv_tcp_init(ctx.loop, &listener);
  uv_ip4_addr("0.0.0.0", port, &addr);
  if ((r = uv_tcp_bind(&listener, (struct sockaddr*)&addr, 0)) != 0) {
    LOG_VERBOSE(1) << stderr << "uv_tcp_bind:" << uv_strerror(r);
    goto Error;
  }
  if ((r = uv_listen((uv_stream_t*)&listener, 128, on_accept)) != 0) {
    LOG_VERBOSE(1) << stderr << "uv_listen:" << uv_strerror(r);
    goto Error;
  }

  listener.data = this;

  return 0;
Error:
  uv_close((uv_handle_t*)&listener, NULL);
  return r;
}

#else

void
HTTPAPIServer::on_accept(h2o_socket_t* listener, const char* err)
{
  HTTPAPIServer* http_server = reinterpret_cast<HTTPAPIServer*>(listener->data);
  h2o_socket_t* sock;

  if (err != NULL) {
    return;
  }

  if ((sock = h2o_evloop_socket_accept(listener)) == NULL)
    return;
  h2o_accept(&http_server->accept_ctx, sock);
}

int
HTTPAPIServer::create_listener(int32_t port)
{
  struct sockaddr_in addr;
  int fd, reuseaddr_flag = 1;

  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(0x7f000001);
  addr.sin_port = htons(port);

  if ((fd = socket(AF_INET, SOCK_STREAM, 0)) == -1 ||
      setsockopt(
          fd, SOL_SOCKET, SO_REUSEADDR, &reuseaddr_flag,
          sizeof(reuseaddr_flag)) != 0 ||
      bind(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0 ||
      listen(fd, SOMAXCONN) != 0) {
    return -1;
  }

  listener_socket =
      h2o_evloop_socket_create(ctx.loop, fd, H2O_SOCKET_FLAG_DONT_READ);
  listener_socket->data = this;
  h2o_socket_read_start(listener_socket, on_accept);

  return 0;
}
#endif

int
HTTPAPIServer::setup_ssl(
    const char* cert_file, const char* key_file, const char* ciphers)
{
  SSL_load_error_strings();
  SSL_library_init();
  OpenSSL_add_all_algorithms();

  accept_ctx.ssl_ctx = SSL_CTX_new(SSLv23_server_method());
  SSL_CTX_set_options(accept_ctx.ssl_ctx, SSL_OP_NO_SSLv2);

#ifdef SSL_CTX_set_ecdh_auto
  SSL_CTX_set_ecdh_auto(accept_ctx.ssl_ctx, 1);
#endif

  /* load certificate and private key */
  if (SSL_CTX_use_certificate_chain_file(accept_ctx.ssl_ctx, cert_file) != 1) {
    fprintf(
        stderr,
        "an error occurred while trying to load server certificate file:%s\n",
        cert_file);
    return -1;
  }
  if (SSL_CTX_use_PrivateKey_file(
          accept_ctx.ssl_ctx, key_file, SSL_FILETYPE_PEM) != 1) {
    fprintf(
        stderr, "an error occurred while trying to load private key file:%s\n",
        key_file);
    return -1;
  }

  if (SSL_CTX_set_cipher_list(accept_ctx.ssl_ctx, ciphers) != 1) {
    fprintf(stderr, "ciphers could not be set: %s\n", ciphers);
    return -1;
  }

/* setup protocol negotiation methods */
#if H2O_USE_NPN
  h2o_ssl_register_npn_protocols(accept_ctx.ssl_ctx, h2o_http2_npn_protocols);
#endif
#if H2O_USE_ALPN
  h2o_ssl_register_alpn_protocols(accept_ctx.ssl_ctx, h2o_http2_alpn_protocols);
#endif

  return 0;
}

h2o_pathconf_t*
HTTPAPIServer::register_handler(
    h2o_hostconf_t* hostconf, const char* path,
    int (*on_req)(h2o_handler_t*, h2o_req_t*))
{
  h2o_pathconf_t* pathconf = h2o_config_register_path(hostconf, path, 0);
  h2o_custom_req_handler_t* handler =
      reinterpret_cast<h2o_custom_req_handler_t*>(
          h2o_create_handler(pathconf, sizeof(*handler)));
  handler->http_server = this;
  handler->super.on_req = on_req;
  return pathconf;
}

int
HTTPAPIServer::health(h2o_handler_t* _self, h2o_req_t* req)
{
  h2o_custom_req_handler_t* self = (h2o_custom_req_handler_t*)_self;
  if (!h2o_memis(req->method.base, req->method.len, H2O_STRLIT("GET"))) {
    req->res.status = 400;
    req->res.reason = "Only GET method is allowed";
    h2o_add_header(
        &req->pool, &req->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL,
        H2O_STRLIT("text/plain"));
    h2o_generator_t generator = {NULL, NULL};
    h2o_iovec_t body = h2o_strdup(&req->pool, "", SIZE_MAX);
    h2o_start_response(req, &generator);
    h2o_send(req, &body, 1, H2O_SEND_STATE_FINAL);
    return 0;
  }

  re2::RE2 health_regex_(R"(/api/health/(live|ready))");
  std::string health_uri =
      std::string(req->path_normalized.base, req->path_normalized.len);
  std::string mode;
  if ((health_uri.empty()) ||
      (!RE2::FullMatch(health_uri, health_regex_, &mode))) {
    req->res.status = 400;
    req->res.reason = "Bad request";
    h2o_add_header(
        &req->pool, &req->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL,
        H2O_STRLIT("text/plain"));
    h2o_generator_t generator = {NULL, NULL};
    h2o_iovec_t body = h2o_strdup(&req->pool, "", SIZE_MAX);
    h2o_start_response(req, &generator);
    h2o_send(req, &body, 1, H2O_SEND_STATE_FINAL);
    return 0;
  }

  TRTSERVER_Error* err = nullptr;
  bool health = false;

  if (mode == "live") {
    err = TRTSERVER_ServerIsLive(self->http_server->server_.get(), &health);
  } else if (mode == "ready") {
    err = TRTSERVER_ServerIsReady(self->http_server->server_.get(), &health);
  }

  h2o_generator_t generator = {NULL, NULL};
  h2o_iovec_t body = h2o_strdup(&req->pool, "", SIZE_MAX);

  RequestStatus request_status;
  RequestStatusUtil::Create(
      &request_status, err, RequestStatusUtil::NextUniqueRequestId(),
      self->http_server->server_id_);

  // Add NV-Status header
  std::string status_header = std::string(kStatusHTTPHeader);
  h2o_iovec_t status_header_content = h2o_strdup(
      &req->pool, request_status.ShortDebugString().c_str(), SIZE_MAX);
  h2o_add_header_by_str(
      &req->pool, &req->res.headers, status_header.c_str(),
      status_header.size(), 0, NULL, status_header_content.base,
      status_header_content.len);
  h2o_add_header(
      &req->pool, &req->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL,
      H2O_STRLIT("text/plain"));

  if (health && (err == nullptr)) {
    req->res.status = 200;
    req->res.reason = "OK";
    req->res.content_length = body.len;
  } else {
    req->res.status = 400;
    req->res.reason = "Bad request";
  }

  h2o_start_response(req, &generator);
  h2o_send(req, &body, 1, H2O_SEND_STATE_FINAL);
  return 0;
}

int
HTTPAPIServer::status(h2o_handler_t* _self, h2o_req_t* req)
{
  h2o_custom_req_handler_t* self = (h2o_custom_req_handler_t*)_self;
  re2::RE2 status_regex(R"(/api/status(/(.*))?)");
  if (!h2o_memis(req->method.base, req->method.len, H2O_STRLIT("GET"))) {
    req->res.status = 400;
    req->res.reason = "Only GET method is allowed";
    h2o_add_header(
        &req->pool, &req->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL,
        H2O_STRLIT("text/plain"));
    h2o_generator_t generator = {NULL, NULL};
    h2o_iovec_t body = h2o_strdup(&req->pool, "", SIZE_MAX);
    h2o_start_response(req, &generator);
    h2o_send(req, &body, 1, H2O_SEND_STATE_FINAL);
    return 0;
  }

  std::string status_uri =
      std::string(req->path_normalized.base, req->path_normalized.len);
  std::string model_name, format = "text";
  if (!status_uri.empty()) {
    if (!RE2::FullMatch(status_uri, status_regex, &model_name)) {
      req->res.status = 400;
      req->res.reason = "Bad request";
      h2o_add_header(
          &req->pool, &req->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL,
          H2O_STRLIT("text/plain"));
      h2o_generator_t generator = {NULL, NULL};
      h2o_iovec_t body = h2o_strdup(&req->pool, "", SIZE_MAX);
      h2o_start_response(req, &generator);
      h2o_send(req, &body, 1, H2O_SEND_STATE_FINAL);
      return 0;
    }
  }

  size_t query_len = req->path.len - req->query_at;
  if (req->query_at != SIZE_MAX && (query_len > 1)) {
    if (h2o_memis(&req->path.base[req->query_at], 8, "?format=", 8)) {
      format = std::string(&req->path.base[req->query_at + 8]);
      format = format.substr(0, format.find(' '));
    } else {
      req->res.status = 400;
      req->res.reason = "Bad request";
      h2o_add_header(
          &req->pool, &req->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL,
          H2O_STRLIT("text/plain"));
      h2o_generator_t generator = {NULL, NULL};
      h2o_iovec_t body = h2o_strdup(&req->pool, "", SIZE_MAX);
      h2o_start_response(req, &generator);
      h2o_send(req, &body, 1, H2O_SEND_STATE_FINAL);
      return 0;
    }
  }

  // if accept: application/json then override format from query
  ssize_t accept_cursor =
      h2o_find_header_by_str(&req->headers, H2O_STRLIT("accept"), -1);
  if (accept_cursor != -1) {
    h2o_iovec_t* slot = &req->headers.entries[accept_cursor].value;
    std::string accept_header = std::string(slot->base, slot->len);
    if (accept_header == "application/json") {
      format = "json";
    }
  }

  TRTSERVER_Protobuf* server_status_protobuf = nullptr;
  TRTSERVER_Error* err =
      (model_name.empty())
          ? TRTSERVER_ServerStatus(
                self->http_server->server_.get(), &server_status_protobuf)
          : TRTSERVER_ServerModelStatus(
                self->http_server->server_.get(), model_name.c_str(),
                &server_status_protobuf);

  h2o_generator_t generator = {NULL, NULL};
  h2o_iovec_t body;

  if (err == nullptr) {
    const char* status_buffer;
    size_t status_byte_size;
    err = TRTSERVER_ProtobufSerialize(
        server_status_protobuf, &status_buffer, &status_byte_size);
    if (err == nullptr) {
      // Request text, binary or json format for status
      if (format == "binary") {
        h2o_add_header(
            &req->pool, &req->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL,
            H2O_STRLIT("application/octet-stream"));
        body = h2o_strdup(&req->pool, status_buffer, SIZE_MAX);
      } else {
        ServerStatus server_status;
        if (!server_status.ParseFromArray(status_buffer, status_byte_size)) {
          err = TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_UNKNOWN, "failed to parse server status");
        } else {
          if (format == "json") {
            std::string server_status_json;
            ::google::protobuf::util::MessageToJsonString(
                server_status, &server_status_json);
            h2o_add_header(
                &req->pool, &req->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL,
                H2O_STRLIT("application/json"));
            body = h2o_strdup(&req->pool, server_status_json.c_str(), SIZE_MAX);
          } else {
            std::string server_status_str = server_status.DebugString();
            h2o_add_header(
                &req->pool, &req->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL,
                H2O_STRLIT("text/plain"));
            body = h2o_strdup(&req->pool, server_status_str.c_str(), SIZE_MAX);
          }
        }
      }
    }
  }

  TRTSERVER_ProtobufDelete(server_status_protobuf);

  RequestStatus request_status;
  RequestStatusUtil::Create(
      &request_status, err, RequestStatusUtil::NextUniqueRequestId(),
      self->http_server->server_id_);

  // Add NV-Status header
  std::string status_header = std::string(kStatusHTTPHeader);
  h2o_iovec_t status_header_content = h2o_strdup(
      &req->pool, request_status.ShortDebugString().c_str(), SIZE_MAX);
  h2o_add_header_by_str(
      &req->pool, &req->res.headers, status_header.c_str(),
      status_header.size(), 0, NULL, status_header_content.base,
      status_header_content.len);

  if (err == nullptr) {
    req->res.status = 200;
    req->res.reason = "OK";
    req->res.content_length = body.len;
  } else {
    req->res.status = 400;
    req->res.reason = "Bad Request";
  }

  h2o_start_response(req, &generator);
  h2o_send(req, &body, 1, H2O_SEND_STATE_FINAL);
  TRTSERVER_ErrorDelete(err);

  return 0;
}

TRTSERVER_Error*
HTTPAPIServer::EVBufferToInput(
    const std::string& model_name, const InferRequestHeader& request_header,
    h2o_iovec_t* input_buffer,
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
  int v_idx = 0;
  int n = input_buffer->len;

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
        // if (byte_size != io.shared_memory().byte_size()) {
        //   return TRTSERVER_ErrorNew(
        //       TRTSERVER_ERROR_INVALID_ARG,
        //       std::string(
        //           "unexpected shared-memory size " +
        //           std::to_string(io.shared_memory().byte_size()) +
        //           " for input '" + io.name() + "', expecting " +
        //           std::to_string(byte_size) + " for model '" + model_name +
        //           "'") .c_str());
        // }
        //
        // void* base;
        // TRTSERVER_Memory_Type memory_type = TRTSERVER_MEMORY_CPU;
        // int64_t memory_type_id;
        // TRTSERVER_SharedMemoryBlock* smb = nullptr;
        // RETURN_IF_ERR(smb_manager_->Get(&smb, io.shared_memory().name()));
        // RETURN_IF_ERR(TRTSERVER_ServerSharedMemoryAddress(
        //     server_.get(), smb, io.shared_memory().offset(),
        //     io.shared_memory().byte_size(), &base));
        // TRTSERVER_SharedMemoryBlockMemoryType(smb, &memory_type);
        // TRTSERVER_SharedMemoryBlockMemoryTypeId(smb, &memory_type_id);
        // RETURN_IF_ERR(TRTSERVER_InferenceRequestProviderSetInputData(
        //     request_provider, io.name().c_str(), base, byte_size,
        //     memory_type, memory_type_id));
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

  // // Initialize System Memory for Output if it uses shared memory
  // for (const auto& io : request_header.output()) {
  //   if (io.has_shared_memory()) {
  //     void* base;
  //     TRTSERVER_SharedMemoryBlock* smb = nullptr;
  //     RETURN_IF_ERR(smb_manager_->Get(&smb, io.shared_memory().name()));
  //     RETURN_IF_ERR(TRTSERVER_ServerSharedMemoryAddress(
  //         server_.get(), smb, io.shared_memory().offset(),
  //         io.shared_memory().byte_size(), &base));
  //
  //     TRTSERVER_Memory_Type memory_type;
  //     int64_t memory_type_id;
  //     TRTSERVER_SharedMemoryBlockMemoryType(smb, &memory_type);
  //     TRTSERVER_SharedMemoryBlockMemoryTypeId(smb, &memory_type_id);
  //     output_shm_map.emplace(
  //         io.name(),
  //         std::make_tuple(
  //             static_cast<const void*>(base), io.shared_memory().byte_size(),
  //             memory_type, memory_type_id));
  //   }
  // }

  return nullptr;  // success
}

int
HTTPAPIServer::Infer(h2o_handler_t* _self, h2o_req_t* req)
{
  h2o_custom_req_handler_t* self = (h2o_custom_req_handler_t*)_self;
  re2::RE2 infer_regex(R"(/api/infer/([^/]+)(?:/(\d+))?)");
  if (!h2o_memis(req->method.base, req->method.len, H2O_STRLIT("POST"))) {
    req->res.status = 400;
    req->res.reason = "Only POST method is allowed";
    h2o_add_header(
        &req->pool, &req->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL,
        H2O_STRLIT("text/plain"));
    h2o_generator_t generator = {NULL, NULL};
    h2o_iovec_t body = h2o_strdup(&req->pool, "", SIZE_MAX);
    h2o_start_response(req, &generator);
    h2o_send(req, &body, 1, H2O_SEND_STATE_FINAL);
    return 0;
  }

  std::string infer_uri =
      std::string(req->path_normalized.base, req->path_normalized.len);
  std::string model_name, model_version_str;
  if ((infer_uri.empty()) ||
      (!RE2::FullMatch(
          infer_uri, infer_regex, &model_name, &model_version_str))) {
    req->res.status = 400;
    req->res.reason = "Bad request";
    h2o_add_header(
        &req->pool, &req->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL,
        H2O_STRLIT("text/plain"));
    h2o_generator_t generator = {NULL, NULL};
    h2o_iovec_t body = h2o_strdup(&req->pool, "", SIZE_MAX);
    h2o_start_response(req, &generator);
    h2o_send(req, &body, 1, H2O_SEND_STATE_FINAL);
    return 0;
  }

  int64_t model_version = -1;
  if (!model_version_str.empty()) {
    model_version = std::atoll(model_version_str.c_str());
  }

  ssize_t infer_request_cursor = h2o_find_header_by_str(
      &req->headers, H2O_STRLIT(kInferRequestHTTPHeader), -1);
  if (infer_request_cursor != -1) {
    h2o_iovec_t* slot = &req->headers.entries[infer_request_cursor].value;
    std::string infer_request_header = std::string(slot->base, slot->len);
    InferRequestHeader request_header_protobuf;
    if (!google::protobuf::TextFormat::ParseFromString(
            infer_request_header, &request_header_protobuf)) {
      req->res.status = 400;
      req->res.reason = "Bad request";
      h2o_add_header(
          &req->pool, &req->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL,
          H2O_STRLIT("text/plain"));
      h2o_generator_t generator = {NULL, NULL};
      h2o_iovec_t body = h2o_strdup(&req->pool, "", SIZE_MAX);
      h2o_start_response(req, &generator);
      h2o_send(req, &body, 1, H2O_SEND_STATE_FINAL);
      return 0;
    }
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
    H2OBufferPair* response_pair(new H2OBufferPair());
    err = H2OBufferToInput(
        model_name, request_header_protobuf, req->entity, request_provider,
        response_pair->second);
    if (err == nullptr) {
      InferRequest* infer_request = new InferRequest(
          req, request_header_protobuf.id(), server_id_, unique_id);

      response_pair->first = req->buffer_out;
      infer_request->response_pair_.reset(response_pair);

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
  TRTSERVER_InferenceRequestOptionsDelete(request_options);

  h2o_generator_t generator = {NULL, NULL};
  h2o_iovec_t body;

  RequestStatus request_status;
  RequestStatusUtil::Create(
      &request_status, err, RequestStatusUtil::NextUniqueRequestId(),
      self->http_server->server_id_);

  // Add NV-Status header
  std::string status_header = std::string(kStatusHTTPHeader);
  h2o_iovec_t status_header_content = h2o_strdup(
      &req->pool, request_status.ShortDebugString().c_str(), SIZE_MAX);
  h2o_add_header_by_str(
      &req->pool, &req->res.headers, status_header.c_str(),
      status_header.size(), 0, NULL, status_header_content.base,
      status_header_content.len);

  if (err == nullptr) {
    req->res.status = 200;
    req->res.reason = "OK";
    req->res.content_length = body.len;
  } else {
    req->res.status = 400;
    req->res.reason = "Bad Request";
  }

  h2o_start_response(req, &generator);
  h2o_send(req, &body, 1, H2O_SEND_STATE_FINAL);
  TRTSERVER_ErrorDelete(err);

  return 0;
}

TRTSERVER_Error*
HTTPAPIServer::Start()
{
  h2o_hostconf_t* hostconf;
  h2o_access_log_filehandle_t* logfh = h2o_access_log_open_handle(
      "/dev/stdout", NULL, H2O_LOGCONF_ESCAPE_APACHE);

  signal(SIGPIPE, SIG_IGN);

  // necessary to zero these structs before using them!
  memset(&accept_ctx, 0, sizeof(accept_ctx));
  memset(&ctx, 0, sizeof(ctx));
  memset(&config, 0, sizeof(config));

  h2o_config_init(&config);
  hostconf = h2o_config_register_host(
      &config, h2o_iovec_init(H2O_STRLIT("default")), 65535);

  // handler->on_req(st_h2o_handler_t(health));
  h2o_pathconf_t* pathconf = register_handler(hostconf, "/api/health", health);
  if (logfh != NULL)
    h2o_access_log_register(pathconf, logfh);

  pathconf = register_handler(hostconf, "/api/status", status);
  if (logfh != NULL)
    h2o_access_log_register(pathconf, logfh);

  pathconf = register_handler(hostconf, "/api/infer", infer);
  if (logfh != NULL)
    h2o_access_log_register(pathconf, logfh);

#if H2O_USE_LIBUV
  uv_loop_t loop;
  uv_loop_init(&loop);
  h2o_context_init(&ctx, &loop, &config);
#else
  h2o_context_init(&ctx, h2o_evloop_create(), &config);
#endif

  if (USE_MEMCACHED)
    h2o_multithread_register_receiver(
        ctx.queue, &libmemcached_receiver, h2o_memcached_receiver);

  // TODO Add server.crt and key files
  if (USE_HTTPS && setup_ssl(
                       "h2o/server.crt", "h2o/server.key",
                       "DEFAULT:!MD5:!DSS:!DES:!RC4:!RC2:!SEED:!IDEA:!NULL:!"
                       "ADH:!EXP:!SRP:!PSK") != 0)
    goto Error;

  accept_ctx.ctx = &ctx;
  accept_ctx.hosts = config.hosts;

  if (create_listener(port_) != 0) {
    LOG_VERBOSE(1) << stderr << "failed to listen to 0.0.0.0:" << port_ << ":"
                   << strerror(errno);
    goto Error;
  }

  exit_loop = false;
  while (!exit_loop) {
#if H2O_USE_LIBUV
    uv_run(ctx.loop, UV_RUN_DEFAULT);
#else
    h2o_evloop_run(ctx.loop, INT32_MAX);
#endif
  }

  return nullptr;

Error:
  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_INTERNAL, "HTTP h2o server failed to run.");
}

TRTSERVER_Error*
HTTPAPIServer::Stop()
{
  if (!exit_loop) {
#if H2O_USE_LIBUV
    uv_close((uv_handle_t*)&listener, NULL);
#else
    h2o_socket_read_stop(listener_socket);
    h2o_socket_close(listener_socket);
#endif

    // this will break the event loop
    exit_loop = true;
  } else {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_UNAVAILABLE, "HTTP h2o server is not running.");
  }

  return nullptr;
}

TRTSERVER_Error*
HTTPAPIServer::ResponseAlloc(
    TRTSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRTSERVER_Memory_Type preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRTSERVER_Memory_Type* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  return nullptr;  // Success
}

TRTSERVER_Error*
HTTPAPIServer::ResponseRelease(
    TRTSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRTSERVER_Memory_Type memory_type, int64_t memory_type_id)
{
  LOG_VERBOSE(1) << "HTTP h2o release: "
                 << "size " << byte_size << ", addr " << buffer;

  // Don't do anything when releasing a buffer since ResponseAlloc
  // wrote directly into the response ebvuffer.
  return nullptr;  // Success
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
        "HTTP is enabled but none of the service endpoints have a valid "
        "port assignment");
  }
  http_servers->clear();
  for (auto const& ep_map : port_map) {
    std::string addr = "0.0.0.0:" + std::to_string(ep_map.first);
    LOG_INFO << "Starting HTTPService at " << addr;
    http_servers->emplace_back(new HTTPAPIServer(
        server, trace_manager, smb_manager, ep_map.second, ep_map.first,
        thread_cnt));
  }

  return nullptr;
}

}}  // namespace nvidia::inferenceserver
