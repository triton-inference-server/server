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
#include <google/protobuf/util/json_util.h>
#include <h2o.h>
#include <h2o/cache.h>
#include <h2o/memcached.h>
#include <h2o/serverutil.h>
#include <openssl/ssl.h>
#include <pthread.h>
#include <re2/re2.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/signalfd.h>
#include <sys/time.h>
#include <unistd.h>
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

namespace nvidia { namespace inferenceserver {

// Handle HTTP requests to inference server APIs
class HTTPAPIServer : public HTTPServer {
 public:
  HTTPAPIServer(
      const std::shared_ptr<TRTSERVER_Server>& server,
      const std::shared_ptr<nvidia::inferenceserver::TraceManager>&
          trace_manager,
      const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
      const std::vector<std::string>& endpoints, const int32_t port);
  ~HTTPAPIServer();

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

  static void OnAccept(uv_stream_t* listener, int status);
  int SetupSsl(
      const std::string& cert_file, const std::string& key_file,
      const std::string& ciphers);
  h2o_pathconf_t* RegisterHandler(
      h2o_hostconf_t* hostconf, const char* path,
      int (*on_req)(h2o_handler_t*, h2o_req_t*));

  static int Health(h2o_handler_t* handler_self, h2o_req_t* req);
  static int Status(h2o_handler_t* handler_self, h2o_req_t* req);

  h2o_globalconf_t config_;
  h2o_context_t ctx_;
  h2o_accept_ctx_t accept_ctx_;
  uv_tcp_t listener_;
  bool exit_loop_ = true;

  std::shared_ptr<TRTSERVER_Server> server_;
  const char* server_id_;

  std::shared_ptr<TraceManager> trace_manager_;
  std::shared_ptr<SharedMemoryBlockManager> smb_manager_;
  std::vector<std::string> endpoint_names_;

  // The allocator that will be used to allocate buffers for the
  // inference result tensors.
  TRTSERVER_ResponseAllocator* allocator_;

  int32_t port_;
  std::thread worker_;
};

HTTPAPIServer::HTTPAPIServer(
    const std::shared_ptr<TRTSERVER_Server>& server,
    const std::shared_ptr<nvidia::inferenceserver::TraceManager>& trace_manager,
    const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
    const std::vector<std::string>& endpoints, const int32_t port)
    : server_(server), trace_manager_(trace_manager), smb_manager_(smb_manager),
      endpoint_names_(endpoints), allocator_(nullptr), port_(port)
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

HTTPAPIServer::~HTTPAPIServer()
{
  Stop();
  LOG_IF_ERR(
      TRTSERVER_ResponseAllocatorDelete(allocator_),
      "deleting response allocator");
}

struct h2o_custom_req_handler_t {
  h2o_handler_t super;
  HTTPAPIServer* http_server;
};

void
HTTPAPIServer::OnAccept(uv_stream_t* listener, int status)
{
  HTTPAPIServer* http_server = reinterpret_cast<HTTPAPIServer*>(listener->data);

  if (status != 0) {
    return;
  }

  uv_tcp_t* conn = reinterpret_cast<uv_tcp_t*>(h2o_mem_alloc(sizeof(*conn)));
  uv_tcp_init(http_server->ctx_.loop, conn);

  if (uv_accept(listener, (uv_stream_t*)conn) != 0) {
    uv_close((uv_handle_t*)conn, (uv_close_cb)free);
    return;
  }

  h2o_socket_t* sock =
      h2o_uv_socket_create((uv_stream_t*)conn, (uv_close_cb)free);
  h2o_accept(&http_server->accept_ctx_, sock);
}

#ifdef USE_HTTPS
int
HTTPAPIServer::SetupSsl(
    const std::string& cert_file, const std::string& key_file,
    const std::string& ciphers)
{
  SSL_load_error_strings();
  SSL_library_init();
  OpenSSL_add_all_algorithms();

  accept_ctx_.ssl_ctx = SSL_CTX_new(SSLv23_server_method());
  SSL_CTX_set_options(accept_ctx_.ssl_ctx, SSL_OP_NO_SSLv2);

#ifdef SSL_CTX_set_ecdh_auto
  SSL_CTX_set_ecdh_auto(accept_ctx_.ssl_ctx, 1);
#endif

  /* load certificate and private key */
  if (SSL_CTX_use_certificate_chain_file(
          accept_ctx_.ssl_ctx, cert_file.c_str()) != 1) {
    fprintf(
        stderr,
        "an error occurred while trying to load server certificate file:%s\n",
        cert_file.c_str());
    return -1;
  }
  if (SSL_CTX_use_PrivateKey_file(
          accept_ctx_.ssl_ctx, key_file.c_str(), SSL_FILETYPE_PEM) != 1) {
    fprintf(
        stderr, "an error occurred while trying to load private key file:%s\n",
        key_file.c_str());
    return -1;
  }

  if (SSL_CTX_set_cipher_list(accept_ctx_.ssl_ctx, ciphers.c_str()) != 1) {
    fprintf(stderr, "ciphers could not be set: %s\n", ciphers.c_str());
    return -1;
  }

  /* setup protocol negotiation methods */
#if H2O_USE_NPN
  h2o_ssl_register_npn_protocols(accept_ctx_.ssl_ctx, h2o_http2_npn_protocols);
#endif
#if H2O_USE_ALPN
  h2o_ssl_register_alpn_protocols(
      accept_ctx_.ssl_ctx, h2o_http2_alpn_protocols);
#endif

  return 0;
}
#endif  // USE_HTTPS

h2o_pathconf_t*
HTTPAPIServer::RegisterHandler(
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
HTTPAPIServer::Health(h2o_handler_t* handler_self, h2o_req_t* req)
{
  h2o_custom_req_handler_t* self = (h2o_custom_req_handler_t*)handler_self;
  if (!h2o_memis(req->method.base, req->method.len, H2O_STRLIT("GET"))) {
    req->res.status = 400;
    req->res.reason = "Only GET method is allowed";
    h2o_add_header(
        &req->pool, &req->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL,
        H2O_STRLIT("text/plain"));
    h2o_generator_t generator = {NULL, NULL};
    h2o_iovec_t body = h2o_strdup(&req->pool, "", SIZE_MAX);
    h2o_start_response(req, &generator);
    h2o_send(req, &body, 1 /* buffer count */, H2O_SEND_STATE_FINAL);
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
    h2o_send(req, &body, 1 /* buffer count */, H2O_SEND_STATE_FINAL);
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
  h2o_send(req, &body, 1 /* buffer count */, H2O_SEND_STATE_FINAL);
  return 0;
}

int
HTTPAPIServer::Status(h2o_handler_t* handler_self, h2o_req_t* req)
{
  h2o_custom_req_handler_t* self = (h2o_custom_req_handler_t*)handler_self;
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
    h2o_send(req, &body, 1 /* buffer count */, H2O_SEND_STATE_FINAL);
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
      h2o_send(req, &body, 1 /* buffer count */, H2O_SEND_STATE_FINAL);
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
      h2o_send(req, &body, 1 /* buffer count */, H2O_SEND_STATE_FINAL);
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
  h2o_send(req, &body, 1 /* buffer count */, H2O_SEND_STATE_FINAL);
  TRTSERVER_ErrorDelete(err);

  return 0;
}

TRTSERVER_Error*
HTTPAPIServer::Start()
{
  if (!worker_.joinable()) {
    h2o_hostconf_t* hostconf;
    h2o_pathconf_t* pathconf;
    h2o_access_log_filehandle_t* logfh = h2o_access_log_open_handle(
        "/dev/stdout", NULL, H2O_LOGCONF_ESCAPE_APACHE);

    // Needed by h2o to know when to terminate
    signal(SIGPIPE, SIG_IGN);

    // necessary to zero these structs before using them!
    memset(&accept_ctx_, 0, sizeof(accept_ctx_));
    memset(&ctx_, 0, sizeof(ctx_));
    memset(&config_, 0, sizeof(config_));

    h2o_config_init(&config_);
    hostconf = h2o_config_register_host(
        &config_, h2o_iovec_init(H2O_STRLIT("default")), 65535);

    if (std::find(endpoint_names_.begin(), endpoint_names_.end(), "health") !=
        endpoint_names_.end()) {
      pathconf = RegisterHandler(hostconf, "/api/health", Health);
      if (logfh != NULL)
        h2o_access_log_register(pathconf, logfh);
    }

    if (std::find(endpoint_names_.begin(), endpoint_names_.end(), "status") !=
        endpoint_names_.end()) {
      pathconf = RegisterHandler(hostconf, "/api/status", Status);
      if (logfh != NULL)
        h2o_access_log_register(pathconf, logfh);
    }

    exit_loop_ = false;
    worker_ = std::thread([&] {
      uv_loop_t loop;
      uv_loop_init(&loop);
      h2o_context_init(&ctx_, &loop, &config_);

#ifdef USE_HTTPS
      // TODO Add server.crt and key files
      if (SetupSsl(
              "h2o/server.crt", "h2o/server.key",
              "DEFAULT:!MD5:!DSS:!DES:!RC4:!RC2:!SEED:!IDEA:!NULL:!"
              "ADH:!EXP:!SRP:!PSK") != 0) {
        return TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INTERNAL, "HTTP h2o server failed to run.");
      }
#endif

      accept_ctx_.ctx = &ctx_;
      accept_ctx_.hosts = config_.hosts;

      // Create listener
      struct sockaddr_in addr;
      uv_tcp_init(ctx_.loop, &listener_);
      uv_ip4_addr("0.0.0.0", port_, &addr);
      int r = uv_tcp_bind(&listener_, (struct sockaddr*)&addr, 0);
      if (r != 0) {
        uv_close((uv_handle_t*)&listener_, NULL);
        return;
      }

      if ((r = uv_listen((uv_stream_t*)&listener_, 128, OnAccept)) != 0) {
        uv_close((uv_handle_t*)&listener_, NULL);
        return;
      }

      listener_.data = this;
      uv_run(ctx_.loop, UV_RUN_DEFAULT);
    });

    return nullptr;
  }

  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_INTERNAL, "HTTP h2o server is already running.");
}

TRTSERVER_Error*
HTTPAPIServer::Stop()
{
  exit_loop_ = true;
  if (worker_.joinable()) {
    uv_stop(ctx_.loop);
    uv_loop_close(uv_default_loop());
    worker_.join();
    return nullptr;
  }

  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNAVAILABLE, "HTTP h2o server is not running.");
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
    const std::map<int32_t, std::vector<std::string>>& port_map,
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
        server, trace_manager, smb_manager, ep_map.second, ep_map.first));
  }

  return nullptr;
}

}}  // namespace nvidia::inferenceserver
