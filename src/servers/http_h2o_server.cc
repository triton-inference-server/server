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

// Generic HTTP server
// class HTTPServerImpl : public HTTPServer {
//  public:
//   explicit HTTPServerImpl(const int32_t port, const int thread_cnt)
//       : port_(port), thread_cnt_(thread_cnt)
//   {
//   }
//
//   // static void Dispatch(evhtp_request_t* req, void* arg);
//
//   // TRTSERVER_Error* Start() override;
//   // TRTSERVER_Error* Stop() override;
//
//  protected:
//   // virtual void Handle(evhtp_request_t* req) = 0;
//   static void StopCallback(int sock, short events, void* arg);
//
//   int32_t port_;
//   int thread_cnt_;
// };

// void
// HTTPServerImpl::StopCallback(int sock, short events, void* arg)
// {
//   struct event_base* base = (struct event_base*)arg;
//   event_base_loopbreak(base);
// }

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
    LOG_IF_ERR(
        TRTSERVER_ResponseAllocatorDelete(allocator_),
        "deleting response allocator");
    Stop();
  }

  // using EVBufferTuple = std::tuple<
  //     evbuffer*,
  //     std::unordered_map<
  //         std::string,
  //         std::tuple<const void*, size_t, TRTSERVER_Memory_Type, int64_t>>,
  //     InferRequest>;

  // Class object associated to evhtp thread, requests received are bounded
  // with the thread that accepts it. Need to keep track of that and let the
  // corresponding thread send back the reply
  //   class InferRequestClass {
  //    public:
  //     InferRequestClass(
  //         h2o_req_t* req, uint64_t request_id, const char* server_id,
  //         uint64_t unique_id);
  //     ~InferRequestClass() = default;
  //
  //     evhtp_request_t* EvHtpRequest() const { return req_; }
  //
  //     static void InferComplete(
  //         TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
  //         TRTSERVER_InferenceResponse* response, void* userp);
  //     evhtp_res FinalizeResponse(TRTSERVER_InferenceResponse* response);
  //
  // #ifdef TRTIS_ENABLE_TRACING
  //     std::unique_ptr<TraceMetaData> trace_meta_data_;
  // #endif  // TRTIS_ENABLE_TRACING
  //
  //     std::unique_ptr<EVBufferTuple> response_tuple_;
  //
  //    private:
  //     const uint64_t request_id_;
  //     const char* const server_id_;
  //     const uint64_t unique_id_;
  //   };

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

  int setup_ssl(
      const char* cert_file, const char* key_file, const char* ciphers);
  h2o_pathconf_t* register_handler(
      h2o_hostconf_t* hostconf, const char* path,
      int (*on_req)(h2o_handler_t*, h2o_req_t*));
  static int post_test(h2o_handler_t* self, h2o_req_t* req);
  static int health(h2o_handler_t* self, h2o_req_t* req);
  static int status(h2o_handler_t* self, h2o_req_t* req);
  // void HandleStatus(evhtp_request_t* req, const std::string& model_name);
  // void HandleInfer(evhtp_request_t* req, const std::string& model_name);

  // #ifdef TRTIS_ENABLE_GPU
  //   TRTSERVER_Error* EVBufferToCudaHandle(
  //       evbuffer* handle_buffer, cudaIpcMemHandle_t** cuda_shm_handle);
  // #endif  // TRTIS_ENABLE_GPU
  //   TRTSERVER_Error* EVBufferToInput(
  //       const std::string& model_name, const InferRequestHeader&
  //       request_header, const InferRequest& request,
  //       TRTSERVER_InferenceRequestProvider* request_provider,
  //       std::unordered_map<
  //           std::string,
  //           std::tuple<const void*, size_t, TRTSERVER_Memory_Type, int64_t>>&
  //           output_shm_map);
  //
  //   static void OKReplyCallback(evthr_t* thr, void* arg, void* shared);
  //   static void BADReplyCallback(evthr_t* thr, void* arg, void* shared);

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

static h2o_globalconf_t config;
static h2o_context_t ctx;
static h2o_multithread_receiver_t libmemcached_receiver;
static h2o_accept_ctx_t accept_ctx;

#if H2O_USE_LIBUV

static void
on_accept(uv_stream_t* listener, int status)
{
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
  h2o_accept(&accept_ctx, sock);
}

static int
create_listener(int32_t port)
{
  static uv_tcp_t listener;
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

  return 0;
Error:
  uv_close((uv_handle_t*)&listener, NULL);
  return r;
}

#else

static void
on_accept(h2o_socket_t* listener, const char* err)
{
  h2o_socket_t* sock;

  if (err != NULL) {
    return;
  }

  if ((sock = h2o_evloop_socket_accept(listener)) == NULL)
    return;
  h2o_accept(&accept_ctx, sock);
}

static int
create_listener(int32_t port)
{
  struct sockaddr_in addr;
  int fd, reuseaddr_flag = 1;
  h2o_socket_t* sock;

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

  sock = h2o_evloop_socket_create(ctx.loop, fd, H2O_SOCKET_FLAG_DONT_READ);
  h2o_socket_read_start(sock, on_accept);

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
  h2o_handler_t* handler = h2o_create_handler(pathconf, sizeof(*handler));
  handler->on_req = on_req;
  return pathconf;
}

int
HTTPAPIServer::post_test(h2o_handler_t* self, h2o_req_t* req)
{
  if (h2o_memis(req->method.base, req->method.len, H2O_STRLIT("POST")) &&
      h2o_memis(
          req->path_normalized.base, req->path_normalized.len,
          H2O_STRLIT("/post-test/"))) {
    LOG_VERBOSE(1) << "Hit here";
    static h2o_generator_t generator = {NULL, NULL};
    req->res.status = 200;
    req->res.reason = "OK";
    h2o_add_header(
        &req->pool, &req->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL,
        H2O_STRLIT("text/plain; charset=utf-8"));
    h2o_start_response(req, &generator);
    h2o_send(req, &req->entity, 1, h2o_send_state_t(1));
    return 0;
  }

  return -1;
}

int
HTTPAPIServer::health(h2o_handler_t* self, h2o_req_t* req)
{
  LOG_VERBOSE(1) << "url: " << std::string(req->path_normalized.base);
  if (!h2o_memis(req->method.base, req->method.len, H2O_STRLIT("GET")))
    return -1;

  if (!h2o_memis(
          req->path_normalized.base, req->path_normalized.len,
          H2O_STRLIT("/api/health")))
    return -1;

  std::string mode;
  size_t query_len = req->path.len - req->query_at;
  if (req->query_at != SIZE_MAX && (query_len > 1)) {
    if (h2o_memis(&req->path.base[req->query_at], 0, "", 6)) {
      mode = std::string(&req->path.base[req->query_at], query_len);
    }
  }
  LOG_VERBOSE(1) << "mode: " << mode;

  req->res.status = 400;
  req->res.reason = "Bad Request";
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

  if (health && (err == nullptr)) {
    req->res.status = 200;
    req->res.reason = "OK";
  }

  h2o_add_header(
      &req->pool, &req->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL,
      H2O_STRLIT("text/plain"));
  h2o_send_inline(req, H2O_STRLIT(request_status.ShortDebugString().c_str()));

  return 0;
}

int
HTTPAPIServer::status(h2o_handler_t* self, h2o_req_t* req)
{
  re2::RE2 status_regex(R"(/api/status(/(.*)?))");
  LOG_VERBOSE(1) << "url: " << std::string(req->path_normalized.base);
  if (!h2o_memis(req->method.base, req->method.len, H2O_STRLIT("GET"))) {
    return -1;
  }

  std::string status_uri =
      std::string(req->path_normalized.base, req->path_normalized.len);
  std::string model_name, format = "text";
  if (!status_uri.empty()) {
    if (!RE2::FullMatch(status_uri, status_regex, &model_name)) {
      return -1;
    }
  }
  LOG_VERBOSE(1) << "model_name: " << model_name;

  size_t query_len = req->path.len - req->query_at;
  if (req->query_at != SIZE_MAX && (query_len > 1)) {
    if (h2o_memis(&req->path.base[req->query_at], 8, "?format=", 6)) {
      format = std::string(&req->path.base[req->query_at], query_len);
    }
  }

  TRTSERVER_Protobuf* server_status_protobuf = nullptr;
  TRTSERVER_Error* err =
      (model_name.empty())
          ? TRTSERVER_ServerStatus(server_.get(), &server_status_protobuf)
          : TRTSERVER_ServerModelStatus(
                server_.get(), model_name.c_str(), &server_status_protobuf);

  h2o_generator_t generator;
  memset(&generator, 0, sizeof(generator));
  h2o_iovec_t body;

  if (err == nullptr) {
    const char* status_buffer;
    size_t status_byte_size;
    err = TRTSERVER_ProtobufSerialize(
        server_status_protobuf, &status_buffer, &status_byte_size);
    if (err == nullptr) {
      // Request text or binary format for status?
      if (format == "binary") {
        h2o_add_header(
            &req->pool, &req->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL,
            H2O_STRLIT("application/octet-stream"));
        body.base = const_cast<char*>(status_buffer);
        body.len = status_byte_size;
      } else {
        ServerStatus server_status;
        if (!server_status.ParseFromArray(status_buffer, status_byte_size)) {
          err = TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_UNKNOWN, "failed to parse server status");
        } else {
          std::string server_status_str = server_status.DebugString();
          h2o_add_header(
              &req->pool, &req->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL,
              H2O_STRLIT("text/plain"));
          body.base = const_cast<char*>(server_status_str.c_str());
          body.len = server_status_str.size();
        }
      }
    }
  }

  TRTSERVER_ProtobufDelete(server_status_protobuf);

  RequestStatus request_status;
  RequestStatusUtil::Create(
      &request_status, err, RequestStatusUtil::NextUniqueRequestId(),
      server_id_);

  std::string status_header = std::string(kStatusHTTPHeader);
  h2o_add_header_by_str(
      &req->pool, &req->res.headers, status_header.c_str(),
      status_header.size(), 0, NULL,
      H2O_STRLIT(request_status.ShortDebugString().c_str()));

  if (err == nullptr) {
    req->res.status = 200;
    req->res.reason = "OK";
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
  h2o_pathconf_t* pathconf;

  signal(SIGPIPE, SIG_IGN);

  h2o_config_init(&config);
  hostconf = h2o_config_register_host(
      &config, h2o_iovec_init(H2O_STRLIT("default")), 65535);

  pathconf = register_handler(hostconf, "/post-test", post_test);
  if (logfh != NULL)
    h2o_access_log_register(pathconf, logfh);

  pathconf = register_handler(hostconf, "/api/health", health);
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

  // Add server.crt and key files
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

#if H2O_USE_LIBUV
  uv_run(ctx.loop, UV_RUN_DEFAULT);
#else
  while (h2o_evloop_run(ctx.loop, INT32_MAX) == 0)
    ;
#endif

  return nullptr;

Error:
  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_INTERNAL, "HTTP server failed to run.");
}

TRTSERVER_Error*
HTTPAPIServer::Stop()
{
  // if (worker_.joinable()) {
  //   // Notify event loop to break via fd write
  //   send(fds_[1], &evbase_, sizeof(event_base*), 0);
  //   worker_.join();
  //   event_free(break_ev_);
  //   evutil_closesocket(fds_[0]);
  //   evutil_closesocket(fds_[1]);
  //   evhtp_unbind_socket(htp_);
  //   evhtp_free(htp_);
  //   event_base_free(evbase_);
  //   return nullptr;
  // }

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
  // auto userp_tuple = reinterpret_cast<EVBufferTuple*>(userp);
  // evbuffer* evhttp_buffer =
  //     reinterpret_cast<evbuffer*>(std::get<0>(*userp_tuple));
  // const std::unordered_map<
  //     std::string,
  //     std::tuple<const void*, size_t, TRTSERVER_Memory_Type, int64_t>>&
  //     output_shm_map = std::get<1>(*userp_tuple);
  //
  // *buffer = nullptr;
  // *buffer_userp = nullptr;
  // *actual_memory_type = preferred_memory_type;
  // *actual_memory_type_id = preferred_memory_type_id;
  //
  // // Don't need to do anything if no memory was requested.
  // if (byte_size > 0) {
  //   auto pr = output_shm_map.find(tensor_name);
  //   if (pr != output_shm_map.end()) {
  //     // If the output is in shared memory then check that the expected
  //     buffer
  //     // size is at least the byte size of the output.
  //     if (byte_size > std::get<1>(pr->second)) {
  //       return TRTSERVER_ErrorNew(
  //           TRTSERVER_ERROR_INTERNAL,
  //           std::string(
  //               "expected buffer size to be at least " +
  //               std::to_string(std::get<1>(pr->second)) + " bytes but gets "
  //               + std::to_string(byte_size) + " bytes in output tensor")
  //               .c_str());
  //     }
  //
  //     *buffer = const_cast<void*>(std::get<0>(pr->second));
  //     *actual_memory_type = std::get<2>(pr->second);
  //     *actual_memory_type_id = std::get<3>(pr->second);
  //   } else {
  //     // Can't allocate for any memory type other than CPU.
  //     if (preferred_memory_type != TRTSERVER_MEMORY_CPU) {
  //       LOG_VERBOSE(1)
  //           << "HTTP V2: unable to provide '" << tensor_name
  //           << "' in TRTSERVER_MEMORY_GPU, will use type
  //           TRTSERVER_MEMORY_CPU";
  //       *actual_memory_type = TRTSERVER_MEMORY_CPU;
  //       *actual_memory_type_id = 0;
  //     }
  //
  //     // Reserve requested space in evbuffer...
  //     struct evbuffer_iovec output_iovec;
  //     if (evbuffer_reserve_space(evhttp_buffer, byte_size, &output_iovec, 1)
  //     !=
  //         1) {
  //       return TRTSERVER_ErrorNew(
  //           TRTSERVER_ERROR_INTERNAL,
  //           std::string(
  //               "failed to reserve " + std::to_string(byte_size) +
  //               " bytes in output tensor buffer")
  //               .c_str());
  //     }
  //
  //     if (output_iovec.iov_len < byte_size) {
  //       return TRTSERVER_ErrorNew(
  //           TRTSERVER_ERROR_INTERNAL,
  //           std::string(
  //               "reserved " + std::to_string(output_iovec.iov_len) +
  //               " bytes in output tensor buffer, need " +
  //               std::to_string(byte_size))
  //               .c_str());
  //     }
  //
  //     output_iovec.iov_len = byte_size;
  //     *buffer = output_iovec.iov_base;
  //
  //     // Immediately commit the buffer space. We are relying on evbuffer
  //     // not to relocate this space. Because we request a contiguous
  //     // chunk every time (above by allowing only a single entry in
  //     // output_iovec), this seems to be a valid assumption.
  //     if (evbuffer_commit_space(evhttp_buffer, &output_iovec, 1) != 0) {
  //       *buffer = nullptr;
  //       return TRTSERVER_ErrorNew(
  //           TRTSERVER_ERROR_INTERNAL,
  //           "failed to commit output tensors to output buffer");
  //     }
  //   }
  // }
  //
  // LOG_VERBOSE(1) << "HTTP V2 allocation: '" << tensor_name
  //                << "', size: " << byte_size << ", addr: " << *buffer;

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

// TRTSERVER_Error*
// HTTPAPIServer::EVBufferToInput(
//     const std::string& model_name, const InferRequestHeader& request_header,
//     const InferRequest& request,
//     TRTSERVER_InferenceRequestProvider* request_provider,
//     std::unordered_map<
//         std::string,
//         std::tuple<const void*, size_t, TRTSERVER_Memory_Type, int64_t>>&
//         output_shm_map)
// {
//   // Extract input data from HTTP body and register in
//   // 'request_provider'.
//   // Get the byte-size for each input and from that get the blocks
//   // holding the data for that input
//   size_t idx = 0;
//   for (const auto& io : request_header.input()) {
//     uint64_t byte_size = 0;
//     RETURN_IF_ERR(TRTSERVER_InferenceRequestProviderInputBatchByteSize(
//         request_provider, io.name().c_str(), &byte_size));
//
//     // If 'byte_size' is zero then need to add an empty input data
//     // block... the provider expects at least one data block for every
//     // input.
//     if (byte_size == 0) {
//       RETURN_IF_ERR(TRTSERVER_InferenceRequestProviderSetInputData(
//           request_provider, io.name().c_str(), nullptr, 0 /* byte_size */,
//           TRTSERVER_MEMORY_CPU, 0 /* memory_type_id */));
//     } else {
//       // If input is in shared memory then verify that the size is
//       // correct and set input from the shared memory.
//       if (io.has_shared_memory()) {
//         if (byte_size != io.shared_memory().byte_size()) {
//           return TRTSERVER_ErrorNew(
//               TRTSERVER_ERROR_INVALID_ARG,
//               std::string(
//                   "unexpected shared-memory size " +
//                   std::to_string(io.shared_memory().byte_size()) +
//                   " for input '" + io.name() + "', expecting " +
//                   std::to_string(byte_size) + " for model '" + model_name +
//                   "'") .c_str());
//         }
//
//         void* base;
//         TRTSERVER_Memory_Type memory_type = TRTSERVER_MEMORY_CPU;
//         int64_t memory_type_id;
//         TRTSERVER_SharedMemoryBlock* smb = nullptr;
//         RETURN_IF_ERR(smb_manager_->Get(&smb, io.shared_memory().name()));
//         RETURN_IF_ERR(TRTSERVER_ServerSharedMemoryAddress(
//             server_.get(), smb, io.shared_memory().offset(),
//             io.shared_memory().byte_size(), &base));
//         TRTSERVER_SharedMemoryBlockMemoryType(smb, &memory_type);
//         TRTSERVER_SharedMemoryBlockMemoryTypeId(smb, &memory_type_id);
//         RETURN_IF_ERR(TRTSERVER_InferenceRequestProviderSetInputData(
//             request_provider, io.name().c_str(), base, byte_size,
//             memory_type, memory_type_id));
//       } else {
//         const std::string& raw = request.raw_input(idx++);
//         const void* base = raw.c_str();
//         size_t request_byte_size = raw.size();
//
//         if (byte_size != request_byte_size) {
//           return TRTSERVER_ErrorNew(
//               TRTSERVER_ERROR_INVALID_ARG,
//               std::string(
//                   "unexpected size " + std::to_string(request_byte_size) +
//                   " for input '" + io.name() + "', expecting " +
//                   std::to_string(byte_size) + " for model '" + model_name +
//                   "'") .c_str());
//         }
//
//         RETURN_IF_ERR(TRTSERVER_InferenceRequestProviderSetInputData(
//             request_provider, io.name().c_str(), base, byte_size,
//             TRTSERVER_MEMORY_CPU, 0 /* memory_type_id */));
//       }
//     }
//   }
//
//   // Initialize System Memory for Output if it uses shared memory
//   for (const auto& io : request_header.output()) {
//     if (io.has_shared_memory()) {
//       void* base;
//       TRTSERVER_SharedMemoryBlock* smb = nullptr;
//       RETURN_IF_ERR(smb_manager_->Get(&smb, io.shared_memory().name()));
//       RETURN_IF_ERR(TRTSERVER_ServerSharedMemoryAddress(
//           server_.get(), smb, io.shared_memory().offset(),
//           io.shared_memory().byte_size(), &base));
//
//       TRTSERVER_Memory_Type memory_type;
//       int64_t memory_type_id;
//       TRTSERVER_SharedMemoryBlockMemoryType(smb, &memory_type);
//       TRTSERVER_SharedMemoryBlockMemoryTypeId(smb, &memory_type_id);
//       output_shm_map.emplace(
//           io.name(),
//           std::make_tuple(
//               static_cast<const void*>(base), io.shared_memory().byte_size(),
//               memory_type, memory_type_id));
//     }
//   }
//
//   return nullptr;  // success
// }

// void
// HTTPAPIServer::HandleInfer(evhtp_request_t* req, const std::string&
// model_name)
// {
//   if (req->method != htp_method_POST) {
//     evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
//     return;
//   }
//
//   // Assume -1 for now
//   std::string model_version_str = "-1";
//   if (model_name.empty()) {
//     evhtp_send_reply(req, EVHTP_RES_BADREQ);
//     return;
//   }
//
//   int64_t model_version = -1;
//   if (!model_version_str.empty()) {
//     model_version = std::atoll(model_version_str.c_str());
//   }
//
// #ifdef TRTIS_ENABLE_TRACING
//   // Timestamps from evhtp are capture in 'req'. We record here since
//   // this is the first place where we have a tracer.
//   std::unique_ptr<TraceMetaData> trace_meta_data;
//   if (trace_manager_ != nullptr) {
//     trace_meta_data.reset(trace_manager_->SampleTrace());
//     if (trace_meta_data != nullptr) {
//       trace_meta_data->tracer_->SetModel(model_name, model_version);
//       trace_meta_data->tracer_->CaptureTimestamp(
//           TRTSERVER_TRACE_LEVEL_MIN, "http recv start",
//           TIMESPEC_TO_NANOS(req->recv_start_ts));
//       trace_meta_data->tracer_->CaptureTimestamp(
//           TRTSERVER_TRACE_LEVEL_MIN, "http recv end",
//           TIMESPEC_TO_NANOS(req->recv_end_ts));
//     }
//   }
// #endif  // TRTIS_ENABLE_TRACING
//
//   std::string infer_request_header(
//       evhtp_kv_find(req->headers_in, kInferRequestHTTPHeader));
//
//   InferRequestHeader request_header;
//   if (!google::protobuf::TextFormat::ParseFromString(
//           infer_request_header, &request_header)) {
//     evhtp_send_reply(req, EVHTP_RES_BADREQ);
//     return;
//   }
//
//   std::string request_header_serialized;
//   if (!request_header.SerializeToString(&request_header_serialized)) {
//     evhtp_send_reply(req, EVHTP_RES_BADREQ);
//     return;
//   }
//
//   // Convert the json string to protobuf message
//   EVBufferTuple* response_tuple(new EVBufferTuple());
//   size_t buffer_length = evbuffer_get_length(req->buffer_in);
//   char* request_buffer = (char*)malloc(sizeof(char) * buffer_length);
//   evbuffer_copyout(req->buffer_in, request_buffer, buffer_length);
//   std::string json_request_string = std::string(request_buffer,
//   buffer_length); if (google::protobuf::util::JsonStringToMessage(
//           json_request_string, &std::get<2>(*response_tuple)) !=
//       google::protobuf::util::Status::OK) {
//     evhtp_send_reply(req, EVHTP_RES_BADREQ);
//     return;
//   }
//   free(request_buffer);
//
//   uint64_t unique_id = RequestStatusUtil::NextUniqueRequestId();
//
//   // Create the inference request provider which provides all the
//   // input information needed for an inference.
//   TRTSERVER_InferenceRequestProvider* request_provider = nullptr;
//   TRTSERVER_Error* err = TRTSERVER_InferenceRequestProviderNew(
//       &request_provider, server_.get(), model_name.c_str(), model_version,
//       request_header_serialized.c_str(), request_header_serialized.size());
//   if (err == nullptr) {
//     err = EVBufferToInput(
//         model_name, request_header, std::get<2>(*response_tuple),
//         request_provider, std::get<1>(*response_tuple));
//     if (err == nullptr) {
//       InferRequestClass* infer_request = new InferRequestClass(
//           req, request_header.id(), server_id_, unique_id);
//
//       std::get<0>(*response_tuple) = req->buffer_out;
//       infer_request->response_tuple_.reset(response_tuple);
//
//       // Provide the trace manager object to use for this request, if nullptr
//       // then no tracing will be performed.
//       TRTSERVER_TraceManager* trace_manager = nullptr;
// #ifdef TRTIS_ENABLE_TRACING
//       if (trace_meta_data != nullptr) {
//         infer_request->trace_meta_data_ = std::move(trace_meta_data);
//         TRTSERVER_TraceManagerNew(
//             &trace_manager, TraceManager::CreateTrace,
//             TraceManager::ReleaseTrace,
//             infer_request->trace_meta_data_.get());
//       }
// #endif  // TRTIS_ENABLE_TRACING
//
//       err = TRTSERVER_ServerInferAsync(
//           server_.get(), trace_manager, request_provider, allocator_,
//           reinterpret_cast<void*>(response_tuple),
//           InferRequestClass::InferComplete,
//           reinterpret_cast<void*>(infer_request));
//       if (err != nullptr) {
//         delete infer_request;
//         infer_request = nullptr;
//       }
//     }
//   }
//
//   // The request provider can be deleted before ServerInferAsync
//   // callback completes.
//   TRTSERVER_InferenceRequestProviderDelete(request_provider);
//
//   if (err != nullptr) {
//     RequestStatus request_status;
//     RequestStatusUtil::Create(&request_status, err, unique_id, server_id_);
//
//     InferResponseHeader response_header;
//     response_header.set_id(request_header.id());
//     evhtp_headers_add_header(
//         req->headers_out,
//         evhtp_header_new(
//             kInferResponseHTTPHeader,
//             response_header.ShortDebugString().c_str(), 1, 1));
//     LOG_VERBOSE(1) << "Infer failed: " << request_status.msg();
//
//     evhtp_headers_add_header(
//         req->headers_out, evhtp_header_new(
//                               kStatusHTTPHeader,
//                               request_status.ShortDebugString().c_str(), 1,
//                               1));
//     evhtp_headers_add_header(
//         req->headers_out,
//         evhtp_header_new("Content-Type", "application/octet-stream", 1, 1));
//
//     evhtp_send_reply(
//         req, (request_status.code() == RequestStatusCode::SUCCESS)
//                  ? EVHTP_RES_OK
//                  : EVHTP_RES_BADREQ);
//   }
//
//   TRTSERVER_ErrorDelete(err);
// }

// void
// HTTPAPIServer::OKReplyCallback(evthr_t* thr, void* arg, void* shared)
// {
//   HTTPAPIServer::InferRequestClass* infer_request =
//       reinterpret_cast<HTTPAPIServer::InferRequestClass*>(arg);
//
//   evhtp_request_t* request = infer_request->EvHtpRequest();
//   evhtp_send_reply(request, EVHTP_RES_OK);
//   evhtp_request_resume(request);
//
// #ifdef TRTIS_ENABLE_TRACING
//   if (infer_request->trace_meta_data_ != nullptr) {
//     infer_request->trace_meta_data_->tracer_->CaptureTimestamp(
//         TRTSERVER_TRACE_LEVEL_MIN, "http send start",
//         TIMESPEC_TO_NANOS(request->send_start_ts));
//     infer_request->trace_meta_data_->tracer_->CaptureTimestamp(
//         TRTSERVER_TRACE_LEVEL_MIN, "http send end",
//         TIMESPEC_TO_NANOS(request->send_end_ts));
//   }
// #endif  // TRTIS_ENABLE_TRACING
//
//   delete infer_request;
// }

// void
// HTTPAPIServer::BADReplyCallback(evthr_t* thr, void* arg, void* shared)
// {
//   HTTPAPIServer::InferRequestClass* infer_request =
//       reinterpret_cast<HTTPAPIServer::InferRequestClass*>(arg);
//
//   evhtp_request_t* request = infer_request->EvHtpRequest();
//   evhtp_send_reply(request, EVHTP_RES_BADREQ);
//   evhtp_request_resume(request);
//
// #ifdef TRTIS_ENABLE_TRACING
//   if (infer_request->trace_meta_data_ != nullptr) {
//     infer_request->trace_meta_data_->tracer_->CaptureTimestamp(
//         TRTSERVER_TRACE_LEVEL_MIN, "http send start",
//         TIMESPEC_TO_NANOS(request->send_start_ts));
//     infer_request->trace_meta_data_->tracer_->CaptureTimestamp(
//         TRTSERVER_TRACE_LEVEL_MIN, "http send end",
//         TIMESPEC_TO_NANOS(request->send_end_ts));
//   }
// #endif  // TRTIS_ENABLE_TRACING
//
//   delete infer_request;
// }

// HTTPAPIServer::InferRequestClass::InferRequestClass(
//     evhtp_request_t* req, uint64_t request_id, const char* server_id,
//     uint64_t unique_id)
//     : req_(req), request_id_(request_id), server_id_(server_id),
//       unique_id_(unique_id)
// {
//   evhtp_connection_t* htpconn = evhtp_request_get_connection(req);
//   thread_ = htpconn->thread;
//   evhtp_request_pause(req);
// }
//
// void
// HTTPAPIServer::InferRequestClass::InferComplete(
//     TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
//     TRTSERVER_InferenceResponse* response, void* userp)
// {
//   HTTPAPIServer::InferRequestClass* infer_request =
//       reinterpret_cast<HTTPAPIServer::InferRequestClass*>(userp);
//   if (infer_request->FinalizeResponse(response) == EVHTP_RES_OK) {
//     evthr_defer(infer_request->thread_, OKReplyCallback, infer_request);
//   } else {
//     evthr_defer(infer_request->thread_, BADReplyCallback, infer_request);
//   }
//
//   // Don't need to explicitly delete 'trace_manager'. It will be deleted by
//   // the TraceMetaData object in 'infer_request'.
//   LOG_IF_ERR(
//       TRTSERVER_InferenceResponseDelete(response), "deleting HTTP response");
// }
//
// evhtp_res
// HTTPAPIServer::InferRequestClass::FinalizeResponse(
//     TRTSERVER_InferenceResponse* response)
// {
//   InferResponseHeader response_header;
//
//   TRTSERVER_Error* response_status =
//       TRTSERVER_InferenceResponseStatus(response);
//   if (response_status == nullptr) {
//     TRTSERVER_Protobuf* response_protobuf = nullptr;
//     response_status =
//         TRTSERVER_InferenceResponseHeader(response, &response_protobuf);
//     if (response_status == nullptr) {
//       const char* buffer;
//       size_t byte_size;
//       response_status =
//           TRTSERVER_ProtobufSerialize(response_protobuf, &buffer,
//           &byte_size);
//       if (response_status == nullptr) {
//         if (!response_header.ParseFromArray(buffer, byte_size)) {
//           response_status = TRTSERVER_ErrorNew(
//               TRTSERVER_ERROR_INTERNAL, "failed to parse response header");
//         }
//       }
//
//       TRTSERVER_ProtobufDelete(response_protobuf);
//     }
//   }
//
//   if (response_status == nullptr) {
//     std::string format;
//     const char* format_c_str = evhtp_kv_find(req_->uri->query, "format");
//     if (format_c_str != NULL) {
//       format = std::string(format_c_str);
//     } else {
//       format = "text";
//     }
//
//     // The description of the raw outputs needs to go in the
//     // kInferResponseHTTPHeader since it is needed to interpret the
//     // body. The entire response (including classifications) is
//     // serialized at the end of the body.
//     response_header.set_id(request_id_);
//
//     std::string rstr;
//     if (format == "binary") {
//       response_header.SerializeToString(&rstr);
//     } else {
//       rstr = response_header.DebugString();
//     }
//
//     evbuffer_add(req_->buffer_out, rstr.c_str(), rstr.size());
//   } else {
//     evbuffer_drain(req_->buffer_out, -1);
//     response_header.Clear();
//     response_header.set_id(request_id_);
//   }
//
//   RequestStatus request_status;
//   RequestStatusUtil::Create(
//       &request_status, response_status, unique_id_, server_id_);
//
//   evhtp_headers_add_header(
//       req_->headers_out, evhtp_header_new(
//                              kInferResponseHTTPHeader,
//                              response_header.ShortDebugString().c_str(), 1,
//                              1));
//   evhtp_headers_add_header(
//       req_->headers_out,
//       evhtp_header_new(
//           kStatusHTTPHeader, request_status.ShortDebugString().c_str(), 1,
//           1));
//   evhtp_headers_add_header(
//       req_->headers_out,
//       evhtp_header_new("Content-Type", "application/octet-stream", 1, 1));
//
//   TRTSERVER_ErrorDelete(response_status);
//
//   return (request_status.code() == RequestStatusCode::SUCCESS)
//              ? EVHTP_RES_OK
//              : EVHTP_RES_BADREQ;
// }

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
