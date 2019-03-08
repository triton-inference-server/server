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

#include "src/core/http_server.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "evhtp/evhtp.h"
#include "libevent/include/event2/buffer.h"
#include "re2/re2.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/request_status.h"
#include "src/core/server.h"

namespace nvidia { namespace inferenceserver {

//
// Handle HTTP requests
//
class HTTPServerImpl : public HTTPServer {
 public:
  explicit HTTPServerImpl(
      InferenceServer* server, uint16_t port, int thread_cnt)
      : server_(server), port_(port), thread_cnt_(thread_cnt),
        api_regex_(R"(/api/(health|profile|infer|status)(.*))"),
        health_regex_(R"(/(live|ready))"),
        infer_regex_(R"(/([^/]+)(?:/(\d+))?)"), status_regex_(R"(/(.*))")
  {
  }

  static void Dispatch(evhtp_request_t* req, void* arg);

  tensorflow::Status Start() override;

  tensorflow::Status Stop() override;

 private:
  // Class object associated to evhtp thread, requests received are bounded
  // with the thread that accepts it. Need to keep track of that and let the
  // corresponding thread send back the reply
  class InferRequest {
   public:
    InferRequest(
        evhtp_request_t* req, uint64_t id,
        const std::shared_ptr<HTTPInferRequestProvider>& request_provider,
        const std::shared_ptr<HTTPInferResponseProvider>& response_provider,
        const std::shared_ptr<ModelInferStats>& infer_stats,
        const std::shared_ptr<ModelInferStats::ScopedTimer>& timer);

    evhtp_res FinalizeResponse();

   private:
    friend class HTTPServerImpl;
    evhtp_request_t* req_;
    evthr_t* thread_;
    uint64_t id_;
    RequestStatus request_status_;
    std::shared_ptr<HTTPInferRequestProvider> request_provider_;
    std::shared_ptr<HTTPInferResponseProvider> response_provider_;
    std::shared_ptr<ModelInferStats> infer_stats_;
    std::shared_ptr<ModelInferStats::ScopedTimer> timer_;
  };

  void Handle(evhtp_request_t* req);
  void Health(evhtp_request_t* req, const std::string& health_uri);
  void Profile(evhtp_request_t* req, const std::string& profile_uri);
  void Infer(evhtp_request_t* req, const std::string& infer_uri);
  void Status(evhtp_request_t* req, const std::string& status_uri);

  void FinishInferResponse(const std::shared_ptr<InferRequest>& req);
  static void OKReplyCallback(evthr_t* thr, void* arg, void* shared);
  static void BADReplyCallback(evthr_t* thr, void* arg, void* shared);

  InferenceServer* server_;
  uint16_t port_;
  int thread_cnt_;
  re2::RE2 api_regex_;
  re2::RE2 health_regex_;
  re2::RE2 infer_regex_;
  re2::RE2 status_regex_;

  evhtp_t* htp_;
  struct event_base* evbase_;
  std::thread worker_;
};

tensorflow::Status
HTTPServerImpl::Start()
{
  if (!worker_.joinable()) {
    evbase_ = event_base_new();
    htp_ = evhtp_new(evbase_, NULL);
    evhtp_set_gencb(htp_, HTTPServerImpl::Dispatch, this);
    evhtp_use_threads_wexit(htp_, NULL, NULL, thread_cnt_, NULL);
    evhtp_bind_socket(htp_, "0.0.0.0", port_, 1024);
    worker_ = std::thread(event_base_loop, evbase_, 0);
    return tensorflow::Status::OK();
  }

  return tensorflow::Status(
      tensorflow::error::ALREADY_EXISTS, "HTTP server is already running.");
}

tensorflow::Status
HTTPServerImpl::Stop()
{
  if (worker_.joinable()) {
    event_base_loopexit(evbase_, NULL);
    worker_.join();
    return tensorflow::Status::OK();
  }

  return tensorflow::Status(
      tensorflow::error::UNAVAILABLE, "HTTP server is not running.");
}

void
HTTPServerImpl::Dispatch(evhtp_request_t* req, void* arg)
{
  (static_cast<HTTPServerImpl*>(arg))->Handle(req);
}

void
HTTPServerImpl::Handle(evhtp_request_t* req)
{
  LOG_VERBOSE(1) << "HTTP request: " << req->method << " "
                 << req->uri->path->full;

  std::string endpoint, rest;
  if (RE2::FullMatch(
          std::string(req->uri->path->full), api_regex_, &endpoint, &rest)) {
    // health
    if (endpoint == "health") {
      Health(req, rest);
      return;
    }
    // profile
    else if (endpoint == "profile") {
      Profile(req, rest);
      return;
    }
    // infer
    else if (endpoint == "infer") {
      Infer(req, rest);
      return;
    }
    // status
    else if (endpoint == "status") {
      Status(req, rest);
      return;
    }
  }

  LOG_VERBOSE(1) << "HTTP error: " << req->method << " " << req->uri->path->full
                 << " - " << static_cast<int>(EVHTP_RES_BADREQ);
  evhtp_send_reply(req, EVHTP_RES_BADREQ);
}

void
HTTPServerImpl::Health(evhtp_request_t* req, const std::string& health_uri)
{
  ServerStatTimerScoped timer(
      server_->StatusManager(), ServerStatTimerScoped::Kind::HEALTH);

  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  std::string mode;
  if (!health_uri.empty()) {
    if (!RE2::FullMatch(health_uri, health_regex_, &mode)) {
      evhtp_send_reply(req, EVHTP_RES_BADREQ);
      return;
    }
  }

  RequestStatus request_status;
  bool health;
  server_->HandleHealth(&request_status, &health, mode);

  evhtp_headers_add_header(
      req->headers_out,
      evhtp_header_new(
          kStatusHTTPHeader, request_status.ShortDebugString().c_str(), 1, 1));

  evhtp_send_reply(req, (health) ? EVHTP_RES_OK : EVHTP_RES_BADREQ);
}

void
HTTPServerImpl::Profile(evhtp_request_t* req, const std::string& profile_uri)
{
  ServerStatTimerScoped timer(
      server_->StatusManager(), ServerStatTimerScoped::Kind::PROFILE);

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

  RequestStatus request_status;
  server_->HandleProfile(&request_status, cmd);

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
HTTPServerImpl::Infer(evhtp_request_t* req, const std::string& infer_uri)
{
  if (req->method != htp_method_POST) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  std::string model_name, model_version_str;
  if (!infer_uri.empty()) {
    if (!RE2::FullMatch(
            infer_uri, infer_regex_, &model_name, &model_version_str)) {
      evhtp_send_reply(req, EVHTP_RES_BADREQ);
      return;
    }
  }

  int64_t model_version = -1;
  if (!model_version_str.empty()) {
    model_version = std::atoll(model_version_str.c_str());
  }

  auto infer_stats =
      std::make_shared<ModelInferStats>(server_->StatusManager(), model_name);
  auto timer = std::make_shared<ModelInferStats::ScopedTimer>();
  infer_stats->StartRequestTimer(timer.get());
  infer_stats->SetRequestedVersion(model_version);

  absl::string_view infer_request_header = absl::string_view(
      evhtp_kv_find(req->headers_in, kInferRequestHTTPHeader));
  std::string infer_request_header_str(
      infer_request_header.data(), infer_request_header.size());

  RequestStatus request_status;

  InferRequestHeader request_header;
  tensorflow::protobuf::TextFormat::ParseFromString(
      infer_request_header_str, &request_header);
  uint64_t id = request_header.id();

  auto backend = std::make_shared<InferenceServer::InferBackendHandle>();
  tensorflow::Status status =
      server_->CreateBackendHandle(model_name, model_version, backend);
  if (status.ok()) {
    infer_stats->SetModelBackend((*backend)());

    std::shared_ptr<HTTPInferRequestProvider> request_provider;
    status = HTTPInferRequestProvider::Create(
        req->buffer_in, *((*backend)()), model_name, model_version,
        infer_request_header_str, &request_provider);
    if (status.ok()) {
      infer_stats->SetBatchSize(request_provider->RequestHeader().batch_size());

      std::shared_ptr<HTTPInferResponseProvider> response_provider;
      status = HTTPInferResponseProvider::Create(
          req->buffer_out, *((*backend)()), request_provider->RequestHeader(),
          &response_provider);
      if (status.ok()) {
        std::shared_ptr<InferRequest> request(new InferRequest(
            req, id, request_provider, response_provider, infer_stats, timer));
        server_->HandleInfer(
            &(request->request_status_), backend, request->request_provider_,
            request->response_provider_, infer_stats,
            [this, request]() mutable { this->FinishInferResponse(request); });
      }
    }
  }

  if (!status.ok()) {
    InferResponseHeader response_header;
    response_header.set_id(id);
    evhtp_headers_add_header(
        req->headers_out,
        evhtp_header_new(
            kInferResponseHTTPHeader,
            response_header.ShortDebugString().c_str(), 1, 1));
    LOG_VERBOSE(1) << "Infer failed: " << status.error_message();
    infer_stats->SetFailed(true);
    RequestStatusFactory::Create(
        &request_status, 0 /* request_id */, server_->Id(), status);

    // this part still needs to be implemented in the completer
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
}

void
HTTPServerImpl::Status(evhtp_request_t* req, const std::string& status_uri)
{
  ServerStatTimerScoped timer(
      server_->StatusManager(), ServerStatTimerScoped::Kind::STATUS);

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

  RequestStatus request_status;
  ServerStatus server_status;
  server_->HandleStatus(&request_status, &server_status, model_name);

  // If got status successfully then send it...
  if (request_status.code() == RequestStatusCode::SUCCESS) {
    std::string format;
    const char* format_c_str = evhtp_kv_find(req->uri->query, "format");
    if (format_c_str != NULL) {
      format = std::string(format_c_str);
    } else {
      format = "text";
    }

    std::string server_status_str;
    if (format == "binary") {
      server_status.SerializeToString(&server_status_str);
      evbuffer_add(
          req->buffer_out, server_status_str.c_str(), server_status_str.size());
      evhtp_headers_add_header(
          req->headers_out,
          evhtp_header_new("Content-Type", "application/octet-stream", 1, 1));
    } else {
      server_status_str = server_status.DebugString();
      evbuffer_add(
          req->buffer_out, server_status_str.c_str(), server_status_str.size());
    }
  }

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
HTTPServerImpl::OKReplyCallback(evthr_t* thr, void* arg, void* shared)
{
  evhtp_request_t* request = (evhtp_request_t*)arg;
  evhtp_send_reply(request, EVHTP_RES_OK);
  evhtp_request_resume(request);
}

void
HTTPServerImpl::BADReplyCallback(evthr_t* thr, void* arg, void* shared)
{
  evhtp_request_t* request = (evhtp_request_t*)arg;
  evhtp_send_reply(request, EVHTP_RES_BADREQ);
  evhtp_request_resume(request);
}

void
HTTPServerImpl::FinishInferResponse(const std::shared_ptr<InferRequest>& req)
{
  if (req->FinalizeResponse() == EVHTP_RES_OK) {
    evthr_defer(req->thread_, OKReplyCallback, req->req_);
  } else {
    evthr_defer(req->thread_, BADReplyCallback, req->req_);
  }
}

HTTPServerImpl::InferRequest::InferRequest(
    evhtp_request_t* req, uint64_t id,
    const std::shared_ptr<HTTPInferRequestProvider>& request_provider,
    const std::shared_ptr<HTTPInferResponseProvider>& response_provider,
    const std::shared_ptr<ModelInferStats>& infer_stats,
    const std::shared_ptr<ModelInferStats::ScopedTimer>& timer)
    : req_(req), id_(id), request_provider_(request_provider),
      response_provider_(response_provider), infer_stats_(infer_stats),
      timer_(timer)
{
  evhtp_connection_t* htpconn = evhtp_request_get_connection(req);
  thread_ = htpconn->thread;
  evhtp_request_pause(req);
}

evhtp_res
HTTPServerImpl::InferRequest::FinalizeResponse()
{
  InferResponseHeader* response_header =
      response_provider_->MutableResponseHeader();
  if (request_status_.code() == RequestStatusCode::SUCCESS) {
    std::string format;
    const char* format_c_str = evhtp_kv_find(req_->uri->query, "format");
    if (format_c_str != NULL) {
      format = std::string(format_c_str);
    } else {
      format = "text";
    }

    // The description of the raw outputs needs to go in
    // the kInferResponseHTTPHeader since it is needed to
    // interpret the body. The entire response (including
    // classifications) is serialized at the end of the
    // body.
    response_header->set_id(id_);

    std::string rstr;
    if (format == "binary") {
      response_header->SerializeToString(&rstr);
    } else {
      rstr = response_header->DebugString();
    }
    evbuffer_add(req_->buffer_out, rstr.c_str(), rstr.size());

    // We do this in destructive manner since we are the
    // last one to use response header from the provider.
    for (int i = 0; i < response_header->output_size(); ++i) {
      InferResponseHeader::Output* output = response_header->mutable_output(i);
      output->clear_batch_classes();
    }
  } else {
    evbuffer_drain(req_->buffer_out, -1);
    response_header->Clear();
    response_header->set_id(id_);
  }
  evhtp_headers_add_header(
      req_->headers_out,
      evhtp_header_new(
          kInferResponseHTTPHeader, response_header->ShortDebugString().c_str(),
          1, 1));
  evhtp_headers_add_header(
      req_->headers_out,
      evhtp_header_new(
          kStatusHTTPHeader, request_status_.ShortDebugString().c_str(), 1, 1));
  evhtp_headers_add_header(
      req_->headers_out,
      evhtp_header_new("Content-Type", "application/octet-stream", 1, 1));

  return (request_status_.code() == RequestStatusCode::SUCCESS)
             ? EVHTP_RES_OK
             : EVHTP_RES_BADREQ;
}

tensorflow::Status
HTTPServer::Create(
    InferenceServer* server, uint16_t port, int thread_cnt,
    std::unique_ptr<HTTPServer>* http_server)
{
  http_server->reset(new HTTPServerImpl(server, port, thread_cnt));
  return tensorflow::Status::OK();
}
}}  // namespace nvidia::inferenceserver
