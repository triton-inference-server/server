// Copyright 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "http_server.h"

#include "common.h"
#include "http_error_json.h"
#include "http_server_macros.h"

#include <atomic>
#include <charconv>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace triton { namespace server {

namespace {

int HttpCodeFromError(TRITONSERVER_Error* error) {
  if (error == nullptr) {
    return EVHTP_RES_OK;
  }
  switch (TRITONSERVER_ErrorCode(error)) {
    case TRITONSERVER_ERROR_INTERNAL:
      return EVHTP_RES_SERVERR;
    case TRITONSERVER_ERROR_NOT_FOUND:
      return EVHTP_RES_NOTFOUND;
    case TRITONSERVER_ERROR_UNAVAILABLE:
      return EVHTP_RES_SERVUNAVAIL;
    case TRITONSERVER_ERROR_UNSUPPORTED:
      return EVHTP_RES_NOTIMPL;
    case TRITONSERVER_ERROR_UNKNOWN:
    case TRITONSERVER_ERROR_INVALID_ARG:
    case TRITONSERVER_ERROR_ALREADY_EXISTS:
    case TRITONSERVER_ERROR_CANCELLED:
      return EVHTP_RES_BADREQ;
  }

  return EVHTP_RES_BADREQ;
}

void AddContentTypeHeader(evhtp_request_t* req, const char* type) {
  auto content_header = evhtp_headers_find_header(req->headers_out, kContentTypeHeader);
  if (content_header) {
    evhtp_header_rm_and_free(req->headers_out, content_header);
  }

  evhtp_headers_add_header(req->headers_out, evhtp_header_new(kContentTypeHeader, type, 1, 1));
}

void AppendJsonEscaped(std::string* out, const std::string& value) {
  out->reserve(out->size() + value.size() + 8);
  for (char c : value) {
    switch (c) {
      case '"':
        out->append("\\\"");
        break;
      case '\\':
        out->append("\\\\");
        break;
      case '\b':
        out->append("\\b");
        break;
      case '\f':
        out->append("\\f");
        break;
      case '\n':
        out->append("\\n");
        break;
      case '\r':
        out->append("\\r");
        break;
      case '\t':
        out->append("\\t");
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[7];
          std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
          out->append(buf);
        } else {
          out->push_back(c);
        }
        break;
    }
  }
}

void AppendScoreToEvbuffer(evbuffer* out, double v) {
  v = RoundToScorePrecision(v);
  char buf[32];
  const auto result = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::fixed, kScoreJsonDecimalPrecision);
  if (result.ec == std::errc()) {
    evbuffer_add(out, buf, static_cast<size_t>(result.ptr - buf));
    return;
  }
  char fallback[32];
  const int len = std::snprintf(fallback, sizeof(fallback), "%.*f", kScoreJsonDecimalPrecision, v);
  if (len > 0) {
    evbuffer_add(out, fallback, static_cast<size_t>(len));
  }
}

bool IsJsonStringSafe(const char* s) {
  if (s == nullptr) {
    return true;
  }
  for (; *s != '\0'; ++s) {
    const unsigned char c = static_cast<unsigned char>(*s);
    if (c < 0x20 || c == '"' || c == '\\') {
      return false;
    }
  }
  return true;
}

void AppendJsonStringToEvbuffer(evbuffer* out, const char* s) {
  if (IsJsonStringSafe(s)) {
    evbuffer_add(out, s, std::strlen(s));
    return;
  }
  std::string escaped;
  AppendJsonEscaped(&escaped, s);
  evbuffer_add(out, escaped.data(), escaped.size());
}

void AppendShardErrorJsonToEvbuffer(evbuffer* out, const std::string& message) {
  evbuffer_add(out, "{\"error\":{\"message\":\"", 21);
  if (IsJsonStringSafe(message.c_str())) {
    evbuffer_add(out, message.data(), message.size());
  } else {
    std::string escaped;
    AppendJsonEscaped(&escaped, message);
    evbuffer_add(out, escaped.data(), escaped.size());
  }
  evbuffer_add(out, "\"}}", 3);
}

struct EvbufferDeleter {
  void operator()(evbuffer* buf) const {
    if (buf != nullptr) {
      evbuffer_free(buf);
    }
  }
};

using EvbufferPtr = std::unique_ptr<evbuffer, EvbufferDeleter>;
using SharedEvbuffer = std::shared_ptr<evbuffer>;

SharedEvbuffer MakeSharedEvbuffer(evbuffer* buf) {
  return SharedEvbuffer(buf, EvbufferDeleter{});
}

TRITONSERVER_Error* GetModelVersionStringFromSlot(triton::common::TritonJson::Value& slot, std::string* ver_out) {
  ver_out->clear();
  triton::common::TritonJson::Value mv;
  if (!slot.Find("model_version", &mv)) {
    return nullptr;
  }
  if (mv.IsString()) {
    const char* s;
    size_t len;
    RETURN_IF_ERR(mv.AsString(&s, &len));
    ver_out->assign(s, len);
    return nullptr;
  }
  if (mv.IsNumber()) {
    int64_t iv;
    RETURN_IF_ERR(mv.AsInt(&iv));
    *ver_out = std::to_string(iv);
    return nullptr;
  }
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "'model_version' must be a string or integer");
}

TRITONSERVER_Error* AppendSlotDedupFingerprint(triton::common::TritonJson::Value& slot, std::string* out) {
  static const char* kMembers[] = {"id", "inputs", "outputs", "parameters"};
  for (const char* name : kMembers) {
    triton::common::TritonJson::Value member;
    if (slot.Find(name, &member)) {
      triton::common::TritonJson::WriteBuffer wb;
      RETURN_IF_ERR(member.Write(&wb));
      out->push_back(static_cast<char>(name[0]));
      out->append(name);
      out->push_back('\0');
      out->append(wb.Base(), wb.Size());
      out->push_back('\0');
    }
  }
  return nullptr;
}

std::string MakeSlotDedupKey(const std::string& model_name, const std::string& model_version_str, triton::common::TritonJson::Value& slot_json) {
  std::string key;
  key.reserve(model_name.size() + model_version_str.size() + 64);
  key.append(model_name);
  key.push_back('\x1f');
  key.append(model_version_str);
  key.push_back('\x1f');
  TRITONSERVER_Error* err = AppendSlotDedupFingerprint(slot_json, &key);
  if (err != nullptr) {
    TRITONSERVER_ErrorDelete(err);
  }
  return key;
}

bool TryFinalizeShardJsonToEvbuffer( TRITONSERVER_InferenceResponse* response, evbuffer* out) {
  TRITONSERVER_Error* err = TRITONSERVER_InferenceResponseError(response);
  if (err != nullptr) {
    TRITONSERVER_ErrorDelete(err);
    return false;
  }

  uint32_t parameter_count = 0;
  if (TRITONSERVER_InferenceResponseParameterCount(response, &parameter_count) != nullptr || parameter_count > 0) {
    return false;
  }

  uint32_t output_count = 0;
  if (TRITONSERVER_InferenceResponseOutputCount(response, &output_count) != nullptr || output_count != 1) {
    return false;
  }

  const char* cname = nullptr;
  TRITONSERVER_DataType datatype = TRITONSERVER_TYPE_INVALID;
  const int64_t* shape = nullptr;
  uint64_t dim_count = 0;
  const void* base = nullptr;
  size_t byte_size = 0;
  TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t memory_type_id = 0;
  void* userp = nullptr;

  if (TRITONSERVER_InferenceResponseOutput(response, 0, &cname, &datatype, &shape, &dim_count, &base, &byte_size, &memory_type, &memory_type_id, &userp) != nullptr) {
    return false;
  }

  if (base == nullptr) {
    return false;
  }

  auto* info = reinterpret_cast<HTTPAPIServer::AllocPayload::OutputInfo*>(userp);
  if (info == nullptr || info->kind_ != HTTPAPIServer::AllocPayload::OutputInfo::JSON || info->class_cnt_ > 0) {
    return false;
  }

  int64_t element_count = 1;
  for (uint64_t j = 0; j < dim_count; ++j) {
    if (shape[j] < 0) {
      return false;
    }
    element_count *= shape[j];
  }
  if (element_count <= 0) {
    return false;
  }

  const char* datatype_str = nullptr;
  switch (datatype) {
    case TRITONSERVER_TYPE_FP32:
      datatype_str = "FP32";
      if (byte_size < static_cast<size_t>(element_count) * sizeof(float)) {
        return false;
      }
      break;
    case TRITONSERVER_TYPE_FP64:
      if (element_count != 1) {
        return false;
      }
      datatype_str = "FP64";
      if (byte_size < sizeof(double)) {
        return false;
      }
      break;
    default:
      return false;
  }

  const char* request_id = "";
  if (TRITONSERVER_InferenceResponseId(response, &request_id) != nullptr) {
    request_id = "";
  }

  const char* model_name = "";
  int64_t model_version = 0;
  if (TRITONSERVER_InferenceResponseModel(response, &model_name, &model_version) != nullptr) {
    return false;
  }

  char version_buf[32];
  std::snprintf(version_buf, sizeof(version_buf), "%" PRId64, model_version);

  // Typical shard: metadata + 6 FP32 scores at 6 decimal places.
  evbuffer_expand(out, 384 + static_cast<size_t>(element_count) * 10);

  evbuffer_add(out, "{", 1);
  if (request_id[0] != '\0') {
    evbuffer_add(out, "\"id\":\"", 6);
    AppendJsonStringToEvbuffer(out, request_id);
    evbuffer_add(out, "\",", 2);
  }
  evbuffer_add(out, "\"model_name\":\"", 14);
  AppendJsonStringToEvbuffer(out, model_name);
  evbuffer_add(out, "\",\"model_version\":\"", 19);
  evbuffer_add(out, version_buf, std::strlen(version_buf));
  evbuffer_add(out, "\",\"outputs\":[{\"name\":\"", 22);
  AppendJsonStringToEvbuffer(out, cname);
  evbuffer_add(out, "\",\"shape\":[", 11);
  for (uint64_t j = 0; j < dim_count; ++j) {
    if (j > 0) {
      evbuffer_add(out, ",", 1);
    }
    char dim_buf[32];
    std::snprintf(dim_buf, sizeof(dim_buf), "%" PRId64, shape[j]);
    evbuffer_add(out, dim_buf, std::strlen(dim_buf));
  }
  evbuffer_add(out, "],\"datatype\":\"", 14);
  evbuffer_add(out, datatype_str, std::strlen(datatype_str));
  evbuffer_add(out, "\",\"data\":[", 10);

  if (datatype == TRITONSERVER_TYPE_FP32) {
    const float* values = reinterpret_cast<const float*>(base);
    for (int64_t i = 0; i < element_count; ++i) {
      if (i > 0) {
        evbuffer_add(out, ",", 1);
      }
      AppendScoreToEvbuffer(out, static_cast<double>(values[i]));
    }
  } else {
    AppendScoreToEvbuffer(out, *reinterpret_cast<const double*>(base));
  }

  evbuffer_add(out, "]}]}", 4);
  return true;
}

TRITONSERVER_Error* FinalizeShardToEvbuffer(TRITONSERVER_InferenceResponse* response, HTTPAPIServer::InferRequestClass* infer_request, evbuffer* out) {
  if (TryFinalizeShardJsonToEvbuffer(response, out)) {
    return nullptr;
  }
  return infer_request->FinalizeResponse(response, out);
}

struct SlotPrep {
  size_t request_idx{0};
  std::string model_name;
  int64_t model_version{0};
  std::string model_version_str;
};

struct UniqueInferGroup {
  size_t source_slot_idx{0};
  std::vector<size_t> response_slots;
};

struct MultiInferBatchState {
  std::string body_json;
  triton::common::TritonJson::Value root;
  std::vector<SlotPrep> slots;
  std::vector<std::shared_ptr<TRITONSERVER_InferenceRequest>> irequests;
  std::vector<std::unique_ptr<HTTPAPIServer::InferRequestClass>> shards;
  std::vector<std::unique_ptr<HTTPAPIServer::RequestReleasePayload>> releases;
};

class MultiInferAggregator : public std::enable_shared_from_this<MultiInferAggregator> {
 private:
  struct FinishPayload {
    std::shared_ptr<MultiInferAggregator> agg;
  };

 public:
  MultiInferAggregator(evhtp_request_t* req, size_t response_slot_count, size_t unique_infer_count, std::vector<std::vector<size_t>> fanout, std::shared_ptr<MultiInferBatchState> batch_state, evthr_t* reply_thread)
      : req_(req), response_slot_count_(response_slot_count), unique_infer_count_(unique_infer_count), fanout_(std::move(fanout)), batch_state_(std::move(batch_state)), reply_thread_(reply_thread), json_buffers_(response_slot_count), error_text_(response_slot_count), have_error_(response_slot_count, false) {}

  void CancelAllSubRequests() {
    if (cancel_sent_.exchange(true, std::memory_order_acq_rel)) {
      return;
    }
    for (auto& ir : batch_state_->irequests) {
      if (ir != nullptr) {
        LOG_TRITONSERVER_ERROR(TRITONSERVER_InferenceRequestCancel(ir.get()), "cancelling multi_infer sub-request");
      }
    }
  }

  void OnUniqueInferDone(
      size_t unique_idx, TRITONSERVER_Error* finalize_err,
      SharedEvbuffer shard_buf)
  {
    if (unique_idx >= fanout_.size()) {
      if (finalize_err != nullptr) {
        TRITONSERVER_ErrorDelete(finalize_err);
      }
      return;
    }

    if (finalize_err != nullptr) {
      const std::string message = TRITONSERVER_ErrorMessage(finalize_err);
      TRITONSERVER_ErrorDelete(finalize_err);
      for (const size_t slot : fanout_[unique_idx]) {
        have_error_[slot] = true;
        error_text_[slot] = message;
      }
    } else if ((shard_buf == nullptr) || (evbuffer_get_length(shard_buf.get()) == 0)) {
      for (const size_t slot : fanout_[unique_idx]) {
        have_error_[slot] = true;
        error_text_[slot] = "empty multi_infer sub-response";
      }
    } else {
      for (const size_t slot : fanout_[unique_idx]) {
        json_buffers_[slot] = shard_buf;
      }
    }

    const size_t prev = done_count_.fetch_add(1, std::memory_order_acq_rel);
    if (prev + 1 < unique_infer_count_) {
      return;
    }

    bool expected = false;
    if (!reply_scheduled_.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_relaxed)) {
      return;
    }

    ScheduleReply();
  }

 private:
  void ScheduleReply() {
    auto* fp = new FinishPayload{shared_from_this()};
    if (!EvthrDeferWithRetry(reply_thread_, SendReplyThunk, fp)) {
#ifdef TRITON_ENABLE_LOGGING
      LOG_ERROR << "failed to defer multi_infer reply to HTTP worker thread";
#endif  // TRITON_ENABLE_LOGGING
      delete fp;
    }
  }

  static void SendReplyThunk(evthr_t* /*thr*/, void* arg, void* /*shared*/) {
    std::unique_ptr<FinishPayload> fp(static_cast<FinishPayload*>(arg));
    fp->agg->BuildHttpResponse();
    fp->agg->SendHttpReply();
  }

  void BuildHttpResponse() {
    response_code_ = EVHTP_RES_OK;

    size_t reserve_size = 16;
    for (size_t i = 0; i < response_slot_count_; ++i) {
      reserve_size += 1;
      if (have_error_[i]) {
        reserve_size += error_text_[i].size() + 32;
      } else if (json_buffers_[i] != nullptr) {
        reserve_size += evbuffer_get_length(json_buffers_[i].get());
      }
    }
    evbuffer_expand(req_->buffer_out, reserve_size);

    AddContentTypeHeader(req_, "application/json");
    evbuffer_add(req_->buffer_out, "{\"responses\":[", 14);

    for (size_t i = 0; i < response_slot_count_; ++i) {
      if (i > 0) {
        evbuffer_add(req_->buffer_out, ",", 1);
      }
      if (have_error_[i]) {
        AppendShardErrorJsonToEvbuffer(req_->buffer_out, error_text_[i]);
      } else if (json_buffers_[i] != nullptr) {
        evbuffer* shard = json_buffers_[i].get();
        const size_t len = evbuffer_get_length(shard);
        if (len == 0) {
          continue;
        }
        if (json_buffers_[i].use_count() == 1) {
          evbuffer_add_buffer(req_->buffer_out, shard);
          json_buffers_[i].reset();
        } else if (const unsigned char* data = evbuffer_pullup(shard, len)) {
          evbuffer_add(req_->buffer_out, data, len);
        }
      }
    }

    evbuffer_add(req_->buffer_out, "]}", 2);
  }

  void SendHttpReply() {
    if (req_ != nullptr) {
      evhtp_send_reply(req_, response_code_);
      evhtp_request_resume(req_);
    }
  }

  evhtp_request_t* req_;
  const size_t response_slot_count_;
  const size_t unique_infer_count_;
  std::vector<std::vector<size_t>> fanout_;
  std::shared_ptr<MultiInferBatchState> batch_state_;
  evthr_t* const reply_thread_;
  std::vector<SharedEvbuffer> json_buffers_;
  std::vector<std::string> error_text_;
  std::vector<bool> have_error_;
  std::atomic<size_t> done_count_{0};
  std::atomic<bool> cancel_sent_{false};
  std::atomic<bool> reply_scheduled_{false};
  evhtp_res response_code_{EVHTP_RES_OK};
};

class MultiInferSingleSlotRequest : public HTTPAPIServer::InferRequestClass {
 public:
  struct ReplyPayload {
    evhtp_request_t* req{nullptr};
    SharedEvbuffer shard_buf;
    std::string error_message;
    bool have_error{false};
  };

  MultiInferSingleSlotRequest(
      TRITONSERVER_Server* server, evhtp_request_t* req,
      DataCompressor::Type response_compression_type,
      const std::shared_ptr<TRITONSERVER_InferenceRequest>& triton_request,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      evthr_t* reply_thread)
      : HTTPAPIServer::InferRequestClass(server, req, response_compression_type, triton_request, shm_manager, false /* pause */, false /* fini hook */), reply_thread_(reply_thread) {}

  static void InferResponseComplete(TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp) {
    auto* infer_request = reinterpret_cast<MultiInferSingleSlotRequest*>(userp);

    if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) == 0) {
      if (response != nullptr) {
        LOG_TRITONSERVER_ERROR(TRITONSERVER_InferenceResponseDelete(response), "deleting non-final multi_infer inference response");
      }
      return;
    }

    auto payload = std::make_unique<ReplyPayload>();
    payload->req = infer_request->req_;

    if (response != nullptr) {
      ++infer_request->response_count_;
    }

    TRITONSERVER_Error* err = nullptr;
    if (infer_request->response_count_ != 1) {
      const std::string msg = "expected a single response, got " + std::to_string(infer_request->response_count_);
      err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, msg.c_str());
    } else if (response != nullptr) {
      payload->shard_buf = MakeSharedEvbuffer(evbuffer_new());
      err = FinalizeShardToEvbuffer(response, infer_request, payload->shard_buf.get());
#ifdef TRITON_ENABLE_TRACING
      if (infer_request->trace_ != nullptr) {
        infer_request->trace_->CaptureTimestamp("INFER_RESPONSE_COMPLETE", TraceManager::CaptureTimestamp());
      }
#endif  // TRITON_ENABLE_TRACING
    }

    if (response != nullptr) {
      LOG_TRITONSERVER_ERROR(TRITONSERVER_InferenceResponseDelete(response), "deleting inference response");
    }

    if (err != nullptr) {
      payload->have_error = true;
      payload->error_message = TRITONSERVER_ErrorMessage(err);
      payload->shard_buf.reset();
      TRITONSERVER_ErrorDelete(err);
    }

    if (!EvthrDeferWithRetry(infer_request->reply_thread_, ReplyThunk, payload.get())) {
#ifdef TRITON_ENABLE_LOGGING
      LOG_ERROR << "failed to defer multi_infer single-slot reply";
#endif  // TRITON_ENABLE_LOGGING
    } else {
      payload.release();
    }

    delete infer_request;
  }

 private:
  static void ReplyThunk(evthr_t* /*thr*/, void* arg, void* /*shared*/) {
    std::unique_ptr<ReplyPayload> payload(static_cast<ReplyPayload*>(arg));
    if (payload->req != nullptr) {
      AddContentTypeHeader(payload->req, "application/json");
      evbuffer_add(payload->req->buffer_out, "{\"responses\":[", 14);
      if (payload->have_error) {
        AppendShardErrorJsonToEvbuffer(payload->req->buffer_out, payload->error_message);
      } else if (payload->shard_buf != nullptr) {
        evbuffer_add_buffer(payload->req->buffer_out, payload->shard_buf.get());
      }
      evbuffer_add(payload->req->buffer_out, "]}", 2);
      evhtp_send_reply(payload->req, EVHTP_RES_OK);
      evhtp_request_resume(payload->req);
    }
  }

  evthr_t* const reply_thread_;
};

class MultiInferShardRequest : public HTTPAPIServer::InferRequestClass {
 public:
  MultiInferShardRequest(TRITONSERVER_Server* server, evhtp_request_t* req, DataCompressor::Type response_compression_type, const std::shared_ptr<TRITONSERVER_InferenceRequest>& triton_request, const std::shared_ptr<SharedMemoryManager>& shm_manager, std::shared_ptr<MultiInferAggregator> aggregator, const size_t unique_idx)
      : HTTPAPIServer::InferRequestClass(server, req, response_compression_type, triton_request, shm_manager, false /* pause */, false /* fini hook */), aggregator_(std::move(aggregator)), unique_idx_(unique_idx) {}

  static void InferResponseComplete(TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp) {
    auto* infer_request = reinterpret_cast<MultiInferShardRequest*>(userp);

    if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) == 0) {
      if (response != nullptr) {
        LOG_TRITONSERVER_ERROR(TRITONSERVER_InferenceResponseDelete(response),"deleting non-final multi_infer inference response");
      }
      return;
    }

    if (response != nullptr) {
      ++infer_request->response_count_;
    }

    TRITONSERVER_Error* err = nullptr;
    SharedEvbuffer shard_buf;
    if (infer_request->response_count_ != 1) {
      const std::string msg = std::string("expected a single response, got ") + std::to_string(infer_request->response_count_);
      err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, msg.c_str());
    } else if (response != nullptr) {
      shard_buf = MakeSharedEvbuffer(evbuffer_new());
      err = FinalizeShardToEvbuffer(response, infer_request, shard_buf.get());
#ifdef TRITON_ENABLE_TRACING
      if (infer_request->trace_ != nullptr) {
        infer_request->trace_->CaptureTimestamp("INFER_RESPONSE_COMPLETE", TraceManager::CaptureTimestamp());
      }
#endif  // TRITON_ENABLE_TRACING
    }

    if (response != nullptr) {
      LOG_TRITONSERVER_ERROR(TRITONSERVER_InferenceResponseDelete(response), "deleting inference response");
    }

    infer_request->aggregator_->OnUniqueInferDone(infer_request->unique_idx_, err, std::move(shard_buf));
    delete infer_request;
  }

 private:
  std::shared_ptr<MultiInferAggregator> aggregator_;
  const size_t unique_idx_;
};

struct DecompressedBodyGuard {
  evbuffer* buffer{nullptr};
  ~DecompressedBodyGuard() {
    if (buffer != nullptr) {
      evbuffer_free(buffer);
    }
  }
  evbuffer* Release() {
    evbuffer* released = buffer;
    buffer = nullptr;
    return released;
  }
};

void RespondWithTritonError(evhtp_request_t* req, TRITONSERVER_Error* err) {
  AddContentTypeHeader(req, "application/json");
  EVBufferAddErrorJson(req->buffer_out, err);
  evhtp_send_reply(req, HttpCodeFromError(err));
  TRITONSERVER_ErrorDelete(err);
  evhtp_request_resume(req);
}

std::string MakeModelVersionDedupKey( const std::string& model_name, const std::string& model_version_str) {
  std::string key = model_name + '\x1f' + model_version_str;
  return key;
}

void BuildUniqueInferGroups(triton::common::TritonJson::Value& requests, std::vector<SlotPrep>& slots, std::vector<UniqueInferGroup>* unique_groups) {
  std::unordered_map<std::string, size_t> dedup_key_to_unique;
  dedup_key_to_unique.reserve(slots.size());
  unique_groups->clear();
  unique_groups->reserve(slots.size());

  for (size_t i = 0; i < slots.size(); ++i) {
    triton::common::TritonJson::Value slot;
    TRITONSERVER_Error* slot_err = requests.At(slots[i].request_idx, &slot);
    if (slot_err != nullptr) {
      TRITONSERVER_ErrorDelete(slot_err);
      continue;
    }

    const std::string model_version_key = MakeModelVersionDedupKey(slots[i].model_name, slots[i].model_version_str);
    const auto model_version_it = dedup_key_to_unique.find(model_version_key);
    if (model_version_it == dedup_key_to_unique.end()) {
      const size_t unique_idx = unique_groups->size();
      dedup_key_to_unique.emplace(model_version_key, unique_idx);
      UniqueInferGroup group;
      group.source_slot_idx = i;
      group.response_slots.push_back(i);
      unique_groups->push_back(std::move(group));
      continue;
    }

    const std::string full_key = MakeSlotDedupKey(slots[i].model_name, slots[i].model_version_str, slot);
    const auto full_it = dedup_key_to_unique.find(full_key);
    if (full_it != dedup_key_to_unique.end()) {
      (*unique_groups)[full_it->second].response_slots.push_back(i);
      continue;
    }

    const size_t first_unique_idx = model_version_it->second;
    const size_t source_slot_idx = (*unique_groups)[first_unique_idx].source_slot_idx;
    triton::common::TritonJson::Value source_slot;
    TRITONSERVER_Error* source_err = requests.At(slots[source_slot_idx].request_idx, &source_slot);
    if (source_err != nullptr) {
      TRITONSERVER_ErrorDelete(source_err);
      const size_t unique_idx = unique_groups->size();
      dedup_key_to_unique.emplace(full_key, unique_idx);
      UniqueInferGroup group;
      group.source_slot_idx = i;
      group.response_slots.push_back(i);
      unique_groups->push_back(std::move(group));
      continue;
    }

    const std::string source_full_key = MakeSlotDedupKey(slots[source_slot_idx].model_name, slots[source_slot_idx].model_version_str, source_slot);
    if (full_key == source_full_key) {
      (*unique_groups)[first_unique_idx].response_slots.push_back(i);
      dedup_key_to_unique.emplace(full_key, first_unique_idx);
      continue;
    }

    const size_t unique_idx = unique_groups->size();
    dedup_key_to_unique.emplace(full_key, unique_idx);
    UniqueInferGroup group;
    group.source_slot_idx = i;
    group.response_slots.push_back(i);
    unique_groups->push_back(std::move(group));
  }
}

}  // namespace

void HTTPAPIServer::HandleMultiInfer(evhtp_request_t* req) {
  RETURN_AND_RESPOND_IF_RESTRICTED(req, RestrictedCategory::INFERENCE, restricted_apis_);

  if (req->method != htp_method_POST) {
    RETURN_AND_RESPOND_WITH_ERR(req, EVHTP_RES_METHNALLOWED, "Method Not Allowed");
  }

  evhtp_request_pause(req);

  DecompressedBodyGuard decompressed_body;
  TRITONSERVER_Error* err = DecompressBuffer(req, &decompressed_body.buffer);
  if (err != nullptr) {
    RespondWithTritonError(req, err);
    return;
  }

  int32_t content_length = 0;
  err = GetContentLength(req, decompressed_body.buffer, &content_length);
  if (err != nullptr) {
    RespondWithTritonError(req, err);
    return;
  }

  size_t header_length = static_cast<size_t>(content_length);
  if (evhtp_kv_find(req->headers_in, kInferHeaderContentLengthHTTPHeader) != nullptr) {
    err = GetInferenceHeaderLength(req, content_length, &header_length);
    if (err != nullptr) {
      RespondWithTritonError(req, err);
      return;
    }
  }

  if (header_length < static_cast<size_t>(content_length)) {
    RespondWithTritonError(req, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED, "POST /v2/multi_infer does not support binary tensor input; send JSON with inline 'data' fields"));
    return;
  }

  evbuffer* body_buf = (decompressed_body.buffer != nullptr) ? decompressed_body.buffer : req->buffer_in;
  const size_t body_size = evbuffer_get_length(body_buf);
  if (body_size < static_cast<size_t>(content_length)) {
    RespondWithTritonError(req, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "request body shorter than Content-Length"));
    return;
  }

  auto batch_state = std::make_shared<MultiInferBatchState>();
  const char* json_ptr = nullptr;
  if (const unsigned char* pulled = evbuffer_pullup(body_buf, body_size)) {
    json_ptr = reinterpret_cast<const char*>(pulled);
  } else {
    batch_state->body_json.resize(body_size);
    if (evbuffer_copyout(body_buf, batch_state->body_json.data(), body_size) != static_cast<ev_ssize_t>(body_size)) {
      RespondWithTritonError(req, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "failed to read multi_infer request body"));
      return;
    }
    json_ptr = batch_state->body_json.data();
  }

  err = batch_state->root.Parse(json_ptr, body_size);
  if (err != nullptr) {
    RespondWithTritonError(req, err);
    return;
  }

  triton::common::TritonJson::Value requests;
  if (!batch_state->root.Find("requests", &requests)) {
    err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "Request body must include a JSON array field 'requests'");
    RespondWithTritonError(req, err);
    return;
  }

  const size_t n = requests.ArraySize();
  if (n == 0) {
    err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "'requests' array must be non-empty");
    RespondWithTritonError(req, err);
    return;
  }

  if (n == 1) {
    triton::common::TritonJson::Value slot;
    err = requests.At(0, &slot);
    if (err != nullptr) {
      RespondWithTritonError(req, err);
      return;
    }

    const char* mn_c;
    size_t mn_len;
    err = slot.MemberAsString("model_name", &mn_c, &mn_len);
    if (err != nullptr) {
      RespondWithTritonError(req, err);
      return;
    }
    const std::string model_name(mn_c, mn_len);

    std::string model_version_str;
    err = GetModelVersionStringFromSlot(slot, &model_version_str);
    if (err != nullptr) {
      RespondWithTritonError(req, err);
      return;
    }
    int64_t model_version = 0;
    err = GetModelVersionFromString(model_version_str, &model_version);
    if (err != nullptr) {
      RespondWithTritonError(req, err);
      return;
    }
    err = CheckTransactionPolicy(req, model_name, model_version);
    if (err != nullptr) {
      RespondWithTritonError(req, err);
      return;
    }

    TRITONSERVER_InferenceRequest* ireq = nullptr;
    err = TRITONSERVER_InferenceRequestNew(&ireq, server_.get(), model_name.c_str(), model_version);
    if (err != nullptr) {
      RespondWithTritonError(req, err);
      return;
    }

    auto ireq_shared = std::shared_ptr<TRITONSERVER_InferenceRequest>(
      ireq, [](TRITONSERVER_InferenceRequest* r) {
        LOG_TRITONSERVER_ERROR(TRITONSERVER_InferenceRequestDelete(r),"deleting HTTP multi_infer sub-request");
    });

    evthr_t* reply_thread = evhtp_request_get_connection(req)->thread;
    auto infer_request = std::make_unique<MultiInferSingleSlotRequest>(server_.get(), req, GetResponseCompressionType(req), ireq_shared, shm_manager_, reply_thread);

    err = FillMultiInferSlotTritonRequest(model_name, slot, ireq_shared.get(), infer_request.get());
    if (err != nullptr) {
      RespondWithTritonError(req, err);
      return;
    }

    evbuffer* body_buffer_holder = decompressed_body.Release();
    auto release_payload = std::make_unique<HTTPAPIServer::RequestReleasePayload>(ireq_shared, body_buffer_holder);

    err = ScheduleInferAsync(req, ireq_shared.get(), infer_request.get(), release_payload.get(), nullptr, MultiInferSingleSlotRequest::InferResponseComplete, false /* forward_headers */);
    if (err != nullptr) {
      if (body_buffer_holder != nullptr) {
        evbuffer_free(body_buffer_holder);
      }
      RespondWithTritonError(req, err);
      return;
    }

    infer_request.release();
    release_payload.release();
    return;
  }

  // Phase 1: validate every slot before scheduling any infer.
  std::vector<SlotPrep> slots;
  slots.reserve(n);
  std::unordered_set<std::string> policy_checked_models;
  policy_checked_models.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    triton::common::TritonJson::Value slot;
    err = requests.At(i, &slot);
    if (err != nullptr) {
      RespondWithTritonError(req, err);
      return;
    }

    SlotPrep prep;
    const char* mn_c;
    size_t mn_len;
    err = slot.MemberAsString("model_name", &mn_c, &mn_len);
    if (err != nullptr) {
      RespondWithTritonError(req, err);
      return;
    }
    prep.model_name.assign(mn_c, mn_len);

    err = GetModelVersionStringFromSlot(slot, &prep.model_version_str);
    if (err != nullptr) {
      RespondWithTritonError(req, err);
      return;
    }
    err = GetModelVersionFromString(prep.model_version_str, &prep.model_version);
    if (err != nullptr) {
      RespondWithTritonError(req, err);
      return;
    }
    if (policy_checked_models.insert(prep.model_name).second) {
      err = CheckTransactionPolicy(req, prep.model_name, prep.model_version);
      if (err != nullptr) {
        RespondWithTritonError(req, err);
        return;
      }
    }

    prep.request_idx = i;
    slots.push_back(std::move(prep));
  }

  batch_state->slots = std::move(slots);

  // Phase 2: deduplicate (lazy fingerprint — full I/O hash only on model+version collision).
  std::vector<UniqueInferGroup> unique_groups;
  BuildUniqueInferGroups(requests, batch_state->slots, &unique_groups);

  const size_t unique_count = unique_groups.size();
  evthr_t* reply_thread = evhtp_request_get_connection(req)->thread;

  batch_state->irequests.reserve(unique_count);
  batch_state->shards.reserve(unique_count);
  batch_state->releases.reserve(unique_count);

  std::vector<std::vector<size_t>> fanout(unique_count);
  for (size_t u = 0; u < unique_count; ++u) {
    fanout[u] = std::move(unique_groups[u].response_slots);
  }

  auto aggregator = std::make_shared<MultiInferAggregator>(req, n, unique_count, std::move(fanout), batch_state, reply_thread);

  // Phase 3: create and fill one infer per unique slot group.
  for (size_t u = 0; u < unique_count; ++u) {
    const size_t source_idx = unique_groups[u].source_slot_idx;
    SlotPrep& prep = batch_state->slots[source_idx];

    TRITONSERVER_InferenceRequest* ireq = nullptr;
    err = TRITONSERVER_InferenceRequestNew(&ireq, server_.get(), prep.model_name.c_str(), prep.model_version);
    if (err != nullptr) {
      aggregator->CancelAllSubRequests();
      RespondWithTritonError(req, err);
      return; 
    }

    auto ireq_shared = std::shared_ptr<TRITONSERVER_InferenceRequest>(
      ireq, [](TRITONSERVER_InferenceRequest* r) {
        LOG_TRITONSERVER_ERROR(TRITONSERVER_InferenceRequestDelete(r), "deleting HTTP multi_infer sub-request");
    });
    batch_state->irequests.push_back(ireq_shared);
    batch_state->shards.push_back(std::make_unique<MultiInferShardRequest>(server_.get(), req, GetResponseCompressionType(req), ireq_shared, shm_manager_, aggregator, u));
    batch_state->releases.push_back(std::make_unique<HTTPAPIServer::RequestReleasePayload>(ireq_shared, nullptr));

    triton::common::TritonJson::Value slot;
    err = requests.At(prep.request_idx, &slot);
    if (err != nullptr) {
      aggregator->CancelAllSubRequests();
      RespondWithTritonError(req, err);
      return;
    }
    err = FillMultiInferSlotTritonRequest(prep.model_name, slot, batch_state->irequests[u].get(), batch_state->shards[u].get());
    if (err != nullptr) {
      aggregator->CancelAllSubRequests();
      RespondWithTritonError(req, err);
      return;
    }
  }

  evbuffer* body_buffer_holder = decompressed_body.Release();

  // Phase 4: schedule all unique infers.
  for (size_t u = 0; u < unique_count; ++u) {
    if ((u == 0) && (body_buffer_holder != nullptr)) {
      batch_state->releases[0].reset(new HTTPAPIServer::RequestReleasePayload(batch_state->irequests[0], body_buffer_holder));
      body_buffer_holder = nullptr;
    }

    err = ScheduleInferAsync(req, batch_state->irequests[u].get(), batch_state->shards[u].get(), batch_state->releases[u].get(), nullptr, MultiInferShardRequest::InferResponseComplete, false /* forward_headers */);
    if (err != nullptr) {
      if (body_buffer_holder != nullptr) {
        evbuffer_free(body_buffer_holder);
        body_buffer_holder = nullptr;
      }
      aggregator->CancelAllSubRequests();
      RespondWithTritonError(req, err);
      return;
    }

    batch_state->releases[u].release();
    batch_state->shards[u].release();
  }

  if (body_buffer_holder != nullptr) {
    evbuffer_free(body_buffer_holder);
  }
}

}}  // namespace triton::server
