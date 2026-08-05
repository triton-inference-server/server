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
#ifdef TRITON_ENABLE_MYSQL_ODBC
#include "transform.h"
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

void AppendJsonEscaped(std::string* out, const std::string& value)
{
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

TRITONSERVER_Error* CopyInferSlotBodyJson(triton::common::TritonJson::Value& slot, triton::common::TritonJson::Value* infer_json) {
  *infer_json = triton::common::TritonJson::Value(triton::common::TritonJson::ValueType::OBJECT);
  {
    triton::common::TritonJson::Value v;
    if (slot.Find("id", &v)) {
      RETURN_IF_ERR(infer_json->Add("id", std::move(v)));
    }
  }
  {
    triton::common::TritonJson::Value v;
    if (!slot.Find("inputs", &v)) {
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "Each entry in 'requests' must contain an 'inputs' array");
    }
    RETURN_IF_ERR(infer_json->Add("inputs", std::move(v)));
  }
  {
    triton::common::TritonJson::Value v;
    if (slot.Find("outputs", &v)) {
      RETURN_IF_ERR(infer_json->Add("outputs", std::move(v)));
    }
  }
  {
    triton::common::TritonJson::Value v;
    if (slot.Find("parameters", &v)) {
      RETURN_IF_ERR(infer_json->Add("parameters", std::move(v)));
    }
  }
  return nullptr;
}

TRITONSERVER_Error* GetModelVersionStringFromSlot(triton::common::TritonJson::Value& slot, std::string* ver_out)
{
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

#ifdef TRITON_ENABLE_MYSQL_ODBC

inline float RoundScore6(float x) {
  if (!std::isfinite(x)) {
    return x;
  }
  const int64_t scaled = static_cast<int64_t>(std::floor(x * 1e6f));
  return static_cast<float>(scaled) / 1e6f;
}

bool WriteImpsShapedMultiInferResponse(const std::vector<std::vector<ImpRouteRow>>& routing_slots, const std::vector<std::string>& slot_model_names, const std::vector<std::vector<float>>& slot_rows, const int imp_count, rapidjson::StringBuffer* sb) {
  if (imp_count <= 0 || routing_slots.empty() || sb == nullptr) {
    return false;
  }
  if (slot_rows.size() != routing_slots.size() || slot_model_names.size() != routing_slots.size()) {
    return false;
  }

  struct CampAgg {
    int32_t cid{0};
    const std::string* mdl{nullptr};
    std::vector<std::vector<float>> by_adsize;
    bool initialized{false};
  };

  size_t total_rows = 0;
  for (size_t si = 0; si < routing_slots.size(); ++si) {
    total_rows += routing_slots[si].size();
  }

  std::vector<std::vector<CampAgg>> camps_per_imp(static_cast<size_t>(imp_count));
  for (size_t si = 0; si < routing_slots.size(); ++si) {
    const auto& slot_r = routing_slots[si];
    const std::string& slot_mdl = slot_model_names[si];
    const size_t R = slot_r.size();
    if (si >= slot_rows.size()) {
      return false;
    }
    const auto& row_scores = slot_rows[si];
    if (row_scores.size() != R) {
      return false;
    }
    for (size_t ri = 0; ri < R; ++ri) {
      const ImpRouteRow& rc = slot_r[ri];
      if (rc.imp_idx < 0 || rc.imp_idx >= imp_count || rc.camp_idx < 0) {
        return false;
      }
      auto& imp_camps = camps_per_imp[static_cast<size_t>(rc.imp_idx)];
      if (static_cast<size_t>(rc.camp_idx) >= imp_camps.size()) {
        imp_camps.resize(static_cast<size_t>(rc.camp_idx) + 1);
      }
      CampAgg& ca = imp_camps[static_cast<size_t>(rc.camp_idx)];
      if (!ca.initialized) {
        ca.cid = rc.cid;
        ca.mdl = &slot_mdl;
        ca.initialized = true;
      } else if (ca.cid != rc.cid || *ca.mdl != slot_mdl) {
        return false;
      }
      const size_t ad_idx = static_cast<size_t>(rc.adsize_idx);
      if (ad_idx >= ca.by_adsize.size()) {
        ca.by_adsize.resize(ad_idx + 1);
      }
      ca.by_adsize[ad_idx] = {row_scores[ri]};
    }
  }

  sb->Clear();
  sb->Reserve(static_cast<rapidjson::SizeType>(128 + total_rows * 96));
  rapidjson::Writer<rapidjson::StringBuffer> writer(*sb);
  writer.SetMaxDecimalPlaces(6);

  writer.StartObject();
  writer.Key("imps");
  writer.StartArray();
  for (int ii = 0; ii < imp_count; ++ii) {
    writer.StartObject();
    writer.Key("camps");
    writer.StartArray();
    for (const CampAgg& ca : camps_per_imp[static_cast<size_t>(ii)]) {
      if (!ca.initialized) {
        continue;
      }
      writer.StartObject();
      writer.Key("cid");
      writer.Int(ca.cid);
      writer.Key("mdl");
      writer.String(ca.mdl->c_str(), static_cast<rapidjson::SizeType>(ca.mdl->size()));
      writer.Key("score");
      writer.StartArray();
      for (const std::vector<float>& vec : ca.by_adsize) {
        if (vec.empty()) {
          continue;
        }
        if (vec.size() == 1) {
          writer.Double(static_cast<double>(RoundScore6(vec[0])));
        } else {
          writer.StartArray();
          for (float f : vec) {
            writer.Double(static_cast<double>(RoundScore6(f)));
          }
          writer.EndArray();
        }
      }
      writer.EndArray();
      writer.EndObject();
    }
    writer.EndArray();
    writer.EndObject();
  }
  writer.EndArray();
  writer.EndObject();
  return true;
}

#endif  // TRITON_ENABLE_MYSQL_ODBC

class MultiInferAggregator : public std::enable_shared_from_this<MultiInferAggregator> {
 private:
  struct FinishPayload {
    std::shared_ptr<MultiInferAggregator> agg;
  };

 public:
  MultiInferAggregator(evhtp_request_t* req, size_t slot_count, evthr_t* reply_thread,
      std::vector<std::shared_ptr<TRITONSERVER_InferenceRequest>> irequests
#ifdef TRITON_ENABLE_MYSQL_ODBC
      , std::vector<std::vector<ImpRouteRow>> imp_routing_slots = {},
      std::vector<std::string> slot_model_names = {},
      int imp_routing_imp_count = 0
#endif
      )
      : req_(req), n_(slot_count), reply_thread_(reply_thread),
        irequests_(std::move(irequests)), success_buffers_(slot_count, nullptr),
        error_text_(slot_count), have_error_(slot_count, 0)
#ifdef TRITON_ENABLE_MYSQL_ODBC
        , imp_routing_slots_(std::move(imp_routing_slots)),
        slot_model_names_(std::move(slot_model_names)),
        imp_routing_imp_count_(imp_routing_imp_count)
#endif
  {
#ifdef TRITON_ENABLE_MYSQL_ODBC
    slot_row_outputs_.assign(n_, {});
#endif
  }

  ~MultiInferAggregator()
  {
    for (evbuffer* buf : success_buffers_) {
      if (buf != nullptr) {
        evbuffer_free(buf);
      }
    }
  }

  std::shared_ptr<TRITONSERVER_InferenceRequest> IrequestAt(size_t i) const
  {
    return irequests_[i];
  }

#ifdef TRITON_ENABLE_MYSQL_ODBC
  bool WantsShardParsedRows() const
  {
    return imp_routing_imp_count_ > 0 && !imp_routing_slots_.empty();
  }

  size_t ExpectedRowsForSlot(size_t slot) const
  {
    return (slot < imp_routing_slots_.size()) ? imp_routing_slots_[slot].size() : 0;
  }
#endif

  void CancelAllSubRequests()
  {
    if (cancel_sent_.exchange(true, std::memory_order_acq_rel)) {
      return;
    }
    for (auto& ir : irequests_) {
      if (ir != nullptr) {
        LOG_TRITONSERVER_ERROR(TRITONSERVER_InferenceRequestCancel(ir.get()), "cancelling multi_infer sub-request");
      }
    }
  }

  void OnShardDone(size_t slot, TRITONSERVER_Error* finalize_err, evbuffer* response_json, std::vector<float> parsed_scalar_output = {}) {
    if (finalize_err != nullptr) {
      have_error_[slot] = 1;
      error_text_[slot] = TRITONSERVER_ErrorMessage(finalize_err);
      TRITONSERVER_ErrorDelete(finalize_err);
      if (response_json != nullptr) {
        evbuffer_free(response_json);
      }
      CancelAllSubRequests();
    } else {
      success_buffers_[slot] = response_json;
#ifdef TRITON_ENABLE_MYSQL_ODBC
      if (!parsed_scalar_output.empty() && slot < slot_row_outputs_.size()) {
        slot_row_outputs_[slot] = std::move(parsed_scalar_output);
      }
#endif
    }

    std::atomic_thread_fence(std::memory_order_release);

    const size_t prev = done_count_.fetch_add(1, std::memory_order_acq_rel);
    if (prev + 1 < n_) {
      return;
    }

    bool expected = false;
    if (!reply_scheduled_.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_relaxed)) {
      return;
    }

    auto* fp = new FinishPayload{shared_from_this()};
    evthr_defer(reply_thread_, FinishThunk, fp);
  }

 private:
  static void FinishThunk(evthr_t* /*thr*/, void* arg, void* /*shared*/) {
    std::unique_ptr<FinishPayload> fp(static_cast<FinishPayload*>(arg));
    fp->agg->WriteHttpReply();
  }

  static void AppendShardErrorJson(evbuffer* out, const std::string& message)
  {
    std::string fragment;
    fragment.reserve(message.size() + 24);
    fragment += "{\"error\":{\"message\":\"";
    AppendJsonEscaped(&fragment, message);
    fragment += "\"}}";
    evbuffer_add(out, fragment.data(), fragment.size());
  }

  void WriteHttpReply() {
    std::atomic_thread_fence(std::memory_order_acquire);

#ifdef TRITON_ENABLE_MYSQL_ODBC
    if (WantsShardParsedRows() && !cancel_sent_.load(std::memory_order_acquire)) {
      bool any_err = false;
      for (size_t i = 0; i < n_; ++i) {
        if (have_error_[i]) {
          any_err = true;
          break;
        }
      }
      if (!any_err) {
        rapidjson::StringBuffer sb;
        if (WriteImpsShapedMultiInferResponse(imp_routing_slots_, slot_model_names_, slot_row_outputs_, imp_routing_imp_count_, &sb)) {
          AddContentTypeHeader(req_, "application/json");
          evbuffer_add(req_->buffer_out, sb.GetString(), sb.GetSize());
          evhtp_send_reply(req_, EVHTP_RES_OK);
          evhtp_request_resume(req_);
          return;
        }
        static const char kFoldErr[] = "{\"error\":\"failed to fold imps response\"}";
        AddContentTypeHeader(req_, "application/json");
        evbuffer_add(req_->buffer_out, kFoldErr, sizeof(kFoldErr) - 1);
        evhtp_send_reply(req_, EVHTP_RES_BADREQ);
        evhtp_request_resume(req_);
        return;
      }

      bool any_shard_error = false;
      for (size_t i = 0; i < n_; ++i) {
        if (have_error_[i]) {
          any_shard_error = true;
          break;
        }
      }
      if (any_shard_error) {
        triton::common::TritonJson::Value root(triton::common::TritonJson::ValueType::OBJECT);
        triton::common::TritonJson::Value errors(root, triton::common::TritonJson::ValueType::ARRAY);
        for (size_t i = 0; i < n_; ++i) {
          TRITONSERVER_Error* ae = nullptr;
          if (have_error_[i]) {
            ae = errors.AppendString(error_text_[i]);
          } else {
            ae = errors.AppendString("");
          }
          if (ae != nullptr) {
            LOG_TRITONSERVER_ERROR(ae, "multi_infer: building errors array");
            TRITONSERVER_ErrorDelete(ae);
          }
        }
        TRITONSERVER_Error* re = root.Add("errors", std::move(errors));
        if (re != nullptr) {
          LOG_TRITONSERVER_ERROR(re, "multi_infer: building root JSON");
          AddContentTypeHeader(req_, "application/json");
          EVBufferAddErrorJson(req_->buffer_out, re);
          evhtp_send_reply(req_, HttpCodeFromError(re));
          TRITONSERVER_ErrorDelete(re);
          evhtp_request_resume(req_);
          return;
        }
        triton::common::TritonJson::WriteBuffer wb;
        TRITONSERVER_Error* we = root.Write(&wb);
        if (we != nullptr) {
          AddContentTypeHeader(req_, "application/json");
          EVBufferAddErrorJson(req_->buffer_out, we);
          evhtp_send_reply(req_, HttpCodeFromError(we));
          TRITONSERVER_ErrorDelete(we);
        } else {
          AddContentTypeHeader(req_, "application/json");
          evbuffer_add(req_->buffer_out, wb.Base(), wb.Size());
          evhtp_send_reply(req_, EVHTP_RES_BADREQ);
        }
        evhtp_request_resume(req_);
        return;
      }
    }
#endif  // TRITON_ENABLE_MYSQL_ODBC

    evbuffer* out = evbuffer_new();
    evbuffer_add(out, "{\"responses\":[", 14);

    for (size_t i = 0; i < n_; ++i) {
      if (i > 0) {
        evbuffer_add(out, ",", 1);
      }
      if (have_error_[i]) {
        AppendShardErrorJson(out, error_text_[i]);
      } else if (success_buffers_[i] == nullptr || evbuffer_get_length(success_buffers_[i]) == 0) {
        AppendShardErrorJson(out, "empty multi_infer sub-response");
        if (success_buffers_[i] != nullptr) {
          evbuffer_free(success_buffers_[i]);
          success_buffers_[i] = nullptr;
        }
      } else {
        evbuffer_add_buffer(out, success_buffers_[i]);
        evbuffer_free(success_buffers_[i]);
        success_buffers_[i] = nullptr;
      }
    }

    evbuffer_add(out, "]}", 2);

    AddContentTypeHeader(req_, "application/json");
    evbuffer_add_buffer(req_->buffer_out, out);
    evbuffer_free(out);
    evhtp_send_reply(req_, EVHTP_RES_OK);
    evhtp_request_resume(req_);
  }

  evhtp_request_t* req_;
  const size_t n_;
  evthr_t* reply_thread_;
  std::vector<std::shared_ptr<TRITONSERVER_InferenceRequest>> irequests_;

  std::atomic<size_t> done_count_{0};
  std::vector<evbuffer*> success_buffers_;
  std::vector<std::string> error_text_;
  std::vector<char> have_error_;
  std::atomic<bool> cancel_sent_{false};
  std::atomic<bool> reply_scheduled_{false};
#ifdef TRITON_ENABLE_MYSQL_ODBC
  std::vector<std::vector<ImpRouteRow>> imp_routing_slots_;
  std::vector<std::string> slot_model_names_;
  std::vector<std::vector<float>> slot_row_outputs_;
  int imp_routing_imp_count_{0};
#endif
};

class MultiInferShardRequest : public HTTPAPIServer::InferRequestClass {
 public:
  MultiInferShardRequest(TRITONSERVER_Server* server, evhtp_request_t* req,
      DataCompressor::Type response_compression_type,
      const std::shared_ptr<TRITONSERVER_InferenceRequest>& triton_request,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      std::shared_ptr<MultiInferAggregator> aggregator, const size_t slot)
      : HTTPAPIServer::InferRequestClass(server, req, response_compression_type, triton_request, shm_manager, false /* pause */, false /* fini hook */), aggregator_(std::move(aggregator)), slot_(slot){}

  static void InferResponseComplete(TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp) {
    auto* infer_request = reinterpret_cast<MultiInferShardRequest*>(userp);

    if (response != nullptr) {
      ++infer_request->response_count_;
    }

    TRITONSERVER_Error* err = nullptr;
    evbuffer* shard_json = nullptr;
    std::vector<float> pre_parsed_scalars;
    if (infer_request->response_count_ != 1) {
      const std::string msg = std::string("expected a single response, got ") + std::to_string(infer_request->response_count_);
      err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, msg.c_str());
    } else if (response != nullptr) {
      bool skip_shard_json = false;
#ifdef TRITON_ENABLE_MYSQL_ODBC
      if (infer_request->aggregator_->WantsShardParsedRows()) {
        const size_t nrows = infer_request->aggregator_->ExpectedRowsForSlot(infer_request->slot_);
        if (nrows > 0u) {
          TRITONSERVER_Error* ex_err = infer_request->ExtractFirstJsonOutputAsScalars(response, nrows, &pre_parsed_scalars);
          if (ex_err == nullptr) {
            skip_shard_json = true;
          } else {
            err = ex_err;
          }
        }
      }
#endif
      if (!skip_shard_json) {
        shard_json = evbuffer_new();
        err = infer_request->FinalizeResponse(response, shard_json);
      }
#ifdef TRITON_ENABLE_TRACING
      if (infer_request->trace_ != nullptr) {
        infer_request->trace_->CaptureTimestamp("INFER_RESPONSE_COMPLETE", TraceManager::CaptureTimestamp());
      }
#endif  // TRITON_ENABLE_TRACING
    }

    LOG_TRITONSERVER_ERROR(TRITONSERVER_InferenceResponseDelete(response), "deleting inference response");

    if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) == 0) {
      if (shard_json != nullptr) {
        evbuffer_free(shard_json);
      }
      return;
    }

    if (err != nullptr) {
      if (shard_json != nullptr) {
        evbuffer_free(shard_json);
      }
      infer_request->aggregator_->OnShardDone(infer_request->slot_, err, nullptr);
    } else {
      infer_request->aggregator_->OnShardDone(infer_request->slot_, nullptr, shard_json, std::move(pre_parsed_scalars));
    }
    evthr_defer(infer_request->thread_, DeleteMultiInferShardRequestThunk, infer_request);
  }

 private:
  static void DeleteMultiInferShardRequestThunk(evthr_t* /*thr*/, void* arg, void* /*shared*/) {
    delete reinterpret_cast<MultiInferShardRequest*>(arg);
  }

  std::shared_ptr<MultiInferAggregator> aggregator_;
  const size_t slot_;
};

struct PostBodyContent {
  evbuffer* decompressed_buffer{nullptr};
  TRITONSERVER_Error* read_error{nullptr};
  const char* data{""};
  size_t size{0};
};

PostBodyContent ReadPostBody(evhtp_request_t* req, evbuffer* decompressed_buffer)
{
  PostBodyContent out;
  out.decompressed_buffer = decompressed_buffer;
  evbuffer* body_buf = (out.decompressed_buffer != nullptr) ? out.decompressed_buffer : req->buffer_in;
  out.size = evbuffer_get_length(body_buf);
  if (out.size > 0) {
    const unsigned char* pulled = evbuffer_pullup(body_buf, -1);
    if (pulled == nullptr) {
      out.read_error = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "failed to read request body");
    } else {
      out.data = reinterpret_cast<const char*>(pulled);
    }
  }
  return out;
}

void FreePostBody(PostBodyContent* body)
{
  if (body->decompressed_buffer != nullptr) {
    evbuffer_free(body->decompressed_buffer);
    body->decompressed_buffer = nullptr;
  }
}

void RespondWithTritonError(evhtp_request_t* req, TRITONSERVER_Error* err)
{
  AddContentTypeHeader(req, "application/json");
  EVBufferAddErrorJson(req->buffer_out, err);
  evhtp_send_reply(req, HttpCodeFromError(err));
  TRITONSERVER_ErrorDelete(err);
  evhtp_request_resume(req);
}

}  // namespace

#ifdef TRITON_ENABLE_MYSQL_ODBC
void HTTPAPIServer::HandlePredict(evhtp_request_t* req) {
  RETURN_AND_RESPOND_IF_RESTRICTED(req, RestrictedCategory::INFERENCE, restricted_apis_);

  if (req->method != htp_method_POST) {
    RETURN_AND_RESPOND_WITH_ERR(req, EVHTP_RES_METHNALLOWED, "Method Not Allowed");
  }

  evhtp_request_pause(req);

  evbuffer* decompressed_buffer = nullptr;
  TRITONSERVER_Error* read_error = DecompressBuffer(req, &decompressed_buffer);
  if (read_error != nullptr) {
    RespondWithTritonError(req, read_error);
    return;
  }

  PostBodyContent body = ReadPostBody(req, decompressed_buffer);
  if (body.read_error != nullptr) {
    RespondWithTritonError(req, body.read_error);
    FreePostBody(&body);
    return;
  }

  if (body.size == 0) {
    RespondWithTritonError(req, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "expected object with imps array"));
    FreePostBody(&body);
    return;
  }

  rapidjson::Document imps_doc;
  imps_doc.Parse(body.data, body.size);
  FreePostBody(&body);

  if (imps_doc.HasParseError() || !imps_doc.IsObject() || !imps_doc.HasMember("imps") || !imps_doc["imps"].IsArray()) {
    RespondWithTritonError(req, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "expected object with imps array"));
    return;
  }

  ImpRoutingTable imp_routing;
  std::vector<ImpsInferSlot> imps_slots;
  TRITONSERVER_Error* err = GenerateImpsInferSlots(imps_doc, server_.get(), &imps_slots, &imp_routing);
  if (err != nullptr) {
    RespondWithTritonError(req, err);
    return;
  }

  const size_t n = imps_slots.size();
  if (n == 0) {
    RespondWithTritonError(req, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "imps request produced no inference sub-requests"));
    return;
  }

  std::unordered_set<std::string> policy_checked_models;
  for (size_t i = 0; i < n; ++i) {
    if (!policy_checked_models.insert(imps_slots[i].model_name).second) {
      continue;
    }
    err = CheckTransactionPolicy(req, imps_slots[i].model_name, imps_slots[i].model_version);
    if (err != nullptr) {
      RespondWithTritonError(req, err);
      return;
    }
  }

  evthr_t* reply_thread = evhtp_request_get_connection(req)->thread;
  std::vector<std::shared_ptr<TRITONSERVER_InferenceRequest>> irequests;
  irequests.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    TRITONSERVER_InferenceRequest* ireq = nullptr;
    err = TRITONSERVER_InferenceRequestNew(&ireq, server_.get(), imps_slots[i].model_name.c_str(), imps_slots[i].model_version);
    if (err != nullptr) {
      for (auto& ir : irequests) {
        if (ir != nullptr) {
          LOG_TRITONSERVER_ERROR(TRITONSERVER_InferenceRequestDelete(ir.get()), "deleting unused predict sub-request");
        }
      }
      RespondWithTritonError(req, err);
      return;
    }
    irequests.emplace_back(ireq, [](TRITONSERVER_InferenceRequest* r) {
      LOG_TRITONSERVER_ERROR(TRITONSERVER_InferenceRequestDelete(r), "deleting HTTP predict sub-request");
    });
  }

  std::vector<std::string> slot_model_names;
  slot_model_names.reserve(n);
  for (const auto& slot : imps_slots) {
    slot_model_names.push_back(slot.original_model_name.empty() ? slot.model_name : slot.original_model_name);
  }

  std::shared_ptr<MultiInferAggregator> aggregator = std::make_shared<MultiInferAggregator>(req, n, reply_thread, irequests, std::move(imp_routing.slots), std::move(slot_model_names), imp_routing.imp_count);
  std::vector<std::unique_ptr<MultiInferShardRequest>> shard_holders;
  std::vector<std::unique_ptr<HTTPAPIServer::RequestReleasePayload>> release_holders;
  shard_holders.reserve(n);
  release_holders.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    auto shard = std::make_unique<MultiInferShardRequest>(server_.get(), req, GetResponseCompressionType(req), irequests[i], shm_manager_, aggregator, i);

    err = FillImpsTritonRequest(irequests[i].get(), shard.get(), std::move(imps_slots[i]));
    if (err != nullptr) {
      aggregator->CancelAllSubRequests();
      RespondWithTritonError(req, err);
      return;
    }

    auto rel = std::make_unique<HTTPAPIServer::RequestReleasePayload>(irequests[i], nullptr /* body buffer */);
    err = ScheduleInferAsync(req, irequests[i].get(), shard.get(), rel.get(), nullptr, MultiInferShardRequest::InferResponseComplete);
    if (err != nullptr) {
      aggregator->CancelAllSubRequests();
      RespondWithTritonError(req, err);
      return;
    }

    shard_holders.push_back(std::move(shard));
    release_holders.push_back(std::move(rel));
  }

  for (size_t i = 0; i < n; ++i) {
    release_holders[i].release();
    shard_holders[i].release();
  }
}
#endif  // TRITON_ENABLE_MYSQL_ODBC

void HTTPAPIServer::HandleMultiInfer(evhtp_request_t* req) {
  RETURN_AND_RESPOND_IF_RESTRICTED(req, RestrictedCategory::INFERENCE, restricted_apis_);

  if (req->method != htp_method_POST) {
    RETURN_AND_RESPOND_WITH_ERR(req, EVHTP_RES_METHNALLOWED, "Method Not Allowed");
  }

  evhtp_request_pause(req);

  evbuffer* decompressed_buffer = nullptr;
  TRITONSERVER_Error* read_error = DecompressBuffer(req, &decompressed_buffer);
  if (read_error != nullptr) {
    RespondWithTritonError(req, read_error);
    return;
  }

  PostBodyContent body = ReadPostBody(req, decompressed_buffer);
  if (body.read_error != nullptr) {
    RespondWithTritonError(req, body.read_error);
    FreePostBody(&body);
    return;
  }

  triton::common::TritonJson::Value root;
  TRITONSERVER_Error* err = root.Parse(body.data, body.size);
  FreePostBody(&body);
  if (err != nullptr) {
    RespondWithTritonError(req, err);
    return;
  }

  triton::common::TritonJson::Value requests;
  if (!root.Find("requests", &requests)) {
    err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "Request body must include a JSON array field 'requests'");
    AddContentTypeHeader(req, "application/json");
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, HttpCodeFromError(err));
    TRITONSERVER_ErrorDelete(err);
    evhtp_request_resume(req);
    return;
  }

  const size_t n = requests.ArraySize();
  if (n == 0) {
    err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "'requests' array must be non-empty");
    AddContentTypeHeader(req, "application/json");
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, HttpCodeFromError(err));
    TRITONSERVER_ErrorDelete(err);
    evhtp_request_resume(req);
    return;
  }

  struct SlotPrep {
    std::string model_name;
    int64_t model_version{0};
    triton::common::TritonJson::Value infer_json;
  };
  std::vector<SlotPrep> slots;
  slots.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    triton::common::TritonJson::Value slot;
    err = requests.At(i, &slot);
    if (err != nullptr) {
      AddContentTypeHeader(req, "application/json");
      EVBufferAddErrorJson(req->buffer_out, err);
      evhtp_send_reply(req, HttpCodeFromError(err));
      TRITONSERVER_ErrorDelete(err);
      evhtp_request_resume(req);
      return;
    }
    const char* mn_c;
    size_t mn_len;
    err = slot.MemberAsString("model_name", &mn_c, &mn_len);
    if (err != nullptr) {
      AddContentTypeHeader(req, "application/json");
      EVBufferAddErrorJson(req->buffer_out, err);
      evhtp_send_reply(req, HttpCodeFromError(err));
      TRITONSERVER_ErrorDelete(err);
      evhtp_request_resume(req);
      return;
    }
    SlotPrep prep;
    prep.model_name.assign(mn_c, mn_len);
    std::string ver_str;
    err = GetModelVersionStringFromSlot(slot, &ver_str);
    if (err != nullptr) {
      AddContentTypeHeader(req, "application/json");
      EVBufferAddErrorJson(req->buffer_out, err);
      evhtp_send_reply(req, HttpCodeFromError(err));
      TRITONSERVER_ErrorDelete(err);
      evhtp_request_resume(req);
      return;
    }
    err = GetModelVersionFromString(ver_str, &prep.model_version);
    if (err != nullptr) {
      AddContentTypeHeader(req, "application/json");
      EVBufferAddErrorJson(req->buffer_out, err);
      evhtp_send_reply(req, HttpCodeFromError(err));
      TRITONSERVER_ErrorDelete(err);
      evhtp_request_resume(req);
      return;
    }
    err = CheckTransactionPolicy(req, prep.model_name, prep.model_version);
    if (err != nullptr) {
      AddContentTypeHeader(req, "application/json");
      EVBufferAddErrorJson(req->buffer_out, err);
      evhtp_send_reply(req, HttpCodeFromError(err));
      TRITONSERVER_ErrorDelete(err);
      evhtp_request_resume(req);
      return;
    }
    triton::common::TritonJson::Value infer_only;
    err = CopyInferSlotBodyJson(slot, &infer_only);
    if (err != nullptr) {
      AddContentTypeHeader(req, "application/json");
      EVBufferAddErrorJson(req->buffer_out, err);
      evhtp_send_reply(req, HttpCodeFromError(err));
      TRITONSERVER_ErrorDelete(err);
      evhtp_request_resume(req);
      return;
    }
    prep.infer_json = std::move(infer_only);
    slots.push_back(std::move(prep));
  }

  evthr_t* reply_thread = evhtp_request_get_connection(req)->thread;
  std::vector<std::shared_ptr<TRITONSERVER_InferenceRequest>> irequests;
  irequests.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    TRITONSERVER_InferenceRequest* ireq = nullptr;
    err = TRITONSERVER_InferenceRequestNew(&ireq, server_.get(), slots[i].model_name.c_str(), slots[i].model_version);
    if (err != nullptr) {
      for (auto& ir : irequests) {
        if (ir != nullptr) {
          LOG_TRITONSERVER_ERROR(TRITONSERVER_InferenceRequestDelete(ir.get()), "deleting unused multi_infer sub-request");
        }
      }
      AddContentTypeHeader(req, "application/json");
      EVBufferAddErrorJson(req->buffer_out, err);
      evhtp_send_reply(req, HttpCodeFromError(err));
      TRITONSERVER_ErrorDelete(err);
      evhtp_request_resume(req);
      return;
    }
    irequests.emplace_back(ireq, [](TRITONSERVER_InferenceRequest* r) {
      LOG_TRITONSERVER_ERROR(TRITONSERVER_InferenceRequestDelete(r),"deleting HTTP multi_infer sub-request");
    });
  }

  std::shared_ptr<MultiInferAggregator> aggregator = std::make_shared<MultiInferAggregator>(req, n, reply_thread, irequests);
  std::vector<std::unique_ptr<MultiInferShardRequest>> shard_holders;
  std::vector<std::unique_ptr<HTTPAPIServer::RequestReleasePayload>> release_holders;
  shard_holders.reserve(n);
  release_holders.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    auto shard = std::make_unique<MultiInferShardRequest>(server_.get(), req, GetResponseCompressionType(req), irequests[i], shm_manager_, aggregator, i);

    err = FillMultiInferSlotTritonRequest(slots[i].model_name, slots[i].infer_json, irequests[i].get(), shard.get());
    if (err != nullptr) {
      aggregator->CancelAllSubRequests();
      AddContentTypeHeader(req, "application/json");
      EVBufferAddErrorJson(req->buffer_out, err);
      evhtp_send_reply(req, HttpCodeFromError(err));
      TRITONSERVER_ErrorDelete(err);
      evhtp_request_resume(req);
      return;
    }

    auto rel = std::make_unique<HTTPAPIServer::RequestReleasePayload>(irequests[i], nullptr /* body buffer */);
    err = ScheduleInferAsync(req, irequests[i].get(), shard.get(), rel.get(), nullptr, MultiInferShardRequest::InferResponseComplete);
    if (err != nullptr) {
      aggregator->CancelAllSubRequests();
      AddContentTypeHeader(req, "application/json");
      EVBufferAddErrorJson(req->buffer_out, err);
      evhtp_send_reply(req, HttpCodeFromError(err));
      TRITONSERVER_ErrorDelete(err);
      evhtp_request_resume(req);
      return;
    }

    shard_holders.push_back(std::move(shard));
    release_holders.push_back(std::move(rel));
  }

  for (size_t i = 0; i < n; ++i) {
    release_holders[i].release();
    shard_holders[i].release();
  }
}

}}  // namespace triton::server
