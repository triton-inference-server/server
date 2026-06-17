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

// POST /v2/multi_infer: full implementation (imps expansion, routing, and
// response folding) is compiled only when TRITON_ENABLE_MYSQL_ODBC is defined.
// Otherwise HandleMultiInfer responds with TRITONSERVER_ERROR_UNAVAILABLE.

#include "http_server.h"

#include "common.h"

#ifdef TRITON_ENABLE_MYSQL_ODBC
#include "classification.h"
#include "transform.h"

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace triton { namespace server {

#include "http_server_macros.h"

#ifdef TRITON_ENABLE_MYSQL_ODBC

namespace {

constexpr size_t kMaxMultiInferRequests = 16;

// Per batched infer row when folding multi_infer back to imps/camps (see
// imp_slot_routing in transform.cc).
struct ImpRouteCell {
  int imp_idx{0};
  int camp_idx{0};
  int adsize_idx{0};
  int32_t cid{0};
  std::string mdl;
};

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

void EVBufferAddErrorJson(evbuffer* buffer, const char* message) {
  triton::common::TritonJson::Value response(triton::common::TritonJson::ValueType::OBJECT);
  response.AddStringRef("error", message, strlen(message));

  triton::common::TritonJson::WriteBuffer buffer_json;
  response.Write(&buffer_json);

  evbuffer_add(buffer, buffer_json.Base(), buffer_json.Size());
}

void EVBufferAddErrorJson(evbuffer* buffer, TRITONSERVER_Error* err) {
  const char* message = TRITONSERVER_ErrorMessage(err);
  EVBufferAddErrorJson(buffer, message);
}

void AddContentTypeHeader(evhtp_request_t* req, const char* type) {
  auto content_header = evhtp_headers_find_header(req->headers_out, kContentTypeHeader);
  if (content_header) {
    evhtp_header_rm_and_free(req->headers_out, content_header);
  }

  evhtp_headers_add_header(req->headers_out, evhtp_header_new(kContentTypeHeader, type, 1, 1));
}

inline double RoundScore6(double x)
{
  constexpr double k = 1e6;
  return std::floor(x * k) / k;
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

bool ParseImpSlotRoutingTable(const std::string& json, const size_t expect_slots, std::vector<std::vector<ImpRouteCell>>* out) {
  out->clear();
  if (json.empty() || expect_slots == 0) {
    return false;
  }
  rapidjson::Document d;
  d.Parse(json.data(), json.size());
  if (d.HasParseError() || !d.IsArray() || d.Size() != expect_slots) {
    return false;
  }
  out->resize(d.Size());
  for (rapidjson::SizeType si = 0; si < d.Size(); ++si) {
    const rapidjson::Value& slot = d[si];
    if (!slot.IsArray()) {
      return false;
    }
    (*out)[si].reserve(slot.Size());
    for (rapidjson::SizeType ri = 0; ri < slot.Size(); ++ri) {
      const rapidjson::Value& cell = slot[ri];
      if (!cell.IsObject() || !cell.HasMember("i") || !cell["i"].IsInt() ||
          !cell.HasMember("c") || !cell["c"].IsInt() || !cell.HasMember("a") ||
          !cell["a"].IsInt() || !cell.HasMember("cid") || !cell["cid"].IsInt() ||
          !cell.HasMember("mdl") || !cell["mdl"].IsString()) {
        return false;
      }
      ImpRouteCell rc;
      rc.imp_idx = cell["i"].GetInt();
      rc.camp_idx = cell["c"].GetInt();
      rc.adsize_idx = cell["a"].GetInt();
      rc.cid = static_cast<int32_t>(cell["cid"].GetInt());
      rc.mdl.assign(cell["mdl"].GetString(), cell["mdl"].GetStringLength());
      (*out)[si].push_back(std::move(rc));
    }
  }
  return true;
}

bool TryBuildImpsShapedMultiInferResponse(const std::vector<std::vector<ImpRouteCell>>& routing_slots, const std::vector<std::vector<std::vector<double>>>& slot_rows, const int imp_count, rapidjson::Document* out) {
  if (imp_count <= 0 || routing_slots.empty()) {
    return false;
  }
  if (slot_rows.size() != routing_slots.size()) {
    return false;
  }

  struct CampAgg {
    int32_t cid{0};
    std::string mdl;
    std::map<int, std::vector<double>> by_adsize;
  };
  std::map<std::pair<int, int>, CampAgg> agg;

  for (size_t si = 0; si < routing_slots.size(); ++si) {
    const auto& slot_r = routing_slots[si];
    const size_t R = slot_r.size();
    if (si >= slot_rows.size()) {
      return false;
    }
    const auto& row_vecs = slot_rows[si];
    if (row_vecs.size() != R) {
      return false;
    }
    for (size_t ri = 0; ri < R; ++ri) {
      const ImpRouteCell& rc = slot_r[ri];
      const auto key = std::make_pair(rc.imp_idx, rc.camp_idx);
      CampAgg& ca = agg[key];
      if (ca.mdl.empty()) {
        ca.cid = rc.cid;
        ca.mdl = rc.mdl;
      } else if (ca.cid != rc.cid || ca.mdl != rc.mdl) {
        return false;
      }
      ca.by_adsize[rc.adsize_idx] = row_vecs[ri];
    }
  }

  out->SetObject();
  auto& alloc = out->GetAllocator();
  rapidjson::Value imps_arr(rapidjson::kArrayType);
  imps_arr.Reserve(static_cast<rapidjson::SizeType>(imp_count), alloc);

  for (int ii = 0; ii < imp_count; ++ii) {
    std::vector<int> camp_indices;
    for (const auto& kv : agg) {
      if (kv.first.first == ii) {
        camp_indices.push_back(kv.first.second);
      }
    }
    std::sort(camp_indices.begin(), camp_indices.end());
    camp_indices.erase(std::unique(camp_indices.begin(), camp_indices.end()), camp_indices.end());

    rapidjson::Value camps_out(rapidjson::kArrayType);
    for (int camp_j : camp_indices) {
      const auto it = agg.find(std::make_pair(ii, camp_j));
      if (it == agg.end()) {
        continue;
      }
      const CampAgg& ca = it->second;
      rapidjson::Value camp_obj(rapidjson::kObjectType);
      camp_obj.AddMember("cid", ca.cid, alloc);
      camp_obj.AddMember("mdl", rapidjson::Value(ca.mdl.c_str(), static_cast<rapidjson::SizeType>(ca.mdl.size()), alloc).Move(),alloc);
      rapidjson::Value score_arr(rapidjson::kArrayType);
      for (const auto& ad_kv : ca.by_adsize) {
        const std::vector<double>& vec = ad_kv.second;
        if (vec.size() == 1) {
          score_arr.PushBack(RoundScore6(vec[0]), alloc);
        } else {
          rapidjson::Value inner(rapidjson::kArrayType);
          inner.Reserve(static_cast<rapidjson::SizeType>(vec.size()), alloc);
          for (double d : vec) {
            inner.PushBack(RoundScore6(d), alloc);
          }
          score_arr.PushBack(inner, alloc);
        }
      }
      camp_obj.AddMember("score", score_arr, alloc);
      camps_out.PushBack(camp_obj, alloc);
    }

    rapidjson::Value imp_wrap(rapidjson::kObjectType);
    imp_wrap.AddMember("camps", camps_out, alloc);
    imps_arr.PushBack(imp_wrap, alloc);
  }

  out->AddMember("imps", imps_arr, alloc);
  return true;
}

class MultiInferAggregator : public std::enable_shared_from_this<MultiInferAggregator> {
 private:
  struct FinishPayload {
    std::shared_ptr<MultiInferAggregator> agg;
  };

 public:
  MultiInferAggregator(evhtp_request_t* req, size_t slot_count, evthr_t* reply_thread,
      std::vector<std::shared_ptr<TRITONSERVER_InferenceRequest>> irequests,
      std::vector<std::vector<ImpRouteCell>> imp_routing_slots = {},
      int imp_routing_imp_count = 0)
      : req_(req), n_(slot_count), reply_thread_(reply_thread),
        irequests_(std::move(irequests)), success_json_(slot_count),
        error_text_(slot_count), have_error_(slot_count, 0),
        imp_routing_slots_(std::move(imp_routing_slots)),
        imp_routing_imp_count_(imp_routing_imp_count)
  {
    slot_row_outputs_.assign(n_, {});
  }

  std::shared_ptr<TRITONSERVER_InferenceRequest> IrequestAt(size_t i) const
  {
    return irequests_[i];
  }

  bool WantsShardParsedRows() const
  {
    return imp_routing_imp_count_ > 0 && !imp_routing_slots_.empty();
  }

  size_t ExpectedRowsForSlot(size_t slot) const
  {
    return (slot < imp_routing_slots_.size()) ? imp_routing_slots_[slot].size() : 0;
  }

  void CancelAllSubRequests()
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (cancel_sent_) {
      return;
    }
    cancel_sent_ = true;
    for (auto& ir : irequests_) {
      if (ir != nullptr) {
        LOG_TRITONSERVER_ERROR(TRITONSERVER_InferenceRequestCancel(ir.get()), "cancelling multi_infer sub-request");
      }
    }
  }

  void OnShardDone(size_t slot, TRITONSERVER_Error* finalize_err, const std::string& response_json, std::vector<std::vector<double>> parsed_first_output = {}) {
    std::shared_ptr<MultiInferAggregator> self;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (finalize_err != nullptr) {
        have_error_[slot] = 1;
        error_text_[slot] = TRITONSERVER_ErrorMessage(finalize_err);
        TRITONSERVER_ErrorDelete(finalize_err);
        if (!cancel_sent_) {
          cancel_sent_ = true;
          for (auto& ir : irequests_) {
            if (ir != nullptr) {
              LOG_TRITONSERVER_ERROR(TRITONSERVER_InferenceRequestCancel(ir.get()), "cancelling multi_infer sub-request");
            }
          }
        }
      } else {
        success_json_[slot] = response_json;
        if (!parsed_first_output.empty() && slot < slot_row_outputs_.size()) {
          slot_row_outputs_[slot] = std::move(parsed_first_output);
        }
      }
      done_count_++;
      if (done_count_ < n_ || reply_scheduled_) {
        return;
      }
      reply_scheduled_ = true;
      self = shared_from_this();
    }

    auto* fp = new FinishPayload{std::move(self)};
    evthr_defer(reply_thread_, FinishThunk, fp);
  }

 private:
  static void FinishThunk(evthr_t* /*thr*/, void* arg, void* /*shared*/) {
    std::unique_ptr<FinishPayload> fp(static_cast<FinishPayload*>(arg));
    fp->agg->WriteHttpReply();
  }

  void WriteHttpReply() {
    if (WantsShardParsedRows() && !cancel_sent_) {
      bool any_err = false;
      for (size_t i = 0; i < n_; ++i) {
        if (have_error_[i]) {
          any_err = true;
          break;
        }
      }
      if (!any_err) {
        rapidjson::Document shaped;
        if (TryBuildImpsShapedMultiInferResponse(imp_routing_slots_, slot_row_outputs_, imp_routing_imp_count_, &shaped)) {
          rapidjson::StringBuffer sb;
          rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
          shaped.Accept(writer);
          AddContentTypeHeader(req_, "application/json");
          evbuffer_add(req_->buffer_out, sb.GetString(), sb.GetSize());
          evhtp_send_reply(req_, EVHTP_RES_OK);
          evhtp_request_resume(req_);
          return;
        }
      }
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

    triton::common::TritonJson::Value root(triton::common::TritonJson::ValueType::OBJECT);
    triton::common::TritonJson::Value responses(root, triton::common::TritonJson::ValueType::ARRAY);
    for (size_t i = 0; i < n_; ++i) {
      triton::common::TritonJson::Value item;
      TRITONSERVER_Error* perr = item.Parse(success_json_[i].c_str(), success_json_[i].size());
      if (perr != nullptr) {
        triton::common::TritonJson::Value wrap(root, triton::common::TritonJson::ValueType::OBJECT);
        triton::common::TritonJson::Value err_part(root, triton::common::TritonJson::ValueType::OBJECT);
        TRITONSERVER_Error* ae = err_part.AddString("message", TRITONSERVER_ErrorMessage(perr));
        TRITONSERVER_ErrorDelete(perr);
        if (ae != nullptr) {
          TRITONSERVER_ErrorDelete(ae);
        }
        TRITONSERVER_Error* be = wrap.Add("error", std::move(err_part));
        if (be != nullptr) {
          TRITONSERVER_ErrorDelete(be);
        }
        TRITONSERVER_Error* ce = responses.Append(std::move(wrap));
        if (ce != nullptr) {
          TRITONSERVER_ErrorDelete(ce);
        }
      } else {
        TRITONSERVER_Error* ce = responses.Append(std::move(item));
        if (ce != nullptr) {
          LOG_TRITONSERVER_ERROR(ce, "multi_infer: appending response");
          TRITONSERVER_ErrorDelete(ce);
        }
      }
    }

    TRITONSERVER_Error* re = root.Add("errors", std::move(responses));
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
    } 
    else {
      AddContentTypeHeader(req_, "application/json");
      evbuffer_add(req_->buffer_out, wb.Base(), wb.Size());
      evhtp_send_reply(req_, EVHTP_RES_OK);
    }
    evhtp_request_resume(req_);
  }

  evhtp_request_t* req_;
  const size_t n_;
  evthr_t* reply_thread_;
  std::vector<std::shared_ptr<TRITONSERVER_InferenceRequest>> irequests_;

  std::mutex mu_;
  size_t done_count_{0};
  std::vector<std::string> success_json_;
  std::vector<std::string> error_text_;
  std::vector<char> have_error_;
  bool cancel_sent_{false};
  bool reply_scheduled_{false};
  std::vector<std::vector<ImpRouteCell>> imp_routing_slots_;
  std::vector<std::vector<std::vector<double>>> slot_row_outputs_;
  int imp_routing_imp_count_{0};
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
    evbuffer* shard_json = evbuffer_new();
    std::vector<std::vector<double>> pre_parsed_rows;
    if (infer_request->response_count_ != 1) {
      const std::string msg = std::string("expected a single response, got ") + std::to_string(infer_request->response_count_);
      err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, msg.c_str());
    } else if (response != nullptr) {
      if (infer_request->aggregator_->WantsShardParsedRows()) {
        const size_t nrows = infer_request->aggregator_->ExpectedRowsForSlot(infer_request->slot_);
        if (nrows > 0u) {
          TRITONSERVER_Error* ex_err = infer_request->ExtractFirstJsonOutputAsRowMajorDoubles(response, nrows, &pre_parsed_rows);
          if (ex_err != nullptr) {
            TRITONSERVER_ErrorDelete(ex_err);
            pre_parsed_rows.clear();
          }
        }
      }
      err = infer_request->FinalizeResponse(response, shard_json);
#ifdef TRITON_ENABLE_TRACING
      if (infer_request->trace_ != nullptr) {
        infer_request->trace_->CaptureTimestamp("INFER_RESPONSE_COMPLETE", TraceManager::CaptureTimestamp());
      }
#endif  // TRITON_ENABLE_TRACING
    }

    LOG_TRITONSERVER_ERROR(TRITONSERVER_InferenceResponseDelete(response), "deleting inference response");

    std::string json_fragment;
    if (err == nullptr) {
      const size_t len = evbuffer_get_length(shard_json);
      if (len > 0) {
        const unsigned char* p = evbuffer_pullup(shard_json, -1);
        if (p != nullptr) {
          json_fragment.assign(reinterpret_cast<const char*>(p), len);
        }
      }
    }
    evbuffer_free(shard_json);

    if (err != nullptr) {
      infer_request->aggregator_->OnShardDone(infer_request->slot_, err, "");
    } else {
      infer_request->aggregator_->OnShardDone(infer_request->slot_, nullptr, json_fragment, std::move(pre_parsed_rows));
    }

    if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) == 0) {
      return;
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

}  // namespace

void HTTPAPIServer::HandleMultiInfer(evhtp_request_t* req) {
  RETURN_AND_RESPOND_IF_RESTRICTED(req, RestrictedCategory::INFERENCE, restricted_apis_);

  if (req->method != htp_method_POST) {
    RETURN_AND_RESPOND_WITH_ERR(req, EVHTP_RES_METHNALLOWED, "Method Not Allowed");
  }

  evhtp_request_pause(req);

  evbuffer* decompressed_buffer = nullptr;
  TRITONSERVER_Error* derr = DecompressBuffer(req, &decompressed_buffer);
  if (derr != nullptr) {
    AddContentTypeHeader(req, "application/json");
    EVBufferAddErrorJson(req->buffer_out, derr);
    evhtp_send_reply(req, HttpCodeFromError(derr));
    TRITONSERVER_ErrorDelete(derr);
    evhtp_request_resume(req);
    return;
  }

  evbuffer* body_buf = (decompressed_buffer != nullptr) ? decompressed_buffer : req->buffer_in;

  triton::common::TritonJson::Value root;
  std::string imp_routing_meta;
  int imp_routing_imp_count = 0;
  TRITONSERVER_Error* err = nullptr;
  const size_t body_len = evbuffer_get_length(body_buf);
  std::vector<char> body_copy(body_len);
  if (body_len > 0) {
    const ssize_t nread = evbuffer_copyout(body_buf, body_copy.data(), body_len);
    if (nread < 0 || static_cast<size_t>(nread) != body_len) {
      err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "failed to read multi_infer request body");
    }
  }
  if (err == nullptr) {
    const std::string body_json(body_copy.data(), body_len);
    rapidjson::Document parsed;
    err = triton::server::ParseRequest(body_json, server_.get(), &parsed);
    if (err == nullptr) {
      if (parsed.IsObject()) {
        if (parsed.HasMember("imp_slot_routing") && parsed["imp_slot_routing"].IsArray()) {
          rapidjson::StringBuffer rbuf;
          rapidjson::Writer<rapidjson::StringBuffer> rw(rbuf);
          parsed["imp_slot_routing"].Accept(rw);
          imp_routing_meta.assign(rbuf.GetString(), rbuf.GetSize());
          parsed.RemoveMember("imp_slot_routing");
        }
        if (parsed.HasMember("imp_routing_imp_count")) {
          const rapidjson::Value& ic = parsed["imp_routing_imp_count"];
          if (ic.IsInt()) {
            imp_routing_imp_count = ic.GetInt();
          } else if (ic.IsUint()) {
            imp_routing_imp_count = static_cast<int>(ic.GetUint());
          }
          parsed.RemoveMember("imp_routing_imp_count");
        }
      }
      rapidjson::StringBuffer sb;
      rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
      parsed.Accept(writer);
      const std::string transformed(sb.GetString(), sb.GetSize());
      err = root.Parse(transformed.c_str(), transformed.size());
    }
  }
  if (decompressed_buffer != nullptr) {
    evbuffer_free(decompressed_buffer);
    decompressed_buffer = nullptr;
  }
  if (err != nullptr) {
    AddContentTypeHeader(req, "application/json");
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, HttpCodeFromError(err));
    TRITONSERVER_ErrorDelete(err);
    evhtp_request_resume(req);
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
  if (n > kMaxMultiInferRequests) {
    const std::string lim =  "At most " + std::to_string(kMaxMultiInferRequests) + " sub-requests are allowed per multi_infer call";
    err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, lim.c_str());
    AddContentTypeHeader(req, "application/json");
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, HttpCodeFromError(err));
    TRITONSERVER_ErrorDelete(err);
    evhtp_request_resume(req);
    return;
  }

  std::vector<std::vector<ImpRouteCell>> imp_routing_table;
  if (!imp_routing_meta.empty() && imp_routing_imp_count > 0) {
    if (!ParseImpSlotRoutingTable(imp_routing_meta, n, &imp_routing_table)) {
      imp_routing_table.clear();
      imp_routing_imp_count = 0;
    }
  } else {
    imp_routing_imp_count = 0;
  }

  struct SlotPrep {
    std::string model_name;
    int64_t model_version{0};
    std::string infer_body_json;
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
    triton::common::TritonJson::WriteBuffer wb;
    err = infer_only.Write(&wb);
    if (err != nullptr) {
      AddContentTypeHeader(req, "application/json");
      EVBufferAddErrorJson(req->buffer_out, err);
      evhtp_send_reply(req, HttpCodeFromError(err));
      TRITONSERVER_ErrorDelete(err);
      evhtp_request_resume(req);
      return;
    }
    prep.infer_body_json.assign(wb.Base(), wb.Size());
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

  std::shared_ptr<MultiInferAggregator> aggregator = std::make_shared<MultiInferAggregator>(req, n, reply_thread, irequests, std::move(imp_routing_table), imp_routing_imp_count);
  std::vector<std::unique_ptr<MultiInferShardRequest>> shard_holders;
  std::vector<std::unique_ptr<HTTPAPIServer::RequestReleasePayload>> release_holders;
  shard_holders.reserve(n);
  release_holders.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    evbuffer* body_i = evbuffer_new();
    evbuffer_add(body_i, slots[i].infer_body_json.data(), slots[i].infer_body_json.size());
    const int32_t content_length = static_cast<int32_t>(evbuffer_get_length(body_i));
    size_t header_length = 0;
    err = GetInferenceHeaderLength(req, content_length, &header_length);
    if (err != nullptr) {
      aggregator->CancelAllSubRequests();
      AddContentTypeHeader(req, "application/json");
      EVBufferAddErrorJson(req->buffer_out, err);
      evhtp_send_reply(req, HttpCodeFromError(err));
      TRITONSERVER_ErrorDelete(err);
      evhtp_request_resume(req);
      return;
    }

    auto shard = std::make_unique<MultiInferShardRequest>(server_.get(), req, GetResponseCompressionType(req), irequests[i], shm_manager_, aggregator, i);

    err = EVRequestToTritonRequest(req, slots[i].model_name, irequests[i].get(), body_i, shard.get(), header_length);
    if (err != nullptr) {
      aggregator->CancelAllSubRequests();
      evbuffer_free(body_i);
      AddContentTypeHeader(req, "application/json");
      EVBufferAddErrorJson(req->buffer_out, err);
      evhtp_send_reply(req, HttpCodeFromError(err));
      TRITONSERVER_ErrorDelete(err);
      evhtp_request_resume(req);
      return;
    }

    auto rel = std::make_unique<HTTPAPIServer::RequestReleasePayload>(irequests[i], body_i);
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

#else  // !TRITON_ENABLE_MYSQL_ODBC

void HTTPAPIServer::HandleMultiInfer(evhtp_request_t* req)
{
  RETURN_AND_RESPOND_IF_RESTRICTED(req, RestrictedCategory::INFERENCE, restricted_apis_);
  evhtp_request_pause(req);
  static const char kMsg[] = "POST /v2/multi_infer requires a server built with TRITON_ENABLE_MYSQL_ODBC";
  auto* content_header = evhtp_headers_find_header(req->headers_out, kContentTypeHeader);
  if (content_header != nullptr) {
    evhtp_header_rm_and_free(req->headers_out, content_header);
  }
  evhtp_headers_add_header(req->headers_out, evhtp_header_new(kContentTypeHeader, "application/json", 1, 1));
  triton::common::TritonJson::Value response(triton::common::TritonJson::ValueType::OBJECT);
  response.AddStringRef("error", kMsg, sizeof(kMsg) - 1);
  triton::common::TritonJson::WriteBuffer buffer_json;
  TRITONSERVER_Error* we = response.Write(&buffer_json);
  if (we != nullptr) {
    TRITONSERVER_ErrorDelete(we);
    evhtp_send_reply(req, EVHTP_RES_SERVERR);
  } else {
    evbuffer_add(req->buffer_out, buffer_json.Base(), buffer_json.Size());
    evhtp_send_reply(req, EVHTP_RES_SERVUNAVAIL);
  }
  evhtp_request_resume(req);
}

#endif  // TRITON_ENABLE_MYSQL_ODBC

}}  // namespace triton::server
