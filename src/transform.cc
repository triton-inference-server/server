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

#ifdef TRITON_ENABLE_MYSQL_ODBC
#include "transform.h"
#include "mysql_odbc_connection_pool.h"
#include <rapidjson/error/en.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <rapidjson/document.h>
#include <vector>

namespace {

int FeatureIdxFromJsonValue(const char* feature_name, const rapidjson::Value& v, const triton::server::FeatureMappingTables* tables) {
  if (tables == nullptr) {
    return -1;
  }
  if (v.IsString()) {
    return triton::server::GetFeatureMappingIdx(feature_name, v.GetString(), tables);
  }
  char num_buf[48];
  int n = 0;
  if (v.IsInt()) {
    n = std::snprintf(num_buf, sizeof(num_buf), "%d", v.GetInt());
  } else if (v.IsUint()) {
    n = std::snprintf(num_buf, sizeof(num_buf), "%u", v.GetUint());
  } else if (v.IsInt64()) {
    n = std::snprintf(num_buf, sizeof(num_buf), "%lld", static_cast<long long>(v.GetInt64()));
  } else if (v.IsUint64()) {
    n = std::snprintf(num_buf, sizeof(num_buf), "%llu", static_cast<unsigned long long>(v.GetUint64()));
  } else {
    return -1;
  }
  if (n <= 0 || static_cast<size_t>(n) >= sizeof(num_buf)) {
    return -1;
  }
  return triton::server::GetFeatureMappingIdx(feature_name, num_buf, tables);
}
}

namespace triton { namespace server {

TRITONSERVER_Error* ParseRequest(const char* json, size_t json_len, TRITONSERVER_Server* server, rapidjson::Document* out_doc) {
  if (out_doc == nullptr) {
    return TRITONSERVER_ErrorNew(
    TRITONSERVER_ERROR_INVALID_ARG, "output document pointer is null");
  }

  rapidjson::Document doc;
  doc.Parse(json, json_len);
  if (doc.HasParseError()) {
    const char* msg = rapidjson::GetParseError_En(doc.GetParseError());
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, msg);
  }

  if (server != nullptr && doc.IsObject() && doc.HasMember(TRITON_BT_JSON_IMPS)) {
    const rapidjson::Value& imps_member = doc[TRITON_BT_JSON_IMPS];
    if (imps_member.IsArray()) {
      return GenerateInputVectors(doc, server, out_doc);
    }
  }

  *out_doc = std::move(doc);
  return nullptr;
}

TRITONSERVER_Error* GetReadyModelNames(TRITONSERVER_Server* server, std::unordered_set<std::string>* out) {
  if (out == nullptr) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "output set pointer is null");
  }
  out->clear();

  TRITONSERVER_Message* message = nullptr;
  TRITONSERVER_Error* err = TRITONSERVER_ServerModelIndex(server, TRITONSERVER_INDEX_FLAG_READY, &message);
  if (err != nullptr) {
    return err;
  }

  const char* buffer = nullptr;
  size_t byte_size = 0;
  err = TRITONSERVER_MessageSerializeToJson(message, &buffer, &byte_size);
  if (err != nullptr) {
    TRITONSERVER_MessageDelete(message);
    return err;
  }
  const std::string index_json(buffer, byte_size);
  TRITONSERVER_MessageDelete(message);

  rapidjson::Document index_doc;
  index_doc.Parse(index_json.data(), index_json.size());
  if (index_doc.HasParseError()) {
    const std::string parse_err = std::string("failed to parse model index JSON: ") + rapidjson::GetParseError_En(index_doc.GetParseError()) + " at offset " + std::to_string(index_doc.GetErrorOffset());
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, parse_err.c_str());
  }
  if (!index_doc.IsArray()) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "model index JSON root is not an array");
  }

  out->reserve(static_cast<size_t>(index_doc.Size()));
  for (rapidjson::SizeType i = 0; i < index_doc.Size(); ++i) {
    const rapidjson::Value& o = index_doc[i];
    if (!o.IsObject() || !o.HasMember("name")) {
      continue;
    }
    const rapidjson::Value& n = o["name"];
    if (!n.IsString()) {
      continue;
    }
    out->emplace(n.GetString());
  }

  return nullptr;
}

TRITONSERVER_Error* AppendRowToNamedDoubleBuffers(NamedDoubleBuffers* buffers, ModelNameToFeatureCount* feature_counts,
  const std::string& vector_name, const std::vector<double>& row, size_t feature_count) {
  if (buffers == nullptr || feature_counts == nullptr) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "buffer pointers are null");
  }
  if (row.size() != feature_count || feature_count == 0) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "row width does not match feature_count");
  }

  auto fc_it = feature_counts->find(vector_name);
  if (fc_it == feature_counts->end()) {
    (*feature_counts)[vector_name] = feature_count;
  } else if (fc_it->second != feature_count) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "inconsistent feature_count for model buffer");
  }

  std::vector<double>& buf = (*buffers)[vector_name];
  buf.insert(buf.end(), row.begin(), row.end());
  return nullptr;
}

TRITONSERVER_Error* BuildMultiInferRequestDocument(const NamedDoubleBuffers& buffers, const ModelNameToFeatureCount& feature_counts, rapidjson::Document* out_doc) {
  if (out_doc == nullptr) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "output document pointer is null");
  }

  std::vector<std::string> names;
  names.reserve(buffers.size());
  for (const auto& kv : buffers) {
    names.push_back(kv.first);
  }
  std::sort(names.begin(), names.end());

  rapidjson::Document doc(rapidjson::kObjectType);
  auto& alloc = doc.GetAllocator();
  rapidjson::Value requests(rapidjson::kArrayType);

  for (const std::string& model_name : names) {
    auto bc_it = buffers.find(model_name);
    auto fc_it = feature_counts.find(model_name);
    if (bc_it == buffers.end() || fc_it == feature_counts.end()) {
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "internal buffer map inconsistency");
    }
    const size_t feature_count = fc_it->second;
    if (feature_count == 0) {
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "zero feature_count for model");
    }
    const std::vector<double>& flat = bc_it->second;
    if (flat.size() % feature_count != 0 || flat.empty()) {
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "buffer length is not a multiple of feature_count");
    }
    const size_t rows = flat.size() / feature_count;

    rapidjson::Value req(rapidjson::kObjectType);
    req.AddMember("model_name", rapidjson::Value(model_name.c_str(), static_cast<rapidjson::SizeType>(model_name.size()), alloc).Move(), alloc);

    rapidjson::Value data(rapidjson::kArrayType);
    data.Reserve(static_cast<rapidjson::SizeType>(flat.size()), alloc);
    for (double v : flat) {
      data.PushBack(v, alloc);
    }

    rapidjson::Value shape(rapidjson::kArrayType);
    shape.PushBack(static_cast<uint64_t>(rows), alloc);
    shape.PushBack(static_cast<uint64_t>(feature_count), alloc);

    rapidjson::Value input0(rapidjson::kObjectType);
    input0.AddMember("name", "input__0", alloc);
    input0.AddMember("datatype", "FP32", alloc);
    input0.AddMember("shape", shape, alloc);
    input0.AddMember("data", data, alloc);

    rapidjson::Value inputs(rapidjson::kArrayType);
    inputs.PushBack(input0, alloc);

    rapidjson::Value outputs(rapidjson::kArrayType);
    rapidjson::Value out0(rapidjson::kObjectType);
    out0.AddMember("name", "output__0", alloc);
    outputs.PushBack(out0, alloc);

    req.AddMember("inputs", inputs, alloc);
    req.AddMember("outputs", outputs, alloc);
    requests.PushBack(req, alloc);
  }

  doc.AddMember("requests", requests, alloc);
  *out_doc = std::move(doc);
  return nullptr;
}

namespace {

struct ImpRouteRow {
  int imp_idx{0};
  int camp_idx{0};
  int adsize_idx{0};
  int32_t cid{0};
};

void AddImpSlotRoutingMembers(const NamedDoubleBuffers& buffers, const std::unordered_map<std::string, std::vector<ImpRouteRow>>& routes_by_model, int imp_count, rapidjson::Document* out_doc) {
  std::vector<std::string> names;
  names.reserve(buffers.size());
  for (const auto& kv : buffers) {
    names.push_back(kv.first);
  }
  std::sort(names.begin(), names.end());

  auto& alloc = out_doc->GetAllocator();
  rapidjson::Value routing(rapidjson::kArrayType);
  routing.Reserve(static_cast<rapidjson::SizeType>(names.size()), alloc);
  for (const std::string& mn : names) {
    rapidjson::Value slot(rapidjson::kArrayType);
    auto it = routes_by_model.find(mn);
    if (it != routes_by_model.end()) {
      slot.Reserve(static_cast<rapidjson::SizeType>(it->second.size()), alloc);
      for (const ImpRouteRow& r : it->second) {
        rapidjson::Value o(rapidjson::kObjectType);
        o.AddMember("i", r.imp_idx, alloc);
        o.AddMember("c", r.camp_idx, alloc);
        o.AddMember("a", r.adsize_idx, alloc);
        o.AddMember(TRITON_BT_JSON_CID, r.cid, alloc);
        o.AddMember("mdl", rapidjson::Value(mn.c_str(), static_cast<rapidjson::SizeType>(mn.size()), alloc).Move(), alloc);
        slot.PushBack(o, alloc);
      }
    }
    routing.PushBack(slot, alloc);
  }
  out_doc->AddMember("imp_slot_routing", routing, alloc);
  out_doc->AddMember("imp_routing_imp_count", imp_count, alloc);
}

}

TRITONSERVER_Error* GenerateInputVectors(const rapidjson::Document& doc, TRITONSERVER_Server* server, rapidjson::Document* out_doc) {
  if (out_doc == nullptr || server == nullptr) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "invalid argument");
  }
  if (!doc.IsObject() || !doc.HasMember(TRITON_BT_JSON_IMPS) || !doc[TRITON_BT_JSON_IMPS].IsArray()) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "expected object with imps array");
  }

  const CampaignToFeatureMappings* cmap = ActiveCampaignToFeatureMappings();
  if (cmap == nullptr || cmap->empty()) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNAVAILABLE, "campaign feature mappings are not loaded");
  }

  std::unordered_set<std::string> ready_model_names;
  TRITONSERVER_Error* err = GetReadyModelNames(server, &ready_model_names);
  if (err != nullptr) {
    return err;
  }
  if (ready_model_names.empty()) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "no ready models reported by server");
  }

  NamedDoubleBuffers buffers;
  ModelNameToFeatureCount counts;
  std::unordered_map<std::string, std::vector<ImpRouteRow>> routes_by_model;

  const rapidjson::Value& imps = doc[TRITON_BT_JSON_IMPS];
  for (rapidjson::SizeType ii = 0; ii < imps.Size(); ++ii) {
    const rapidjson::Value& imp = imps[ii];
    if (!imp.IsObject()) {
      continue;
    }
    if (!imp.HasMember(TRITON_BT_JSON_CAMPS) || !imp[TRITON_BT_JSON_CAMPS].IsArray() || imp[TRITON_BT_JSON_CAMPS].Size() == 0) {
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "each impression must include a non-empty camps array");
    }
    const rapidjson::Value& camps = imp[TRITON_BT_JSON_CAMPS];
    for (rapidjson::SizeType ci = 0; ci < camps.Size(); ++ci) {
      const rapidjson::Value& camp = camps[ci];
      if (!camp.IsObject() || !camp.HasMember(TRITON_BT_JSON_CID) || !camp[TRITON_BT_JSON_CID].IsInt()) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "each camp must be an object with integer 'cid'");
      }
      const int32_t campaign_id = camp[TRITON_BT_JSON_CID].GetInt();

      auto cmap_it = cmap->find(campaign_id);
      if(cmap_it == cmap->end()) {
        cmap_it = cmap->find(0);
      }
      if (cmap_it == cmap->end()) {
        const std::string unknown_campaign = std::string("unknown campaign_id ") + std::to_string(campaign_id);
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, unknown_campaign.c_str());
      }
      const std::string& model_name = cmap_it->second.model_name;
      const std::vector<std::string>& feature_sequence = cmap_it->second.feature_sequence;
      const FeatureMappingTables& tables = cmap_it->second.feature_mapping;

      if (ready_model_names.find(model_name) == ready_model_names.end()) {
        const std::string not_ready = "model " + model_name + " not ready";
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, not_ready.c_str());
      }

      std::vector<double> row(feature_sequence.size(), 0.0);
      int adsize_idx = -1;

      for (size_t fi = 0; fi < feature_sequence.size(); ++fi) {
        const std::string& feature = feature_sequence[fi];
        const char* fkey = feature.c_str();

        if (feature == TRITON_BT_FEATURE_ADSIZE) {
          adsize_idx = static_cast<int>(fi);
          continue;
        }

        const rapidjson::Value* src = nullptr;
        if (feature == TRITON_BT_FEATURE_COOKIE || feature == TRITON_BT_FEATURE_RNK) {
          if (camp.HasMember(fkey)) {
            src = &camp[fkey];
          }
        }
        else if (feature == TRITON_BT_FEATURE_CAMPID) {
          if (camp.HasMember(TRITON_BT_JSON_CID)) {
            src = &camp[TRITON_BT_JSON_CID];
          }
        }
        else {
          if (imp.HasMember(fkey)) {
            src = &imp[fkey];
          }
        }

        if (src == nullptr) {
          const std::string missing_field = std::string("missing JSON field for feature '") + feature + "'";
          return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, missing_field.c_str());
        }

        const bool use_raw_numeric = (feature == TRITON_BT_FEATURE_UID) ||
            (feature == TRITON_BT_FEATURE_VIDEO_VPW) || (feature == TRITON_BT_FEATURE_VIDEO_VPH) ||
            (feature == TRITON_BT_FEATURE_MOBILEID) || (feature == TRITON_BT_FEATURE_VIEW) ||
            (feature == TRITON_BT_FEATURE_COOKIE) || (feature == TRITON_BT_FEATURE_RNK);

        if (use_raw_numeric) {
          const rapidjson::Value& v = *src;
          if (v.IsInt()) {
            row[fi] = static_cast<double>(v.GetInt());
          } else if (v.IsUint()) {
            row[fi] = static_cast<double>(v.GetUint());
          } else if (v.IsInt64()) {
            row[fi] = static_cast<double>(v.GetInt64());
          } else if (v.IsUint64()) {
            row[fi] = static_cast<double>(v.GetUint64());
          } else if (v.IsDouble()) {
            row[fi] = v.GetDouble();
          } else {
            const std::string msg = std::string("feature '") + feature + "' must be a JSON number for raw passthrough";
            return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, msg.c_str());
          }
        } else {
          const int idx = FeatureIdxFromJsonValue(feature.c_str(), *src, &tables);
          row[fi] = static_cast<double>(idx);
        }
      }
      if (adsize_idx >= 0 && camp.HasMember(TRITON_BT_FEATURE_ADSIZE) &&
          camp[TRITON_BT_FEATURE_ADSIZE].IsArray()) {
        const rapidjson::Value& adsize = camp[TRITON_BT_FEATURE_ADSIZE];
        for (rapidjson::SizeType ai = 0; ai < adsize.Size(); ++ai) {
          const rapidjson::Value& adsize_item = adsize[ai];
          int mapped = FeatureIdxFromJsonValue(TRITON_BT_FEATURE_ADSIZE, adsize_item, &tables);
          row[static_cast<size_t>(adsize_idx)] = static_cast<double>(mapped);
          err = AppendRowToNamedDoubleBuffers(&buffers, &counts, model_name, row, feature_sequence.size());
          if (err != nullptr) {
            return err;
          }
          routes_by_model[model_name].push_back(ImpRouteRow{
          static_cast<int>(ii), static_cast<int>(ci), static_cast<int>(ai), campaign_id});
        }
      }
    }
  }
  err = BuildMultiInferRequestDocument(buffers, counts, out_doc);
  if (err != nullptr) {
    return err;
  }
  AddImpSlotRoutingMembers(buffers, routes_by_model, static_cast<int>(imps.Size()), out_doc);
  return nullptr;
}
} }  // namespace triton::server
#endif  // TRITON_ENABLE_MYSQL_ODBC
