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
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <rapidjson/document.h>
#include <vector>

namespace {

std::unordered_set<std::string> g_ready_model_names;
std::atomic<bool> g_ready_models_valid{false};

int FeatureIdxFromJsonValue(const char* feature_name, const rapidjson::Value& v, const triton::server::FeatureMappingTables* tables) {
  if (tables == nullptr) {
    return -1;
  }
  if (v.IsString()) {
    return triton::server::GetFeatureMappingIdx(feature_name, v.GetString(), tables);
  }
  if (v.IsInt()) {
    return triton::server::GetFeatureMappingIdxForInt64(feature_name, static_cast<int64_t>(v.GetInt()), tables);
  }
  if (v.IsUint()) {
    return triton::server::GetFeatureMappingIdxForInt64(feature_name, static_cast<int64_t>(v.GetUint()), tables);
  }
  if (v.IsInt64()) {
    return triton::server::GetFeatureMappingIdxForInt64(feature_name, v.GetInt64(), tables);
  }
  if (v.IsUint64()) {
    const uint64_t uv = v.GetUint64();
    if (uv > static_cast<uint64_t>(INT64_MAX)) {
      char num_buf[32];
      const int n = std::snprintf(num_buf, sizeof(num_buf), "%llu", static_cast<unsigned long long>(uv));
      if (n <= 0 || static_cast<size_t>(n) >= sizeof(num_buf)) {
        return -1;
      }
      return triton::server::GetFeatureMappingIdx(feature_name, num_buf, tables);
    }
    return triton::server::GetFeatureMappingIdxForInt64(feature_name, static_cast<int64_t>(uv), tables);
  }
  return -1;
}

bool IsCampLevelFeature(const std::string& feature) {
  return feature == TRITON_BT_FEATURE_COOKIE || feature == TRITON_BT_FEATURE_RNK || feature == TRITON_BT_FEATURE_CAMPID;
}

bool UsesRawNumericFeature(const std::string& feature) {
  return (feature == TRITON_BT_FEATURE_UID) || (feature == TRITON_BT_FEATURE_VIDEO_VPW) || (feature == TRITON_BT_FEATURE_VIDEO_VPH) || (feature == TRITON_BT_FEATURE_MOBILEID) || (feature == TRITON_BT_FEATURE_VIEW) || (feature == TRITON_BT_FEATURE_COOKIE);
}

TRITONSERVER_Error* FillRawNumericFeature(const std::string& feature, const rapidjson::Value& v, float* out) {
  if (v.IsInt()) {
    *out = static_cast<float>(v.GetInt());
  } else if (v.IsUint()) {
    *out = static_cast<float>(v.GetUint());
  } else if (v.IsInt64()) {
    *out = static_cast<float>(v.GetInt64());
  } else if (v.IsUint64()) {
    *out = static_cast<float>(v.GetUint64());
  } else if (v.IsDouble()) {
    *out = static_cast<float>(v.GetDouble());
  } else {
    const std::string msg = std::string("feature '") + feature + "' must be a JSON number for raw passthrough";
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, msg.c_str());
  }
  return nullptr;
}

TRITONSERVER_Error* BuildImpBaseRow(const rapidjson::Value& imp, const std::vector<std::string>& feature_sequence, const triton::server::FeatureMappingTables& tables, std::vector<float>* out_row) {
  out_row->assign(feature_sequence.size(), 0.0f);
  for (size_t fi = 0; fi < feature_sequence.size(); ++fi) {
    const std::string& feature = feature_sequence[fi];
    if (feature == TRITON_BT_FEATURE_ADSIZE || IsCampLevelFeature(feature)) {
      continue;
    }

    const char* fkey = feature.c_str();
    if (!imp.HasMember(fkey)) {
      const std::string missing_field = std::string("missing JSON field for feature '") + feature + "'";
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, missing_field.c_str());
    }
    const rapidjson::Value& v = imp[fkey];

    TRITONSERVER_Error* err = nullptr;
    if (UsesRawNumericFeature(feature)) {
      err = FillRawNumericFeature(feature, v, &(*out_row)[fi]);
    } else {
      (*out_row)[fi] = static_cast<float>(FeatureIdxFromJsonValue(feature.c_str(), v, &tables));
    }
    if (err != nullptr) {
      return err;
    }
  }
  return nullptr;
}

TRITONSERVER_Error* FillCampFeaturesInRow(const rapidjson::Value& camp, int32_t campaign_id, const std::vector<std::string>& feature_sequence, const triton::server::FeatureMappingTables& tables, std::vector<float>* row) {
  for (size_t fi = 0; fi < feature_sequence.size(); ++fi) {
    const std::string& feature = feature_sequence[fi];
    if (feature == TRITON_BT_FEATURE_ADSIZE) {
      continue;
    }
    if (!IsCampLevelFeature(feature)) {
      continue;
    }

    const rapidjson::Value* src = nullptr;
    if (feature == TRITON_BT_FEATURE_CAMPID) {
      src = &camp[TRITON_BT_JSON_CID];
    } else if (camp.HasMember(feature.c_str())) {
      src = &camp[feature.c_str()];
    } else {
      const std::string missing_field = std::string("missing JSON field for feature '") + feature + "'";
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, missing_field.c_str());
    }

    TRITONSERVER_Error* err = nullptr;
    if (UsesRawNumericFeature(feature)) {
      err = FillRawNumericFeature(feature, *src, &(*row)[fi]);
    } else {
      (*row)[fi] = static_cast<float>(FeatureIdxFromJsonValue(feature.c_str(), *src, &tables));
    }
    if (err != nullptr) {
      return err;
    }
  }
  return nullptr;
}

TRITONSERVER_Error* RefreshReadyModelNamesInto(std::unordered_set<std::string>* out, TRITONSERVER_Server* server) {
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

}  // namespace

namespace triton { namespace server {

namespace {

struct ModelSlotBuild {
  size_t feature_count{0};
  std::vector<char> tensor;
  std::vector<ImpRouteRow> routes;
};

void AppendFloatRowToTensor(std::vector<char>* tensor, const std::vector<float>& row)
{
  const size_t nbytes = row.size() * sizeof(float);
  if (nbytes == 0) {
    return;
  }
  const char* bytes = reinterpret_cast<const char*>(row.data());
  tensor->insert(tensor->end(), bytes, bytes + nbytes);
}

TRITONSERVER_Error* CheckModelReadyCached(TRITONSERVER_Server* server, const std::string& model_name, std::unordered_map<std::string, bool>* ready_cache) {
  auto it = ready_cache->find(model_name);
  if (it != ready_cache->end()) {
    if (!it->second) {
      const std::string not_ready = "model " + model_name + " not ready";
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, not_ready.c_str());
    }
    return nullptr;
  }

  bool ready = false;
  TRITONSERVER_Error* err = TRITONSERVER_ServerModelIsReady(server, model_name.c_str(), -1 /* latest version */, &ready);
  if (err != nullptr) {
    return err;
  }
  (*ready_cache)[model_name] = ready;
  if (!ready) {
    const std::string not_ready = "model " + model_name + " not ready";
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, not_ready.c_str());
  }
  return nullptr;
}

}  // namespace

TRITONSERVER_Error* InitializeReadyModelNames(TRITONSERVER_Server* server) {
  if (server == nullptr) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "server pointer is null");
  }
  if (g_ready_models_valid.load(std::memory_order_acquire)) {
    return nullptr;
  }
  TRITONSERVER_Error* err = RefreshReadyModelNamesInto(&g_ready_model_names, server);
  if (err != nullptr) {
    return err;
  }
  g_ready_models_valid.store(true, std::memory_order_release);
  return nullptr;
}

const std::unordered_set<std::string>* ActiveReadyModelNames() {
  if (!g_ready_models_valid.load(std::memory_order_acquire)) {
    return nullptr;
  }
  return &g_ready_model_names;
}

TRITONSERVER_Error* GenerateImpsInferSlots(const rapidjson::Document& doc, TRITONSERVER_Server* server, std::vector<ImpsInferSlot>* out_slots, ImpRoutingTable* out_routing) {
  if (out_slots == nullptr || server == nullptr) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "invalid argument");
  }
  if (!doc.IsObject() || !doc.HasMember(TRITON_BT_JSON_IMPS) || !doc[TRITON_BT_JSON_IMPS].IsArray()) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "expected object with imps array");
  }

  const CampaignToFeatureMappings* cmap = ActiveCampaignToFeatureMappings();
  if (cmap == nullptr || cmap->empty()) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNAVAILABLE, "campaign feature mappings are not loaded");
  }

  std::unordered_map<std::string, bool> ready_cache;
  std::unordered_map<std::string, ModelSlotBuild> slots_by_model;
  const std::string* cached_slot_model_name = nullptr;
  ModelSlotBuild* cached_slot_build = nullptr;
  TRITONSERVER_Error* err = nullptr;

  out_slots->clear();
  if (out_routing != nullptr) {
    out_routing->imp_count = 0;
    out_routing->slots.clear();
  }

  const rapidjson::Value& imps = doc[TRITON_BT_JSON_IMPS];
  std::vector<float> row;
  std::unordered_map<std::string, std::vector<float>> imp_base_by_model;

  for (rapidjson::SizeType ii = 0; ii < imps.Size(); ++ii) {
    const rapidjson::Value& imp = imps[ii];
    if (!imp.IsObject()) {
      continue;
    }
    if (!imp.HasMember(TRITON_BT_JSON_CAMPS) || !imp[TRITON_BT_JSON_CAMPS].IsArray() || imp[TRITON_BT_JSON_CAMPS].Size() == 0) {
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "each impression must include a non-empty camps array");
    }
    const rapidjson::Value& camps = imp[TRITON_BT_JSON_CAMPS];
    imp_base_by_model.clear();

    for (rapidjson::SizeType ci = 0; ci < camps.Size(); ++ci) {
      const rapidjson::Value& camp = camps[ci];
      if (!camp.IsObject() || !camp.HasMember(TRITON_BT_JSON_CID) || !camp[TRITON_BT_JSON_CID].IsInt()) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "each camp must be an object with integer 'cid'");
      }
      const int32_t campaign_id = camp[TRITON_BT_JSON_CID].GetInt();

      auto cmap_it = cmap->find(campaign_id);
      if (cmap_it == cmap->end()) {
        cmap_it = cmap->find(0);
      }
      if (cmap_it == cmap->end()) {
        const std::string unknown_campaign = std::string("unknown campaign_id ") + std::to_string(campaign_id);
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, unknown_campaign.c_str());
      }
      const std::string& model_name = cmap_it->second.model_name;
      const std::vector<std::string>& feature_sequence = cmap_it->second.feature_sequence;
      const FeatureMappingTables& tables = cmap_it->second.feature_mapping;

      err = CheckModelReadyCached(server, model_name, &ready_cache);
      if (err != nullptr) {
        return err;
      }

      ModelSlotBuild* slot_build = nullptr;
      if (cached_slot_model_name != nullptr && model_name == *cached_slot_model_name) {
        slot_build = cached_slot_build;
      } else {
        auto slot_it = slots_by_model.find(model_name);
        if (slot_it == slots_by_model.end()) {
          slot_it = slots_by_model.emplace(model_name, ModelSlotBuild{}).first;
        }
        cached_slot_model_name = &slot_it->first;
        cached_slot_build = &slot_it->second;
        slot_build = cached_slot_build;
      }
      const size_t feature_count = feature_sequence.size();
      if (slot_build->feature_count == 0) {
        slot_build->feature_count = feature_count;
      } 
      else if (slot_build->feature_count != feature_count) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "inconsistent feature_count for model buffer");
      }

      auto base_it = imp_base_by_model.find(model_name);
      if (base_it == imp_base_by_model.end()) {
        std::vector<float> base_row;
        err = BuildImpBaseRow(imp, feature_sequence, tables, &base_row);
        if (err != nullptr) {
          return err;
        }
        base_it = imp_base_by_model.emplace(model_name, std::move(base_row)).first;
      }

      row = base_it->second;
      err = FillCampFeaturesInRow(camp, campaign_id, feature_sequence, tables, &row);
      if (err != nullptr) {
        return err;
      }

      int adsize_idx = -1;
      for (size_t fi = 0; fi < feature_sequence.size(); ++fi) {
        if (feature_sequence[fi] == TRITON_BT_FEATURE_ADSIZE) {
          adsize_idx = static_cast<int>(fi);
          break;
        }
      }

      if (adsize_idx >= 0 && camp.HasMember(TRITON_BT_FEATURE_ADSIZE) && camp[TRITON_BT_FEATURE_ADSIZE].IsArray()) {
        const rapidjson::Value& adsize = camp[TRITON_BT_FEATURE_ADSIZE];
        for (rapidjson::SizeType ai = 0; ai < adsize.Size(); ++ai) {
          const rapidjson::Value& adsize_item = adsize[ai];
          row[static_cast<size_t>(adsize_idx)] = static_cast<float>(FeatureIdxFromJsonValue(TRITON_BT_FEATURE_ADSIZE, adsize_item, &tables));
          AppendFloatRowToTensor(&slot_build->tensor, row);
          if (out_routing != nullptr) {
            slot_build->routes.push_back(ImpRouteRow{static_cast<int>(ii), static_cast<int>(ci), static_cast<int>(ai), campaign_id});
          }
        }
      }
    }
  }

  std::vector<std::string> model_names;
  model_names.reserve(slots_by_model.size());
  for (const auto& kv : slots_by_model) {
    model_names.push_back(kv.first);
  }
  std::sort(model_names.begin(), model_names.end());

  out_slots->reserve(model_names.size());
  if (out_routing != nullptr) {
    out_routing->imp_count = static_cast<int>(imps.Size());
    out_routing->slots.reserve(model_names.size());
  }

  for (const std::string& model_name : model_names) {
    auto it = slots_by_model.find(model_name);
    if (it == slots_by_model.end()) {
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "internal buffer map inconsistency");
    }
    ModelSlotBuild& built = it->second;
    if (built.feature_count == 0 || built.tensor.empty()) {
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "empty model buffer after transform");
    }
    if (built.tensor.size() % (built.feature_count * sizeof(float)) != 0) {
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "buffer length is not a multiple of feature_count");
    }

    ImpsInferSlot slot;
    slot.model_name = model_name;
    slot.model_version = 0;
    slot.feature_count = built.feature_count;
    slot.rows = built.tensor.size() / (built.feature_count * sizeof(float));
    slot.input_tensor = std::move(built.tensor);
    out_slots->push_back(std::move(slot));

    if (out_routing != nullptr) {
      out_routing->slots.push_back(std::move(built.routes));
    }
  }

  return nullptr;
}

}}  // namespace triton::server
#endif  // TRITON_ENABLE_MYSQL_ODBC
