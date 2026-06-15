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
#include <string>
#include <utility>
#include <rapidjson/document.h>
#include <vector>
#include <iostream>

namespace {

int FeatureIdxFromJsonValue(const char* feature_name, const rapidjson::Value& v, const triton::server::FeatureMappingTables* tables) {
  if (tables == nullptr) {
    return -1;
  }
  if (v.IsString()) {
    return triton::server::GetFeatureMappingIdx(feature_name, v.GetString(), tables);
  }
  if (v.IsInt()) {
    const std::string tmp = std::to_string(v.GetInt());
    return triton::server::GetFeatureMappingIdx(feature_name, tmp.c_str(), tables);
  }
  if (v.IsUint()) {
    const std::string tmp = std::to_string(v.GetUint());
    return triton::server::GetFeatureMappingIdx(feature_name, tmp.c_str(), tables);
  }
  if (v.IsInt64()) {
    const std::string tmp = std::to_string(v.GetInt64());
    return triton::server::GetFeatureMappingIdx(feature_name, tmp.c_str(), tables);
  }
  if (v.IsUint64()) {
    const std::string tmp = std::to_string(v.GetUint64());
    return triton::server::GetFeatureMappingIdx(feature_name, tmp.c_str(), tables);
  }
  return -1;
}
}  // namespace

namespace triton { namespace server {

TRITONSERVER_Error* ParseRequest(const std::string& json, TRITONSERVER_Server* server, rapidjson::Document* out_doc) {
  if (out_doc == nullptr) {
    return TRITONSERVER_ErrorNew(
    TRITONSERVER_ERROR_INVALID_ARG, "output document pointer is null");
  }

  rapidjson::Document doc;
  doc.Parse(json.data(), json.size());
  if (doc.HasParseError()) {
    const char* msg = rapidjson::GetParseError_En(doc.GetParseError());
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, msg);
  }

  if (server != nullptr && doc.IsObject() && doc.HasMember("imps") && doc["imps"].IsArray()) {
    return GenerateInputVectors(doc, server, out_doc);
  }

  *out_doc = std::move(doc);
  return nullptr;
}

TRITONSERVER_Error* GetReadyModelNames(TRITONSERVER_Server* server, std::vector<std::string>* out) {
  if (out == nullptr) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "output vector pointer is null");
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

  for (rapidjson::SizeType i = 0; i < index_doc.Size(); ++i) {
    const rapidjson::Value& o = index_doc[i];
    if (!o.IsObject() || !o.HasMember("name")) {
      continue;
    }
    const rapidjson::Value& n = o["name"];
    if (!n.IsString()) {
      continue;
    }
    out->emplace_back(n.GetString());
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

TRITONSERVER_Error* GenerateInputVectors(const rapidjson::Document& doc, TRITONSERVER_Server* server, rapidjson::Document* out_doc) {
  if (out_doc == nullptr || server == nullptr) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "invalid argument");
  }
  if (!doc.IsObject() || !doc.HasMember("imps") || !doc["imps"].IsArray()) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "expected object with imps array");
  }

  const CampaignToFeatureMappings* cmap = ActiveCampaignToFeatureMappings();
  if (cmap == nullptr || cmap->empty()) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNAVAILABLE, "campaign feature mappings are not loaded");
  }

  std::vector<std::string> ready_model_names;
  TRITONSERVER_Error* err = GetReadyModelNames(server, &ready_model_names);
  if (err != nullptr) {
    return err;
  }
  if (ready_model_names.empty()) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "no ready models reported by server");
  }

  NamedDoubleBuffers buffers;
  ModelNameToFeatureCount counts;

  const rapidjson::Value& imps = doc["imps"];
  for (rapidjson::SizeType ii = 0; ii < imps.Size(); ++ii) {
    const rapidjson::Value& imp = imps[ii];
    if (!imp.IsObject()) {
      continue;
    }
    if (!imp.HasMember("camps") || !imp["camps"].IsArray() || imp["camps"].Size() == 0) {
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "each impression must include a non-empty camps array");
    }
    const rapidjson::Value& camps = imp["camps"];
    for (rapidjson::SizeType ci = 0; ci < camps.Size(); ++ci) {
      const rapidjson::Value& camp = camps[ci];
      if (!camp.IsObject() || !camp.HasMember("cid") || !camp["cid"].IsInt()) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "each camp must be an object with integer 'cid'");
      }
      const int32_t campaign_id = camp["cid"].GetInt();

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

      if (std::find(ready_model_names.begin(), ready_model_names.end(), model_name) == ready_model_names.end()) {
        const std::string not_ready = "model " + model_name + " not ready";
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, not_ready.c_str());
      }

      std::vector<double> row(feature_sequence.size(), 0.0);
      int adsize_idx = -1;

      for (size_t fi = 0; fi < feature_sequence.size(); ++fi) {
        const std::string& feature = feature_sequence[fi];
        const char* fkey = feature.c_str();

        if (feature == "adsize") {
          adsize_idx = static_cast<int>(fi);
          continue;
        }

        const rapidjson::Value* src = nullptr;
        if (feature == "cookie" || feature == "rnk") {
          if (camp.IsObject() && camp.HasMember(fkey)) {
            src = &camp[fkey];
          }
        } 
        else if(feature == "campid") {
          if (camp.IsObject() && camp.HasMember("cid")) {
            src = &camp["cid"];
          }
        }
        else {
          if (imp.IsObject() && imp.HasMember(fkey)) {
            src = &imp[fkey];
          }
        }

        if (src == nullptr) {
          const std::string missing_field = std::string("missing JSON field for feature '") + feature + "'";
          return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, missing_field.c_str());
        }
        int idx = FeatureIdxFromJsonValue(feature.c_str(), *src, &tables);
        row[fi] = static_cast<double>(idx);
        }

        if (adsize_idx >= 0 && camp.HasMember("adsize") && camp["adsize"].IsArray()) {
          const rapidjson::Value& adsize = camp["adsize"];
          for (rapidjson::SizeType ai = 0; ai < adsize.Size(); ++ai) {
            const rapidjson::Value& adsize_item = adsize[ai];
            int mapped = FeatureIdxFromJsonValue("adsize", adsize_item, &tables);
            row[static_cast<size_t>(adsize_idx)] = static_cast<double>(mapped);
            err = AppendRowToNamedDoubleBuffers(&buffers, &counts, model_name, row, feature_sequence.size());
            if (err != nullptr) {
              return err;
            }
          }
        }
      }
    }
    err = BuildMultiInferRequestDocument(buffers, counts, out_doc);
    return err;
  }
}
} // namespace triton::server
#endif  // TRITON_ENABLE_MYSQL_ODBC
