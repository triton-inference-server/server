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
#pragma once

#include <rapidjson/document.h>
#include <string>
#include <unordered_set>
#include <vector>

#include "triton/core/tritonserver.h"

namespace triton { namespace server {

#ifdef TRITON_ENABLE_MYSQL_ODBC

#define TRITON_BT_FEATURE_ADSIZE "adsize"
#define TRITON_BT_FEATURE_COOKIE "cookie"
#define TRITON_BT_FEATURE_RNK "rnk"
#define TRITON_BT_FEATURE_CAMPID "campid"
#define TRITON_BT_JSON_IMPS "imps"
#define TRITON_BT_JSON_CAMPS "camps"
#define TRITON_BT_JSON_CID "cid"
#define TRITON_BT_FEATURE_UID "uid"
#define TRITON_BT_FEATURE_VIDEO_VPW "video_vpw"
#define TRITON_BT_FEATURE_VIDEO_VPH "video_vph"
#define TRITON_BT_FEATURE_MOBILEID "mobileid"
#define TRITON_BT_FEATURE_VIEW "view"

// Per batched infer row when folding multi_infer results back to imps/camps.
struct ImpRouteRow {
  int imp_idx{0};
  int camp_idx{0};
  int adsize_idx{0};
  int32_t cid{0};
};

// In-memory routing for imps-shaped requests (not serialized into multi_infer JSON).
struct ImpRoutingTable {
  int imp_count{0};
  // One vector per multi_infer request slot (sorted model name order).
  std::vector<std::vector<ImpRouteRow>> slots;
};

// One model's imps feature matrix ready for direct TRITONSERVER_InferenceRequest
// fill (replaces per-slot infer JSON for the imps fast path).
struct ImpsInferSlot {
  std::string model_name;
  int64_t model_version{0};
  // Row-major FP32 tensor bytes (rows * feature_count * sizeof(float)).
  std::vector<char> input_tensor;
  size_t rows{0};
  size_t feature_count{0};
};

constexpr const char* kImpsInputTensorName = "input__0";
constexpr const char* kImpsOutputTensorName = "output__0";

// Populate the ready-model snapshot once at process startup (after models are loaded).
TRITONSERVER_Error* InitializeReadyModelNames(TRITONSERVER_Server* server);

// Lock-free read of the snapshot initialized by InitializeReadyModelNames.
const std::unordered_set<std::string>* ActiveReadyModelNames();

// Feature mapping + FP32 tensor build for POST /v2/multi_infer imps requests.
TRITONSERVER_Error* GenerateImpsInferSlots(
    const rapidjson::Document& doc, TRITONSERVER_Server* server,
    std::vector<ImpsInferSlot>* out_slots,
    ImpRoutingTable* out_routing = nullptr);

// Populates TRITONSERVER_InferenceRequest from a slot: HTTPAPIServer::FillImpsTritonRequest
// in http_server.cc (requires InferRequestClass for input lifetime and output alloc).

#endif  // TRITON_ENABLE_MYSQL_ODBC

}}  // namespace triton::server
