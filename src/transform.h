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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "triton/core/tritonserver.h"

namespace triton { namespace server {

// Parses `json` into a document. If built with `TRITON_ENABLE_MYSQL_ODBC` and
// `server` is non-null and the root object contains `"imps"`, replaces
// `*out_doc` with a multi-infer request built from impressions; otherwise
// stores the parsed root in `*out_doc`.
// Returns nullptr on success, or a TRITONSERVER_Error that the caller must
// delete with TRITONSERVER_ErrorDelete on failure.
TRITONSERVER_Error* ParseRequest(const std::string& json, TRITONSERVER_Server* server, rapidjson::Document* out_doc);

#ifdef TRITON_ENABLE_MYSQL_ODBC

// Builds `*out_doc` as a `POST /v2/multi_infer` body from `doc` (imps/camps).
TRITONSERVER_Error* GenerateInputVectors(
    const rapidjson::Document& doc, TRITONSERVER_Server* server,
    rapidjson::Document* out_doc);

TRITONSERVER_Error* GetReadyModelNames(
    TRITONSERVER_Server* server, std::unordered_set<std::string>* out);

using NamedDoubleBuffers = std::unordered_map<std::string, std::vector<double>>;
using ModelNameToFeatureCount = std::unordered_map<std::string, size_t>;

TRITONSERVER_Error* AppendRowToNamedDoubleBuffers(
    NamedDoubleBuffers* buffers, ModelNameToFeatureCount* feature_counts,
    const std::string& vector_name, const std::vector<double>& row,
    size_t feature_count);

TRITONSERVER_Error* BuildMultiInferRequestDocument(
    const NamedDoubleBuffers& buffers,
    const ModelNameToFeatureCount& feature_counts, rapidjson::Document* out_doc);

#endif  // TRITON_ENABLE_MYSQL_ODBC

}}  // namespace triton::server
