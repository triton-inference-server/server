// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include <stdint.h>

namespace nvidia { namespace inferenceserver {

constexpr char kInferRequestHTTPHeader[] = "NV-InferRequest";
constexpr char kInferResponseHTTPHeader[] = "NV-InferResponse";
constexpr char kStatusHTTPHeader[] = "NV-Status";

constexpr char kInferRESTEndpoint[] = "api/infer";
constexpr char kStatusRESTEndpoint[] = "api/status";
constexpr char kProfileRESTEndpoint[] = "api/profile";
constexpr char kHealthRESTEndpoint[] = "api/health";

constexpr char kTensorFlowGraphDefPlatform[] = "tensorflow_graphdef";
constexpr char kTensorFlowSavedModelPlatform[] = "tensorflow_savedmodel";
constexpr char kTensorRTPlanPlatform[] = "tensorrt_plan";
constexpr char kCaffe2NetDefPlatform[] = "caffe2_netdef";
constexpr char kCustomPlatform[] = "custom";

constexpr char kModelConfigPbTxt[] = "config.pbtxt";
constexpr char kTensorRTPlanFilename[] = "model.plan";
constexpr char kTensorFlowGraphDefFilename[] = "model.graphdef";
constexpr char kTensorFlowSavedModelFilename[] = "model.savedmodel";
constexpr char kCaffe2NetDefFilename[] = "model.netdef";
constexpr char kCaffe2NetDefInitFilenamePrefix[] = "init_";
constexpr char kCustomFilename[] = "libcustom.so";

constexpr char kMetricsLabelModelName[] = "model";
constexpr char kMetricsLabelModelVersion[] = "version";
constexpr char kMetricsLabelGpuUuid[] = "gpu_uuid";

constexpr uint64_t NANOS_PER_SECOND = 1000000000;
constexpr int MAX_GRPC_MESSAGE_SIZE = INT32_MAX;
constexpr int SCHEDULER_DEFAULT_NICE = 5;

}}  // namespace nvidia::inferenceserver
