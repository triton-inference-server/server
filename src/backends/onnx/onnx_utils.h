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

#pragma once

#include <onnxruntime_c_api.h>
#include "src/core/model_config.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

#define RETURN_IF_ORT_ERROR(S)                                           \
  do {                                                                   \
    OrtStatus* status__ = (S);                                           \
    if (status__ != nullptr) {                                           \
      OrtErrorCode code = OrtGetErrorCode(status__);                     \
      std::string msg = std::string(OrtGetErrorMessage(status__));       \
      OrtReleaseStatus(status__);                                        \
      return Status(                                                     \
          RequestStatusCode::INTERNAL, "onnx runtime error " +           \
                                           std::to_string(code) + ": " + \
                                           std::string(msg));            \
    }                                                                    \
  } while (false)

struct OnnxTensorInfo {
  OnnxTensorInfo(ONNXTensorElementDataType type, std::vector<int64_t> dims)
      : type_(type), dims_(dims)
  {
  }

  ONNXTensorElementDataType type_;
  std::vector<int64_t> dims_;
};

// Wrapper class for ORT resources. Mainly used for destruction
template <class T>
class OrtResourceWrapper {
 public:
  OrtResourceWrapper(T ort_resource, void (*ReleaseFunction)(T))
      : resource_(ort_resource), ReleaseFunc_(ReleaseFunction)
  {
  }
  ~OrtResourceWrapper() { ReleaseFunc_(resource_); }

 private:
  T resource_;
  void (*ReleaseFunc_)(T);
};

using OnnxTensorInfoMap = std::unordered_map<std::string, OnnxTensorInfo>;

std::string OnnxDataTypeName(ONNXTensorElementDataType onnx_type);

DataType ConvertFromOnnxDataType(ONNXTensorElementDataType onnx_type);

ONNXTensorElementDataType ConvertToOnnxDataType(DataType data_type);

Status InputNames(OrtSession* session, std::set<std::string>& names);

Status OutputNames(OrtSession* session, std::set<std::string>& names);

Status InputInfos(
    OrtSession* session, OrtAllocator* allocator, OnnxTensorInfoMap& infos);

Status OutputInfos(
    OrtSession* session, OrtAllocator* allocator, OnnxTensorInfoMap& infos);

Status CompareDimsSupported(
    const std::string& model_name, const std::string& tensor_name,
    const std::vector<int64_t>& model_shape, const DimsList& dims,
    const int max_batch_size);

}}  // namespace nvidia::inferenceserver
