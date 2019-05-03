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

#include "src/backends/tensorrt/plan_utils.h"

namespace nvidia { namespace inferenceserver {

DataType
ConvertDatatype(nvinfer1::DataType trt_type)
{
  switch (trt_type) {
    case nvinfer1::DataType::kFLOAT:
      return TYPE_FP32;
    case nvinfer1::DataType::kHALF:
      return TYPE_FP16;
    case nvinfer1::DataType::kINT8:
      return TYPE_INT8;
    case nvinfer1::DataType::kINT32:
      return TYPE_INT32;
  }

  return TYPE_INVALID;
}

bool
CompareDims(const nvinfer1::Dims& model_dims, const DimsList& dims)
{
  if (model_dims.nbDims != dims.size()) {
    return false;
  }

  for (int i = 0; i < model_dims.nbDims; ++i) {
    if (model_dims.d[i] != dims[i]) {
      return false;
    }
  }

  return true;
}

const std::string
DimsDebugString(const nvinfer1::Dims& dims)
{
  bool first = true;
  std::string str;
  str.append("[");
  for (int i = 0; i < dims.nbDims; ++i) {
    if (!first) {
      str.append(",");
    }
    str.append(std::to_string(dims.d[i]));
    first = false;
  }
  str.append("]");
  return str;
}

}}  // namespace nvidia::inferenceserver
