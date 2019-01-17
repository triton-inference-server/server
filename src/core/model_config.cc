// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "src/core/model_config.h"

#include "src/core/constants.h"

namespace nvidia { namespace inferenceserver {

bool
IsFixedSizeDataType(const DataType dtype)
{
  return dtype != TYPE_STRING;
}

size_t
GetDataTypeByteSize(const DataType dtype)
{
  switch (dtype) {
    case TYPE_BOOL:
      return 1;
    case TYPE_UINT8:
      return 1;
    case TYPE_UINT16:
      return 2;
    case TYPE_UINT32:
      return 4;
    case TYPE_UINT64:
      return 8;
    case TYPE_INT8:
      return 1;
    case TYPE_INT16:
      return 2;
    case TYPE_INT32:
      return 4;
    case TYPE_INT64:
      return 8;
    case TYPE_FP16:
      return 2;
    case TYPE_FP32:
      return 4;
    case TYPE_FP64:
      return 8;
    case TYPE_STRING:
      return 0;
    default:
      break;
  }

  return 0;
}

int64_t
GetElementCount(const DimsList& dims)
{
  int64_t cnt = 0;
  for (auto dim : dims) {
    if (dim == WILDCARD_DIM) {
      return -1;
    }

    if (cnt == 0) {
      cnt = dim;
    } else {
      cnt *= dim;
    }
  }

  return cnt;
}

int64_t
GetElementCount(const std::vector<int64_t>& dims)
{
  int64_t cnt = 0;
  for (auto dim : dims) {
    if (dim == WILDCARD_DIM) {
      return -1;
    }

    if (cnt == 0) {
      cnt = dim;
    } else {
      cnt *= dim;
    }
  }

  return cnt;
}

int64_t
GetElementCount(const ModelInput& mio)
{
  return GetElementCount(mio.dims());
}

int64_t
GetElementCount(const ModelOutput& mio)
{
  return GetElementCount(mio.dims());
}

uint64_t
GetByteSize(const DataType& dtype, const DimsList& dims)
{
  size_t dt_size = GetDataTypeByteSize(dtype);
  if (dt_size <= 0) {
    return 0;
  }

  int64_t cnt = GetElementCount(dims);
  if (cnt == -1) {
    return 0;
  }

  return cnt * dt_size;
}

uint64_t
GetByteSize(const DataType& dtype, const std::vector<int64_t>& dims)
{
  size_t dt_size = GetDataTypeByteSize(dtype);
  if (dt_size <= 0) {
    return 0;
  }

  int64_t cnt = GetElementCount(dims);
  if (cnt == -1) {
    return 0;
  }

  return cnt * dt_size;
}

uint64_t
GetByteSize(const ModelInput& mio)
{
  return GetByteSize(mio.data_type(), mio.dims());
}

uint64_t
GetByteSize(const ModelOutput& mio)
{
  return GetByteSize(mio.data_type(), mio.dims());
}

Platform
GetPlatform(const std::string& platform_str)
{
  if (platform_str == kTensorFlowGraphDefPlatform) {
    return Platform::PLATFORM_TENSORFLOW_GRAPHDEF;
  } else if (platform_str == kTensorFlowSavedModelPlatform) {
    return Platform::PLATFORM_TENSORFLOW_SAVEDMODEL;
  } else if (platform_str == kTensorRTPlanPlatform) {
    return Platform::PLATFORM_TENSORRT_PLAN;
  } else if (platform_str == kCaffe2NetDefPlatform) {
    return Platform::PLATFORM_CAFFE2_NETDEF;
  } else if (platform_str == kCustomPlatform) {
    return Platform::PLATFORM_CUSTOM;
  }

  return Platform::PLATFORM_UNKNOWN;
}

int
GetCpuNiceLevel(const ModelConfig& config)
{
  int nice = SCHEDULER_DEFAULT_NICE;
  if (config.has_optimization()) {
    switch (config.optimization().priority()) {
      case ModelOptimizationPolicy::PRIORITY_MAX:
        nice = 0;
        break;
      case ModelOptimizationPolicy::PRIORITY_MIN:
        nice = 19;
        break;
      default:
        nice = SCHEDULER_DEFAULT_NICE;
        break;
    }
  }

  return nice;
}

bool
CompareDims(const DimsList& dims0, const DimsList& dims1)
{
  if (dims0.size() != dims1.size()) {
    return false;
  }

  for (int i = 0; i < dims0.size(); ++i) {
    if (dims0[i] != dims1[i]) {
      return false;
    }
  }

  return true;
}

bool
CompareDimsWithWildcard(const DimsList& dims0, const DimsList& dims1)
{
  if (dims0.size() != dims1.size()) {
    return false;
  }

  for (int i = 0; i < dims0.size(); ++i) {
    if ((dims0[i] != WILDCARD_DIM) && (dims1[i] != WILDCARD_DIM) &&
        (dims0[i] != dims1[i])) {
      return false;
    }
  }

  return true;
}

}}  // namespace nvidia::inferenceserver
