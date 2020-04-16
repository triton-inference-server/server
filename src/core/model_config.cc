// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
  bool first = true;
  int64_t cnt = 0;
  for (auto dim : dims) {
    if (dim == WILDCARD_DIM) {
      return -1;
    }

    if (first) {
      cnt = dim;
      first = false;
    } else {
      cnt *= dim;
    }
  }

  return cnt;
}

int64_t
GetElementCount(const std::vector<int64_t>& dims)
{
  bool first = true;
  int64_t cnt = 0;
  for (auto dim : dims) {
    if (dim == WILDCARD_DIM) {
      return -1;
    }

    if (first) {
      cnt = dim;
      first = false;
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

int64_t
GetByteSize(const DataType& dtype, const DimsList& dims)
{
  size_t dt_size = GetDataTypeByteSize(dtype);
  if (dt_size == 0) {
    return -1;
  }

  int64_t cnt = GetElementCount(dims);
  if (cnt == -1) {
    return -1;
  }

  return cnt * dt_size;
}

int64_t
GetByteSize(const DataType& dtype, const std::vector<int64_t>& dims)
{
  size_t dt_size = GetDataTypeByteSize(dtype);
  if (dt_size == 0) {
    return -1;
  }

  int64_t cnt = GetElementCount(dims);
  if (cnt == -1) {
    return -1;
  }

  return cnt * dt_size;
}

int64_t
GetByteSize(const int batch_size, const DataType& dtype, const DimsList& dims)
{
  if (dims.size() == 0) {
    return batch_size * GetDataTypeByteSize(dtype);
  }

  int64_t bs = GetByteSize(dtype, dims);
  if (bs == -1) {
    return -1;
  }

  return std::max(1, batch_size) * bs;
}

int64_t
GetByteSize(
    const int batch_size, const DataType& dtype,
    const std::vector<int64_t>& dims)
{
  if (dims.size() == 0) {
    return batch_size * GetDataTypeByteSize(dtype);
  }

  int64_t bs = GetByteSize(dtype, dims);
  if (bs == -1) {
    return -1;
  }

  return std::max(1, batch_size) * bs;
}

int64_t
GetByteSize(const ModelInput& mio)
{
  return GetByteSize(mio.data_type(), mio.dims());
}

int64_t
GetByteSize(const ModelOutput& mio)
{
  return GetByteSize(mio.data_type(), mio.dims());
}

Platform
GetPlatform(const std::string& platform_str)
{
#ifdef TRTIS_ENABLE_TENSORFLOW
  if (platform_str == kTensorFlowGraphDefPlatform) {
    return Platform::PLATFORM_TENSORFLOW_GRAPHDEF;
  }
  if (platform_str == kTensorFlowSavedModelPlatform) {
    return Platform::PLATFORM_TENSORFLOW_SAVEDMODEL;
  }
#endif  // TRTIS_ENABLE_TENSORFLOW

#ifdef TRTIS_ENABLE_TENSORRT
  if (platform_str == kTensorRTPlanPlatform) {
    return Platform::PLATFORM_TENSORRT_PLAN;
  }
#endif  // TRTIS_ENABLE_TENSORRT

#ifdef TRTIS_ENABLE_CAFFE2
  if (platform_str == kCaffe2NetDefPlatform) {
    return Platform::PLATFORM_CAFFE2_NETDEF;
  }
#endif  // TRTIS_ENABLE_CAFFE2

#ifdef TRTIS_ENABLE_CUSTOM
  if (platform_str == kCustomPlatform) {
    return Platform::PLATFORM_CUSTOM;
  }
#endif  // TRTIS_ENABLE_CUSTOM

#ifdef TRTIS_ENABLE_ONNXRUNTIME
  if (platform_str == kOnnxRuntimeOnnxPlatform) {
    return Platform::PLATFORM_ONNXRUNTIME_ONNX;
  }
#endif  // TRTIS_ENABLE_ONNXRUNTIME

#ifdef TRTIS_ENABLE_PYTORCH
  if (platform_str == kPyTorchLibTorchPlatform) {
    return Platform::PLATFORM_PYTORCH_LIBTORCH;
  }
#endif  // TRTIS_ENABLE_PYTORCH

#ifdef TRTIS_ENABLE_ENSEMBLE
  if (platform_str == kEnsemblePlatform) {
    return Platform::PLATFORM_ENSEMBLE;
  }
#endif  // TRTIS_ENABLE_ENSEMBLE

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
CompareDims(
    const std::vector<int64_t>& dims0, const std::vector<int64_t>& dims1)
{
  if (dims0.size() != dims1.size()) {
    return false;
  }

  for (size_t i = 0; i < dims0.size(); ++i) {
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

bool
CompareDimsWithWildcard(
    const DimsList& dims0, const std::vector<int64_t>& dims1)
{
  if (dims0.size() != (int64_t)dims1.size()) {
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

std::string
DimsListToString(const DimsList& dims)
{
  bool first = true;

  std::string str("[");
  for (const auto& dim : dims) {
    if (!first) {
      str += ",";
    }
    str += std::to_string(dim);
    first = false;
  }

  str += "]";
  return str;
}

std::string
DimsListToString(const std::vector<int64_t>& dims, const int start_idx)
{
  int idx = 0;

  std::string str("[");
  for (const auto& dim : dims) {
    if (idx >= start_idx) {
      if (idx > start_idx) {
        str += ",";
      }
      str += std::to_string(dim);
    }

    idx++;
  }

  str += "]";
  return str;
}

const char*
DataTypeToProtocolString(const DataType dtype)
{
  switch (dtype) {
    case TYPE_BOOL:
      return "BOOL";
    case TYPE_UINT8:
      return "UINT8";
    case TYPE_UINT16:
      return "UINT16";
    case TYPE_UINT32:
      return "UINT32";
    case TYPE_UINT64:
      return "UINT64";
    case TYPE_INT8:
      return "INT8";
    case TYPE_INT16:
      return "INT16";
    case TYPE_INT32:
      return "INT32";
    case TYPE_INT64:
      return "INT64";
    case TYPE_FP16:
      return "FP16";
    case TYPE_FP32:
      return "FP32";
    case TYPE_FP64:
      return "FP64";
    case TYPE_STRING:
      return "BYTES";
    default:
      break;
  }

  return "<invalid>";
}

DataType
ProtocolStringToDataType(const std::string& dtype)
{
  return ProtocolStringToDataType(dtype.c_str(), dtype.size());
}

DataType
ProtocolStringToDataType(const char* dtype, size_t len)
{
  if (len < 4 || len > 6) {
    return TYPE_INVALID;
  }

  if ((*dtype == 'I') && (len != 6)) {
    if ((dtype[1] == 'N') && (dtype[2] == 'T')) {
      if ((dtype[3] == '8') && (len == 4)) {
        return TYPE_INT8;
      } else if ((dtype[3] == '1') && (dtype[4] == '6')) {
        return TYPE_INT16;
      } else if ((dtype[3] == '3') && (dtype[4] == '2')) {
        return TYPE_INT32;
      } else if ((dtype[3] == '6') && (dtype[4] == '4')) {
        return TYPE_INT64;
      }
    }
  } else if ((*dtype == 'U') && (len != 4)) {
    if ((dtype[1] == 'I') && (dtype[2] == 'N') && (dtype[3] == 'T')) {
      if ((dtype[4] == '8') && (len == 5)) {
        return TYPE_UINT8;
      } else if ((dtype[4] == '1') && (dtype[5] == '6')) {
        return TYPE_UINT16;
      } else if ((dtype[4] == '3') && (dtype[5] == '2')) {
        return TYPE_UINT32;
      } else if ((dtype[4] == '6') && (dtype[5] == '4')) {
        return TYPE_UINT64;
      }
    }
  } else if ((*dtype == 'F') && (dtype[1] == 'P') && (len == 4)) {
    if ((dtype[2] == '1') && (dtype[3] == '6')) {
      return TYPE_FP16;
    } else if ((dtype[2] == '3') && (dtype[3] == '2')) {
      return TYPE_FP32;
    } else if ((dtype[2] == '6') && (dtype[3] == '4')) {
      return TYPE_FP64;
    }
  } else if (*dtype == 'B') {
    if (dtype[1] == 'Y') {
      if (!strcmp(dtype + 2, "TES")) {
        return TYPE_STRING;
      }
    } else if (!strcmp(dtype + 1, "OOL")) {
      return TYPE_BOOL;
    }
  }

  return TYPE_INVALID;
}

}}  // namespace nvidia::inferenceserver
