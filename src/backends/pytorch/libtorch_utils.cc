// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "libtorch_utils.h"

namespace triton { namespace backend { namespace pytorch {

TRITONSERVER_DataType
ConvertTorchTypeToDataType(const torch::ScalarType& stype)
{
  switch (stype) {
    case torch::kBool:
      return TRITONSERVER_TYPE_BOOL;
    case torch::kByte:
      return TRITONSERVER_TYPE_UINT8;
    case torch::kChar:
      return TRITONSERVER_TYPE_INT8;
    case torch::kShort:
      return TRITONSERVER_TYPE_INT16;
    case torch::kInt:
      return TRITONSERVER_TYPE_INT32;
    case torch::kLong:
      return TRITONSERVER_TYPE_INT64;
    case torch::kHalf:
      return TRITONSERVER_TYPE_FP16;
    case torch::kFloat:
      return TRITONSERVER_TYPE_FP32;
    case torch::kDouble:
      return TRITONSERVER_TYPE_FP64;
    default:
      break;
  }

  return TRITONSERVER_TYPE_INVALID;
}

std::pair<bool, torch::ScalarType>
ConvertDataTypeToTorchType(const TRITONSERVER_DataType& dtype)
{
  torch::ScalarType type = torch::kInt;
  switch (dtype) {
    case TRITONSERVER_TYPE_BOOL:
      type = torch::kBool;
      break;
    case TRITONSERVER_TYPE_UINT8:
      type = torch::kByte;
      break;
    case TRITONSERVER_TYPE_INT8:
      type = torch::kChar;
      break;
    case TRITONSERVER_TYPE_INT16:
      type = torch::kShort;
      break;
    case TRITONSERVER_TYPE_INT32:
      type = torch::kInt;
      break;
    case TRITONSERVER_TYPE_INT64:
      type = torch::kLong;
      break;
    case TRITONSERVER_TYPE_FP16:
      type = torch::kHalf;
      break;
    case TRITONSERVER_TYPE_FP32:
      type = torch::kFloat;
      break;
    case TRITONSERVER_TYPE_FP64:
      type = torch::kDouble;
      break;
    case TRITONSERVER_TYPE_UINT16:
    case TRITONSERVER_TYPE_UINT32:
    case TRITONSERVER_TYPE_UINT64:
    case TRITONSERVER_TYPE_BYTES:
    default:
      return std::make_pair(false, type);
  }

  return std::make_pair(true, type);
}

std::pair<bool, torch::ScalarType>
ModelConfigDataTypeToTorchType(const std::string& data_type_str)
{
  torch::ScalarType type = torch::kInt;

  // Must start with "TYPE_".
  if (data_type_str.rfind("TYPE_", 0) != 0) {
    return std::make_pair(false, type);
  }

  const std::string dtype = data_type_str.substr(strlen("TYPE_"));

  if (dtype == "BOOL") {
    type = torch::kBool;
  } else if (dtype == "UINT8") {
    type = torch::kByte;
  } else if (dtype == "INT8") {
    type = torch::kChar;
  } else if (dtype == "INT16") {
    type = torch::kShort;
  } else if (dtype == "INT32") {
    type = torch::kInt;
  } else if (dtype == "INT64") {
    type = torch::kLong;
  } else if (dtype == "FP16") {
    type = torch::kHalf;
  } else if (dtype == "FP32") {
    type = torch::kFloat;
  } else if (dtype == "FP64") {
    type = torch::kDouble;
  }

  return std::make_pair(false, type);
}

}}}  // namespace triton::backend::pytorch