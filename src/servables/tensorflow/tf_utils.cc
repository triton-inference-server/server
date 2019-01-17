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

#include "src/servables/tensorflow/tf_utils.h"

namespace nvidia { namespace inferenceserver {

bool
CompareDimsExact(
    const tensorflow::TensorShapeProto& model_shape, const DimsList& dims,
    const bool supports_batching)
{
  // If the model configuration expects batching support in the model,
  // then the tensorflow shape first dimension must be -1.
  if (supports_batching) {
    if ((model_shape.dim().size() == 0) || (model_shape.dim(0).size() != -1)) {
      return false;
    }
  }

  if (model_shape.dim().size() != (dims.size() + (supports_batching ? 1 : 0))) {
    return false;
  }

  for (int i = 0; i < dims.size(); ++i) {
    if (model_shape.dim(i + (supports_batching ? 1 : 0)).size() != dims[i]) {
      return false;
    }
  }

  return true;
}

bool
CompareDimsSupported(
    const tensorflow::TensorShapeProto& model_shape, const DimsList& dims,
    const bool supports_batching)
{
  // If the model configuration expects batching support in the model,
  // then the tensorflow shape first dimension must be -1.
  if (supports_batching) {
    if ((model_shape.dim().size() == 0) || (model_shape.dim(0).size() != -1)) {
      return false;
    }
  }

  if (model_shape.dim().size() != (dims.size() + (supports_batching ? 1 : 0))) {
    return false;
  }

  for (int i = 0; i < dims.size(); ++i) {
    int64_t model_dim = model_shape.dim(i + (supports_batching ? 1 : 0)).size();
    if (model_dim == -1) {
      continue;
    }

    if (model_dim != dims[i]) {
      return false;
    }
  }

  return true;
}

bool
CompareDataType(tensorflow::DataType model_dtype, DataType dtype)
{
  tensorflow::DataType cdtype = ConvertDataType(dtype);
  if (cdtype == tensorflow::DT_INVALID) {
    return false;
  }

  return model_dtype == cdtype;
}

const std::string
DimsDebugString(const DimsList& dims)
{
  bool first = true;
  std::string str;
  str.append("[");
  for (int i = 0; i < dims.size(); ++i) {
    if (!first) {
      str.append(",");
    }
    str.append(std::to_string(dims[i]));
    first = false;
  }
  str.append("]");
  return str;
}

const std::string
DimsDebugString(const tensorflow::TensorShapeProto& dims)
{
  bool first = true;
  std::string str;
  str.append("[");
  for (int i = 0; i < dims.dim().size(); ++i) {
    if (!first) {
      str.append(",");
    }
    str.append(std::to_string(dims.dim(i).size()));
    first = false;
  }
  str.append("]");
  return str;
}

tensorflow::DataType
ConvertDataType(DataType dtype)
{
  switch (dtype) {
    case DataType::TYPE_INVALID:
      return tensorflow::DT_INVALID;
    case DataType::TYPE_BOOL:
      return tensorflow::DT_BOOL;
    case DataType::TYPE_UINT8:
      return tensorflow::DT_UINT8;
    case DataType::TYPE_UINT16:
      return tensorflow::DT_UINT16;
    case DataType::TYPE_UINT32:
      return tensorflow::DT_UINT32;
    case DataType::TYPE_UINT64:
      return tensorflow::DT_UINT64;
    case DataType::TYPE_INT8:
      return tensorflow::DT_INT8;
    case DataType::TYPE_INT16:
      return tensorflow::DT_INT16;
    case DataType::TYPE_INT32:
      return tensorflow::DT_INT32;
    case DataType::TYPE_INT64:
      return tensorflow::DT_INT64;
    case DataType::TYPE_FP16:
      return tensorflow::DT_HALF;
    case DataType::TYPE_FP32:
      return tensorflow::DT_FLOAT;
    case DataType::TYPE_FP64:
      return tensorflow::DT_DOUBLE;
    case DataType::TYPE_STRING:
      return tensorflow::DT_STRING;
    default:
      break;
  }

  return tensorflow::DT_INVALID;
}

DataType
ConvertDataType(tensorflow::DataType dtype)
{
  switch (dtype) {
    case tensorflow::DT_INVALID:
      return DataType::TYPE_INVALID;
    case tensorflow::DT_BOOL:
      return DataType::TYPE_BOOL;
    case tensorflow::DT_UINT8:
      return DataType::TYPE_UINT8;
    case tensorflow::DT_UINT16:
      return DataType::TYPE_UINT16;
    case tensorflow::DT_UINT32:
      return DataType::TYPE_UINT32;
    case tensorflow::DT_UINT64:
      return DataType::TYPE_UINT64;
    case tensorflow::DT_INT8:
      return DataType::TYPE_INT8;
    case tensorflow::DT_INT16:
      return DataType::TYPE_INT16;
    case tensorflow::DT_INT32:
      return DataType::TYPE_INT32;
    case tensorflow::DT_INT64:
      return DataType::TYPE_INT64;
    case tensorflow::DT_HALF:
      return DataType::TYPE_FP16;
    case tensorflow::DT_FLOAT:
      return DataType::TYPE_FP32;
    case tensorflow::DT_DOUBLE:
      return DataType::TYPE_FP64;
    case tensorflow::DT_STRING:
      return DataType::TYPE_STRING;
    default:
      break;
  }

  return DataType::TYPE_INVALID;
}

}}  // namespace nvidia::inferenceserver
