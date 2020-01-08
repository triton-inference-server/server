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

#include "src/backends/tensorflow/tf_utils.h"

namespace nvidia { namespace inferenceserver {

Status
CompareDims(
    const std::string& model_name, const std::string& tensor_name,
    const TRTISTF_Shape* model_shape, const DimsList& dims,
    const bool supports_batching, const bool compare_exact)
{
  // If the model configuration expects batching support in the model,
  // then the tensorflow shape first dimension must be -1.
  if (supports_batching) {
    if ((model_shape->rank_ == 0) || (model_shape->dims_[0] != -1)) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "model '" + model_name + "', tensor '" + tensor_name +
              "': for the model to support batching the shape should have at "
              "least 1 dimension and the first dimension must be -1; but shape "
              "expected by the model is " +
              ShapeToString(model_shape));
    }

    DimsList full_dims;
    full_dims.Add(-1);
    for (int i = 0; i < dims.size(); ++i) {
      full_dims.Add(dims[i]);
    }

    bool succ = (model_shape->rank_ == (size_t)full_dims.size());
    if (succ) {
      for (int i = 0; i < full_dims.size(); ++i) {
        const int64_t model_dim = model_shape->dims_[i];
        if (compare_exact || (model_dim != -1)) {
          succ &= (model_dim == full_dims[i]);
        }
      }
    }

    if (!succ) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "model '" + model_name + "', tensor '" + tensor_name +
              "': the model expects " + std::to_string(model_shape->rank_) +
              " dimensions (shape " + ShapeToString(model_shape) +
              ") but the model configuration specifies " +
              std::to_string(full_dims.size()) +
              " dimensions (an initial batch dimension because max_batch_size "
              "> 0 followed by the explicit tensor shape, making complete "
              "shape " +
              DimsListToString(full_dims) + ")");
    }
  } else {
    // ! supports_batching
    bool succ = (model_shape->rank_ == (size_t)dims.size());
    if (succ) {
      for (int i = 0; i < dims.size(); ++i) {
        const int64_t model_dim = model_shape->dims_[i];
        if (compare_exact || (model_dim != -1)) {
          succ &= (model_dim == dims[i]);
        }
      }
    }

    if (!succ) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "model '" + model_name + "', tensor '" + tensor_name +
              "': the model expects " + std::to_string(model_shape->rank_) +
              " dimensions (shape " + ShapeToString(model_shape) +
              ") but the model configuration specifies " +
              std::to_string(dims.size()) + " dimensions (shape " +
              DimsListToString(dims) + ")");
    }
  }

  return Status::Success;
}

const TRTISTF_IO*
FindIOByName(const TRTISTF_IOList* ios, const std::string& name)
{
  for (const TRTISTF_IOList* itr = ios; itr != nullptr; itr = itr->next_) {
    if (itr->io_->name_ == name) {
      return itr->io_;
    }
  }

  return nullptr;
}

std::string
ShapeToString(const TRTISTF_Shape* shape, const size_t start_idx)
{
  std::string str("[");
  for (size_t idx = start_idx; idx < shape->rank_; idx++) {
    if (idx > start_idx) {
      str += ",";
    }

    str += std::to_string(shape->dims_[idx]);
  }

  str += "]";
  return str;
}

bool
CompareDataType(TRTISTF_DataType model_dtype, DataType dtype)
{
  DataType cdtype = ConvertDataType(model_dtype);
  if (cdtype == DataType::TYPE_INVALID) {
    return false;
  }

  return dtype == cdtype;
}

DataType
ConvertDataType(TRTISTF_DataType dtype)
{
  switch (dtype) {
    case TRTISTF_DataType::TRTISTF_TYPE_INVALID:
      return DataType::TYPE_INVALID;
    case TRTISTF_DataType::TRTISTF_TYPE_BOOL:
      return DataType::TYPE_BOOL;
    case TRTISTF_DataType::TRTISTF_TYPE_UINT8:
      return DataType::TYPE_UINT8;
    case TRTISTF_DataType::TRTISTF_TYPE_UINT16:
      return DataType::TYPE_UINT16;
    case TRTISTF_DataType::TRTISTF_TYPE_UINT32:
      return DataType::TYPE_UINT32;
    case TRTISTF_DataType::TRTISTF_TYPE_UINT64:
      return DataType::TYPE_UINT64;
    case TRTISTF_DataType::TRTISTF_TYPE_INT8:
      return DataType::TYPE_INT8;
    case TRTISTF_DataType::TRTISTF_TYPE_INT16:
      return DataType::TYPE_INT16;
    case TRTISTF_DataType::TRTISTF_TYPE_INT32:
      return DataType::TYPE_INT32;
    case TRTISTF_DataType::TRTISTF_TYPE_INT64:
      return DataType::TYPE_INT64;
    case TRTISTF_DataType::TRTISTF_TYPE_FP16:
      return DataType::TYPE_FP16;
    case TRTISTF_DataType::TRTISTF_TYPE_FP32:
      return DataType::TYPE_FP32;
    case TRTISTF_DataType::TRTISTF_TYPE_FP64:
      return DataType::TYPE_FP64;
    case TRTISTF_DataType::TRTISTF_TYPE_STRING:
      return DataType::TYPE_STRING;
    default:
      break;
  }

  return DataType::TYPE_INVALID;
}

TRTISTF_DataType
ConvertDataType(DataType dtype)
{
  switch (dtype) {
    case DataType::TYPE_INVALID:
      return TRTISTF_DataType::TRTISTF_TYPE_INVALID;
    case DataType::TYPE_BOOL:
      return TRTISTF_DataType::TRTISTF_TYPE_BOOL;
    case DataType::TYPE_UINT8:
      return TRTISTF_DataType::TRTISTF_TYPE_UINT8;
    case DataType::TYPE_UINT16:
      return TRTISTF_DataType::TRTISTF_TYPE_UINT16;
    case DataType::TYPE_UINT32:
      return TRTISTF_DataType::TRTISTF_TYPE_UINT32;
    case DataType::TYPE_UINT64:
      return TRTISTF_DataType::TRTISTF_TYPE_UINT64;
    case DataType::TYPE_INT8:
      return TRTISTF_DataType::TRTISTF_TYPE_INT8;
    case DataType::TYPE_INT16:
      return TRTISTF_DataType::TRTISTF_TYPE_INT16;
    case DataType::TYPE_INT32:
      return TRTISTF_DataType::TRTISTF_TYPE_INT32;
    case DataType::TYPE_INT64:
      return TRTISTF_DataType::TRTISTF_TYPE_INT64;
    case DataType::TYPE_FP16:
      return TRTISTF_DataType::TRTISTF_TYPE_FP16;
    case DataType::TYPE_FP32:
      return TRTISTF_DataType::TRTISTF_TYPE_FP32;
    case DataType::TYPE_FP64:
      return TRTISTF_DataType::TRTISTF_TYPE_FP64;
    case DataType::TYPE_STRING:
      return TRTISTF_DataType::TRTISTF_TYPE_STRING;
    default:
      break;
  }

  return TRTISTF_DataType::TRTISTF_TYPE_INVALID;
}

}}  // namespace nvidia::inferenceserver
