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

#pragma once

#include <NvInfer.h>
#include "src/core/model_config.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

bool UseTensorRTv2API(const std::shared_ptr<nvinfer1::ICudaEngine>& engine);

inference::DataType ConvertTrtTypeToDataType(nvinfer1::DataType trt_type);

std::pair<bool, nvinfer1::DataType> ConvertDataTypeToTrtType(
    const inference::DataType& dtype);

bool CompareDims(const nvinfer1::Dims& model_dims, const DimsList& dims);
bool CompareDims(const nvinfer1::Dims& ldims, const nvinfer1::Dims& rdims);

Status ValidateDimension(
    const nvinfer1::Dims& this_dims, const nvinfer1::Dims& min_dims,
    const nvinfer1::Dims& max_dims, const bool skip_first_dimension);

template <typename T>
Status ValidateDimension(
    const T& this_dims, const nvinfer1::Dims& min_dims,
    const nvinfer1::Dims& max_dims, const bool skip_first_dimension);

Status CompareDimsSupported(
    const std::string& model_name, const std::string& tensor_name,
    const nvinfer1::Dims& model_dims, const DimsList& dims,
    const bool supports_batching, const bool contains_explicit_batch,
    const bool compare_exact);

Status CompareShapeDimsSupported(
    const std::string& model_name, const std::string& tensor_name,
    const nvinfer1::Dims& model_dims, const DimsList& dims,
    const bool supports_batching);

Status ValidateControlDimsDynamic(
    const nvinfer1::Dims& dims, const bool support_batching);

Status ValidateShapeValues(
    const std::vector<int32_t>& request_shape_values,
    const int32_t* min_shape_values, const int32_t* max_shape_values,
    size_t nb_shape_values, const bool support_batching);

Status MaximumDims(
    const nvinfer1::Dims& max_profile_dims, const DimsList& dims,
    const bool support_batching, const int max_batch_size,
    std::vector<int64_t>* maximum_dims);

void DimsToDimVec(const nvinfer1::Dims& model_dims, std::vector<int64_t>* dims);

bool DimVecToDims(const std::vector<int64_t>& dim_vec, nvinfer1::Dims* dims);

int64_t GetElementCount(const nvinfer1::Dims& dims);

bool ContainsWildcard(const nvinfer1::Dims& dims);

bool ContainsWildcardAtExplicitBatchDim(const nvinfer1::Dims& dims);

const std::string DimsDebugString(const nvinfer1::Dims& dims);

//
// Templates
//

template <typename T>
Status
ValidateDimension(
    const T& this_dims, const nvinfer1::Dims& min_dims,
    const nvinfer1::Dims& max_dims, const bool skip_first_dimension)
{
  const int nonbatch_start_idx = (skip_first_dimension ? 1 : 0);
  if (int(this_dims.size() + nonbatch_start_idx) != max_dims.nbDims) {
    return Status(
        Status::Code::INTERNAL,
        "model expected " +
            std::to_string(max_dims.nbDims - nonbatch_start_idx) +
            " dimensions but received " + std::to_string(this_dims.size()) +
            " dimensions");
  }

  for (int i = 0; i < int(this_dims.size()); i++) {
    if (this_dims[i] == -1) {
      continue;
    }
    if (this_dims[i] < min_dims.d[i + nonbatch_start_idx] ||
        this_dims[i] > max_dims.d[i + nonbatch_start_idx]) {
      return Status(
          Status::Code::INTERNAL,
          "model expected the shape of dimension " + std::to_string(i) +
              " to be between " +
              std::to_string(min_dims.d[i + nonbatch_start_idx]) + " and " +
              std::to_string(max_dims.d[i + nonbatch_start_idx]) +
              " but received " + std::to_string(this_dims[i]));
    }
  }
  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
