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

#include <NvInfer.h>
#include "src/core/model_config.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

// The memory layouts for i/o tensors
enum class MemoryFormat {
  // Row major linear format.
  LINEAR,
  // Two wide channel vectorized row major format.
  CHW2,
  // Four wide channel vectorized row major format.
  CHW4,
  // Eight channel format where C is padded to a multiple of 8.
  HCW8,
  // Sixteen wide channel vectorized row major format.
  CHW16,
  // Thirty-two wide channel vectorized row major format.
  CHW32,
  // Invalid Memory format
  INVALID
};

MemoryFormat ConvertTrtFmtToFmt(nvinfer1::TensorFormat trt_fmt);

const std::string MemoryFormat_Name(MemoryFormat fmt);

DataType ConvertTrtTypeToDataType(nvinfer1::DataType trt_type);

std::pair<bool, nvinfer1::DataType> ConvertDataTypeToTrtType(
    const DataType& dtype);

bool CompareDims(const nvinfer1::Dims& model_dims, const DimsList& dims);

Status ValidateDimension(
    const nvinfer1::Dims& this_dims, const nvinfer1::Dims& min_dims,
    const nvinfer1::Dims& max_dims, const bool skip_first_dimension);

Status ValidateDimension(
    const DimsList& this_dims, const nvinfer1::Dims& min_dims,
    const nvinfer1::Dims& max_dims, const bool skip_first_dimension);

Status CompareDimsSupported(
    const std::string& model_name, const std::string& tensor_name,
    const nvinfer1::Dims& model_dims, const DimsList& dims,
    const bool supports_batching, const bool is_dynamic);

Status ValidateControlDimsDynamic(
    const nvinfer1::Dims& dims, const bool support_batching);

Status MaximumDims(
    const nvinfer1::Dims& max_profile_dims, const DimsList& dims,
    const bool support_batching, const int max_batch_size,
    std::vector<int64_t>* maximum_dims);

void DimsToDimVec(const nvinfer1::Dims& model_dims, std::vector<int64_t>* dims);

bool DimVecToDims(const std::vector<int64_t>& dim_vec, nvinfer1::Dims* dims);

bool ContainsWildcard(const nvinfer1::Dims& dims);

const std::string DimsDebugString(const nvinfer1::Dims& dims);

}}  // namespace nvidia::inferenceserver
