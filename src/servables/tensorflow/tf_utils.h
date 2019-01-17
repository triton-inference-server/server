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

#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace nvidia { namespace inferenceserver {

/// \return true if a TensorFlow shape exactly matches a model
/// configuration shape. Dimensions with variable size are represented
/// by -1 in both the TensorFlow shape and the model configuration
/// shape and these must match as well.
/// \param supports_batching If True then the configuration expects
/// the model to support batching and so the shape must have the
/// appropriate batch dimension.
bool CompareDimsExact(
    const tensorflow::TensorShapeProto& model_shape, const DimsList& dims,
    const bool supports_batching);

/// \return true if a TensorFlow shape can support a model
/// configuration shape. Dimensions with variable size in the
/// TensorFlow shape can support any size in the corresponding model
/// configuration shape dimension. Dimensions with variable size in
/// the model configuration shape must be variable size in the
/// TensorFlow shape. All fixed-sized dimensions must match exactly.
/// \param supports_batching If True then the configuration expects
/// the model to support batching and so the shape must have the
/// appropriate batch dimension.
bool CompareDimsSupported(
    const tensorflow::TensorShapeProto& model_shape, const DimsList& dims,
    const bool supports_batching);

/// \return true if a TensorFlow data-type matches a model
/// configuration data-type.
bool CompareDataType(tensorflow::DataType model_dtype, DataType dtype);

/// \return the string representation of a model configuration shape.
const std::string DimsDebugString(const DimsList& dims);

/// \return the string representation of a TensorFlow shape.
const std::string DimsDebugString(const tensorflow::TensorShapeProto& dims);

/// \return the TensorFlow data-type that corresponds to a model
/// configuration data-type.
tensorflow::DataType ConvertDataType(DataType dtype);

/// \return the model configuration data-type that corresponds to a
/// TensorFlow data-type.
DataType ConvertDataType(tensorflow::DataType dtype);

}}  // namespace nvidia::inferenceserver
