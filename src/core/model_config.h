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
#pragma once

#include <stdint.h>
#include "src/core/model_config.pb.h"

namespace nvidia { namespace inferenceserver {

/// The type for the correlation ID in an inference request. This must
/// match the protobuf type used in InferRequestHeader.correlation_id.
using CorrelationID = uint64_t;

/// The type for a repeated dims field (used for shape)
using DimsList = ::google::protobuf::RepeatedField<::google::protobuf::int64>;

/// The value for a dimension in a shape that indicates that that
/// dimension can take on any size.
constexpr int WILDCARD_DIM = -1;

/// Enumeration for the different platform types.
enum Platform {
  PLATFORM_UNKNOWN = 0,
  PLATFORM_TENSORRT_PLAN = 1,
  PLATFORM_TENSORFLOW_GRAPHDEF = 2,
  PLATFORM_TENSORFLOW_SAVEDMODEL = 3,
  PLATFORM_CAFFE2_NETDEF = 4,
  PLATFORM_CUSTOM = 5
};

/// Get the number of elements in a shape.
/// \param dims The shape.
/// \return The number of elements, or -1 if the number of elements
/// cannot be determined because the shape contains one or more
/// wilcard dimensions.
int64_t GetElementCount(const DimsList& dims);

/// Get the number of elements in a shape.
/// \param dims The shape.
/// \return The number of elements, or -1 if the number of elements
/// cannot be determined because the shape contains one or more
/// wilcard dimensions.
int64_t GetElementCount(const std::vector<int64_t>& dims);

/// Get the number of elements in the shape of a model input.
/// \param mio The model input.
/// \return The number of elements, or -1 if the number of elements
/// cannot be determined because the shape contains one or more
/// wilcard dimensions.
int64_t GetElementCount(const ModelInput& mio);

/// Get the number of elements in the shape of a model output.
/// \param mio The model output.
/// \return The number of elements, or -1 if the number of elements
/// cannot be determined because the shape contains one or more
/// wilcard dimensions.
int64_t GetElementCount(const ModelOutput& mio);

/// Are values of a datatype fixed-size, or variable-sized.
/// \param dtype The data-type.
/// \return True if datatype values are fixed-sized, false if
/// variable-sized.
bool IsFixedSizeDataType(const DataType dtype);

/// Get the size of objects of a given datatype in bytes.
/// \param dtype The data-type.
/// \return The size, in bytes, of objects of the datatype, or 0 if
/// size cannot be determine (for example, values of type TYPE_STRING
/// have variable length and so size cannot be determine just from the
/// type).
size_t GetDataTypeByteSize(const DataType dtype);

/// Get the size, in bytes, of a tensor based on datatype and
/// shape.
/// \param dtype The data-type.
/// \param dims The shape.
/// \return The size, in bytes, of the corresponding tensor, or 0 if
/// unable to determine the size.
uint64_t GetByteSize(const DataType& dtype, const DimsList& dims);

/// Get the size, in bytes, of a tensor based on datatype and
/// shape.
/// \param dtype The data-type.
/// \param dims The shape.
/// \return The size, in bytes, of the corresponding tensor, or 0 if
/// unable to determine the size.
uint64_t GetByteSize(const DataType& dtype, const std::vector<int64_t>& dims);

/// Get the size, in bytes, of a tensor based on ModelInput.
/// \param mio The ModelInput protobuf.
/// \return The size, in bytes, of the corresponding tensor, or 0 if
/// unable to determine the size.
uint64_t GetByteSize(const ModelInput& mio);

/// Get the size, in bytes, of a tensor based on ModelOutput.
/// \param mio The ModelOutput protobuf.
/// \return The size, in bytes, of the corresponding tensor, or 0 if
/// unable to determine the size.
uint64_t GetByteSize(const ModelOutput& mio);

/// Get the Platform value for a platform name.
/// \param platform_name The platform name.
/// \return The Platform or Platform::UNKNOWN if the platform string
/// is not recognized.
Platform GetPlatform(const std::string& platform_name);

/// Get the CPU thread nice level associate with a model
/// configuration's priority.
/// \param config The model configuration.
/// \return The nice level.
int GetCpuNiceLevel(const ModelConfig& config);

/// Compare two model configuration shapes for equality. Wildcard
/// dimensions (that is, dimensions with size WILDCARD_DIM) are
/// compared literally so that to be equal the two shapes must both
/// specify WILDCARD_DIM in the same dimensions.
/// \params dims0 The first shape.
/// \params dims1 The second shape.
/// \return True if the shapes are equal, false if not equal.
bool CompareDims(const DimsList& dims0, const DimsList& dims1);

/// Compare two model configuration shapes for equality. Wildcard
/// dimensions (that is, dimensions with size WILDCARD_DIM) are
/// allowed to match with any value. So, a dimension in one shape
/// specified as WILDCARD_DIM will always match the same dimension in
/// the other shape.
/// \params dims0 The first shape.
/// \params dims1 The second shape.
/// \return True if the shapes are equal, false if not equal.
bool CompareDimsWithWildcard(const DimsList& dims0, const DimsList& dims1);

}}  // namespace nvidia::inferenceserver
