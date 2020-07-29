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

#include <google/protobuf/any.pb.h>
#include <stdint.h>
#include "src/core/model_config.pb.h"
#include "src/core/tritonserver.h"

namespace nvidia { namespace inferenceserver {

/// The type for a repeated dims field (used for shape).
using DimsList = ::google::protobuf::RepeatedField<::google::protobuf::int64>;

/// The type for the metric_tags map.
using MetricTagsMap = ::google::protobuf::Map<
    ::google::protobuf::string, ::google::protobuf::string>;

// Map from backend name to list of setting=value pairs of cmdline
// settings for the backend.
using BackendCmdlineConfig = std::vector<std::pair<std::string, std::string>>;
using BackendCmdlineConfigMap =
    std::unordered_map<std::string, BackendCmdlineConfig>;

/// The type from platform name to the backend configuration for that
/// platform. The configuration is determined primarily by
/// command-line options and is the same for every backend created for
/// a given platform.
struct BackendConfig {
};
using BackendConfigMap =
    std::unordered_map<std::string, std::shared_ptr<BackendConfig>>;

/// The value for a dimension in a shape that indicates that that
/// dimension can take on any size.
constexpr int WILDCARD_DIM = -1;

/// Enumeration for the different platform types.
enum Platform {
  PLATFORM_UNKNOWN = 0,
#ifdef TRITON_ENABLE_TENSORRT
  PLATFORM_TENSORRT_PLAN = 1,
#endif  // TRITON_ENABLE_TENSORRT
#ifdef TRITON_ENABLE_TENSORFLOW
  PLATFORM_TENSORFLOW_GRAPHDEF = 2,
  PLATFORM_TENSORFLOW_SAVEDMODEL = 3,
#endif  // TRITON_ENABLE_TENSORFLOW
#ifdef TRITON_ENABLE_CAFFE2
  PLATFORM_CAFFE2_NETDEF = 4,
#endif  // TRITON_ENABLE_CAFFE2
#ifdef TRITON_ENABLE_CUSTOM
  PLATFORM_CUSTOM = 5,
#endif  // TRITON_ENABLE_CUSTOM
#ifdef TRITON_ENABLE_ENSEMBLE
  PLATFORM_ENSEMBLE = 6,
#endif  // TRITON_ENABLE_ENSEMBLE
#ifdef TRITON_ENABLE_ONNXRUNTIME
  PLATFORM_ONNXRUNTIME_ONNX = 7,
#endif  // TRITON_ENABLE_ONNXRUNTIME
#ifdef TRITON_ENABLE_PYTORCH
  PLATFORM_PYTORCH_LIBTORCH = 8
#endif  // TRITON_ENABLE_PYTORCH
};

/// Enumeration for the different backend types.
enum BackendType {
  BACKEND_TYPE_UNKNOWN = 0,
#ifdef TRITON_ENABLE_TENSORRT
  BACKEND_TYPE_TENSORRT = 1,
#endif  // TRITON_ENABLE_TENSORRT
#ifdef TRITON_ENABLE_TENSORFLOW
  BACKEND_TYPE_TENSORFLOW = 2,
#endif  // TRITON_ENABLE_TENSORFLOW
#ifdef TRITON_ENABLE_ONNXRUNTIME
  BACKEND_TYPE_ONNXRUNTIME = 3,
#endif  // TRITON_ENABLE_ONNXRUNTIME
#ifdef TRITON_ENABLE_PYTORCH
  BACKEND_TYPE_PYTORCH = 4
#endif  // TRITON_ENABLE_PYTORCH
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
/// \return The size, in bytes, of the corresponding tensor, or -1 if
/// unable to determine the size.
int64_t GetByteSize(const DataType& dtype, const DimsList& dims);

/// Get the size, in bytes, of a tensor based on datatype and
/// shape.
/// \param dtype The data-type.
/// \param dims The shape.
/// \return The size, in bytes, of the corresponding tensor, or -1 if
/// unable to determine the size.
int64_t GetByteSize(const DataType& dtype, const std::vector<int64_t>& dims);

/// Get the size, in bytes, of a tensor based on batch-size, datatype
/// and shape. A tensor that has empty shape [] and non-zero
/// batch-size is sized as a tensor with shape [ batch-size ].
/// \param batch_size The batch-size. May be 0 to indicate no
/// batching.
/// \param dtype The data-type.
/// \param dims The shape.
/// \return The size, in bytes, of the corresponding tensor, or -1 if
/// unable to determine the size.
int64_t GetByteSize(
    const int batch_size, const DataType& dtype, const DimsList& dims);

/// Get the size, in bytes, of a tensor based on batch-size, datatype
/// and shape. A tensor that has empty shape [] and non-zero
/// batch-size is sized as a tensor with shape [ batch-size ].
/// \param batch_size The batch-size. May be 0 to indicate no
/// batching.
/// \param dtype The data-type.
/// \param dims The shape.
/// \return The size, in bytes, of the corresponding tensor, or -1 if
/// unable to determine the size.
int64_t GetByteSize(
    const int batch_size, const DataType& dtype,
    const std::vector<int64_t>& dims);

/// Get the size, in bytes, of a tensor based on ModelInput.
/// \param mio The ModelInput protobuf.
/// \return The size, in bytes, of the corresponding tensor, or -1 if
/// unable to determine the size.
int64_t GetByteSize(const ModelInput& mio);

/// Get the size, in bytes, of a tensor based on ModelOutput.
/// \param mio The ModelOutput protobuf.
/// \return The size, in bytes, of the corresponding tensor, or -1 if
/// unable to determine the size.
int64_t GetByteSize(const ModelOutput& mio);

/// Get the Platform value for a platform name.
/// \param platform_name The platform name.
/// \return The Platform or Platform::UNKNOWN if the platform string
/// is not recognized.
Platform GetPlatform(const std::string& platform_name);

/// Get the BackendType value for a platform name.
/// \param platform_name The platform name.
/// \return The BackendType or BackendType::UNKNOWN if the platform string
/// is not recognized.
BackendType GetBackendTypeFromPlatform(const std::string& platform_name);

/// Get the BackendType value for a backend name.
/// \param backend_name The backend name.
/// \return The BackendType or BackendType::UNKNOWN if the platform string
/// is not recognized.
BackendType GetBackendType(const std::string& backend_name);

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
/// compared literally so that to be equal the two shapes must both
/// specify WILDCARD_DIM in the same dimensions.
/// \params dims0 The first shape.
/// \params dims1 The second shape.
/// \return True if the shapes are equal, false if not equal.
bool CompareDims(
    const std::vector<int64_t>& dims0, const std::vector<int64_t>& dims1);

/// Compare two model configuration shapes for equality. Wildcard
/// dimensions (that is, dimensions with size WILDCARD_DIM) are
/// allowed to match with any value. So, a dimension in one shape
/// specified as WILDCARD_DIM will always match the same dimension in
/// the other shape.
/// \params dims0 The first shape.
/// \params dims1 The second shape.
/// \return True if the shapes are equal, false if not equal.
bool CompareDimsWithWildcard(const DimsList& dims0, const DimsList& dims1);

/// Compare two model configuration shapes for equality. Wildcard
/// dimensions (that is, dimensions with size WILDCARD_DIM) are
/// allowed to match with any value. So, a dimension in one shape
/// specified as WILDCARD_DIM will always match the same dimension in
/// the other shape.
/// \params dims0 The first shape.
/// \params dims1 The second shape.
/// \return True if the shapes are equal, false if not equal.
bool CompareDimsWithWildcard(
    const DimsList& dims0, const std::vector<int64_t>& dims1);

/// Convert a DimsList to string representation.
/// \param dims The DimsList to be converted.
/// \return String representation of the DimsList in pattern
/// "[d0,d1,...,dn]"
std::string DimsListToString(const DimsList& dims);

/// Convert a vector representing a shape to string representation.
/// \param dims The vector of dimensions to be converted.
/// \return String representation of the vector in pattern
/// "[d0,d1,...,dn]"
std::string DimsListToString(
    const std::vector<int64_t>& dims, const int start_idx = 0);

/// Get the server protocol string representation of a datatype.
/// \param dtype The data type.
/// \return The string representation.
const char* DataTypeToProtocolString(const DataType dtype);

/// Get the datatype corresponding to a server protocol string
/// representation of a datatype.
/// \param dtype string representation.
/// \return The data type.
DataType ProtocolStringToDataType(const std::string& dtype);

/// Get the datatype corresponding to a server protocol string
/// representation of a datatype.
/// \param dtype Pointer to string.
/// \param len Length of the string.
/// \return The data type.
DataType ProtocolStringToDataType(const char* dtype, size_t len);

/// Get the Triton server data type corresponding to a data type.
/// \param dtype The data type.
/// \return The Triton server data type.
TRITONSERVER_DataType DataTypeToTriton(const DataType dtype);

/// Get the data type corresponding to a Triton server data type.
/// \param dtype The Triton server data type.
/// \return The data type.
DataType TritonToDataType(const TRITONSERVER_DataType dtype);

}}  // namespace nvidia::inferenceserver
