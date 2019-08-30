// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "src/backends/onnx/onnx_utils.h"

namespace nvidia { namespace inferenceserver {

namespace {

Status
InputOutputNames(
    OrtSession* session, bool is_input, std::set<std::string>& names)
{
  names.clear();

  size_t num_nodes;
  if (is_input) {
    RETURN_IF_ORT_ERROR(OrtSessionGetInputCount(session, &num_nodes));
  } else {
    RETURN_IF_ORT_ERROR(OrtSessionGetOutputCount(session, &num_nodes));
  }

  // iterate over all input / output nodes
  OrtAllocator* allocator;
  RETURN_IF_ORT_ERROR(OrtGetAllocatorWithDefaultOptions(&allocator));
  OrtStatus* onnx_status = nullptr;
  for (size_t i = 0; i < num_nodes; i++) {
    char* node_name;
    if (is_input) {
      onnx_status = OrtSessionGetInputName(session, i, allocator, &node_name);
    } else {
      onnx_status = OrtSessionGetOutputName(session, i, allocator, &node_name);
    }

    if (onnx_status != nullptr) {
      break;
    }
    names.emplace(node_name);
  }
  RETURN_IF_ORT_ERROR(onnx_status);

  return Status::Success;
}

Status
InputOutputInfos(
    OrtSession* session, OrtAllocator* allocator, bool is_input,
    OnnxTensorInfoMap& infos)
{
  infos.clear();

  size_t num_nodes;
  if (is_input) {
    RETURN_IF_ORT_ERROR(OrtSessionGetInputCount(session, &num_nodes));
  } else {
    RETURN_IF_ORT_ERROR(OrtSessionGetOutputCount(session, &num_nodes));
  }

  // iterate over all nodes
  for (size_t i = 0; i < num_nodes; i++) {
    char* name;
    if (is_input) {
      RETURN_IF_ORT_ERROR(OrtSessionGetInputName(session, i, allocator, &name));
    } else {
      RETURN_IF_ORT_ERROR(
          OrtSessionGetOutputName(session, i, allocator, &name));
    }

    OrtTypeInfo* typeinfo;
    if (is_input) {
      RETURN_IF_ORT_ERROR(OrtSessionGetInputTypeInfo(session, i, &typeinfo));
    } else {
      RETURN_IF_ORT_ERROR(OrtSessionGetOutputTypeInfo(session, i, &typeinfo));
    }

    OrtResourceWrapper<OrtTypeInfo*> typeinfo_wrapper(
        typeinfo, &OrtReleaseTypeInfo);
    const OrtTensorTypeAndShapeInfo* tensor_info;
    RETURN_IF_ORT_ERROR(OrtCastTypeInfoToTensorInfo(typeinfo, &tensor_info));

    ONNXTensorElementDataType type;
    RETURN_IF_ORT_ERROR(OrtGetTensorElementType(tensor_info, &type));

    size_t num_dims;
    RETURN_IF_ORT_ERROR(OrtGetDimensionsCount(tensor_info, &num_dims));

    std::vector<int64_t> dims(num_dims);
    RETURN_IF_ORT_ERROR(
        OrtGetDimensions(tensor_info, (int64_t*)dims.data(), num_dims));

    infos.emplace(name, OnnxTensorInfo(type, dims));
  }

  return Status::Success;
}

}  // namespace

std::string
OnnxDataTypeName(ONNXTensorElementDataType onnx_type)
{
  switch (onnx_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "FLOAT";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return "UINT8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return "INT8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return "UINT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return "INT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return "INT32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "INT64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      return "STRING";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return "BOOL";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return "FLOAT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return "DOUBLE";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return "UINT32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return "UINT64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      return "COMPLEX64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      return "COMPLEX64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return "BFLOAT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
    default:
      break;
  }

  return "UNDEFINED";
}

DataType
ConvertFromOnnxDataType(ONNXTensorElementDataType onnx_type)
{
  switch (onnx_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      // maps to c type float (4 bytes)
      return TYPE_FP32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return TYPE_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return TYPE_INT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return TYPE_UINT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return TYPE_INT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return TYPE_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return TYPE_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      return TYPE_STRING;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return TYPE_BOOL;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return TYPE_FP16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      // maps to c type double (8 bytes)
      return TYPE_FP64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return TYPE_UINT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return TYPE_UINT64;
    // The following types are not supported:
    // complex with float32 real and imaginary components
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    // complex with float64 real and imaginary components
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
    // Non-IEEE floating-point format based on IEEE754 single-precision
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
    default:
      break;
  }

  return TYPE_INVALID;
}

ONNXTensorElementDataType
ConvertToOnnxDataType(DataType data_type)
{
  switch (data_type) {
    case TYPE_UINT8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case TYPE_UINT16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    case TYPE_UINT32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    case TYPE_UINT64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    case TYPE_INT8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case TYPE_INT16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    case TYPE_INT32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case TYPE_INT64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case TYPE_FP16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case TYPE_FP32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case TYPE_FP64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case TYPE_STRING:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    default:
      break;
  }

  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

Status
InputNames(OrtSession* session, std::set<std::string>& names)
{
  return InputOutputNames(session, true, names);
}

Status
OutputNames(OrtSession* session, std::set<std::string>& names)
{
  return InputOutputNames(session, false, names);
}

Status
InputInfos(
    OrtSession* session, OrtAllocator* allocator, OnnxTensorInfoMap& infos)
{
  return InputOutputInfos(session, allocator, true, infos);
}

Status
OutputInfos(
    OrtSession* session, OrtAllocator* allocator, OnnxTensorInfoMap& infos)
{
  return InputOutputInfos(session, allocator, false, infos);
}

Status
CompareDimsSupported(
    const std::string& model_name, const std::string& tensor_name,
    const std::vector<int64_t>& model_shape, const DimsList& dims,
    const int max_batch_size)
{
  // If the model configuration expects batching support in the model,
  // then the onnx shape first dimension must be -1.
  const bool supports_batching = (max_batch_size > 0);
  if (supports_batching &&
      ((model_shape.size() == 0) || (model_shape[0] != -1))) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "unable to load model '" + model_name +
            "', model configuration supports batching but first dimension of "
            "tensor '" +
            tensor_name +
            "' expected by framework is not a variable-size batch dimension: " +
            DimsListToString(model_shape) +
            " whereas model configuration shape is: " + DimsListToString(dims));
  }

  const int nonbatch_start_idx = (supports_batching ? 1 : 0);
  std::vector<int64_t> debatched_model_shape;
  for (size_t i = nonbatch_start_idx; i < model_shape.size(); i++) {
    debatched_model_shape.push_back(model_shape[i]);
  }

  // Tensor rank in configuration must match what framework expects.
  if (debatched_model_shape.size() != (size_t)dims.size()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "unable to load model '" + model_name + "', tensor '" + tensor_name +
            "' shape expected by framework " +
            DimsListToString(debatched_model_shape) +
            " doesn't match model configuration shape " +
            DimsListToString(dims));
  }

  for (int i = 0; i < dims.size(); ++i) {
    int64_t model_dim = debatched_model_shape[i];
    if (model_dim == -1) {
      continue;
    }

    if (model_dim != dims[i]) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "unable to load model '" + model_name + "', tensor '" + tensor_name +
              "' shape expected by framework " +
              DimsListToString(debatched_model_shape) +
              " doesn't match model configuration shape " +
              DimsListToString(dims));
    }
  }

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
