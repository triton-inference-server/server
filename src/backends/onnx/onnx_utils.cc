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
  RETURN_IF_ORT_ERROR(OrtCreateDefaultAllocator(&allocator));
  OrtStatus* onnx_status = nullptr;
  for (size_t i = 0; i < num_nodes; i++) {
    char* node_name;
    if (is_input) {
      onnx_status =
        OrtSessionGetInputName(session, i, allocator, &node_name);
    } else {
      onnx_status =
        OrtSessionGetOutputName(session, i, allocator, &node_name);
    }
    
    if (onnx_status != nullptr) {
      break;
    }
    names.emplace(node_name);
  }
  OrtReleaseAllocator(allocator);
  RETURN_IF_ORT_ERROR(onnx_status);

  return Status::Success;
}

} // namespace

DataType
ConvertDatatype(ONNXTensorElementDataType onnx_type)
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
ConvertDataType(DataType data_type)
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

}}  // namespace nvidia::inferenceserver
