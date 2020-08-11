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

#include "src/backends/backend/onnxruntime/onnx_utils.h"

#include "src/backends/backend/examples/backend_utils.h"

namespace nib = nvidia::inferenceserver::backend;

namespace triton { namespace backend { namespace onnxruntime {

const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

namespace {

std::string
OnnxTypeName(ONNXType onnx_type)
{
  switch (onnx_type) {
    case ONNX_TYPE_TENSOR:
      return "ONNX_TYPE_TENSOR";
    case ONNX_TYPE_SEQUENCE:
      return "ONNX_TYPE_SEQUENCE";
    case ONNX_TYPE_MAP:
      return "ONNX_TYPE_MAP";
    case ONNX_TYPE_OPAQUE:
      return "ONNX_TYPE_OPAQUE";
    case ONNX_TYPE_SPARSETENSOR:
      return "ONNX_TYPE_SPARSETENSOR";
    case ONNX_TYPE_UNKNOWN:
    default:
      break;
  }

  return "ONNX_TYPE_UNKNOWN";
}

TRITONSERVER_Error*
InputOutputNames(
    OrtSession* session, bool is_input, std::set<std::string>& names)
{
  names.clear();

  size_t num_nodes;
  if (is_input) {
    RETURN_IF_ORT_ERROR(ort_api->SessionGetInputCount(session, &num_nodes));
  } else {
    RETURN_IF_ORT_ERROR(ort_api->SessionGetOutputCount(session, &num_nodes));
  }

  // iterate over all input / output nodes
  OrtAllocator* allocator;
  RETURN_IF_ORT_ERROR(ort_api->GetAllocatorWithDefaultOptions(&allocator));
  OrtStatus* onnx_status = nullptr;
  for (size_t i = 0; i < num_nodes; i++) {
    char* node_name;
    if (is_input) {
      onnx_status =
          ort_api->SessionGetInputName(session, i, allocator, &node_name);
    } else {
      onnx_status =
          ort_api->SessionGetOutputName(session, i, allocator, &node_name);
    }

    if (onnx_status != nullptr) {
      break;
    }
    names.emplace(node_name);
  }
  RETURN_IF_ORT_ERROR(onnx_status);

  return nullptr;  // success
}

TRITONSERVER_Error*
InputOutputInfos(
    OrtSession* session, OrtAllocator* allocator, bool is_input,
    OnnxTensorInfoMap& infos)
{
  infos.clear();

  size_t num_nodes;
  if (is_input) {
    RETURN_IF_ORT_ERROR(ort_api->SessionGetInputCount(session, &num_nodes));
  } else {
    RETURN_IF_ORT_ERROR(ort_api->SessionGetOutputCount(session, &num_nodes));
  }

  // iterate over all nodes
  for (size_t i = 0; i < num_nodes; i++) {
    char* name;
    if (is_input) {
      RETURN_IF_ORT_ERROR(
          ort_api->SessionGetInputName(session, i, allocator, &name));
    } else {
      RETURN_IF_ORT_ERROR(
          ort_api->SessionGetOutputName(session, i, allocator, &name));
    }

    OrtTypeInfo* typeinfo;
    if (is_input) {
      RETURN_IF_ORT_ERROR(
          ort_api->SessionGetInputTypeInfo(session, i, &typeinfo));
    } else {
      RETURN_IF_ORT_ERROR(
          ort_api->SessionGetOutputTypeInfo(session, i, &typeinfo));
    }

    std::unique_ptr<OrtTypeInfo, TypeInfoDeleter> typeinfo_wrapper(typeinfo);

    ONNXType onnx_type;
    RETURN_IF_ORT_ERROR(ort_api->GetOnnxTypeFromTypeInfo(typeinfo, &onnx_type));
    RETURN_ERROR_IF_TRUE(
        onnx_type != ONNX_TYPE_TENSOR, TRITONSERVER_ERROR_UNSUPPORTED,
        std::string("Unsupported ONNX Type '") + OnnxTypeName(onnx_type) +
            "' for I/O '" + name + "', expected '" +
            OnnxTypeName(ONNX_TYPE_TENSOR) + "'.");

    const OrtTensorTypeAndShapeInfo* tensor_info;
    RETURN_IF_ORT_ERROR(
        ort_api->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));

    ONNXTensorElementDataType type;
    RETURN_IF_ORT_ERROR(ort_api->GetTensorElementType(tensor_info, &type));

    size_t num_dims;
    RETURN_IF_ORT_ERROR(ort_api->GetDimensionsCount(tensor_info, &num_dims));

    std::vector<int64_t> dims(num_dims);
    RETURN_IF_ORT_ERROR(
        ort_api->GetDimensions(tensor_info, (int64_t*)dims.data(), num_dims));

    infos.emplace(name, OnnxTensorInfo(type, dims));
  }

  return nullptr;  // success
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

TRITONSERVER_DataType
ConvertFromOnnxDataType(ONNXTensorElementDataType onnx_type)
{
  switch (onnx_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      // maps to c type float (4 bytes)
      return TRITONSERVER_TYPE_FP32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return TRITONSERVER_TYPE_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return TRITONSERVER_TYPE_INT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return TRITONSERVER_TYPE_UINT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return TRITONSERVER_TYPE_INT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return TRITONSERVER_TYPE_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return TRITONSERVER_TYPE_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      return TRITONSERVER_TYPE_BYTES;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return TRITONSERVER_TYPE_BOOL;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return TRITONSERVER_TYPE_FP16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      // maps to c type double (8 bytes)
      return TRITONSERVER_TYPE_FP64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return TRITONSERVER_TYPE_UINT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return TRITONSERVER_TYPE_UINT64;
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

  return TRITONSERVER_TYPE_INVALID;
}

ONNXTensorElementDataType
ConvertToOnnxDataType(TRITONSERVER_DataType data_type)
{
  switch (data_type) {
    case TRITONSERVER_TYPE_UINT8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case TRITONSERVER_TYPE_UINT16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    case TRITONSERVER_TYPE_UINT32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    case TRITONSERVER_TYPE_UINT64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    case TRITONSERVER_TYPE_INT8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case TRITONSERVER_TYPE_INT16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    case TRITONSERVER_TYPE_INT32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case TRITONSERVER_TYPE_INT64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case TRITONSERVER_TYPE_FP16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case TRITONSERVER_TYPE_FP32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case TRITONSERVER_TYPE_FP64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case TRITONSERVER_TYPE_BYTES:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    case TRITONSERVER_TYPE_BOOL:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    default:
      break;
  }

  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

ONNXTensorElementDataType
ConvertToOnnxDataType(const std::string& data_type_str)
{
  TRITONSERVER_DataType data_type =
      TRITONSERVER_StringToDataType(data_type_str.c_str());
  return ConvertToOnnxDataType(data_type);
}

ONNXTensorElementDataType
ModelConfigDataTypeToOnnxDataType(const std::string& data_type_str)
{
  // Must start with "TYPE_".
  if (data_type_str.rfind("TYPE_", 0) != 0) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }

  const std::string dtype = data_type_str.substr(strlen("TYPE_"));

  if (dtype == "BOOL") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  } else if (dtype == "UINT8") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  } else if (dtype == "UINT16") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
  } else if (dtype == "UINT32") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
  } else if (dtype == "UINT64") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
  } else if (dtype == "INT8") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  } else if (dtype == "INT16") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
  } else if (dtype == "INT32") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  } else if (dtype == "INT64") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  } else if (dtype == "FP16") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  } else if (dtype == "FP32") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  } else if (dtype == "FP64") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  } else if (dtype == "STRING") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  }

  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

TRITONSERVER_Error*
InputNames(OrtSession* session, std::set<std::string>& names)
{
  return InputOutputNames(session, true, names);
}

TRITONSERVER_Error*
OutputNames(OrtSession* session, std::set<std::string>& names)
{
  return InputOutputNames(session, false, names);
}

TRITONSERVER_Error*
InputInfos(
    OrtSession* session, OrtAllocator* allocator, OnnxTensorInfoMap& infos)
{
  return InputOutputInfos(session, allocator, true, infos);
}

TRITONSERVER_Error*
OutputInfos(
    OrtSession* session, OrtAllocator* allocator, OnnxTensorInfoMap& infos)
{
  return InputOutputInfos(session, allocator, false, infos);
}

TRITONSERVER_Error*
CompareDimsSupported(
    const std::string& model_name, const std::string& tensor_name,
    const std::vector<int64_t>& model_shape, const std::vector<int64_t>& dims,
    const int max_batch_size, const bool compare_exact)
{
  // If the model configuration expects batching support in the model,
  // then the onnx shape first dimension must be -1.
  const bool supports_batching = (max_batch_size > 0);
  if (supports_batching) {
    RETURN_ERROR_IF_TRUE(
        (model_shape.size() == 0) || (model_shape[0] != -1),
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model '") + model_name + "', tensor '" + tensor_name +
            "': for the model to support batching the shape should have at "
            "least 1 dimension and the first dimension must be -1; but shape "
            "expected by the model is " +
            nib::ShapeToString(model_shape));

    std::vector<int64_t> full_dims;
    full_dims.reserve(1 + dims.size());
    full_dims.push_back(-1);
    full_dims.insert(full_dims.end(), dims.begin(), dims.end());

    bool succ = (model_shape.size() == (size_t)full_dims.size());
    if (succ) {
      for (size_t i = 0; i < full_dims.size(); ++i) {
        const int64_t model_dim = model_shape[i];
        if (compare_exact || (model_dim != -1)) {
          succ &= (model_dim == full_dims[i]);
        }
      }
    }

    RETURN_ERROR_IF_TRUE(
        !succ, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model '") + model_name + "', tensor '" + tensor_name +
            "': the model expects " + std::to_string(model_shape.size()) +
            " dimensions (shape " + nib::ShapeToString(model_shape) +
            ") but the model configuration specifies " +
            std::to_string(full_dims.size()) +
            " dimensions (an initial batch dimension because max_batch_size "
            "> 0 followed by the explicit tensor shape, making complete "
            "shape " +
            nib::ShapeToString(full_dims) + ")");
  } else {
    // ! supports_batching
    bool succ = (model_shape.size() == dims.size());
    if (succ) {
      for (size_t i = 0; i < dims.size(); ++i) {
        const int64_t model_dim = model_shape[i];
        if (compare_exact || (model_dim != -1)) {
          succ &= (model_dim == dims[i]);
        }
      }
    }

    RETURN_ERROR_IF_TRUE(
        !succ, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model '") + model_name + "', tensor '" + tensor_name +
            "': the model expects " + std::to_string(model_shape.size()) +
            " dimensions (shape " + nib::ShapeToString(model_shape) +
            ") but the model configuration specifies " +
            std::to_string(dims.size()) + " dimensions (shape " +
            nib::ShapeToString(dims) + ")");
  }

  return nullptr;  // success
}

}}}  // namespace triton::backend::onnxruntime
