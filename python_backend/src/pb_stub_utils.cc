// Copyright 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pb_stub_utils.h"

#include "pb_utils.h"

namespace triton { namespace backend { namespace python {

TRITONSERVER_DataType
numpy_to_triton_type(py::object data_type)
{
  py::module np = py::module::import("numpy");
  if (data_type.equal(np.attr("bool_")))
    return TRITONSERVER_TYPE_BOOL;
  else if (data_type.equal(np.attr("uint8")))
    return TRITONSERVER_TYPE_UINT8;
  else if (data_type.equal(np.attr("uint16")))
    return TRITONSERVER_TYPE_UINT16;
  else if (data_type.equal(np.attr("uint32")))
    return TRITONSERVER_TYPE_UINT32;
  else if (data_type.equal(np.attr("uint64")))
    return TRITONSERVER_TYPE_UINT64;
  else if (data_type.equal(np.attr("int8")))
    return TRITONSERVER_TYPE_INT8;
  else if (data_type.equal(np.attr("int16")))
    return TRITONSERVER_TYPE_INT16;
  else if (data_type.equal(np.attr("int32")))
    return TRITONSERVER_TYPE_INT32;
  else if (data_type.equal(np.attr("int64")))
    return TRITONSERVER_TYPE_INT64;
  else if (data_type.equal(np.attr("float16")))
    return TRITONSERVER_TYPE_FP16;
  else if (data_type.equal(np.attr("float32")))
    return TRITONSERVER_TYPE_FP32;
  else if (data_type.equal(np.attr("float64")))
    return TRITONSERVER_TYPE_FP64;
  else if (
      data_type.equal(np.attr("object_")) ||
      data_type.equal(np.attr("bytes_")) ||
      data_type.attr("type").equal(np.attr("bytes_")))
    return TRITONSERVER_TYPE_BYTES;
  throw PythonBackendException("NumPy dtype is not supported.");
}

py::object
triton_to_numpy_type(TRITONSERVER_DataType data_type)
{
  py::module np = py::module::import("numpy");
  py::object np_type;
  switch (data_type) {
    case TRITONSERVER_TYPE_BOOL:
      np_type = np.attr("bool_");
      break;
    case TRITONSERVER_TYPE_UINT8:
      np_type = np.attr("uint8");
      break;
    case TRITONSERVER_TYPE_UINT16:
      np_type = np.attr("uint16");
      break;
    case TRITONSERVER_TYPE_UINT32:
      np_type = np.attr("uint32");
      break;
    case TRITONSERVER_TYPE_UINT64:
      np_type = np.attr("uint64");
      break;
    case TRITONSERVER_TYPE_INT8:
      np_type = np.attr("int8");
      break;
    case TRITONSERVER_TYPE_INT16:
      np_type = np.attr("int16");
      break;
    case TRITONSERVER_TYPE_INT32:
      np_type = np.attr("int32");
      break;
    case TRITONSERVER_TYPE_INT64:
      np_type = np.attr("int64");
      break;
    case TRITONSERVER_TYPE_FP16:
      np_type = np.attr("float16");
      break;
    case TRITONSERVER_TYPE_FP32:
      np_type = np.attr("float32");
      break;
    case TRITONSERVER_TYPE_FP64:
      np_type = np.attr("float64");
      break;
    case TRITONSERVER_TYPE_BYTES:
      np_type = np.attr("object_");
      break;
    default:
      throw PythonBackendException(
          "Unsupported triton dtype" +
          std::to_string(static_cast<int>(data_type)));
  }

  return np_type;
}

py::dtype
triton_to_pybind_dtype(TRITONSERVER_DataType data_type)
{
  py::dtype dtype_numpy;

  switch (data_type) {
    case TRITONSERVER_TYPE_BOOL:
      dtype_numpy = py::dtype(py::format_descriptor<bool>::format());
      break;
    case TRITONSERVER_TYPE_UINT8:
      dtype_numpy = py::dtype(py::format_descriptor<uint8_t>::format());
      break;
    case TRITONSERVER_TYPE_UINT16:
      dtype_numpy = py::dtype(py::format_descriptor<uint16_t>::format());
      break;
    case TRITONSERVER_TYPE_UINT32:
      dtype_numpy = py::dtype(py::format_descriptor<uint32_t>::format());
      break;
    case TRITONSERVER_TYPE_UINT64:
      dtype_numpy = py::dtype(py::format_descriptor<uint64_t>::format());
      break;
    case TRITONSERVER_TYPE_INT8:
      dtype_numpy = py::dtype(py::format_descriptor<int8_t>::format());
      break;
    case TRITONSERVER_TYPE_INT16:
      dtype_numpy = py::dtype(py::format_descriptor<int16_t>::format());
      break;
    case TRITONSERVER_TYPE_INT32:
      dtype_numpy = py::dtype(py::format_descriptor<int32_t>::format());
      break;
    case TRITONSERVER_TYPE_INT64:
      dtype_numpy = py::dtype(py::format_descriptor<int64_t>::format());
      break;
    case TRITONSERVER_TYPE_FP16:
      // Will be reinterpreted in the python code.
      dtype_numpy = py::dtype(py::format_descriptor<uint16_t>::format());
      break;
    case TRITONSERVER_TYPE_FP32:
      dtype_numpy = py::dtype(py::format_descriptor<float>::format());
      break;
    case TRITONSERVER_TYPE_FP64:
      dtype_numpy = py::dtype(py::format_descriptor<double>::format());
      break;
    case TRITONSERVER_TYPE_BYTES:
      // Will be reinterpreted in the python code.
      dtype_numpy = py::dtype(py::format_descriptor<uint8_t>::format());
      break;
    case TRITONSERVER_TYPE_BF16:
      // NOTE: Currently skipping this call via `if (BF16)` check, but may
      // want to better handle this or set some default/invalid dtype.
      throw PythonBackendException("TYPE_BF16 not currently supported.");
    case TRITONSERVER_TYPE_INVALID:
      throw PythonBackendException("Dtype is invalid.");
    default:
      throw PythonBackendException("Unsupported triton dtype.");
  }

  return dtype_numpy;
}

DLDataType
triton_to_dlpack_type(TRITONSERVER_DataType triton_dtype)
{
  DLDataType dl_dtype;
  DLDataTypeCode dl_code;

  // Number of bits required for the data type.
  size_t dt_size = 0;

  dl_dtype.lanes = 1;
  switch (triton_dtype) {
    case TRITONSERVER_TYPE_BOOL:
      dl_code = DLDataTypeCode::kDLBool;
      dt_size = 8;
      break;
    case TRITONSERVER_TYPE_UINT8:
      dl_code = DLDataTypeCode::kDLUInt;
      dt_size = 8;
      break;
    case TRITONSERVER_TYPE_UINT16:
      dl_code = DLDataTypeCode::kDLUInt;
      dt_size = 16;
      break;
    case TRITONSERVER_TYPE_UINT32:
      dl_code = DLDataTypeCode::kDLUInt;
      dt_size = 32;
      break;
    case TRITONSERVER_TYPE_UINT64:
      dl_code = DLDataTypeCode::kDLUInt;
      dt_size = 64;
      break;
    case TRITONSERVER_TYPE_INT8:
      dl_code = DLDataTypeCode::kDLInt;
      dt_size = 8;
      break;
    case TRITONSERVER_TYPE_INT16:
      dl_code = DLDataTypeCode::kDLInt;
      dt_size = 16;
      break;
    case TRITONSERVER_TYPE_INT32:
      dl_code = DLDataTypeCode::kDLInt;
      dt_size = 32;
      break;
    case TRITONSERVER_TYPE_INT64:
      dl_code = DLDataTypeCode::kDLInt;
      dt_size = 64;
      break;
    case TRITONSERVER_TYPE_FP16:
      dl_code = DLDataTypeCode::kDLFloat;
      dt_size = 16;
      break;
    case TRITONSERVER_TYPE_FP32:
      dl_code = DLDataTypeCode::kDLFloat;
      dt_size = 32;
      break;
    case TRITONSERVER_TYPE_FP64:
      dl_code = DLDataTypeCode::kDLFloat;
      dt_size = 64;
      break;
    case TRITONSERVER_TYPE_BYTES:
      throw PythonBackendException(
          "TYPE_BYTES tensors cannot be converted to DLPack.");
    case TRITONSERVER_TYPE_BF16:
      dl_code = DLDataTypeCode::kDLBfloat;
      dt_size = 16;
      break;

    default:
      throw PythonBackendException(
          std::string("DType code \"") +
          std::to_string(static_cast<int>(triton_dtype)) +
          "\" is not supported.");
  }

  dl_dtype.code = dl_code;
  dl_dtype.bits = dt_size;
  return dl_dtype;
}

TRITONSERVER_DataType
dlpack_to_triton_type(const DLDataType& data_type)
{
  if (data_type.lanes != 1) {
    // lanes != 1 is not supported in Python backend.
    return TRITONSERVER_TYPE_INVALID;
  }

  if (data_type.code == DLDataTypeCode::kDLFloat) {
    if (data_type.bits == 16) {
      return TRITONSERVER_TYPE_FP16;
    } else if (data_type.bits == 32) {
      return TRITONSERVER_TYPE_FP32;
    } else if (data_type.bits == 64) {
      return TRITONSERVER_TYPE_FP64;
    }
  }

  if (data_type.code == DLDataTypeCode::kDLInt) {
    if (data_type.bits == 8) {
      return TRITONSERVER_TYPE_INT8;
    } else if (data_type.bits == 16) {
      return TRITONSERVER_TYPE_INT16;
    } else if (data_type.bits == 32) {
      return TRITONSERVER_TYPE_INT32;
    } else if (data_type.bits == 64) {
      return TRITONSERVER_TYPE_INT64;
    }
  }

  if (data_type.code == DLDataTypeCode::kDLUInt) {
    if (data_type.bits == 8) {
      return TRITONSERVER_TYPE_UINT8;
    } else if (data_type.bits == 16) {
      return TRITONSERVER_TYPE_UINT16;
    } else if (data_type.bits == 32) {
      return TRITONSERVER_TYPE_UINT32;
    } else if (data_type.bits == 64) {
      return TRITONSERVER_TYPE_UINT64;
    }
  }

  if (data_type.code == DLDataTypeCode::kDLBool) {
    if (data_type.bits == 8) {
      return TRITONSERVER_TYPE_BOOL;
    }
  }

  if (data_type.code == DLDataTypeCode::kDLBfloat) {
    if (data_type.bits != 16) {
      throw PythonBackendException(
          "Expected BF16 tensor to have 16 bits, but had: " +
          std::to_string(data_type.bits));
    }
    return TRITONSERVER_TYPE_BF16;
  }

  return TRITONSERVER_TYPE_INVALID;
}
}}}  // namespace triton::backend::python
