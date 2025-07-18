// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <dlpack/dlpack.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "triton/core/tritonserver.h"

namespace py = pybind11;
namespace triton { namespace backend { namespace python {

/// Convert numpy dtype to triton dtype
/// \param data_type numpy data type to be converted.
/// \return equivalent triton dtype
TRITONSERVER_DataType numpy_to_triton_type(py::object data_type);

/// Convert triton dtype to numpy dtype
/// \param data_type triton dtype to be converted.
/// \return equivalent numpy data type.
py::object triton_to_numpy_type(TRITONSERVER_DataType data_type);

/// Convert triton dtype to dlpack dtype
/// \param data_type triton dtype to be converted
/// \return equivalent DLPack data type.
DLDataType triton_to_dlpack_type(TRITONSERVER_DataType data_type);

/// Convert dlpack type to triton type
/// \param data_type triton dtype to be converted
/// \return equivalent Triton dtype
TRITONSERVER_DataType dlpack_to_triton_type(const DLDataType& data_type);

/// Convert triton data to pybind data type.
/// \param data_type triton dtype to be converted.
/// \return equivalent pybind numpy dtype.
py::dtype triton_to_pybind_dtype(TRITONSERVER_DataType data_type);
}}}  // namespace triton::backend::python
