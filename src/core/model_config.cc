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

#include "src/core/model_config.h"

namespace nvidia { namespace inferenceserver {

size_t
GetDataTypeByteSize(const DataType dtype)
{
  switch (dtype) {
    case TYPE_BOOL:
      return 1;
    case TYPE_UINT8:
      return 1;
    case TYPE_UINT16:
      return 2;
    case TYPE_UINT32:
      return 4;
    case TYPE_UINT64:
      return 8;
    case TYPE_INT8:
      return 1;
    case TYPE_INT16:
      return 2;
    case TYPE_INT32:
      return 4;
    case TYPE_INT64:
      return 8;
    case TYPE_FP16:
      return 2;
    case TYPE_FP32:
      return 4;
    case TYPE_FP64:
      return 8;
    default:
      break;
  }

  return 0;
}

uint64_t
GetSize(const DataType& dtype, const DimsList& dims)
{
  size_t dt_size = GetDataTypeByteSize(dtype);
  if (dt_size <= 0) {
    return 0;
  }

  int64_t size = 0;
  for (auto dim : dims) {
    if (size == 0) {
      size = dim;
    } else {
      size *= dim;
    }
  }

  return size * dt_size;
}

uint64_t
GetSize(const ModelInput& mio)
{
  return GetSize(mio.data_type(), mio.dims());
}

uint64_t
GetSize(const ModelOutput& mio)
{
  return GetSize(mio.data_type(), mio.dims());
}

}}  // namespace nvidia::inferenceserver
