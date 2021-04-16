// Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

const Status Status::Success(Status::Code::SUCCESS);

Status::Code
TritonCodeToStatusCode(TRITONSERVER_Error_Code code)
{
  switch (code) {
    case TRITONSERVER_ERROR_UNKNOWN:
      return Status::Code::UNKNOWN;
    case TRITONSERVER_ERROR_INTERNAL:
      return Status::Code::INTERNAL;
    case TRITONSERVER_ERROR_NOT_FOUND:
      return Status::Code::NOT_FOUND;
    case TRITONSERVER_ERROR_INVALID_ARG:
      return Status::Code::INVALID_ARG;
    case TRITONSERVER_ERROR_UNAVAILABLE:
      return Status::Code::UNAVAILABLE;
    case TRITONSERVER_ERROR_UNSUPPORTED:
      return Status::Code::UNSUPPORTED;
    case TRITONSERVER_ERROR_ALREADY_EXISTS:
      return Status::Code::ALREADY_EXISTS;

    default:
      break;
  }

  return Status::Code::UNKNOWN;
}

TRITONSERVER_Error_Code
StatusCodeToTritonCode(Status::Code status_code)
{
  switch (status_code) {
    case Status::Code::UNKNOWN:
      return TRITONSERVER_ERROR_UNKNOWN;
    case Status::Code::INTERNAL:
      return TRITONSERVER_ERROR_INTERNAL;
    case Status::Code::NOT_FOUND:
      return TRITONSERVER_ERROR_NOT_FOUND;
    case Status::Code::INVALID_ARG:
      return TRITONSERVER_ERROR_INVALID_ARG;
    case Status::Code::UNAVAILABLE:
      return TRITONSERVER_ERROR_UNAVAILABLE;
    case Status::Code::UNSUPPORTED:
      return TRITONSERVER_ERROR_UNSUPPORTED;
    case Status::Code::ALREADY_EXISTS:
      return TRITONSERVER_ERROR_ALREADY_EXISTS;

    default:
      break;
  }

  return TRITONSERVER_ERROR_UNKNOWN;
}

}}  // namespace nvidia::inferenceserver
