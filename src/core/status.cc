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

#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

const Status Status::Success(RequestStatusCode::SUCCESS);

std::string
Status::AsString() const
{
  std::string str;

  switch (code_) {
    case RequestStatusCode::INVALID:
      str = "Invalid";
      break;
    case RequestStatusCode::SUCCESS:
      str = "OK";
      break;
    case RequestStatusCode::UNKNOWN:
      str = "Unknown";
      break;
    case RequestStatusCode::INTERNAL:
      str = "Internal";
      break;
    case RequestStatusCode::NOT_FOUND:
      str = "Not found";
      break;
    case RequestStatusCode::INVALID_ARG:
      str = "Invalid argument";
      break;
    case RequestStatusCode::UNAVAILABLE:
      str = "Unavailable";
      break;
    case RequestStatusCode::UNSUPPORTED:
      str = "Unsupported";
      break;
    case RequestStatusCode::ALREADY_EXISTS:
      str = "Already exists";
      break;

    default:
      str = "Unknown status code (" + std::to_string(code_) + ")";
      break;
  }

  str += ": " + msg_;
  return str;
}

}}  // namespace nvidia::inferenceserver
