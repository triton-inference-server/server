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

#include "src/custom/sdk/error_codes.h"

namespace nvidia { namespace inferenceserver { namespace custom {

ErrorCodes::ErrorCodes()
{
  RegisterError(Success, "success");
  RegisterError(CreationFailure, "failed to create instance");
  RegisterError(InvalidModelConfig, "invalid model configuration");
  RegisterError(
      InvalidInvocationV1,
      "invalid V1 function invocation while the custom backend is not V1");
  RegisterError(
      InvalidInvocationV2,
      "invalid V2 function invocation while the custom backend is not V2");
  RegisterError(Unknown, "unknown error");
}

const char*
ErrorCodes::ErrorString(int error) const
{
  if (ErrorIsRegistered(error)) {
    return err_messages_[error].c_str();
  }

  return err_messages_[Unknown].c_str();
}

int
ErrorCodes::RegisterError(const std::string& error_string)
{
  err_messages_.push_back(error_string);
  return static_cast<int>(err_messages_.size() - 1);
}

void
ErrorCodes::RegisterError(int error, const std::string& error_string)
{
  if (ErrorIsRegistered(error)) {
    err_messages_[error] = error_string;
  }
}

}}}  // namespace nvidia::inferenceserver::custom
