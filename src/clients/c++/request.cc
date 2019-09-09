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

#define DLL_EXPORTING

#include "src/clients/c++/request.h"

namespace nvidia { namespace inferenceserver { namespace client {

//==============================================================================

ServerHealthContext::~ServerHealthContext() {}
ServerStatusContext::~ServerStatusContext() {}
ModelControlContext::~ModelControlContext() {}
SharedMemoryControlContext::~SharedMemoryControlContext() {}
TraceControlContext::Options::~Options() {}
TraceControlContext::~TraceControlContext() {}
InferContext::Input::~Input() {}
InferContext::Output::~Output() {}
InferContext::Result::~Result() {}
InferContext::Options::~Options() {}
InferContext::Request::~Request() {}
InferContext::~InferContext() {}

//==============================================================================

template <>
Error
InferContext::Result::GetRawAtCursor(size_t batch_idx, std::string* out)
{
  Error err;

  const uint8_t* len_ptr;
  err = GetRawAtCursor(batch_idx, &len_ptr, sizeof(uint32_t));
  if (!err.IsOk()) {
    return err;
  }

  const uint32_t len = *(reinterpret_cast<const uint32_t*>(len_ptr));

  const uint8_t* str_ptr;
  err = GetRawAtCursor(batch_idx, &str_ptr, len);
  if (!err.IsOk()) {
    return err;
  }

  out->clear();
  std::copy(str_ptr, str_ptr + len, std::back_inserter(*out));

  return Error::Success;
}

//==============================================================================

const Error Error::Success(RequestStatusCode::SUCCESS);

Error::Error(RequestStatusCode code, const std::string& msg)
    : code_(code), msg_(msg), request_id_(0)
{
}

Error::Error(RequestStatusCode code) : code_(code), request_id_(0) {}

Error::Error(const RequestStatus& status) : Error(status.code(), status.msg())
{
  server_id_ = status.server_id();
  request_id_ = status.request_id();
}

std::ostream&
operator<<(std::ostream& out, const Error& err)
{
  out << "[" << err.server_id_ << " " << err.request_id_ << "] "
      << RequestStatusCode_Name(err.code_);
  if (!err.msg_.empty()) {
    out << " - " << err.msg_;
  }
  return out;
}

}}}  // namespace nvidia::inferenceserver::client
