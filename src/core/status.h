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
#pragma once

#include "src/core/request_status.pb.h"

namespace nvidia { namespace inferenceserver {

class Status {
 public:
  // Construct a status from a code with no message.
  explicit Status(RequestStatusCode code = RequestStatusCode::SUCCESS)
      : code_(code)
  {
  }

  // Construct a status from a code and message.
  explicit Status(RequestStatusCode code, const std::string& msg)
      : code_(code), msg_(msg)
  {
  }

  // Convenience "success" value. Can be used as Status::Success to
  // indicate no error.
  static const Status Success;

  // Return the RequestStatusCode for this status.
  RequestStatusCode Code() const { return code_; }

  // Return the message for this status.
  const std::string& Message() const { return msg_; }

  // Return true if this status indicates "ok"/"success", false if
  // status indicates some kind of failure.
  bool IsOk() const { return code_ == RequestStatusCode::SUCCESS; }

  // Return the status as a string.
  std::string AsString() const;

  // Convert a TensorFlow status code to inference server status code.
  static RequestStatusCode FromTFError(const int tf_code);

 private:
  RequestStatusCode code_;
  std::string msg_;
};

// If status is non-OK, exit.
#define CHECK_IF_ERROR(S)                                \
  do {                                                   \
    const Status& status__ = (S);                        \
    if (status__.Code() != RequestStatusCode::SUCCESS) { \
      std::cerr << status__.AsString() << std::endl;     \
      exit(1);                                           \
    }                                                    \
  } while (false)

// If status is non-OK, return the Status.
#define RETURN_IF_ERROR(S)                               \
  do {                                                   \
    const Status& status__ = (S);                        \
    if (status__.Code() != RequestStatusCode::SUCCESS) { \
      return status__;                                   \
    }                                                    \
  } while (false)

// If TensorFlow status is non-OK, return the equivalent Status.
#define RETURN_IF_TF_ERROR(TFS)                                            \
  do {                                                                     \
    const tensorflow::Status& status__ = (TFS);                            \
    if (status__.code() != 0) {                                            \
      return Status(                                                       \
          Status::FromTFError(status__.code()), status__.error_message()); \
    }                                                                      \
  } while (false)

}}  // namespace nvidia::inferenceserver
