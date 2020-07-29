// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <string>
#include "src/core/status.h"

#define TRITONJSON_STATUSTYPE Status
#define TRITONJSON_STATUSRETURN(M) return Status(Status::Code::INTERNAL, (M))
#define TRITONJSON_STATUSSUCCESS Status::Success
#include "src/core/json.h"

namespace nvidia { namespace inferenceserver {

//
// Implementation for TRITONSERVER_Message.
//
class TritonServerMessage {
 public:
  TritonServerMessage(const TritonJson::Value& msg)
  {
    json_buffer_.Clear();
    msg.Write(&json_buffer_);
    base_ = json_buffer_.Base();
    byte_size_ = json_buffer_.Size();
    from_json_ = true;
  }

  TritonServerMessage(std::string&& msg)
  {
    str_buffer_ = std::move(msg);
    base_ = str_buffer_.data();
    byte_size_ = str_buffer_.size();
    from_json_ = false;
  }

  TritonServerMessage(const TritonServerMessage& rhs)
  {
    from_json_ = rhs.from_json_;
    if (from_json_) {
      json_buffer_ = rhs.json_buffer_;
      base_ = json_buffer_.Base();
      byte_size_ = json_buffer_.Size();
    } else {
      str_buffer_ = rhs.str_buffer_;
      base_ = str_buffer_.data();
      byte_size_ = str_buffer_.size();
    }
  }

  void Serialize(const char** base, size_t* byte_size) const
  {
    *base = base_;
    *byte_size = byte_size_;
  }

 private:
  bool from_json_;
  TritonJson::WriteBuffer json_buffer_;
  std::string str_buffer_;

  const char* base_;
  size_t byte_size_;
};

}}  // namespace nvidia::inferenceserver
