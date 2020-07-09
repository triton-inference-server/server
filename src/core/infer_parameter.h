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

#include <iostream>
#include <string>
#include "src/core/tritonserver.h"

namespace nvidia { namespace inferenceserver {

//
// An inference parameter.
//
class InferenceParameter {
 public:
  InferenceParameter(const char* name, const char* value)
      : name_(name), type_(TRITONSERVER_PARAMETER_STRING), value_string_(value)
  {
  }

  InferenceParameter(const char* name, const int64_t value)
      : name_(name), type_(TRITONSERVER_PARAMETER_INT), value_int64_(value)
  {
  }

  InferenceParameter(const char* name, const bool value)
      : name_(name), type_(TRITONSERVER_PARAMETER_BOOL), value_bool_(value)
  {
  }

  // The name of the parametre.
  const std::string& Name() const { return name_; }

  // Data type of the parameter.
  TRITONSERVER_ParameterType Type() const { return type_; }

  // Return a pointer to the parameter. This returned pointer must be
  // cast correctly based on 'type_'.
  //   TRITONSERVER_PARAMETER_STRING -> const char*
  //   TRITONSERVER_PARAMETER_INT -> int64_t*
  //   TRITONSERVER_PARAMETER_BOOL -> bool*
  const void* ValuePointer() const;

 private:
  friend std::ostream& operator<<(
      std::ostream& out, const InferenceParameter& parameter);

  std::string name_;
  TRITONSERVER_ParameterType type_;

  std::string value_string_;
  int64_t value_int64_;
  bool value_bool_;
};

std::ostream& operator<<(
    std::ostream& out, const InferenceParameter& parameter);

}}  // namespace nvidia::inferenceserver
