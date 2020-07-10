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

#include "src/core/infer_parameter.h"

namespace nvidia { namespace inferenceserver {


const void*
InferenceParameter::ValuePointer() const
{
  switch (type_) {
    case TRITONSERVER_PARAMETER_STRING:
      return reinterpret_cast<const void*>(value_string_.c_str());
    case TRITONSERVER_PARAMETER_INT:
      return reinterpret_cast<const void*>(&value_int64_);
    case TRITONSERVER_PARAMETER_BOOL:
      return reinterpret_cast<const void*>(&value_bool_);
    default:
      break;
  }

  return nullptr;
}

std::ostream&
operator<<(std::ostream& out, const InferenceParameter& parameter)
{
  out << "[0x" << std::addressof(parameter) << "] "
      << "name: " << parameter.Name()
      << ", type: " << TRITONSERVER_ParameterTypeString(parameter.Type())
      << ", value: ";
  return out;
}

}}  // namespace nvidia::inferenceserver
