// Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <unordered_map>
#include <variant>

#include "triton/core/tritonserver.h"

namespace triton { namespace server {

using VariantType = std::variant<int, bool, std::string>;
using UnorderedMapType = std::unordered_map<std::string, VariantType>;

template <typename T>
T
get_value(const UnorderedMapType& options, const std::string& key)
{
  auto curr = options.find(key);
  bool is_present = (curr != options.end());
  bool correct_type = std::holds_alternative<T>(curr->second);

  if (!is_present || !correct_type) {
    if (curr == options.end())
      std::cerr << "Error: Key " << key << " not found." << std::endl;
    else
      std::cerr << "Error: Type mismatch for key." << std::endl;
  }
  std::cout << "Key " << key << " found." << std::endl;
  return std::get<T>(curr->second);
}

class Server_Interface {
 public:
  virtual bool CreateWrapper(
      std::shared_ptr<TRITONSERVER_Server>& server, UnorderedMapType& data,
      std::unique_ptr<Server_Interface>* service,
      const RestrictedFeatures& restricted_features);
  virtual TRITONSERVER_Error* Start() = 0;
  virtual TRITONSERVER_Error* Stop() = 0;
};
}}  // namespace triton::server
