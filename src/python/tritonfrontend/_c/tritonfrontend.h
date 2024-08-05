// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include <unistd.h>  // For sleep

#include <memory>  // For shared_ptr
#include <unordered_map>
#include <variant>

#include "../../../grpc/grpc_server.h"
#include "../../../http_server.h"
#include "../../../restricted_features.h"
#include "triton/core/tritonserver.h"

struct TRITONSERVER_Server {};

namespace triton { namespace server { namespace python {

template <typename Base_Server, typename Frontend_Server>
class TritonFrontend {
  // static_assert(std::is_base_of<Server_Interface, Frontend_Server>::value, "T
  // must be derived from Base");

 private:
  std::shared_ptr<TRITONSERVER_Server> server_;
  std::unique_ptr<Base_Server> service;
  triton::server::RestrictedFeatures restricted_features;

 public:
  TritonFrontend(uintptr_t server_mem_addr, UnorderedMapType data)
  {
    TRITONSERVER_Server* server_ptr =
        reinterpret_cast<TRITONSERVER_Server*>(server_mem_addr);
    server_.reset(server_ptr, TRITONSERVER_CustomDestroy);

    // For debugging
    // for (const auto& [key, value] : data) {
    //     std::cout << "Key: " << key << std::endl;
    //     printVariant(value);
    // }

    bool res = Frontend_Server::CreateWrapper(
        server_, data, &service, restricted_features);
  };

  bool StartService() { return (service->Start() == nullptr); };
  bool StopService() { return (service->Stop() == nullptr); };


  void printVariant(const VariantType& v)
  {
    std::visit(
        [](auto&& arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, std::string>) {
            std::cout << "Value (string): " << arg << std::endl;
          } else if constexpr (std::is_same_v<T, int>) {
            std::cout << "Value (int): " << arg << std::endl;
          } else if constexpr (std::is_same_v<T, bool>) {
            std::cout << "Value (bool): " << std::boolalpha << arg << std::endl;
          }
        },
        v);
  };

  static TRITONSERVER_Error* TRITONSERVER_CustomDestroy(
      TRITONSERVER_Server* obj)
  {
    return nullptr;  // Prevents double-free
  };
};

}}}  // namespace triton::server::python
