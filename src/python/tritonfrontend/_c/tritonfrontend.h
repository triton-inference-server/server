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

#include <memory>  // For shared_ptr
#include <unordered_map>
#include <variant>


#ifdef TRITON_ENABLE_GRPC
#include "../../../grpc/grpc_server.h"
#endif


#if defined(TRITON_ENABLE_HTTP) || defined(TRITON_ENABLE_METRICS)
#include "../../../http_server.h"
#endif


#include "../../../common.h"
#include "../../../restricted_features.h"
#include "../../../shared_memory_manager.h"
#include "triton/common/logging.h"
#include "triton/common/triton_json.h"
#include "triton/core/tritonserver.h"


struct TRITONSERVER_Server {};

namespace triton { namespace server { namespace python {

// base exception for all Triton error code
struct TritonError : public std::runtime_error {
  explicit TritonError(const std::string& what) : std::runtime_error(what) {}
};

// triton::core::python exceptions map 1:1 to TRITONSERVER_Error_Code.
struct UnknownError : public TritonError {
  explicit UnknownError(const std::string& what) : TritonError(what) {}
};
struct InternalError : public TritonError {
  explicit InternalError(const std::string& what) : TritonError(what) {}
};
struct NotFoundError : public TritonError {
  explicit NotFoundError(const std::string& what) : TritonError(what) {}
};
struct InvalidArgumentError : public TritonError {
  explicit InvalidArgumentError(const std::string& what) : TritonError(what) {}
};
struct UnavailableError : public TritonError {
  explicit UnavailableError(const std::string& what) : TritonError(what) {}
};
struct UnsupportedError : public TritonError {
  explicit UnsupportedError(const std::string& what) : TritonError(what) {}
};
struct AlreadyExistsError : public TritonError {
  explicit AlreadyExistsError(const std::string& what) : TritonError(what) {}
};

void
ThrowIfError(TRITONSERVER_Error* err)
{
  if (err == nullptr) {
    return;
  }
  std::shared_ptr<TRITONSERVER_Error> managed_err(
      err, TRITONSERVER_ErrorDelete);
  std::string msg = TRITONSERVER_ErrorMessage(err);
  switch (TRITONSERVER_ErrorCode(err)) {
    case TRITONSERVER_ERROR_INTERNAL:
      throw InternalError(std::move(msg));
    case TRITONSERVER_ERROR_NOT_FOUND:
      throw NotFoundError(std::move(msg));
    case TRITONSERVER_ERROR_INVALID_ARG:
      throw InvalidArgumentError(std::move(msg));
    case TRITONSERVER_ERROR_UNAVAILABLE:
      throw UnavailableError(std::move(msg));
    case TRITONSERVER_ERROR_UNSUPPORTED:
      throw UnsupportedError(std::move(msg));
    case TRITONSERVER_ERROR_ALREADY_EXISTS:
      throw AlreadyExistsError(std::move(msg));
    default:
      throw UnknownError(std::move(msg));
  }
}


template <typename Base, typename FrontendServer>
class TritonFrontend {
 private:
  std::shared_ptr<TRITONSERVER_Server> server_;
  std::unique_ptr<Base> service;
  triton::server::RestrictedFeatures restricted_features;
  // TODO: [DLIS-7194] Add support for TraceManager & SharedMemoryManager
  // triton::server::TraceManager trace_manager_;
  // triton::server::SharedMemoryManager shm_manager_;

 public:
  TritonFrontend(uintptr_t server_mem_addr, UnorderedMapType data)
  {
    TRITONSERVER_Server* server_ptr =
        reinterpret_cast<TRITONSERVER_Server*>(server_mem_addr);

    server_.reset(server_ptr, EmptyDeleter);
    TritonFrontend::_populate_restricted_features(data, restricted_features);

#ifdef TRITON_ENABLE_HTTP
    if constexpr (std::is_same_v<FrontendServer, HTTPAPIServer>) {
      ThrowIfError(FrontendServer::Create(
          server_, data, nullptr /* TraceManager */,
          nullptr /* SharedMemoryManager */, restricted_features, &service));
    }
#endif

#ifdef TRITON_ENABLE_GRPC
    if constexpr (std::is_same_v<
                      FrontendServer, triton::server::grpc::Server>) {
      ThrowIfError(FrontendServer::Create(
          server_, data, nullptr /* TraceManager */,
          nullptr /* SharedMemoryManager */, restricted_features, &service));
    }
#endif

#ifdef TRITON_ENABLE_METRICS
    if constexpr (std::is_same_v<FrontendServer, HTTPMetricsServer>) {
      ThrowIfError(FrontendServer::Create(server_, data, &service));
    }
#endif
  };

  // TODO: [DLIS-7194] Add support for TraceManager & SharedMemoryManager
  // TritonFrontend(
  //     uintptr_t server_mem_addr, UnorderedMapType data,
  //     TraceManager trace_manager, SharedMemoryManager shm_manager)

  void StartService() { ThrowIfError(service->Start()); };
  void StopService() { ThrowIfError(service->Stop()); };

  // The frontend does not own the TRITONSERVER_Server* object.
  // Hence, deleting the underlying server instance,
  // will cause a double-free when the core bindings attempt to
  // delete the TRITONSERVER_Server instance.
  static void EmptyDeleter(TRITONSERVER_Server* obj){};

  static void _populate_restricted_features(
      UnorderedMapType& data, RestrictedFeatures& rest_features)
  {
    std::string map_key =
        "restricted_features";  // Name of option in UnorderedMap
    std::string key_prefix;     // Prefix for header key
    if (std::is_same_v<FrontendServer, triton::server::HTTPAPIServer>) {
      key_prefix = "";
    } else if (std::is_same_v<FrontendServer, triton::server::grpc::Server>) {
      key_prefix = "triton-grpc-protocol-";
    } else {
      // Restricted Features is not supported for this class.
      return;
    }

    std::string restricted_info;
    ThrowIfError(GetValue(data, map_key, &restricted_info));

    triton::common::TritonJson::Value rf_groups;
    ThrowIfError(rf_groups.Parse(restricted_info));

    std::string key, value, feature;
    for (size_t group_idx = 0; group_idx < rf_groups.ArraySize(); group_idx++) {
      triton::common::TritonJson::Value feature_group;
      ThrowIfError(rf_groups.IndexAsObject(group_idx, &feature_group));

      // Extract key and value
      ThrowIfError(feature_group.MemberAsString("key", &key));
      ThrowIfError(feature_group.MemberAsString("value", &value));

      triton::common::TritonJson::Value features;
      ThrowIfError(feature_group.MemberAsArray("features", &features));

      // Extract feature list
      for (size_t feature_idx = 0; feature_idx < features.ArraySize();
           feature_idx++) {
        ThrowIfError(features.IndexAsString(feature_idx, &feature));

        rest_features.Insert(
            RestrictedFeatures::ToCategory(feature),
            std::make_pair(key_prefix + key, value));
      }
    }
  };
};
}}}  // namespace triton::server::python
