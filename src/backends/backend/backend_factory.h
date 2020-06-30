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

#include "src/backends/backend/triton_model.h"
#include "src/core/model_config.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

// Adapter that connects backend to Triton using the legacy
// mechanisms. Will be unnecessary once we transition to new arch.
class TritonBackendFactory {
 public:
  struct Config : public BackendConfig {
  };

  static Status Create(
      InferenceServer* server,
      const std::shared_ptr<BackendConfig>& backend_config,
      std::unique_ptr<TritonBackendFactory>* factory)
  {
    LOG_VERBOSE(1) << "Create TritonBackendFactory";

    auto triton_backend_config =
        std::static_pointer_cast<Config>(backend_config);
    factory->reset(new TritonBackendFactory(server, triton_backend_config));
    return Status::Success;
  }

  Status CreateBackend(
      const std::string& model_repository_path, const std::string& model_name,
      const int64_t version, const ModelConfig& model_config,
      const double min_compute_capability,
      std::unique_ptr<InferenceBackend>* backend)
  {
    std::unique_ptr<TritonModel> model;
    RETURN_IF_ERROR(TritonModel::Create(
        server_, model_repository_path, model_name, version, model_config,
        min_compute_capability, &model));
    backend->reset(model.release());
    return Status::Success;
  }

  ~TritonBackendFactory() = default;

 private:
  DISALLOW_COPY_AND_ASSIGN(TritonBackendFactory);

  TritonBackendFactory(
      InferenceServer* server, const std::shared_ptr<Config>& backend_config)
      : server_(server), backend_config_(backend_config)
  {
  }

  InferenceServer* server_;
  const std::shared_ptr<Config> backend_config_;
};

}}  // namespace nvidia::inferenceserver
