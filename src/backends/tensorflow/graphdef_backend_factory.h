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
#pragma once

#include "src/core/constants.h"
#include "src/core/model_config.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class InferenceBackend;

// Adapter that converts storage paths pointing to GraphDef files into
// the corresponding graphdef backend.
class GraphDefBackendFactory {
 public:
  struct Config : public BackendConfig {
    // Autofill missing required model configuration settings based on
    // model definition file.
    bool autofill;

    bool allow_gpu_memory_growth;
    float per_process_gpu_memory_fraction;
    bool allow_soft_placement;
    std::map<int, std::vector<float>> memory_limit_mb;
  };

  static Status Create(
      const std::shared_ptr<BackendConfig>& backend_config,
      std::unique_ptr<GraphDefBackendFactory>* factory);

  Status CreateBackend(
      const std::string& path, const ModelConfig& model_config,
      std::unique_ptr<InferenceBackend>* backend);

  ~GraphDefBackendFactory() = default;

 private:
  DISALLOW_COPY_AND_ASSIGN(GraphDefBackendFactory);

  GraphDefBackendFactory(const std::shared_ptr<Config>& backend_config)
      : backend_config_(backend_config)
  {
  }

  const std::shared_ptr<Config> backend_config_;
};

}}  // namespace nvidia::inferenceserver
