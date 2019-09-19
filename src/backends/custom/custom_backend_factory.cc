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

#include "src/backends/custom/custom_backend_factory.h"

#include <memory>
#include <string>
#include <vector>

#include "src/core/constants.h"
#include "src/core/filesystem.h"
#include "src/core/logging.h"
#include "src/core/model_config.pb.h"
#include "src/core/model_config_utils.h"

namespace nvidia { namespace inferenceserver {

Status
CustomBackendFactory::Create(
    const std::shared_ptr<BackendConfig>& backend_config,
    std::unique_ptr<CustomBackendFactory>* factory)
{
  LOG_VERBOSE(1) << "Create CustomBackendFactory";

  auto custom_backend_config = std::static_pointer_cast<Config>(backend_config);
  factory->reset(new CustomBackendFactory(custom_backend_config));
  return Status::Success;
}

Status
CustomBackendFactory::CreateBackend(
    const std::string& model_repository_path, const std::string& model_name,
    const int64_t version, const ModelConfig& model_config,
    std::unique_ptr<InferenceBackend>* backend)
{
  const auto path =
      JoinPath({model_repository_path, model_name, std::to_string(version)});

  // Read all the files in 'path'.
  std::set<std::string> custom_files;
  RETURN_IF_ERROR(GetDirectoryFiles(path, &custom_files));

  std::unordered_map<std::string, std::string> custom_paths;
  for (const auto& filename : custom_files) {
    const auto custom_path = JoinPath({path, filename});

    custom_paths.emplace(
        std::piecewise_construct, std::make_tuple(filename),
        std::make_tuple(custom_path));
  }

  // Create the vector of server parameter values, indexed by the
  // CustomServerParameter value.
  std::vector<std::string> server_params(CUSTOM_SERVER_PARAMETER_CNT);
  server_params[CustomServerParameter::INFERENCE_SERVER_VERSION] =
      backend_config_->inference_server_version;
  server_params[CustomServerParameter::MODEL_REPOSITORY_PATH] =
      model_repository_path;

  // Create the backend for the model and all the execution contexts
  // requested for this model.
  std::unique_ptr<CustomBackend> local_backend(new CustomBackend);
  RETURN_IF_ERROR(local_backend->Init(path, server_params, model_config));
  RETURN_IF_ERROR(local_backend->CreateExecutionContexts(custom_paths));

  *backend = std::move(local_backend);
  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
