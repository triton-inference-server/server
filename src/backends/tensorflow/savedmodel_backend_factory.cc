// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "src/backends/tensorflow/savedmodel_backend_factory.h"

#include <memory>
#include <string>
#include <vector>

#include "src/backends/tensorflow/savedmodel_backend.h"
#include "src/backends/tensorflow/tf_virtual_device.h"
#include "src/core/constants.h"
#include "src/core/filesystem.h"
#include "src/core/logging.h"
#include "src/core/model_config.pb.h"
#include "src/core/model_config_utils.h"

namespace nvidia { namespace inferenceserver {

Status
SavedModelBackendFactory::Create(
    const std::shared_ptr<BackendConfig>& backend_config,
    std::unique_ptr<SavedModelBackendFactory>* factory)
{
  LOG_VERBOSE(1) << "Create SavedModelBackendFactory";

  auto savedmodel_backend_config =
      std::static_pointer_cast<GraphDefBackendFactory::Config>(backend_config);
  factory->reset(new SavedModelBackendFactory(savedmodel_backend_config));

  // Initialize VGPUs if required
  VirtualDeviceTracker::Init(savedmodel_backend_config->memory_limit_mb);

  return Status::Success;
}

Status
SavedModelBackendFactory::CreateBackend(
    const std::string& path, const ModelConfig& model_config,
    const double min_compute_capability,
    std::unique_ptr<InferenceBackend>* backend)
{
  // Read all the savedmodel directories in 'path'.
  std::set<std::string> savedmodel_subdirs;
  RETURN_IF_ERROR(GetDirectorySubdirs(path, &savedmodel_subdirs));

  std::unordered_map<std::string, std::string> models;
  std::vector<std::shared_ptr<LocalizedDirectory>> local_savedmodel_path;

  for (const auto& filename : savedmodel_subdirs) {
    const auto savedmodel_path = JoinPath({path, filename});
    local_savedmodel_path.push_back(
        std::shared_ptr<LocalizedDirectory>(nullptr));
    RETURN_IF_ERROR(
        LocalizeDirectory(savedmodel_path, &local_savedmodel_path.back()));
    models.emplace(
        std::piecewise_construct, std::make_tuple(filename),
        std::make_tuple(local_savedmodel_path.back()->Path()));
  }

  // Create the backend for the model and all the execution contexts
  // requested for this model.
  std::unique_ptr<SavedModelBackend> local_backend(
      new SavedModelBackend(min_compute_capability));
  RETURN_IF_ERROR(local_backend->Init(
      path, model_config, backend_config_.get(),
      kTensorFlowSavedModelPlatform));
  RETURN_IF_ERROR(local_backend->CreateExecutionContexts(models));

  *backend = std::move(local_backend);
  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
