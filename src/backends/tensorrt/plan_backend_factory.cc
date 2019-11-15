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

#include "src/backends/tensorrt/plan_backend_factory.h"

#include <memory>
#include <string>
#include <vector>

#include <NvInferPlugin.h>
#include "src/backends/tensorrt/logging.h"
#include "src/core/constants.h"
#include "src/core/filesystem.h"
#include "src/core/logging.h"
#include "src/core/model_config.pb.h"
#include "src/core/model_config_utils.h"

namespace nvidia { namespace inferenceserver {

Status
PlanBackendFactory::Create(
    const std::shared_ptr<BackendConfig>& backend_config,
    std::unique_ptr<PlanBackendFactory>* factory)
{
  LOG_VERBOSE(1) << "Create PlanBackendFactory";

  // Register all the default plugins that come with TensorRT
  bool success = true;
  std::once_flag onceFlag;
  {
    std::call_once(onceFlag, [&success] {
      LOG_VERBOSE(1) << "Registering TensorRT Plugins";
      success = initLibNvInferPlugins(&tensorrt_logger, "");
    });
  }
  if (!success) {
    return Status(
        RequestStatusCode::INTERNAL,
        "unable to register default TensorRT Plugins");
  }

  auto plan_backend_config = std::static_pointer_cast<Config>(backend_config);
  factory->reset(new PlanBackendFactory(plan_backend_config));
  return Status::Success;
}

Status
PlanBackendFactory::CreateBackend(
    const std::string& path, const ModelConfig& model_config,
    std::unique_ptr<InferenceBackend>* backend)
{
  std::set<std::string> plan_files;
  RETURN_IF_ERROR(
      GetDirectoryFiles(path, true /* skip_hidden_files */, &plan_files));

  std::unordered_map<std::string, std::vector<char>> models;
  for (const auto& filename : plan_files) {
    const auto plan_path = JoinPath({path, filename});
    std::string model_data_str;
    RETURN_IF_ERROR(ReadTextFile(plan_path, &model_data_str));
    std::vector<char> model_data(model_data_str.begin(), model_data_str.end());
    models.emplace(filename, std::move(model_data));
  }

  // Create the backend for the model and all the execution contexts
  // requested for this model.
  std::unique_ptr<PlanBackend> local_backend(new PlanBackend);
  RETURN_IF_ERROR(
      local_backend->Init(path, model_config, kTensorRTPlanPlatform));
  RETURN_IF_ERROR(local_backend->CreateExecutionContexts(models));

  *backend = std::move(local_backend);
  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
