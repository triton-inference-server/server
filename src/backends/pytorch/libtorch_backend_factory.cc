// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "src/backends/pytorch/libtorch_backend_factory.h"

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
LibTorchBackendFactory::Create(
    const LibTorchPlatformConfig& platform_config,
    std::unique_ptr<LibTorchBackendFactory>* factory)
{
  LOG_VERBOSE(1) << "Create LibTorchBackendFactory for platform config \""
                 << platform_config.DebugString() << "\"";

  factory->reset(new LibTorchBackendFactory(platform_config));
  return Status::Success;
}

Status
LibTorchBackendFactory::CreateBackend(
    const std::string& path, const ModelConfig& model_config,
    std::unique_ptr<InferenceBackend>* backend)
{
  // Read all the *.pt files in 'path'.
  std::set<std::string> torch_files;
  RETURN_IF_ERROR(GetDirectoryFiles(path, &torch_files));

  std::unordered_map<std::string, std::string> torch_paths;
  for (const auto& filename : torch_files) {
    const auto torch_path = JoinPath({path, filename});
    torch_paths.emplace(filename, std::move(torch_path));
  }

  // Create the backend for the model and all the execution contexts
  // requested for this model.
  std::unique_ptr<LibTorchBackend> local_backend(new LibTorchBackend);
  RETURN_IF_ERROR(local_backend->Init(path, model_config));
  RETURN_IF_ERROR(local_backend->CreateExecutionContexts(torch_paths));

  *backend = std::move(local_backend);
  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
