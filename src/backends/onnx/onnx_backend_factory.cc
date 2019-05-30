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

#include "src/backends/onnx/onnx_backend_factory.h"

#include <memory>
#include <string>
#include <vector>
#include "src/backends/onnx/loader.h"
#include "src/backends/onnx/onnx_utils.h"
#include "src/core/constants.h"
#include "src/core/filesystem.h"
#include "src/core/logging.h"
#include "src/core/model_config.pb.h"
#include "src/core/model_config_utils.h"

namespace nvidia { namespace inferenceserver {

OnnxBackendFactory::~OnnxBackendFactory()
{
  OnnxLoader::Stop();
}

Status
OnnxBackendFactory::Create(
    const OnnxPlatformConfig& platform_config,
    std::unique_ptr<OnnxBackendFactory>* factory)
{
  LOG_VERBOSE(1) << "Create OnnxBackendFactory for platform config \""
                 << platform_config.DebugString() << "\"";

  std::unique_ptr<OnnxBackendFactory> local(
      new OnnxBackendFactory(platform_config));
  RETURN_IF_ERROR(OnnxLoader::Init());

  *factory = std::move(local);
  return Status::Success;
}

Status
OnnxBackendFactory::CreateBackend(
    const std::string& path, const ModelConfig& model_config,
    std::unique_ptr<InferenceBackend>* backend)
{
  std::set<std::string> onnx_files;
  RETURN_IF_ERROR(GetDirectoryFiles(path, &onnx_files));

  std::unordered_map<std::string, std::string> onnx_paths;
  for (const auto& filename : onnx_files) {
    const auto onnx_path = JoinPath({path, filename});
    onnx_paths.emplace(
        std::piecewise_construct, std::make_tuple(filename),
        std::make_tuple(onnx_path));
  }

  // Create the backend for the model and all the execution contexts
  // requested for this model.
  std::unique_ptr<OnnxBackend> local_backend(new OnnxBackend);
  RETURN_IF_ERROR(local_backend->Init(path, model_config));
  RETURN_IF_ERROR(local_backend->CreateExecutionContexts(onnx_paths));

  *backend = std::move(local_backend);
  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
