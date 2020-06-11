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

#include "src/backends/backend/triton_model.h"

#include <vector>
#include "src/core/filesystem.h"

namespace nvidia { namespace inferenceserver {

Status
TritonModel::Create(
    const std::string& model_repository_path, const std::string& model_name,
    const int64_t version, const ModelConfig& model_config,
    const double min_compute_capability, std::unique_ptr<TritonModel>* model)
{
  model->reset();

  // The model configuration must specify a backend. The name of the
  // corresponding shared library must be libtriton_<backend>.so.
  if (model_config.backend().empty()) {
    return Status(
        Status::Code::INVALID_ARG,
        "must specify 'backend' for '" + model_config.name() + "'");
  }

  const std::string backend_libname =
      "libtriton_" + model_config.backend() + ".so";

  // Get the path to the backend shared library. Search path is
  // version directory, model directory, global backend directory.
  const auto version_path =
      JoinPath({model_repository_path, model_name, std::to_string(version)});
  const auto model_path = JoinPath({model_repository_path, model_name});
  const std::string global_path =
      "/opt/tritonserver/backends";  // FIXME need cmdline flag
  const std::vector<std::string> search_paths = {version_path, model_path,
                                                 global_path};

  std::string backend_libpath;
  for (const auto& path : search_paths) {
    const auto full_path = JoinPath({path, backend_libname});
    bool exists = false;
    RETURN_IF_ERROR(FileExists(full_path, &exists));
    if (exists) {
      backend_libpath = full_path;
      break;
    }
  }

  if (backend_libpath.empty()) {
    return Status(
        Status::Code::INVALID_ARG, "unable to find '" + backend_libname +
                                       "' for model '" + model_config.name() +
                                       "', searched: " + version_path + ", " +
                                       model_path + ", " + global_path);
  }

  // Create and intialize the model and model instances.
  std::unique_ptr<TritonModel> local_model(
      new TritonModel(min_compute_capability));
  //    RETURN_IF_ERROR(local_backend->Init(path, server_params,
  //    model_config));
  //    RETURN_IF_ERROR(local_backend->CreateExecutionContexts(custom_paths));

  *model = std::move(local_model);
  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
