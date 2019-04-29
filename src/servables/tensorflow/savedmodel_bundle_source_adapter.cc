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

#include "src/servables/tensorflow/savedmodel_bundle_source_adapter.h"

#include <memory>
#include <string>
#include <vector>

#include "src/core/constants.h"
#include "src/core/filesystem.h"
#include "src/core/logging.h"
#include "src/core/model_config.pb.h"
#include "src/core/model_config_utils.h"
#include "src/core/model_repository_manager.h"
#include "tensorflow/core/lib/core/errors.h"

namespace nvidia { namespace inferenceserver {

namespace {

tensorflow::Status
CreateSavedModelBundle(
    const SavedModelBundleSourceAdapterConfig& adapter_config,
    const std::string& path, std::unique_ptr<SavedModelBundle>* bundle)
{
  const auto model_path = DirName(path);
  const auto model_name = BaseName(model_path);

  ModelConfig model_config;
  Status status = ModelRepositoryManager::GetModelConfig(
      std::string(model_name), &model_config);
  if (!status.IsOk()) {
    return tensorflow::errors::Internal(status.Message());
  }

  // Read all the savedmodel directories in 'path'.
  std::set<std::string> savedmodel_subdirs;
  status = GetDirectorySubdirs(path, &savedmodel_subdirs);
  if (!status.IsOk()) {
    return tensorflow::errors::Internal(status.Message());
  }

  std::unordered_map<std::string, std::string> savedmodel_paths;
  for (const auto& filename : savedmodel_subdirs) {
    const auto savedmodel_path = JoinPath({path, filename});
    savedmodel_paths.emplace(
        std::piecewise_construct, std::make_tuple(filename),
        std::make_tuple(savedmodel_path));
  }

  bundle->reset(new SavedModelBundle);
  status = (*bundle)->Init(path, model_config);
  if (status.IsOk()) {
    status = (*bundle)->CreateExecutionContexts(
        adapter_config.session_config(), savedmodel_paths);
  }
  if (!status.IsOk()) {
    bundle->reset();
    return tensorflow::errors::Internal(status.Message());
  }

  return tensorflow::Status::OK();
}

}  // namespace

tensorflow::Status
SavedModelBundleSourceAdapter::Create(
    const SavedModelBundleSourceAdapterConfig& config,
    std::unique_ptr<
        tfs::SourceAdapter<tfs::StoragePath, std::unique_ptr<tfs::Loader>>>*
        adapter)
{
  LOG_VERBOSE(1) << "Create SavedModelBundleSourceAdaptor for config \""
                 << config.DebugString() << "\"";

  Creator creator = std::bind(
      &CreateSavedModelBundle, config, std::placeholders::_1,
      std::placeholders::_2);

  adapter->reset(new SavedModelBundleSourceAdapter(
      config, creator, SimpleSourceAdapter::EstimateNoResources()));
  return tensorflow::Status::OK();
}

SavedModelBundleSourceAdapter::~SavedModelBundleSourceAdapter()
{
  Detach();
}

}}  // namespace nvidia::inferenceserver

namespace tensorflow { namespace serving {

REGISTER_STORAGE_PATH_SOURCE_ADAPTER(
    nvidia::inferenceserver::SavedModelBundleSourceAdapter,
    nvidia::inferenceserver::SavedModelBundleSourceAdapterConfig);
}}  // namespace tensorflow::serving
