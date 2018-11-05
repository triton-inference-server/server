// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "src/core/logging.h"
#include "src/core/model_config.pb.h"
#include "src/core/utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"

namespace nvidia { namespace inferenceserver {

namespace {

tensorflow::Status
CreateSavedModelBundle(
  const tfs::SavedModelBundleSourceAdapterConfig& adapter_config,
  const std::string& path, std::unique_ptr<SavedModelBundle>* bundle)
{
  const auto model_path = tensorflow::io::Dirname(path);

  ModelConfig model_config;
  TF_RETURN_IF_ERROR(GetNormalizedModelConfig(model_path, &model_config));

  // Read all the savedmodel directories in 'path'. GetChildren()
  // returns all descendants instead for cloud storage like GCS, so
  // filter out all non-direct descendants.
  std::vector<std::string> possible_children;
  TF_RETURN_IF_ERROR(
    tensorflow::Env::Default()->GetChildren(path, &possible_children));
  std::set<std::string> children;
  for (const auto& child : possible_children) {
    children.insert(child.substr(0, child.find_first_of('/')));
  }

  std::unordered_map<std::string, std::string> savedmodel_paths;
  for (const auto& filename : children) {
    const auto savedmodel_path = tensorflow::io::JoinPath(path, filename);
    savedmodel_paths.emplace(
      std::piecewise_construct, std::make_tuple(filename),
      std::make_tuple(savedmodel_path));
  }

  bundle->reset(new SavedModelBundle);
  tensorflow::Status status = (*bundle)->Init(path, model_config);
  if (status.ok()) {
    status = (*bundle)->CreateExecutionContexts(
      adapter_config.legacy_config().session_config(), savedmodel_paths);
  }
  if (!status.ok()) {
    bundle->reset();
  }

  return status;
}

}  // namespace


tensorflow::Status
SavedModelBundleSourceAdapter::Create(
  const tfs::SavedModelBundleSourceAdapterConfig& config,
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
  SavedModelBundleSourceAdapterConfig);

}}  // namespace tensorflow::serving
