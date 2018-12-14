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

#include "src/servables/custom/custom_bundle_source_adapter.h"

#include <memory>
#include <string>
#include <vector>

#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.pb.h"
#include "src/core/model_repository_manager.h"
#include "src/core/utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"

namespace nvidia { namespace inferenceserver {

namespace {

tensorflow::Status
CreateCustomBundle(
  const CustomBundleSourceAdapterConfig& adapter_config,
  const std::string& path, std::unique_ptr<CustomBundle>* bundle)
{
  const auto model_path = tensorflow::io::Dirname(path);
  const auto model_name = tensorflow::io::Basename(model_path);

  ModelConfig model_config;
  TF_RETURN_IF_ERROR(ModelRepositoryManager::GetModelConfig(
    std::string(model_name), &model_config));

  // Read all the files in 'path'. GetChildren() returns all
  // descendants instead for cloud storage like GCS, so filter out all
  // non-direct descendants.
  std::vector<std::string> possible_children;
  TF_RETURN_IF_ERROR(
    tensorflow::Env::Default()->GetChildren(path, &possible_children));
  std::set<std::string> children;
  for (const auto& child : possible_children) {
    children.insert(child.substr(0, child.find_first_of('/')));
  }

  std::unordered_map<std::string, std::string> custom_paths;
  for (const auto& filename : children) {
    const auto custom_path = tensorflow::io::JoinPath(path, filename);
    if (!tensorflow::Env::Default()->IsDirectory(custom_path).ok()) {
      custom_paths.emplace(
        std::piecewise_construct, std::make_tuple(filename),
        std::make_tuple(custom_path));
    }
  }

  // Create the bundle for the model and all the execution contexts
  // requested for this model.
  bundle->reset(new CustomBundle);
  tensorflow::Status status = (*bundle)->Init(path, model_config);
  if (status.ok()) {
    status = (*bundle)->CreateExecutionContexts(custom_paths);
  }
  if (!status.ok()) {
    bundle->reset();
  }

  return status;
}

}  // namespace


tensorflow::Status
CustomBundleSourceAdapter::Create(
  const CustomBundleSourceAdapterConfig& config,
  std::unique_ptr<
    SourceAdapter<tfs::StoragePath, std::unique_ptr<tfs::Loader>>>* adapter)
{
  LOG_VERBOSE(1) << "Create CustomBundleSourceAdaptor for config \""
                 << config.DebugString() << "\"";

  Creator creator = std::bind(
    &CreateCustomBundle, config, std::placeholders::_1, std::placeholders::_2);

  adapter->reset(new CustomBundleSourceAdapter(
    config, creator, SimpleSourceAdapter::EstimateNoResources()));
  return tensorflow::Status::OK();
}

CustomBundleSourceAdapter::~CustomBundleSourceAdapter()
{
  Detach();
}

}}  // namespace nvidia::inferenceserver

namespace tensorflow { namespace serving {

REGISTER_STORAGE_PATH_SOURCE_ADAPTER(
  nvidia::inferenceserver::CustomBundleSourceAdapter,
  nvidia::inferenceserver::CustomBundleSourceAdapterConfig);
}}  // namespace tensorflow::serving
