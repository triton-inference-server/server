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
//

#include "src/core/model_config_manager.h"

#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"

namespace nvidia { namespace inferenceserver {

ModelConfigManager*
ModelConfigManager::GetSingleton()
{
  static ModelConfigManager singleton;
  return &singleton;
}

tensorflow::Status
ModelConfigManager::SetModelConfigs(const ModelConfigMap& model_configs)
{
  ModelConfigManager* singleton = GetSingleton();
  singleton->configs_ = model_configs;
  return tensorflow::Status::OK();
}

tensorflow::Status
ModelConfigManager::GetModelConfig(
  const std::string& name, const ModelConfig** model_config)
{
  ModelConfigManager* singleton = GetSingleton();
  const auto itr = singleton->configs_.find(name);
  if (itr == singleton->configs_.end()) {
    return tensorflow::errors::NotFound(
      "no configuration for model '", name, "'");
  }

  *model_config = &itr->second;
  return tensorflow::Status::OK();
}

tensorflow::Status
ModelConfigManager::ReadModelConfigs(
  const std::string& model_store_path, const bool autofill,
  ModelConfigMap* model_configs, tfs::ModelServerConfig* tfs_model_configs)
{
  // Each subdirectory of model_store_path is a model directory from
  // which we read the model configuration.
  std::vector<std::string> children;
  TF_RETURN_IF_ERROR(
    tensorflow::Env::Default()->GetChildren(model_store_path, &children));

  // GetChildren() returns all descendants instead for cloud storage
  // like GCS.  In such case we should filter out all non-direct
  // descendants.
  std::set<std::string> real_children;
  for (size_t i = 0; i < children.size(); ++i) {
    const std::string& child = children[i];
    real_children.insert(child.substr(0, child.find_first_of('/')));
  }

  for (const auto& child : real_children) {
    const auto full_path = tensorflow::io::JoinPath(model_store_path, child);
    if (!tensorflow::Env::Default()->IsDirectory(full_path).ok()) {
      continue;
    }

    const auto& ret = model_configs->emplace(child, ModelConfig{});
    if (!ret.second) {
      return tensorflow::errors::InvalidArgument(
        "repeated model name '", child, "'");
    }

    ModelConfig* model_config = &(ret.first->second);

    // If enabled, try to automatically generate missing parts of the
    // model configuration from the model definition. In all cases
    // normalize and validate the config.
    TF_RETURN_IF_ERROR(
      GetNormalizedModelConfig(full_path, autofill, model_config));
    TF_RETURN_IF_ERROR(ValidateModelConfig(*model_config, std::string()));

    // Make sure the name of the model matches the name of the
    // directory. This is a somewhat arbitrary requirement but seems
    // like good practice to require it of the user. It also acts as a
    // check to make sure we don't have two different models with the
    // same name.
    if (model_config->name() != child) {
      return tensorflow::errors::InvalidArgument(
        "unexpected directory name '", child, "' for model '",
        model_config->name(), "', directory name must equal model name");
    }

    tfs::ModelConfig* tfs_config =
      tfs_model_configs->mutable_model_config_list()->add_config();
    tfs_config->set_name(model_config->name());
    tfs_config->set_base_path(full_path);
    tfs_config->set_model_platform(model_config->platform());

    // Create the appropriate TFS version policy from the model
    // configuration policy.
    if (model_config->version_policy().has_latest()) {
      tfs::FileSystemStoragePathSourceConfig::ServableVersionPolicy::Latest
        latest;
      latest.set_num_versions(
        model_config->version_policy().latest().num_versions());
      tfs_config->mutable_model_version_policy()->mutable_latest()->CopyFrom(
        latest);
    } else if (model_config->version_policy().has_all()) {
      tfs::FileSystemStoragePathSourceConfig::ServableVersionPolicy::All all;
      tfs_config->mutable_model_version_policy()->mutable_all()->CopyFrom(all);
    } else if (model_config->version_policy().has_specific()) {
      tfs::FileSystemStoragePathSourceConfig::ServableVersionPolicy::Specific
        specific;
      specific.mutable_versions()->CopyFrom(
        model_config->version_policy().specific().versions());
      tfs_config->mutable_model_version_policy()->mutable_specific()->CopyFrom(
        specific);
    } else {
      return tensorflow::errors::Internal(
        "expected version policy for model '", model_config->name());
    }
  }

  return tensorflow::Status::OK();
}

bool
ModelConfigManager::CompareModelConfigs(
  const ModelConfigMap& next, std::set<std::string>* added,
  std::set<std::string>* removed)
{
  ModelConfigManager* singleton = GetSingleton();

  std::set<std::string> current_names, next_names;
  for (const auto& p : singleton->configs_) {
    current_names.insert(p.first);
  }
  for (const auto& p : next) {
    next_names.insert(p.first);
  }

  if (added != nullptr) {
    std::set_difference(
      next_names.begin(), next_names.end(), current_names.begin(),
      current_names.end(), std::inserter(*added, added->end()));
  }

  if (removed != nullptr) {
    std::set_difference(
      current_names.begin(), current_names.end(), next_names.begin(),
      next_names.end(), std::inserter(*removed, removed->end()));
  }

  return current_names != next_names;
}


}}  // namespace nvidia::inferenceserver
