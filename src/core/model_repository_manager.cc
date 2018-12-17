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

#include "src/core/model_repository_manager.h"

#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_statistics.h"

namespace nvidia { namespace inferenceserver {

namespace {

int64_t
GetModifiedTime(const std::string& path)
{
  // If there is an error in any step the fall-back default
  // modification time is 0. This means that in error cases 'path'
  // will show as not modified. This is the safe fall-back to avoid
  // assuming a model is constantly being modified.

  // If 'path' is a file return its mtime.
  if (!tensorflow::Env::Default()->IsDirectory(path).ok()) {
    tensorflow::FileStatistics stat;
    if (!tensorflow::Env::Default()->Stat(path, &stat).ok()) {
      LOG_ERROR << "Failed to determine modification time for '" << path
                << "', assuming 0";
      return 0;
    }

    return stat.mtime_nsec;
  }

  // 'path' is a directory. Return the most recent mtime of the
  // contents of the directory.
  //
  // GetChildren() returns all descendants instead for cloud storage
  // like GCS.  In such case we should filter out all non-direct
  // descendants.
  std::vector<std::string> children;
  if (!tensorflow::Env::Default()->GetChildren(path, &children).ok()) {
    LOG_ERROR << "Failed to determine modification time for '" << path
              << "', assuming 0";
  }

  std::set<std::string> real_children;
  for (size_t i = 0; i < children.size(); ++i) {
    const std::string& child = children[i];
    real_children.insert(child.substr(0, child.find_first_of('/')));
  }

  int64_t mtime = 0;
  for (const auto& child : real_children) {
    const auto full_path = tensorflow::io::JoinPath(path, child);
    mtime = std::max(mtime, GetModifiedTime(full_path));
  }

  return mtime;
}

// Return true if any file in the subdirectory root at 'path' has been
// modified more recently than 'last'. Return the most-recent modified
// time in 'last'.
bool
IsModified(const std::string& path, int64_t* last_ns)
{
  const int64_t repo_ns = GetModifiedTime(path);
  bool modified = repo_ns > *last_ns;
  *last_ns = repo_ns;
  return modified;
}

}  // namespace

ModelRepositoryManager* ModelRepositoryManager::singleton = nullptr;

ModelRepositoryManager::ModelRepositoryManager(
    const std::string& repository_path, const bool autofill)
    : repository_path_(repository_path), autofill_(autofill)
{
}

tensorflow::Status
ModelRepositoryManager::Create(
    const std::string& repository_path, const bool autofill)
{
  if (singleton != nullptr) {
    return tensorflow::errors::AlreadyExists(
        "ModelRepositoryManager singleton already created");
  }

  singleton = new ModelRepositoryManager(repository_path, autofill);

  return tensorflow::Status::OK();
}

tensorflow::Status
ModelRepositoryManager::GetModelConfig(
    const std::string& name, ModelConfig* model_config)
{
  std::lock_guard<std::mutex> lock(singleton->infos_mu_);

  const auto itr = singleton->infos_.find(name);
  if (itr == singleton->infos_.end()) {
    return tensorflow::errors::NotFound(
        "no configuration for model '", name, "'");
  }

  *model_config = itr->second.model_config_;
  return tensorflow::Status::OK();
}

tensorflow::Status
ModelRepositoryManager::GetTFSModelConfig(
    const std::string& name, tfs::ModelConfig* tfs_model_config)
{
  std::lock_guard<std::mutex> lock(singleton->infos_mu_);

  const auto itr = singleton->infos_.find(name);
  if (itr == singleton->infos_.end()) {
    return tensorflow::errors::NotFound(
        "no TFS configuration for model '", name, "'");
  }

  *tfs_model_config = itr->second.tfs_model_config_;
  return tensorflow::Status::OK();
}

tensorflow::Status
ModelRepositoryManager::GetModelPlatform(
    const std::string& name, Platform* platform)
{
  std::lock_guard<std::mutex> lock(singleton->infos_mu_);

  const auto itr = singleton->infos_.find(name);
  if (itr == singleton->infos_.end()) {
    *platform = Platform::PLATFORM_UNKNOWN;
  } else {
    *platform = itr->second.platform_;
  }

  if (*platform == Platform::PLATFORM_UNKNOWN) {
    return tensorflow::errors::NotFound(
        "unknown platform for model '", name, "'");
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
ModelRepositoryManager::Poll(
    std::set<std::string>* added, std::set<std::string>* deleted,
    std::set<std::string>* modified, std::set<std::string>* unmodified)
{
  // Serialize all polling operation...
  std::lock_guard<std::mutex> lock(singleton->poll_mu_);

  added->clear();
  deleted->clear();
  modified->clear();
  unmodified->clear();

  // We don't modify 'infos_' in place to minimize how long we need to
  // hold the lock and also prevent any partial changes to do an error
  // during processing.
  ModelInfoMap new_infos;

  // Each subdirectory of repository path is a model directory from
  // which we read the model configuration.
  std::vector<std::string> children;
  TF_RETURN_IF_ERROR(tensorflow::Env::Default()->GetChildren(
      singleton->repository_path_, &children));

  // GetChildren() returns all descendants instead for cloud storage
  // like GCS.  In such case we should filter out all non-direct
  // descendants.
  std::set<std::string> real_children;
  for (size_t i = 0; i < children.size(); ++i) {
    const std::string& child = children[i];
    real_children.insert(child.substr(0, child.find_first_of('/')));
  }

  for (const auto& child : real_children) {
    const auto full_path =
        tensorflow::io::JoinPath(singleton->repository_path_, child);
    if (!tensorflow::Env::Default()->IsDirectory(full_path).ok()) {
      continue;
    }

    // If 'child' is a new model or an existing model that has been
    // modified since the last time it was polled, then need to
    // (re)load, normalize and validate the configuration.
    bool need_load = false;
    int64_t mtime_ns;
    const auto iitr = singleton->infos_.find(child);
    if (iitr == singleton->infos_.end()) {
      added->insert(child);
      mtime_ns = GetModifiedTime(std::string(full_path));
      need_load = true;
    } else {
      mtime_ns = iitr->second.mtime_nsec_;
      if (IsModified(std::string(full_path), &mtime_ns)) {
        modified->insert(child);
        need_load = true;
      } else {
        unmodified->insert(child);
        const auto& ret = new_infos.emplace(child, iitr->second);
        if (!ret.second) {
          return tensorflow::errors::AlreadyExists(
              "unexpected model info for model '", child, "'");
        }
      }
    }

    if (need_load) {
      const auto& ret = new_infos.emplace(child, ModelInfo{});
      if (!ret.second) {
        return tensorflow::errors::AlreadyExists(
            "unexpected model info for model '", child, "'");
      }

      ModelInfo& model_info = ret.first->second;
      ModelConfig& model_config = model_info.model_config_;
      tfs::ModelConfig& tfs_config = model_info.tfs_model_config_;
      model_info.mtime_nsec_ = mtime_ns;

      // If enabled, try to automatically generate missing parts of
      // the model configuration (autofill) from the model
      // definition. In all cases normalize and validate the config.
      TF_RETURN_IF_ERROR(GetNormalizedModelConfig(
          full_path, singleton->autofill_, &model_config));
      TF_RETURN_IF_ERROR(ValidateModelConfig(model_config, std::string()));

      model_info.platform_ = GetPlatform(model_config.platform());

      // Make sure the name of the model matches the name of the
      // directory. This is a somewhat arbitrary requirement but seems
      // like good practice to require it of the user. It also acts as a
      // check to make sure we don't have two different models with the
      // same name.
      if (model_config.name() != child) {
        return tensorflow::errors::InvalidArgument(
            "unexpected directory name '", child, "' for model '",
            model_config.name(), "', directory name must equal model name");
      }

      tfs_config.set_name(model_config.name());
      tfs_config.set_base_path(full_path);
      tfs_config.set_model_platform(model_config.platform());

      // Create the appropriate TFS version policy from the model
      // configuration policy.
      if (model_config.version_policy().has_latest()) {
        tfs::FileSystemStoragePathSourceConfig::ServableVersionPolicy::Latest
            latest;
        latest.set_num_versions(
            model_config.version_policy().latest().num_versions());
        tfs_config.mutable_model_version_policy()->mutable_latest()->CopyFrom(
            latest);
      } else if (model_config.version_policy().has_all()) {
        tfs::FileSystemStoragePathSourceConfig::ServableVersionPolicy::All all;
        tfs_config.mutable_model_version_policy()->mutable_all()->CopyFrom(all);
      } else if (model_config.version_policy().has_specific()) {
        tfs::FileSystemStoragePathSourceConfig::ServableVersionPolicy::Specific
            specific;
        specific.mutable_versions()->CopyFrom(
            model_config.version_policy().specific().versions());
        tfs_config.mutable_model_version_policy()->mutable_specific()->CopyFrom(
            specific);
      } else {
        return tensorflow::errors::Internal(
            "expected version policy for model '", model_config.name());
      }
    }
  }

  // Anything in 'infos_' that is not in "added", "modified", or
  // "unmodified" is deleted.
  for (const auto& pr : singleton->infos_) {
    if ((added->find(pr.first) == added->end()) &&
        (modified->find(pr.first) == modified->end()) &&
        (unmodified->find(pr.first) == unmodified->end())) {
      deleted->insert(pr.first);
    }
  }

  // Swap the new infos in place under a short-lived lock and only if
  // there were no errors encountered during polling.
  {
    std::lock_guard<std::mutex> lock(singleton->infos_mu_);
    singleton->infos_.swap(new_infos);
  }

  return tensorflow::Status::OK();
}

}}  // namespace nvidia::inferenceserver
