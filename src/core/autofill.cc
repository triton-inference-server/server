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

#include "src/core/autofill.h"

#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/servables/tensorflow/autofill.h"
#include "src/servables/tensorrt/autofill.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"

namespace nvidia { namespace inferenceserver {

//
// AutoFillNull
//
class AutoFillNull : public AutoFill {
 public:
  static tensorflow::Status Create(std::unique_ptr<AutoFillNull>* autofill);
  tensorflow::Status Fix(ModelConfig* config);

 private:
  AutoFillNull() : AutoFill(std::string()) {}
};

tensorflow::Status
AutoFillNull::Create(std::unique_ptr<AutoFillNull>* autofill)
{
  autofill->reset(new AutoFillNull);
  return tensorflow::Status::OK();
}

tensorflow::Status
AutoFillNull::Fix(ModelConfig* config)
{
  return tensorflow::Status::OK();
}

//
// AutoFillSimple
//
class AutoFillSimple : public AutoFill {
 public:
  static tensorflow::Status Create(
    const std::string& model_name, std::unique_ptr<AutoFillSimple>* autofill);
  tensorflow::Status Fix(ModelConfig* config);

 private:
  AutoFillSimple(const std::string& model_name) : AutoFill(model_name) {}
};

tensorflow::Status
AutoFillSimple::Create(
  const std::string& model_name, std::unique_ptr<AutoFillSimple>* autofill)
{
  autofill->reset(new AutoFillSimple(model_name));
  return tensorflow::Status::OK();
}

tensorflow::Status
AutoFillSimple::Fix(ModelConfig* config)
{
  // Set name if not already set.
  if (config->name().empty()) {
    config->set_name(model_name_);
  }

  return tensorflow::Status::OK();
}

//
// AutoFillNetDef
//
class AutoFillNetDef : public AutoFill {
 public:
  static tensorflow::Status Create(
    const std::string& model_name, const std::string& model_path,
    std::unique_ptr<AutoFillNetDef>* autofill);
  tensorflow::Status Fix(ModelConfig* config);

 private:
  AutoFillNetDef(const std::string& model_name) : AutoFill(model_name) {}
};

tensorflow::Status
AutoFillNetDef::Create(
  const std::string& model_name, const std::string& model_path,
  std::unique_ptr<AutoFillNetDef>* autofill)
{
  std::set<std::string> version_dirs;
  TF_RETURN_IF_ERROR(GetSubdirs(model_path, &version_dirs));

  // There must be at least one version directory that we can inspect
  // to attempt to determine the platform. For now we only handle the
  // case where there is one version directory.
  if (version_dirs.size() != 1) {
    return tensorflow::errors::Internal(
      "unable to autofill for '", model_name, "' due to multiple versions");
  }

  const auto version_path =
    tensorflow::io::JoinPath(model_path, *(version_dirs.begin()));

  // There must be a single netdef file within the version directory...
  std::set<std::string> netdef_files;
  TF_RETURN_IF_ERROR(GetFiles(version_path, &netdef_files));
  if (netdef_files.size() != 1) {
    return tensorflow::errors::Internal(
      "unable to autofill for '", model_name, "', unable to find netdef file");
  }

  const std::string netdef_file = *(netdef_files.begin());
  const auto netdef_path = tensorflow::io::JoinPath(version_path, netdef_file);

  // FIXME better than just recognize by name
  if (netdef_file != kCaffe2NetDefFilename) {
    return tensorflow::errors::Internal(
      "unable to autofill for '", model_name,
      "', unable to find netdef file named '", kCaffe2NetDefFilename, "'");
  }

  autofill->reset(new AutoFillNetDef(model_name));
  return tensorflow::Status::OK();
}

tensorflow::Status
AutoFillNetDef::Fix(ModelConfig* config)
{
  // Set name if not already set.
  if (config->name().empty()) {
    config->set_name(model_name_);
  }

  config->set_platform(kCaffe2NetDefPlatform);
  return tensorflow::Status::OK();
}

//
// AutoFill
//
tensorflow::Status
AutoFill::Create(
  const std::string& model_name, const std::string& model_path,
  const ModelConfig& config, std::unique_ptr<AutoFill>* autofill)
{
  autofill->reset();

  // If the config specifies a platform use it to create the
  // appropriate autofill object, otherwise just try creating each
  // autofill object to see if one can detect the platform.
  const Platform platform = GetPlatform(config.platform());

  if (
    (platform == Platform::PLATFORM_TENSORFLOW_SAVEDMODEL) ||
    (platform == Platform::PLATFORM_UNKNOWN)) {
    std::unique_ptr<AutoFillSavedModel> afsm;
    tensorflow::Status status =
      AutoFillSavedModel::Create(model_name, model_path, &afsm);
    if (status.ok()) {
      *autofill = std::move(afsm);
      return tensorflow::Status::OK();
    }
  }

  if (
    (platform == Platform::PLATFORM_TENSORFLOW_GRAPHDEF) ||
    (platform == Platform::PLATFORM_UNKNOWN)) {
    std::unique_ptr<AutoFillGraphDef> afgd;
    tensorflow::Status status =
      AutoFillGraphDef::Create(model_name, model_path, &afgd);
    if (status.ok()) {
      *autofill = std::move(afgd);
      return tensorflow::Status::OK();
    }
  }

  if (
    (platform == Platform::PLATFORM_TENSORRT_PLAN) ||
    (platform == Platform::PLATFORM_UNKNOWN)) {
    std::unique_ptr<AutoFillPlan> afp;
    tensorflow::Status status =
      AutoFillPlan::Create(model_name, model_path, &afp);
    if (status.ok()) {
      *autofill = std::move(afp);
      return tensorflow::Status::OK();
    }
  }

  if (
    (platform == Platform::PLATFORM_CAFFE2_NETDEF) ||
    (platform == Platform::PLATFORM_UNKNOWN)) {
    std::unique_ptr<AutoFillNetDef> afnd;
    tensorflow::Status status =
      AutoFillNetDef::Create(model_name, model_path, &afnd);
    if (status.ok()) {
      *autofill = std::move(afnd);
      return tensorflow::Status::OK();
    }
  }

  // Unable to determine the platform so just use the simple autofill,
  // or null if that fails.
  {
    std::unique_ptr<AutoFillSimple> afs;
    tensorflow::Status status = AutoFillSimple::Create(model_name, &afs);
    if (status.ok()) {
      *autofill = std::move(afs);
    } else {
      std::unique_ptr<AutoFillNull> afn;
      TF_RETURN_IF_ERROR(AutoFillNull::Create(&afn));
      *autofill = std::move(afn);
    }
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
AutoFill::GetSubdirs(const std::string& path, std::set<std::string>* subdirs)
{
  std::vector<std::string> childs;
  TF_RETURN_IF_ERROR(tensorflow::Env::Default()->GetChildren(path, &childs));

  // GetChildren() returns all descendants instead for cloud storage
  // like GCS. In such case we should filter out all non-direct
  // descendants.
  std::set<std::string> real_childs;
  for (const std::string& child : childs) {
    real_childs.insert(child.substr(0, child.find_first_of('/')));
  }

  for (const auto& child : real_childs) {
    const auto vp = tensorflow::io::JoinPath(path, child);
    if (tensorflow::Env::Default()->IsDirectory(vp).ok()) {
      subdirs->insert(child);
    }
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
AutoFill::GetFiles(const std::string& path, std::set<std::string>* files)
{
  std::vector<std::string> childs;
  TF_RETURN_IF_ERROR(tensorflow::Env::Default()->GetChildren(path, &childs));

  // GetChildren() returns all descendants instead for cloud storage
  // like GCS. In such case we should filter out all non-direct
  // descendants.
  std::set<std::string> real_childs;
  for (const std::string& child : childs) {
    real_childs.insert(child.substr(0, child.find_first_of('/')));
  }

  for (const auto& child : real_childs) {
    const auto vp = tensorflow::io::JoinPath(path, child);
    if (!tensorflow::Env::Default()->IsDirectory(vp).ok()) {
      files->insert(child);
    }
  }

  return tensorflow::Status::OK();
}

}}  // namespace nvidia::inferenceserver
