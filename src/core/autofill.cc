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
#include "src/servables/caffe2/autofill.h"
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
  static Status Create(std::unique_ptr<AutoFillNull>* autofill);
  Status Fix(ModelConfig* config);

 private:
  AutoFillNull() : AutoFill(std::string()) {}
};

Status
AutoFillNull::Create(std::unique_ptr<AutoFillNull>* autofill)
{
  autofill->reset(new AutoFillNull);
  return Status::Success;
}

Status
AutoFillNull::Fix(ModelConfig* config)
{
  return Status::Success;
}

//
// AutoFillSimple
//
class AutoFillSimple : public AutoFill {
 public:
  static Status Create(
      const std::string& model_name, std::unique_ptr<AutoFillSimple>* autofill);
  Status Fix(ModelConfig* config);

 private:
  AutoFillSimple(const std::string& model_name) : AutoFill(model_name) {}
};

Status
AutoFillSimple::Create(
    const std::string& model_name, std::unique_ptr<AutoFillSimple>* autofill)
{
  autofill->reset(new AutoFillSimple(model_name));
  return Status::Success;
}

Status
AutoFillSimple::Fix(ModelConfig* config)
{
  // Set name if not already set.
  if (config->name().empty()) {
    config->set_name(model_name_);
  }

  return Status::Success;
}

//
// AutoFill
//
Status
AutoFill::Create(
    const std::string& model_name,
    const tfs::PlatformConfigMap& platform_config_map,
    const std::string& model_path, const ModelConfig& config,
    std::unique_ptr<AutoFill>* autofill)
{
  autofill->reset();

  // If the config specifies a platform use it to create the
  // appropriate autofill object, otherwise just try creating each
  // autofill object to see if one can detect the platform.
  const Platform platform = GetPlatform(config.platform());

  if ((platform == Platform::PLATFORM_TENSORFLOW_SAVEDMODEL) ||
      (platform == Platform::PLATFORM_UNKNOWN)) {
    std::unique_ptr<AutoFillSavedModel> afsm;
    ::google::protobuf::Any platform_config;
    auto it = platform_config_map.platform_configs().find(
        kTensorFlowSavedModelPlatform);
    if (it != platform_config_map.platform_configs().end()) {
      platform_config = it->second.source_adapter_config();
    }
    Status status = AutoFillSavedModel::Create(
        model_name, platform_config, model_path, &afsm);
    if (status.IsOk()) {
      *autofill = std::move(afsm);
      return Status::Success;
    }
  }

  if ((platform == Platform::PLATFORM_TENSORFLOW_GRAPHDEF) ||
      (platform == Platform::PLATFORM_UNKNOWN)) {
    std::unique_ptr<AutoFillGraphDef> afgd;
    Status status = AutoFillGraphDef::Create(model_name, model_path, &afgd);
    if (status.IsOk()) {
      *autofill = std::move(afgd);
      return Status::Success;
    }
  }

  if ((platform == Platform::PLATFORM_TENSORRT_PLAN) ||
      (platform == Platform::PLATFORM_UNKNOWN)) {
    std::unique_ptr<AutoFillPlan> afp;
    Status status = AutoFillPlan::Create(model_name, model_path, &afp);
    if (status.IsOk()) {
      *autofill = std::move(afp);
      return Status::Success;
    }
  }

  if ((platform == Platform::PLATFORM_CAFFE2_NETDEF) ||
      (platform == Platform::PLATFORM_UNKNOWN)) {
    std::unique_ptr<AutoFillNetDef> afnd;
    Status status = AutoFillNetDef::Create(model_name, model_path, &afnd);
    if (status.IsOk()) {
      *autofill = std::move(afnd);
      return Status::Success;
    }
  }

  // Unable to determine the platform so just use the simple autofill,
  // or null if that fails.
  {
    std::unique_ptr<AutoFillSimple> afs;
    Status status = AutoFillSimple::Create(model_name, &afs);
    if (status.IsOk()) {
      *autofill = std::move(afs);
    } else {
      std::unique_ptr<AutoFillNull> afn;
      RETURN_IF_ERROR(AutoFillNull::Create(&afn));
      *autofill = std::move(afn);
    }
  }

  return Status::Success;
}

Status
AutoFill::GetSubdirs(const std::string& path, std::set<std::string>* subdirs)
{
  std::vector<std::string> childs;
  RETURN_IF_TF_ERROR(tensorflow::Env::Default()->GetChildren(path, &childs));

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

  return Status::Success;
}

Status
AutoFill::GetFiles(const std::string& path, std::set<std::string>* files)
{
  std::vector<std::string> childs;
  RETURN_IF_TF_ERROR(tensorflow::Env::Default()->GetChildren(path, &childs));

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

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
