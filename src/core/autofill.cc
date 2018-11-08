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
#include "tensorflow/c/c_api.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"

namespace nvidia { namespace inferenceserver {

namespace {

DataType
ConvertDataType(tensorflow::DataType dtype)
{
  switch (dtype) {
    case tensorflow::DT_INVALID:
      return DataType::TYPE_INVALID;
    case tensorflow::DT_BOOL:
      return DataType::TYPE_BOOL;
    case tensorflow::DT_UINT8:
      return DataType::TYPE_UINT8;
    case tensorflow::DT_UINT16:
      return DataType::TYPE_UINT16;
    case tensorflow::DT_UINT32:
      return DataType::TYPE_UINT32;
    case tensorflow::DT_UINT64:
      return DataType::TYPE_UINT64;
    case tensorflow::DT_INT8:
      return DataType::TYPE_INT8;
    case tensorflow::DT_INT16:
      return DataType::TYPE_INT16;
    case tensorflow::DT_INT32:
      return DataType::TYPE_INT32;
    case tensorflow::DT_INT64:
      return DataType::TYPE_INT64;
    case tensorflow::DT_HALF:
      return DataType::TYPE_FP16;
    case tensorflow::DT_FLOAT:
      return DataType::TYPE_FP32;
    case tensorflow::DT_DOUBLE:
      return DataType::TYPE_FP64;
    default:
      break;
  }

  return DataType::TYPE_INVALID;
}

}  // namespace

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
// AutoFillSavedModel
//
class AutoFillSavedModel : public AutoFill {
 public:
  static tensorflow::Status Create(
    const std::string& model_name, const std::string& model_path,
    std::unique_ptr<AutoFillSavedModel>* autofill);
  tensorflow::Status Fix(ModelConfig* config);

 private:
  AutoFillSavedModel(
    const std::string& model_name, const std::string& savedmodel_dirname,
    const tensorflow::SignatureDef& sig)
      : AutoFill(model_name), savedmodel_dirname_(savedmodel_dirname), sig_(sig)
  {
  }

  const std::string savedmodel_dirname_;
  const tensorflow::SignatureDef sig_;
};

tensorflow::Status
AutoFillSavedModel::Create(
  const std::string& model_name, const std::string& model_path,
  std::unique_ptr<AutoFillSavedModel>* autofill)
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

  // There must be a single savedmodel directory within the version
  // directory...
  std::set<std::string> savedmodel_dirs;
  TF_RETURN_IF_ERROR(GetSubdirs(version_path, &savedmodel_dirs));
  if (savedmodel_dirs.size() != 1) {
    return tensorflow::errors::Internal(
      "unable to autofill for '", model_name,
      "', unable to find savedmodel directory");
  }

  const std::string savedmodel_dir = *(savedmodel_dirs.begin());
  const auto savedmodel_path =
    tensorflow::io::JoinPath(version_path, savedmodel_dir);
  std::unique_ptr<tensorflow::SavedModelBundle> bundle(
    new tensorflow::SavedModelBundle);

  std::unordered_set<std::string> saved_model_tags;
  saved_model_tags.insert(tensorflow::kSavedModelTagServe);

  tensorflow::SessionOptions session_options;
  tensorflow::RunOptions run_options;
  TF_RETURN_IF_ERROR(tensorflow::LoadSavedModel(
    session_options, run_options, savedmodel_path, saved_model_tags,
    bundle.get()));

  // Only autofill if there is a "serve" tag...
  bool found_serve_tag = false;
  for (const auto& tag : bundle->meta_graph_def.meta_info_def().tags()) {
    if (tag == tensorflow::kSavedModelTagServe) {
      found_serve_tag = true;
      break;
    }
  }
  if (!found_serve_tag) {
    return tensorflow::errors::Internal(
      "unable to autofill for '", model_name, "', unable to find '",
      tensorflow::kSavedModelTagServe, "' tag");
  }

  // Must have a "serving_default" signature as that is what will be
  // used to determine the inputs and outputs.
  static const std::string DEFAULT_SERVING_SIGNATURE_DEF_KEY("serving_default");
  const auto& sig_itr = bundle->meta_graph_def.signature_def().find(
    DEFAULT_SERVING_SIGNATURE_DEF_KEY);
  if (sig_itr == bundle->meta_graph_def.signature_def().end()) {
    return tensorflow::errors::InvalidArgument(
      "unable to autofill for '", model_name, "', require '",
      DEFAULT_SERVING_SIGNATURE_DEF_KEY, "' signature");
  }

  // Save the name of the savedmodel directory since it may not be the
  // default value. Save the SignatureDef as that is what we need to
  // autofill inputs and outputs.
  const tensorflow::SignatureDef& sig = sig_itr->second;

  autofill->reset(new AutoFillSavedModel(model_name, savedmodel_dir, sig));
  return tensorflow::Status::OK();
}

tensorflow::Status
AutoFillSavedModel::Fix(ModelConfig* config)
{
  config->set_platform(kTensorFlowSavedModelPlatform);

  if (config->name().empty()) {
    config->set_name(model_name_);
  }

  if (config->default_model_filename().empty()) {
    config->set_default_model_filename(savedmodel_dirname_);
  }

  // Assume model doesn't support batching unless we see a batch
  // dimension in the input or output.
  bool supports_batch = false;

  // Inputs
  if (config->input().size() == 0) {
    for (const auto& sin : sig_.inputs()) {
      ModelInput* config_input = config->add_input();
      config_input->set_name(sin.first);
      config_input->set_data_type(ConvertDataType(sin.second.dtype()));


      // The first model dimension can be -1 to serve as a placeholder
      // for batch. The batch dim doesn't appear in the configuration
      // 'dims'.
      const tensorflow::TensorShapeProto& shape = sin.second.tensor_shape();
      const bool has_batch_dim =
        (shape.dim().size() >= 1) && (shape.dim(0).size() == -1);

      for (int i = (has_batch_dim ? 1 : 0); i < shape.dim().size(); ++i) {
        config_input->mutable_dims()->Add(shape.dim(i).size());
      }

      supports_batch |= has_batch_dim;
    }
  }

  // Outputs
  if (config->output().size() == 0) {
    for (const auto& sout : sig_.outputs()) {
      ModelOutput* config_output = config->add_output();
      config_output->set_name(sout.first);
      config_output->set_data_type(ConvertDataType(sout.second.dtype()));


      // The first model dimension can be -1 to serve as a placeholder
      // for batch. The batch dim doesn't appear in the configuration
      // 'dims'.
      const tensorflow::TensorShapeProto& shape = sout.second.tensor_shape();
      const bool has_batch_dim =
        (shape.dim().size() >= 1) && (shape.dim(0).size() == -1);

      for (int i = (has_batch_dim ? 1 : 0); i < shape.dim().size(); ++i) {
        config_output->mutable_dims()->Add(shape.dim(i).size());
      }

      supports_batch |= has_batch_dim;
    }
  }

  // Set max-batch-size to 1 if the model supports it and it is not
  // already set.
  if (supports_batch && (config->max_batch_size() == 0)) {
    config->set_max_batch_size(1);
  }

  return tensorflow::Status::OK();
}

//
// AutoFillGraphDef
//
class AutoFillGraphDef : public AutoFill {
 public:
  static tensorflow::Status Create(
    const std::string& model_name, const std::string& model_path,
    std::unique_ptr<AutoFillGraphDef>* autofill);
  tensorflow::Status Fix(ModelConfig* config);

 private:
  AutoFillGraphDef(const std::string& model_name) : AutoFill(model_name) {}
};

tensorflow::Status
AutoFillGraphDef::Create(
  const std::string& model_name, const std::string& model_path,
  std::unique_ptr<AutoFillGraphDef>* autofill)
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

  // There must be a single graphdef file within the version
  // directory...
  std::set<std::string> graphdef_files;
  TF_RETURN_IF_ERROR(GetFiles(version_path, &graphdef_files));
  if (graphdef_files.size() != 1) {
    return tensorflow::errors::Internal(
      "unable to autofill for '", model_name,
      "', unable to find graphdef file");
  }

  const std::string graphdef_file = *(graphdef_files.begin());
  const auto graphdef_path =
    tensorflow::io::JoinPath(version_path, graphdef_file);

  // FIXME better than just recognize by name
  if (graphdef_file != kTensorFlowGraphDefFilename) {
    return tensorflow::errors::Internal(
      "unable to autofill for '", model_name,
      "', unable to find graphdef file named '", kTensorFlowGraphDefFilename,
      "'");
  }

  autofill->reset(new AutoFillGraphDef(model_name));
  return tensorflow::Status::OK();
}

tensorflow::Status
AutoFillGraphDef::Fix(ModelConfig* config)
{
  // Set name if not already set.
  if (config->name().empty()) {
    config->set_name(model_name_);
  }

  config->set_platform(kTensorFlowGraphDefPlatform);
  return tensorflow::Status::OK();
}

//
// AutoFillPlan
//
class AutoFillPlan : public AutoFill {
 public:
  static tensorflow::Status Create(
    const std::string& model_name, const std::string& model_path,
    std::unique_ptr<AutoFillPlan>* autofill);
  tensorflow::Status Fix(ModelConfig* config);

 private:
  AutoFillPlan(const std::string& model_name) : AutoFill(model_name) {}
};

tensorflow::Status
AutoFillPlan::Create(
  const std::string& model_name, const std::string& model_path,
  std::unique_ptr<AutoFillPlan>* autofill)
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

  // There must be a single plan file within the version directory...
  std::set<std::string> plan_files;
  TF_RETURN_IF_ERROR(GetFiles(version_path, &plan_files));
  if (plan_files.size() != 1) {
    return tensorflow::errors::Internal(
      "unable to autofill for '", model_name, "', unable to find plan file");
  }

  const std::string plan_file = *(plan_files.begin());
  const auto plan_path = tensorflow::io::JoinPath(version_path, plan_file);

  // FIXME better than just recognize by name
  if (plan_file != kTensorRTPlanFilename) {
    return tensorflow::errors::Internal(
      "unable to autofill for '", model_name,
      "', unable to find plan file named '", kTensorRTPlanFilename, "'");
  }

  autofill->reset(new AutoFillPlan(model_name));
  return tensorflow::Status::OK();
}

tensorflow::Status
AutoFillPlan::Fix(ModelConfig* config)
{
  // Set name if not already set.
  if (config->name().empty()) {
    config->set_name(model_name_);
  }

  config->set_platform(kTensorRTPlanPlatform);
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
