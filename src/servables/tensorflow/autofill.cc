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

#include "src/servables/tensorflow/autofill.h"

#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/servables/tensorflow/loader.h"
#include "src/servables/tensorflow/tf_utils.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"

namespace nvidia { namespace inferenceserver {

//
// AutoFillSavedModel
//
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

  std::unique_ptr<tensorflow::SavedModelBundle> bundle;
  tensorflow::SessionOptions session_options;
  tensorflow::SignatureDef sig;
  TF_RETURN_IF_ERROR(LoadSavedModel(
    model_name, savedmodel_path, session_options, &bundle, &sig));

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

  // If find a file named with the default graphdef name then assume
  // it is a graphdef. We could be smarter here and try to parse to
  // see if it really is a graphdef. We could also guess thae
  // placeholders are inputs... but we have no way to know what the
  // outputs are.
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
  config->set_platform(kTensorFlowGraphDefPlatform);

  if (config->name().empty()) {
    config->set_name(model_name_);
  }

  return tensorflow::Status::OK();
}

}}  // namespace nvidia::inferenceserver
