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

#include "src/backends/tensorflow/autofill.h"

#include "src/backends/tensorflow/tensorflow_backend_tf.h"
#include "src/backends/tensorflow/tf_utils.h"
#include "src/core/autofill.h"
#include "src/core/constants.h"
#include "src/core/filesystem.h"
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"

namespace nvidia { namespace inferenceserver {

//
// AutoFillSavedModelImpl
//
class AutoFillSavedModelImpl : public AutoFill {
 public:
  AutoFillSavedModelImpl(
      const std::string& model_name, const std::string& savedmodel_dirname,
      TFWorkspace* tfws)
      : AutoFill(model_name), savedmodel_dirname_(savedmodel_dirname),
        tfws_(tfws)
  {
  }

  Status Fix(ModelConfig* config) override;

 private:
  const std::string savedmodel_dirname_;
  std::unique_ptr<TFWorkspace> tfws_;
};

Status
AutoFillSavedModelImpl::Fix(ModelConfig* config)
{
  config->set_platform(kTensorFlowSavedModelPlatform);

  if (config->name().empty()) {
    config->set_name(model_name_);
  }

  if (config->default_model_filename().empty()) {
    config->set_default_model_filename(savedmodel_dirname_);
  }

  const TFWorkspace::IOList& inputs = tfws_->Inputs();
  const TFWorkspace::IOList& outputs = tfws_->Outputs();

  // Assume model doesn't support batching unless we see a batch
  // dimension (-1) on signature of every model input and output.
  bool sig_supports_batch = true;
  if (config->input().size() == 0) {
    for (const auto& io : inputs) {
      if ((io.shape_.size() == 0) || (io.shape_[0] != -1)) {
        sig_supports_batch = false;
      }
    }
  }
  if (config->output().size() == 0) {
    for (const auto& io : outputs) {
      if ((io.shape_.size() == 0) || (io.shape_[0] != -1)) {
        sig_supports_batch = false;
      }
    }
  }

  // If max-batch-size is explicitly set to non-zero but the model
  // signature doesn't support batching then can't autofill.
  if (!sig_supports_batch && (config->max_batch_size() > 0)) {
    return Status(
        RequestStatusCode::INTERNAL,
        "unable to autofill for '" + model_name_ +
            "', configuration specified max-batch " +
            std::to_string(config->max_batch_size()) +
            " but model signature does not support batching");
  }

  // Set max-batch-size to 1 if the model signature supports it and it
  // is not already set.
  if (sig_supports_batch && (config->max_batch_size() == 0)) {
    config->set_max_batch_size(1);
  }

  // Inputs
  if (config->input().size() == 0) {
    for (const auto& io : inputs) {
      ModelInput* config_input = config->add_input();
      config_input->set_name(io.name_);
      config_input->set_data_type(ConvertDataType(io.data_type_));

      // The model signature supports batching then the first
      // dimension is -1 and should not appear in the model
      // configuration 'dims' that we are creating.
      for (size_t i = (sig_supports_batch ? 1 : 0); i < io.shape_.size(); ++i) {
        config_input->mutable_dims()->Add(io.shape_[i]);
      }

      // If input dims are empty then must use a reshape for the
      // input, since 'dims' is not allowed to be empty.
      if (config_input->dims_size() == 0) {
        config_input->mutable_dims()->Add(1);
        config_input->mutable_reshape();
      }
    }
  }

  // Outputs
  if (config->output().size() == 0) {
    for (const auto& io : outputs) {
      ModelOutput* config_output = config->add_output();
      config_output->set_name(io.name_);
      config_output->set_data_type(ConvertDataType(io.data_type_));

      // The model signature supports batching then the first
      // dimension is -1 and should not appear in the model
      // configuration 'dims' that we are creating.
      for (size_t i = (sig_supports_batch ? 1 : 0); i < io.shape_.size(); ++i) {
        config_output->mutable_dims()->Add(io.shape_[i]);
      }

      // If output dims are empty then must use a reshape for the
      // output, since 'dims' is not allowed to be empty.
      if (config_output->dims_size() == 0) {
        config_output->mutable_dims()->Add(1);
        config_output->mutable_reshape();
      }
    }
  }

  return Status::Success;
}

Status
AutoFillSavedModel::Create(
    const std::string& model_name,
    const std::shared_ptr<BackendConfig>& backend_config,
    const std::string& model_path, std::unique_ptr<AutoFill>* autofill)
{
  std::set<std::string> version_dirs;
  RETURN_IF_ERROR(GetDirectorySubdirs(model_path, &version_dirs));

  // There must be at least one version directory that we can inspect
  // to attempt to determine the platform. For now we only handle the
  // case where there is one version directory.
  if (version_dirs.size() != 1) {
    return Status(
        RequestStatusCode::INTERNAL,
        "unable to autofill for '" + model_name + "' due to multiple versions");
  }

  const auto version_path = JoinPath({model_path, *(version_dirs.begin())});

  // There must be a single savedmodel directory within the version
  // directory...
  std::set<std::string> savedmodel_dirs;
  RETURN_IF_ERROR(GetDirectorySubdirs(version_path, &savedmodel_dirs));
  if (savedmodel_dirs.size() != 1) {
    return Status(
        RequestStatusCode::INTERNAL,
        "unable to autofill for '" + model_name +
            "', unable to find savedmodel directory");
  }

  const std::string savedmodel_dir = *(savedmodel_dirs.begin());
  const auto savedmodel_path = JoinPath({version_path, savedmodel_dir});

  auto graphdef_backend_config =
      std::static_pointer_cast<GraphDefBackendFactory::Config>(backend_config);

  TFWorkspace* tfws = nullptr;
  TFWorkspace::Error error = TFWorkspaceCreateFromSavedModel(
      &tfws, model_name, savedmodel_path, TFWorkspace::NO_GPU_DEVICE,
      false /* have_graph */, 0 /* graph_level */,
      graphdef_backend_config->allow_gpu_memory_growth,
      graphdef_backend_config->per_process_gpu_memory_fraction,
      graphdef_backend_config->allow_soft_placement);
  if (!error.IsOk()) {
    return Status(RequestStatusCode::INTERNAL, error.Message());
  }

  autofill->reset(new AutoFillSavedModelImpl(model_name, savedmodel_dir, tfws));
  return Status::Success;
}

//
// AutoFillGraphDefImpl
//
class AutoFillGraphDefImpl : public AutoFill {
 public:
  AutoFillGraphDefImpl(const std::string& model_name) : AutoFill(model_name) {}
  Status Fix(ModelConfig* config) override;
};

Status
AutoFillGraphDefImpl::Fix(ModelConfig* config)
{
  config->set_platform(kTensorFlowGraphDefPlatform);

  if (config->name().empty()) {
    config->set_name(model_name_);
  }

  return Status::Success;
}

Status
AutoFillGraphDef::Create(
    const std::string& model_name, const std::string& model_path,
    std::unique_ptr<AutoFill>* autofill)
{
  std::set<std::string> version_dirs;
  RETURN_IF_ERROR(GetDirectorySubdirs(model_path, &version_dirs));

  // There must be at least one version directory that we can inspect
  // to attempt to determine the platform. For now we only handle the
  // case where there is one version directory.
  if (version_dirs.size() != 1) {
    return Status(
        RequestStatusCode::INTERNAL,
        "unable to autofill for '" + model_name + "' due to multiple versions");
  }

  const auto version_path = JoinPath({model_path, *(version_dirs.begin())});

  // There must be a single graphdef file within the version
  // directory...
  std::set<std::string> graphdef_files;
  RETURN_IF_ERROR(GetDirectoryFiles(version_path, &graphdef_files));
  if (graphdef_files.size() != 1) {
    return Status(
        RequestStatusCode::INTERNAL, "unable to autofill for '" + model_name +
                                         "', unable to find graphdef file");
  }

  const std::string graphdef_file = *(graphdef_files.begin());
  const auto graphdef_path = JoinPath({version_path, graphdef_file});

  // If find a file named with the default graphdef name then assume
  // it is a graphdef. We could be smarter here and try to parse to
  // see if it really is a graphdef. We could also guess thae
  // placeholders are inputs... but we have no way to know what the
  // outputs are.
  if (graphdef_file != kTensorFlowGraphDefFilename) {
    return Status(
        RequestStatusCode::INTERNAL,
        "unable to autofill for '" + model_name +
            "', unable to find graphdef file named '" +
            kTensorFlowGraphDefFilename + "'");
  }

  autofill->reset(new AutoFillGraphDefImpl(model_name));
  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
