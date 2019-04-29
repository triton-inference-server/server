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

#include "src/servables/tensorflow/autofill.h"

#include "src/core/autofill.h"
#include "src/core/constants.h"
#include "src/core/filesystem.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/servables/tensorflow/loader.h"
#include "src/servables/tensorflow/savedmodel_bundle.pb.h"
#include "src/servables/tensorflow/tf_utils.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/lib/io/path.h"

namespace nvidia { namespace inferenceserver {

//
// AutoFillSavedModelImpl
//
class AutoFillSavedModelImpl : public AutoFill {
 public:
  AutoFillSavedModelImpl(
      const std::string& model_name, const std::string& savedmodel_dirname,
      const tensorflow::SignatureDef& sig)
      : AutoFill(model_name), savedmodel_dirname_(savedmodel_dirname), sig_(sig)
  {
  }

  Status Fix(ModelConfig* config) override;

 private:
  const std::string savedmodel_dirname_;
  const tensorflow::SignatureDef sig_;
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

  // Assume model doesn't support batching unless we see a batch
  // dimension (-1) on signature of every model input and output.
  bool sig_supports_batch = true;
  if (config->input().size() == 0) {
    for (const auto& sin : sig_.inputs()) {
      const tensorflow::TensorShapeProto& shape = sin.second.tensor_shape();
      if ((shape.dim().size() == 0) || (shape.dim(0).size() != -1)) {
        sig_supports_batch = false;
      }
    }
  }
  if (config->output().size() == 0) {
    for (const auto& sout : sig_.outputs()) {
      const tensorflow::TensorShapeProto& shape = sout.second.tensor_shape();
      if ((shape.dim().size() == 0) || (shape.dim(0).size() != -1)) {
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
    for (const auto& sin : sig_.inputs()) {
      ModelInput* config_input = config->add_input();
      config_input->set_name(sin.first);

      const DataType dt = ConvertDataType(sin.second.dtype());
      if (dt == DataType::TYPE_INVALID) {
        return Status(
            RequestStatusCode::INTERNAL,
            "unable to autofill for '" + model_name_ +
                "', unsupported data-type '" +
                tensorflow::DataType_Name(sin.second.dtype()) + "'");
      }

      config_input->set_data_type(dt);

      // The model signature supports batching then the first
      // dimension is -1 and should not appear in the model
      // configuration 'dims' that we are creating.
      const tensorflow::TensorShapeProto& shape = sin.second.tensor_shape();
      for (int i = (sig_supports_batch ? 1 : 0); i < shape.dim().size(); ++i) {
        config_input->mutable_dims()->Add(shape.dim(i).size());
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
    for (const auto& sout : sig_.outputs()) {
      ModelOutput* config_output = config->add_output();
      config_output->set_name(sout.first);

      const DataType dt = ConvertDataType(sout.second.dtype());
      if (dt == DataType::TYPE_INVALID) {
        return Status(
            RequestStatusCode::INTERNAL,
            "unable to autofill for '" + model_name_ +
                "', unsupported data-type '" +
                tensorflow::DataType_Name(sout.second.dtype()) + "'");
      }

      config_output->set_data_type(dt);

      // The model signature supports batching then the first
      // dimension is -1 and should not appear in the model
      // configuration 'dims' that we are creating.
      const tensorflow::TensorShapeProto& shape = sout.second.tensor_shape();
      for (int i = (sig_supports_batch ? 1 : 0); i < shape.dim().size(); ++i) {
        config_output->mutable_dims()->Add(shape.dim(i).size());
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
    const ::google::protobuf::Any& platform_config,
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

  std::unique_ptr<tensorflow::SavedModelBundle> bundle;
  SavedModelBundleSourceAdapterConfig saved_model_config;
  platform_config.UnpackTo(&saved_model_config);
  tensorflow::SessionOptions session_options;
  session_options.config = saved_model_config.session_config();
  tensorflow::SignatureDef sig;
  RETURN_IF_ERROR(LoadSavedModel(
      model_name, savedmodel_path, session_options, &bundle, &sig));

  autofill->reset(new AutoFillSavedModelImpl(model_name, savedmodel_dir, sig));
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
