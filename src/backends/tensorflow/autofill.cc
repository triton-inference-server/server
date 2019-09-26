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
      TRTISTF_Model* trtistf_model)
      : AutoFill(model_name), savedmodel_dirname_(savedmodel_dirname),
        trtistf_model_(trtistf_model, TRTISTF_ModelDelete)
  {
  }

  Status Fix(ModelConfig* config) override;

 private:
  template <class ModelIO>
  using IOList = ::google::protobuf::RepeatedPtrField<ModelIO>;

  template <class IO>
  Status FixIOConfig(
      const TRTISTF_IOList* reference_list, IOList<IO>* mutable_list);

  Status FixBatchingSupport(ModelConfig* config);

  const std::string savedmodel_dirname_;
  std::unique_ptr<TRTISTF_Model, decltype(&TRTISTF_ModelDelete)> trtistf_model_;
  bool model_support_batching_;
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

  // Validate and fill 'max_batch_size' based on model signature and config hint
  RETURN_IF_ERROR(FixBatchingSupport(config));

  // Inputs
  const TRTISTF_IOList* inputs = TRTISTF_ModelInputs(trtistf_model_.get());
  RETURN_IF_ERROR(FixIOConfig(inputs, config->mutable_input()));

  // Outputs
  const TRTISTF_IOList* outputs = TRTISTF_ModelOutputs(trtistf_model_.get());
  RETURN_IF_ERROR(FixIOConfig(outputs, config->mutable_output()));

  return Status::Success;
}

Status
AutoFillSavedModelImpl::FixBatchingSupport(ModelConfig* config)
{
  const TRTISTF_IOList* inputs = TRTISTF_ModelInputs(trtistf_model_.get());
  const TRTISTF_IOList* outputs = TRTISTF_ModelOutputs(trtistf_model_.get());

  // Assume model doesn't support batching unless we see a batch
  // dimension (-1) on signature of every model input and output.
  bool sig_supports_batch = true;
  for (const TRTISTF_IOList* itr = inputs; itr != nullptr; itr = itr->next_) {
    TRTISTF_IO* io = itr->io_;
    if ((io->shape_->rank_ == 0) || (io->shape_->dims_[0] != -1)) {
      sig_supports_batch = false;
    }
  }
  for (const TRTISTF_IOList* itr = outputs; itr != nullptr; itr = itr->next_) {
    TRTISTF_IO* io = itr->io_;
    if ((io->shape_->rank_ == 0) || (io->shape_->dims_[0] != -1)) {
      sig_supports_batch = false;
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

  // 'model_support_batching_' is set to be true when all model inputs have
  // variable size first dimension, but it is not necessary to be the case
  // (i.e. non-batch model with variable size tensors). As 'max_batch_size == 0'
  // from existing config is also ambiguous, it can be either unspecified or
  // no-batch, autofill will check specified input/output (if any) for hint.
  model_support_batching_ = sig_supports_batch;
  if (model_support_batching_ && (config->max_batch_size() == 0)) {
    bool config_batch_hint = false;
    if ((config->input_size() != 0) || (config->output_size() != 0)) {
      for (const auto& config_io : config->input()) {
        if (!config_io.dims().empty()) {
          // look up corresponding io info from model
          for (const TRTISTF_IOList* itr = inputs; itr != nullptr;
               itr = itr->next_) {
            TRTISTF_IO* io = itr->io_;
            if (config_io.name() == io->name_) {
              bool should_batch =
                  (static_cast<int>(io->shape_->rank_) ==
                   (config_io.dims_size() + 1));
              // inconsistent hint
              if (config_batch_hint &&
                  (model_support_batching_ != should_batch)) {
                return Status(
                    RequestStatusCode::INTERNAL,
                    "unable to autofill for '" + model_name_ +
                        "', model tensor configurations are contradicting " +
                        "each other in terms of whether batching is supported");
              }
              config_batch_hint = true;
              model_support_batching_ = should_batch;
            }
          }
        }
      }
      for (const auto& config_io : config->output()) {
        if (!config_io.dims().empty()) {
          // look up corresponding io info from model
          for (const TRTISTF_IOList* itr = outputs; itr != nullptr;
               itr = itr->next_) {
            TRTISTF_IO* io = itr->io_;
            if (config_io.name() == io->name_) {
              bool should_batch =
                  (static_cast<int>(io->shape_->rank_) ==
                   (config_io.dims_size() + 1));
              // inconsistent hint
              if (config_batch_hint &&
                  (model_support_batching_ != should_batch)) {
                return Status(
                    RequestStatusCode::INTERNAL,
                    "unable to autofill for '" + model_name_ +
                        "', model tensor configurations are contradicting " +
                        "each other in terms of whether batching is supported");
              }
              config_batch_hint = true;
              model_support_batching_ = should_batch;
            }
          }
        }
      }
    }
  }

  // Set max-batch-size to 1 if the model signature and config hint agree
  if (config->max_batch_size() == 0) {
    config->set_max_batch_size(model_support_batching_ ? 1 : 0);
  }
  return Status::Success;
}

template <class IO>
Status
AutoFillSavedModelImpl::FixIOConfig(
    const TRTISTF_IOList* reference_list, IOList<IO>* mutable_list)
{
  bool config_io_specified = (mutable_list->size() > 0);
  for (const TRTISTF_IOList* itr = reference_list; itr != nullptr;
       itr = itr->next_) {
    TRTISTF_IO* io = itr->io_;

    // Add new IO or find corresponding IO in config to be filled
    IO* config_io = nullptr;
    if (config_io_specified) {
      for (auto& mutable_io : *mutable_list) {
        if (mutable_io.name() == io->name_) {
          config_io = &mutable_io;
          break;
        }
      }
    } else {
      config_io = mutable_list->Add();
    }
    if (config_io == nullptr) {
      continue;
    }

    config_io->set_name(io->name_);

    // only set type and shape if they are not set
    if (config_io->data_type() == DataType::TYPE_INVALID) {
      config_io->set_data_type(ConvertDataType(io->data_type_));
    }
    if (config_io->dims_size() == 0) {
      // The model signature supports batching then the first
      // dimension is -1 and should not appear in the model
      // configuration 'dims' that we are creating.
      for (size_t i = (model_support_batching_ ? 1 : 0); i < io->shape_->rank_;
           ++i) {
        config_io->mutable_dims()->Add(io->shape_->dims_[i]);
      }

      // If io dims are empty then must use a reshape for the
      // io, since 'dims' is not allowed to be empty.
      if (config_io->dims_size() == 0) {
        config_io->mutable_dims()->Add(1);
        config_io->mutable_reshape();
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

  // There must be at least one version directory that we can inspect to
  // attempt to determine the platform. For now we allow multiple versions
  // and only inspect the first verison directory to ensure it is valid.
  // We can add more aggressive checks later.
  if (version_dirs.size() == 0) {
    return Status(
        RequestStatusCode::INTERNAL, "unable to autofill for '" + model_name +
                                         "' due to no version directories");
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

  TRTISTF_Model* trtistf_model = nullptr;
  RETURN_IF_TRTISTF_ERROR(TRTISTF_ModelCreateFromSavedModel(
      &trtistf_model, model_name.c_str(), savedmodel_path.c_str(),
      TRTISTF_NO_GPU_DEVICE, false /* have_graph */, 0 /* graph_level */,
      graphdef_backend_config->allow_gpu_memory_growth,
      graphdef_backend_config->per_process_gpu_memory_fraction,
      graphdef_backend_config->allow_soft_placement,
      graphdef_backend_config->memory_limit_mb, nullptr /* tftrt_config */));

  autofill->reset(
      new AutoFillSavedModelImpl(model_name, savedmodel_dir, trtistf_model));
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

  // There must be at least one version directory that we can inspect to
  // attempt to determine the platform. For now we allow multiple versions
  // and only inspect the first verison directory to ensure it is valid.
  // We can add more aggressive checks later.
  if (version_dirs.size() == 0) {
    return Status(
        RequestStatusCode::INTERNAL, "unable to autofill for '" + model_name +
                                         "' due to no version directories");
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
