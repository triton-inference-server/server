// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "src/backends/tensorrt/autofill.h"

#include <NvInfer.h>
#include <vector>
#include "src/backends/tensorrt/loader.h"
#include "src/backends/tensorrt/plan_utils.h"
#include "src/core/autofill.h"
#include "src/core/constants.h"
#include "src/core/filesystem.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/model_config_utils.h"

namespace nvidia { namespace inferenceserver {

class AutoFillPlanImpl : public AutoFill {
 public:
  AutoFillPlanImpl(
      const std::string& model_name, const std::string& plan_filename,
      const std::shared_ptr<nvinfer1::ICudaEngine>& engine,
      const std::shared_ptr<nvinfer1::IRuntime>& runtime)
      : AutoFill(model_name), plan_filename_(plan_filename), engine_(engine),
        runtime_(runtime), max_batch_size_(0), num_profile_bindings_(0)
  {
    if (!UseTensorRTv2API(engine_)) {
      num_profile_bindings_ = engine_->getNbBindings();
    }
  }

  Status Fix(inference::ModelConfig* config) override;

 private:
  template <class ModelIO>
  using IOList = ::google::protobuf::RepeatedPtrField<ModelIO>;
  using DimsList = ::google::protobuf::RepeatedField<int64_t>;

  Status Init(inference::ModelConfig* config);

  Status GetMaxSupportedBatchSize(inference::ModelConfig* config);

  Status GetProfileMaxBatchSize(
      const int profile_index, int* max_profile_batch_size);

  Status GetProfileIndices(
      inference::ModelConfig* config, std::set<int>* config_profiles);

  Status ExtractBatchHintFromIOConfig(
      const std::string& tensor_name, const DimsList& dims,
      bool* config_batch_hint);

  Status FixBatchingSupport(inference::ModelConfig* config);

  void InitIOLists();

  template <class IO>
  void InitIODims(nvinfer1::Dims& dims, bool is_shape_binding, IO* config_io);

  template <class IO>
  Status FixIO(const IOList<IO>& reference_list, IOList<IO>* mutable_list);

  const std::string plan_filename_;
  inference::ModelConfig config_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IRuntime> runtime_;
  int max_batch_size_;
  int num_profile_bindings_;
};

Status
AutoFillPlanImpl::Fix(inference::ModelConfig* config)
{
  config->set_platform(kTensorRTPlanPlatform);

  // Set name if not already set.
  if (config->name().empty()) {
    config->set_name(model_name_);
  }

  if (config->default_model_filename().empty()) {
    config->set_default_model_filename(plan_filename_);
  }

  // Initialize the autofiller
  RETURN_IF_ERROR(Init(config));

  // fix batching support
  RETURN_IF_ERROR(FixBatchingSupport(config));

  // Get the reference IO lists
  InitIOLists();

  // Inputs
  RETURN_IF_ERROR(FixIO(config_.input(), config->mutable_input()));
  // Outputs
  RETURN_IF_ERROR(FixIO(config_.output(), config->mutable_output()));

  return Status::Success;
}

Status
AutoFillPlanImpl::Init(inference::ModelConfig* config)
{
  if (engine_->hasImplicitBatchDimension()) {
    // If engine has implicit batch dimension then retrieve the value and exit
    max_batch_size_ = engine_->getMaxBatchSize();
    return Status::Success;
  } else {
    // Assuming the first dimension to be batch dimension, until and unless
    // proven otherwise.
    RETURN_IF_ERROR(GetMaxSupportedBatchSize(config));
  }

  // For batching support, the number of dimensions specified in model config
  // match should be 1 less than the number of dimensions specified in engine.
  // Will use that as a hint to ascertain whether or not to enable batching.
  bool config_batch_hint = false;
  // The number of IO Tensors with shape specification in config
  int tensors_with_config_shape_cnt = 0;
  if ((config->input_size() != 0) || (config->output_size() != 0)) {
    for (const auto& config_io : config->input()) {
      if (!config_io.dims().empty()) {
        tensors_with_config_shape_cnt++;
        RETURN_IF_ERROR(ExtractBatchHintFromIOConfig(
            config_io.name(), config_io.dims(), &config_batch_hint));
      }
    }
    for (const auto& config_io : config->output()) {
      if (!config_io.dims().empty()) {
        tensors_with_config_shape_cnt++;
        RETURN_IF_ERROR(ExtractBatchHintFromIOConfig(
            config_io.name(), config_io.dims(), &config_batch_hint));
      }
    }
  }

  // Validate cases with incomplete input and output shapes
  if (tensors_with_config_shape_cnt != 0 &&
      tensors_with_config_shape_cnt != num_profile_bindings_) {
    return Status(
        Status::Code::INTERNAL,
        "unable to autofill for '" + model_name_ +
            "', either all model tensor configuration should specify their "
            "dims or none.");
  }

  if (config_batch_hint && max_batch_size_ == 0) {
    return Status(
        Status::Code::INTERNAL,
        "unable to autofill for '" + model_name_ +
            "', model tensor  shape configuration hints for dynamic batching "
            "but the underlying engine doesn't support batching.");
  } else if (tensors_with_config_shape_cnt != 0 && !config_batch_hint) {
    // if no hint for batching in config io
    LOG_WARNING << "The specified dimensions in model config for "
                << model_name_ << " hints that batching is unavailable";
    max_batch_size_ = 0;
  }

  return Status::Success;
}

Status
AutoFillPlanImpl::GetMaxSupportedBatchSize(inference::ModelConfig* config)
{
  std::set<int> profile_indices;
  RETURN_IF_ERROR(GetProfileIndices(config, &profile_indices));

  int running_max = 0;
  for (const auto profile_index : profile_indices) {
    int max_profile_batch_size;
    RETURN_IF_ERROR(
        GetProfileMaxBatchSize(profile_index, &max_profile_batch_size));
    if (max_profile_batch_size > running_max) {
      running_max = max_profile_batch_size;
    }
  }

  max_batch_size_ = running_max;

  return Status::Success;
}

Status
AutoFillPlanImpl::GetProfileIndices(
    inference::ModelConfig* config, std::set<int>* config_profiles)
{
  int num_profiles = engine_->getNbOptimizationProfiles();
  num_profile_bindings_ = engine_->getNbBindings() / num_profiles;

  for (const auto& group : config->instance_group()) {
    for (const auto& profile : group.profile()) {
      int profile_idx;
      RETURN_IF_ERROR(GetProfileIndex(profile, &profile_idx));
      if (profile_idx < 0 || profile_idx >= num_profiles) {
        return Status(
            Status::Code::INTERNAL,
            "unable to autofill for '" + model_name_ +
                "', configuration specified invalid profile " + profile +
                " . Number of profiles supported by TensorRT engine: " +
                std::to_string(num_profiles));
      }
      config_profiles->insert(profile_idx);
    }
  }

  if (config_profiles->empty()) {
    // If not specified then use the default.
    config_profiles->insert(0);
  }

  return Status::Success;
}

Status
AutoFillPlanImpl::GetProfileMaxBatchSize(
    int profile_index, int* max_profile_batch_size)
{
  *max_profile_batch_size = INT_MAX;

  // Visit all the bindings of the profile to capture the maximum and
  // minimum batch size supported.
  for (int binding_index = 0; binding_index < num_profile_bindings_;
       binding_index++) {
    int effective_binding_index =
        (profile_index * num_profile_bindings_) + binding_index;
    if (engine_->bindingIsInput(effective_binding_index)) {
      if (!engine_->isShapeBinding(effective_binding_index)) {
        nvinfer1::Dims max_shape = engine_->getProfileDimensions(
            effective_binding_index, profile_index,
            nvinfer1::OptProfileSelector::kMAX);
        if (*max_profile_batch_size > max_shape.d[0]) {
          *max_profile_batch_size = max_shape.d[0];
        }

      } else {
        const int32_t* max_shapes = engine_->getProfileShapeValues(
            effective_binding_index, profile_index,
            nvinfer1::OptProfileSelector::kMAX);
        if (*max_profile_batch_size > *max_shapes) {
          *max_profile_batch_size = *max_shapes;
        }
      }
    }
  }
  return Status::Success;
}


Status
AutoFillPlanImpl::ExtractBatchHintFromIOConfig(
    const std::string& tensor_name, const DimsList& dims,
    bool* config_batch_hint)
{
  // look up corresponding io info from model
  for (int binding_index = 0; binding_index < num_profile_bindings_;
       binding_index++) {
    if (tensor_name == engine_->getBindingName(binding_index)) {
      nvinfer1::Dims shape = engine_->getBindingDimensions(binding_index);
      bool should_batch;
      if (!engine_->isShapeBinding(binding_index)) {
        should_batch = (shape.nbDims == (dims.size() + 1));
      } else {
        should_batch = (shape.d[0] == (dims[0] + 1));
      }
      if (should_batch) {
        *config_batch_hint = true;
      }
      if (*config_batch_hint && (!should_batch)) {
        return Status(
            Status::Code::INTERNAL,
            "unable to autofill for '" + model_name_ +
                "', model tensor configurations are contradicting " +
                "each other in terms of whether batching is supported");
      }
    }
  }
  return Status::Success;
}

Status
AutoFillPlanImpl::FixBatchingSupport(inference::ModelConfig* config)
{
  if (config->max_batch_size() == 0) {
    config->set_max_batch_size(max_batch_size_);
  } else if (config->max_batch_size() > max_batch_size_) {
    return Status(
        Status::Code::INTERNAL,
        "unable to autofill for '" + model_name_ +
            "', configuration specified max-batch " +
            std::to_string(config->max_batch_size()) +
            " but TensorRT engine only supports max-batch " +
            std::to_string(max_batch_size_));
  }
  return Status::Success;
}

void
AutoFillPlanImpl::InitIOLists()
{
  for (int i = 0; i < num_profile_bindings_; ++i) {
    nvinfer1::Dims dims = engine_->getBindingDimensions(i);
    bool is_shape_binding = engine_->isShapeBinding(i);
    if (engine_->bindingIsInput(i)) {
      inference::ModelInput* config_input = config_.add_input();
      std::string input_name{engine_->getBindingName(i)};
      config_input->set_name(input_name.substr(0, input_name.find(" ")));
      config_input->set_data_type(
          ConvertTrtTypeToDataType(engine_->getBindingDataType(i)));
      InitIODims(dims, is_shape_binding, config_input);
      config_input->set_is_shape_tensor(is_shape_binding);
    } else {
      inference::ModelOutput* config_output = config_.add_output();
      std::string output_name{engine_->getBindingName(i)};
      config_output->set_name(output_name.substr(0, output_name.find(" ")));
      config_output->set_data_type(
          ConvertTrtTypeToDataType(engine_->getBindingDataType(i)));
      InitIODims(dims, is_shape_binding, config_output);
      config_output->set_is_shape_tensor(is_shape_binding);
    }
  }
}

template <class IO>
void
AutoFillPlanImpl::InitIODims(
    nvinfer1::Dims& dims, bool is_shape_binding, IO* config_io)
{
  bool skip_first =
      (max_batch_size_ != 0) && (!engine_->hasImplicitBatchDimension());
  auto config_dims = config_io->mutable_dims();
  if (!is_shape_binding) {
    for (int didx = (skip_first ? 1 : 0); didx < dims.nbDims; ++didx) {
      config_dims->Add(dims.d[didx]);
    }
    // If tensor dims are empty then must use a reshape for the
    // tensor, since 'dims' is not allowed to be empty.
    if (config_io->dims_size() == 0) {
      config_io->mutable_dims()->Add(1);
      config_io->mutable_reshape();
    }
  } else {
    if (dims.nbDims != 0) {
      if (skip_first) {
        config_dims->Add(dims.d[0] - 1);
      } else {
        config_dims->Add(dims.d[0]);
      }
    }
  }
}

template <class IO>
Status
AutoFillPlanImpl::FixIO(
    const IOList<IO>& reference_list, IOList<IO>* mutable_list)
{
  if (mutable_list->size() == 0) {
    mutable_list->CopyFrom(reference_list);
  } else {
    for (auto& io : *mutable_list) {
      for (const auto& io_ref : reference_list) {
        if (io.name() == io_ref.name()) {
          // only set type and shape if they are not set
          if (io.data_type() == inference::DataType::TYPE_INVALID) {
            io.set_data_type(io_ref.data_type());
          }
          if (io.dims_size() == 0) {
            io.mutable_dims()->CopyFrom(io_ref.dims());
            if (io_ref.has_reshape()) {
              io.mutable_reshape()->CopyFrom(io_ref.reshape());
            }
          }
          // Check if the IO is a shape tensor.
          bool is_shape_tensor = false;
          int io_index = engine_->getBindingIndex(io.name().c_str());
          if (io_index == -1) {
            return Status(
                Status::Code::INVALID_ARG,
                "binding for '" + io.name() + "' not found in the model.");
          }
          is_shape_tensor = engine_->isShapeBinding(io_index);
          if (io.is_shape_tensor() && (!is_shape_tensor)) {
            return Status(
                Status::Code::INVALID_ARG,
                "'" + io.name() +
                    "' is incorrectly specified as a shape tensor.");
          }
          io.set_is_shape_tensor(is_shape_tensor);
          break;
        }
      }
    }
  }
  return Status::Success;
}

Status
AutoFillPlan::Create(
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
        Status::Code::INTERNAL, "unable to autofill for '" + model_name +
                                    "' due to no version directories");
  }

  // The model configuration will be the same across all the version directories
  const auto version_path = JoinPath({model_path, *(version_dirs.begin())});

  std::set<std::string> plan_files;
  RETURN_IF_ERROR(GetDirectoryFiles(
      version_path, true /* skip_hidden_files */, &plan_files));

  std::shared_ptr<nvinfer1::IRuntime> runtime;
  std::shared_ptr<nvinfer1::ICudaEngine> engine;
  std::string plan_file;
  Status status;
  bool found = false;

  for (auto file : plan_files) {
    const auto plan_path = JoinPath({version_path, file});

    std::string plan_data_str;
    status = ReadTextFile(plan_path, &plan_data_str);
    if (!status.IsOk()) {
      continue;
    }
    std::vector<char> plan_data(plan_data_str.begin(), plan_data_str.end());

    if (!LoadPlan(plan_data, -1 /* dla_core_id */, &runtime, &engine).IsOk()) {
      if (engine != nullptr) {
        engine.reset();
      }
      if (runtime != nullptr) {
        runtime.reset();
      }
    } else {
      plan_file = file;
      found = true;
      break;
    }
  }

  if (!found) {
    return Status(
        Status::Code::INTERNAL,
        "unable to autofill for '" + model_name +
            "', unable to find a compatible plan file.");
  }

  autofill->reset(new AutoFillPlanImpl(model_name, plan_file, engine, runtime));

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
