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

#include "src/backends/tensorrt/autofill.h"

#include <NvInfer.h>
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
      nvinfer1::ICudaEngine* engine, nvinfer1::IRuntime* runtime)
      : AutoFill(model_name), plan_filename_(plan_filename), engine_(engine),
        runtime_(runtime), skip_first_dim_(false), max_batch_size_(0)
  {
  }

  ~AutoFillPlanImpl()
  {
    if (engine_ != nullptr) {
      engine_->destroy();
    }

    if (runtime_ != nullptr) {
      runtime_->destroy();
    }
  }

  Status Fix(ModelConfig* config) override;

 private:
  template <class ModelIO>
  using IOList = ::google::protobuf::RepeatedPtrField<ModelIO>;

  Status Init(ModelConfig* config);

  Status FixBatchingSupport(ModelConfig* config);

  void InitIOLists();

  template <class IO>
  Status FixIO(const IOList<IO>& reference_list, IOList<IO>* mutable_list);

  const std::string plan_filename_;
  ModelConfig config_;
  nvinfer1::ICudaEngine* engine_;
  nvinfer1::IRuntime* runtime_;
  bool skip_first_dim_;
  int max_batch_size_;
};

Status
AutoFillPlanImpl::Fix(ModelConfig* config)
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
AutoFillPlanImpl::Init(ModelConfig* config)
{
  bool is_dynamic = false;
  bool first_dim_variable = true;
  int num_profiles = 0;

  int num_model_bindings = engine_->getNbBindings();

  // Visits all the model bindings in the engine and verifies if
  // dynamic shapes are in use. Also verifies if the first dimension
  // can be the batch dimension.
  for (int i = 0; i < num_model_bindings; ++i) {
    nvinfer1::Dims dims = engine_->getBindingDimensions(i);
    if (engine_->bindingIsInput(i)) {
      if (!is_dynamic) {
        for (int didx = 0; didx < dims.nbDims; ++didx) {
          if (dims.d[didx] == -1) {
            // Initialize the necessary variables
            num_profiles = engine_->getNbOptimizationProfiles();
            num_model_bindings /= num_profiles;
            is_dynamic = true;
            break;
          }
        }
      }
    }
    if (dims.d[0] != -1 || dims.nbDims == 1) {
      first_dim_variable = false;
      break;
    }
  }

  if (!is_dynamic) {
    // If engine doesn't contain any dynamic shape then get the max batch size
    // from the engine and exit
    max_batch_size_ = engine_->getMaxBatchSize();
    return Status::Success;
  } else if (!first_dim_variable) {
    LOG_WARNING << "The TRT engine doesn't specify appropriate dimensions to "
                   "support dynamic batching";
    max_batch_size_ = 0;
  } else {
    // Generate the set of profiles that supports dynamic batching. A profile to
    // support dynamic batching should have minimum shape of first dim for every
    // binding to be 1.
    std::set<int> supported_profiles;
    for (int profile = 0; profile < num_profiles; profile++) {
      bool supports_batching = true;
      for (int binding = 0; binding < num_model_bindings; binding++) {
        int effective_binding_index = binding + (profile * num_model_bindings);
        if (engine_->bindingIsInput(effective_binding_index)) {
          nvinfer1::Dims min_shape = engine_->getProfileDimensions(
              effective_binding_index, profile,
              nvinfer1::OptProfileSelector::kMIN);
          if (min_shape.d[0] != 1) {
            supports_batching = false;
            break;
          }
        }
      }
      if (supports_batching) {
        supported_profiles.insert(profile);
      }
    }
    // Generate the map between supported profile and the corresponding maximum
    // possible batch size
    std::map<int, int> supported_profile_bs_map;
    for (auto profile : supported_profiles) {
      int min_max_shape = INT_MAX;
      for (int binding = 0; binding < num_model_bindings; binding++) {
        int effective_binding_index = binding + (profile * num_model_bindings);
        if (engine_->bindingIsInput(effective_binding_index)) {
          nvinfer1::Dims max_shape = engine_->getProfileDimensions(
              effective_binding_index, profile,
              nvinfer1::OptProfileSelector::kMAX);
          if (min_max_shape > max_shape.d[0]) {
            min_max_shape = max_shape.d[0];
          }
        }
      }
      supported_profile_bs_map[profile] = min_max_shape;
    }

    if (supported_profiles.empty()) {
      // no supported profiles
      max_batch_size_ = 0;
    } else {
      std::set<int> config_profiles;
      // Verify all the profiles in the instance groups are supported or not
      bool supports_batching = true;
      for (const auto& group : config->instance_group()) {
        for (const auto& profile : group.profile()) {
          int profile_idx;
          RETURN_IF_ERROR(GetProfileIndex(profile, &profile_idx));
          if (profile_idx < 0 || profile_idx >= num_profiles) {
            return Status(
                RequestStatusCode::INTERNAL,
                "unable to autofill for '" + model_name_ +
                    "', configuration specified invalid profile " + profile +
                    " . Number of profiles supported by TensorRT engine: " +
                    std::to_string(num_profiles));
          }
          config_profiles.insert(profile_idx);
          if (supported_profiles.find(profile_idx) ==
              supported_profiles.end()) {
            supports_batching = false;
            break;
          }
        }
      }
      // Get the max batch size if batching is supported
      if (supports_batching) {
        // Get the minimum of the maximum shape in config_profiles if non-empty,
        // else get the minimum of the maximum shape in first supported profile.
        if (!config_profiles.empty()) {
          max_batch_size_ = INT_MAX;
          for (auto profiles : config_profiles) {
            if (max_batch_size_ > supported_profile_bs_map[profiles]) {
              max_batch_size_ = supported_profile_bs_map[profiles];
            }
          }
        } else {
          LOG_WARNING
              << "No profiles specified in the model config. Will be selecting "
                 "profile index 0 to determine the max_batch_size for "
              << model_name_;
          if (supported_profile_bs_map.find(0) !=
              supported_profile_bs_map.end()) {
            max_batch_size_ = supported_profile_bs_map[0];
          } else {
            LOG_WARNING << "Profile index 0 for " << model_name_
                        << " does not support batching.";
            max_batch_size_ = 0;
          }
        }
      } else {
        LOG_WARNING << "Some of the profiles for " << model_name_
                    << " does not support batching.";
        max_batch_size_ = 0;
      }
    }
  }

  // For dynamic batching, the number of dimensions specified in model config
  // match should be 1 less than the number of dimensions specified in engine.
  bool config_batch_hint = false;
  // The number of IO Tensors with shape specification in config
  int tensors_with_config_shape_cnt = 0;
  if ((config->input_size() != 0) || (config->output_size() != 0)) {
    for (const auto& config_io : config->input()) {
      if (!config_io.dims().empty()) {
        tensors_with_config_shape_cnt++;
        // look up corresponding io info from model
        for (int binding = 0; binding < num_model_bindings; binding++) {
          if (config_io.name() == engine_->getBindingName(binding)) {
            nvinfer1::Dims shape = engine_->getBindingDimensions(binding);
            bool should_batch = (shape.nbDims == (config_io.dims_size() + 1));
            if (should_batch) {
              config_batch_hint = true;
            }
            if (config_batch_hint && (!should_batch)) {
              return Status(
                  RequestStatusCode::INTERNAL,
                  "unable to autofill for '" + model_name_ +
                      "', model tensor configurations are contradicting " +
                      "each other in terms of whether batching is supported");
            }
          }
        }
      }
    }
    for (const auto& config_io : config->output()) {
      if (!config_io.dims().empty()) {
        tensors_with_config_shape_cnt++;
        // look up corresponding io info from model
        for (int binding = 0; binding < num_model_bindings; binding++) {
          if (config_io.name() == engine_->getBindingName(binding)) {
            nvinfer1::Dims shape = engine_->getBindingDimensions(binding);
            bool should_batch = (shape.nbDims == (config_io.dims_size() + 1));
            if (should_batch) {
              config_batch_hint = true;
            }
            if (config_batch_hint && (!should_batch)) {
              return Status(
                  RequestStatusCode::INTERNAL,
                  "unable to autofill for '" + model_name_ +
                      "', model tensor configurations are contradicting " +
                      "each other in terms of whether batching is supported");
            }
          }
        }
      }
    }
  }

  // Validate cases with incomplete input and output shapes
  if (tensors_with_config_shape_cnt != 0 &&
      tensors_with_config_shape_cnt != num_model_bindings) {
    return Status(
        RequestStatusCode::INTERNAL,
        "unable to autofill for '" + model_name_ +
            "', either all model tensor configuration should specify their "
            "dims or none.");
  }

  if (config_batch_hint && max_batch_size_ == 0) {
    return Status(
        RequestStatusCode::INTERNAL,
        "unable to autofill for '" + model_name_ +
            "', model tensor  shape configuration hints for dynamic batching "
            "but the underlying engine doesn't support batching.");
  } else if (tensors_with_config_shape_cnt != 0 && !config_batch_hint) {
    // if no hint for batching in config io
    LOG_WARNING << "The specified dimensions in model config for "
                << model_name_ << " hints that batching is unavailable";
    max_batch_size_ = 0;
  }


  if (max_batch_size_ != 0) {
    skip_first_dim_ = true;
  }
  return Status::Success;
}

Status
AutoFillPlanImpl::FixBatchingSupport(ModelConfig* config)
{
  if (config->max_batch_size() == 0) {
    config->set_max_batch_size(max_batch_size_);
  } else if (config->max_batch_size() > max_batch_size_) {
    return Status(
        RequestStatusCode::INTERNAL,
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
  int num_model_bindings =
      engine_->getNbBindings() / engine_->getNbOptimizationProfiles();
  for (int i = 0; i < num_model_bindings; ++i) {
    if (engine_->bindingIsInput(i)) {
      ModelInput* config_input = config_.add_input();
      std::string input_name{engine_->getBindingName(i)};
      config_input->set_name(input_name.substr(0, input_name.find(" ")));
      config_input->set_data_type(
          ConvertTrtTypeToDataType(engine_->getBindingDataType(i)));
      nvinfer1::Dims dims = engine_->getBindingDimensions(i);
      for (int didx = skip_first_dim_ ? 1 : 0; didx < dims.nbDims; ++didx) {
        config_input->mutable_dims()->Add(dims.d[didx]);
      }
    } else {
      ModelOutput* config_output = config_.add_output();
      std::string output_name{engine_->getBindingName(i)};
      config_output->set_name(output_name.substr(0, output_name.find(" ")));
      config_output->set_data_type(
          ConvertTrtTypeToDataType(engine_->getBindingDataType(i)));
      nvinfer1::Dims dims = engine_->getBindingDimensions(i);
      for (int didx = skip_first_dim_ ? 1 : 0; didx < dims.nbDims; ++didx) {
        config_output->mutable_dims()->Add(dims.d[didx]);
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
          if (io.data_type() == DataType::TYPE_INVALID) {
            io.set_data_type(io_ref.data_type());
          }
          if (io.dims_size() == 0) {
            io.mutable_dims()->CopyFrom(io_ref.dims());
            if (io_ref.has_reshape()) {
              io.mutable_reshape()->CopyFrom(io_ref.reshape());
            }
          }
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
        RequestStatusCode::INTERNAL, "unable to autofill for '" + model_name +
                                         "' due to no version directories");
  }

  // The model configuration will be the same across all the version directories
  const auto version_path = JoinPath({model_path, *(version_dirs.begin())});

  std::set<std::string> plan_files;
  RETURN_IF_ERROR(GetDirectoryFiles(
      version_path, true /* skip_hidden_files */, &plan_files));

  nvinfer1::IRuntime* runtime = nullptr;
  nvinfer1::ICudaEngine* engine = nullptr;
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

    if (!LoadPlan(plan_data, &runtime, &engine).IsOk()) {
      if (engine != nullptr) {
        engine->destroy();
      }
      if (runtime != nullptr) {
        runtime->destroy();
      }
    } else {
      plan_file = file;
      found = true;
      break;
    }
  }

  if (!found) {
    return Status(
        RequestStatusCode::INTERNAL,
        "unable to autofill for '" + model_name +
            "', unable to find a compatible plan file.");
  }

  autofill->reset(new AutoFillPlanImpl(model_name, plan_file, engine, runtime));

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
