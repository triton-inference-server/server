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

#include "src/core/model_config_utils.h"

#include <deque>
#include <set>
#include "src/core/autofill.h"
#include "src/core/constants.h"
#include "src/core/cuda_utils.h"
#include "src/core/filesystem.h"
#include "src/core/logging.h"

#ifdef TRTIS_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRTIS_ENABLE_GPU

namespace nvidia { namespace inferenceserver {

Status
GetModelVersionFromString(const std::string& version_string, int64_t* version)
{
  if (version_string.empty()) {
    *version = -1;
    return Status::Success;
  }

  try {
    *version = std::stol(version_string);
  }
  catch (std::exception& e) {
    return Status(
        Status::Code::INVALID_ARG,
        "failed to get model version from specified version string '" +
            version_string + "' (details: " + e.what() +
            "), version should be an integral value > 0");
  }

  if (*version < 0) {
    return Status(
        Status::Code::INVALID_ARG,
        "invalid model version specified '" + version_string +
            "' , version should be an integral value > 0");
  }

  return Status::Success;
}

Status
GetModelVersionFromPath(const std::string& path, int64_t* version)
{
  auto version_dir = BaseName(path);

  // Determine the version from the last segment of 'path'
  try {
    *version = std::atoll(version_dir.c_str());
  }
  catch (...) {
    return Status(
        Status::Code::INTERNAL,
        "unable to determine model version from " + path);
  }

  return Status::Success;
}

Status
GetBooleanSequenceControlProperties(
    const ModelSequenceBatching& batcher, const std::string& model_name,
    const ModelSequenceBatching::Control::Kind control_kind,
    const bool required, std::string* tensor_name, DataType* tensor_datatype,
    float* fp32_false_value, float* fp32_true_value, int32_t* int32_false_value,
    int32_t* int32_true_value)
{
  // Make sure same tensor is not configured for multiple controls
  std::set<std::string> seen_tensors;

  // Make sure the control kind is not mentioned multiple times.
  bool seen_control = false;

  for (const auto& control_input : batcher.control_input()) {
    if (control_input.name().empty()) {
      return Status(
          Status::Code::INVALID_ARG,
          "sequence batching control tensor must have a name for " +
              model_name);
    }

    if (seen_tensors.find(control_input.name()) != seen_tensors.end()) {
      return Status(
          Status::Code::INVALID_ARG,
          "sequence batching control tensor '" + control_input.name() +
              "' is specified for multiple control kinds for " + model_name);
    }

    seen_tensors.insert(control_input.name());

    for (const auto& c : control_input.control()) {
      if (c.kind() == control_kind) {
        if (seen_control) {
          return Status(
              Status::Code::INVALID_ARG,
              "sequence batching specifies multiple " +
                  ModelSequenceBatching_Control_Kind_Name(control_kind) +
                  " tensors for " + model_name);
        }

        *tensor_name = control_input.name();
        seen_control = true;

        if (c.int32_false_true_size() > 0) {
          if (c.fp32_false_true_size() != 0) {
            return Status(
                Status::Code::INVALID_ARG,
                "sequence batching specifies both 'int32_false_true' and "
                "'fp32_false_true' for " +
                    ModelSequenceBatching_Control_Kind_Name(control_kind) +
                    " for " + model_name);
          }

          if (c.int32_false_true_size() != 2) {
            return Status(
                Status::Code::INVALID_ARG,
                "sequence batching control 'int32_false_true' must have "
                "exactly 2 entries for " +
                    ModelSequenceBatching_Control_Kind_Name(control_kind) +
                    " for " + model_name);
          }

          if (tensor_datatype != nullptr) {
            *tensor_datatype = DataType::TYPE_INT32;
          }
          if (int32_false_value != nullptr) {
            *int32_false_value = c.int32_false_true(0);
          }
          if (int32_true_value != nullptr) {
            *int32_true_value = c.int32_false_true(1);
          }
        } else {
          if (c.fp32_false_true_size() == 0) {
            return Status(
                Status::Code::INVALID_ARG,
                "sequence batching must specify either 'int32_false_true' or "
                "'fp32_false_true' for " +
                    ModelSequenceBatching_Control_Kind_Name(control_kind) +
                    " for " + model_name);
          }

          if (c.fp32_false_true_size() != 2) {
            return Status(
                Status::Code::INVALID_ARG,
                "sequence batching control 'fp32_false_true' must have exactly "
                "2 entries for " +
                    ModelSequenceBatching_Control_Kind_Name(control_kind) +
                    " for " + model_name);
          }

          if (tensor_datatype != nullptr) {
            *tensor_datatype = DataType::TYPE_FP32;
          }
          if (fp32_false_value != nullptr) {
            *fp32_false_value = c.fp32_false_true(0);
          }
          if (fp32_true_value != nullptr) {
            *fp32_true_value = c.fp32_false_true(1);
          }
        }
      }
    }
  }

  if (!seen_control) {
    if (required) {
      return Status(
          Status::Code::INVALID_ARG,
          "sequence batching control tensor must specify a " +
              ModelSequenceBatching_Control_Kind_Name(control_kind) +
              " value for " + model_name);
    }

    tensor_name->clear();
  }

  return Status::Success;
}

Status
GetTypedSequenceControlProperties(
    const ModelSequenceBatching& batcher, const std::string& model_name,
    const ModelSequenceBatching::Control::Kind control_kind,
    const bool required, std::string* tensor_name, DataType* tensor_datatype)
{
  // Make sure same tensor is not configured for multiple controls
  std::set<std::string> seen_tensors;

  // Make sure the control kind is not mentioned multiple times.
  bool seen_control = false;

  for (const auto& control_input : batcher.control_input()) {
    if (control_input.name().empty()) {
      return Status(
          Status::Code::INVALID_ARG,
          "sequence batching control tensor must have a name for " +
              model_name);
    }

    if (seen_tensors.find(control_input.name()) != seen_tensors.end()) {
      return Status(
          Status::Code::INVALID_ARG,
          "sequence batching control tensor '" + control_input.name() +
              "' is specified for multiple control kinds for " + model_name);
    }

    seen_tensors.insert(control_input.name());

    for (const auto& c : control_input.control()) {
      if (c.kind() == control_kind) {
        if (seen_control) {
          return Status(
              Status::Code::INVALID_ARG,
              "sequence batching specifies multiple " +
                  ModelSequenceBatching_Control_Kind_Name(control_kind) +
                  " tensors for " + model_name);
        }

        *tensor_name = control_input.name();
        if (tensor_datatype != nullptr) {
          *tensor_datatype = c.data_type();
        }

        seen_control = true;

        if ((c.int32_false_true_size() > 0) || (c.fp32_false_true_size() > 0)) {
          return Status(
              Status::Code::INVALID_ARG,
              "sequence batching must not specify either 'int32_false_true' "
              "nor 'fp32_false_true' for " +
                  ModelSequenceBatching_Control_Kind_Name(control_kind) +
                  " for " + model_name);
        }
      }
    }
  }

  if (!seen_control) {
    if (required) {
      return Status(
          Status::Code::INVALID_ARG,
          "sequence batching control tensor must specify a " +
              ModelSequenceBatching_Control_Kind_Name(control_kind) +
              " value for " + model_name);
    }

    tensor_name->clear();
  }

  return Status::Success;
}

Status
GetNormalizedModelConfig(
    const std::string& path, const BackendConfigMap& backend_config_map,
    const bool autofill, const double min_compute_capability,
    ModelConfig* config)
{
  // If 'autofill' then the configuration file can be empty.
  const auto config_path = JoinPath({path, kModelConfigPbTxt});
  bool model_config_exists;
  RETURN_IF_ERROR(FileExists(config_path, &model_config_exists));
  if (autofill && !model_config_exists) {
    config->Clear();
  } else {
    RETURN_IF_ERROR(ReadTextProto(config_path, config));
  }

  // Autofill if requested...
  if (autofill) {
    const std::string model_name(BaseName(path));
    std::unique_ptr<AutoFill> af;
    RETURN_IF_ERROR(AutoFill::Create(
        model_name, backend_config_map, std::string(path), *config, &af));
    RETURN_IF_ERROR(af->Fix(config));

    LOG_VERBOSE(1) << "autofilled config: " << config->DebugString();
  }

  if (config->platform().empty()) {
    return Status(
        Status::Code::INVALID_ARG,
        "must specify platform for model '" + config->name() + "'");
  }

  // If 'default_model_filename' is not specified set it appropriately
  // based upon 'platform'.
  if (config->default_model_filename().empty()) {
#ifdef TRTIS_ENABLE_TENSORFLOW
    if (config->platform() == kTensorFlowGraphDefPlatform) {
      config->set_default_model_filename(kTensorFlowGraphDefFilename);
    } else if (config->platform() == kTensorFlowSavedModelPlatform) {
      config->set_default_model_filename(kTensorFlowSavedModelFilename);
    } else
#endif  // TRTIS_ENABLE_TENSORFLOW
#ifdef TRTIS_ENABLE_TENSORRT
        if (config->platform() == kTensorRTPlanPlatform) {
      config->set_default_model_filename(kTensorRTPlanFilename);
    } else
#endif  // TRTIS_ENABLE_TENSORRT
#ifdef TRTIS_ENABLE_CAFFE2
        if (config->platform() == kCaffe2NetDefPlatform) {
      config->set_default_model_filename(kCaffe2NetDefFilename);
    } else
#endif  // TRTIS_ENABLE_CAFFE2
#ifdef TRTIS_ENABLE_ONNXRUNTIME
        if (config->platform() == kOnnxRuntimeOnnxPlatform) {
      config->set_default_model_filename(kOnnxRuntimeOnnxFilename);
    } else
#endif  // TRTIS_ENABLE_ONNXRUNTIME
#ifdef TRTIS_ENABLE_PYTORCH
        if (config->platform() == kPyTorchLibTorchPlatform) {
      config->set_default_model_filename(kPyTorchLibTorchFilename);
    } else
#endif  // TRTIS_ENABLE_PYTORCH
#ifdef TRTIS_ENABLE_CUSTOM
        if (config->platform() == kCustomPlatform) {
      config->set_default_model_filename(kCustomFilename);
    } else
#endif  // TRTIS_ENABLE_CUSTOM
        if (config->platform() == kEnsemblePlatform) {
      // No actual model file is needed to be loaded for ensemble.
    } else {
      return Status(
          Status::Code::INTERNAL, "unexpected platform type " +
                                      config->platform() + " for " +
                                      config->name());
    }
  }

  // If version_policy is not specified, default to Latest 1 version.
  if (!config->has_version_policy()) {
    ModelVersionPolicy::Latest latest;
    latest.set_num_versions(1);
    config->mutable_version_policy()->mutable_latest()->CopyFrom(latest);
  }

  // If dynamic batching is specified...
  if (config->has_dynamic_batching()) {
    // If preferred batch size is not specified choose
    // automatically. For now we just choose 4, 8 as those are
    // generally good values for GPUs.
    if (config->dynamic_batching().preferred_batch_size().size() == 0) {
      if (config->max_batch_size() >= 4) {
        config->mutable_dynamic_batching()->mutable_preferred_batch_size()->Add(
            4);
      }
      if (config->max_batch_size() >= 8) {
        config->mutable_dynamic_batching()->mutable_preferred_batch_size()->Add(
            8);
      }
    }
  }

  // If sequence batching is specified...
  if (config->has_sequence_batching()) {
    // Set default idle is not specified.
    if (config->sequence_batching().max_sequence_idle_microseconds() == 0) {
      config->mutable_sequence_batching()->set_max_sequence_idle_microseconds(
          SEQUENCE_IDLE_DEFAULT_MICROSECONDS);
    }
  }

  // If model ensembling is specified, don't attempt to normalize instance_group
  // as it is not allowed in ensemble scheduling
  if (!config->has_ensemble_scheduling()) {
    auto optimization = config->mutable_optimization();
    if (!optimization->has_input_pinned_memory()) {
      optimization->mutable_input_pinned_memory()->set_enable(true);
    }
    if (!optimization->has_output_pinned_memory()) {
      optimization->mutable_output_pinned_memory()->set_enable(true);
    }

    // Make sure there is at least one instance_group.
    if (config->instance_group().size() == 0) {
      ModelInstanceGroup* group = config->add_instance_group();
      group->set_name(config->name());
    }

    // Creates a set of supported GPU device ids
    std::set<int> supported_gpus;
#ifdef TRTIS_ENABLE_GPU
    // Get the total number of GPUs from the runtime library.
    Status status = GetSupportedGPUs(&supported_gpus, min_compute_capability);
    if (!status.IsOk()) {
      return status;
    }

#endif  // TRTIS_ENABLE_GPU

    // Assign default name, kind and count to each instance group that
    // doesn't give those values explicitly. For KIND_GPU, set GPUs to
    // all available if not specified explicitly.
    size_t cnt = 0;
    for (auto& group : *config->mutable_instance_group()) {
      // Name
      if (group.name().empty()) {
        group.set_name(config->name() + "_" + std::to_string(cnt));
      }
      cnt++;

      // For KIND_AUTO... if there are no GPUs or if any of the listed
      // 'gpu's are not present, then use KIND_CPU.
      if (group.kind() == ModelInstanceGroup::KIND_AUTO) {
        if (supported_gpus.empty()) {
          group.set_kind(ModelInstanceGroup::KIND_CPU);
        } else {
          for (const int32_t gid : group.gpus()) {
            if (supported_gpus.find(gid) == supported_gpus.end()) {
              group.set_kind(ModelInstanceGroup::KIND_CPU);
              break;
            }
          }
        }

        if (group.kind() == ModelInstanceGroup::KIND_AUTO) {
          group.set_kind(ModelInstanceGroup::KIND_GPU);
        }
      }

      // Count
      if (group.count() < 1) {
        group.set_count(1);
      }

      // GPUs
      if ((group.kind() == ModelInstanceGroup::KIND_GPU) &&
          (group.gpus().size() == 0)) {
        for (auto d : supported_gpus) {
          group.add_gpus(d);
        }
      }
    }
  }

  return Status::Success;
}

Status
ValidateModelConfig(
    const ModelConfig& config, const std::string& expected_platform,
    const double min_compute_capability)
{
  if (config.name().empty()) {
    return Status(
        Status::Code::INVALID_ARG, "model configuration must specify 'name'");
  }

  if (config.platform().empty()) {
    return Status(
        Status::Code::INVALID_ARG,
        "must specify 'platform' for " + config.name());
  }

  if (config.max_batch_size() < 0) {
    return Status(
        Status::Code::INVALID_ARG,
        "'max_batch_size' must be non-negative value for " + config.name());
  }

  if (!expected_platform.empty() && (config.platform() != expected_platform)) {
    return Status(
        Status::Code::NOT_FOUND, "expected model of type " + expected_platform +
                                     " for " + config.name());
  }

  if (GetPlatform(config.platform()) == Platform::PLATFORM_UNKNOWN) {
    return Status(
        Status::Code::INVALID_ARG, "unexpected platform type \'" +
                                       config.platform() + "\' for " +
                                       config.name());
  }

  if (!config.has_version_policy()) {
    return Status(
        Status::Code::INVALID_ARG,
        "must specify 'version policy' for " + config.name());
  }

  Status status;
  for (const auto& io : config.input()) {
    status = ValidateModelInput(io, config.max_batch_size(), config.platform());
    if (!status.IsOk()) {
      return Status(
          status.StatusCode(), status.Message() + " for " + config.name());
    }
  }
  for (const auto& io : config.output()) {
    status =
        ValidateModelOutput(io, config.max_batch_size(), config.platform());
    if (!status.IsOk()) {
      return Status(
          status.StatusCode(), status.Message() + " for " + config.name());
    }
  }

  // If dynamic batching is specified make sure the preferred batch
  // sizes are positive and don't exceed maximum batch size.
  if (config.has_dynamic_batching()) {
    for (const auto size : config.dynamic_batching().preferred_batch_size()) {
      if (size <= 0) {
        return Status(
            Status::Code::INVALID_ARG,
            "dynamic batching preferred size must be positive for " +
                config.name());
      }
      if (size > config.max_batch_size()) {
        return Status(
            Status::Code::INVALID_ARG,
            "dynamic batching preferred size must be <= max batch size for " +
                config.name());
      }
    }

    // Priority queue is specified
    const auto priority_levels = config.dynamic_batching().priority_levels();
    if (priority_levels != 0) {
      if ((config.dynamic_batching().default_priority_level() == 0) ||
          (config.dynamic_batching().default_priority_level() >
           priority_levels)) {
        return Status(
            Status::Code::INVALID_ARG,
            "default priority level must be in range [1, " +
                std::to_string(priority_levels) + "] for " + config.name());
      }
      for (const auto& queue_policy :
           config.dynamic_batching().priority_queue_policy()) {
        if ((queue_policy.first == 0) ||
            (queue_policy.first > priority_levels)) {
          return Status(
              Status::Code::INVALID_ARG,
              "priority queue policy must have priority level in range [1, " +
                  std::to_string(priority_levels) + "] for " + config.name());
        }
      }
    }

    // preserve ordering option will conflict with priorities and delay policy
    if (config.dynamic_batching().preserve_ordering()) {
      if (priority_levels > 1) {
        return Status(
            Status::Code::INVALID_ARG,
            "Only one priority level is allowed when 'preserve_ordering' is "
            "true for " +
                config.name());
      }
      const auto& default_policy =
          config.dynamic_batching().default_queue_policy();
      if ((default_policy.default_timeout_microseconds() != 0) &&
          (default_policy.timeout_action() == ModelQueuePolicy::DELAY)) {
        return Status(
            Status::Code::INVALID_ARG,
            "Queue policy can not have DELAY as timeout action when "
            "'preserve_ordering' is true for " +
                config.name());
      }
      // Also need to check policy in 'priority_queue_policy'
      // for single priority case
      for (const auto& policy :
           config.dynamic_batching().priority_queue_policy()) {
        if ((policy.second.default_timeout_microseconds() != 0) &&
            (policy.second.timeout_action() == ModelQueuePolicy::DELAY)) {
          return Status(
              Status::Code::INVALID_ARG,
              "Queue policy can not have DELAY as timeout action when "
              "'preserve_ordering' is true for " +
                  config.name());
        }
      }
    }
  }

  // If sequence batching is specified make sure the control is
  // specified correctly.
  if (config.has_sequence_batching()) {
    const auto& batcher = config.sequence_batching();

    if (batcher.control_input_size() == 0) {
      return Status(
          Status::Code::INVALID_ARG,
          "sequence batching must specify at least one control tensor for " +
              config.name());
    }

    // Check boolean controls...
    std::string tensor_name;
    RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
        batcher, config.name(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_START,
        false /* required */, &tensor_name, nullptr, nullptr, nullptr, nullptr,
        nullptr));
    RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
        batcher, config.name(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_END,
        false /* required */, &tensor_name, nullptr, nullptr, nullptr, nullptr,
        nullptr));
    RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
        batcher, config.name(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_READY,
        false /* required */, &tensor_name, nullptr, nullptr, nullptr, nullptr,
        nullptr));

    // Check CORRID control and make sure it is one of the allowed types.
    DataType tensor_datatype;
    RETURN_IF_ERROR(GetTypedSequenceControlProperties(
        batcher, config.name(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_CORRID,
        false /* required */, &tensor_name, &tensor_datatype));
    if (!tensor_name.empty()) {
      if ((tensor_datatype != TYPE_UINT64) && (tensor_datatype != TYPE_INT64) &&
          (tensor_datatype != TYPE_UINT32) && (tensor_datatype != TYPE_INT32)) {
        return Status(
            Status::Code::INVALID_ARG,
            "unexpected data type for control " +
                ModelSequenceBatching_Control_Kind_Name(
                    ModelSequenceBatching::Control::CONTROL_SEQUENCE_CORRID) +
                " for " + config.name() +
                ". Allowed data types are TYPE_UINT64, TYPE_INT64, TYPE_UINT32 "
                "and TYPE_INT32");
      }
    }

    // If oldest-first strategy is enabled make sure the preferred
    // batch sizes are positive and don't exceed maximum batch size.
    if (config.sequence_batching().has_oldest()) {
      for (const auto size :
           config.sequence_batching().oldest().preferred_batch_size()) {
        if (size <= 0) {
          return Status(
              Status::Code::INVALID_ARG,
              "sequence batching preferred batch size must be positive for " +
                  config.name());
        }
        if (size > config.max_batch_size()) {
          return Status(
              Status::Code::INVALID_ARG,
              "sequence batching preferred batch size must be <= max batch "
              "size for " +
                  config.name());
        }
      }
    }
  }

  // If ensemble scheduling is specified, validate it.
  // Otherwise, must validate platform and instance_group
  if (config.has_ensemble_scheduling()) {
    RETURN_IF_ERROR(ValidateEnsembleSchedulingConfig(config));
  } else {
    if (config.platform() == kEnsemblePlatform) {
      return Status(
          Status::Code::INVALID_ARG,
          "ensemble scheduling must be set for ensemble " + config.name() +
              " whose platform is " + kEnsemblePlatform);
    }

    if (config.instance_group().size() == 0) {
      return Status(
          Status::Code::INVALID_ARG,
          "must specify one or more 'instance group's for " + config.name());
    }

    // Make sure KIND_GPU instance group specifies at least one GPU and
    // doesn't specify a non-existent GPU. Make sure non-KIND_GPU does
    // not specify any GPUs.
#ifdef TRTIS_ENABLE_GPU
    std::set<int> supported_gpus;
    Status status = GetSupportedGPUs(&supported_gpus, min_compute_capability);
    if (!status.IsOk()) {
      return status;
    }
#endif  // TRTIS_ENABLE_GPU

    for (const auto& group : config.instance_group()) {
      // KIND_MODEL is supported only on TensorFlow.
      if (group.kind() == ModelInstanceGroup::KIND_MODEL) {
        if (group.gpus().size() > 0) {
          return Status(
              Status::Code::INVALID_ARG,
              "instance group " + group.name() + " of model " + config.name() +
                  " has kind KIND_MODEL but specifies one or more GPUs");
        }
#ifdef TRTIS_ENABLE_TENSORFLOW
        if (!(config.platform() == kTensorFlowGraphDefPlatform ||
              config.platform() == kTensorFlowSavedModelPlatform))
#endif  // TRTIS_ENABLE_TENSORFLOW
        {
          return Status(
              Status::Code::INVALID_ARG,
              "instance group " + group.name() + " of model " + config.name() +
                  "on platform " + config.platform() +
                  " has kind KIND_MODEL which is supported only on TensorFlow "
                  "models");
        }
      } else if (group.kind() == ModelInstanceGroup::KIND_GPU) {
#ifndef TRTIS_ENABLE_GPU
        return Status(
            Status::Code::INVALID_ARG,
            "instance group " + group.name() + " of model " + config.name() +
                " has kind KIND_GPU but server does not support GPUs");
#else
        if (group.gpus().size() == 0) {
          if (supported_gpus.size() == 0) {
            return Status(
                Status::Code::INVALID_ARG,
                "instance group " + group.name() + " of model " +
                    config.name() +
                    " has kind KIND_GPU but no GPUs are available");
          } else {
            return Status(
                Status::Code::INVALID_ARG,
                "instance group " + group.name() + " of model " +
                    config.name() + " has kind KIND_GPU but specifies no GPUs");
          }
        }

        for (const int32_t gid : group.gpus()) {
          if (supported_gpus.find(gid) == supported_gpus.end()) {
            return Status(
                Status::Code::INVALID_ARG,
                "instance group " + group.name() + " of model " +
                    config.name() +
                    " specifies invalid or unsupported gpu id of " +
                    std::to_string(gid) +
                    ". The minimum required CUDA compute compatibility is " +
                    std::to_string(TRTIS_MIN_COMPUTE_CAPABILITY));
          }
        }
#endif  // !TRTIS_ENABLE_GPU
      } else if (group.kind() == ModelInstanceGroup::KIND_CPU) {
        if (group.gpus().size() > 0) {
          return Status(
              Status::Code::INVALID_ARG,
              "instance group " + group.name() + " of model " + config.name() +
                  " has kind KIND_CPU but specifies one or more GPUs");
        }
      } else {
        return Status(
            Status::Code::INTERNAL, "instance group " + group.name() +
                                        " of model " + config.name() +
                                        " has unexpected kind KIND_AUTO");
      }

      if (
#ifdef TRTIS_ENABLE_TENSORRT
          (config.platform() != kTensorRTPlanPlatform) &&
#endif  // TRTIS_ENABLE_TENSORRT
          !group.profile().empty()) {
        return Status(
            Status::Code::INVALID_ARG,
            "instance group " + group.name() + " of model " + config.name() +
                " and platform " + config.platform() +
                "specifies profile field which is only supported for "
                "TensorRT models");
      } else if (!group.profile().empty()) {
        for (const auto& profile : group.profile()) {
          int profile_index;
          RETURN_IF_ERROR(GetProfileIndex(profile, &profile_index));
          if (profile_index < 0) {
            return Status(
                Status::Code::INVALID_ARG,
                "instance group " + group.name() + " of model " +
                    config.name() + " and platform " + config.platform() +
                    " specifies invalid profile " + profile +
                    ". The field should contain the string representation of a "
                    "non-negative integer.");
          }
        }
      }
    }
  }

  return Status::Success;
}

namespace {

struct EnsembleTensor {
  EnsembleTensor(bool isOutput) : ready(false), isOutput(isOutput) {}
  bool ready;
  bool isOutput;
  std::vector<EnsembleTensor*> prev_nodes;
  std::vector<EnsembleTensor*> next_nodes;
};

/// Build a graph that represents the data flow in the ensemble specified in
/// given model config. the node (ensemble tensor) in the graph can be looked
/// up using its name as key.
/// \param ensemble_config The model configuration that specifies
/// ensemble_scheduling field.
/// \param keyed_ensemble_graph Returned the ensemble graph.
/// \return The error status. A non-OK status indicates the build fails because
/// the ensemble configuration is not valid.
Status
BuildEnsembleGraph(
    const ModelConfig& config,
    std::unordered_map<std::string, EnsembleTensor>& keyed_ensemble_graph)
{
  keyed_ensemble_graph.clear();
  size_t step_idx = 0;
  for (const auto& element : config.ensemble_scheduling().step()) {
    if (element.model_name().empty()) {
      return Status(
          Status::Code::INVALID_ARG,
          "must specify 'model_name' in step " + std::to_string(step_idx) +
              " of ensemble '" + config.name() + "'");
    }
    if (element.input_map().size() == 0) {
      return Status(
          Status::Code::INVALID_ARG,
          "must specify 'input_map' in step " + std::to_string(step_idx) +
              " of ensemble '" + config.name() + "'");
    }
    if (element.output_map().size() == 0) {
      return Status(
          Status::Code::INVALID_ARG,
          "must specify 'output_map' in step " + std::to_string(step_idx) +
              " of ensemble '" + config.name() + "'");
    }

    // Link ensemble tensors
    std::vector<EnsembleTensor*> tensor_as_output;
    for (const auto& output_map : element.output_map()) {
      auto it = keyed_ensemble_graph.find(output_map.second);
      if (it != keyed_ensemble_graph.end()) {
        if (it->second.isOutput) {
          return Status(
              Status::Code::INVALID_ARG,
              "ensemble tensor '" + it->first +
                  "' can appear in an output map only once for ensemble '" +
                  config.name() + "' step " + std::to_string(step_idx));
        } else {
          it->second.isOutput = true;
        }
      } else {
        it = keyed_ensemble_graph
                 .emplace(
                     std::make_pair(output_map.second, EnsembleTensor(true)))
                 .first;
      }
      tensor_as_output.push_back(&(it->second));
    }

    std::set<std::string> model_inputs;
    for (const auto& input_map : element.input_map()) {
      if (model_inputs.find(input_map.first) != model_inputs.end()) {
        return Status(
            Status::Code::INVALID_ARG,
            "input '" + input_map.first + "' in model '" +
                element.model_name() +
                "' is mapped to multiple ensemble tensors for ensemble '" +
                config.name() + "' step " + std::to_string(step_idx));
      } else {
        model_inputs.emplace(input_map.first);
      }
      auto it = keyed_ensemble_graph.find(input_map.second);
      if (it == keyed_ensemble_graph.end()) {
        it = keyed_ensemble_graph
                 .emplace(
                     std::make_pair(input_map.second, EnsembleTensor(false)))
                 .first;
      }
      for (auto output : tensor_as_output) {
        output->prev_nodes.push_back(&(it->second));
        it->second.next_nodes.push_back(output);
      }
    }

    step_idx++;
  }

  return Status::Success;
}

}  // namespace

Status
ValidateEnsembleSchedulingConfig(const ModelConfig& config)
{
  if (config.platform() != kEnsemblePlatform) {
    return Status(
        Status::Code::INVALID_ARG,
        "ensemble scheduling cannot be set for model '" + config.name() +
            "' whose platform is not " + kEnsemblePlatform);
  }
  if (config.instance_group().size() != 0) {
    return Status(
        Status::Code::INVALID_ARG,
        "instance group should not be specified for ensemble '" +
            config.name() + "'");
  }
  if (config.has_optimization()) {
    return Status(
        Status::Code::INVALID_ARG,
        "optimization should not be specified for ensemble '" + config.name() +
            "'");
  }
  if (config.model_warmup_size() != 0) {
    return Status(
        Status::Code::INVALID_ARG,
        "model_warmup can not be specified for ensemble '" + config.name() +
            "'");
  }

  // Make sure step is not empty and all fields are set
  if (config.ensemble_scheduling().step_size() == 0) {
    return Status(
        Status::Code::INVALID_ARG,
        "must specify 'step' for ensemble '" + config.name() + "'");
  }

  std::unordered_map<std::string, EnsembleTensor> tensors;

  RETURN_IF_ERROR(BuildEnsembleGraph(config, tensors));

  // check data flow
  std::deque<EnsembleTensor*> ready_queue;
  for (const auto& input : config.input()) {
    auto it = tensors.find(input.name());
    if (it == tensors.end()) {
      return Status(
          Status::Code::INVALID_ARG, "ensemble input '" + input.name() +
                                         "' for ensemble " + config.name() +
                                         "' is not used");
    }
    it->second.ready = true;
    ready_queue.push_back(&(it->second));
  }
  while (!ready_queue.empty()) {
    auto& ready_node = ready_queue.front();
    for (auto& next_node : ready_node->next_nodes) {
      if (next_node->ready) {
        continue;
      }
      bool next_node_ready = true;
      for (auto& prev_node : next_node->prev_nodes) {
        if (!prev_node->ready) {
          next_node_ready = false;
          break;
        }
      }
      next_node->ready = next_node_ready;
      if (next_node_ready) {
        ready_queue.push_back(next_node);
      }
    }
    ready_queue.pop_front();
  }
  std::set<std::string> outputs;
  for (const auto& output : config.output()) {
    auto it = tensors.find(output.name());
    if (it == tensors.end()) {
      return Status(
          Status::Code::INVALID_ARG, "ensemble output '" + output.name() +
                                         "' for ensemble " + config.name() +
                                         "' is not used");
    }
    if (!it->second.ready) {
      return Status(
          Status::Code::INVALID_ARG, "output '" + output.name() +
                                         "' for ensemble '" + config.name() +
                                         "' is not written");
    } else {
      outputs.insert(it->first);
    }
  }
  // Check redundant ensemble tensors
  for (const auto& tensor : tensors) {
    // skip ensemble outputs as they have been checked and can have no
    // next nodes
    if (outputs.find(tensor.first) != outputs.end()) {
      continue;
    }
    if (!tensor.second.ready || (tensor.second.next_nodes.size() == 0)) {
      return Status(
          Status::Code::INVALID_ARG, "ensemble tensor '" + tensor.first +
                                         "' is unused in ensemble '" +
                                         config.name() + "'");
    }
  }
  return Status::Success;
}

namespace {

template <class ModelIO>
Status
ValidateIOShape(
    const ModelIO& io, int32_t max_batch_size,
    const std::string& message_prefix = "")
{
  if (io.name().empty()) {
    return Status(
        Status::Code::INVALID_ARG, message_prefix + "must specify 'name'");
  }

  if (io.data_type() == DataType::TYPE_INVALID) {
    return Status(
        Status::Code::INVALID_ARG, "model output must specify 'data_type'");
  }

  if (io.dims_size() == 0) {
    return Status(
        Status::Code::INVALID_ARG, message_prefix + "must specify 'dims'");
  }

  // If the configuration is non-batching, then no input or output
  // reshape can be empty as that would mean that input or output was
  // always empty (no data).
  if (io.has_reshape() && (io.reshape().shape_size() == 0) &&
      (max_batch_size == 0)) {
    return Status(
        Status::Code::INVALID_ARG,
        message_prefix + "cannot have empty reshape for non-batching model");
  }

  for (auto dim : io.dims()) {
    // Dimension cannot be 0.
    if ((dim < 1) && (dim != WILDCARD_DIM)) {
      return Status(
          Status::Code::INVALID_ARG,
          message_prefix + "dimension must be integer >= 1, or " +
              std::to_string(WILDCARD_DIM) +
              " to indicate a variable-size dimension");
    }
  }

  if (io.has_reshape()) {
    // Zeros are not allowed in reshape.
    for (auto dim : io.reshape().shape()) {
      if ((dim < 1) && (dim != WILDCARD_DIM)) {
        return Status(
            Status::Code::INVALID_ARG,
            message_prefix + "reshape dimensions must be integer >= 1, or " +
                std::to_string(WILDCARD_DIM) +
                " to indicate a variable-size dimension");
      }
    }

    const int64_t dims_size = GetElementCount(io.dims());
    const int64_t reshape_size = GetElementCount(io.reshape().shape());

    // dims and reshape must both have same element count
    // or both have variable-size dimension.
    // Special case for empty reshape... expect dims to have element
    // count of 1.
    if ((dims_size != reshape_size) &&
        ((reshape_size != 0) || (dims_size != 1))) {
      return Status(
          Status::Code::INVALID_ARG,
          message_prefix + "has different size for dims and reshape");
    }

    // shape contains variable-size dimension, in this case we compare if
    // each pair of the trunks separated by variable-size dimension has
    // the same element count. For instance, from [2, 4, -1, 6] to [8, -1, 1, 6]
    // is valid reshape as 2 * 4 = 8 and 6 = 1 * 6.
    if (dims_size == -1) {
      std::vector<int64_t> dim_element_cnts;
      std::vector<int64_t> reshape_element_cnts;
      int64_t current_cnt = 1;
      for (const auto& dim : io.dims()) {
        if (dim != -1) {
          current_cnt *= dim;
        } else {
          dim_element_cnts.push_back(current_cnt);
          current_cnt = 1;
        }
      }
      dim_element_cnts.push_back(current_cnt);

      current_cnt = 1;
      for (const auto& dim : io.reshape().shape()) {
        if (dim != -1) {
          current_cnt *= dim;
        } else {
          reshape_element_cnts.push_back(current_cnt);
          current_cnt = 1;
        }
      }
      reshape_element_cnts.push_back(current_cnt);

      if (dim_element_cnts.size() != reshape_element_cnts.size()) {
        return Status(
            Status::Code::INVALID_ARG,
            message_prefix +
                "has different number of variable-size dimensions for dims "
                "and reshape");
      }
      for (size_t idx = 0; idx < dim_element_cnts.size(); idx++) {
        if (dim_element_cnts[idx] != reshape_element_cnts[idx]) {
          return Status(
              Status::Code::INVALID_ARG,
              message_prefix + "has different size for dims and reshape");
        }
      }
    }
  }

  return Status::Success;
}

}  // namespace

Status
ValidateModelInput(
    const ModelInput& io, int32_t max_batch_size, const std::string& platform)
{
  RETURN_IF_ERROR(ValidateIOShape(io, max_batch_size, "model input "));

  if (((io.format() == ModelInput::FORMAT_NHWC) ||
       (io.format() == ModelInput::FORMAT_NCHW)) &&
      (io.dims_size() != 3)) {
    return Status(
        Status::Code::INVALID_ARG, "model input NHWC/NCHW require 3 dims");
  }

  if (
#ifdef TRTIS_ENABLE_TENSORRT
      (platform != kTensorRTPlanPlatform) &&
#endif  // TRTIS_ENABLE_TENSORRT
      io.is_shape_tensor()) {
    return Status(
        Status::Code::INVALID_ARG,
        "shape tensors are only supported for TensorRT platform");
  }

  if (
#ifdef TRTIS_ENABLE_CUSTOM
      (platform != kCustomPlatform) &&
#endif  // TRTIS_ENABLE_CUSTOM
      io.allow_ragged_batch()) {
    return Status(
        Status::Code::INVALID_ARG,
        "ragged-batch input tensors are only supported for custom platform");
  }

  return Status::Success;
}

Status
CheckAllowedModelInput(
    const ModelInput& io, const std::set<std::string>& allowed)
{
  if (allowed.find(io.name()) == allowed.end()) {
    std::string astr;
    for (const auto& a : allowed) {
      if (!astr.empty()) {
        astr.append(", ");
      }
      astr.append(a);
    }

    return Status(
        Status::Code::INVALID_ARG, "unexpected inference input '" + io.name() +
                                       "', allowed inputs are: " + astr);
  }
  return Status::Success;
}

Status
ValidateModelOutput(
    const ModelOutput& io, int32_t max_batch_size, const std::string& platform)
{
  RETURN_IF_ERROR(ValidateIOShape(io, max_batch_size, "model output "));

  if (
#ifdef TRTIS_ENABLE_TENSORRT
      (platform != kTensorRTPlanPlatform) &&
#endif  // TRTIS_ENABLE_TENSORRT
      io.is_shape_tensor()) {
    return Status(
        Status::Code::INVALID_ARG,
        "shape tensors are only supported for TensorRT platform");
  }

  return Status::Success;
}

Status
CheckAllowedModelOutput(
    const ModelOutput& io, const std::set<std::string>& allowed)
{
  if (allowed.find(io.name()) == allowed.end()) {
    std::string astr;
    for (const auto& a : allowed) {
      if (!astr.empty()) {
        astr.append(", ");
      }
      astr.append(a);
    }

    return Status(
        Status::Code::INVALID_ARG, "unexpected inference output '" + io.name() +
                                       "', allowed outputs are: " + astr);
  }

  return Status::Success;
}

Status
ParseBoolParameter(
    const std::string& key, std::string value, bool* parsed_value)
{
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return std::tolower(c); });

  if ((value == "true") || (value == "1")) {
    *parsed_value = true;
  } else if ((value == "false") || (value == "0")) {
    *parsed_value = false;
  } else {
    return Status(
        Status::Code::INVALID_ARG,
        "failed to convert " + key + " '" + value + "' to boolean value");
  }

  return Status::Success;
}

Status
ParseLongLongParameter(
    const std::string& key, const std::string& value, int64_t* parsed_value)
{
  try {
    *parsed_value = std::stoll(value);
  }
  catch (const std::invalid_argument& ia) {
    return Status(
        Status::Code::INVALID_ARG,
        "failed to convert " + key + " '" + value + "' to integral number");
  }

  return Status::Success;
}

Status
GetProfileIndex(const std::string& profile_name, int* profile_index)
{
  if (profile_name.empty()) {
    return Status(Status::Code::INVALID_ARG, "profile name must not be empty");
  } else {
    try {
      *profile_index = stoi(profile_name);
    }
    catch (const std::invalid_argument& ia) {
      return Status(
          Status::Code::INVALID_ARG,
          "unable to parse '" + profile_name + "': " + ia.what());
    }
    return Status::Success;
  }
}

}}  // namespace nvidia::inferenceserver
