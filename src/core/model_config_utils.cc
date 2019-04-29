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

#include "src/core/model_config_utils.h"

#include <deque>
#include <set>
#include "absl/strings/numbers.h"
#include "cuda/include/cuda_runtime_api.h"
#include "src/core/autofill.h"
#include "src/core/constants.h"
#include "src/core/filesystem.h"
#include "src/core/logging.h"

namespace nvidia { namespace inferenceserver {

Status
GetModelVersionFromPath(const std::string& path, int64_t* version)
{
  auto version_dir = BaseName(path);

  // Determine the version from the last segment of 'path'
  if (!absl::SimpleAtoi(version_dir, version)) {
    return Status(
        RequestStatusCode::INTERNAL,
        "unable to determine model version from " + path);
  }

  return Status::Success;
}

Status
GetSequenceControlProperties(
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
          RequestStatusCode::INVALID_ARG,
          "sequence batching control tensor must have a name for " +
              model_name);
    }

    if (seen_tensors.find(control_input.name()) != seen_tensors.end()) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "sequence batching control tensor '" + control_input.name() +
              "' is specified for multiple control kinds for " + model_name);
    }

    seen_tensors.insert(control_input.name());

    for (const auto& c : control_input.control()) {
      if (c.kind() == control_kind) {
        if (seen_control) {
          return Status(
              RequestStatusCode::INVALID_ARG,
              "sequence batching specifies multiple " +
                  ModelSequenceBatching_Control_Kind_Name(control_kind) +
                  " tensors for " + model_name);
        }

        *tensor_name = control_input.name();
        seen_control = true;

        if (c.int32_false_true_size() > 0) {
          if (c.fp32_false_true_size() != 0) {
            return Status(
                RequestStatusCode::INVALID_ARG,
                "sequence batching specifies both 'int32_false_true' and "
                "'fp32_false_true' for " +
                    ModelSequenceBatching_Control_Kind_Name(control_kind) +
                    " for " + model_name);
          }

          if (c.int32_false_true_size() != 2) {
            return Status(
                RequestStatusCode::INVALID_ARG,
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
                RequestStatusCode::INVALID_ARG,
                "sequence batching must specify either 'int32_false_true' or "
                "'fp32_false_true' for " +
                    ModelSequenceBatching_Control_Kind_Name(control_kind) +
                    " for " + model_name);
          }

          if (c.fp32_false_true_size() != 2) {
            return Status(
                RequestStatusCode::INVALID_ARG,
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
          RequestStatusCode::INVALID_ARG,
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
    const std::string& path, const PlatformConfigMap& platform_config_map,
    const bool autofill, ModelConfig* config)
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
        model_name, platform_config_map, std::string(path), *config, &af));
    RETURN_IF_ERROR(af->Fix(config));

    LOG_VERBOSE(1) << "autofilled config: " << config->DebugString();
  }

  if (config->platform().empty()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "must specify platform for model '" + config->name() + "'");
  }

  // If 'default_model_filename' is not specified set it appropriately
  // based upon 'platform'.
  if (config->default_model_filename().empty()) {
    if (config->platform() == kTensorFlowGraphDefPlatform) {
      config->set_default_model_filename(kTensorFlowGraphDefFilename);
    } else if (config->platform() == kTensorFlowSavedModelPlatform) {
      config->set_default_model_filename(kTensorFlowSavedModelFilename);
    } else if (config->platform() == kTensorRTPlanPlatform) {
      config->set_default_model_filename(kTensorRTPlanFilename);
    } else if (config->platform() == kCaffe2NetDefPlatform) {
      config->set_default_model_filename(kCaffe2NetDefFilename);
    } else if (config->platform() == kCustomPlatform) {
      config->set_default_model_filename(kCustomFilename);
    } else if (config->platform() == kEnsemblePlatform) {
      // No actual model file is needed to be loaded for ensemble.
    } else {
      return Status(
          RequestStatusCode::INTERNAL, "unexpected platform type " +
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
    // Make sure there is at least one instance_group.
    if (config->instance_group().size() == 0) {
      ModelInstanceGroup* group = config->add_instance_group();
      group->set_name(config->name());
    }

    int device_cnt;
    cudaError_t cuerr = cudaGetDeviceCount(&device_cnt);
    if (cuerr == cudaErrorNoDevice) {
      device_cnt = 0;
    } else if (cuerr != cudaSuccess) {
      return Status(
          RequestStatusCode::INTERNAL,
          "unable to get number of CUDA devices for " + config->name() + ": " +
              cudaGetErrorString(cuerr));
    }

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
        if (device_cnt == 0) {
          group.set_kind(ModelInstanceGroup::KIND_CPU);
        } else {
          for (const int32_t gid : group.gpus()) {
            if ((gid < 0) || (gid >= device_cnt)) {
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
        for (int d = 0; d < device_cnt; d++) {
          group.add_gpus(d);
        }
      }
    }
  }

  return Status::Success;
}

Status
ValidateModelConfig(
    const ModelConfig& config, const std::string& expected_platform)
{
  if (config.name().empty()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "model configuration must specify 'name'");
  }

  if (config.platform().empty()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "must specify 'platform' for " + config.name());
  }

  if (!expected_platform.empty() && (config.platform() != expected_platform)) {
    return Status(
        RequestStatusCode::NOT_FOUND, "expected model of type " +
                                          expected_platform + " for " +
                                          config.name());
  }

  if (!config.has_version_policy()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "must specify 'version policy' for " + config.name());
  }

  Status status;
  for (const auto& io : config.input()) {
    status = ValidateModelInput(io, config.max_batch_size());
    if (!status.IsOk()) {
      return Status(status.Code(), status.Message() + " for " + config.name());
    }
  }
  for (const auto& io : config.output()) {
    status = ValidateModelOutput(io, config.max_batch_size());
    if (!status.IsOk()) {
      return Status(status.Code(), status.Message() + " for " + config.name());
    }
  }

  // If dynamic batching is specified make sure the preferred batch
  // sizes are positive and don't exceed maximum batch size. Make sure
  // the max delay is non-negative.
  if (config.has_dynamic_batching()) {
    for (const auto size : config.dynamic_batching().preferred_batch_size()) {
      if (size <= 0) {
        return Status(
            RequestStatusCode::INVALID_ARG,
            "dynamic batching preferred size must be positive for " +
                config.name());
      }
      if (size > config.max_batch_size()) {
        return Status(
            RequestStatusCode::INVALID_ARG,
            "dynamic batching preferred size must be <= max batch size for " +
                config.name());
      }
    }
  }

  // If sequence batching is specified make sure the control is
  // specified correctly.
  if (config.has_sequence_batching()) {
    const auto& batcher = config.sequence_batching();

    if (batcher.control_input_size() == 0) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "sequence batching must specify at least one control tensor for " +
              config.name());
    }

    // Make sure at most one SEQUENCE_START and one SEQUENCE_READY
    // control is specified.
    std::string tensor_name;
    RETURN_IF_ERROR(GetSequenceControlProperties(
        batcher, config.name(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_START,
        true /* required */, &tensor_name, nullptr, nullptr, nullptr, nullptr,
        nullptr));
    RETURN_IF_ERROR(GetSequenceControlProperties(
        batcher, config.name(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_READY,
        true /* required */, &tensor_name, nullptr, nullptr, nullptr, nullptr,
        nullptr));
  }

  // If ensemble scheduling is specified, validate it.
  // Otherwise, must validate platform and instance_group
  if (config.has_ensemble_scheduling()) {
    RETURN_IF_ERROR(ValidateEnsembleSchedulingConfig(config));
  } else {
    if (config.platform() == kEnsemblePlatform) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "ensemble scheduling must be set for ensemble " + config.name() +
              " whose platform is " + kEnsemblePlatform);
    }

    if (config.instance_group().size() == 0) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "must specify one or more 'instance group's for " + config.name());
    }

    // Make sure KIND_GPU instance group specifies at least one GPU and
    // doesn't specify a non-existent GPU. Make sure non-KIND_GPU does
    // not specify any GPUs.
    int dcnt;
    cudaError_t cuerr = cudaGetDeviceCount(&dcnt);
    if (cuerr == cudaErrorNoDevice) {
      dcnt = 0;
    } else if (cuerr != cudaSuccess) {
      return Status(
          RequestStatusCode::INTERNAL,
          "failed to get device count for validation of model " +
              config.name() + ": " + cudaGetErrorString(cuerr));
    }

    for (const auto& group : config.instance_group()) {
      if (group.kind() == ModelInstanceGroup::KIND_GPU) {
        if (group.gpus().size() == 0) {
          return Status(
              RequestStatusCode::INVALID_ARG,
              "instance group " + group.name() + " of model " + config.name() +
                  " has kind KIND_GPU but specifies no GPUs");
        }

        for (const int32_t gid : group.gpus()) {
          if ((gid < 0) || (gid >= dcnt)) {
            return Status(
                RequestStatusCode::INVALID_ARG,
                "instance group " + group.name() + " of model " +
                    config.name() + " specifies invalid GPU id " +
                    std::to_string(gid) + ", valid GPUs are 0 - " +
                    std::to_string(dcnt - 1));
          }
        }
      } else if (group.kind() == ModelInstanceGroup::KIND_CPU) {
        if (group.gpus().size() > 0) {
          return Status(
              RequestStatusCode::INVALID_ARG,
              "instance group " + group.name() + " of model " + config.name() +
                  " has kind KIND_CPU but specifies one or more GPUs");
        }
      } else {
        return Status(
            RequestStatusCode::INTERNAL, "instance group " + group.name() +
                                             " of model " + config.name() +
                                             " has unexpected kind KIND_AUTO");
      }
    }
  }

  return Status::Success;
}

Status
ValidateEnsembleSchedulingConfig(const ModelConfig& config)
{
  if (config.platform() != kEnsemblePlatform) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "ensemble scheduling can not be set for model " + config.name() +
            " whose platform is not " + kEnsemblePlatform);
  }
  if (config.instance_group().size() != 0) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "instance group should not be specified for ensemble " + config.name());
  }
  if (config.has_optimization()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "optimization should not be specified for ensemble " + config.name());
  }

  // Make sure step is not empty and all fields are set
  if (config.ensemble_scheduling().step_size() == 0) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "must specify 'step' for ensemble " + config.name());
  }

  std::unordered_map<std::string, EnsembleTensor> tensors;

  RETURN_IF_ERROR(BuildEnsembleGraph(config, tensors));

  // check data flow
  std::deque<EnsembleTensor*> ready_queue;
  for (const auto& input : config.input()) {
    auto it = tensors.find(input.name());
    if (it == tensors.end()) {
      return Status(
          RequestStatusCode::INVALID_ARG, "must map ensemble input " +
                                              input.name() + " for ensemble " +
                                              config.name());
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
          RequestStatusCode::INVALID_ARG, "must map ensemble output " +
                                              output.name() + " for ensemble " +
                                              config.name());
    }
    if (!it->second.ready) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "no data will be written to ensemble output " + output.name() +
              " for ensemble " + config.name());
    } else {
      outputs.insert(it->first);
    }
  }
  // Check redundant ensemble tensors
  for (const auto& tensor : tensors) {
    // skip ensemble outputs as they have been check and can have no next nodes
    if (outputs.find(tensor.first) != outputs.end()) {
      continue;
    }
    if (!tensor.second.ready) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "ensemble tensor " + tensor.first +
              " is redundant as no data will be written to it for ensemble " +
              config.name());
    } else if (tensor.second.next_nodes.size() == 0) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "ensemble tensor " + tensor.first +
              " is redundant as it will not be used in any models for "
              "ensemble " +
              config.name());
    }
  }
  return Status::Success;
}

Status
BuildEnsembleGraph(
    const ModelConfig& config,
    std::unordered_map<std::string, EnsembleTensor>& keyed_ensemble_graph)
{
  keyed_ensemble_graph.clear();
  for (const auto& element : config.ensemble_scheduling().step()) {
    if (element.model_name().empty()) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "must specify 'model_name' in step of ensemble " + config.name());
    }
    if (element.input_map().size() == 0) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "must specify one or more 'input_map' in step of ensemble " +
              config.name());
    }
    if (element.output_map().size() == 0) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "must specify one or more 'output_map' in step of ensemble " +
              config.name());
    }

    // Link ensemble tensors
    std::vector<EnsembleTensor*> tensor_as_output;
    for (const auto& output_map : element.output_map()) {
      auto it = keyed_ensemble_graph.find(output_map.second);
      if (it != keyed_ensemble_graph.end()) {
        if (it->second.isOutput) {
          return Status(
              RequestStatusCode::INVALID_ARG,
              "ensemble tensor " + it->first +
                  " can appear in an output map only once for ensemble " +
                  config.name());
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
            RequestStatusCode::INVALID_ARG,
            "input " + input_map.first + " in model " + element.model_name() +
                " is mapped to multiple ensemble tensors in one step for "
                "ensemble " +
                config.name());
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
  }

  return Status::Success;
}

Status
ValidateModelInput(const ModelInput& io, int32_t max_batch_size)
{
  if (io.name().empty()) {
    return Status(
        RequestStatusCode::INVALID_ARG, "model input must specify 'name'");
  }

  if (io.data_type() == DataType::TYPE_INVALID) {
    return Status(
        RequestStatusCode::INVALID_ARG, "model input must specify 'data_type'");
  }

  if (io.dims_size() == 0) {
    return Status(
        RequestStatusCode::INVALID_ARG, "model input must specify 'dims'");
  }

  // If the configuration is non-batching, then no input or output
  // reshape can be empty as that would mean that input or output was
  // always empty (no data).
  if (io.has_reshape() && (io.reshape().shape_size() == 0) &&
      (max_batch_size == 0)) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "model input cannot have empty reshape for non-batching model");
  }

  for (auto dim : io.dims()) {
    // Dimension cannot be 0.
    if ((dim < 1) && (dim != WILDCARD_DIM)) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "model input dimension must be integer >= 1, or " +
              std::to_string(WILDCARD_DIM) +
              " to indicate a variable-size dimension");
    }

    // Wildcards are not allowed in dims if there is a reshape.
    if (io.has_reshape() && (dim == WILDCARD_DIM)) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "model input using reshape cannot have variable-size dimension " +
              std::to_string(WILDCARD_DIM));
    }
  }

  // Wildcards and zeros are not allowed in reshape. Make sure the
  // element count input dims matches the reshape.
  if (io.has_reshape()) {
    for (auto dim : io.reshape().shape()) {
      if (dim <= 0) {
        return Status(
            RequestStatusCode::INVALID_ARG,
            "model input reshape dimensions must be integer >= 1");
      }
    }

    // Special case for empty reshape... expect dims to have element
    // count of 1.
    const int64_t reshape_size =
        std::max((int64_t)1, GetElementCount(io.reshape().shape()));
    const int64_t dims_size = GetElementCount(io.dims());
    if (dims_size != reshape_size) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "model input has different size for dims and reshape");
    }
  }

  if (((io.format() == ModelInput::FORMAT_NHWC) ||
       (io.format() == ModelInput::FORMAT_NCHW)) &&
      (io.dims_size() != 3)) {
    return Status(
        RequestStatusCode::INVALID_ARG, "model input NHWC/NCHW require 3 dims");
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
        RequestStatusCode::INVALID_ARG, "unexpected inference input '" +
                                            io.name() +
                                            "', allowed inputs are: " + astr);
  }
  return Status::Success;
}

Status
ValidateModelOutput(const ModelOutput& io, int32_t max_batch_size)
{
  if (io.name().empty()) {
    return Status(
        RequestStatusCode::INVALID_ARG, "model output must specify 'name'");
  }

  if (io.data_type() == DataType::TYPE_INVALID) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "model output must specify 'data_type'");
  }

  if (io.dims_size() == 0) {
    return Status(
        RequestStatusCode::INVALID_ARG, "model output must specify 'dims'");
  }

  // If the configuration is non-batching, then no input or output
  // reshape can be empty as that would mean that input or output was
  // always empty (no data).
  if (io.has_reshape() && (io.reshape().shape_size() == 0) &&
      (max_batch_size == 0)) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "model output cannot have empty reshape for non-batching model");
  }

  for (auto dim : io.dims()) {
    // Dimension cannot be 0.
    if ((dim < 1) && (dim != WILDCARD_DIM)) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "model output dimension must be integer >= 1, or " +
              std::to_string(WILDCARD_DIM) +
              " to indicate a variable-size dimension");
    }

    // Wildcards are not allowed in dims if there is a reshape.
    if (io.has_reshape() && (dim == WILDCARD_DIM)) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "model output using reshape cannot have variable-size dimension " +
              std::to_string(WILDCARD_DIM));
    }
  }

  // Wildcards and zeros are not allowed in reshape. Make sure the
  // element count output dims matches the reshape.
  if (io.has_reshape()) {
    for (auto dim : io.reshape().shape()) {
      if (dim <= 0) {
        return Status(
            RequestStatusCode::INVALID_ARG,
            "model output reshape dimensions must be integer >= 1");
      }
    }

    // Special case for empty reshape... expect dims to have element
    // count of 1.
    const int64_t reshape_size =
        std::max((int64_t)1, GetElementCount(io.reshape().shape()));
    const int64_t dims_size = GetElementCount(io.dims());
    if (dims_size != reshape_size) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "model output has different size for dims and reshape");
    }
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
        RequestStatusCode::INVALID_ARG, "unexpected inference output '" +
                                            io.name() +
                                            "', allowed outputs are: " + astr);
  }

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
