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

#include "src/core/utils.h"

#include <set>
#include "absl/strings/numbers.h"
#include "cuda/include/cuda_runtime_api.h"
#include "src/core/autofill.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"

namespace nvidia { namespace inferenceserver {

namespace {

struct TensorNode {
  TensorNode(std::string model, DataType type, DimsList dims)
      : model_name(model), type(type), dims(dims), ready(false)
  {
  }

  std::string model_name;
  DataType type;
  DimsList dims;
  bool ready;
  std::vector<struct TensorNode*> prev_nodes;
  std::vector<struct TensorNode*> next_nodes;
};

std::string
DimsListToString(const DimsList& list)
{
  std::string res = "[ ";
  for (const auto& dim : list) {
    res = res + std::to_string(dim) + " ";
  }
  return res + "]";
}

tensorflow::Status
ValidateTensorConsistency(
    const struct TensorNode& lhs, const struct TensorNode& rhs,
    const std::string& message)
{
  if (lhs.type != rhs.type) {
    return tensorflow::errors::InvalidArgument(
        message, "inconsistent data type: ", lhs.type,
        " is inferred from model ", lhs.model_name, " while ", rhs.type,
        " is inferred from model ", rhs.model_name);
  }
  bool consistent = (lhs.dims.size() == rhs.dims.size());
  if (consistent) {
    for (size_t i = 0; i < lhs.dims.size(); i++) {
      if (lhs.dims[i] != rhs.dims[i]) {
        consistent = false;
        break;
      }
    }
  }
  if (!consistent) {
    return tensorflow::errors::InvalidArgument(
        message, "inconsistent shape: ", DimsListToString(lhs.dims),
        " is inferred from model ", lhs.model_name, " while ",
        DimsListToString(rhs.dims), " is inferred from model ", rhs.model_name);
  }
  return tensorflow::Status::OK();
}

tensorflow::Status
ValidateEnsembleConfig(
    const std::string& ensemble,
    const std::unordered_map<std::string, ModelConfig>& config_map,
    const std::unordered_map<std::string, std::string>& invalid_model_names,
    std::unordered_map<std::string, bool>& ensembles,
    std::deque<std::string>& ensemble_dependency)
{
  std::unordered_map<std::string, struct TensorNode> ensemble_tensors;

  const auto& ensemble_config = config_map.at(ensemble);

  for (const auto& input : ensemble_config.input()) {
    struct TensorNode input_node(ensemble, input.data_type(), input.dims());
    ensemble_tensors.emplace(std::make_pair(input.name(), input_node));
  }
  for (const auto& output : ensemble_config.output()) {
    struct TensorNode output_node(ensemble, output.data_type(), output.dims());
    ensemble_tensors.emplace(std::make_pair(output.name(), output_node));
  }

  for (const auto& step : ensemble_config.ensemble_scheduling().step()) {
    const auto& model_name = step.model_name();
    if (invalid_model_names.find(model_name) != invalid_model_names.end()) {
      return tensorflow::errors::InvalidArgument(
          "ensemble ", ensemble, " contains invalid model ", model_name, " : ",
          invalid_model_names.at(model_name));
    }
    auto it = config_map.find(model_name);
    if (it == config_map.end()) {
      return tensorflow::errors::InvalidArgument(
          "ensemble ", ensemble, " contains model ", model_name,
          " which is not in the available models");
    }
    const auto& model_config = it->second;
    if (model_config.max_batch_size() < ensemble_config.max_batch_size()) {
      return tensorflow::errors::InvalidArgument(
          "ensemble ", ensemble, " allows maximum batch size ",
          ensemble_config.max_batch_size(), ", but it contains model ",
          model_name, " which only allows  maximum batch size to be ",
          model_config.max_batch_size());
    }

    if (model_config.has_ensemble_scheduling()) {
      for (const auto& name : ensemble_dependency) {
        if (name == model_name) {
          return tensorflow::errors::InvalidArgument(
              "circular dependency between ensembles: ", name, " -> ... -> ",
              ensemble, " -> ", name);
        }
      }

      if ((ensembles.find(model_name))->second == false) {
        ensemble_dependency.push_back(ensemble);
        TF_RETURN_IF_ERROR(ValidateEnsembleConfig(
            model_name, config_map, invalid_model_names, ensembles,
            ensemble_dependency));
        ensemble_dependency.pop_back();
      }
    }

    // Check all inputs are mapped and no mapping to invalid inputs
    std::set<std::string> input_names;
    for (const auto& model_input : model_config.input()) {
      input_names.insert(model_input.name());
    }
    for (const auto& input_map : step.input_map()) {
      if (input_names.find(input_map.second) == input_names.end()) {
        return tensorflow::errors::InvalidArgument(
            "in ensemble ", ensemble, ", ensemble tensor ", input_map.first,
            " is mapping to non-existing input ", input_map.second,
            " in model ", step.model_name());
      }
    }
    for (const auto& model_input : model_config.input()) {
      bool found = false;
      for (const auto& input_map : step.input_map()) {
        found = (model_input.name() == input_map.second);
        if (found) {
          struct TensorNode model_tensor(
              step.model_name(), model_input.data_type(), model_input.dims());
          auto it = ensemble_tensors.find(input_map.first);
          if (it != ensemble_tensors.end()) {
            TF_RETURN_IF_ERROR(ValidateTensorConsistency(
                it->second, model_tensor,
                "in ensemble " + ensemble + ", ensemble tensor " +
                    input_map.first + ": "));
          } else {
            ensemble_tensors.emplace(
                std::make_pair(input_map.first, model_tensor));
          }
        }
      }
      if (!found) {
        return tensorflow::errors::InvalidArgument(
            "in ensemble ", ensemble, ", input ", model_input.name(),
            " in model ", model_config.name(),
            " is not mapped to any ensemble tensors");
      }
    }

    // Check no multiple mappings to same ensemble tensor
    // and no mapping from invalid outputs
    std::set<std::string> mapped;
    for (const auto& output_map : step.output_map()) {
      if (mapped.find(output_map.second) == mapped.end()) {
        mapped.insert(output_map.second);
      } else {
        return tensorflow::errors::InvalidArgument(
            "in ensemble " + ensemble + ", multiple outputs in model ",
            model_config.name(), " are mapped to the same ensemble tensor ",
            output_map.second);
      }
      bool found = false;
      for (const auto& model_output : model_config.output()) {
        found = (model_output.name() == output_map.first);
        if (found) {
          struct TensorNode model_tensor(
              step.model_name(), model_output.data_type(), model_output.dims());
          auto it = ensemble_tensors.find(output_map.second);
          if (it != ensemble_tensors.end()) {
            TF_RETURN_IF_ERROR(ValidateTensorConsistency(
                it->second, model_tensor,
                "in ensemble " + ensemble + ", ensemble tensor " +
                    output_map.second + ": "));
          } else {
            ensemble_tensors.emplace(
                std::make_pair(output_map.second, model_tensor));
          }
        }
      }
      if (!found) {
        return tensorflow::errors::InvalidArgument(
            "in ensemble ", ensemble, ", ensemble tensor ", output_map.second,
            " is mapped from non-existing output ", output_map.first,
            " in model ", step.model_name());
      }
    }

    // link ensemble tensors
    for (const auto& output_map : step.output_map()) {
      auto& node = ensemble_tensors.find(output_map.second)->second;
      for (const auto& input_map : step.input_map()) {
        auto& prev_node = ensemble_tensors.find(input_map.first)->second;
        node.prev_nodes.push_back(&prev_node);
        prev_node.next_nodes.push_back(&node);
      }
    }
  }

  // Check data flow
  std::deque<struct TensorNode*> ready_queue;
  for (const auto& input : ensemble_config.input()) {
    auto it = ensemble_tensors.find(input.name());
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
  for (const auto& output : ensemble_config.output()) {
    auto it = ensemble_tensors.find(output.name());
    if (!it->second.ready) {
      return tensorflow::errors::InvalidArgument(
          "in ensemble ", ensemble, ", no data will be written to ",
          "ensemble output ", output.name(), " under optimistic assumption");
    }
  }
  (ensembles.find(ensemble))->second = true;
  return tensorflow::Status::OK();
}

}  // namespace

tensorflow::Status
GetModelVersionFromPath(const tensorflow::StringPiece& path, int64_t* version)
{
  auto version_dir = tensorflow::io::Basename(path);

  // Determine the version from the last segment of 'path'
  if (!absl::SimpleAtoi(version_dir, version)) {
    return tensorflow::errors::Internal(
        "unable to determine model version from ", path);
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
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
      return tensorflow::errors::InvalidArgument(
          "sequence batching control tensor must have a name for ", model_name);
    }

    if (seen_tensors.find(control_input.name()) != seen_tensors.end()) {
      return tensorflow::errors::InvalidArgument(
          "sequence batching control tensor '", control_input.name(),
          "' is specified for multiple control kinds for ", model_name);
    }

    seen_tensors.insert(control_input.name());

    for (const auto& c : control_input.control()) {
      if (c.kind() == control_kind) {
        if (seen_control) {
          return tensorflow::errors::InvalidArgument(
              "sequence batching specifies multiple ",
              ModelSequenceBatching_Control_Kind_Name(control_kind),
              " tensors for ", model_name);
        }

        *tensor_name = control_input.name();
        seen_control = true;

        if (c.int32_false_true_size() > 0) {
          if (c.fp32_false_true_size() != 0) {
            return tensorflow::errors::InvalidArgument(
                "sequence batching specifies both 'int32_false_true' and "
                "'fp32_false_true' for ",
                ModelSequenceBatching_Control_Kind_Name(control_kind), " for ",
                model_name);
          }

          if (c.int32_false_true_size() != 2) {
            return tensorflow::errors::InvalidArgument(
                "sequence batching control 'int32_false_true' must have "
                "exactly 2 entries for ",
                ModelSequenceBatching_Control_Kind_Name(control_kind), " for ",
                model_name);
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
            return tensorflow::errors::InvalidArgument(
                "sequence batching must specify either 'int32_false_true' or "
                "'fp32_false_true' for ",
                ModelSequenceBatching_Control_Kind_Name(control_kind), " for ",
                model_name);
          }

          if (c.fp32_false_true_size() != 2) {
            return tensorflow::errors::InvalidArgument(
                "sequence batching control 'fp32_false_true' must have exactly "
                "2 entries for ",
                ModelSequenceBatching_Control_Kind_Name(control_kind), " for ",
                model_name);
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
      return tensorflow::errors::InvalidArgument(
          "sequence batching control tensor must specify a ",
          ModelSequenceBatching_Control_Kind_Name(control_kind), " value for ",
          model_name);
    }

    tensor_name->clear();
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
GetNormalizedModelConfig(
    const tensorflow::StringPiece& path,
    const tfs::PlatformConfigMap& platform_config_map, const bool autofill,
    ModelConfig* config)
{
  // If 'autofill' then the configuration file can be empty.
  const auto config_path = tensorflow::io::JoinPath(path, kModelConfigPbTxt);
  if (autofill && !tensorflow::Env::Default()->FileExists(config_path).ok()) {
    config->Clear();
  } else {
    TF_RETURN_IF_ERROR(
        ReadTextProto(tensorflow::Env::Default(), config_path, config));
  }

  // Autofill if requested...
  if (autofill) {
    const std::string model_name(tensorflow::io::Basename(path));
    std::unique_ptr<AutoFill> af;
    TF_RETURN_IF_ERROR(AutoFill::Create(
        model_name, platform_config_map, std::string(path), *config, &af));
    TF_RETURN_IF_ERROR(af->Fix(config));

    LOG_VERBOSE(1) << "autofilled config: " << config->DebugString();
  }

  if (config->platform().empty()) {
    return tensorflow::errors::InvalidArgument(
        "must specify platform for model '", config->name(), "'");
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
      return tensorflow::errors::Internal(
          "unexpected platform type ", config->platform(), " for ",
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
      return tensorflow::errors::Internal(
          "unable to get number of CUDA devices for ", config->name(), ": ",
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

  return tensorflow::Status::OK();
}

tensorflow::Status
ValidateModelConfig(
    const ModelConfig& config, const std::string& expected_platform)
{
  if (config.name().empty()) {
    return tensorflow::errors::InvalidArgument(
        "model configuration must specify 'name'");
  }

  if (config.platform().empty()) {
    return tensorflow::errors::InvalidArgument(
        "must specify 'platform' for ", config.name());
  }

  if (!expected_platform.empty() && (config.platform() != expected_platform)) {
    return tensorflow::errors::NotFound(
        "expected model of type ", expected_platform, " for ", config.name());
  }

  if (!config.has_version_policy()) {
    return tensorflow::errors::InvalidArgument(
        "must specify 'version policy' for ", config.name());
  }

  // If dynamic batching is specified make sure the preferred batch
  // sizes are positive and don't exceed maximum batch size. Make sure
  // the max delay is non-negative.
  if (config.has_dynamic_batching()) {
    for (const auto size : config.dynamic_batching().preferred_batch_size()) {
      if (size <= 0) {
        return tensorflow::errors::InvalidArgument(
            "dynamic batching preferred size must be positive for ",
            config.name());
      }
      if (size > config.max_batch_size()) {
        return tensorflow::errors::InvalidArgument(
            "dynamic batching preferred size must be <= max batch size for ",
            config.name());
      }
    }
  }

  // If sequence batching is specified make sure the control is
  // specified correctly.
  if (config.has_sequence_batching()) {
    const auto& batcher = config.sequence_batching();

    if (batcher.control_input_size() == 0) {
      return tensorflow::errors::InvalidArgument(
          "sequence batching must specify at least one control tensor for ",
          config.name());
    }

    // Make sure at most one SEQUENCE_START and one SEQUENCE_READY
    // control is specified.
    std::string tensor_name;
    TF_RETURN_IF_ERROR(GetSequenceControlProperties(
        batcher, config.name(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_START,
        true /* required */, &tensor_name, nullptr, nullptr, nullptr, nullptr,
        nullptr));
    TF_RETURN_IF_ERROR(GetSequenceControlProperties(
        batcher, config.name(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_READY,
        true /* required */, &tensor_name, nullptr, nullptr, nullptr, nullptr,
        nullptr));
  }

  // If ensemble scheduling is specified make sure the other fields are
  // specified correctly.
  if (config.has_ensemble_scheduling()) {
    if (config.platform() != kEnsemblePlatform) {
      return tensorflow::errors::InvalidArgument(
          "ensemble scheduling can not be set for model ", config.name(),
          " whose platform is not ", kEnsemblePlatform);
    }
    if (config.instance_group().size() != 0) {
      return tensorflow::errors::InvalidArgument(
          "instance group should not be specified for ensemble ",
          config.name());
    }
    if (config.has_optimization()) {
      return tensorflow::errors::InvalidArgument(
          "optimization should not be specified for ensemble ", config.name());
    }
  } 
  // if not specifed, then must validate platform and instance_group
  else {
    if (config.platform() == kEnsemblePlatform) {
      return tensorflow::errors::InvalidArgument(
          "ensemble scheduling must be set for ensemble ", config.name(),
          " whose platform is ", kEnsemblePlatform);
    }

    if (config.instance_group().size() == 0) {
      return tensorflow::errors::InvalidArgument(
          "must specify one or more 'instance group's for ", config.name());
    }

    // Make sure KIND_GPU instance group specifies at least one GPU and
    // doesn't specify a non-existent GPU. Make sure non-KIND_GPU does
    // not specify any GPUs.
    int dcnt;
    cudaError_t cuerr = cudaGetDeviceCount(&dcnt);
    if (cuerr == cudaErrorNoDevice) {
      dcnt = 0;
    } else if (cuerr != cudaSuccess) {
      return tensorflow::errors::Internal(
          "failed to get device count for validation of model ", config.name(),
          ": ", cudaGetErrorString(cuerr));
    }

    for (const auto& group : config.instance_group()) {
      if (group.kind() == ModelInstanceGroup::KIND_GPU) {
        if (group.gpus().size() == 0) {
          return tensorflow::errors::InvalidArgument(
              "instance group ", group.name(), " of model ", config.name(),
              " has kind KIND_GPU but specifies no GPUs");
        }

        for (const int32_t gid : group.gpus()) {
          if ((gid < 0) || (gid >= dcnt)) {
            return tensorflow::errors::InvalidArgument(
                "instance group ", group.name(), " of model ", config.name(),
                " specifies invalid GPU id ", gid, ", valid GPUs are 0 - ",
                (dcnt - 1));
          }
        }
      } else if (group.kind() == ModelInstanceGroup::KIND_CPU) {
        if (group.gpus().size() > 0) {
          return tensorflow::errors::InvalidArgument(
              "instance group ", group.name(), " of model ", config.name(),
              " has kind KIND_CPU but specifies one or more GPUs");
        }
      } else {
        return tensorflow::errors::Internal(
            "instance group ", group.name(), " of model ", config.name(),
            " has unexpected kind KIND_AUTO");
      }
    }
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
ValidateEnsembleConfig(
    const std::unordered_map<std::string, ModelConfig>& config_map)
{
  std::unordered_map<std::string, std::string> invalid_model_names;
  std::unordered_map<std::string, bool> ensembles;

  for (const auto& pair : config_map) {
    tensorflow::Status status;
    for (const auto& input : pair.second.input()) {
      status = ValidateModelInput(input);
      if (!status.ok()) {
        break;
      }
    }
    if (status.ok()) {
      for (const auto& output : pair.second.output()) {
        status = ValidateModelOutput(output);
        if (!status.ok()) {
          break;
        }
      }
    }
    if (!status.ok()) {
      // Return error if the inputs / outputs of one ensemble is not correct.
      if (pair.second.has_ensemble_scheduling()) {
        return tensorflow::errors::InvalidArgument(
            "ensemble", pair.first, ": ", status.error_message());
      }
      invalid_model_names.emplace(pair.first, status.error_message());
    } else if (pair.second.has_ensemble_scheduling()) {
      ensembles.emplace(std::make_pair(pair.first, false));
    }
  }

  std::deque<std::string> ensemble_dependency;
  for (const auto& pair : ensembles) {
    if (pair.second) {
      continue;
    }
    TF_RETURN_IF_ERROR(ValidateEnsembleConfig(
        pair.first, config_map, invalid_model_names, ensembles,
        ensemble_dependency));
  }
  return tensorflow::Status::OK();
}

tensorflow::Status
ValidateModelInput(const ModelInput& io)
{
  std::set<std::string> allowed;
  return ValidateModelInput(io, allowed);
}

tensorflow::Status
ValidateModelInput(const ModelInput& io, const std::set<std::string>& allowed)
{
  if (io.name().empty()) {
    return tensorflow::errors::InvalidArgument(
        "model input must specify 'name'");
  }

  if (io.data_type() == DataType::TYPE_INVALID) {
    return tensorflow::errors::InvalidArgument(
        "model input must specify 'data_type'");
  }

  if (io.dims_size() == 0) {
    return tensorflow::errors::InvalidArgument(
        "model input must specify 'dims'");
  }

  for (auto dim : io.dims()) {
    if ((dim < 1) && (dim != WILDCARD_DIM)) {
      return tensorflow::errors::InvalidArgument(
          "model input dimension must be integer >= 1, or ",
          std::to_string(WILDCARD_DIM),
          " to indicate a variable-size dimension");
    }
  }

  if (((io.format() == ModelInput::FORMAT_NHWC) ||
       (io.format() == ModelInput::FORMAT_NCHW)) &&
      (io.dims_size() != 3)) {
    return tensorflow::errors::InvalidArgument(
        "model input NHWC/NCHW require 3 dims");
  }

  if (!allowed.empty() && (allowed.find(io.name()) == allowed.end())) {
    std::string astr;
    for (const auto& a : allowed) {
      if (!astr.empty()) {
        astr.append(", ");
      }
      astr.append(a);
    }

    return tensorflow::errors::InvalidArgument(
        "unexpected inference input '", io.name(),
        "', allowed inputs are: ", astr);
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
ValidateModelOutput(const ModelOutput& io)
{
  std::set<std::string> allowed;
  return ValidateModelOutput(io, allowed);
}

tensorflow::Status
ValidateModelOutput(const ModelOutput& io, const std::set<std::string>& allowed)
{
  if (io.name().empty()) {
    return tensorflow::errors::InvalidArgument(
        "model output must specify 'name'");
  }

  if (io.data_type() == DataType::TYPE_INVALID) {
    return tensorflow::errors::InvalidArgument(
        "model output must specify 'data_type'");
  }

  if (io.dims_size() == 0) {
    return tensorflow::errors::InvalidArgument(
        "model output must specify 'dims'");
  }

  for (auto dim : io.dims()) {
    if ((dim < 1) && (dim != WILDCARD_DIM)) {
      return tensorflow::errors::InvalidArgument(
          "model input dimension must be integer >= 1, or ",
          std::to_string(WILDCARD_DIM),
          " to indicate a variable-size dimension");
    }
  }

  if (!allowed.empty() && (allowed.find(io.name()) == allowed.end())) {
    std::string astr;
    for (const auto& a : allowed) {
      if (!astr.empty()) {
        astr.append(", ");
      }
      astr.append(a);
    }

    return tensorflow::errors::InvalidArgument(
        "unexpected inference output '", io.name(),
        "', allowed outputs are: ", astr);
  }

  return tensorflow::Status::OK();
}

}}  // namespace nvidia::inferenceserver
