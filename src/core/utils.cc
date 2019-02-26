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

#include "absl/strings/numbers.h"
#include "cuda/include/cuda_runtime_api.h"
#include "src/core/autofill.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"

namespace nvidia { namespace inferenceserver {

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
    const tensorflow::StringPiece& path, const bool autofill,
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
    TF_RETURN_IF_ERROR(
        AutoFill::Create(model_name, std::string(path), *config, &af));
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

  if (config.instance_group().size() == 0) {
    return tensorflow::errors::InvalidArgument(
        "must specify one or more 'instance group's for ", config.name());
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
