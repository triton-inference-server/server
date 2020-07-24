// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "src/clients/c++/perf_client/model_parser.h"

nic::Error
ModelParser::Init(
    const inference::ModelMetadataResponse& metadata, const inference::ModelConfig& config,
    const std::string& model_version,
    const std::unordered_map<std::string, std::vector<int64_t>>& input_shapes,
    std::unique_ptr<TritonClientWrapper>& client_wrapper)
{
  model_name_ = metadata.name();
  model_version_ = model_version;
  // Get the scheduler type for the model
  if (config.has_ensemble_scheduling()) {
    bool is_sequential = false;
    RETURN_IF_ERROR(GetEnsembleSchedulerType(
        config, model_version, client_wrapper, &is_sequential));
    if (is_sequential) {
      scheduler_type_ = ENSEMBLE_SEQUENCE;
    } else {
      scheduler_type_ = ENSEMBLE;
    }
  } else if (config.has_sequence_batching()) {
    scheduler_type_ = SEQUENCE;
  } else if (config.has_dynamic_batching()) {
    scheduler_type_ = DYNAMIC;
  } else {
    scheduler_type_ = NONE;
  }

  max_batch_size_ = config.max_batch_size();

  is_decoupled_ = config.model_transaction_policy().decoupled();

  // Get the information about inputs from metadata
  for (const auto& input : metadata.inputs()) {
    auto it = inputs_->emplace(input.name(), ModelTensor()).first;
    it->second.name_ = input.name();
    it->second.datatype_ = input.datatype();
    bool is_dynamic = false;
    // Skip the batch size in the shape
    bool skip = (max_batch_size_ > 0);
    for (const auto dim : input.shape()) {
      if (skip) {
        skip = false;
        continue;
      }
      if (dim == -1) {
        is_dynamic = true;
      }
      it->second.shape_.push_back(dim);
    }

    if (is_dynamic) {
      const auto user_shape_it = input_shapes.find(input.name());
      if (user_shape_it != input_shapes.end()) {
        // Update the default shape to be used.
        it->second.shape_.clear();
        for (const auto dim : user_shape_it->second) {
          it->second.shape_.push_back(dim);
        }
      }
    }
  }

  // Check whether the tensor is shape tensor or not from config.
  for (const auto& input_config : config.input()) {
    const auto& itr = inputs_->find(input_config.name());
    if (itr == inputs_->end()) {
      return nic::Error(
          "no metadata found for input tensor " + input_config.name());
    }
    itr->second.is_shape_tensor_ = input_config.is_shape_tensor();
  }

  // Get the information about outputs from metadata
  for (const auto& output : metadata.outputs()) {
    auto it = outputs_->emplace(output.name(), ModelTensor()).first;
    it->second.name_ = output.name();
    it->second.datatype_ = output.datatype();
    // Skip the batch size in the shape
    bool skip = (max_batch_size_ > 0);
    for (const auto dim : output.shape()) {
      if (skip) {
        skip = false;
        continue;
      }
      it->second.shape_.push_back(dim);
    }
  }

  // Check whether the tensor is shape tensor or not from config.
  for (const auto& output_config : config.output()) {
    const auto& itr = outputs_->find(output_config.name());
    if (itr == outputs_->end()) {
      return nic::Error(
          "no metadata found for output tensor " + output_config.name());
    }
    itr->second.is_shape_tensor_ = output_config.is_shape_tensor();
  }
  return nic::Error::Success;
}


nic::Error
ModelParser::GetEnsembleSchedulerType(
    const inference::ModelConfig& config, const std::string& model_version,
    std::unique_ptr<TritonClientWrapper>& client_wrapper, bool* is_sequential)
{
  if (config.has_sequence_batching()) {
    *is_sequential = true;
  }

  if (config.platform() == "ensemble") {
    for (const auto& step : config.ensemble_scheduling().step()) {
      inference::ModelConfigResponse model_config;
      std::string step_version;
      if (step.model_version() != -1) {
        step_version = std::to_string(step.model_version());
      }
      (*composing_models_map_)[config.name()].emplace(
          step.model_name(), step_version);
      RETURN_IF_ERROR(client_wrapper->ModelConfig(
          &model_config, step.model_name(), step_version));
      RETURN_IF_ERROR(GetEnsembleSchedulerType(
          model_config.config(), step_version, client_wrapper, is_sequential));
    }
  }

  return nic::Error::Success;
}

nic::Error
ModelParser::Init(
    const rapidjson::Document& metadata, const rapidjson::Document& config,
    const std::string& model_version,
    const std::unordered_map<std::string, std::vector<int64_t>>& input_shapes,
    std::unique_ptr<TritonClientWrapper>& client_wrapper)
{
  model_name_ = metadata["name"].GetString();
  model_version_ = model_version;
  // Get the scheduler type for the model
  scheduler_type_ = NONE;
  const auto& ensemble_itr = config.FindMember("ensemble_scheduling");
  if (ensemble_itr != config.MemberEnd()) {
    bool is_sequential = false;
    RETURN_IF_ERROR(GetEnsembleSchedulerType(
        config, model_version, client_wrapper, &is_sequential));
    if (is_sequential) {
      scheduler_type_ = ENSEMBLE_SEQUENCE;
    } else {
      scheduler_type_ = ENSEMBLE;
    }
  } else {
    const auto& sequence_itr = config.FindMember("sequence_batching");
    if (sequence_itr != config.MemberEnd()) {
      scheduler_type_ = SEQUENCE;
    } else {
      const auto& dynamic_itr = config.FindMember("dynamic_batching");
      if (dynamic_itr != config.MemberEnd()) {
        scheduler_type_ = DYNAMIC;
      }
    }
  }

  max_batch_size_ = 0;
  const auto bs_itr = config.FindMember("max_batch_size");
  if (bs_itr != config.MemberEnd()) {
    max_batch_size_ = bs_itr->value.GetInt();
  }

  const auto txn_itr = config.FindMember("model_transaction_policy");
  if (txn_itr != config.MemberEnd()) {
    is_decoupled_ = txn_itr->value["decoupled"].GetBool();
  }

  // Get the information about inputs from metadata
  const auto inputs_itr = metadata.FindMember("inputs");
  if (inputs_itr != metadata.MemberEnd()) {
    for (const auto& input : inputs_itr->value.GetArray()) {
      auto it =
          inputs_->emplace(input["name"].GetString(), ModelTensor()).first;
      it->second.name_ = input["name"].GetString();
      it->second.datatype_ = input["datatype"].GetString();
      bool is_dynamic = false;
      bool skip = (max_batch_size_ > 0);
      for (const auto& dim : input["shape"].GetArray()) {
        if (skip) {
          skip = false;
          continue;
        }
        if (dim.GetInt() == -1) {
          is_dynamic = true;
        }
        it->second.shape_.push_back(dim.GetInt());
      }

      if (is_dynamic) {
        const auto user_shape_it = input_shapes.find(it->second.name_);
        if (user_shape_it != input_shapes.end()) {
          // Update the default shape to be used.
          it->second.shape_.clear();
          for (const auto dim : user_shape_it->second) {
            it->second.shape_.push_back(dim);
          }
        }
      }
    }
  }

  // Check whether the tensor is shape tensor or not from config.
  const auto inputs_config_itr = config.FindMember("input");
  if (inputs_config_itr != config.MemberEnd()) {
    for (const auto& input_config : inputs_config_itr->value.GetArray()) {
      const auto name = std::string(
          input_config["name"].GetString(),
          input_config["name"].GetStringLength());
      auto it = inputs_->find(name);
      if (it == inputs_->end()) {
        return nic::Error("no metadata found for input tensor " + name);
      }
      const auto& shape_tensor_itr = input_config.FindMember("is_shape_tensor");
      if (shape_tensor_itr != input_config.MemberEnd()) {
        it->second.is_shape_tensor_ = shape_tensor_itr->value.GetBool();
      }
    }
  }

  // Get the information about outputs from metadata
  const auto outputs_itr = metadata.FindMember("outputs");
  if (outputs_itr != metadata.MemberEnd()) {
    for (const auto& output : outputs_itr->value.GetArray()) {
      auto it =
          outputs_->emplace(output["name"].GetString(), ModelTensor()).first;
      it->second.name_ = output["name"].GetString();
      it->second.datatype_ = output["datatype"].GetString();
      bool skip = (max_batch_size_ > 0);
      for (const auto& dim : output["shape"].GetArray()) {
        if (skip) {
          skip = false;
          continue;
        }
        it->second.shape_.push_back(dim.GetInt());
      }
    }
  }

  // Check whether the tensor is shape tensor or not from config.
  const auto output_config_itr = config.FindMember("output");
  if (output_config_itr != config.MemberEnd()) {
    for (const auto& output_config : output_config_itr->value.GetArray()) {
      const auto name = std::string(
          output_config["name"].GetString(),
          output_config["name"].GetStringLength());
      auto itr = outputs_->find(name);
      if (itr == outputs_->end()) {
        return nic::Error("no metadata found for output tensor " + name);
      }
      const auto& shape_tensor_itr =
          output_config.FindMember("is_shape_tensor");
      if (shape_tensor_itr != output_config.MemberEnd()) {
        itr->second.is_shape_tensor_ = shape_tensor_itr->value.GetBool();
      }
    }
  }
  return nic::Error::Success;
}


nic::Error
ModelParser::GetEnsembleSchedulerType(
    const rapidjson::Document& config, const std::string& model_version,
    std::unique_ptr<TritonClientWrapper>& client_wrapper, bool* is_sequential)
{
  const auto& sequence_itr = config.FindMember("sequence_batching");
  if (sequence_itr != config.MemberEnd()) {
    *is_sequential = true;
  }

  if (std::string(config["platform"].GetString()).compare("ensemble") == 0) {
    const auto step_itr = config["ensemble_scheduling"].FindMember("step");
    for (const auto& step : step_itr->value.GetArray()) {
      std::string step_model_version;
      const int64_t model_version_int = step["model_version"].GetInt64();
      if (model_version_int == -1) {
        step_model_version = "";
      } else {
        step_model_version = std::to_string(model_version_int);
      }
      (*composing_models_map_)[config["name"].GetString()].emplace(
          std::string(step["model_name"].GetString()), step_model_version);

      rapidjson::Document model_config;
      RETURN_IF_ERROR(client_wrapper->ModelConfig(
          &model_config, step["model_name"].GetString(), step_model_version));
      RETURN_IF_ERROR(GetEnsembleSchedulerType(
          model_config, step_model_version, client_wrapper, is_sequential));
    }
  }

  return nic::Error::Success;
}
