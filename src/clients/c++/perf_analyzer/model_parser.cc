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

#include "src/clients/c++/perf_analyzer/model_parser.h"

#include "rapidjson/writer.h"

namespace perfanalyzer {

namespace {
// In the json produced by protobuf, int64 and uint64 values are
// represented as strings. Protobuf doesn't provide an option to
// disable this (sigh) so we need to correctly parse these fields
// for ModelParser to receive appopriate requests.
cb::Error
GetInt(const rapidjson::Value& value, int64_t* integer_value)
{
  if (value.IsString()) {
    std::string str(value.GetString(), value.GetStringLength());

    try {
      *integer_value = std::atoll(str.c_str());
    }
    catch (...) {
      return cb::Error(
          std::string("unable to convert '") + str + "' to integer");
    }

  } else if (value.IsInt64()) {
    *integer_value = value.GetInt64();
  } else if (value.IsInt()) {
    *integer_value = value.GetInt();
  } else {
    return cb::Error("failed to parse the integer value");
  }

  return cb::Error::Success;
}

}  // namespace

cb::Error
ModelParser::InitTriton(
    const rapidjson::Document& metadata, const rapidjson::Document& config,
    const std::string& model_version,
    const std::unordered_map<std::string, std::vector<int64_t>>& input_shapes,
    std::unique_ptr<cb::ClientBackend>& backend)
{
  model_name_ = metadata["name"].GetString();
  model_version_ = model_version;
  // Get the scheduler type for the model
  scheduler_type_ = NONE;
  const auto& ensemble_itr = config.FindMember("ensemble_scheduling");
  if (ensemble_itr != config.MemberEnd()) {
    bool is_sequential = false;
    RETURN_IF_ERROR(GetEnsembleSchedulerType(
        config, model_version, backend, &is_sequential));
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
    int64_t mbs;
    RETURN_IF_ERROR(GetInt(bs_itr->value, &mbs));
    max_batch_size_ = mbs;
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
        int64_t dim_int;
        RETURN_IF_ERROR(GetInt(dim, &dim_int));
        if (dim_int == -1) {
          is_dynamic = true;
        }
        it->second.shape_.push_back(dim_int);
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
        return cb::Error("no metadata found for input tensor " + name);
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
        int64_t dim_int;
        RETURN_IF_ERROR(GetInt(dim, &dim_int));
        it->second.shape_.push_back(dim_int);
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
        return cb::Error("no metadata found for output tensor " + name);
      }
      const auto& shape_tensor_itr =
          output_config.FindMember("is_shape_tensor");
      if (shape_tensor_itr != output_config.MemberEnd()) {
        itr->second.is_shape_tensor_ = shape_tensor_itr->value.GetBool();
      }
    }
  }
  return cb::Error::Success;
}

cb::Error
ModelParser::InitTFServe(
    const rapidjson::Document& metadata, const std::string& model_name,
    const std::string& model_version, const std::string& model_signature_name,
    const int32_t batch_size,
    const std::unordered_map<std::string, std::vector<int64_t>>& input_shapes,
    std::unique_ptr<cb::ClientBackend>& backend)
{
  model_name_ = model_name;
  model_version_ = model_version;
  model_signature_name_ = model_signature_name;
  // Get the scheduler type for the model
  scheduler_type_ = NONE;

  // Will use the user provided batch size as max. Relies on the service
  // to throw an error if not supported.
  max_batch_size_ = batch_size;

  const rapidjson::Value& signature_config =
      metadata["metadata"]["signature_def"]["signature_def"];
  if (!signature_config.HasMember(model_signature_name.c_str())) {
    return cb::Error(
        "Failed to find signature_name \"" + model_signature_name +
        "\" in the metadata");
  }

  // Get the information about inputs from metadata
  if (signature_config[model_signature_name.c_str()].HasMember("inputs")) {
    const rapidjson::Value& inputs =
        signature_config[model_signature_name.c_str()]["inputs"];
    for (rapidjson::Value::ConstMemberIterator json_itr = inputs.MemberBegin();
         json_itr != inputs.MemberEnd(); ++json_itr) {
      auto it =
          inputs_->emplace(json_itr->name.GetString(), ModelTensor()).first;
      it->second.name_ = json_itr->name.GetString();
      RETURN_IF_ERROR(ConvertDTypeFromTFS(
          json_itr->value["dtype"].GetString(), &it->second.datatype_));

      bool is_dynamic = false;
      if (json_itr->value["tensor_shape"]["unknown_rank"].GetBool()) {
        if (max_batch_size_ != 0) {
          return cb::Error(
              "Can not specify -b flag for saved model with unknown ranked "
              "inputs");
        }
        is_dynamic = true;
      } else {
        bool first_dim = true;
        for (const auto& dim :
             json_itr->value["tensor_shape"]["dim"].GetArray()) {
          int64_t dim_int;
          RETURN_IF_ERROR(GetInt(dim["size"], &dim_int));
          if (first_dim && (max_batch_size_ != 0)) {
            if (dim_int != -1) {
              return cb::Error(
                  "Can not specify -b flag for saved model with input not "
                  "having their first dim as -1");
            }
            first_dim = false;
          } else {
            if (dim_int == -1) {
              is_dynamic = true;
            }
            it->second.shape_.push_back(dim_int);
          }
        }
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

  // Will not extract the information about the information about the outputs.
  // As by default, the TensorFlow serving will return all the output tensors
  // if none are requested.
  // See here
  // https://github.com/tensorflow/serving/blob/2.3.0/tensorflow_serving/apis/predict.proto#L27

  return cb::Error::Success;
}

cb::Error
ModelParser::InitTorchServe(
    const std::string& model_name, const std::string& model_version,
    const int32_t batch_size)
{
  // TorchServe does not return model metadata hence we can not obtain any
  // parameters.
  model_name_ = model_name;
  model_version_ = model_version;
  max_batch_size_ = batch_size;

  // TorchServe needs to upload a file to the server. The input will hold the
  // path to the file which should be provided as json to --input-data
  auto it = inputs_->emplace("TORCHSERVE_INPUT", ModelTensor()).first;
  it->second.name_ = "TORCHSERVE_INPUT";
  it->second.datatype_ = "BYTES";
  // Supports only a single input file
  it->second.shape_.push_back(1);

  return cb::Error::Success;
}

cb::Error
ModelParser::GetEnsembleSchedulerType(
    const rapidjson::Document& config, const std::string& model_version,
    std::unique_ptr<cb::ClientBackend>& backend, bool* is_sequential)
{
  const auto& sequence_itr = config.FindMember("sequence_batching");
  if (sequence_itr != config.MemberEnd()) {
    *is_sequential = true;
  }

  if (std::string(config["platform"].GetString()).compare("ensemble") == 0) {
    const auto step_itr = config["ensemble_scheduling"].FindMember("step");
    for (const auto& step : step_itr->value.GetArray()) {
      std::string step_model_version;
      int64_t model_version_int;
      RETURN_IF_ERROR(GetInt(step["model_version"], &model_version_int));
      if (model_version_int == -1) {
        step_model_version = "";
      } else {
        step_model_version = std::to_string(model_version_int);
      }
      (*composing_models_map_)[config["name"].GetString()].emplace(
          std::string(step["model_name"].GetString()), step_model_version);

      rapidjson::Document model_config;
      RETURN_IF_ERROR(backend->ModelConfig(
          &model_config, step["model_name"].GetString(), step_model_version));
      RETURN_IF_ERROR(GetEnsembleSchedulerType(
          model_config, step_model_version, backend, is_sequential));
    }
  }

  return cb::Error::Success;
}

}  // namespace perfanalyzer
