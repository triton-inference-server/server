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
#pragma once

#include "src/clients/c++/perf_client/perf_utils.h"
#include "src/clients/c++/perf_client/triton_client_wrapper.h"

struct ModelTensor {
  ModelTensor() : is_shape_tensor_(false) {}
  std::string name_;
  std::string datatype_;
  std::vector<int64_t> shape_;
  bool is_shape_tensor_;
};

using ModelTensorMap = std::map<std::string, ModelTensor>;
using ComposingModelMap = std::map<std::string, std::set<ModelIdentifier>>;

//==============================================================================
/// ModelParser is a helper class to parse the information about the target
/// model from the metadata and configuration returned by the server.
///
/// Perf Client depends upon the various properties of the model to correctly
/// generate and issue inference request for the model. The object of this
/// class will provide these necessary details.
class ModelParser {
 public:
  enum ModelSchedulerType {
    NONE,
    DYNAMIC,
    SEQUENCE,
    ENSEMBLE,
    ENSEMBLE_SEQUENCE
  };

  ModelParser()
      : inputs_(std::make_shared<ModelTensorMap>()),
        outputs_(std::make_shared<ModelTensorMap>()),
        composing_models_map_(std::make_shared<ComposingModelMap>()),
        scheduler_type_(NONE), max_batch_size_(0), is_decoupled_(false)
  {
  }

  /// Initializes the ModelParser with the metadata and config messages
  /// for the target model
  /// \param metadata The metadata of the target model.
  /// \param config The config of the target model.
  /// \param model_version The version of target model.
  /// \param input_shapes The user provided default shapes which will be use
  /// if a certain input has wildcard in its dimension.
  /// \param client_wrapper The wrapped triton client object.
  /// \return Error object indicating success or failure.
  nic::Error Init(
      const inference::ModelMetadataResponse& metadata, const inference::ModelConfig& config,
      const std::string& model_version,
      const std::unordered_map<std::string, std::vector<int64_t>>& input_shapes,
      std::unique_ptr<TritonClientWrapper>& client_wrapper);

  /// Initializes the ModelParser with the metadata and config rapidjson DOM
  /// for the target model
  /// \param metadata The metadata of the target model.
  /// \param config The config of the target model.
  /// \param model_version The version of target model.
  /// \param input_shapes The user provided default shapes which will be use
  /// if a certain input has wildcard in its dimension.
  /// \param client_wrapper The wrapped triton client object.
  /// \return Error object indicating success or failure.
  nic::Error Init(
      const rapidjson::Document& metadata, const rapidjson::Document& config,
      const std::string& model_version,
      const std::unordered_map<std::string, std::vector<int64_t>>& input_shapes,
      std::unique_ptr<TritonClientWrapper>& client_wrapper);

  /// Get the name of the target model
  /// \return Model name as string
  const std::string& ModelName() const { return model_name_; }

  /// Get the version of target model
  /// \return Model version as string
  const std::string& ModelVersion() const { return model_version_; }

  /// Get the scheduler type for the model
  ModelSchedulerType SchedulerType() const { return scheduler_type_; }

  /// Get the max batch size supported by the model. Returns 0 if the model
  /// does not support batching.
  /// \return The maximum supported batch size.
  size_t MaxBatchSize() const { return max_batch_size_; }

  /// Returns whether or not the model is decoupled
  /// \return the truth value of whether the model is decoupled
  bool IsDecoupled() const { return is_decoupled_; }

  /// Get the details about the model inputs.
  /// \return The map with tensor_name and the tensor details
  /// stored as key-value pair.
  const std::shared_ptr<ModelTensorMap>& Inputs() { return inputs_; }

  /// Get the details about the model outputs.
  /// \return The map with tensor_name and the tensor details
  /// stored as key-value pair.
  const std::shared_ptr<ModelTensorMap>& Outputs() { return outputs_; }

  /// Get the composing maps for the target model.
  /// \return The pointer to the nested map descriping the
  /// nested flow in the target model.
  const std::shared_ptr<ComposingModelMap>& GetComposingModelMap()
  {
    return composing_models_map_;
  }


 private:
  nic::Error GetEnsembleSchedulerType(
      const inference::ModelConfig& config, const std::string& model_version,
      std::unique_ptr<TritonClientWrapper>& client_wrapper,
      bool* is_sequential);

  nic::Error GetEnsembleSchedulerType(
      const rapidjson::Document& config, const std::string& model_version,
      std::unique_ptr<TritonClientWrapper>& client_wrapper,
      bool* is_sequential);


  std::shared_ptr<ModelTensorMap> inputs_;
  std::shared_ptr<ModelTensorMap> outputs_;
  std::shared_ptr<ComposingModelMap> composing_models_map_;

  std::string model_name_;
  std::string model_version_;
  ModelSchedulerType scheduler_type_;
  size_t max_batch_size_;
  bool is_decoupled_;
};
