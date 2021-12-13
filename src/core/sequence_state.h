// Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <functional>
#include <map>
#include <memory>
#include "memory.h"
#include "model_config.h"
#include "status.h"

#pragma once

namespace nvidia { namespace inferenceserver {

//
// Sequence state tensors.
//
class SequenceState {
 public:
  SequenceState();
  SequenceState(
      const std::string& name, const inference::DataType datatype,
      const std::vector<int64_t>& shape);
  SequenceState(
      const std::string& name, const inference::DataType datatype,
      const int64_t* shape, const uint64_t dim_count);

  // The name of the state tensor.
  const std::string& Name() const { return name_; }

  // Data type of the state tensor.
  inference::DataType DType() const { return datatype_; }

  // Mutable data type of the state tensor.
  inference::DataType* MutableDType() { return &datatype_; }

  // The shape of the state tensor after normalization.
  const std::vector<int64_t>& Shape() const { return shape_; }
  std::vector<int64_t>* MutableShape() { return &shape_; }

  // The data for this shape.
  std::shared_ptr<Memory>& Data() { return data_; }

  // Set the data for this shape. Error if state already has some
  // data.
  Status SetData(const std::shared_ptr<Memory>& data);

  // Remove all existing data for the state.
  Status RemoveAllData();

  // Set the state update callback.
  void SetStateUpdateCallback(std::function<Status()>&& state_update_cb)
  {
    state_update_cb_ = std::move(state_update_cb);
  }

  // Call the state update callback. This function will be called when
  // TRITONBACKEND_StateUpdate is called.
  Status Update() { return state_update_cb_(); }

 private:
  DISALLOW_COPY_AND_ASSIGN(SequenceState);
  std::string name_;
  inference::DataType datatype_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> batch_dim_;
  std::shared_ptr<Memory> data_;
  std::function<Status()> state_update_cb_ = []() {
    // By default calling the TRITONBACKEND_StateUpdate will return an error.
    return Status(
        Status::Code::INVALID_ARG,
        "TRITONBACKEND_StateUpdate called when sequence batching is disabled "
        "or the 'states' section of the model configuration is empty.");
  };
};

class SequenceStates {
 public:
  struct InitialStateData {
    InitialStateData(const std::string& state_init_name)
        : state_init_name_(state_init_name)
    {
    }

    std::string state_init_name_;
    std::shared_ptr<MutableMemory> data_;
  };

  // Initialize the state tensors according to the state model configuration.
  // Will use a default value of 1 for the variable dimensions in the state
  // tensor configuration.
  Status Initialize(
      const std::unordered_map<
          std::string, const inference::ModelSequenceBatching_State&>&
          state_output_config_map,
      const size_t max_batch_size,
      const std::unordered_map<std::string, InitialStateData>& initial_state);

  // Get a buffer holding the output state.
  Status OutputState(
      const std::string& name, const inference::DataType datatype,
      const int64_t* shape, const uint64_t dim_count,
      SequenceState** output_state);
  Status OutputState(
      const std::string& name, const inference::DataType datatype,
      const std::vector<int64_t>& shape, SequenceState** output_state);

  // Create a copy of the 'from' sequence states for NULL requests.
  static std::shared_ptr<SequenceStates> CopyAsNull(
      const std::shared_ptr<SequenceStates>& from);

  const std::map<std::string, std::unique_ptr<SequenceState>>& InputStates()
  {
    return input_states_;
  }

  std::map<std::string, std::unique_ptr<SequenceState>>& OutputStates()
  {
    return output_states_;
  }

  void SetNullSequenceStates(std::shared_ptr<SequenceStates> sequence_states)
  {
    null_sequence_states_ = sequence_states;
    is_null_request_ = true;
  }

  const std::shared_ptr<SequenceStates>& NullSequenceStates()
  {
    return null_sequence_states_;
  }

  bool IsNullRequest() { return is_null_request_; }

 private:
  std::map<std::string, std::unique_ptr<SequenceState>> input_states_;
  std::map<std::string, std::unique_ptr<SequenceState>> output_states_;
  std::shared_ptr<SequenceStates> null_sequence_states_;
  bool is_null_request_ = false;
};

}}  // namespace nvidia::inferenceserver
