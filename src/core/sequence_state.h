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
// State input/output tensor
//
class State {
 public:
  State();
  State(
      const std::string& name, const inference::DataType datatype,
      const std::vector<int64_t>& shape);
  State(
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
  Status Update()
  {
    std::cout << "State update is called." << std::endl;
    return state_update_cb_();
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(State);
  std::string name_;
  inference::DataType datatype_;
  std::vector<int64_t> shape_;
  std::shared_ptr<Memory> data_;
  std::function<Status()> state_update_cb_ = []() {
    return Status(
        Status::Code::INVALID_ARG,
        "TRITONBACKEND_StateUpdate called when sequence batching is disabled "
        "or the 'states' section of the model configuration is empty.");
  };
};

struct SequenceState {
  std::map<std::string, std::unique_ptr<State>> input_states_;
  std::map<std::string, std::unique_ptr<State>> output_states_;
};

}}  // namespace nvidia::inferenceserver
