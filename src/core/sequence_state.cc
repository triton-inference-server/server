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

#include "src/core/sequence_state.h"

#include "memory.h"
#include "src/core/logging.h"

namespace nvidia { namespace inferenceserver {

SequenceState::SequenceState() : data_(new MemoryReference) {}

SequenceState::SequenceState(
    const std::string& name, const inference::DataType datatype,
    const int64_t* shape, const uint64_t dim_count)
    : name_(name), datatype_(datatype), shape_(shape, shape + dim_count),
      data_(new MemoryReference)
{
}

SequenceState::SequenceState(
    const std::string& name, const inference::DataType datatype,
    const std::vector<int64_t>& shape)
    : name_(name), datatype_(datatype), shape_(shape),
      data_(new MemoryReference)
{
}

Status
SequenceState::SetData(const std::shared_ptr<Memory>& data)
{
  if (data_->TotalByteSize() != 0) {
    return Status(
        Status::Code::INVALID_ARG,
        "state '" + name_ + "' already has data, can't overwrite");
  }

  data_ = data;
  return Status::Success;
}

Status
SequenceState::RemoveAllData()
{
  data_ = std::make_shared<MemoryReference>();
  return Status::Success;
}

Status
SequenceStates::Initialize(
    const std::unordered_map<
        std::string, const inference::ModelSequenceBatching_State&>&
        state_output_config_map)
{
  input_states_.clear();
  output_states_.clear();

  for (auto& state : state_output_config_map) {
    auto& state_config = state.second;

    // Convert the variable dimensions to 1 for the first request.
    std::vector<int64_t> dims;
    for (auto& dim : state_config.dims()) {
      if (dim == -1) {
        dims.push_back(1);
      } else {
        dims.push_back(dim);
      }
    }

    const size_t state_size = GetByteSize(state.second.data_type(), dims);
    auto data = std::make_shared<AllocatedMemory>(
        state_size, TRITONSERVER_MEMORY_CPU, 0);

    const auto& input_pair = input_states_.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(state_config.input_name()),
        std::forward_as_tuple(new SequenceState(
            state_config.input_name(), state.second.data_type(), dims)));

    if (!input_pair.second) {
      LOG_WARNING
          << "Detected duplicate 'input_name' in the state configuration: '"
          << state_config.input_name()
          << ".' This state configuration will be ignored.";
      continue;
    }
    RETURN_IF_ERROR(input_pair.first->second->SetData(data));

    const auto& output_pair = output_states_.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(state_config.output_name()),
        std::forward_as_tuple());
    if (!output_pair.second) {
      // Remove the corresponding state from the input_states_map
      input_states_.erase(state_config.input_name());
      LOG_WARNING << "Detected duplicate 'output_name' in the state "
                  "configuration: '"
               << state_config.output_name()
               << ".' This state configuration will be ignored.";

      continue;
    }
  }

  return Status::Success;
}

Status
SequenceStates::GetOutputState(
    const std::string& name, const inference::DataType datatype,
    const int64_t* shape, const uint64_t dim_count,
    SequenceState** output_state)
{
  const auto& output_state_itr = output_states_.find(name);

  // If the state name is not valid return an error.
  if (output_state_itr == output_states_.end()) {
    return Status(
        Status::Code::INVALID_ARG,
        "state '" + name + "' is not a valid state name.");
  }

  if (output_states_[name] == nullptr) {
    output_states_[name] = std::unique_ptr<SequenceState>(
        new SequenceState(name, datatype, shape, dim_count));
  } else {
    // A new SequenceState is created here in case the shape for the new output
    // state is different from the shape of the originally stored state.
    std::unique_ptr<SequenceState> output_state(
        new SequenceState(name, datatype, shape, dim_count));

    // Transfer the previously allocated buffer to the new output_state.
    output_state->SetData(output_states_[name]->Data());
    output_states_[name] = std::move(output_state);
  }

  auto& output_state_r = output_states_[name];
  size_t iter_advance =
      std::distance(output_states_.begin(), output_states_.find(name));

  // Find the input state corresponding to this output state.
  auto input_states_itr = input_states_.begin();
  std::advance(input_states_itr, iter_advance);
  auto& input_state_r = input_states_[input_states_itr->first];

  if (output_state != nullptr) {
    *output_state = output_states_[name].get();
  }

  output_state_r->SetStateUpdateCallback([&output_state_r, &input_state_r]() {
    // Swap the internal memory if the size of the input and output state is
    // equal
    if (output_state_r->Data()->TotalByteSize() ==
        input_state_r->Data()->TotalByteSize()) {
      std::shared_ptr<Memory> temp_memory = input_state_r->Data();
      RETURN_IF_ERROR(input_state_r->RemoveAllData());
      RETURN_IF_ERROR(input_state_r->SetData(output_state_r->Data()));
      RETURN_IF_ERROR(output_state_r->RemoveAllData());
      RETURN_IF_ERROR(output_state_r->SetData(temp_memory));
    } else {
      // If the size of output state is different from the input state, allocate
      // a new memory for the input state with the same size as output state.
      TRITONSERVER_MemoryType memory_type;
      int64_t memory_type_id;

      const std::shared_ptr<AllocatedMemory>& input_memory =
          reinterpret_cast<const std::shared_ptr<AllocatedMemory>&>(
              input_state_r->Data());

      input_memory->MutableBuffer(&memory_type, &memory_type_id);
      std::shared_ptr<AllocatedMemory> memory =
          std::make_shared<AllocatedMemory>(
              output_state_r->Data()->TotalByteSize(), memory_type,
              memory_type_id);
      RETURN_IF_ERROR(input_state_r->RemoveAllData());
      RETURN_IF_ERROR(input_state_r->SetData(output_state_r->Data()));
      RETURN_IF_ERROR(output_state_r->RemoveAllData());
      RETURN_IF_ERROR(output_state_r->SetData(memory));
    }

    // Update the shape and data type of the output state if it doesn't match
    // the input state.
    if (input_state_r->Shape() != output_state_r->Shape()) {
      *input_state_r->MutableShape() = output_state_r->Shape();
    }

    if (input_state_r->DType() != output_state_r->DType()) {
      *input_state_r->MutableDType() = output_state_r->DType();
    }

    return Status::Success;
  });

  return Status::Success;
}

Status
SequenceStates::GetOutputState(
    const std::string& name, const inference::DataType datatype,
    const std::vector<int64_t>& shape, SequenceState** output_state)
{
  return GetOutputState(
      name, datatype, shape.data(), shape.size(), output_state);
}

}}  // namespace nvidia::inferenceserver
