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

#include "src/core/infer_request.h"

#include <deque>
#include "src/core/backend.h"
#include "src/core/logging.h"
#include "src/core/server.h"

namespace nvidia { namespace inferenceserver {

const std::string&
InferenceRequest::ModelName() const
{
  return backend_raw_->Name();
}

int64_t
InferenceRequest::ActualModelVersion() const
{
  return backend_raw_->Version();
}

Status
InferenceRequest::Run(std::unique_ptr<InferenceRequest>& request)
{
  return request->backend_raw_->Enqueue(request);
}

void
InferenceRequest::RespondWithError(
    std::unique_ptr<InferenceRequest>& request, const Status& status,
    const bool release_request)
{
  if (status.IsOk()) {
    return;
  }

  // Use the response factory to create a response, set the status,
  // and send it. If something goes wrong all we can do is log the
  // error.
  std::unique_ptr<InferenceResponse> response;
  LOG_STATUS_ERROR(
      request->response_factory_.CreateResponse(&response),
      "failed to create error response");
  response->SetResponseStatus(status);
  LOG_STATUS_ERROR(
      InferenceResponse::Send(std::move(response)),
      "failed to send error response");

  // If releasing the request then invoke the release callback which
  // gives ownership to the callback. So can't access 'request' after
  // this point.
  if (release_request) {
    Release(std::move(request));
  }
}

void
InferenceRequest::RespondWithError(
    std::vector<std::unique_ptr<InferenceRequest>>& requests,
    const Status& status, const bool release_requests)
{
  if (status.IsOk()) {
    return;
  }

  for (auto& request : requests) {
    RespondWithError(request, status, release_requests);
  }
}

void
InferenceRequest::Release(std::unique_ptr<InferenceRequest>&& request)
{
  void* userp = request->release_userp_;
  request->release_fn_(
      reinterpret_cast<TRITONSERVER_InferenceRequest*>(request.release()),
      userp);
}

InferenceRequest*
InferenceRequest::CopyAsNull(const InferenceRequest& from)
{
  // FIXME
  LOG_ERROR << "CopyAsNull: NYI";
  return nullptr;
}

Status
InferenceRequest::MutableOriginalInput(
    const std::string& name, InferenceRequest::Input** input)
{
  auto itr = original_inputs_.find(name);
  if (itr == original_inputs_.end()) {
    return Status(
        Status::Code::INVALID_ARG,
        "input '" + name + "' does not exist in request");
  }

  *input = &(itr->second);
  // FIXME remove and refine
  // needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::ImmutableInput(
    const std::string& name, const InferenceRequest::Input** input) const
{
  auto itr = inputs_.find(name);
  if (itr == inputs_.end()) {
    return Status(
        Status::Code::INVALID_ARG,
        "input '" + name + "' does not exist in request");
  }

  *input = itr->second;
  return Status::Success;
}

Status
InferenceRequest::MutableRequestedOutput(
    const std::string& name, RequestedOutput** output)
{
  auto itr = requested_outputs_.find(name);
  if (itr == requested_outputs_.end()) {
    return Status(
        Status::Code::INVALID_ARG,
        "output '" + name + "' does not exist in request");
  }

  *output = &(itr->second);
  needs_normalization_ = true;
  return Status::Success;
}


Status
InferenceRequest::AddOriginalInput(
    const std::string& name, const DataType datatype, const int64_t* shape,
    const uint64_t dim_count, InferenceRequest::Input** input)
{
  const auto& pr = original_inputs_.emplace(
      std::piecewise_construct, std::forward_as_tuple(name),
      std::forward_as_tuple(name, datatype, shape, dim_count));
  if (!pr.second) {
    return Status(
        Status::Code::INVALID_ARG,
        "input '" + name + "' already exists in request");
  }

  LOG_VERBOSE(1) << "add original input: " << *this;

  if (input != nullptr) {
    *input = std::addressof(pr.first->second);
  }

  needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::AddOriginalInput(
    const std::string& name, const DataType datatype,
    const std::vector<int64_t>& shape, InferenceRequest::Input** input)
{
  return AddOriginalInput(name, datatype, &shape[0], shape.size(), input);
}

Status
InferenceRequest::RemoveOriginalInput(const std::string& name)
{
  if (original_inputs_.erase(name) != 1) {
    return Status(
        Status::Code::INVALID_ARG,
        "input '" + name + "' does not exist in request");
  }

  needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::RemoveAllOriginalInputs()
{
  original_inputs_.clear();
  needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::AddOverrideInput(
    const std::string& name, const DataType datatype,
    const std::vector<int64_t>& shape,
    std::shared_ptr<InferenceRequest::Input>* input)
{
  std::shared_ptr<Input> i = std::make_shared<Input>(name, datatype, shape);
  *(i->MutableShape()) = i->OriginalShape();

  RETURN_IF_ERROR(AddOverrideInput(i));
  if (input != nullptr) {
    *input = std::move(i);
  }

  return Status::Success;
}

Status
InferenceRequest::AddOverrideInput(
    const std::shared_ptr<InferenceRequest::Input>& input)
{
  LOG_VERBOSE(1) << "adding input override for " << input->Name() << ": "
                 << *this;

  const auto& pr =
      override_inputs_.emplace(std::make_pair(input->Name(), input));
  if (!pr.second) {
    pr.first->second = input;
  }

  // Add or replace this override in the inputs...
  const auto res = inputs_.emplace(std::make_pair(input->Name(), input.get()));
  if (!res.second) {
    res.first->second = input.get();
  }

  LOG_VERBOSE(1) << "added input override for " << input->Name() << ": "
                 << *this;

  return Status::Success;
}

Status
InferenceRequest::AddRequestedOutput(
    const std::string& name, const uint32_t classification_cnt)
{
  const auto& pr = requested_outputs_.emplace(
      std::piecewise_construct, std::forward_as_tuple(name),
      std::forward_as_tuple(name, classification_cnt));

  if (!pr.second) {
    return Status(
        Status::Code::INVALID_ARG, "output '" + name + "' already requested");
  }

  needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::RemoveRequestedOutput(const std::string& name)
{
  if (requested_outputs_.erase(name) != 1) {
    return Status(
        Status::Code::INVALID_ARG,
        "output '" + name + "' does not exist in request");
  }

  needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::RemoveAllRequestedOutputs()
{
  requested_outputs_.clear();
  needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::PrepareForInference()
{
  // Remove override inputs as those are added during any previous
  // inference execution.
  inputs_.clear();
  override_inputs_.clear();

  // If anything has potentially changed in the inference request then
  // need to renormalize.
  if (needs_normalization_) {
    RETURN_IF_ERROR(Normalize());
    needs_normalization_ = false;
  }

  // Initially show the actual inputs to be only the original
  // inputs. If overrides are added later they will be added to
  // 'inputs_'.
  for (auto& pr : original_inputs_) {
    inputs_.emplace(std::make_pair(pr.first, std::addressof(pr.second)));
  }

  LOG_VERBOSE(1) << "prepared: " << *this;

  return Status::Success;
}

Status
InferenceRequest::Normalize()
{
  const ModelConfig& model_config = backend_raw_->Config();

  if ((priority_ == 0) || (priority_ > backend_raw_->MaxPriorityLevel())) {
    priority_ = backend_raw_->DefaultPriorityLevel();
  }

  // FIXMEV2 need original requested
  // If requested_outputs_ is empty return all outputs specified in model config
  if (requested_outputs_.size() == 0) {
    for (const auto& output : model_config.output()) {
      AddRequestedOutput(output.name(), 0 /* classification_count */);
    }
  } else {
    // Validate if the requested output name exists in the model configuration
    for (const auto& pr : requested_outputs_) {
      const ModelOutput* output_config;
      RETURN_IF_ERROR(backend_raw_->GetOutput(pr.first, &output_config));
    }
  }

  // Make sure that the request is providing the same number of inputs
  // as is expected by the model.
  if (original_inputs_.size() != (size_t)model_config.input_size()) {
    return Status(
        Status::Code::INVALID_ARG,
        "expected " + std::to_string(model_config.input_size()) +
            " inputs but got " + std::to_string(original_inputs_.size()) +
            " inputs for model '" + ModelName() + "'");
  }

  // Determine the batch size and shape of each input.
  if (model_config.max_batch_size() == 0) {
    // Model does not support Triton-style batching so treat as
    // batch-size 1 and leave the tensor shapes as they are.
    batch_size_ = 1;
    for (auto& pr : original_inputs_) {
      auto& input = pr.second;
      *input.MutableShape() = input.OriginalShape();
    }
  } else {
    // Model does support Triton-style batching so each input tensor
    // must have the same first dimension which is the batch
    // size. Adjust the shape of the input tensors to remove the batch
    // dimension.
    batch_size_ = 0;
    for (auto& pr : original_inputs_) {
      auto& input = pr.second;

      // Keep shape tensor's shape as it is
      const ModelInput* input_config;
      RETURN_IF_ERROR(backend.GetInput(pr.first, &input_config));
      if (input_config->is_shape_tensor()) {
        *input.MutableShape() = input.OriginalShape();
        continue;
      }

      if (input.OriginalShape().size() == 0) {
        return Status(
            Status::Code::INVALID_ARG,
            "input '" + input.Name() +
                "' has no shape but model requires batch dimension for '" +
                ModelName() + "'");
      }

      if (batch_size_ == 0) {
        batch_size_ = input.OriginalShape()[0];
      } else if (input.OriginalShape()[0] != batch_size_) {
        return Status(
            Status::Code::INVALID_ARG,
            "input '" + input.Name() +
                "' batch size does not match other inputs for '" + ModelName() +
                "'");
      }

      input.MutableShape()->assign(
          input.OriginalShape().begin() + 1, input.OriginalShape().end());
    }
  }

  // Make sure the request has a batch-size > 0. Even for models that
  // don't support batching the requested batch size must be 1.
  if (batch_size_ < 1) {
    return Status(
        Status::Code::INVALID_ARG,
        "inference request batch-size must be >= 1 for '" + ModelName() + "'");
  }

  // Make sure request batch-size doesn't exceed what is supported by
  // the model. For models that don't support batching the request
  // batch-size will still be 1.
  if ((batch_size_ != 1) &&
      ((int)batch_size_ > model_config.max_batch_size())) {
    return Status(
        Status::Code::INVALID_ARG,
        "inference request batch-size must be <= " +
            std::to_string(model_config.max_batch_size()) + " for '" +
            ModelName() + "'");
  }

  // Verify that each input shape is valid for the model, make
  // adjustments for reshapes and find the total tensor size.
  for (auto& pr : original_inputs_) {
    const ModelInput* input_config;
    RETURN_IF_ERROR(backend_raw_->GetInput(pr.first, &input_config));

    auto& input = pr.second;
    auto shape = input.MutableShape();

    if (input.DType() != input_config->data_type()) {
      return Status(
          Status::Code::INVALID_ARG,
          "inference input data-type is '" +
              std::string(DataTypeToProtocolString(input.DType())) +
              "', model expects '" +
              std::string(DataTypeToProtocolString(input_config->data_type())) +
              "' for '" + ModelName() + "'");
    }

    if (!CompareDimsWithWildcard(input_config->dims(), *shape)) {
      return Status(
          Status::Code::INVALID_ARG,
          "unexpected shape for input '" + pr.first + "' for model '" +
              ModelName() + "'. Expected " +
              DimsListToString(input_config->dims()) + ", got " +
              DimsListToString(*shape));
    }

    // If there is a reshape for this input then adjust them to
    // match the reshape. As reshape may have variable-size
    // dimensions, we need to record corresponding value so that we
    // can set the value correctly for reshape.
    if (input_config->has_reshape()) {
      std::deque<int64_t> variable_size_values;
      for (int64_t idx = 0; idx < input_config->dims_size(); idx++) {
        if (input_config->dims(idx) == -1) {
          variable_size_values.push_back((*shape)[idx]);
        }
      }

      shape->clear();
      for (const auto& dim : input_config->reshape().shape()) {
        if (dim == -1) {
          shape->push_back(variable_size_values.front());
          variable_size_values.pop_front();
        } else {
          shape->push_back(dim);
        }
      }
    }
  }

  return Status::Success;
}

//
// Input
//
InferenceRequest::Input::Input()
    : data_byte_size_(0), data_(new MemoryReference)
{
}

InferenceRequest::Input::Input(
    const std::string& name, const DataType datatype, const int64_t* shape,
    const uint64_t dim_count)
    : name_(name), datatype_(datatype),
      original_shape_(shape, shape + dim_count), data_byte_size_(0),
      data_(new MemoryReference)
{
}

InferenceRequest::Input::Input(
    const std::string& name, const DataType datatype,
    const std::vector<int64_t>& shape)
    : name_(name), datatype_(datatype), original_shape_(shape),
      data_byte_size_(0), data_(new MemoryReference)
{
}

Status
InferenceRequest::Input::AppendData(
    const void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  if (byte_size > 0) {
    std::static_pointer_cast<MemoryReference>(data_)->AddBuffer(
        static_cast<const char*>(base), byte_size, memory_type, memory_type_id);
    data_byte_size_ = data_->TotalByteSize();
  }

  return Status::Success;
}

Status
InferenceRequest::Input::SetData(const std::shared_ptr<Memory>& data)
{
  if (data_byte_size_ != 0) {
    return Status(
        Status::Code::INVALID_ARG,
        "input '" + name_ + "' already has data, can't overwrite");
  }

  data_ = data;
  data_byte_size_ = data_->TotalByteSize();

  return Status::Success;
}

Status
InferenceRequest::Input::RemoveAllData()
{
  data_ = std::make_shared<MemoryReference>();
  data_byte_size_ = 0;
  return Status::Success;
}

Status
InferenceRequest::Input::DataBuffer(
    const size_t idx, const void** base, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id) const
{
  *base = data_->BufferAt(idx, byte_size, memory_type, memory_type_id);

  return Status::Success;
}

//
// RequestedOutput
//
InferenceRequest::RequestedOutput::RequestedOutput(
    const std::string& name, const uint32_t classification_cnt)
    : name_(name), classification_cnt_(classification_cnt)
{
}

std::ostream&
operator<<(std::ostream& out, const InferenceRequest& request)
{
  out << "[0x" << std::addressof(request) << "] "
      << "request id: " << request.Id() << ", model: " << request.ModelName()
      << ", requested version: " << request.RequestedModelVersion()
      << ", actual version: " << request.ActualModelVersion() << ", flags: 0x"
      << std::hex << request.Flags() << std::dec
      << ", correlation id: " << request.CorrelationId()
      << ", batch size: " << request.BatchSize()
      << ", priority: " << request.Priority()
      << ", timeout (us): " << request.TimeoutMicroseconds() << std::endl;

  out << "original inputs:" << std::endl;
  for (const auto& itr : request.OriginalInputs()) {
    out << "[0x" << std::addressof(itr.second) << "] " << itr.second
        << std::endl;
  }

  out << "override inputs:" << std::endl;
  for (const auto& itr : request.OverrideInputs()) {
    out << "[0x" << itr.second.get() << "] " << *itr.second << std::endl;
  }

  out << "inputs:" << std::endl;
  for (const auto& itr : request.ImmutableInputs()) {
    out << "[0x" << itr.second << "] " << *itr.second << std::endl;
  }

  out << "requested outputs:" << std::endl;
  for (const auto& itr : request.RequestedOutputs()) {
    out << itr.second << std::endl;
  }

  return out;
}

std::ostream&
operator<<(std::ostream& out, const InferenceRequest::Input& input)
{
  out << "input: " << input.Name()
      << ", type: " << DataTypeToProtocolString(input.DType())
      << ", original shape: " << DimsListToString(input.OriginalShape())
      << ", shape: " << DimsListToString(input.Shape());
  return out;
}

std::ostream&
operator<<(std::ostream& out, const InferenceRequest::RequestedOutput& output)
{
  out << "requested output: " << output.Name()
      << ", class count: " << output.ClassificationCount();
  return out;
}

}}  // namespace nvidia::inferenceserver
