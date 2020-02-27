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
#include "src/core/server.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

InferenceRequest::InferenceRequest() : InferenceRequest(std::string(), -1, 1) {}
InferenceRequest::InferenceRequest(
    const std::string& model_name, const int64_t model_version,
    const uint32_t protocol_version)
    : model_name_(model_name), requested_model_version_(model_version),
      protocol_version_(protocol_version)
{
}

Status
InferenceRequest::AddInput(
    const std::string& name, const DimsList& shape,
    const uint64_t batch_byte_size, InferenceRequest::Input** input)
{
  std::vector<int64_t> lshape;
  for (const auto d : shape) {
    lshape.push_back(d);
  }

  const auto& pr = inputs_.emplace(std::make_pair(
      name, InferenceRequest::Input(name, lshape, batch_byte_size)));
  if (!pr.second) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "input '" + name + "' already exists in request");
  }

  if (input != nullptr) {
    *input = std::addressof(pr.first->second);
  }

  return Status::Success;
}

Status
InferenceRequest::AddInput(
    const std::string& name, const std::vector<int64_t>& shape,
    const uint64_t batch_byte_size, InferenceRequest::Input** input)
{
  const auto& pr = inputs_.emplace(std::make_pair(
      name, InferenceRequest::Input(name, shape, batch_byte_size)));
  if (!pr.second) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "input '" + name + "' already exists in request");
  }

  if (input != nullptr) {
    *input = std::addressof(pr.first->second);
  }

  return Status::Success;
}

Status
InferenceRequest::RequestOutput(
    const std::string& name, const uint32_t classification_cnt)
{
  const auto& pr = requested_outputs_.emplace(std::make_pair(
      name, InferenceRequest::RequestedOutput(name, classification_cnt)));
  if (!pr.second) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "output '" + name + "' already requested");
  }

  return Status::Success;
}

Status
InferenceRequest::Normalize(const InferenceBackend& backend)
{
  // FIXMEV2 these checks and normalization should move to when the
  // shapes are set in the request (so that we don't need to run this
  // code every time, only when tensor added. Also, should not
  // overwrite the registered shape but instead create reshaped shape
  // as separate member.

  const ModelConfig& model_config = backend.Config();

  // FIXMEV2 For V2 protocol we must adjust the shape of the input
  // tensors to remove the batch dimension and instead report that as
  // batch-size.
  if (protocol_version_ == 2) {
    if (model_config.max_batch_size() == 0) {
      batch_size_ = 1;
    } else {
      // All inputs should have same size in first dimension (the
      // batch dimension).
      for (auto& pr : inputs_) {
        if (pr.second.Shape().size() > 0) {
          batch_size_ = pr.second.Shape()[0];

          // Inefficient but will be unnecessary once V2 is only
          // protocol
          for (size_t i = 0; i < pr.second.Shape().size() - 1; ++i) {
            pr.second.MutableShape()->at(i) = pr.second.Shape()[i + 1];
          }
          pr.second.MutableShape()->pop_back();
        }
      }
    }
  }

  // Make sure the request has a batch-size > 0. Even for models that
  // don't support batching the requested batch size must be 1.
  if (batch_size_ < 1) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "inference request batch-size must be >= 1 for '" + model_name_ + "'");
  }

  // Make sure request batch-size doesn't exceed what is supported by
  // the model. For models that don't support batching the request
  // batch-size will still be 1.
  if ((batch_size_ != 1) &&
      ((int)batch_size_ > model_config.max_batch_size())) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "inference request batch-size must be <= " +
            std::to_string(model_config.max_batch_size()) + " for '" +
            model_name_ + "'");
  }

  // Validate if the requested output name exists in the model configuration
  for (const auto& pr : requested_outputs_) {
    const ModelOutput* output_config;
    RETURN_IF_ERROR(backend.GetOutput(pr.first, &output_config));
  }

  // Make sure that the request is providing the same number of inputs
  // as is expected by the model.
  if (inputs_.size() != (size_t)model_config.input_size()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "expected " + std::to_string(model_config.input_size()) +
            " inputs but got " + std::to_string(inputs_.size()) +
            " inputs for model '" + model_name_ + "'");
  }

  // Update each input to have shape and batch-byte-size.
  for (auto& pr : inputs_) {
    const ModelInput* input_config;
    RETURN_IF_ERROR(backend.GetInput(pr.first, &input_config));

    auto& shape = *pr.second.MutableShape();

    // If the inference request specifies a shape for an input, make
    // sure it matches what the model expects.
    if (shape.size() > 0) {
      if (!CompareDimsWithWildcard(input_config->dims(), shape)) {
        return Status(
            RequestStatusCode::INVALID_ARG,
            "unexpected shape for input '" + pr.first + "' for model '" +
                model_name_ + "'. Expected " +
                DimsListToString(input_config->dims()) + ", got " +
                DimsListToString(shape));
      }

      // If there is a reshape for this input then clear the dims and
      // set them to the reshape. As reshape may have variable-size
      // dimensions, we need to record corresponding value so that we
      // can set the value correctly for reshape.
      if (input_config->has_reshape()) {
        std::deque<int64_t> variable_size_values;
        for (int64_t idx = 0; idx < input_config->dims_size(); idx++) {
          if (input_config->dims(idx) == -1) {
            variable_size_values.push_back(shape[idx]);
          }
        }

        shape.clear();
        for (const auto& dim : input_config->reshape().shape()) {
          if (dim == -1) {
            shape.push_back(variable_size_values.front());
            variable_size_values.pop_front();
          } else {
            shape.push_back(dim);
          }
        }
      }
    }

    // If we don't have shape for the input at this point then the
    // request didn't specify it, or it has a reshape that we must use
    // instead. FIXMEV2 shape required for V2 so reevaluate this code.
    if (shape.size() == 0) {
      const DimsList& dims = (input_config->has_reshape())
                                 ? input_config->reshape().shape()
                                 : input_config->dims();

      // Inference request doesn't specify shape, make sure input
      // shape is fully specified in the model and calculate expected
      // size from the model configuration.
      for (auto dim : dims) {
        if (dim < 0) {
          return Status(
              RequestStatusCode::INVALID_ARG,
              "model supports variable-size for input '" + pr.first +
                  "', request must specify input shape for model '" +
                  model_name_ + "'");
        }

        shape.push_back(dim);
      }
    }

    // For fixed-size datatype the tensor used to calculate byte-size
    // is:
    //
    //   [ batch-size, tensor-shape ] : for batching model and
    //   non-zero-rank tensor. For example, batch-size 4 and dims [ 1,
    //   2 ] the full tensor shape is [ 4, 1, 2 ].
    //
    //   [ tensor-shape ] : for non-batching model and non-zero-rank
    //   tensor. For example, dims [ 1, 2 ] the full tensor shape is [
    //   1, 2 ].
    //
    //   [ batch-size ] : for batching model and zero-rank tensor. For
    //   example, batch-size 4 with dims [ 1 ] and reshape [ ], the
    //   full tensor shape is [ 4 ].
    //
    // Note that non-batching zero-rank tensor is not allowed since
    // that will always be shape [], i.e. a tensor with no contents.
    //
    uint64_t bs = 0;
    if (IsFixedSizeDataType(input_config->data_type())) {
      bs = GetByteSize(input_config->data_type(), shape);
      int multiplier = (input_config->is_shape_tensor() ? 1 : batch_size_);
      if (model_config.max_batch_size() > 0) {
        if (shape.size() == 0) {
          bs = GetDataTypeByteSize(input_config->data_type()) * multiplier;
        } else {
          bs *= multiplier;
        }
      }

      // If batch-byte-size is given check to make sure that the
      // calculated batch size matches
      if ((pr.second.BatchByteSize() != 0) &&
          (pr.second.BatchByteSize() != bs)) {
        return Status(
            RequestStatusCode::INVALID_ARG,
            "specific batch-byte-size for input '" + pr.first +
                "' does not match expected byte-size calculated from shape and "
                "datatype for model '" +
                model_name_ + "'");
      }
    } else {
      // The input's datatype is not fixed-sized (like TYPE_STRING),
      // use the full-batch size specified by the request.
      bs = pr.second.BatchByteSize();
    }

    pr.second.SetBatchByteSize(bs);
  }

  return Status::Success;
}

//
// Input
//
InferenceRequest::Input::Input(
    const std::string& name, const std::vector<int64_t>& shape,
    const uint64_t batch_byte_size)
    : name_(name), shape_(shape), batch_byte_size_(batch_byte_size)
{
}

Status
InferenceRequest::Input::AppendData(
    const void* base, size_t byte_size, TRTSERVER_Memory_Type memory_type,
    int64_t memory_type_id)
{
  if (data_ == nullptr) {
    data_.reset(new MemoryReference());
    data_idx_ = 0;
  }

  if (byte_size > 0) {
    std::static_pointer_cast<MemoryReference>(data_)->AddBuffer(
        static_cast<const char*>(base), byte_size, memory_type, memory_type_id);
  }

  return Status::Success;
}

Status
InferenceRequest::Input::SetData(const std::shared_ptr<Memory>& data)
{
  if (data_ != nullptr) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "input '" + name_ + "' already has data, can't overwrite");
  }

  data_ = data;
  data_idx_ = 0;

  return Status::Success;
}

Status
InferenceRequest::Input::NextContent(
    const void** content, size_t* content_byte_size,
    TRTSERVER_Memory_Type* memory_type, int64_t* memory_type_id)
{
  if (*content_byte_size == 0) {
    *content = nullptr;
    return Status::Success;
  }

  *content = data_->BufferAt(
      data_idx_++, content_byte_size, memory_type, memory_type_id);

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

}}  // namespace nvidia::inferenceserver
