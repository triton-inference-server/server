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

#include <string>
#include <unordered_map>
#include <vector>
#include "src/core/memory.h"
#include "src/core/model_config.h"
#include "src/core/status.h"
#include "src/core/tritonserver.h"

namespace nvidia { namespace inferenceserver {

// FIXMEV2 shouldn't need the conversions
TRTSERVER_Memory_Type TritonMemTypeToTrt(TRITONSERVER_Memory_Type mem_type);

TRITONSERVER_Memory_Type TrtMemTypeToTriton(TRTSERVER_Memory_Type mem_type);

class InferenceBackend;
class InferenceServer;

//
// An inference request. A request can be used multiple times for
// inference but before each inference PrepareForInference() must be
// called to verify and prepare the request. Verification involves
// ensuring that any changes made since the last inference are
// valid. Preparing involves removing/resetting any state left over
// from the previous inference. After each inference
// CleanupFromInference() must be called.
//
class InferenceRequest {
 public:
  // Input tensor
  class Input {
   public:
    Input();
    Input(
        const std::string& name, const std::vector<int64_t>& shape,
        const uint64_t batch_byte_size);
    Input(
        const std::string& name, const DataType datatype,
        const std::vector<int64_t>& shape, const uint64_t batch_byte_size);
    Input(
        const std::string& name, const DataType datatype, const int64_t* shape,
        const uint64_t dim_count);

    // The name of the input tensor. There is no mutable operator for
    // the name because it is used in a InferenceReference map and a
    // mutable method would allow it to get out-of-sync.
    const std::string& Name() const { return name_; }

    // Data type of the input tensor.
    DataType DType() const { return datatype_; }

    // FIXMEV2 Datatype should always be set by constructor for V2
    void SetDType(const DataType t) { datatype_ = t; }

    // The original shape of the input tensor.
    const std::vector<int64_t>& OriginalShape() const
    {
      return original_shape_;
    }

    // The shape of the input tensor after normalization. This shape
    // is the original shape modified as required/expected by
    // inference processing.
    const std::vector<int64_t>& Shape() const { return shape_; }
    std::vector<int64_t>* MutableShape() { return &shape_; }

    // The size, in bytes, of the entire input tensor.
    uint64_t BatchByteSize() const { return batch_byte_size_; }
    void SetBatchByteSize(uint64_t b) { batch_byte_size_ = b; }

    // The data for this input.
    const std::shared_ptr<Memory>& Data() const { return data_; }

    // Append a new buffer of data to this input.
    Status AppendData(
        const void* base, size_t byte_size, TRTSERVER_Memory_Type memory_type,
        int64_t memory_type_id);

    // Append a new buffer of data to this input.
    Status AppendData(
        const void* base, size_t byte_size,
        TRITONSERVER_Memory_Type memory_type, int64_t memory_type_id);

    // Set the data for this input. Error if input already has some
    // data.
    Status SetData(const std::shared_ptr<Memory>& data);

    // Remove all existing data for the input.
    Status RemoveAllData();

    // Get the number of buffers that containing the input tensor
    // data.
    size_t ContentBufferCount() const
    {
      return (data_ == nullptr) ? 0 : data_->BufferCount();
    }

    // Get the 'idx' buffer containing a contiguous chunk of bytes for
    // the input. Return error is 'idx' refers to a buffer that does
    // not exist. Return a pointer to the chunk in 'content' and the
    // size of the chunk in 'content_byte_size'. 'memory_type' acts as
    // both input and output. On input 'memory_type' is the buffer
    // memory type preferred by the function caller. On return
    // 'memory_type' gives the actual memory type of the chunk pointed
    // to by 'content'.  'memory_type_id' acts as both input and
    // output. On input 'memory_type_id' is the buffer memory type id
    // preferred by the function caller.  On return 'memory_type_id'
    // gives the actual memory type id of the chunk pointed to by
    // 'content'.
    Status Content(
        const size_t idx, const void** content, size_t* content_byte_size,
        TRTSERVER_Memory_Type* memory_type, int64_t* memory_type_id) const;

   private:
    friend std::ostream& operator<<(
        std::ostream& out, const InferenceRequest::Input& input);

    std::string name_;
    DataType datatype_;
    std::vector<int64_t> original_shape_;
    std::vector<int64_t> shape_;

    // FIXMEV2 why needed? Should get total data size from data_.
    uint64_t batch_byte_size_;

    std::shared_ptr<Memory> data_;
  };

  // Requested output tensor
  class RequestedOutput {
   public:
    RequestedOutput() = default;
    RequestedOutput(const std::string& name, const uint32_t classification_cnt);

    // The name of the output tensor. There is no mutable operator for
    // the name because it is used in a InferenceReference map and a
    // mutable method would allow it to get out-of-sync.
    const std::string& Name() const { return name_; }

    // The classification count for the output. If zero then the
    // result is returned as a raw tensor. If > 0 then result is
    // returned as a classification of the indicated number of
    // classes.
    uint32_t ClassificationCount() const { return classification_cnt_; }
    void SetClassificationCount(uint32_t c) { classification_cnt_ = c; }

   private:
    friend std::ostream& operator<<(
        std::ostream& out, const InferenceRequest::RequestedOutput& output);

    std::string name_;

    // If > 0 then return result as a classification with the
    // indicated number of classes.
    uint32_t classification_cnt_;
  };

  // InferenceRequest
  InferenceRequest(
      const std::string& model_name, const int64_t requested_model_version,
      const int64_t actual_model_version, const uint32_t protocol_version);

  uint32_t ProtocolVersion() const { return protocol_version_; }
  const std::string& ModelName() const { return model_name_; }
  int64_t RequestedModelVersion() const { return requested_model_version_; }
  int64_t ActualModelVersion() const { return actual_model_version_; }

  uint64_t Id() const { return id_; }
  void SetId(uint64_t i) { id_ = i; }

  // FIXMEV2 this replaces "Id" once V2 is only option.
  const std::string& IdStr() const { return id_str_; }
  void SetIdStr(const std::string& i) { id_str_ = i; }

  uint32_t Flags() const { return flags_; }
  void SetFlags(uint32_t f) { flags_ = f; }

  uint64_t CorrelationId() const { return correlation_id_; }
  void SetCorrelationId(uint64_t c) { correlation_id_ = c; }

  // FIXMEV2 remove setter as batch size will only be set during
  // normalization
  uint32_t BatchSize() const { return batch_size_; }
  void SetBatchSize(uint32_t b)
  {
    needs_normalization_ = true;
    batch_size_ = b;
  }

  uint32_t Priority() const { return priority_; }
  void SetPriority(uint32_t p) { priority_ = p; }

  uint64_t TimeoutMicroseconds() const { return timeout_us_; }
  void SetTimeoutMicroseconds(uint64_t t) { timeout_us_ = t; }

  // The original inputs are the inputs added to the request before
  // the inference executed (that is before
  // TRITONSERVER_ServerInferAsync or equivalent is called). Once
  // execution has started the original inputs should not be modified
  // until execution completes (and those modifications will apply to
  // the next inference execution).
  Status MutableOriginalInput(const std::string& name, Input** input);
  std::unordered_map<std::string, Input>* MutableOriginalInputs()
  {
    needs_normalization_ = true;
    return &original_inputs_;
  }
  const std::unordered_map<std::string, Input>& OriginalInputs() const
  {
    return original_inputs_;
  }

  // The override inputs are the inputs added to the request after
  // inference execution has started (that is after
  // TRITONSERVER_ServerInferAsync or equivalent is called). During
  // inference processing, if Triton needs to change an original input
  // it will add an override instead of changing the original. Triton
  // will also use an override if it needs to add a new input to the
  // request. Overrides are recorded as shared_ptr so that the same
  // override can be used efficiently multiple times or even in
  // multiple requests simultaneously. Must be careful not to modify
  // an override input if it is being shared unless you want that
  // change to be reflected in all requests that hold that override
  // input. Override inputs within a specific request are not
  // persisted across inference calls.
  std::unordered_map<std::string, std::shared_ptr<Input>>*
  MutableOverrideInputs()
  {
    return &override_inputs_;
  }
  const std::unordered_map<std::string, std::shared_ptr<Input>>&
  OverrideInputs() const
  {
    return override_inputs_;
  }

  // Get an input taking into account both original inputs and
  // overrides. If an override input is available use it, otherwise
  // use the original input. Accessing inputs via this method is not
  // valid until after PrepareForInference is called.
  Status ImmutableInput(const std::string& name, const Input** input) const;
  const std::unordered_map<std::string, Input*>& ImmutableInputs() const
  {
    return inputs_;
  }

  Status MutableRequestedOutput(
      const std::string& name, RequestedOutput** output);
  const std::unordered_map<std::string, RequestedOutput>& RequestedOutputs()
      const
  {
    return requested_outputs_;
  }

  // Add an original input to the request. If 'input' is non-null
  // return a pointer to the newly added input.
  Status AddOriginalInput(
      const std::string& name, const DimsList& shape,
      const uint64_t batch_byte_size, Input** input = nullptr);
  Status AddOriginalInput(
      const std::string& name, const std::vector<int64_t>& shape,
      const uint64_t batch_byte_size, Input** input = nullptr);
  Status AddOriginalInput(
      const std::string& name, const DataType datatype, const int64_t* shape,
      const uint64_t dim_count, Input** input = nullptr);

  // Remove a single original input or all inputs.
  Status RemoveOriginalInput(const std::string& name);
  Status RemoveAllOriginalInputs();

  // Add an override input to the request. If 'input' is non-null
  // return a pointer to the newly added input.
  Status AddOverrideInput(
      const std::string& name, const DataType datatype,
      const std::vector<int64_t>& shape, const uint64_t batch_byte_size,
      std::shared_ptr<Input>* input = nullptr);

  // Add an override input to the request.
  Status AddOverrideInput(const std::shared_ptr<Input>& input);

  // Request an output.
  Status AddRequestedOutput(
      const std::string& name, const uint32_t classification_cnt = 0);

  // Remove a single requested output or all requested outputs.
  Status RemoveRequestedOutput(const std::string& name);
  Status OutputNames(const char** name_vec, uint64_t* name_count);
  Status RemoveAllRequestedOutputs();

  // Prepare this request for inference. We pass backend here as
  // non-shared-ptr because normalize must be used in contexts where
  // the backend shared_ptr does not yet exist (e.g. warmup).
  Status PrepareForInference(const InferenceBackend& backend);

 private:
  friend std::ostream& operator<<(
      std::ostream& out, const InferenceRequest& request);

  Status NormalizeV1(const InferenceBackend& backend);
  Status NormalizeV2(const InferenceBackend& backend);

  // Has anything in the request potentially changed in a way that
  // causes normalization to be required when preparing the request
  // for inference.
  bool needs_normalization_;

  std::string model_name_;

  // The model version as requested and based on version policy the
  // specific version that is actually used for inference.
  int64_t requested_model_version_;
  int64_t actual_model_version_;

  // FIXMEV2 remove
  uint32_t protocol_version_;

  // For V1 id is an int, for V2 it is a string.
  uint64_t id_;
  std::string id_str_;

  uint32_t flags_;
  uint64_t correlation_id_;
  uint32_t batch_size_;
  uint32_t priority_;
  uint64_t timeout_us_;

  std::unordered_map<std::string, Input> original_inputs_;
  std::unordered_map<std::string, std::shared_ptr<Input>> override_inputs_;
  std::unordered_map<std::string, Input*> inputs_;
  std::unordered_map<std::string, RequestedOutput> requested_outputs_;
};

std::ostream& operator<<(std::ostream& out, const InferenceRequest& request);
std::ostream& operator<<(
    std::ostream& out, const InferenceRequest::Input& input);
std::ostream& operator<<(
    std::ostream& out, const InferenceRequest::RequestedOutput& output);

}}  // namespace nvidia::inferenceserver
