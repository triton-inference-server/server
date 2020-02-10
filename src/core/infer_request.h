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

namespace nvidia { namespace inferenceserver {

class InferenceBackend;
class InferenceServer;
class InferSharedMemory;

//
// An inference request
//
class InferenceRequest {
 public:
  // Reference to a block of memory in a shared memory region
  class SharedMemory {
   public:
    SharedMemory() = default;
    SharedMemory(
        const std::string& name, const uint64_t offset,
        const uint64_t byte_size)
        : name_(name), offset_(offset), byte_size_(byte_size)
    {
    }

    // The name given during registration of a shared memory region
    // that holds the input data (or where the output data should be
    // written).
    const std::string& Name() const { return name_; }

    // The offset from the start of the shared memory region to the
    // start of the memory block.
    uint64_t Offset() const { return offset_; }

    // Size of the memory block, in bytes.
    uint64_t ByteSize() const { return byte_size_; }

   private:
    std::string name_;
    uint64_t offset_;
    uint64_t byte_size_;
  };

  // Input tensor
  class Input {
   public:
    Input() = default;
    Input(const std::string& name, const Input& other);
    Input(
        const std::string& name, const std::vector<int64_t>& shape,
        const uint64_t batch_byte_size);
    Input(
        const std::string& name, const std::vector<int64_t>& shape,
        const uint64_t batch_byte_size, const InferSharedMemory& shared_memory);

    // The name of the input tensor. There is no mutable operator for
    // the name because it is used in a InferenceReference map and a
    // mutable method would allow it to get out-of-sync.
    const std::string& Name() const { return name_; }

    // The shape of the input tensor.
    const std::vector<int64_t>& Shape() const { return shape_; }
    std::vector<int64_t>* MutableShape() { return &shape_; }

    // The size, in bytes, of the entire input tensor.
    uint64_t BatchByteSize() const { return batch_byte_size_; }
    void SetBatchByteSize(uint64_t b) { batch_byte_size_ = b; }

    // Shared-memory information for input tensors being communicated
    // via shared-memory.
    bool SharedMemoryEnable() const { return use_shared_memory_; }
    const InferenceRequest::SharedMemory& SharedMemory() const
    {
      return shared_memory_;
    }

   private:
    std::string name_;
    std::vector<int64_t> shape_;
    uint64_t batch_byte_size_;

    bool use_shared_memory_;
    InferenceRequest::SharedMemory shared_memory_;
  };

  // Requested output tensor
  class RequestedOutput {
   public:
    RequestedOutput() = default;
    RequestedOutput(const std::string& name, const uint32_t classification_cnt);
    RequestedOutput(
        const std::string& name, const uint32_t classification_cnt,
        const InferSharedMemory& shared_memory);

    // The name of the output tensor. There is no mutable operator for
    // the name because it is used in a InferenceReference map and a
    // mutable method would allow it to get out-of-sync.
    const std::string& Name() const { return name_; }

    // The classification count for the output. If zero then the
    // result is returned as a raw tensor. If > 0 then result is
    // returned as a classification of the indicated number of
    // classes.
    uint32_t ClassificationCount() const { return classification_cnt_; }

    // Shared-memory information for output tensors being communicated
    // via shared-memory.
    bool SharedMemoryEnable() const { return use_shared_memory_; }
    const InferenceRequest::SharedMemory& SharedMemory() const
    {
      return shared_memory_;
    }

   private:
    std::string name_;

    // If > 0 then return result as a classification with the
    // indicated number of classes.
    uint32_t classification_cnt_;

    bool use_shared_memory_;
    InferenceRequest::SharedMemory shared_memory_;
  };

  InferenceRequest();
  InferenceRequest(
      const std::string& model_name, const int64_t model_version,
      const uint32_t protocol_version);

  // Initialize this request so that it is prepared to run on
  // 'server'. All options and tensor data must be set before calling.
  Status Init(InferenceServer* server);
  Status Init(const std::shared_ptr<InferenceBackend>& backend);

  // Finalize this request after inference is complete. Options and
  // tensor data are still accessible after finalizing but backend is
  // no longer valid.
  Status Fini();

  uint32_t ProtocolVersion() const { return protocol_version_; }

  const std::string& ModelName() const { return model_name_; }
  void SetModelName(const std::string& n) { model_name_ = n; }

  int64_t RequestedModelVersion() const { return requested_model_version_; }
  void SetRequestedModelVersion(int64_t v) { requested_model_version_ = v; }

  int64_t ActualModelVersion() const { return actual_model_version_; }
  const std::shared_ptr<InferenceBackend>& Backend() const { return backend_; }

  uint64_t Id() const { return id_; }
  void SetId(uint64_t i) { id_ = i; }

  // FIXMEV2 this replaces "Id" once V2 is only option.
  const std::string& IdStr() const { return id_str_; }
  void SetIdStr(const std::string& i) { id_str_ = i; }

  uint32_t Flags() const { return flags_; }
  void SetFlags(uint32_t f) { flags_ = f; }

  uint64_t CorrelationId() const { return correlation_id_; }
  void SetCorrelationId(uint64_t c) { correlation_id_ = c; }

  uint32_t BatchSize() const { return batch_size_; }
  void SetBatchSize(uint32_t b) { batch_size_ = b; }

  const std::unordered_map<std::string, Input>& Inputs() const
  {
    return inputs_;
  }

  const std::unordered_map<std::string, RequestedOutput>& RequestedOutputs()
      const
  {
    return requested_outputs_;
  }

  // Add an input to the request. If 'input' is non-null return a
  // pointer to the newly added input.
  Status AddInput(
      const std::string& name, const Input& other, Input** input = nullptr);
  Status AddInput(
      const std::string& name, const DimsList& shape,
      const uint64_t batch_byte_size, Input** input = nullptr);
  Status AddInput(
      const std::string& name, const std::vector<int64_t>& shape,
      const uint64_t batch_byte_size, Input** input = nullptr);
  Status AddInput(
      const std::string& name, const DimsList& shape,
      const uint64_t batch_byte_size, const InferSharedMemory& shared_memory,
      Input** input = nullptr);
  Status AddInput(
      const std::string& name, const std::vector<int64_t>& shape,
      const uint64_t batch_byte_size, const InferSharedMemory& shared_memory,
      Input** input = nullptr);

  Status RequestOutput(
      const std::string& name, const uint32_t classification_cnt);
  Status RequestOutput(
      const std::string& name, const uint32_t classification_cnt,
      const InferSharedMemory& shared_memory);

  // Append a new buffer of data to this input.
  Status AppendInputData(
      const char* name, const void* base, size_t byte_size,
      TRTSERVER_Memory_Type memory_type, int64_t memory_type_id);

  // Set the data for this input. Error is input already has some data.
  Status SetInputData(
      const std::string& name, const std::shared_ptr<Memory>& data);

  const std::unordered_map<std::string, std::shared_ptr<Memory>>& InputDataMap()
      const
  {
    return input_map_;
  }

  Status Normalize();

 private:
  std::string model_name_;

  // A model version is requested and based on version policy a
  // specific version is actually used for inference.
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

  std::unordered_map<std::string, Input> inputs_;
  std::unordered_map<std::string, RequestedOutput> requested_outputs_;

  // The model backend associated with this request.
  std::shared_ptr<InferenceBackend> backend_;

  // The data for each input. FIXMEV2 should have memory/data for each
  // input in the Input object.
  std::unordered_map<std::string, std::shared_ptr<Memory>> input_map_;
};

}}  // namespace nvidia::inferenceserver
