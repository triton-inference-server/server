// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <torch/script.h>  // One-stop header.
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "src/core/backend.h"
#include "src/core/backend_context.h"
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/core/scheduler.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class AllocatedSystemMemory;

class LibTorchBackend : public InferenceBackend {
 public:
  LibTorchBackend() = default;
  LibTorchBackend(LibTorchBackend&&) = default;

  Status Init(const std::string& path, const ModelConfig& config);

  // Create a context for execution for each instance for the
  // serialized .pt models specified in 'paths'.
  Status CreateExecutionContexts(
      const std::unordered_map<std::string, std::string>& paths);
  Status CreateExecutionContext(
      const std::string& instance_name, const int gpu_device,
      const std::unordered_map<std::string, std::string>& paths);

 private:
  // Run model on the context associated with 'runner_idx' to
  // execute for one or more requests.
  void Run(
      uint32_t runner_idx, std::vector<Scheduler::Payload>* payloads,
      std::function<void(Status)> OnCompleteQueuedPayloads);

 private:
  DISALLOW_COPY_AND_ASSIGN(LibTorchBackend);
  friend std::ostream& operator<<(std::ostream&, const LibTorchBackend&);

  // For each model instance there is a context.
  struct Context : BackendContext {
    Context(
        const std::string& name, const int gpu_device,
        const int max_batch_size);
    ~Context();

    DISALLOW_COPY_AND_ASSIGN(Context);
    DISALLOW_MOVE(Context);

    Status ValidateInputs(
        const ::google::protobuf::RepeatedPtrField<ModelInput>& ios);
    Status ValidateOutputs(
        const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios);

    // Set an input tensor data from payloads.
    Status SetInput(
        std::vector<torch::jit::IValue>* inputs_, const std::string& name,
        const int& ip_index, const DataType datatype, const DimsList& dims,
        const size_t total_batch_size,
        std::vector<Scheduler::Payload>* payloads,
        std::vector<std::unique_ptr<AllocatedSystemMemory>>* input_buffers,
        bool* cuda_copy);

    // Run model to execute for one or more requests. This function
    // assumes that it is only called by the single runner thread that
    // is assigned to this context. A non-OK return status indicates
    // an internal error that prevents any of the of requests from
    // completing. If an error is isolate to a single request payload
    // it will be reported in that payload.
    Status Run(
        const LibTorchBackend* base, std::vector<Scheduler::Payload>* payloads);

    // Set an input tensor from one or more payloads.
    Status SetFixedSizedInputTensor(
        std::vector<torch::jit::IValue>* inputs_, const std::string& name,
        const int& ip_index, const std::vector<int64_t>& shape,
        const DataType dtype, const size_t batch1_byte_size,
        const size_t total_byte_size, std::vector<Scheduler::Payload>* payloads,
        std::vector<std::unique_ptr<AllocatedSystemMemory>>* input_buffers,
        bool* cuda_copy);

    // Read an output tensor into one or more payloads.
    Status ReadFixedSizedOutputTensor(
        std::vector<torch::Tensor>* outputs_, const std::string& name,
        const int& op_index, const DataType dtype, const size_t dtype_byte_size,
        const size_t total_batch_size, const DimsList& dims,
        std::vector<Scheduler::Payload>* payloads, bool* cuda_copy);

    Status SetInputTensor(
        std::vector<torch::jit::IValue>* inputs_, const std::string& name,
        const int& ip_index, const std::vector<int64_t>& shape,
        const DataType dtype, char* content, const size_t byte_size,
        const TRTSERVER_Memory_Type memory_type);

    Status GetOutputTensor(
        std::vector<torch::Tensor>* outputs_, const int& op_index,
        const std::string& name, const DataType dtype, void** content,
        size_t* byte_size, std::vector<int64_t>* content_shape);

    Status Execute(
        std::vector<torch::jit::IValue>* inputs_,
        std::vector<torch::Tensor>* outputs_);

    std::shared_ptr<torch::jit::script::Module> torch_model_;
    torch::Device device_;
    std::unordered_map<std::string, int> input_index_map_;
    std::unordered_map<std::string, int> output_index_map_;
  };

  std::vector<std::unique_ptr<Context>> contexts_;
};

std::ostream& operator<<(std::ostream& out, const LibTorchBackend& pb);

}}  // namespace nvidia::inferenceserver
