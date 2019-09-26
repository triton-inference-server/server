// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "src/core/backend.h"
#include "src/core/backend_context.h"
#include "src/core/model_config.pb.h"
#include "src/core/scheduler.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class PlanBackend : public InferenceBackend {
 public:
  PlanBackend() = default;
  PlanBackend(PlanBackend&&) = default;

  Status Init(const std::string& path, const ModelConfig& config);

  // Create a context for execution for each instance for the
  // serialized plans specified in 'models'.
  Status CreateExecutionContexts(
      const std::unordered_map<std::string, std::vector<char>>& models);
  Status CreateExecutionContext(
      const std::string& instance_name, const int gpu_device,
      const std::unordered_map<std::string, std::vector<char>>& models,
      const std::string profile_index);

 private:
  // Run model on the context associated with 'runner_idx' to
  // execute for one or more requests.
  void Run(
      uint32_t runner_idx, std::vector<Scheduler::Payload>* payloads,
      std::function<void(Status)> OnCompleteQueuedPayloads);

 private:
  DISALLOW_COPY_AND_ASSIGN(PlanBackend);
  friend std::ostream& operator<<(std::ostream&, const PlanBackend&);

  // For each model instance there is a context.
  struct Context : BackendContext {
    Context(
        const std::string& name, const int gpu_device, const int max_batch_size,
        const int profile_index);
    ~Context();

    DISALLOW_MOVE(Context);
    DISALLOW_COPY_AND_ASSIGN(Context);

    Status ValidateInputs(
        const ::google::protobuf::RepeatedPtrField<ModelInput>& ios);
    Status ValidateOutputs(
        const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios);

    Status InitializeInputBinding(
        const std::string& input_name, const DataType input_datatype,
        const DimsList& input_dims, const bool support_batching,
        const bool is_control = false);
    Status InitializeSequenceControlInputBindings(
        const ModelConfig& config, const bool support_batching);
    Status InitializeConfigInputBindings(
        const ::google::protobuf::RepeatedPtrField<ModelInput>& ios,
        const bool support_batching);
    Status InitializeConfigOutputBindings(
        const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios,
        const bool support_batching);
    bool BuildCudaGraph(const int batch_size);

    void InitProfile();

    // Run model to execute for one or more requests. This function
    // assumes that it is only called by the single runner thread that
    // is assigned to this context. A non-OK return status indicates
    // an internal error that prevents any of the of requests from
    // completing. If an error is isolate to a single request payload
    // it will be reported in that payload.
    Status Run(std::vector<Scheduler::Payload>* payloads);

    // TensorRT components for the model
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;

    // Is set true if the loaded model has one or more dynamic shaped inputs
    bool is_dynamic_;
    // The configured optimization profile index
    int profile_index_;
    // Offset used for addressing bindings from the configured optmization
    // profile
    int binding_offset_;
    // Min Dimensions per bindings
    nvinfer1::Dims* min_dims_;
    // Max Dimensions per bindings
    nvinfer1::Dims* max_dims_;

    // Stores the minimum of the maximum possible value of the first dimension
    int max_dynamic_batch_size_;

    // The total number of bindings
    int total_bindings_;
    // The number of expected bindings to the model. In case of dynamic shapes,
    // it is the number of expected bindings to the configured optimization
    // profile.
    int num_expected_bindings_;

    // For each binding index of the TensorRT engine, the size of the
    // corresponding tensor and pointer to the CUDA buffer for the
    // tensor. These are arrays with size equal to
    // Context::num_expected_bindings_
    uint64_t* byte_sizes_;
    void** buffers_;

    // The CUDA graphs captured for the model for different
    // batch-sizes.
    std::unordered_map<int, cudaGraph_t> cuda_graphs_;
    std::unordered_map<int, cudaGraphExec_t> cuda_graph_execs_;
  };

  std::vector<std::unique_ptr<Context>> contexts_;
};

std::ostream& operator<<(std::ostream& out, const PlanBackend& pb);

}}  // namespace nvidia::inferenceserver
