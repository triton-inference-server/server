// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
#include <thread>
#include "src/core/backend.h"
#include "src/core/backend_context.h"
#include "src/core/model_config.pb.h"
#include "src/core/scheduler.h"
#include "src/core/status.h"
#include "src/core/sync_queue.h"

namespace nvidia { namespace inferenceserver {

class PlanBackend : public InferenceBackend {
 public:
  PlanBackend() = default;
  PlanBackend(PlanBackend&&) = default;

  void Run(
      uint32_t runner_idx, std::vector<Scheduler::Payload>* payloads,
      std::function<void(Status)> OnCompleteQueuedPayloads) override;

  // Create a context for execution for each instance for the
  // serialized plans specified in 'models'.
  Status CreateExecutionContexts(
      const std::unordered_map<std::string, std::vector<char>>& models);
  Status CreateExecutionContext(
      const std::string& instance_name, const int gpu_device,
      const std::unordered_map<std::string, std::vector<char>>& models,
      const ::google::protobuf::RepeatedPtrField<std::string>& profile_names,
      const std::shared_ptr<SyncQueue<size_t>>& context_queue);

 private:
  DISALLOW_COPY_AND_ASSIGN(PlanBackend);
  friend std::ostream& operator<<(std::ostream&, const PlanBackend&);

  // For each model instance there is a context.
  struct Context : BackendContext {
    Context(
        const std::string& name, const int gpu_device,
        const int max_batch_size);
    ~Context();

    DISALLOW_MOVE(Context);
    DISALLOW_COPY_AND_ASSIGN(Context);

    struct TensorRTContext;

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
    bool BuildCudaGraph(TensorRTContext* trt_context, const int batch_size);

    Status InitOptimizationProfiles(
        const ::google::protobuf::RepeatedPtrField<std::string>& profile_names);

    // See BackendContext::Run()
    Status Run(
        const InferenceBackend* base,
        std::vector<Scheduler::Payload>* payloads) override;

    void ProcessResponse(
        size_t context_idx, std::shared_ptr<SyncQueue<size_t>> context_queue);

    // A struct to hold TensorRT execution context and its meta data, a backend
    // context can have multiple of this struct if multiple optimization
    // profiles is specified.
    struct TensorRTContext {
      TensorRTContext(const std::string& profile_name, int binding_cnts)
          : profile_name_(profile_name), context_(nullptr),
            min_dims_(binding_cnts), max_dims_(binding_cnts),
            opt_dims_(binding_cnts)
      {
      }
      std::string profile_name_;
      nvinfer1::IExecutionContext* context_;

      // The CUDA graphs captured for the model for different
      // batch-sizes.
      std::unordered_map<int, cudaGraph_t> cuda_graphs_;
      std::unordered_map<int, cudaGraphExec_t> cuda_graph_execs_;

      // Min Dimensions per bindings
      std::vector<nvinfer1::Dims> min_dims_;

      // Max Dimensions per bindings
      std::vector<nvinfer1::Dims> max_dims_;

      // Optimized Dimensions per bindings
      std::vector<nvinfer1::Dims> opt_dims_;
    };

    // A group of CUDA events that signals different stages of the request.
    // One group should be used for one request at any given moment.
    struct CUDAEventSet {
      // CUDA event to signal input buffer availability.
      cudaEvent_t ready_for_input_;
      cudaEvent_t input_ready_;

      // CUDA event for capturing correct timestamp.
      cudaEvent_t ready_for_output_;
      cudaEvent_t output_ready_;
    };

    // Number of CUDA event set for each instance.
    static constexpr int EVENT_SET_COUNT = 2;

    Status InitEventSet();
    Status DestroyEventSet();

    std::map<int, TensorRTContext>::iterator GetMostOptimizedProfile(
        size_t total_batch_size,
        const std::shared_ptr<InferRequestProvider>& input_request_provider);

    // TensorRT components for the model
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;

    // Additional CUDA stream to overlap copy and execution.
    cudaStream_t input_copy_stream_;

    // Use two sets of events each for current request and next request.
    CUDAEventSet events_[EVENT_SET_COUNT];
    size_t next_set_;

    // Completion thread for handling items in the corresponding completion
    // queue. One thread per instance so that the thread logic is simple as this
    // avoids busy-looping on different model executions' event states.
    std::thread completion_thread_;

    // Assume that the lifetime of the payload is extended until the completion
    // callback is called
    SyncQueue<std::tuple<
        std::function<void(Status)>, std::vector<Scheduler::Payload>*, size_t>>
        completion_queue_;

    // Map from profile index to the corresponding TensorRT context. Use map
    // to ensure each profile index is mapped to exactly one TensorRT context.
    std::map<int, TensorRTContext> trt_contexts_;

    // Is set true if the loaded model has one or more dynamic shaped inputs
    bool is_dynamic_;

    // The total number of bindings
    int total_bindings_;

    // The number of expected bindings to the model. In case of dynamic shapes,
    // it is the number of expected bindings to the configured optimization
    // profile.
    int num_expected_bindings_;

    // The maximum possible size of the TensorRT tensor and the corresponding
    // allocated GPU buffer across all optimization
    // profile. The array sizes are equal to Context::num_expected_bindings_
    std::vector<uint64_t> byte_sizes_;
    std::vector<void*> buffers_;

    // The pointer to the CUDA buffer for each binding index of the TensorRT
    // engine. This is used to match the TensorRT context execution declaration
    // while minimizing memory allocation.
    // The array size is equal to Context::total_bindings_
    std::vector<void*> buffer_bindings_;
  };

  // vector for storing available context queue associated with a runner
  std::vector<std::shared_ptr<SyncQueue<size_t>>> available_context_queue_;

  // Next context to be used for the runner.
  std::vector<size_t> next_context_;
};

std::ostream& operator<<(std::ostream& out, const PlanBackend& pb);

}}  // namespace nvidia::inferenceserver
