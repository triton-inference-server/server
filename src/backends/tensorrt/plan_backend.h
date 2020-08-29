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
#include "src/core/metric_model_reporter.h"
#include "src/core/model_config.pb.h"
#include "src/core/scheduler.h"
#include "src/core/status.h"
#include "src/core/sync_queue.h"

namespace nvidia { namespace inferenceserver {


class PlanBackend : public InferenceBackend {
 public:
  explicit PlanBackend(const double min_compute_capability)
      : InferenceBackend(min_compute_capability)
  {
  }
  PlanBackend(PlanBackend&&) = default;
  ~PlanBackend();

  void Run(
      uint32_t runner_idx,
      std::vector<std::unique_ptr<InferenceRequest>>&& requests) override;

  void WarmUp(uint32_t runner_idx, WarmupData& sample) override;

  // Create a context for execution for each instance for the
  // serialized plans specified in 'models'.
  Status CreateExecutionContexts(
      const std::unordered_map<std::string, std::vector<char>>& models);
  Status CreateExecutionContext(
      const std::string& instance_name, const int gpu_device,
      const std::vector<char>& models,
      const ::google::protobuf::RepeatedPtrField<std::string>& profile_names,
      const std::shared_ptr<SyncQueue<size_t>>& context_queue);

 private:
  DISALLOW_COPY_AND_ASSIGN(PlanBackend);
  friend std::ostream& operator<<(std::ostream&, const PlanBackend&);

#ifdef TRITON_ENABLE_CUDA_GRAPH
  struct GraphSpec {
    GraphSpec() : batch_size_(0), captured_(false) {}
    int batch_size_;
    std::map<std::string, std::vector<int64_t>> shapes_;
    bool captured_;
  };
  Status InitializeGraphSpecs(std::vector<GraphSpec>* graph_specs);
#endif

  Status DuplicateWarmupRequests(
      const std::vector<std::unique_ptr<InferenceRequest>>& warmup_requests,
      std::vector<std::unique_ptr<InferenceRequest>>* requests);

  // For each model instance there is a context.
  struct Context : BackendContext {
    Context(
        const std::string& name, const int gpu_device, const int max_batch_size,
        const bool enable_pinned_input, const bool enable_pinned_output,
        std::unique_ptr<MetricModelReporter>&& metric_reporter);
    ~Context();

    DISALLOW_MOVE(Context);
    DISALLOW_COPY_AND_ASSIGN(Context);

    struct TensorRTContext;

    Status ValidateInputs(
        const ::google::protobuf::RepeatedPtrField<inference::ModelInput>& ios,
        const std::set<std::string>& allowed_shape_tensors);
    Status ValidateOutputs(
        const ::google::protobuf::RepeatedPtrField<inference::ModelOutput>& ios,
        const std::set<std::string>& allowed_shape_tensors);

    Status InitializeExecuteInputBinding(
        const std::string& input_name, const inference::DataType input_datatype,
        const DimsList& input_dims, const bool is_control = false,
        const bool is_ragged = false);
    Status InitializeShapeInputBinding(
        const std::string& input_name, const inference::DataType input_datatype,
        const DimsList& input_dims);
    Status InitializeSequenceControlInputBindings(
        const inference::ModelConfig& config);
    Status InitializeBatchInputBindings(const inference::ModelConfig& config);
    Status InitializeConfigExecuteInputBindings(
        const ::google::protobuf::RepeatedPtrField<inference::ModelInput>& ios);
    Status InitializeConfigShapeInputBindings(
        const ::google::protobuf::RepeatedPtrField<inference::ModelInput>& ios);
    Status InitializeConfigExecuteOutputBindings(
        const ::google::protobuf::RepeatedPtrField<inference::ModelOutput>&
            ios);
    Status InitializeConfigShapeOutputBindings(
        const ::google::protobuf::RepeatedPtrField<inference::ModelOutput>&
            ios);
    Status InitializeBatchOutputBindings(const inference::ModelConfig& config);
#ifdef TRITON_ENABLE_CUDA_GRAPH
    bool BuildCudaGraph(
        TensorRTContext* trt_context, const GraphSpec& graph_spec);
    bool BuildCudaGraphDynamic(
        TensorRTContext* trt_context, const GraphSpec& graph_spec);
#endif

    Status InitOptimizationProfiles(
        const ::google::protobuf::RepeatedPtrField<std::string>& profile_names);

    // Helper function to populate the shape value of specified shape
    // input that corresponds with the batch size. The first shape
    // value is asssumed to be the batch size. Its the user's
    // responsibility to ensure it is called only for the shape
    // tensors. Return true if cudaMemcpyAsync is called, and the
    // caller should call cudaStreamSynchronize before using the
    // data. Otherwise, return false.
    bool SetShapeInputBuffer(
        const std::string& name, const int32_t total_batch_size,
        const int expected_byte_size, const bool support_batching,
        std::unique_ptr<InferenceRequest>& request,
        TRITONSERVER_MemoryType dst_memory_type, int64_t dst_memory_type_id,
        char* input_buffer);

    // Helper function to set output buffer for a shape tensor. It is
    // callers resposibilty to ensure this method is called only for
    // the shape tensors. Return true if cudaMemcpyAsync is called,
    // and the caller should call cudaStreamSynchronize before using
    // the data. Otherwise, return false.
    bool SetOutputShapeTensorBuffer(
        const int32_t* content, std::unique_ptr<InferenceResponse>* response,
        InferenceResponse::Output* response_output,
        const size_t tensor_element_count, const int64_t batch_size,
        cudaStream_t stream);

    // See BackendContext::Run()
    void Run(
        InferenceBackend* base,
        std::vector<std::unique_ptr<InferenceRequest>>&& requests) override;

    void ProcessResponse(
        size_t context_idx, std::shared_ptr<SyncQueue<size_t>> context_queue);

    // A struct to hold TensorRT execution context and its meta data, a backend
    // context can have multiple of this struct if multiple optimization
    // profiles is specified.
    struct TensorRTContext {
      TensorRTContext(
          const std::string& profile_name, const int profile_idx,
          const int binding_cnts, const int event_set_cnts)
          : profile_name_(profile_name), profile_idx_(profile_idx),
            context_(nullptr), cuda_graph_execs_(event_set_cnts),
            min_dims_(binding_cnts), max_dims_(binding_cnts),
            opt_dims_(binding_cnts), min_shapes_(binding_cnts),
            max_shapes_(binding_cnts), opt_shapes_(binding_cnts)
      {
      }
      std::string profile_name_;
      int profile_idx_;
      nvinfer1::IExecutionContext* context_;

      // Struct that holds cudaGraphExec_t and the dimensions of the inputs
      // used to capture the graph
      struct CudaGraph {
        CudaGraph() : cuda_graph_exec_(nullptr) {}
        // Store in the order of the bindng index
        std::vector<std::vector<int64_t>> input_dims_;
        cudaGraphExec_t cuda_graph_exec_;
      };

      // The CUDA graphs captured for the model for different
      // batch-sizes.
      std::vector<cudaGraph_t> cuda_graphs_;
      // The key is packed input dimensions prepended by batch size, so that
      // uniqueness is guaranteed and the CUDA graphs are sorted to provide
      // convinence to find the closest CUDA graph in the future.
      std::vector<std::map<std::vector<int64_t>, CudaGraph>> cuda_graph_execs_;

      // Min Dimensions per bindings
      std::vector<nvinfer1::Dims> min_dims_;

      // Max Dimensions per bindings
      std::vector<nvinfer1::Dims> max_dims_;

      // Optimized Dimensions per bindings
      std::vector<nvinfer1::Dims> opt_dims_;

      // Min shape values per bindings
      std::vector<const int32_t*> min_shapes_;

      // Max shape values per bindings
      std::vector<const int32_t*> max_shapes_;

      // Optimized shape values per bindings
      std::vector<const int32_t*> opt_shapes_;

      // The number of shape values
      size_t nb_shape_values_;
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

    Status InitEventSet(bool busy_wait_events);
    Status DestroyEventSet();
    Status GetRequestShapeValues(
        size_t total_batch_size,
        const std::unique_ptr<InferenceRequest>& request,
        std::map<int, std::vector<int32_t>>* request_shape_values);

    std::map<int, TensorRTContext>::iterator GetMostOptimizedProfile(
        size_t total_batch_size,
        const std::vector<std::unique_ptr<InferenceRequest>>& requests,
        const std::map<int, std::vector<int32_t>>& request_shape_values);

    Status SetBindingDimensions(
        const std::string& input_name, const std::vector<int64_t>& shape,
        const TensorRTContext& trt_context, const size_t binding_idx,
        const size_t io_idx, std::vector<int64_t>* input_dims);
    Status SetCudaGraphShape(
        TensorRTContext* trt_context, const GraphSpec& graph_spec,
        std::vector<int64_t>* cuda_graph_key,
        TensorRTContext::CudaGraph* cuda_graph);
    void FindClosestCudaGraph(
        const TensorRTContext& trt_context,
        const std::vector<int64_t>& cuda_graph_key,
        const TensorRTContext::CudaGraph** cuda_graph, bool* found_exact);

    // The engine used for the context. If the model uses dynamic shape, then
    // the CUDA engine is owned by the context. Otherwise, the engine is shared
    // across all contexts and it must not be destroyed by the contexts.
    // In the future version of TensorRT, the engine may be shared even in the
    // dynamic shape case.
    nvinfer1::ICudaEngine* engine_;
    bool is_shared_engine_;

    // Additional CUDA stream to overlap copy and execution.
    cudaStream_t input_copy_stream_;

    // Use two sets of events each for current request and next request.
    CUDAEventSet events_[EVENT_SET_COUNT];
    size_t next_set_;

    // Completion thread for handling items in the corresponding completion
    // queue. One thread per instance so that the thread logic is simple as this
    // avoids busy-looping on different model executions' event states.
    std::thread completion_thread_;

    // The details needed by the completion thread to finalize the response for
    // a model execution.
    struct Payload {
      explicit Payload(
          InferenceBackend* inference_backend, size_t event_set_idx,
          std::vector<std::unique_ptr<InferenceRequest>>&& requests)
          : inference_backend_(inference_backend),
            event_set_idx_(event_set_idx), total_batch_size_(0),
            compute_start_ns_(0), requests_(std::move(requests))
      {
      }

      // The pointer to the backend handling the request
      InferenceBackend* inference_backend_;

      // The index to the event set handling the request
      size_t event_set_idx_;

      // The total batch size for the request
      size_t total_batch_size_;

      // The timestamps for reporting stats
      uint64_t compute_start_ns_;

      // All the composing InferenceRequest objects
      std::vector<std::unique_ptr<InferenceRequest>> requests_;
      // All the generated InferenceResponse objects
      std::vector<std::unique_ptr<InferenceResponse>> responses_;
      // Responder of the payload
      std::unique_ptr<BackendResponder> responder_;
    };

    // Assume that the lifetime of composing completion data to extend till
    // the responses are returned.
    SyncQueue<std::unique_ptr<Payload>> completion_queue_;

    // Map from profile index to the corresponding TensorRT context. Use map
    // to ensure each profile index is mapped to exactly one TensorRT context.
    std::map<int, TensorRTContext> trt_contexts_;

    // Is set true if the configuration supports batching
    bool support_batching_;

    // Is set true if the loaded model has one or more dynamic shaped inputs
    bool is_dynamic_;

    // Whether inexact match is allowed for finding CUDA graph
    bool allow_inexact_match_;

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
    std::vector<bool> buffer_is_ragged_;
    // Instructions on constructing the batch input and the CPU buffer for
    // storing mutable data
    using BatchInputData =
        std::pair<inference::BatchInput, std::unique_ptr<AllocatedMemory>>;
    std::vector<std::shared_ptr<BatchInputData>> batch_inputs_;
    // Store the pair of input name to look up and output shape
    // for output scattering
    std::vector<std::pair<std::string, std::vector<int64_t>>> io_shape_mapping_;

    // The pointer to the CUDA buffer for each binding index of the TensorRT
    // engine. This is used to match the TensorRT context execution declaration
    // while minimizing memory allocation.
    // The array size is equal to Context::total_bindings_
    std::vector<void*> buffer_bindings_;

    // The request details of the ongoing model execution
    std::unique_ptr<Payload> payload_;

    // map from binding_index to pair of index of full dims to
    // be padded and the padding offset.
    std::unordered_map<int, std::pair<int, int64_t>> padding_info_;
  };

  // CUDA engine shared across all model instances on the same device.
  std::map<int, std::pair<nvinfer1::IRuntime*, nvinfer1::ICudaEngine*>>
      device_engines_;

  // vector for storing available context queue associated with a runner
  std::vector<std::shared_ptr<SyncQueue<size_t>>> available_context_queue_;

  // Next context to be used for the runner.
  std::vector<size_t> next_context_;
};

std::ostream& operator<<(std::ostream& out, const PlanBackend& pb);

}}  // namespace nvidia::inferenceserver
