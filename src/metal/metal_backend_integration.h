// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "metal_command.h"
#include <memory>
#include <string>
#include <vector>

namespace triton { namespace metal {

// Forward declarations
class MetalKernelLibrary;
class MetalBuffer;

// Backend execution context for Metal
class MetalBackendContext {
 public:
  // Create context with specified device and configuration
  static std::unique_ptr<MetalBackendContext> Create(
      int device_id = 0,
      size_t num_streams = 1,
      bool enable_profiling = false);
  
  ~MetalBackendContext() = default;
  
  // Stream (queue) selection
  MetalCommandQueue* GetStream(size_t stream_id = 0);
  MetalCommandQueue* GetDefaultStream() { return GetStream(0); }
  
  // Command buffer management
  std::shared_ptr<MetalCommandBuffer> GetCommandBuffer(size_t stream_id = 0);
  void ReturnCommandBuffer(std::shared_ptr<MetalCommandBuffer> buffer, size_t stream_id = 0);
  
  // Synchronization
  MetalResult SynchronizeStream(size_t stream_id);
  MetalResult SynchronizeAllStreams();
  
  // Device properties
  int GetDeviceId() const { return device_id_; }
  size_t GetNumStreams() const { return command_pool_ ? num_streams_ : 0; }
  bool IsProfilingEnabled() const { return enable_profiling_; }
  
  // Performance metrics
  struct ContextMetrics {
    MetalCommandPool::PoolMetrics pool_metrics;
    size_t total_buffers_created;
    size_t total_buffers_reused;
    double total_execution_time_ms;
  };
  ContextMetrics GetMetrics() const;
  
 private:
  MetalBackendContext(int device_id, size_t num_streams, bool enable_profiling);
  MetalResult Initialize();
  
  int device_id_;
  size_t num_streams_;
  bool enable_profiling_;
  
  std::unique_ptr<MetalCommandPool> command_pool_;
  mutable std::mutex metrics_mutex_;
  
  // Metrics
  size_t total_buffers_created_ = 0;
  size_t total_buffers_reused_ = 0;
  double total_execution_time_ms_ = 0.0;
};

// Inference request executor for Metal backend
class MetalInferenceExecutor {
 public:
  MetalInferenceExecutor(MetalBackendContext* context);
  ~MetalInferenceExecutor() = default;
  
  // Execution stages
  MetalResult BeginInference(size_t stream_id = 0);
  MetalResult EndInference();
  
  // Kernel dispatch helpers
  MetalResult DispatchPreprocessing(
      const std::vector<MetalBuffer*>& inputs,
      const std::vector<MetalBuffer*>& outputs);
  
  MetalResult DispatchInference(
      const std::string& model_kernel,
      const std::vector<MetalBuffer*>& inputs,
      const std::vector<MetalBuffer*>& outputs,
      const std::vector<size_t>& dimensions);
  
  MetalResult DispatchPostprocessing(
      const std::vector<MetalBuffer*>& inputs,
      const std::vector<MetalBuffer*>& outputs);
  
  // Synchronous execution
  MetalResult Execute(bool wait_for_completion = true);
  
  // Asynchronous execution with callback
  MetalResult ExecuteAsync(CompletionCallback callback);
  
  // Get timing information (only valid after completion)
  double GetExecutionTimeMs() const { return last_execution_time_ms_; }
  
 private:
  MetalBackendContext* context_;
  size_t current_stream_id_ = 0;
  std::shared_ptr<MetalCommandBuffer> current_buffer_;
  bool is_recording_ = false;
  double last_execution_time_ms_ = 0.0;
};

// Utility class for managing multiple concurrent inferences
class MetalInferenceScheduler {
 public:
  MetalInferenceScheduler(MetalBackendContext* context);
  ~MetalInferenceScheduler() = default;
  
  // Schedule inference request
  struct InferenceRequest {
    std::string request_id;
    std::string model_name;
    std::vector<MetalBuffer*> inputs;
    std::vector<MetalBuffer*> outputs;
    CompletionCallback callback;
    ExecutionOptions::Priority priority = ExecutionOptions::Priority::DEFAULT;
  };
  
  MetalResult ScheduleInference(const InferenceRequest& request);
  
  // Wait for specific request or all requests
  MetalResult WaitForRequest(const std::string& request_id, 
                            std::chrono::milliseconds timeout = std::chrono::milliseconds(0));
  MetalResult WaitForAll(std::chrono::milliseconds timeout = std::chrono::milliseconds(0));
  
  // Get scheduler statistics
  struct SchedulerStats {
    size_t pending_requests;
    size_t active_requests;
    size_t completed_requests;
    double average_latency_ms;
    double p99_latency_ms;
  };
  SchedulerStats GetStats() const;
  
 private:
  struct RequestState {
    InferenceRequest request;
    MetalInferenceExecutor executor;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::atomic<bool> completed{false};
  };
  
  MetalBackendContext* context_;
  std::unordered_map<std::string, std::unique_ptr<RequestState>> active_requests_;
  mutable std::mutex requests_mutex_;
  std::condition_variable completion_cv_;
  
  // Statistics
  std::atomic<size_t> completed_count_{0};
  std::vector<double> latency_history_;
  mutable std::mutex stats_mutex_;
};

// Helper functions for common patterns
MetalResult ExecuteSingleKernel(
    MetalBackendContext* context,
    const std::string& kernel_name,
    const std::vector<void*>& arguments,
    const std::array<size_t, 3>& grid_size,
    const std::array<size_t, 3>& block_size,
    size_t stream_id = 0);

MetalResult ExecuteKernelSequence(
    MetalBackendContext* context,
    const std::vector<std::pair<std::string, std::vector<void*>>>& kernels,
    size_t stream_id = 0);

}}  // namespace triton::metal