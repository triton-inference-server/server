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

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef __APPLE__
#include <Metal/Metal.h>
#include <dispatch/dispatch.h>
#endif

namespace triton { namespace metal {

// Forward declarations
class MetalCommandBuffer;
class MetalCommandQueue;
class MetalCommandPool;

// Error codes specific to Metal operations
enum class MetalErrorCode {
  SUCCESS = 0,
  DEVICE_NOT_FOUND,
  QUEUE_CREATION_FAILED,
  BUFFER_CREATION_FAILED,
  COMMAND_ENCODING_FAILED,
  SYNCHRONIZATION_FAILED,
  INVALID_OPERATION,
  OUT_OF_MEMORY,
  UNKNOWN_ERROR
};

// Metal execution result
struct MetalResult {
  MetalErrorCode code;
  std::string message;
  std::chrono::duration<double, std::milli> execution_time;
  
  bool IsSuccess() const { return code == MetalErrorCode::SUCCESS; }
  static MetalResult Success() { return {MetalErrorCode::SUCCESS, "", {}}; }
  static MetalResult Error(MetalErrorCode code, const std::string& msg) {
    return {code, msg, {}};
  }
};

// Execution completion callback
using CompletionCallback = std::function<void(const MetalResult&)>;

// Execution options
struct ExecutionOptions {
  bool synchronous = true;
  bool enable_profiling = false;
  CompletionCallback completion_callback = nullptr;
  // Priority levels matching Metal's QoS
  enum Priority {
    BACKGROUND = 0,
    UTILITY = 1,
    DEFAULT = 2,
    USER_INITIATED = 3,
    USER_INTERACTIVE = 4
  };
  Priority priority = Priority::DEFAULT;
  
  // Timeout for synchronous execution (0 = no timeout)
  std::chrono::milliseconds timeout{0};
};

// Command buffer state
enum class CommandBufferState {
  IDLE,
  ENCODING,
  COMMITTED,
  SCHEDULED,
  EXECUTING,
  COMPLETED,
  ERROR
};

// Metal command buffer wrapper
class MetalCommandBuffer {
 public:
  MetalCommandBuffer() = default;
  ~MetalCommandBuffer();
  
  // Disable copy, enable move
  MetalCommandBuffer(const MetalCommandBuffer&) = delete;
  MetalCommandBuffer& operator=(const MetalCommandBuffer&) = delete;
  MetalCommandBuffer(MetalCommandBuffer&&) noexcept;
  MetalCommandBuffer& operator=(MetalCommandBuffer&&) noexcept;
  
  // Command buffer lifecycle
  MetalResult Begin();
  MetalResult End();
  MetalResult Reset();
  
  // State query
  CommandBufferState GetState() const;
  bool IsRecording() const { return GetState() == CommandBufferState::ENCODING; }
  bool IsExecutable() const;
  
  // Kernel dispatch interface (to be extended in kernel implementation)
  MetalResult DispatchKernel(
      const std::string& kernel_name,
      const std::vector<void*>& arguments,
      const std::array<size_t, 3>& grid_size,
      const std::array<size_t, 3>& block_size);
  
  // Memory barrier
  MetalResult InsertMemoryBarrier();
  
  // Timing markers for profiling
  MetalResult PushDebugGroup(const std::string& label);
  MetalResult PopDebugGroup();
  
#ifdef __APPLE__
  id<MTLCommandBuffer> GetHandle() const { return command_buffer_; }
  void SetHandle(id<MTLCommandBuffer> buffer);
#endif

 private:
  friend class MetalCommandQueue;
  friend class MetalCommandPool;
  
#ifdef __APPLE__
  id<MTLCommandBuffer> command_buffer_ = nil;
#endif
  std::atomic<CommandBufferState> state_{CommandBufferState::IDLE};
  mutable std::mutex state_mutex_;
  bool is_pooled_ = false;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_time_;
};

// Metal command queue wrapper
class MetalCommandQueue {
 public:
  // Create command queue with specific device and options
  static std::unique_ptr<MetalCommandQueue> Create(
      int device_id = 0,
      ExecutionOptions::Priority priority = ExecutionOptions::Priority::DEFAULT);
  
  ~MetalCommandQueue();
  
  // Disable copy
  MetalCommandQueue(const MetalCommandQueue&) = delete;
  MetalCommandQueue& operator=(const MetalCommandQueue&) = delete;
  
  // Command buffer creation
  std::unique_ptr<MetalCommandBuffer> CreateCommandBuffer();
  
  // Command buffer pool management
  std::shared_ptr<MetalCommandBuffer> GetPooledCommandBuffer();
  void ReturnToPool(std::shared_ptr<MetalCommandBuffer> buffer);
  
  // Execution
  MetalResult Execute(MetalCommandBuffer& buffer, const ExecutionOptions& options = {});
  MetalResult ExecuteBatch(
      const std::vector<MetalCommandBuffer*>& buffers,
      const ExecutionOptions& options = {});
  
  // Synchronization
  MetalResult Synchronize();
  MetalResult WaitForCompletion(std::chrono::milliseconds timeout = std::chrono::milliseconds(0));
  
  // Queue properties
  int GetDeviceId() const { return device_id_; }
  size_t GetQueueDepth() const;
  bool IsIdle() const;
  
  // Performance metrics
  struct QueueMetrics {
    size_t total_executions;
    size_t failed_executions;
    double average_execution_time_ms;
    double peak_execution_time_ms;
    size_t active_buffers;
    size_t pooled_buffers;
  };
  QueueMetrics GetMetrics() const;
  void ResetMetrics();
  
#ifdef __APPLE__
  id<MTLCommandQueue> GetHandle() const { return command_queue_; }
  id<MTLDevice> GetDevice() const { return device_; }
#endif

 private:
  MetalCommandQueue(int device_id);
  MetalResult Initialize(ExecutionOptions::Priority priority);
  
  // Internal execution implementation
  MetalResult ExecuteInternal(
      MetalCommandBuffer& buffer,
      const ExecutionOptions& options,
      bool is_batch = false);
  
  // Pool management
  void InitializePool(size_t initial_size = 4);
  void CleanupPool();
  
#ifdef __APPLE__
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> command_queue_ = nil;
  dispatch_queue_t dispatch_queue_ = nullptr;
#endif
  
  int device_id_;
  mutable std::mutex queue_mutex_;
  mutable std::mutex metrics_mutex_;
  
  // Command buffer pool
  std::queue<std::shared_ptr<MetalCommandBuffer>> buffer_pool_;
  std::mutex pool_mutex_;
  size_t max_pool_size_ = 16;
  
  // Metrics
  std::atomic<size_t> total_executions_{0};
  std::atomic<size_t> failed_executions_{0};
  std::atomic<size_t> active_buffers_{0};
  double total_execution_time_ms_ = 0.0;
  double peak_execution_time_ms_ = 0.0;
  
  // Active command tracking
  std::unordered_map<MetalCommandBuffer*, CompletionCallback> active_commands_;
  std::mutex active_commands_mutex_;
};

// Command pool for managing multiple queues (multi-stream equivalent)
class MetalCommandPool {
 public:
  // Create pool with multiple queues
  static std::unique_ptr<MetalCommandPool> Create(
      int device_id = 0,
      size_t num_queues = 1,
      ExecutionOptions::Priority priority = ExecutionOptions::Priority::DEFAULT);
  
  ~MetalCommandPool() = default;
  
  // Get queue by index (round-robin or specific)
  MetalCommandQueue* GetQueue(size_t index = SIZE_MAX);
  
  // Execute on next available queue
  MetalResult Execute(MetalCommandBuffer& buffer, const ExecutionOptions& options = {});
  
  // Synchronize all queues
  MetalResult SynchronizeAll();
  
  // Pool metrics
  struct PoolMetrics {
    size_t num_queues;
    std::vector<MetalCommandQueue::QueueMetrics> queue_metrics;
    size_t total_executions;
    double average_queue_utilization;
  };
  PoolMetrics GetMetrics() const;
  
 private:
  MetalCommandPool() = default;
  
  std::vector<std::unique_ptr<MetalCommandQueue>> queues_;
  std::atomic<size_t> next_queue_{0};
  mutable std::mutex pool_mutex_;
};

// Utility functions
std::vector<int> GetAvailableDevices();
std::string GetDeviceName(int device_id);
size_t GetDeviceMemory(int device_id);
bool IsDeviceAvailable(int device_id);

// Thread-local command queue for optimal performance
MetalCommandQueue* GetThreadLocalQueue(int device_id = 0);

}}  // namespace triton::metal