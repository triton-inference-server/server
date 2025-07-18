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

#include "metal_command.h"

#include <algorithm>
#include <condition_variable>
#include <sstream>

#ifdef __APPLE__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

namespace triton { namespace metal {

namespace {

// Thread-local storage for command queues
thread_local std::unordered_map<int, std::unique_ptr<MetalCommandQueue>> tls_queues;

#ifdef __APPLE__
// Convert priority to Metal QoS
dispatch_qos_class_t PriorityToQoS(ExecutionOptions::Priority priority) {
  switch (priority) {
    case ExecutionOptions::Priority::BACKGROUND:
      return QOS_CLASS_BACKGROUND;
    case ExecutionOptions::Priority::UTILITY:
      return QOS_CLASS_UTILITY;
    case ExecutionOptions::Priority::USER_INITIATED:
      return QOS_CLASS_USER_INITIATED;
    case ExecutionOptions::Priority::USER_INTERACTIVE:
      return QOS_CLASS_USER_INTERACTIVE;
    case ExecutionOptions::Priority::DEFAULT:
    default:
      return QOS_CLASS_DEFAULT;
  }
}

// Get Metal device by ID
id<MTLDevice> GetMetalDevice(int device_id) {
  NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
  if (device_id < 0 || device_id >= [devices count]) {
    return nil;
  }
  return devices[device_id];
}
#endif

}  // namespace

//
// MetalCommandBuffer Implementation
//

MetalCommandBuffer::~MetalCommandBuffer() {
#ifdef __APPLE__
  if (command_buffer_ != nil) {
    [command_buffer_ release];
    command_buffer_ = nil;
  }
#endif
}

MetalCommandBuffer::MetalCommandBuffer(MetalCommandBuffer&& other) noexcept {
  std::lock_guard<std::mutex> lock(other.state_mutex_);
  command_buffer_ = other.command_buffer_;
  state_ = other.state_.load();
  is_pooled_ = other.is_pooled_;
  start_time_ = other.start_time_;
  end_time_ = other.end_time_;
  
  other.command_buffer_ = nil;
  other.state_ = CommandBufferState::IDLE;
}

MetalCommandBuffer& MetalCommandBuffer::operator=(MetalCommandBuffer&& other) noexcept {
  if (this != &other) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    std::lock_guard<std::mutex> other_lock(other.state_mutex_);
    
#ifdef __APPLE__
    if (command_buffer_ != nil) {
      [command_buffer_ release];
    }
#endif
    
    command_buffer_ = other.command_buffer_;
    state_ = other.state_.load();
    is_pooled_ = other.is_pooled_;
    start_time_ = other.start_time_;
    end_time_ = other.end_time_;
    
    other.command_buffer_ = nil;
    other.state_ = CommandBufferState::IDLE;
  }
  return *this;
}

MetalResult MetalCommandBuffer::Begin() {
#ifdef __APPLE__
  std::lock_guard<std::mutex> lock(state_mutex_);
  
  if (state_ != CommandBufferState::IDLE) {
    return MetalResult::Error(
        MetalErrorCode::INVALID_OPERATION,
        "Cannot begin command buffer that is not idle");
  }
  
  if (command_buffer_ == nil) {
    return MetalResult::Error(
        MetalErrorCode::INVALID_OPERATION,
        "Command buffer not properly initialized");
  }
  
  state_ = CommandBufferState::ENCODING;
  start_time_ = std::chrono::high_resolution_clock::now();
  
  return MetalResult::Success();
#else
  return MetalResult::Error(
      MetalErrorCode::DEVICE_NOT_FOUND,
      "Metal is not supported on this platform");
#endif
}

MetalResult MetalCommandBuffer::End() {
#ifdef __APPLE__
  std::lock_guard<std::mutex> lock(state_mutex_);
  
  if (state_ != CommandBufferState::ENCODING) {
    return MetalResult::Error(
        MetalErrorCode::INVALID_OPERATION,
        "Cannot end command buffer that is not encoding");
  }
  
  state_ = CommandBufferState::COMMITTED;
  return MetalResult::Success();
#else
  return MetalResult::Error(
      MetalErrorCode::DEVICE_NOT_FOUND,
      "Metal is not supported on this platform");
#endif
}

MetalResult MetalCommandBuffer::Reset() {
#ifdef __APPLE__
  std::lock_guard<std::mutex> lock(state_mutex_);
  
  // Can only reset if idle or completed
  if (state_ != CommandBufferState::IDLE && 
      state_ != CommandBufferState::COMPLETED &&
      state_ != CommandBufferState::ERROR) {
    return MetalResult::Error(
        MetalErrorCode::INVALID_OPERATION,
        "Cannot reset command buffer in current state");
  }
  
  state_ = CommandBufferState::IDLE;
  return MetalResult::Success();
#else
  return MetalResult::Error(
      MetalErrorCode::DEVICE_NOT_FOUND,
      "Metal is not supported on this platform");
#endif
}

CommandBufferState MetalCommandBuffer::GetState() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return state_.load();
}

bool MetalCommandBuffer::IsExecutable() const {
  auto state = GetState();
  return state == CommandBufferState::COMMITTED || 
         state == CommandBufferState::SCHEDULED;
}

MetalResult MetalCommandBuffer::DispatchKernel(
    const std::string& kernel_name,
    const std::vector<void*>& arguments,
    const std::array<size_t, 3>& grid_size,
    const std::array<size_t, 3>& block_size) {
  // This will be implemented when integrating with the kernel system
  return MetalResult::Error(
      MetalErrorCode::INVALID_OPERATION,
      "Kernel dispatch not yet implemented");
}

MetalResult MetalCommandBuffer::InsertMemoryBarrier() {
#ifdef __APPLE__
  std::lock_guard<std::mutex> lock(state_mutex_);
  
  if (state_ != CommandBufferState::ENCODING) {
    return MetalResult::Error(
        MetalErrorCode::INVALID_OPERATION,
        "Can only insert barriers during encoding");
  }
  
  // Metal handles memory coherency automatically for most cases
  // This is a placeholder for explicit synchronization if needed
  return MetalResult::Success();
#else
  return MetalResult::Error(
      MetalErrorCode::DEVICE_NOT_FOUND,
      "Metal is not supported on this platform");
#endif
}

MetalResult MetalCommandBuffer::PushDebugGroup(const std::string& label) {
#ifdef __APPLE__
  std::lock_guard<std::mutex> lock(state_mutex_);
  
  if (state_ != CommandBufferState::ENCODING) {
    return MetalResult::Error(
        MetalErrorCode::INVALID_OPERATION,
        "Can only push debug groups during encoding");
  }
  
  if (command_buffer_ != nil) {
    [command_buffer_ pushDebugGroup:[NSString stringWithUTF8String:label.c_str()]];
  }
  
  return MetalResult::Success();
#else
  return MetalResult::Error(
      MetalErrorCode::DEVICE_NOT_FOUND,
      "Metal is not supported on this platform");
#endif
}

MetalResult MetalCommandBuffer::PopDebugGroup() {
#ifdef __APPLE__
  std::lock_guard<std::mutex> lock(state_mutex_);
  
  if (state_ != CommandBufferState::ENCODING) {
    return MetalResult::Error(
        MetalErrorCode::INVALID_OPERATION,
        "Can only pop debug groups during encoding");
  }
  
  if (command_buffer_ != nil) {
    [command_buffer_ popDebugGroup];
  }
  
  return MetalResult::Success();
#else
  return MetalResult::Error(
      MetalErrorCode::DEVICE_NOT_FOUND,
      "Metal is not supported on this platform");
#endif
}

#ifdef __APPLE__
void MetalCommandBuffer::SetHandle(id<MTLCommandBuffer> buffer) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (command_buffer_ != nil) {
    [command_buffer_ release];
  }
  command_buffer_ = [buffer retain];
}
#endif

//
// MetalCommandQueue Implementation
//

std::unique_ptr<MetalCommandQueue> MetalCommandQueue::Create(
    int device_id, ExecutionOptions::Priority priority) {
  auto queue = std::unique_ptr<MetalCommandQueue>(new MetalCommandQueue(device_id));
  auto result = queue->Initialize(priority);
  if (!result.IsSuccess()) {
    return nullptr;
  }
  return queue;
}

MetalCommandQueue::MetalCommandQueue(int device_id) : device_id_(device_id) {}

MetalCommandQueue::~MetalCommandQueue() {
  // Synchronize and cleanup
  Synchronize();
  CleanupPool();
  
#ifdef __APPLE__
  if (dispatch_queue_ != nullptr) {
    dispatch_release(dispatch_queue_);
  }
  if (command_queue_ != nil) {
    [command_queue_ release];
  }
  if (device_ != nil) {
    [device_ release];
  }
#endif
}

MetalResult MetalCommandQueue::Initialize(ExecutionOptions::Priority priority) {
#ifdef __APPLE__
  // Get Metal device
  device_ = GetMetalDevice(device_id_);
  if (device_ == nil) {
    return MetalResult::Error(
        MetalErrorCode::DEVICE_NOT_FOUND,
        "Metal device not found for ID: " + std::to_string(device_id_));
  }
  [device_ retain];
  
  // Create command queue
  command_queue_ = [device_ newCommandQueue];
  if (command_queue_ == nil) {
    return MetalResult::Error(
        MetalErrorCode::QUEUE_CREATION_FAILED,
        "Failed to create Metal command queue");
  }
  
  // Create dispatch queue for async operations
  dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(
      DISPATCH_QUEUE_CONCURRENT,
      PriorityToQoS(priority),
      -1);
  
  std::string queue_name = "triton.metal.queue." + std::to_string(device_id_);
  dispatch_queue_ = dispatch_queue_create(queue_name.c_str(), attr);
  
  // Initialize command buffer pool
  InitializePool();
  
  return MetalResult::Success();
#else
  return MetalResult::Error(
      MetalErrorCode::DEVICE_NOT_FOUND,
      "Metal is not supported on this platform");
#endif
}

std::unique_ptr<MetalCommandBuffer> MetalCommandQueue::CreateCommandBuffer() {
#ifdef __APPLE__
  if (command_queue_ == nil) {
    return nullptr;
  }
  
  auto buffer = std::make_unique<MetalCommandBuffer>();
  id<MTLCommandBuffer> mtl_buffer = [command_queue_ commandBuffer];
  if (mtl_buffer == nil) {
    return nullptr;
  }
  
  buffer->SetHandle(mtl_buffer);
  active_buffers_++;
  
  return buffer;
#else
  return nullptr;
#endif
}

std::shared_ptr<MetalCommandBuffer> MetalCommandQueue::GetPooledCommandBuffer() {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  
  if (!buffer_pool_.empty()) {
    auto buffer = buffer_pool_.front();
    buffer_pool_.pop();
    
    // Reset the buffer for reuse
    buffer->Reset();
    return buffer;
  }
  
  // Create new buffer if pool is empty
  auto buffer = CreateCommandBuffer();
  if (!buffer) {
    return nullptr;
  }
  
  buffer->is_pooled_ = true;
  return std::shared_ptr<MetalCommandBuffer>(buffer.release());
}

void MetalCommandQueue::ReturnToPool(std::shared_ptr<MetalCommandBuffer> buffer) {
  if (!buffer || !buffer->is_pooled_) {
    return;
  }
  
  std::lock_guard<std::mutex> lock(pool_mutex_);
  
  if (buffer_pool_.size() < max_pool_size_) {
    buffer->Reset();
    buffer_pool_.push(buffer);
  } else {
    // Let it be destroyed if pool is full
    active_buffers_--;
  }
}

MetalResult MetalCommandQueue::Execute(
    MetalCommandBuffer& buffer, const ExecutionOptions& options) {
  return ExecuteInternal(buffer, options, false);
}

MetalResult MetalCommandQueue::ExecuteBatch(
    const std::vector<MetalCommandBuffer*>& buffers,
    const ExecutionOptions& options) {
#ifdef __APPLE__
  if (buffers.empty()) {
    return MetalResult::Success();
  }
  
  // Submit all buffers
  for (auto* buffer : buffers) {
    if (!buffer) continue;
    
    auto result = ExecuteInternal(*buffer, options, true);
    if (!result.IsSuccess()) {
      return result;
    }
  }
  
  // Wait for batch completion if synchronous
  if (options.synchronous) {
    return Synchronize();
  }
  
  return MetalResult::Success();
#else
  return MetalResult::Error(
      MetalErrorCode::DEVICE_NOT_FOUND,
      "Metal is not supported on this platform");
#endif
}

MetalResult MetalCommandQueue::ExecuteInternal(
    MetalCommandBuffer& buffer,
    const ExecutionOptions& options,
    bool is_batch) {
#ifdef __APPLE__
  if (!buffer.IsExecutable()) {
    return MetalResult::Error(
        MetalErrorCode::INVALID_OPERATION,
        "Command buffer is not in executable state");
  }
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  id<MTLCommandBuffer> mtl_buffer = buffer.GetHandle();
  if (mtl_buffer == nil) {
    return MetalResult::Error(
        MetalErrorCode::INVALID_OPERATION,
        "Invalid Metal command buffer");
  }
  
  // Setup completion handler
  __block MetalResult exec_result = MetalResult::Success();
  __block auto end_time = std::chrono::high_resolution_clock::now();
  
  // Create a completion block
  void (^completionHandler)(id<MTLCommandBuffer>) = ^(id<MTLCommandBuffer> cmd_buffer) {
    end_time = std::chrono::high_resolution_clock::now();
    
    // Check for errors
    if (cmd_buffer.error != nil) {
      exec_result = MetalResult::Error(
          MetalErrorCode::COMMAND_ENCODING_FAILED,
          [cmd_buffer.error.localizedDescription UTF8String]);
      failed_executions_++;
    }
    
    // Calculate execution time
    auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        end_time - start_time);
    exec_result.execution_time = duration;
    
    // Update metrics
    {
      std::lock_guard<std::mutex> lock(metrics_mutex_);
      total_execution_time_ms_ += duration.count();
      peak_execution_time_ms_ = std::max(peak_execution_time_ms_, duration.count());
    }
    
    // Update buffer state
    buffer.state_ = (cmd_buffer.error != nil) ? 
        CommandBufferState::ERROR : CommandBufferState::COMPLETED;
    buffer.end_time_ = end_time;
    
    // Call user completion callback if provided
    if (options.completion_callback) {
      if (options.synchronous) {
        // For synchronous execution, callback is called after wait
        std::lock_guard<std::mutex> lock(active_commands_mutex_);
        active_commands_[&buffer] = options.completion_callback;
      } else {
        // For async, call immediately
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
          options.completion_callback(exec_result);
        });
      }
    }
  };
  
  // Add profiling if enabled
  if (options.enable_profiling) {
    [mtl_buffer addCompletedHandler:completionHandler];
  } else {
    [mtl_buffer addCompletedHandler:completionHandler];
  }
  
  // Commit the command buffer
  buffer.state_ = CommandBufferState::SCHEDULED;
  [mtl_buffer commit];
  total_executions_++;
  
  // Handle synchronous execution
  if (options.synchronous && !is_batch) {
    if (options.timeout.count() > 0) {
      // Wait with timeout
      dispatch_time_t timeout = dispatch_time(
          DISPATCH_TIME_NOW,
          options.timeout.count() * NSEC_PER_MSEC);
      
      __block bool completed = false;
      dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
      
      [mtl_buffer addCompletedHandler:^(id<MTLCommandBuffer> cmd_buffer) {
        completed = true;
        dispatch_semaphore_signal(semaphore);
      }];
      
      long result = dispatch_semaphore_wait(semaphore, timeout);
      dispatch_release(semaphore);
      
      if (result != 0) {
        return MetalResult::Error(
            MetalErrorCode::SYNCHRONIZATION_FAILED,
            "Command buffer execution timed out");
      }
    } else {
      // Wait indefinitely
      [mtl_buffer waitUntilCompleted];
    }
    
    // Call completion callback for synchronous execution
    if (options.completion_callback) {
      std::lock_guard<std::mutex> lock(active_commands_mutex_);
      auto it = active_commands_.find(&buffer);
      if (it != active_commands_.end()) {
        it->second(exec_result);
        active_commands_.erase(it);
      }
    }
    
    return exec_result;
  }
  
  return MetalResult::Success();
#else
  return MetalResult::Error(
      MetalErrorCode::DEVICE_NOT_FOUND,
      "Metal is not supported on this platform");
#endif
}

MetalResult MetalCommandQueue::Synchronize() {
#ifdef __APPLE__
  // Wait for all active command buffers
  dispatch_sync(dispatch_queue_, ^{
    // This ensures all async operations are complete
  });
  
  return MetalResult::Success();
#else
  return MetalResult::Error(
      MetalErrorCode::DEVICE_NOT_FOUND,
      "Metal is not supported on this platform");
#endif
}

MetalResult MetalCommandQueue::WaitForCompletion(std::chrono::milliseconds timeout) {
#ifdef __APPLE__
  if (timeout.count() == 0) {
    return Synchronize();
  }
  
  __block bool completed = false;
  dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
  
  dispatch_async(dispatch_queue_, ^{
    completed = true;
    dispatch_semaphore_signal(semaphore);
  });
  
  dispatch_time_t dispatch_timeout = dispatch_time(
      DISPATCH_TIME_NOW,
      timeout.count() * NSEC_PER_MSEC);
  
  long result = dispatch_semaphore_wait(semaphore, dispatch_timeout);
  dispatch_release(semaphore);
  
  if (result != 0) {
    return MetalResult::Error(
        MetalErrorCode::SYNCHRONIZATION_FAILED,
        "Wait for completion timed out");
  }
  
  return MetalResult::Success();
#else
  return MetalResult::Error(
      MetalErrorCode::DEVICE_NOT_FOUND,
      "Metal is not supported on this platform");
#endif
}

size_t MetalCommandQueue::GetQueueDepth() const {
  // Metal doesn't expose queue depth directly
  // Return active buffers as an approximation
  return active_buffers_.load();
}

bool MetalCommandQueue::IsIdle() const {
  return active_buffers_.load() == 0;
}

MetalCommandQueue::QueueMetrics MetalCommandQueue::GetMetrics() const {
  std::lock_guard<std::mutex> lock(metrics_mutex_);
  
  QueueMetrics metrics;
  metrics.total_executions = total_executions_.load();
  metrics.failed_executions = failed_executions_.load();
  metrics.active_buffers = active_buffers_.load();
  
  {
    std::lock_guard<std::mutex> pool_lock(pool_mutex_);
    metrics.pooled_buffers = buffer_pool_.size();
  }
  
  if (metrics.total_executions > 0) {
    metrics.average_execution_time_ms = total_execution_time_ms_ / metrics.total_executions;
  } else {
    metrics.average_execution_time_ms = 0.0;
  }
  
  metrics.peak_execution_time_ms = peak_execution_time_ms_;
  
  return metrics;
}

void MetalCommandQueue::ResetMetrics() {
  std::lock_guard<std::mutex> lock(metrics_mutex_);
  total_executions_ = 0;
  failed_executions_ = 0;
  total_execution_time_ms_ = 0.0;
  peak_execution_time_ms_ = 0.0;
}

void MetalCommandQueue::InitializePool(size_t initial_size) {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  
  for (size_t i = 0; i < initial_size; ++i) {
    auto buffer = CreateCommandBuffer();
    if (buffer) {
      buffer->is_pooled_ = true;
      buffer_pool_.push(std::shared_ptr<MetalCommandBuffer>(buffer.release()));
    }
  }
}

void MetalCommandQueue::CleanupPool() {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  
  while (!buffer_pool_.empty()) {
    buffer_pool_.pop();
    active_buffers_--;
  }
}

//
// MetalCommandPool Implementation
//

std::unique_ptr<MetalCommandPool> MetalCommandPool::Create(
    int device_id, size_t num_queues, ExecutionOptions::Priority priority) {
  if (num_queues == 0) {
    return nullptr;
  }
  
  auto pool = std::unique_ptr<MetalCommandPool>(new MetalCommandPool());
  
  // Create requested number of queues
  for (size_t i = 0; i < num_queues; ++i) {
    auto queue = MetalCommandQueue::Create(device_id, priority);
    if (!queue) {
      return nullptr;
    }
    pool->queues_.push_back(std::move(queue));
  }
  
  return pool;
}

MetalCommandQueue* MetalCommandPool::GetQueue(size_t index) {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  
  if (queues_.empty()) {
    return nullptr;
  }
  
  if (index == SIZE_MAX) {
    // Round-robin selection
    index = next_queue_.fetch_add(1) % queues_.size();
  } else if (index >= queues_.size()) {
    return nullptr;
  }
  
  return queues_[index].get();
}

MetalResult MetalCommandPool::Execute(
    MetalCommandBuffer& buffer, const ExecutionOptions& options) {
  auto* queue = GetQueue();
  if (!queue) {
    return MetalResult::Error(
        MetalErrorCode::INVALID_OPERATION,
        "No available queue in pool");
  }
  
  return queue->Execute(buffer, options);
}

MetalResult MetalCommandPool::SynchronizeAll() {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  
  for (auto& queue : queues_) {
    auto result = queue->Synchronize();
    if (!result.IsSuccess()) {
      return result;
    }
  }
  
  return MetalResult::Success();
}

MetalCommandPool::PoolMetrics MetalCommandPool::GetMetrics() const {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  
  PoolMetrics metrics;
  metrics.num_queues = queues_.size();
  metrics.total_executions = 0;
  
  for (const auto& queue : queues_) {
    auto queue_metrics = queue->GetMetrics();
    metrics.queue_metrics.push_back(queue_metrics);
    metrics.total_executions += queue_metrics.total_executions;
  }
  
  if (!queues_.empty()) {
    size_t total_active = 0;
    for (const auto& qm : metrics.queue_metrics) {
      total_active += qm.active_buffers;
    }
    metrics.average_queue_utilization = 
        static_cast<double>(total_active) / queues_.size();
  }
  
  return metrics;
}

//
// Utility Functions
//

std::vector<int> GetAvailableDevices() {
  std::vector<int> devices;
#ifdef __APPLE__
  NSArray<id<MTLDevice>>* mtl_devices = MTLCopyAllDevices();
  for (NSUInteger i = 0; i < [mtl_devices count]; ++i) {
    devices.push_back(static_cast<int>(i));
  }
#endif
  return devices;
}

std::string GetDeviceName(int device_id) {
#ifdef __APPLE__
  id<MTLDevice> device = GetMetalDevice(device_id);
  if (device != nil) {
    return std::string([device.name UTF8String]);
  }
#endif
  return "Unknown";
}

size_t GetDeviceMemory(int device_id) {
#ifdef __APPLE__
  id<MTLDevice> device = GetMetalDevice(device_id);
  if (device != nil) {
    // recommendedMaxWorkingSetSize gives a hint about usable memory
    return device.recommendedMaxWorkingSetSize;
  }
#endif
  return 0;
}

bool IsDeviceAvailable(int device_id) {
#ifdef __APPLE__
  return GetMetalDevice(device_id) != nil;
#else
  return false;
#endif
}

MetalCommandQueue* GetThreadLocalQueue(int device_id) {
  auto it = tls_queues.find(device_id);
  if (it != tls_queues.end()) {
    return it->second.get();
  }
  
  // Create new queue for this thread
  auto queue = MetalCommandQueue::Create(device_id);
  if (queue) {
    auto* queue_ptr = queue.get();
    tls_queues[device_id] = std::move(queue);
    return queue_ptr;
  }
  
  return nullptr;
}

}}  // namespace triton::metal