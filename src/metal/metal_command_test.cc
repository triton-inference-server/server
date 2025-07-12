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

#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using namespace triton::metal;

// Example 1: Basic synchronous execution
void TestSynchronousExecution() {
  std::cout << "=== Test: Synchronous Execution ===" << std::endl;
  
  // Create command queue
  auto queue = MetalCommandQueue::Create(0);
  if (!queue) {
    std::cout << "Failed to create Metal command queue" << std::endl;
    return;
  }
  
  // Create command buffer
  auto buffer = queue->CreateCommandBuffer();
  if (!buffer) {
    std::cout << "Failed to create command buffer" << std::endl;
    return;
  }
  
  // Begin recording
  auto result = buffer->Begin();
  if (!result.IsSuccess()) {
    std::cout << "Failed to begin command buffer: " << result.message << std::endl;
    return;
  }
  
  // Add debug marker
  buffer->PushDebugGroup("Test Kernel");
  
  // Add a memory barrier to test command encoding
  auto result = buffer->InsertMemoryBarrier();
  if (!result.IsSuccess()) {
    std::cout << "Failed to insert memory barrier: " << result.message << std::endl;
  }
  
  buffer->PopDebugGroup();
  
  // End recording
  result = buffer->End();
  if (!result.IsSuccess()) {
    std::cout << "Failed to end command buffer: " << result.message << std::endl;
    return;
  }
  
  // Execute synchronously
  ExecutionOptions options;
  options.synchronous = true;
  options.enable_profiling = true;
  
  result = queue->Execute(*buffer, options);
  if (result.IsSuccess()) {
    std::cout << "Execution completed in " 
              << result.execution_time.count() << " ms" << std::endl;
  } else {
    std::cout << "Execution failed: " << result.message << std::endl;
  }
  
  // Print queue metrics
  auto metrics = queue->GetMetrics();
  std::cout << "Queue metrics:" << std::endl;
  std::cout << "  Total executions: " << metrics.total_executions << std::endl;
  std::cout << "  Failed executions: " << metrics.failed_executions << std::endl;
  std::cout << "  Average execution time: " << metrics.average_execution_time_ms << " ms" << std::endl;
}

// Example 2: Asynchronous execution with callbacks
void TestAsynchronousExecution() {
  std::cout << "\n=== Test: Asynchronous Execution ===" << std::endl;
  
  auto queue = MetalCommandQueue::Create(0);
  if (!queue) {
    std::cout << "Failed to create Metal command queue" << std::endl;
    return;
  }
  
  // Create multiple command buffers
  const int num_buffers = 5;
  std::vector<std::unique_ptr<MetalCommandBuffer>> buffers;
  
  for (int i = 0; i < num_buffers; ++i) {
    auto buffer = queue->CreateCommandBuffer();
    if (buffer) {
      buffer->Begin();
      // Add memory barrier as test command
      buffer->InsertMemoryBarrier();
      buffer->End();
      buffers.push_back(std::move(buffer));
    }
  }
  
  // Execute asynchronously with completion callbacks
  std::atomic<int> completed_count(0);
  std::condition_variable cv;
  std::mutex cv_mutex;
  
  ExecutionOptions options;
  options.synchronous = false;
  options.enable_profiling = true;
  options.completion_callback = [&](const MetalResult& result) {
    if (result.IsSuccess()) {
      std::cout << "Async execution completed in " 
                << result.execution_time.count() << " ms" << std::endl;
    } else {
      std::cout << "Async execution failed: " << result.message << std::endl;
    }
    
    completed_count++;
    cv.notify_one();
  };
  
  // Submit all buffers
  for (auto& buffer : buffers) {
    queue->Execute(*buffer, options);
  }
  
  // Wait for all completions
  std::unique_lock<std::mutex> lock(cv_mutex);
  cv.wait(lock, [&] { return completed_count.load() == num_buffers; });
  
  std::cout << "All async executions completed" << std::endl;
}

// Example 3: Command buffer pooling
void TestCommandBufferPooling() {
  std::cout << "\n=== Test: Command Buffer Pooling ===" << std::endl;
  
  auto queue = MetalCommandQueue::Create(0);
  if (!queue) {
    std::cout << "Failed to create Metal command queue" << std::endl;
    return;
  }
  
  // Get pooled buffers
  std::vector<std::shared_ptr<MetalCommandBuffer>> pooled_buffers;
  
  for (int i = 0; i < 3; ++i) {
    auto buffer = queue->GetPooledCommandBuffer();
    if (buffer) {
      buffer->Begin();
      // Add memory barrier as test command
      buffer->InsertMemoryBarrier();
      buffer->End();
      pooled_buffers.push_back(buffer);
    }
  }
  
  // Execute and return to pool
  ExecutionOptions options;
  options.synchronous = true;
  
  for (auto& buffer : pooled_buffers) {
    auto result = queue->Execute(*buffer, options);
    if (result.IsSuccess()) {
      std::cout << "Pooled buffer executed successfully" << std::endl;
    }
    
    // Return to pool for reuse
    queue->ReturnToPool(buffer);
  }
  
  // Verify pool reuse
  auto reused_buffer = queue->GetPooledCommandBuffer();
  if (reused_buffer) {
    std::cout << "Successfully retrieved reused buffer from pool" << std::endl;
  }
}

// Example 4: Batch execution
void TestBatchExecution() {
  std::cout << "\n=== Test: Batch Execution ===" << std::endl;
  
  auto queue = MetalCommandQueue::Create(0);
  if (!queue) {
    std::cout << "Failed to create Metal command queue" << std::endl;
    return;
  }
  
  // Create batch of command buffers
  std::vector<std::unique_ptr<MetalCommandBuffer>> buffers;
  std::vector<MetalCommandBuffer*> buffer_ptrs;
  
  for (int i = 0; i < 10; ++i) {
    auto buffer = queue->CreateCommandBuffer();
    if (buffer) {
      buffer->Begin();
      buffer->PushDebugGroup("Batch " + std::to_string(i));
      // Add memory barrier as test command
      buffer->InsertMemoryBarrier();
      buffer->PopDebugGroup();
      buffer->End();
      
      buffer_ptrs.push_back(buffer.get());
      buffers.push_back(std::move(buffer));
    }
  }
  
  // Execute as batch
  ExecutionOptions options;
  options.synchronous = true;
  options.enable_profiling = true;
  
  auto start = std::chrono::high_resolution_clock::now();
  auto result = queue->ExecuteBatch(buffer_ptrs, options);
  auto end = std::chrono::high_resolution_clock::now();
  
  if (result.IsSuccess()) {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Batch execution completed in " << duration.count() << " ms" << std::endl;
  } else {
    std::cout << "Batch execution failed: " << result.message << std::endl;
  }
}

// Example 5: Multi-queue (multi-stream) execution
void TestMultiQueueExecution() {
  std::cout << "\n=== Test: Multi-Queue Execution ===" << std::endl;
  
  // Create command pool with multiple queues
  const size_t num_queues = 4;
  auto pool = MetalCommandPool::Create(0, num_queues);
  if (!pool) {
    std::cout << "Failed to create Metal command pool" << std::endl;
    return;
  }
  
  // Submit work to different queues
  std::vector<std::thread> threads;
  
  for (size_t i = 0; i < num_queues * 2; ++i) {
    threads.emplace_back([&pool, i]() {
      auto* queue = pool->GetQueue();  // Round-robin selection
      if (!queue) return;
      
      auto buffer = queue->CreateCommandBuffer();
      if (!buffer) return;
      
      buffer->Begin();
      buffer->PushDebugGroup("Queue " + std::to_string(i));
      // Add memory barrier as test command
      buffer->InsertMemoryBarrier();
      buffer->PopDebugGroup();
      buffer->End();
      
      ExecutionOptions options;
      options.synchronous = false;
      
      auto result = queue->Execute(*buffer, options);
      if (!result.IsSuccess()) {
        std::cout << "Queue " << i << " execution failed" << std::endl;
      }
    });
  }
  
  // Wait for all threads
  for (auto& t : threads) {
    t.join();
  }
  
  // Synchronize all queues
  auto result = pool->SynchronizeAll();
  if (result.IsSuccess()) {
    std::cout << "All queues synchronized successfully" << std::endl;
  }
  
  // Print pool metrics
  auto metrics = pool->GetMetrics();
  std::cout << "Pool metrics:" << std::endl;
  std::cout << "  Number of queues: " << metrics.num_queues << std::endl;
  std::cout << "  Total executions: " << metrics.total_executions << std::endl;
  std::cout << "  Average queue utilization: " << metrics.average_queue_utilization << std::endl;
}

// Example 6: Priority-based execution
void TestPriorityExecution() {
  std::cout << "\n=== Test: Priority-based Execution ===" << std::endl;
  
  // Create queues with different priorities
  auto high_priority_queue = MetalCommandQueue::Create(
      0, ExecutionOptions::Priority::USER_INTERACTIVE);
  auto low_priority_queue = MetalCommandQueue::Create(
      0, ExecutionOptions::Priority::BACKGROUND);
  
  if (!high_priority_queue || !low_priority_queue) {
    std::cout << "Failed to create priority queues" << std::endl;
    return;
  }
  
  // Submit work to both queues
  auto submit_work = [](MetalCommandQueue* queue, const std::string& label) {
    auto buffer = queue->CreateCommandBuffer();
    if (!buffer) return;
    
    buffer->Begin();
    buffer->PushDebugGroup(label);
    
    // Add actual Metal compute commands
    // Example: dispatch a simple compute kernel
    buffer->DispatchThreads({1024, 1, 1}, {256, 1, 1});
    
    // Add memory synchronization
    buffer->InsertMemoryBarrier();
    
    buffer->PopDebugGroup();
    buffer->End();
    
    ExecutionOptions options;
    options.synchronous = false;
    
    queue->Execute(*buffer, options);
  };
  
  // Submit multiple tasks
  for (int i = 0; i < 5; ++i) {
    submit_work(high_priority_queue.get(), "High Priority " + std::to_string(i));
    submit_work(low_priority_queue.get(), "Low Priority " + std::to_string(i));
  }
  
  // Synchronize both queues
  high_priority_queue->Synchronize();
  low_priority_queue->Synchronize();
  
  std::cout << "Priority-based execution completed" << std::endl;
}

// Example 7: Error handling and timeout
void TestErrorHandling() {
  std::cout << "\n=== Test: Error Handling ===" << std::endl;
  
  auto queue = MetalCommandQueue::Create(0);
  if (!queue) {
    std::cout << "Failed to create Metal command queue" << std::endl;
    return;
  }
  
  // Test invalid operations
  auto buffer = queue->CreateCommandBuffer();
  if (!buffer) return;
  
  // Try to end without begin
  auto result = buffer->End();
  if (!result.IsSuccess()) {
    std::cout << "Expected error: " << result.message << std::endl;
  }
  
  // Begin properly
  buffer->Begin();
  buffer->End();
  
  // Try to begin on committed buffer
  result = buffer->Begin();
  if (!result.IsSuccess()) {
    std::cout << "Expected error: " << result.message << std::endl;
  }
  
  // Test timeout
  buffer->Reset();
  buffer->Begin();
  buffer->End();
  
  ExecutionOptions options;
  options.synchronous = true;
  options.timeout = std::chrono::milliseconds(100);  // 100ms timeout
  
  result = queue->Execute(*buffer, options);
  if (result.IsSuccess()) {
    std::cout << "Execution with timeout completed" << std::endl;
  }
}

// Utility function to print device info
void PrintDeviceInfo() {
  std::cout << "\n=== Metal Device Information ===" << std::endl;
  
  auto devices = GetAvailableDevices();
  std::cout << "Found " << devices.size() << " Metal device(s)" << std::endl;
  
  for (int device_id : devices) {
    std::cout << "Device " << device_id << ": " << GetDeviceName(device_id) << std::endl;
    std::cout << "  Memory: " << GetDeviceMemory(device_id) / (1024.0 * 1024.0 * 1024.0) 
              << " GB" << std::endl;
    std::cout << "  Available: " << (IsDeviceAvailable(device_id) ? "Yes" : "No") 
              << std::endl;
  }
}

int main() {
  std::cout << "Metal Command Buffer Interface Test Suite" << std::endl;
  std::cout << "=========================================" << std::endl;
  
  // Print device information
  PrintDeviceInfo();
  
  // Run all tests
  TestSynchronousExecution();
  TestAsynchronousExecution();
  TestCommandBufferPooling();
  TestBatchExecution();
  TestMultiQueueExecution();
  TestPriorityExecution();
  TestErrorHandling();
  
  std::cout << "\nAll tests completed" << std::endl;
  
  return 0;
}