// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include <gtest/gtest.h>
#include <chrono>
#include <memory>
#include <random>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

#include "tritonserver.h"

namespace triton { namespace server { namespace test {

// Test configuration
struct MetalTestConfig {
  bool skip_if_no_metal = true;
  bool verbose = false;
  bool enable_profiling = false;
  std::string test_data_path = "./test_data";
  size_t timeout_seconds = 300;
  int device_id = 0;
};

// Metal device capabilities
struct MetalDeviceCapabilities {
  std::string name;
  size_t max_buffer_length;
  size_t max_threadgroup_memory;
  bool supports_unified_memory;
  bool supports_raytracing;
  bool supports_function_pointers;
  int gpu_family;
  int feature_set;
};

// Test data generator
class TestDataGenerator {
 public:
  TestDataGenerator(unsigned seed = 42) : rng_(seed) {}
  
  // Generate random float data
  std::vector<float> GenerateFloatData(size_t count, float min = -1.0f, float max = 1.0f);
  
  // Generate random int data
  std::vector<int32_t> GenerateIntData(size_t count, int32_t min = 0, int32_t max = 100);
  
  // Generate pattern data (for validation)
  std::vector<float> GeneratePatternData(size_t count, float start = 0.0f, float step = 1.0f);
  
  // Generate tensor data with specific shape
  std::vector<float> GenerateTensorData(const std::vector<int64_t>& shape);
  
 private:
  std::mt19937 rng_;
};

// Performance measurement utilities
class PerformanceTimer {
 public:
  void Start() {
    start_ = std::chrono::high_resolution_clock::now();
  }
  
  double ElapsedMs() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start_).count();
  }
  
  double ElapsedUs() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(end - start_).count();
  }
  
 private:
  std::chrono::high_resolution_clock::time_point start_;
};

// Memory usage tracker
class MemoryTracker {
 public:
  void RecordAllocation(size_t bytes) {
    current_usage_ += bytes;
    peak_usage_ = std::max(peak_usage_, current_usage_);
    total_allocated_ += bytes;
  }
  
  void RecordDeallocation(size_t bytes) {
    current_usage_ -= bytes;
  }
  
  size_t CurrentUsage() const { return current_usage_; }
  size_t PeakUsage() const { return peak_usage_; }
  size_t TotalAllocated() const { return total_allocated_; }
  
  void Reset() {
    current_usage_ = 0;
    peak_usage_ = 0;
    total_allocated_ = 0;
  }
  
 private:
  size_t current_usage_ = 0;
  size_t peak_usage_ = 0;
  size_t total_allocated_ = 0;
};

// Test result collector
struct TestResult {
  std::string test_name;
  bool passed;
  double duration_ms;
  size_t memory_used;
  std::string error_message;
  std::map<std::string, double> metrics;
};

class TestResultCollector {
 public:
  void AddResult(const TestResult& result);
  void PrintSummary() const;
  void SaveToFile(const std::string& filename) const;
  
 private:
  std::vector<TestResult> results_;
};

// Metal-specific test helpers
#ifdef __APPLE__

// Check if Metal is available
bool IsMetalAvailable();

// Get Metal device capabilities
MetalDeviceCapabilities GetDeviceCapabilities(id<MTLDevice> device);

// Create test Metal buffer
id<MTLBuffer> CreateTestBuffer(id<MTLDevice> device, size_t size, 
                               const void* data = nullptr);

// Create test compute pipeline
id<MTLComputePipelineState> CreateTestComputePipeline(
    id<MTLDevice> device, const std::string& kernel_source,
    const std::string& function_name);

// Validate buffer contents
bool ValidateBuffer(id<MTLBuffer> buffer, const void* expected_data,
                   size_t size, float tolerance = 1e-6f);

// Performance profiling helpers
class MetalProfiler {
 public:
  MetalProfiler(id<MTLCommandBuffer> command_buffer);
  ~MetalProfiler();
  
  double GetGPUTime() const;
  double GetKernelTime(const std::string& kernel_name) const;
  
 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

#endif  // __APPLE__

// Test macros
#define SKIP_IF_NO_METAL() \
  do { \
    if (!IsMetalAvailable()) { \
      GTEST_SKIP() << "Metal not available on this system"; \
    } \
  } while (0)

#define ASSERT_TRITON_OK(expr) \
  do { \
    auto err = (expr); \
    ASSERT_EQ(err, nullptr) << "Triton error: " \
        << (err ? TRITONSERVER_ErrorMessage(err) : "unknown"); \
    if (err) TRITONSERVER_ErrorDelete(err); \
  } while (0)

#define EXPECT_TRITON_OK(expr) \
  do { \
    auto err = (expr); \
    EXPECT_EQ(err, nullptr) << "Triton error: " \
        << (err ? TRITONSERVER_ErrorMessage(err) : "unknown"); \
    if (err) TRITONSERVER_ErrorDelete(err); \
  } while (0)

// Benchmark macros
#define BENCHMARK_ITERATIONS(name, iterations, code) \
  do { \
    PerformanceTimer timer; \
    timer.Start(); \
    for (int i = 0; i < iterations; ++i) { \
      code; \
    } \
    double elapsed = timer.ElapsedMs(); \
    std::cout << name << ": " << elapsed / iterations \
              << " ms/iteration" << std::endl; \
  } while (0)

}}}  // namespace triton::server::test