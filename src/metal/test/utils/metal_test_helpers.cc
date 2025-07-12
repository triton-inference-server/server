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

#include "metal_test_helpers.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

namespace triton { namespace server { namespace test {

// TestDataGenerator implementation
std::vector<float>
TestDataGenerator::GenerateFloatData(size_t count, float min, float max)
{
  std::vector<float> data(count);
  std::uniform_real_distribution<float> dist(min, max);
  
  for (size_t i = 0; i < count; ++i) {
    data[i] = dist(rng_);
  }
  
  return data;
}

std::vector<int32_t>
TestDataGenerator::GenerateIntData(size_t count, int32_t min, int32_t max)
{
  std::vector<int32_t> data(count);
  std::uniform_int_distribution<int32_t> dist(min, max);
  
  for (size_t i = 0; i < count; ++i) {
    data[i] = dist(rng_);
  }
  
  return data;
}

std::vector<float>
TestDataGenerator::GeneratePatternData(size_t count, float start, float step)
{
  std::vector<float> data(count);
  
  for (size_t i = 0; i < count; ++i) {
    data[i] = start + i * step;
  }
  
  return data;
}

std::vector<float>
TestDataGenerator::GenerateTensorData(const std::vector<int64_t>& shape)
{
  size_t total_size = 1;
  for (auto dim : shape) {
    total_size *= dim;
  }
  
  return GenerateFloatData(total_size);
}

// TestResultCollector implementation
void
TestResultCollector::AddResult(const TestResult& result)
{
  results_.push_back(result);
}

void
TestResultCollector::PrintSummary() const
{
  std::cout << "\n========== Test Summary ==========\n";
  
  int passed = 0;
  int failed = 0;
  double total_time = 0;
  size_t total_memory = 0;
  
  for (const auto& result : results_) {
    if (result.passed) {
      passed++;
    } else {
      failed++;
    }
    total_time += result.duration_ms;
    total_memory += result.memory_used;
  }
  
  std::cout << "Total tests: " << results_.size() << "\n";
  std::cout << "Passed: " << passed << "\n";
  std::cout << "Failed: " << failed << "\n";
  std::cout << "Total time: " << total_time << " ms\n";
  std::cout << "Total memory: " << total_memory / (1024 * 1024) << " MB\n";
  
  if (failed > 0) {
    std::cout << "\nFailed tests:\n";
    for (const auto& result : results_) {
      if (!result.passed) {
        std::cout << "  - " << result.test_name << ": " 
                  << result.error_message << "\n";
      }
    }
  }
  
  std::cout << "\nPerformance metrics:\n";
  for (const auto& result : results_) {
    if (!result.metrics.empty()) {
      std::cout << result.test_name << ":\n";
      for (const auto& [metric, value] : result.metrics) {
        std::cout << "  " << metric << ": " << value << "\n";
      }
    }
  }
}

void
TestResultCollector::SaveToFile(const std::string& filename) const
{
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << "\n";
    return;
  }
  
  file << "Test Results\n";
  file << "============\n\n";
  
  for (const auto& result : results_) {
    file << "Test: " << result.test_name << "\n";
    file << "Status: " << (result.passed ? "PASSED" : "FAILED") << "\n";
    file << "Duration: " << result.duration_ms << " ms\n";
    file << "Memory: " << result.memory_used / 1024 << " KB\n";
    
    if (!result.passed) {
      file << "Error: " << result.error_message << "\n";
    }
    
    if (!result.metrics.empty()) {
      file << "Metrics:\n";
      for (const auto& [metric, value] : result.metrics) {
        file << "  " << metric << ": " << value << "\n";
      }
    }
    
    file << "\n";
  }
}

#ifdef __APPLE__

bool IsMetalAvailable()
{
  @autoreleasepool {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    bool available = devices.count > 0;
    return available;
  }
}

MetalDeviceCapabilities GetDeviceCapabilities(id<MTLDevice> device)
{
  MetalDeviceCapabilities caps;
  
  caps.name = [device.name UTF8String];
  caps.max_buffer_length = device.maxBufferLength;
  caps.max_threadgroup_memory = device.maxThreadgroupMemoryLength;
  
  // Check features
  caps.supports_unified_memory = device.hasUnifiedMemory;
  
  if (@available(macOS 11.0, *)) {
    caps.supports_raytracing = device.supportsRaytracing;
    caps.supports_function_pointers = device.supportsFunctionPointers;
  } else {
    caps.supports_raytracing = false;
    caps.supports_function_pointers = false;
  }
  
  // Determine GPU family
  if ([device supportsFamily:MTLGPUFamilyApple7]) {
    caps.gpu_family = 7;
  } else if ([device supportsFamily:MTLGPUFamilyApple6]) {
    caps.gpu_family = 6;
  } else if ([device supportsFamily:MTLGPUFamilyApple5]) {
    caps.gpu_family = 5;
  } else if ([device supportsFamily:MTLGPUFamilyApple4]) {
    caps.gpu_family = 4;
  } else if ([device supportsFamily:MTLGPUFamilyApple3]) {
    caps.gpu_family = 3;
  } else if ([device supportsFamily:MTLGPUFamilyApple2]) {
    caps.gpu_family = 2;
  } else if ([device supportsFamily:MTLGPUFamilyApple1]) {
    caps.gpu_family = 1;
  } else {
    caps.gpu_family = 0;  // Intel or unknown
  }
  
  // Determine feature set
  if ([device supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily2_v1]) {
    caps.feature_set = 2;
  } else if ([device supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily1_v4]) {
    caps.feature_set = 1;
  } else {
    caps.feature_set = 0;
  }
  
  return caps;
}

id<MTLBuffer> CreateTestBuffer(id<MTLDevice> device, size_t size, 
                               const void* data)
{
  MTLResourceOptions options = MTLResourceStorageModeShared;
  
  if (data) {
    return [device newBufferWithBytes:data length:size options:options];
  } else {
    return [device newBufferWithLength:size options:options];
  }
}

id<MTLComputePipelineState> CreateTestComputePipeline(
    id<MTLDevice> device, const std::string& kernel_source,
    const std::string& function_name)
{
  NSError* error = nil;
  
  // Create library from source
  NSString* source = [NSString stringWithUTF8String:kernel_source.c_str()];
  id<MTLLibrary> library = [device newLibraryWithSource:source 
                                                options:nil 
                                                  error:&error];
  if (!library) {
    NSLog(@"Failed to create library: %@", error);
    return nil;
  }
  
  // Get function
  NSString* funcName = [NSString stringWithUTF8String:function_name.c_str()];
  id<MTLFunction> function = [library newFunctionWithName:funcName];
  if (!function) {
    NSLog(@"Failed to find function: %@", funcName);
    return nil;
  }
  
  // Create pipeline
  id<MTLComputePipelineState> pipeline = 
      [device newComputePipelineStateWithFunction:function error:&error];
  if (!pipeline) {
    NSLog(@"Failed to create pipeline: %@", error);
    return nil;
  }
  
  return pipeline;
}

bool ValidateBuffer(id<MTLBuffer> buffer, const void* expected_data,
                   size_t size, float tolerance)
{
  const float* buffer_data = static_cast<const float*>(buffer.contents);
  const float* expected = static_cast<const float*>(expected_data);
  
  size_t count = size / sizeof(float);
  
  for (size_t i = 0; i < count; ++i) {
    float diff = std::abs(buffer_data[i] - expected[i]);
    if (diff > tolerance) {
      std::cerr << "Validation failed at index " << i 
                << ": expected " << expected[i]
                << ", got " << buffer_data[i]
                << " (diff: " << diff << ")\n";
      return false;
    }
  }
  
  return true;
}

// MetalProfiler implementation
class MetalProfiler::Impl {
 public:
  Impl(id<MTLCommandBuffer> command_buffer) : command_buffer_(command_buffer) {
    if (@available(macOS 10.15, *)) {
      // Enable GPU timing
      [command_buffer_ addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        gpu_start_time_ = buffer.GPUStartTime;
        gpu_end_time_ = buffer.GPUEndTime;
      }];
    }
  }
  
  double GetGPUTime() const {
    if (@available(macOS 10.15, *)) {
      return (gpu_end_time_ - gpu_start_time_) * 1000.0;  // Convert to ms
    }
    return 0.0;
  }
  
 private:
  id<MTLCommandBuffer> command_buffer_;
  double gpu_start_time_ = 0.0;
  double gpu_end_time_ = 0.0;
};

MetalProfiler::MetalProfiler(id<MTLCommandBuffer> command_buffer)
    : impl_(std::make_unique<Impl>(command_buffer))
{
}

MetalProfiler::~MetalProfiler() = default;

double MetalProfiler::GetGPUTime() const
{
  return impl_->GetGPUTime();
}

double MetalProfiler::GetKernelTime(const std::string& kernel_name) const
{
  // This would require more sophisticated profiling with encoders
  // For now, return total GPU time
  return GetGPUTime();
}

#endif  // __APPLE__

}}}  // namespace triton::server::test