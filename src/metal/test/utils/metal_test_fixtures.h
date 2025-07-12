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

#include "metal_test_helpers.h"
#include "src/metal/metal_allocator.h"
#include "src/metal/metal_device.h"
#include "src/metal/metal_memory_manager.h"

#include <memory>

namespace triton { namespace server { namespace test {

// Base test fixture for Metal tests
class MetalTestBase : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;
  
  // Helper to skip test if Metal not available
  void RequireMetal();
  
  // Get test configuration
  const MetalTestConfig& GetConfig() const { return config_; }
  
  // Test data directory
  std::string GetTestDataPath(const std::string& filename) const;
  
 protected:
  MetalTestConfig config_;
  TestResultCollector result_collector_;
  PerformanceTimer test_timer_;
  MemoryTracker memory_tracker_;
};

// Test fixture with Metal device
class MetalDeviceTest : public MetalTestBase {
 protected:
  void SetUp() override;
  void TearDown() override;
  
#ifdef __APPLE__
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> command_queue_ = nil;
  MetalDeviceCapabilities capabilities_;
#endif
};

// Test fixture with Metal allocator
class MetalAllocatorTest : public MetalDeviceTest {
 protected:
  void SetUp() override;
  void TearDown() override;
  
  std::shared_ptr<MetalAllocator> allocator_;
};

// Test fixture with Metal memory manager
class MetalMemoryTest : public MetalDeviceTest {
 protected:
  void SetUp() override;
  void TearDown() override;
  
  std::unique_ptr<MetalMemoryManager> memory_manager_;
};

// Test fixture for backend integration tests
class MetalBackendTest : public MetalTestBase {
 protected:
  void SetUp() override;
  void TearDown() override;
  
  // Create test model
  TRITONSERVER_InferenceRequest* CreateTestRequest(
      const std::string& model_name,
      const std::vector<std::pair<std::string, std::vector<int64_t>>>& inputs,
      const std::vector<std::string>& outputs);
  
  // Validate response
  bool ValidateResponse(
      TRITONSERVER_InferenceResponse* response,
      const std::map<std::string, std::vector<float>>& expected_outputs,
      float tolerance = 1e-5f);
  
 protected:
  TRITONSERVER_Server* server_ = nullptr;
  TRITONSERVER_ServerOptions* server_options_ = nullptr;
};

// Test fixture for performance benchmarks
class MetalBenchmarkTest : public MetalDeviceTest {
 protected:
  void SetUp() override;
  void TearDown() override;
  
  // Run benchmark with warmup
  template<typename Func>
  double RunBenchmark(const std::string& name, int warmup_iterations,
                     int benchmark_iterations, Func&& func);
  
  // Report benchmark results
  void ReportResults(const std::string& benchmark_name,
                    const std::map<std::string, double>& metrics);
  
 protected:
  bool enable_profiling_ = false;
  std::string benchmark_output_file_;
};

// Test fixture for stress tests
class MetalStressTest : public MetalDeviceTest {
 protected:
  void SetUp() override;
  void TearDown() override;
  
  // Run stress test
  void RunStressTest(const std::string& test_name,
                    std::function<void()> test_func,
                    size_t duration_seconds,
                    size_t num_threads = 1);
  
  // Monitor system resources
  void StartResourceMonitoring();
  void StopResourceMonitoring();
  
 protected:
  std::atomic<bool> stop_stress_test_{false};
  std::vector<double> cpu_usage_samples_;
  std::vector<size_t> memory_usage_samples_;
};

// Test fixture for platform compatibility tests  
class MetalPlatformTest : public MetalTestBase {
 protected:
  void SetUp() override;
  void TearDown() override;
  
  // Check platform features
  bool IsAppleSilicon() const;
  bool IsIntelMac() const;
  int GetMacOSVersion() const;
  std::vector<std::string> GetAvailableGPUs() const;
  
  // Platform-specific tests
  void TestOnAllDevices(std::function<void(int device_id)> test_func);
  
 protected:
  std::vector<int> available_devices_;
  std::map<int, MetalDeviceCapabilities> device_capabilities_;
};

// Test fixture for end-to-end tests
class MetalE2ETest : public MetalBackendTest {
 protected:
  void SetUp() override;
  void TearDown() override;
  
  // Load test model
  void LoadModel(const std::string& model_path,
                const std::string& model_name);
  
  // Run inference
  TRITONSERVER_InferenceResponse* RunInference(
      const std::string& model_name,
      const std::map<std::string, std::vector<float>>& inputs);
  
  // Measure end-to-end latency
  double MeasureInferenceLatency(
      const std::string& model_name,
      const std::map<std::string, std::vector<float>>& inputs,
      int num_iterations = 100);
  
 protected:
  std::string model_repository_path_;
  std::map<std::string, std::string> loaded_models_;
};

// Template implementation
template<typename Func>
double MetalBenchmarkTest::RunBenchmark(
    const std::string& name, int warmup_iterations,
    int benchmark_iterations, Func&& func)
{
  // Warmup
  for (int i = 0; i < warmup_iterations; ++i) {
    func();
  }
  
  // Benchmark
  PerformanceTimer timer;
  timer.Start();
  
  for (int i = 0; i < benchmark_iterations; ++i) {
    func();
  }
  
#ifdef __APPLE__
  // Wait for GPU work to complete
  if (command_queue_) {
    @autoreleasepool {
      id<MTLCommandBuffer> sync_buffer = [command_queue_ commandBuffer];
      [sync_buffer commit];
      [sync_buffer waitUntilCompleted];
    }
  }
#endif
  
  double elapsed_ms = timer.ElapsedMs();
  double ms_per_iteration = elapsed_ms / benchmark_iterations;
  
  std::cout << name << ": " << ms_per_iteration << " ms/iteration\n";
  
  return ms_per_iteration;
}

}}}  // namespace triton::server::test