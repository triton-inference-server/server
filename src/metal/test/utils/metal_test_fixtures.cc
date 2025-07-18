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

#include "metal_test_fixtures.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <thread>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif

namespace triton { namespace server { namespace test {

// MetalTestBase implementation
void MetalTestBase::SetUp()
{
  ::testing::Test::SetUp();
  
  // Load configuration from environment
  if (const char* verbose = std::getenv("METAL_TEST_VERBOSE")) {
    config_.verbose = std::string(verbose) == "1";
  }
  
  if (const char* profiling = std::getenv("METAL_TEST_PROFILING")) {
    config_.enable_profiling = std::string(profiling) == "1";
  }
  
  if (const char* data_path = std::getenv("METAL_TEST_DATA_PATH")) {
    config_.test_data_path = data_path;
  }
  
  test_timer_.Start();
  memory_tracker_.Reset();
}

void MetalTestBase::TearDown()
{
  double duration = test_timer_.ElapsedMs();
  
  TestResult result;
  result.test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
  result.passed = !HasFailure();
  result.duration_ms = duration;
  result.memory_used = memory_tracker_.PeakUsage();
  
  if (HasFailure()) {
    result.error_message = "Test failed";
  }
  
  result_collector_.AddResult(result);
  
  ::testing::Test::TearDown();
}

void MetalTestBase::RequireMetal()
{
#ifdef __APPLE__
  if (!IsMetalAvailable()) {
    GTEST_SKIP() << "Metal not available on this system";
  }
#else
  GTEST_SKIP() << "Metal tests require macOS";
#endif
}

std::string MetalTestBase::GetTestDataPath(const std::string& filename) const
{
  return config_.test_data_path + "/" + filename;
}

// MetalDeviceTest implementation
void MetalDeviceTest::SetUp()
{
  MetalTestBase::SetUp();
  RequireMetal();
  
#ifdef __APPLE__
  @autoreleasepool {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (config_.device_id < devices.count) {
      device_ = devices[config_.device_id];
      command_queue_ = [device_ newCommandQueue];
      capabilities_ = GetDeviceCapabilities(device_);
      
      if (config_.verbose) {
        std::cout << "Using Metal device: " << capabilities_.name << "\n";
        std::cout << "  Max buffer length: " 
                  << capabilities_.max_buffer_length / (1024*1024*1024) << " GB\n";
        std::cout << "  Unified memory: " 
                  << (capabilities_.supports_unified_memory ? "Yes" : "No") << "\n";
        std::cout << "  GPU family: " << capabilities_.gpu_family << "\n";
      }
    } else {
      GTEST_SKIP() << "Device " << config_.device_id << " not available";
    }
  }
#endif
}

void MetalDeviceTest::TearDown()
{
#ifdef __APPLE__
  command_queue_ = nil;
  device_ = nil;
#endif
  
  MetalTestBase::TearDown();
}

// MetalAllocatorTest implementation  
void MetalAllocatorTest::SetUp()
{
  MetalDeviceTest::SetUp();
  
  allocator_ = std::make_shared<MetalAllocator>(config_.device_id);
}

void MetalAllocatorTest::TearDown()
{
  allocator_.reset();
  
  MetalDeviceTest::TearDown();
}

// MetalMemoryTest implementation
void MetalMemoryTest::SetUp()
{
  MetalDeviceTest::SetUp();
  
  memory_manager_ = std::make_unique<MetalMemoryManager>();
  auto err = memory_manager_->Initialize(config_.device_id);
  if (err != nullptr) {
    TRITONSERVER_ErrorDelete(err);
    GTEST_SKIP() << "Failed to initialize memory manager";
  }
}

void MetalMemoryTest::TearDown()
{
  memory_manager_.reset();
  
  MetalDeviceTest::TearDown();
}

// MetalBackendTest implementation
void MetalBackendTest::SetUp()
{
  MetalTestBase::SetUp();
  RequireMetal();
  
  // Create server options
  ASSERT_TRITON_OK(TRITONSERVER_ServerOptionsNew(&server_options_));
  
  // Set model repository
  std::string model_repo = GetTestDataPath("models");
  ASSERT_TRITON_OK(TRITONSERVER_ServerOptionsSetModelRepositoryPath(
      server_options_, model_repo.c_str()));
  
  // Enable Metal backend
  ASSERT_TRITON_OK(TRITONSERVER_ServerOptionsSetBackendDirectory(
      server_options_, "/usr/local/lib/tritonserver/backends"));
  
  // Create server
  ASSERT_TRITON_OK(TRITONSERVER_ServerNew(&server_, server_options_));
}

void MetalBackendTest::TearDown()
{
  if (server_) {
    TRITONSERVER_ServerDelete(server_);
  }
  
  if (server_options_) {
    TRITONSERVER_ServerOptionsDelete(server_options_);
  }
  
  MetalTestBase::TearDown();
}

TRITONSERVER_InferenceRequest*
MetalBackendTest::CreateTestRequest(
    const std::string& model_name,
    const std::vector<std::pair<std::string, std::vector<int64_t>>>& inputs,
    const std::vector<std::string>& outputs)
{
  TRITONSERVER_InferenceRequest* request = nullptr;
  
  // Create request
  auto err = TRITONSERVER_InferenceRequestNew(
      &request, server_, model_name.c_str(), -1 /* model_version */);
  if (err != nullptr) {
    TRITONSERVER_ErrorDelete(err);
    return nullptr;
  }
  
  // Set inputs
  for (const auto& [name, shape] : inputs) {
    err = TRITONSERVER_InferenceRequestAddInput(
        request, name.c_str(), TRITONSERVER_TYPE_FP32,
        shape.data(), shape.size());
    if (err != nullptr) {
      TRITONSERVER_ErrorDelete(err);
      TRITONSERVER_InferenceRequestDelete(request);
      return nullptr;
    }
  }
  
  // Set outputs
  for (const auto& name : outputs) {
    err = TRITONSERVER_InferenceRequestAddRequestedOutput(request, name.c_str());
    if (err != nullptr) {
      TRITONSERVER_ErrorDelete(err);
      TRITONSERVER_InferenceRequestDelete(request);
      return nullptr;
    }
  }
  
  return request;
}

bool
MetalBackendTest::ValidateResponse(
    TRITONSERVER_InferenceResponse* response,
    const std::map<std::string, std::vector<float>>& expected_outputs,
    float tolerance)
{
  // Check for errors
  if (TRITONSERVER_InferenceResponseError(response)) {
    const char* error_message;
    TRITONSERVER_InferenceResponseError(response, &error_message);
    std::cerr << "Response error: " << error_message << "\n";
    return false;
  }
  
  // Validate each output
  for (const auto& [name, expected] : expected_outputs) {
    const void* base;
    size_t byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    
    auto err = TRITONSERVER_InferenceResponseOutput(
        response, name.c_str(), &base, &byte_size,
        &memory_type, &memory_type_id);
    
    if (err != nullptr) {
      std::cerr << "Failed to get output " << name << ": "
                << TRITONSERVER_ErrorMessage(err) << "\n";
      TRITONSERVER_ErrorDelete(err);
      return false;
    }
    
    // Compare values
    const float* actual = static_cast<const float*>(base);
    size_t count = byte_size / sizeof(float);
    
    if (count != expected.size()) {
      std::cerr << "Output " << name << " size mismatch: expected "
                << expected.size() << ", got " << count << "\n";
      return false;
    }
    
    for (size_t i = 0; i < count; ++i) {
      float diff = std::abs(actual[i] - expected[i]);
      if (diff > tolerance) {
        std::cerr << "Output " << name << " mismatch at index " << i
                  << ": expected " << expected[i] << ", got " << actual[i]
                  << " (diff: " << diff << ")\n";
        return false;
      }
    }
  }
  
  return true;
}

// MetalBenchmarkTest implementation
void MetalBenchmarkTest::SetUp()
{
  MetalDeviceTest::SetUp();
  
  enable_profiling_ = config_.enable_profiling;
  
  if (const char* output = std::getenv("METAL_BENCHMARK_OUTPUT")) {
    benchmark_output_file_ = output;
  }
}

void MetalBenchmarkTest::TearDown()
{
  MetalDeviceTest::TearDown();
}

void MetalBenchmarkTest::ReportResults(
    const std::string& benchmark_name,
    const std::map<std::string, double>& metrics)
{
  std::cout << "\nBenchmark: " << benchmark_name << "\n";
  std::cout << "Results:\n";
  
  for (const auto& [metric, value] : metrics) {
    std::cout << "  " << metric << ": " << value << "\n";
  }
  
  if (!benchmark_output_file_.empty()) {
    std::ofstream file(benchmark_output_file_, std::ios::app);
    file << benchmark_name << ",";
    for (const auto& [metric, value] : metrics) {
      file << metric << "," << value << ",";
    }
    file << "\n";
  }
}

// MetalStressTest implementation
void MetalStressTest::SetUp()
{
  MetalDeviceTest::SetUp();
}

void MetalStressTest::TearDown()
{
  stop_stress_test_ = true;
  MetalDeviceTest::TearDown();
}

void MetalStressTest::RunStressTest(
    const std::string& test_name,
    std::function<void()> test_func,
    size_t duration_seconds,
    size_t num_threads)
{
  std::cout << "Running stress test: " << test_name << "\n";
  std::cout << "Duration: " << duration_seconds << " seconds\n";
  std::cout << "Threads: " << num_threads << "\n";
  
  stop_stress_test_ = false;
  std::atomic<size_t> iterations{0};
  std::atomic<size_t> errors{0};
  
  auto worker = [&]() {
    while (!stop_stress_test_) {
      try {
        test_func();
        iterations.fetch_add(1);
      } catch (const std::exception& e) {
        errors.fetch_add(1);
        if (config_.verbose) {
          std::cerr << "Stress test error: " << e.what() << "\n";
        }
      }
    }
  };
  
  // Start monitoring
  StartResourceMonitoring();
  
  // Start worker threads
  std::vector<std::thread> threads;
  for (size_t i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker);
  }
  
  // Run for specified duration
  std::this_thread::sleep_for(std::chrono::seconds(duration_seconds));
  stop_stress_test_ = true;
  
  // Wait for threads
  for (auto& t : threads) {
    t.join();
  }
  
  // Stop monitoring
  StopResourceMonitoring();
  
  // Report results
  std::cout << "Stress test completed:\n";
  std::cout << "  Total iterations: " << iterations.load() << "\n";
  std::cout << "  Errors: " << errors.load() << "\n";
  std::cout << "  Iterations/second: " 
            << iterations.load() / duration_seconds << "\n";
  
  if (!cpu_usage_samples_.empty()) {
    double avg_cpu = std::accumulate(cpu_usage_samples_.begin(),
                                   cpu_usage_samples_.end(), 0.0) /
                    cpu_usage_samples_.size();
    std::cout << "  Average CPU usage: " << avg_cpu << "%\n";
  }
  
  if (!memory_usage_samples_.empty()) {
    size_t peak_memory = *std::max_element(memory_usage_samples_.begin(),
                                         memory_usage_samples_.end());
    std::cout << "  Peak memory usage: " << peak_memory / (1024*1024) << " MB\n";
  }
}

void MetalStressTest::StartResourceMonitoring()
{
  // This would start a background thread to monitor CPU and memory
  // Implementation depends on platform-specific APIs
}

void MetalStressTest::StopResourceMonitoring()
{
  // Stop the monitoring thread
}

// MetalPlatformTest implementation
void MetalPlatformTest::SetUp()
{
  MetalTestBase::SetUp();
  
#ifdef __APPLE__
  @autoreleasepool {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    for (NSUInteger i = 0; i < devices.count; ++i) {
      available_devices_.push_back(i);
      device_capabilities_[i] = GetDeviceCapabilities(devices[i]);
    }
  }
#endif
}

void MetalPlatformTest::TearDown()
{
  MetalTestBase::TearDown();
}

bool MetalPlatformTest::IsAppleSilicon() const
{
#ifdef __APPLE__
  // Check if running on Apple Silicon
  int ret = 0;
  size_t size = sizeof(ret);
  if (sysctlbyname("hw.optional.arm64", &ret, &size, NULL, 0) == 0) {
    return ret == 1;
  }
#endif
  return false;
}

bool MetalPlatformTest::IsIntelMac() const
{
  return !IsAppleSilicon();
}

int MetalPlatformTest::GetMacOSVersion() const
{
#ifdef __APPLE__
  // Get macOS version
  NSOperatingSystemVersion version = 
      [[NSProcessInfo processInfo] operatingSystemVersion];
  return version.majorVersion * 100 + version.minorVersion;
#else
  return 0;
#endif
}

std::vector<std::string> MetalPlatformTest::GetAvailableGPUs() const
{
  std::vector<std::string> gpus;
  
#ifdef __APPLE__
  for (const auto& [id, caps] : device_capabilities_) {
    gpus.push_back(caps.name);
  }
#endif
  
  return gpus;
}

void MetalPlatformTest::TestOnAllDevices(
    std::function<void(int device_id)> test_func)
{
  for (int device_id : available_devices_) {
    std::cout << "Testing on device " << device_id << ": "
              << device_capabilities_[device_id].name << "\n";
    
    test_func(device_id);
  }
}

// MetalE2ETest implementation
void MetalE2ETest::SetUp()
{
  MetalBackendTest::SetUp();
  
  model_repository_path_ = GetTestDataPath("model_repository");
}

void MetalE2ETest::TearDown()
{
  loaded_models_.clear();
  
  MetalBackendTest::TearDown();
}

void MetalE2ETest::LoadModel(
    const std::string& model_path,
    const std::string& model_name)
{
  // Copy model to repository
  std::string dest_path = model_repository_path_ + "/" + model_name;
  
  // In a real implementation, this would copy the model files
  // For now, just track that it's loaded
  loaded_models_[model_name] = model_path;
  
  // Load model in server
  auto err = TRITONSERVER_ServerLoadModel(server_, model_name.c_str());
  if (err != nullptr) {
    std::cerr << "Failed to load model " << model_name << ": "
              << TRITONSERVER_ErrorMessage(err) << "\n";
    TRITONSERVER_ErrorDelete(err);
  }
}

TRITONSERVER_InferenceResponse*
MetalE2ETest::RunInference(
    const std::string& model_name,
    const std::map<std::string, std::vector<float>>& inputs)
{
  // Implementation would create request, set inputs, and run inference
  // This is a simplified version
  return nullptr;
}

double MetalE2ETest::MeasureInferenceLatency(
    const std::string& model_name,
    const std::map<std::string, std::vector<float>>& inputs,
    int num_iterations)
{
  // Warmup
  for (int i = 0; i < 10; ++i) {
    auto response = RunInference(model_name, inputs);
    if (response) {
      TRITONSERVER_InferenceResponseDelete(response);
    }
  }
  
  // Measure
  PerformanceTimer timer;
  timer.Start();
  
  for (int i = 0; i < num_iterations; ++i) {
    auto response = RunInference(model_name, inputs);
    if (response) {
      TRITONSERVER_InferenceResponseDelete(response);
    }
  }
  
  return timer.ElapsedMs() / num_iterations;
}

}}}  // namespace triton::server::test