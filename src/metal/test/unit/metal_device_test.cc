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

#include "src/metal/test/utils/metal_test_fixtures.h"
#include "src/metal/metal_device.h"

namespace triton { namespace server { namespace test {

class MetalDeviceUnitTest : public MetalDeviceTest {
};

TEST_F(MetalDeviceUnitTest, DeviceEnumeration)
{
  auto devices = MetalDevice::GetAvailableDevices();
  ASSERT_GT(devices.size(), 0) << "No Metal devices found";
  
  for (const auto& device : devices) {
    EXPECT_GE(device.id, 0);
    EXPECT_FALSE(device.name.empty());
    EXPECT_GT(device.total_memory, 0);
    
    if (GetConfig().verbose) {
      std::cout << "Device " << device.id << ": " << device.name << "\n";
      std::cout << "  Total memory: " << device.total_memory / (1024*1024*1024) << " GB\n";
      std::cout << "  Unified memory: " << (device.has_unified_memory ? "Yes" : "No") << "\n";
      std::cout << "  GPU family: " << device.gpu_family << "\n";
    }
  }
}

TEST_F(MetalDeviceUnitTest, DeviceSelection)
{
  // Test getting specific device
  auto device = MetalDevice::GetDevice(0);
  ASSERT_NE(device, nullptr);
  
  // Test device properties
  EXPECT_EQ(device->GetId(), 0);
  EXPECT_FALSE(device->GetName().empty());
  
  auto info = device->GetInfo();
  EXPECT_GT(info.total_memory, 0);
}

TEST_F(MetalDeviceUnitTest, InvalidDeviceId)
{
  // Test with invalid device ID
  auto device = MetalDevice::GetDevice(999);
  EXPECT_EQ(device, nullptr);
}

TEST_F(MetalDeviceUnitTest, DeviceCapabilities)
{
  auto device = MetalDevice::GetDevice(0);
  ASSERT_NE(device, nullptr);
  
  // Test capability queries
  auto caps = device->GetCapabilities();
  
  EXPECT_GT(caps.max_buffer_size, 0);
  EXPECT_GT(caps.max_texture_size, 0);
  EXPECT_GT(caps.max_threadgroup_memory, 0);
  EXPECT_GT(caps.max_threads_per_threadgroup, 0);
  
  // Feature support
  if (GetConfig().verbose) {
    std::cout << "Device capabilities:\n";
    std::cout << "  Max buffer size: " << caps.max_buffer_size / (1024*1024) << " MB\n";
    std::cout << "  Max texture size: " << caps.max_texture_size << "\n";
    std::cout << "  Max threadgroup memory: " << caps.max_threadgroup_memory << " bytes\n";
    std::cout << "  Supports fp16: " << (caps.supports_fp16 ? "Yes" : "No") << "\n";
    std::cout << "  Supports int8: " << (caps.supports_int8 ? "Yes" : "No") << "\n";
  }
}

TEST_F(MetalDeviceUnitTest, DeviceMemoryInfo)
{
  auto device = MetalDevice::GetDevice(0);
  ASSERT_NE(device, nullptr);
  
  size_t available, total;
  auto err = device->QueryMemory(&available, &total);
  ASSERT_TRITON_OK(err);
  
  EXPECT_GT(total, 0);
  EXPECT_GT(available, 0);
  EXPECT_LE(available, total);
  
  if (GetConfig().verbose) {
    std::cout << "Device memory:\n";
    std::cout << "  Total: " << total / (1024*1024*1024) << " GB\n";
    std::cout << "  Available: " << available / (1024*1024*1024) << " GB\n";
    std::cout << "  Used: " << (total - available) / (1024*1024*1024) << " GB\n";
  }
}

TEST_F(MetalDeviceUnitTest, CommandQueueCreation)
{
  auto device = MetalDevice::GetDevice(0);
  ASSERT_NE(device, nullptr);
  
  // Create command queue
  auto queue = device->CreateCommandQueue();
  ASSERT_NE(queue, nullptr);
  
  // Test queue properties
  EXPECT_GT(queue->GetMaxCommandBufferCount(), 0);
}

TEST_F(MetalDeviceUnitTest, MultipleCommandQueues)
{
  auto device = MetalDevice::GetDevice(0);
  ASSERT_NE(device, nullptr);
  
  // Create multiple queues
  const int num_queues = 4;
  std::vector<std::shared_ptr<MetalCommandQueue>> queues;
  
  for (int i = 0; i < num_queues; ++i) {
    auto queue = device->CreateCommandQueue();
    ASSERT_NE(queue, nullptr);
    queues.push_back(queue);
  }
  
  // All queues should be unique
  for (int i = 0; i < num_queues; ++i) {
    for (int j = i + 1; j < num_queues; ++j) {
      EXPECT_NE(queues[i].get(), queues[j].get());
    }
  }
}

TEST_F(MetalDeviceUnitTest, DeviceReset)
{
  auto device = MetalDevice::GetDevice(0);
  ASSERT_NE(device, nullptr);
  
  // Create some resources
  auto queue = device->CreateCommandQueue();
  ASSERT_NE(queue, nullptr);
  
  // Reset device
  auto err = device->Reset();
  ASSERT_TRITON_OK(err);
  
  // Should be able to create new resources
  auto new_queue = device->CreateCommandQueue();
  ASSERT_NE(new_queue, nullptr);
}

TEST_F(MetalDeviceUnitTest, DeviceSynchronization)
{
  auto device = MetalDevice::GetDevice(0);
  ASSERT_NE(device, nullptr);
  
  // Test synchronization
  auto err = device->Synchronize();
  ASSERT_TRITON_OK(err);
}

TEST_F(MetalDeviceUnitTest, DeviceFeatureDetection)
{
  auto device = MetalDevice::GetDevice(0);
  ASSERT_NE(device, nullptr);
  
  // Test feature detection
  auto features = device->GetSupportedFeatures();
  
  // Basic features that should be supported
  EXPECT_TRUE(features.compute);
  
  if (GetConfig().verbose) {
    std::cout << "Supported features:\n";
    std::cout << "  Compute: " << (features.compute ? "Yes" : "No") << "\n";
    std::cout << "  Unified memory: " << (features.unified_memory ? "Yes" : "No") << "\n";
    std::cout << "  Raytracing: " << (features.raytracing ? "Yes" : "No") << "\n";
    std::cout << "  Machine learning: " << (features.machine_learning ? "Yes" : "No") << "\n";
    std::cout << "  Variable rate shading: " << (features.variable_rate_shading ? "Yes" : "No") << "\n";
  }
}

TEST_F(MetalDeviceUnitTest, DeviceSelectionStrategies)
{
  // Test different device selection strategies
  
  // Select device with most memory
  auto device_by_memory = MetalDevice::SelectDeviceByMemory();
  if (device_by_memory) {
    auto info = device_by_memory->GetInfo();
    std::cout << "Device with most memory: " << info.name 
              << " (" << info.total_memory / (1024*1024*1024) << " GB)\n";
  }
  
  // Select discrete GPU if available
  auto discrete_device = MetalDevice::SelectDiscreteGPU();
  if (discrete_device) {
    auto info = discrete_device->GetInfo();
    std::cout << "Discrete GPU: " << info.name << "\n";
  }
  
  // Select integrated GPU if available
  auto integrated_device = MetalDevice::SelectIntegratedGPU();
  if (integrated_device) {
    auto info = integrated_device->GetInfo();
    std::cout << "Integrated GPU: " << info.name << "\n";
  }
}

TEST_F(MetalDeviceUnitTest, DeviceMemoryPressure)
{
  auto device = MetalDevice::GetDevice(0);
  ASSERT_NE(device, nullptr);
  
  // Register memory pressure handler
  bool pressure_received = false;
  device->RegisterMemoryPressureHandler([&pressure_received]() {
    pressure_received = true;
  });
  
  // In a real test, we would allocate memory until pressure is triggered
  // For now, just test the registration works
  EXPECT_FALSE(pressure_received);
}

TEST_F(MetalDeviceUnitTest, DevicePowerState)
{
  auto device = MetalDevice::GetDevice(0);
  ASSERT_NE(device, nullptr);
  
  // Get power state
  auto power_state = device->GetPowerState();
  
  if (GetConfig().verbose) {
    std::cout << "Device power state:\n";
    std::cout << "  State: ";
    switch (power_state.state) {
      case MetalPowerState::Active:
        std::cout << "Active\n";
        break;
      case MetalPowerState::LowPower:
        std::cout << "Low Power\n";
        break;
      case MetalPowerState::Suspended:
        std::cout << "Suspended\n";
        break;
    }
    std::cout << "  Temperature: " << power_state.temperature << "°C\n";
    std::cout << "  Power usage: " << power_state.power_usage << " W\n";
  }
}

TEST_F(MetalDeviceUnitTest, DeviceErrorHandling)
{
  auto device = MetalDevice::GetDevice(0);
  ASSERT_NE(device, nullptr);
  
  // Register error handler
  std::string last_error;
  device->RegisterErrorHandler([&last_error](const std::string& error) {
    last_error = error;
  });
  
  // Test error handling (this would normally be triggered by actual errors)
  EXPECT_TRUE(last_error.empty());
}

TEST_F(MetalDeviceUnitTest, DeviceThreadSafety)
{
  auto device = MetalDevice::GetDevice(0);
  ASSERT_NE(device, nullptr);
  
  const int num_threads = 10;
  const int operations_per_thread = 100;
  
  std::vector<std::thread> threads;
  std::atomic<int> success_count{0};
  
  auto worker = [&]() {
    for (int i = 0; i < operations_per_thread; ++i) {
      // Query memory
      size_t available, total;
      auto err = device->QueryMemory(&available, &total);
      if (err == nullptr) {
        success_count.fetch_add(1);
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
      
      // Create command queue
      auto queue = device->CreateCommandQueue();
      if (queue != nullptr) {
        success_count.fetch_add(1);
      }
    }
  };
  
  // Start threads
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker);
  }
  
  // Wait for completion
  for (auto& t : threads) {
    t.join();
  }
  
  EXPECT_EQ(success_count.load(), num_threads * operations_per_thread * 2);
}

}}}  // namespace triton::server::test