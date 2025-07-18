// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifdef TRITON_ENABLE_METAL

#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "../metal/metal_unified_memory.h"
#include "../metal/metal_memory.h"
#include "../status.h"

using namespace triton::core;

// Benchmark configuration
struct BenchmarkConfig {
  size_t buffer_size;
  size_t num_iterations;
  UnifiedMemoryPattern access_pattern;
  bool use_unified_optimization;
  int num_threads;
};

// Benchmark results
struct BenchmarkResult {
  std::string test_name;
  double total_time_ms;
  double avg_time_ms;
  double throughput_gbps;
  size_t transfers_eliminated;
  size_t bytes_saved;
};

class UnifiedMemoryBenchmark {
 public:
  UnifiedMemoryBenchmark() {
    // Initialize the unified memory optimizer
    UnifiedMemoryConfig config;
    config.enable_auto_placement = true;
    config.enable_zero_copy = true;
    config.enable_prefetching = true;
    config.enable_numa_optimization = true;
    
    auto status = UnifiedMemoryOptimizer::Initialize(config);
    if (!status.IsOk()) {
      std::cerr << "Failed to initialize unified memory optimizer: " 
                << status.Message() << std::endl;
      exit(1);
    }
  }
  
  ~UnifiedMemoryBenchmark() {
    UnifiedMemoryOptimizer::Shutdown();
  }
  
  void RunAllBenchmarks() {
    std::vector<BenchmarkResult> results;
    
    // Test different buffer sizes
    std::vector<size_t> buffer_sizes = {
      1 * 1024 * 1024,      // 1 MB
      16 * 1024 * 1024,     // 16 MB
      64 * 1024 * 1024,     // 64 MB
      256 * 1024 * 1024,    // 256 MB
      1024 * 1024 * 1024    // 1 GB
    };
    
    // Test different access patterns
    std::vector<UnifiedMemoryPattern> patterns = {
      UnifiedMemoryPattern::CPU_DOMINANT,
      UnifiedMemoryPattern::GPU_DOMINANT,
      UnifiedMemoryPattern::BALANCED,
      UnifiedMemoryPattern::STREAMING
    };
    
    std::cout << "Running Unified Memory Optimization Benchmarks...\n\n";
    
    // 1. Zero-copy tensor creation benchmark
    for (size_t size : buffer_sizes) {
      results.push_back(BenchmarkZeroCopyTensorCreation(size));
    }
    
    // 2. Memory transfer elimination benchmark
    for (size_t size : buffer_sizes) {
      results.push_back(BenchmarkTransferElimination(size, false)); // Without optimization
      results.push_back(BenchmarkTransferElimination(size, true));  // With optimization
    }
    
    // 3. Access pattern optimization benchmark
    for (auto pattern : patterns) {
      for (size_t size : {16 * 1024 * 1024, 256 * 1024 * 1024}) {
        results.push_back(BenchmarkAccessPatternOptimization(size, pattern));
      }
    }
    
    // 4. Memory pool performance benchmark
    results.push_back(BenchmarkMemoryPool(1000, 1 * 1024 * 1024));
    results.push_back(BenchmarkMemoryPool(1000, 16 * 1024 * 1024));
    
    // 5. NUMA optimization benchmark (Mac Studio)
    if (IsNUMASystem()) {
      results.push_back(BenchmarkNUMAOptimization(256 * 1024 * 1024));
    }
    
    // 6. Concurrent access benchmark
    for (int threads : {1, 2, 4, 8}) {
      results.push_back(BenchmarkConcurrentAccess(64 * 1024 * 1024, threads));
    }
    
    // Print results
    PrintResults(results);
  }
  
 private:
  BenchmarkResult BenchmarkZeroCopyTensorCreation(size_t size) {
    BenchmarkResult result;
    result.test_name = "Zero-Copy Tensor Creation (" + FormatSize(size) + ")";
    
    const size_t iterations = 100;
    std::vector<float> cpu_data(size / sizeof(float));
    
    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& val : cpu_data) {
      val = dist(gen);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
      std::unique_ptr<ZeroCopyTensor> tensor;
      std::vector<int64_t> shape = {static_cast<int64_t>(cpu_data.size())};
      
      auto status = ZeroCopyTensor::CreateFromCPUMemory(
          tensor, cpu_data.data(), size, shape, TRITONSERVER_TYPE_FP32);
      
      if (!status.IsOk()) {
        std::cerr << "Failed to create zero-copy tensor: " 
                  << status.Message() << std::endl;
        break;
      }
      
      // Simulate GPU access
      ScopedMemoryAccess access(tensor->Data(), size, false, true);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    result.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.avg_time_ms = result.total_time_ms / iterations;
    result.throughput_gbps = (size * iterations / (1024.0 * 1024.0 * 1024.0)) / 
                            (result.total_time_ms / 1000.0);
    
    // Get transfer statistics
    TransferEliminationTracker::GetStatistics(
        result.transfers_eliminated, result.bytes_saved,
        result.transfers_eliminated, result.transfers_eliminated);
    
    return result;
  }
  
  BenchmarkResult BenchmarkTransferElimination(size_t size, bool use_optimization) {
    BenchmarkResult result;
    result.test_name = "Memory Transfer " + std::string(use_optimization ? "WITH" : "WITHOUT") + 
                      " Optimization (" + FormatSize(size) + ")";
    
    const size_t iterations = 50;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if (use_optimization) {
      // With unified memory optimization
      for (size_t i = 0; i < iterations; ++i) {
        std::unique_ptr<MetalBuffer> buffer;
        auto status = UnifiedMemoryOptimizer::AllocateOptimized(
            buffer, size, UnifiedMemoryPattern::BALANCED);
        
        if (!status.IsOk()) {
          std::cerr << "Failed to allocate optimized buffer: " 
                    << status.Message() << std::endl;
          break;
        }
        
        // Simulate data processing
        void* data = buffer->Data();
        if (data) {
          // CPU write
          ScopedMemoryAccess cpu_write(data, size, true, false);
          memset(data, 0xAA, size);
          
          // GPU read
          ScopedMemoryAccess gpu_read(data, size, false, true);
          
          // GPU write
          ScopedMemoryAccess gpu_write(data, size, false, false);
          
          // CPU read
          ScopedMemoryAccess cpu_read(data, size, true, true);
          volatile char dummy = static_cast<char*>(data)[0];
          (void)dummy;
        }
      }
    } else {
      // Without optimization - simulate traditional copy
      for (size_t i = 0; i < iterations; ++i) {
        // Allocate CPU memory
        std::vector<char> cpu_buffer(size);
        
        // Allocate GPU memory
        std::unique_ptr<MetalBuffer> gpu_buffer;
        auto status = MetalBuffer::Create(
            gpu_buffer, size, MetalMemoryType::METAL_BUFFER);
        
        if (!status.IsOk()) {
          std::cerr << "Failed to allocate GPU buffer: " 
                    << status.Message() << std::endl;
          break;
        }
        
        // CPU write
        memset(cpu_buffer.data(), 0xAA, size);
        
        // Copy to GPU
        gpu_buffer->CopyFromHost(cpu_buffer.data(), size);
        
        // GPU processing (simulated)
        
        // Copy back to CPU
        gpu_buffer->CopyToHost(cpu_buffer.data(), size);
        
        // CPU read
        volatile char dummy = cpu_buffer[0];
        (void)dummy;
      }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    result.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.avg_time_ms = result.total_time_ms / iterations;
    result.throughput_gbps = (size * iterations * 2 / (1024.0 * 1024.0 * 1024.0)) / 
                            (result.total_time_ms / 1000.0);
    
    if (use_optimization) {
      TransferEliminationTracker::GetStatistics(
          result.transfers_eliminated, result.bytes_saved,
          result.transfers_eliminated, result.transfers_eliminated);
    }
    
    return result;
  }
  
  BenchmarkResult BenchmarkAccessPatternOptimization(
      size_t size, UnifiedMemoryPattern pattern) {
    BenchmarkResult result;
    result.test_name = "Access Pattern Optimization - " + 
                      PatternToString(pattern) + " (" + FormatSize(size) + ")";
    
    const size_t iterations = 100;
    
    // Allocate with specific pattern hint
    std::unique_ptr<MetalBuffer> buffer;
    auto status = UnifiedMemoryOptimizer::AllocateOptimized(
        buffer, size, pattern);
    
    if (!status.IsOk()) {
      std::cerr << "Failed to allocate buffer: " << status.Message() << std::endl;
      return result;
    }
    
    void* data = buffer->Data();
    if (!data) {
      std::cerr << "Failed to get buffer data pointer" << std::endl;
      return result;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Simulate access pattern
    for (size_t i = 0; i < iterations; ++i) {
      switch (pattern) {
        case UnifiedMemoryPattern::CPU_DOMINANT:
          // 80% CPU, 20% GPU
          for (int j = 0; j < 8; ++j) {
            ScopedMemoryAccess access(data, size, true, j % 2 == 0);
            ProcessData(data, size);
          }
          for (int j = 0; j < 2; ++j) {
            ScopedMemoryAccess access(data, size, false, true);
          }
          break;
          
        case UnifiedMemoryPattern::GPU_DOMINANT:
          // 20% CPU, 80% GPU
          for (int j = 0; j < 2; ++j) {
            ScopedMemoryAccess access(data, size, true, true);
            ProcessData(data, size);
          }
          for (int j = 0; j < 8; ++j) {
            ScopedMemoryAccess access(data, size, false, j % 2 == 0);
          }
          break;
          
        case UnifiedMemoryPattern::BALANCED:
          // 50% CPU, 50% GPU
          for (int j = 0; j < 5; ++j) {
            ScopedMemoryAccess cpu_access(data, size, true, j % 2 == 0);
            ProcessData(data, size);
            ScopedMemoryAccess gpu_access(data, size, false, j % 2 == 1);
          }
          break;
          
        case UnifiedMemoryPattern::STREAMING:
          // Sequential access pattern
          ScopedMemoryAccess write_access(data, size, true, false);
          ProcessData(data, size);
          ScopedMemoryAccess read_access(data, size, false, true);
          break;
          
        default:
          break;
      }
      
      // Trigger optimization check
      UnifiedMemoryOptimizer::OptimizePlacement(data);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    result.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.avg_time_ms = result.total_time_ms / iterations;
    result.throughput_gbps = (size * iterations / (1024.0 * 1024.0 * 1024.0)) / 
                            (result.total_time_ms / 1000.0);
    
    return result;
  }
  
  BenchmarkResult BenchmarkMemoryPool(size_t num_allocations, size_t size) {
    BenchmarkResult result;
    result.test_name = "Memory Pool Performance (" + 
                      std::to_string(num_allocations) + " allocations of " + 
                      FormatSize(size) + ")";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::unique_ptr<MetalBuffer>> buffers;
    
    // Allocate and free multiple times to test pool
    for (size_t round = 0; round < 10; ++round) {
      // Allocate
      for (size_t i = 0; i < num_allocations; ++i) {
        std::unique_ptr<MetalBuffer> buffer;
        auto status = UnifiedMemoryOptimizer::GetPooledBuffer(
            buffer, size, UnifiedMemoryPattern::BALANCED);
        
        if (!status.IsOk()) {
          std::cerr << "Failed to get pooled buffer: " 
                    << status.Message() << std::endl;
          break;
        }
        
        buffers.push_back(std::move(buffer));
      }
      
      // Return to pool
      for (auto& buffer : buffers) {
        UnifiedMemoryOptimizer::ReturnToPool(std::move(buffer));
      }
      buffers.clear();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    result.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.avg_time_ms = result.total_time_ms / (num_allocations * 10);
    
    return result;
  }
  
  BenchmarkResult BenchmarkNUMAOptimization(size_t size) {
    BenchmarkResult result;
    result.test_name = "NUMA Optimization (" + FormatSize(size) + ")";
    
    const size_t iterations = 50;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
      // Allocate on specific NUMA node
      std::unique_ptr<MetalBuffer> buffer;
      auto status = UnifiedMemoryOptimizer::AllocateNUMAOptimized(
          buffer, size, i % 2);  // Alternate between nodes
      
      if (!status.IsOk()) {
        std::cerr << "Failed to allocate NUMA-optimized buffer: " 
                  << status.Message() << std::endl;
        break;
      }
      
      // Process data
      void* data = buffer->Data();
      if (data) {
        ProcessData(data, size);
      }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    result.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.avg_time_ms = result.total_time_ms / iterations;
    result.throughput_gbps = (size * iterations / (1024.0 * 1024.0 * 1024.0)) / 
                            (result.total_time_ms / 1000.0);
    
    return result;
  }
  
  BenchmarkResult BenchmarkConcurrentAccess(size_t size, int num_threads) {
    BenchmarkResult result;
    result.test_name = "Concurrent Access (" + std::to_string(num_threads) + 
                      " threads, " + FormatSize(size) + ")";
    
    // Allocate shared buffer
    std::unique_ptr<MetalBuffer> buffer;
    auto status = UnifiedMemoryOptimizer::AllocateOptimized(
        buffer, size, UnifiedMemoryPattern::BALANCED);
    
    if (!status.IsOk()) {
      std::cerr << "Failed to allocate buffer: " << status.Message() << std::endl;
      return result;
    }
    
    void* data = buffer->Data();
    if (!data) {
      std::cerr << "Failed to get buffer data pointer" << std::endl;
      return result;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    size_t chunk_size = size / num_threads;
    
    for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back([=]() {
        size_t offset = t * chunk_size;
        void* chunk_data = static_cast<char*>(data) + offset;
        
        for (int i = 0; i < 100; ++i) {
          // Alternate between CPU and GPU access
          if (i % 2 == 0) {
            ScopedMemoryAccess access(chunk_data, chunk_size, true, false);
            ProcessData(chunk_data, chunk_size);
          } else {
            ScopedMemoryAccess access(chunk_data, chunk_size, false, true);
          }
        }
      });
    }
    
    for (auto& t : threads) {
      t.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    result.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.throughput_gbps = (size * 100 / (1024.0 * 1024.0 * 1024.0)) / 
                            (result.total_time_ms / 1000.0);
    
    return result;
  }
  
  void ProcessData(void* data, size_t size) {
    // Simulate data processing
    char* bytes = static_cast<char*>(data);
    for (size_t i = 0; i < size; i += 4096) {
      bytes[i] = bytes[i] ^ 0xFF;
    }
  }
  
  bool IsNUMASystem() {
    // Simple check for Mac Studio (high memory systems)
#ifdef __APPLE__
    return sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE) > 64ULL * 1024 * 1024 * 1024;
#else
    return false;
#endif
  }
  
  std::string FormatSize(size_t size) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit_idx = 0;
    double value = static_cast<double>(size);
    
    while (value >= 1024.0 && unit_idx < 3) {
      value /= 1024.0;
      unit_idx++;
    }
    
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%.1f%s", value, units[unit_idx]);
    return std::string(buffer);
  }
  
  std::string PatternToString(UnifiedMemoryPattern pattern) {
    switch (pattern) {
      case UnifiedMemoryPattern::CPU_DOMINANT: return "CPU Dominant";
      case UnifiedMemoryPattern::GPU_DOMINANT: return "GPU Dominant";
      case UnifiedMemoryPattern::BALANCED: return "Balanced";
      case UnifiedMemoryPattern::STREAMING: return "Streaming";
      case UnifiedMemoryPattern::UNKNOWN: return "Unknown";
      default: return "Invalid";
    }
  }
  
  void PrintResults(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n=== Benchmark Results ===\n\n";
    std::cout << std::left;
    std::cout.width(60);
    std::cout << "Test Name";
    std::cout.width(15);
    std::cout << "Total (ms)";
    std::cout.width(15);
    std::cout << "Avg (ms)";
    std::cout.width(15);
    std::cout << "Throughput";
    std::cout.width(20);
    std::cout << "Transfers Saved";
    std::cout.width(15);
    std::cout << "Bytes Saved";
    std::cout << "\n";
    std::cout << std::string(140, '-') << "\n";
    
    for (const auto& result : results) {
      std::cout.width(60);
      std::cout << result.test_name;
      std::cout.width(15);
      std::cout << std::fixed << std::setprecision(2) << result.total_time_ms;
      std::cout.width(15);
      std::cout << std::fixed << std::setprecision(4) << result.avg_time_ms;
      std::cout.width(15);
      if (result.throughput_gbps > 0) {
        std::cout << std::fixed << std::setprecision(2) << result.throughput_gbps << " GB/s";
      } else {
        std::cout << "N/A";
      }
      std::cout.width(20);
      std::cout << result.transfers_eliminated;
      std::cout.width(15);
      std::cout << FormatSize(result.bytes_saved);
      std::cout << "\n";
    }
    
    // Print summary statistics
    std::cout << "\n=== Summary ===\n";
    
    size_t total_transfers_eliminated, total_bytes_saved, cpu_to_gpu, gpu_to_cpu;
    TransferEliminationTracker::GetStatistics(
        total_transfers_eliminated, total_bytes_saved, cpu_to_gpu, gpu_to_cpu);
    
    std::cout << "Total Transfers Eliminated: " << total_transfers_eliminated << "\n";
    std::cout << "Total Bytes Saved: " << FormatSize(total_bytes_saved) << "\n";
    std::cout << "CPU->GPU Transfers Eliminated: " << cpu_to_gpu << "\n";
    std::cout << "GPU->CPU Transfers Eliminated: " << gpu_to_cpu << "\n";
    
    // Get memory statistics
    size_t total_allocated, unified_memory_used;
    std::unordered_map<UnifiedMemoryPattern, size_t> pattern_distribution;
    
    UnifiedMemoryOptimizer::GetMemoryStats(
        total_allocated, unified_memory_used, total_transfers_eliminated, pattern_distribution);
    
    std::cout << "\nMemory Usage:\n";
    std::cout << "Total Allocated: " << FormatSize(total_allocated) << "\n";
    std::cout << "Unified Memory Used: " << FormatSize(unified_memory_used) << "\n";
    
    std::cout << "\nPattern Distribution:\n";
    for (const auto& [pattern, size] : pattern_distribution) {
      std::cout << PatternToString(pattern) << ": " << FormatSize(size) << "\n";
    }
  }
};

int main(int argc, char* argv[]) {
  std::cout << "Apple Silicon Unified Memory Optimization Benchmark\n";
  std::cout << "==================================================\n\n";
  
  // Check if Metal is available
  if (!MetalMemoryManager::IsAvailable()) {
    std::cerr << "Metal is not available on this system\n";
    return 1;
  }
  
  // Print device information
  size_t device_count = MetalMemoryManager::DeviceCount();
  std::cout << "Found " << device_count << " Metal device(s)\n";
  
  for (size_t i = 0; i < device_count; ++i) {
    std::string device_name;
    size_t total_memory, available_memory;
    
    MetalMemoryManager::GetDeviceName(i, device_name);
    MetalMemoryManager::GetDeviceMemoryInfo(i, total_memory, available_memory);
    
    std::cout << "Device " << i << ": " << device_name << "\n";
    std::cout << "  Total Memory: " << (total_memory / (1024.0 * 1024.0 * 1024.0)) 
              << " GB\n";
    std::cout << "  Available Memory: " << (available_memory / (1024.0 * 1024.0 * 1024.0)) 
              << " GB\n";
  }
  
  std::cout << "\n";
  
  // Run benchmarks
  UnifiedMemoryBenchmark benchmark;
  benchmark.RunAllBenchmarks();
  
  // Dump profiling data if requested
  if (argc > 1 && std::string(argv[1]) == "--profile") {
    UnifiedMemoryOptimizer::EnableProfiling(true);
    UnifiedMemoryOptimizer::DumpProfilingData("unified_memory_profile.csv");
    std::cout << "\nProfiling data saved to unified_memory_profile.csv\n";
  }
  
  return 0;
}

#endif  // TRITON_ENABLE_METAL