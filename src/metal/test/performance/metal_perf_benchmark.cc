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
#include "src/metal/metal_allocator.h"
#include "src/metal/kernels/metal_kernel_library.h"

#include <iomanip>
#include <numeric>

namespace triton { namespace server { namespace test {

class MetalPerformanceBenchmark : public MetalBenchmarkTest {
 protected:
  struct BenchmarkResult {
    std::string name;
    double avg_ms;
    double min_ms;
    double max_ms;
    double std_dev;
    double throughput_gbps;
    double gflops;
  };
  
  void ReportBenchmark(const BenchmarkResult& result) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n" << result.name << ":\n";
    std::cout << "  Average: " << result.avg_ms << " ms\n";
    std::cout << "  Min: " << result.min_ms << " ms\n";
    std::cout << "  Max: " << result.max_ms << " ms\n";
    std::cout << "  Std Dev: " << result.std_dev << " ms\n";
    
    if (result.throughput_gbps > 0) {
      std::cout << "  Throughput: " << result.throughput_gbps << " GB/s\n";
    }
    
    if (result.gflops > 0) {
      std::cout << "  Performance: " << result.gflops << " GFLOPS\n";
    }
    
    // Also report to base class
    std::map<std::string, double> metrics = {
        {"avg_ms", result.avg_ms},
        {"min_ms", result.min_ms},
        {"max_ms", result.max_ms},
        {"std_dev", result.std_dev}
    };
    
    if (result.throughput_gbps > 0) {
      metrics["throughput_gbps"] = result.throughput_gbps;
    }
    
    if (result.gflops > 0) {
      metrics["gflops"] = result.gflops;
    }
    
    ReportResults(result.name, metrics);
  }
  
  BenchmarkResult RunTimedBenchmark(
      const std::string& name,
      int warmup_iterations,
      int benchmark_iterations,
      std::function<void()> func,
      size_t bytes_processed = 0,
      size_t flops = 0) {
    
    // Warmup
    for (int i = 0; i < warmup_iterations; ++i) {
      func();
    }
    
    // Benchmark
    std::vector<double> times;
    times.reserve(benchmark_iterations);
    
    for (int i = 0; i < benchmark_iterations; ++i) {
      PerformanceTimer timer;
      timer.Start();
      func();
      
#ifdef __APPLE__
      // Ensure GPU work completes
      @autoreleasepool {
        id<MTLCommandBuffer> sync_buffer = [command_queue_ commandBuffer];
        [sync_buffer commit];
        [sync_buffer waitUntilCompleted];
      }
#endif
      
      times.push_back(timer.ElapsedMs());
    }
    
    // Calculate statistics
    BenchmarkResult result;
    result.name = name;
    result.avg_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    result.min_ms = *std::min_element(times.begin(), times.end());
    result.max_ms = *std::max_element(times.begin(), times.end());
    
    // Standard deviation
    double variance = 0.0;
    for (double t : times) {
      variance += (t - result.avg_ms) * (t - result.avg_ms);
    }
    result.std_dev = std::sqrt(variance / times.size());
    
    // Throughput
    if (bytes_processed > 0) {
      result.throughput_gbps = (bytes_processed / 1e9) / (result.avg_ms / 1000.0);
    }
    
    // FLOPS
    if (flops > 0) {
      result.gflops = (flops / 1e9) / (result.avg_ms / 1000.0);
    }
    
    return result;
  }
};

TEST_F(MetalPerformanceBenchmark, MemoryAllocationBenchmark)
{
  std::vector<size_t> sizes = {
      1024,           // 1 KB
      1024 * 1024,    // 1 MB
      10 * 1024 * 1024,   // 10 MB
      100 * 1024 * 1024,  // 100 MB
      1024 * 1024 * 1024  // 1 GB
  };
  
  auto allocator = std::make_shared<MetalAllocator>(0);
  
  for (size_t size : sizes) {
    std::string name = "Allocation_" + std::to_string(size / (1024 * 1024)) + "MB";
    
    auto result = RunTimedBenchmark(
        name, 10, 100,
        [&]() {
          void* buffer = nullptr;
          MetalAllocation* allocation = nullptr;
          
          auto err = allocator->Allocate(size, &buffer, &allocation);
          if (err == nullptr) {
            allocator->Free(allocation);
          } else {
            TRITONSERVER_ErrorDelete(err);
          }
        });
    
    ReportBenchmark(result);
  }
}

TEST_F(MetalPerformanceBenchmark, MemoryPoolBenchmark)
{
  // Compare pooled vs non-pooled allocation
  const size_t alloc_size = 4 * 1024 * 1024;  // 4 MB
  const int iterations = 1000;
  
  // Non-pooled allocator
  {
    MetalPoolConfig config;
    config.size_classes = {};  // No pools
    auto allocator = std::make_shared<MetalAllocator>(0, config);
    
    auto result = RunTimedBenchmark(
        "Non-pooled_Allocation", 10, iterations,
        [&]() {
          void* buffer = nullptr;
          MetalAllocation* allocation = nullptr;
          
          auto err = allocator->Allocate(alloc_size, &buffer, &allocation);
          if (err == nullptr) {
            allocator->Free(allocation);
          } else {
            TRITONSERVER_ErrorDelete(err);
          }
        });
    
    ReportBenchmark(result);
  }
  
  // Pooled allocator
  {
    MetalPoolConfig config;
    config.size_classes = {alloc_size};
    config.initial_pool_sizes = {100};
    config.max_pool_sizes = {1000};
    
    auto allocator = std::make_shared<MetalAllocator>(0, config);
    
    auto result = RunTimedBenchmark(
        "Pooled_Allocation", 10, iterations,
        [&]() {
          void* buffer = nullptr;
          MetalAllocation* allocation = nullptr;
          
          auto err = allocator->Allocate(alloc_size, &buffer, &allocation);
          if (err == nullptr) {
            allocator->Free(allocation);
          } else {
            TRITONSERVER_ErrorDelete(err);
          }
        });
    
    ReportBenchmark(result);
  }
}

TEST_F(MetalPerformanceBenchmark, MemoryCopyBenchmark)
{
#ifdef __APPLE__
  std::vector<size_t> sizes = {
      1024 * 1024,        // 1 MB
      16 * 1024 * 1024,   // 16 MB
      64 * 1024 * 1024,   // 64 MB
      256 * 1024 * 1024   // 256 MB
  };
  
  for (size_t size : sizes) {
    // Create buffers
    id<MTLBuffer> src_buffer = CreateTestBuffer(device_, size);
    id<MTLBuffer> dst_buffer = CreateTestBuffer(device_, size);
    
    ASSERT_NE(src_buffer, nil);
    ASSERT_NE(dst_buffer, nil);
    
    // Fill source with test data
    TestDataGenerator gen;
    auto test_data = gen.GenerateFloatData(size / sizeof(float));
    memcpy([src_buffer contents], test_data.data(), size);
    
    std::string name = "MemoryCopy_" + std::to_string(size / (1024 * 1024)) + "MB";
    
    auto result = RunTimedBenchmark(
        name, 10, 100,
        [&]() {
          @autoreleasepool {
            id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
            id<MTLBlitCommandEncoder> blit_encoder = [command_buffer blitCommandEncoder];
            
            [blit_encoder copyFromBuffer:src_buffer
                            sourceOffset:0
                                toBuffer:dst_buffer
                       destinationOffset:0
                                    size:size];
            
            [blit_encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
          }
        },
        size * 2);  // Read + write
    
    ReportBenchmark(result);
  }
#endif
}

TEST_F(MetalPerformanceBenchmark, KernelExecutionBenchmark)
{
#ifdef __APPLE__
  // Benchmark various kernel operations
  
  // 1. Vector addition
  {
    const size_t vector_size = 16 * 1024 * 1024;  // 16M elements
    const size_t bytes = vector_size * sizeof(float);
    
    // Create buffers
    id<MTLBuffer> a_buffer = CreateTestBuffer(device_, bytes);
    id<MTLBuffer> b_buffer = CreateTestBuffer(device_, bytes);
    id<MTLBuffer> c_buffer = CreateTestBuffer(device_, bytes);
    
    // Fill with test data
    TestDataGenerator gen;
    auto a_data = gen.GenerateFloatData(vector_size);
    auto b_data = gen.GenerateFloatData(vector_size);
    
    memcpy([a_buffer contents], a_data.data(), bytes);
    memcpy([b_buffer contents], b_data.data(), bytes);
    
    // Create compute pipeline
    NSString* kernel_source = @R"(
      #include <metal_stdlib>
      using namespace metal;
      
      kernel void vector_add(
          device const float* a [[buffer(0)]],
          device const float* b [[buffer(1)]],
          device float* c [[buffer(2)]],
          uint index [[thread_position_in_grid]])
      {
          c[index] = a[index] + b[index];
      }
    )";
    
    auto pipeline = CreateTestComputePipeline(device_, 
        [kernel_source UTF8String], "vector_add");
    ASSERT_NE(pipeline, nil);
    
    auto result = RunTimedBenchmark(
        "VectorAdd_16M", 10, 100,
        [&]() {
          @autoreleasepool {
            id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:a_buffer offset:0 atIndex:0];
            [encoder setBuffer:b_buffer offset:0 atIndex:1];
            [encoder setBuffer:c_buffer offset:0 atIndex:2];
            
            MTLSize grid_size = MTLSizeMake(vector_size, 1, 1);
            MTLSize thread_group_size = MTLSizeMake(
                pipeline.maxTotalThreadsPerThreadgroup, 1, 1);
            
            [encoder dispatchThreads:grid_size
                threadsPerThreadgroup:thread_group_size];
            
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
          }
        },
        bytes * 3,  // Read a, b; write c
        vector_size);  // One add per element
    
    ReportBenchmark(result);
  }
  
  // 2. Matrix multiplication (GEMM)
  {
    const int M = 1024, N = 1024, K = 1024;
    const size_t a_size = M * K * sizeof(float);
    const size_t b_size = K * N * sizeof(float);
    const size_t c_size = M * N * sizeof(float);
    
    // Create buffers
    id<MTLBuffer> a_buffer = CreateTestBuffer(device_, a_size);
    id<MTLBuffer> b_buffer = CreateTestBuffer(device_, b_size);
    id<MTLBuffer> c_buffer = CreateTestBuffer(device_, c_size);
    
    // Fill with test data
    TestDataGenerator gen;
    auto a_data = gen.GenerateFloatData(M * K);
    auto b_data = gen.GenerateFloatData(K * N);
    
    memcpy([a_buffer contents], a_data.data(), a_size);
    memcpy([b_buffer contents], b_data.data(), b_size);
    
    // Use MPS for GEMM
    MPSMatrixDescriptor* a_desc = [MPSMatrixDescriptor 
        matrixDescriptorWithRows:M columns:K 
        rowBytes:K*sizeof(float) dataType:MPSDataTypeFloat32];
    
    MPSMatrixDescriptor* b_desc = [MPSMatrixDescriptor 
        matrixDescriptorWithRows:K columns:N 
        rowBytes:N*sizeof(float) dataType:MPSDataTypeFloat32];
    
    MPSMatrixDescriptor* c_desc = [MPSMatrixDescriptor 
        matrixDescriptorWithRows:M columns:N 
        rowBytes:N*sizeof(float) dataType:MPSDataTypeFloat32];
    
    MPSMatrix* a_matrix = [[MPSMatrix alloc] initWithBuffer:a_buffer 
                                                  descriptor:a_desc];
    MPSMatrix* b_matrix = [[MPSMatrix alloc] initWithBuffer:b_buffer 
                                                  descriptor:b_desc];
    MPSMatrix* c_matrix = [[MPSMatrix alloc] initWithBuffer:c_buffer 
                                                  descriptor:c_desc];
    
    MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc]
        initWithDevice:device_ transposeLeft:NO transposeRight:NO
        resultRows:M resultColumns:N interiorColumns:K
        alpha:1.0 beta:0.0];
    
    auto result = RunTimedBenchmark(
        "GEMM_1024x1024x1024", 10, 50,
        [&]() {
          @autoreleasepool {
            id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
            
            [gemm encodeToCommandBuffer:command_buffer
                             leftMatrix:a_matrix
                            rightMatrix:b_matrix
                           resultMatrix:c_matrix];
            
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
          }
        },
        a_size + b_size + c_size,
        2L * M * N * K);  // 2 * M * N * K FLOPs for GEMM
    
    ReportBenchmark(result);
  }
  
  // 3. Convolution
  {
    const int batch = 1, in_channels = 64, out_channels = 128;
    const int height = 56, width = 56;
    const int kernel_h = 3, kernel_w = 3;
    
    size_t input_size = batch * in_channels * height * width * sizeof(float);
    size_t weight_size = out_channels * in_channels * kernel_h * kernel_w * sizeof(float);
    size_t output_size = batch * out_channels * height * width * sizeof(float);
    
    // Create buffers
    id<MTLBuffer> input_buffer = CreateTestBuffer(device_, input_size);
    id<MTLBuffer> weight_buffer = CreateTestBuffer(device_, weight_size);
    id<MTLBuffer> output_buffer = CreateTestBuffer(device_, output_size);
    
    // Fill with test data
    TestDataGenerator gen;
    auto input_data = gen.GenerateFloatData(input_size / sizeof(float));
    auto weight_data = gen.GenerateFloatData(weight_size / sizeof(float));
    
    memcpy([input_buffer contents], input_data.data(), input_size);
    memcpy([weight_buffer contents], weight_data.data(), weight_size);
    
    // Create convolution descriptor
    MPSCNNConvolutionDescriptor* conv_desc = [MPSCNNConvolutionDescriptor
        cnnConvolutionDescriptorWithKernelWidth:kernel_w
                                    kernelHeight:kernel_h
                            inputFeatureChannels:in_channels
                           outputFeatureChannels:out_channels];
    
    conv_desc.strideInPixelsX = 1;
    conv_desc.strideInPixelsY = 1;
    
    // Create convolution
    MPSCNNConvolution* convolution = [[MPSCNNConvolution alloc]
        initWithDevice:device_
        convolutionDescriptor:conv_desc
        kernelWeights:[weight_buffer contents]
        biasTerms:nil
        flags:MPSCNNConvolutionFlagsNone];
    
    convolution.padding = [MPSNNDefaultPadding paddingWithMethod:MPSNNPaddingMethodAddZeros];
    
    // Create image descriptors
    MPSImageDescriptor* input_desc = [MPSImageDescriptor
        imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
        width:width height:height featureChannels:in_channels];
    
    MPSImageDescriptor* output_desc = [MPSImageDescriptor
        imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
        width:width height:height featureChannels:out_channels];
    
    MPSImage* input_image = [[MPSImage alloc] initWithDevice:device_ 
                                               imageDescriptor:input_desc];
    MPSImage* output_image = [[MPSImage alloc] initWithDevice:device_ 
                                                imageDescriptor:output_desc];
    
    auto result = RunTimedBenchmark(
        "Conv2D_3x3_64to128", 10, 50,
        [&]() {
          @autoreleasepool {
            id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
            
            // Copy input data to image
            id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
            [blit copyFromBuffer:input_buffer sourceOffset:0
                    toTexture:input_image.texture
                    destinationSlice:0 destinationLevel:0
                    destinationOrigin:MTLOriginMake(0, 0, 0)
                    sourceSize:MTLSizeMake(width, height, in_channels)];
            [blit endEncoding];
            
            // Perform convolution
            [convolution encodeToCommandBuffer:command_buffer
                                   sourceImage:input_image
                              destinationImage:output_image];
            
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
          }
        },
        input_size + weight_size + output_size,
        2L * batch * out_channels * height * width * in_channels * kernel_h * kernel_w);
    
    ReportBenchmark(result);
  }
#endif
}

TEST_F(MetalPerformanceBenchmark, DataTypePerformance)
{
#ifdef __APPLE__
  // Compare performance of different data types
  const size_t num_elements = 16 * 1024 * 1024;  // 16M elements
  
  // Float32 performance
  {
    size_t bytes = num_elements * sizeof(float);
    id<MTLBuffer> a_buffer = CreateTestBuffer(device_, bytes);
    id<MTLBuffer> b_buffer = CreateTestBuffer(device_, bytes);
    id<MTLBuffer> c_buffer = CreateTestBuffer(device_, bytes);
    
    NSString* kernel_source = @R"(
      #include <metal_stdlib>
      using namespace metal;
      
      kernel void add_float32(
          device const float* a [[buffer(0)]],
          device const float* b [[buffer(1)]],
          device float* c [[buffer(2)]],
          uint index [[thread_position_in_grid]])
      {
          c[index] = a[index] + b[index];
      }
    )";
    
    auto pipeline = CreateTestComputePipeline(device_, 
        [kernel_source UTF8String], "add_float32");
    
    auto result = RunTimedBenchmark(
        "Float32_Addition", 10, 100,
        [&]() {
          @autoreleasepool {
            id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:a_buffer offset:0 atIndex:0];
            [encoder setBuffer:b_buffer offset:0 atIndex:1];
            [encoder setBuffer:c_buffer offset:0 atIndex:2];
            
            MTLSize grid_size = MTLSizeMake(num_elements, 1, 1);
            MTLSize thread_group_size = MTLSizeMake(256, 1, 1);
            
            [encoder dispatchThreads:grid_size
                threadsPerThreadgroup:thread_group_size];
            
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
          }
        },
        bytes * 3, num_elements);
    
    ReportBenchmark(result);
  }
  
  // Float16 performance
  if (capabilities_.supports_fp16) {
    size_t bytes = num_elements * sizeof(uint16_t);
    id<MTLBuffer> a_buffer = CreateTestBuffer(device_, bytes);
    id<MTLBuffer> b_buffer = CreateTestBuffer(device_, bytes);
    id<MTLBuffer> c_buffer = CreateTestBuffer(device_, bytes);
    
    NSString* kernel_source = @R"(
      #include <metal_stdlib>
      using namespace metal;
      
      kernel void add_float16(
          device const half* a [[buffer(0)]],
          device const half* b [[buffer(1)]],
          device half* c [[buffer(2)]],
          uint index [[thread_position_in_grid]])
      {
          c[index] = a[index] + b[index];
      }
    )";
    
    auto pipeline = CreateTestComputePipeline(device_, 
        [kernel_source UTF8String], "add_float16");
    
    auto result = RunTimedBenchmark(
        "Float16_Addition", 10, 100,
        [&]() {
          @autoreleasepool {
            id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:a_buffer offset:0 atIndex:0];
            [encoder setBuffer:b_buffer offset:0 atIndex:1];
            [encoder setBuffer:c_buffer offset:0 atIndex:2];
            
            MTLSize grid_size = MTLSizeMake(num_elements, 1, 1);
            MTLSize thread_group_size = MTLSizeMake(256, 1, 1);
            
            [encoder dispatchThreads:grid_size
                threadsPerThreadgroup:thread_group_size];
            
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
          }
        },
        bytes * 3, num_elements);
    
    ReportBenchmark(result);
  }
  
  // Int8 performance
  if (capabilities_.supports_int8) {
    size_t bytes = num_elements * sizeof(int8_t);
    id<MTLBuffer> a_buffer = CreateTestBuffer(device_, bytes);
    id<MTLBuffer> b_buffer = CreateTestBuffer(device_, bytes);
    id<MTLBuffer> c_buffer = CreateTestBuffer(device_, bytes);
    
    NSString* kernel_source = @R"(
      #include <metal_stdlib>
      using namespace metal;
      
      kernel void add_int8(
          device const char* a [[buffer(0)]],
          device const char* b [[buffer(1)]],
          device char* c [[buffer(2)]],
          uint index [[thread_position_in_grid]])
      {
          c[index] = a[index] + b[index];
      }
    )";
    
    auto pipeline = CreateTestComputePipeline(device_, 
        [kernel_source UTF8String], "add_int8");
    
    auto result = RunTimedBenchmark(
        "Int8_Addition", 10, 100,
        [&]() {
          @autoreleasepool {
            id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:a_buffer offset:0 atIndex:0];
            [encoder setBuffer:b_buffer offset:0 atIndex:1];
            [encoder setBuffer:c_buffer offset:0 atIndex:2];
            
            MTLSize grid_size = MTLSizeMake(num_elements, 1, 1);
            MTLSize thread_group_size = MTLSizeMake(256, 1, 1);
            
            [encoder dispatchThreads:grid_size
                threadsPerThreadgroup:thread_group_size];
            
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
          }
        },
        bytes * 3, num_elements);
    
    ReportBenchmark(result);
  }
#endif
}

TEST_F(MetalPerformanceBenchmark, ParallelExecutionBenchmark)
{
#ifdef __APPLE__
  // Test parallel command buffer execution
  const int num_streams = 4;
  const size_t vector_size = 4 * 1024 * 1024;  // 4M elements per stream
  const size_t bytes = vector_size * sizeof(float);
  
  // Create multiple command queues
  std::vector<id<MTLCommandQueue>> queues;
  for (int i = 0; i < num_streams; ++i) {
    queues.push_back([device_ newCommandQueue]);
  }
  
  // Create buffers for each stream
  std::vector<id<MTLBuffer>> a_buffers, b_buffers, c_buffers;
  for (int i = 0; i < num_streams; ++i) {
    a_buffers.push_back(CreateTestBuffer(device_, bytes));
    b_buffers.push_back(CreateTestBuffer(device_, bytes));
    c_buffers.push_back(CreateTestBuffer(device_, bytes));
  }
  
  // Create compute pipeline
  NSString* kernel_source = @R"(
    #include <metal_stdlib>
    using namespace metal;
    
    kernel void vector_add(
        device const float* a [[buffer(0)]],
        device const float* b [[buffer(1)]],
        device float* c [[buffer(2)]],
        uint index [[thread_position_in_grid]])
    {
        c[index] = a[index] + b[index];
    }
  )";
  
  auto pipeline = CreateTestComputePipeline(device_, 
      [kernel_source UTF8String], "vector_add");
  
  // Sequential execution
  auto seq_result = RunTimedBenchmark(
      "Sequential_Execution", 10, 50,
      [&]() {
        @autoreleasepool {
          for (int i = 0; i < num_streams; ++i) {
            id<MTLCommandBuffer> command_buffer = [queues[i] commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:a_buffers[i] offset:0 atIndex:0];
            [encoder setBuffer:b_buffers[i] offset:0 atIndex:1];
            [encoder setBuffer:c_buffers[i] offset:0 atIndex:2];
            
            MTLSize grid_size = MTLSizeMake(vector_size, 1, 1);
            MTLSize thread_group_size = MTLSizeMake(256, 1, 1);
            
            [encoder dispatchThreads:grid_size
                threadsPerThreadgroup:thread_group_size];
            
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
          }
        }
      },
      bytes * 3 * num_streams,
      vector_size * num_streams);
  
  ReportBenchmark(seq_result);
  
  // Parallel execution
  auto par_result = RunTimedBenchmark(
      "Parallel_Execution", 10, 50,
      [&]() {
        @autoreleasepool {
          std::vector<id<MTLCommandBuffer>> command_buffers;
          
          // Submit all command buffers
          for (int i = 0; i < num_streams; ++i) {
            id<MTLCommandBuffer> command_buffer = [queues[i] commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:a_buffers[i] offset:0 atIndex:0];
            [encoder setBuffer:b_buffers[i] offset:0 atIndex:1];
            [encoder setBuffer:c_buffers[i] offset:0 atIndex:2];
            
            MTLSize grid_size = MTLSizeMake(vector_size, 1, 1);
            MTLSize thread_group_size = MTLSizeMake(256, 1, 1);
            
            [encoder dispatchThreads:grid_size
                threadsPerThreadgroup:thread_group_size];
            
            [encoder endEncoding];
            [command_buffer commit];
            command_buffers.push_back(command_buffer);
          }
          
          // Wait for all to complete
          for (auto buffer : command_buffers) {
            [buffer waitUntilCompleted];
          }
        }
      },
      bytes * 3 * num_streams,
      vector_size * num_streams);
  
  ReportBenchmark(par_result);
  
  // Calculate speedup
  double speedup = seq_result.avg_ms / par_result.avg_ms;
  std::cout << "\nParallel speedup: " << speedup << "x\n";
  std::cout << "Efficiency: " << (speedup / num_streams * 100) << "%\n";
#endif
}

TEST_F(MetalPerformanceBenchmark, MemoryBandwidthTest)
{
#ifdef __APPLE__
  // Measure peak memory bandwidth
  std::vector<size_t> sizes = {
      1 * 1024 * 1024,      // 1 MB
      16 * 1024 * 1024,     // 16 MB
      64 * 1024 * 1024,     // 64 MB
      256 * 1024 * 1024,    // 256 MB
      1024 * 1024 * 1024    // 1 GB
  };
  
  for (size_t size : sizes) {
    // Create buffers
    id<MTLBuffer> src_buffer = CreateTestBuffer(device_, size);
    id<MTLBuffer> dst_buffer = CreateTestBuffer(device_, size);
    
    // Kernel that reads and writes every element
    NSString* kernel_source = @R"(
      #include <metal_stdlib>
      using namespace metal;
      
      kernel void memory_bandwidth(
          device const float* src [[buffer(0)]],
          device float* dst [[buffer(1)]],
          uint index [[thread_position_in_grid]])
      {
          dst[index] = src[index];
      }
    )";
    
    auto pipeline = CreateTestComputePipeline(device_, 
        [kernel_source UTF8String], "memory_bandwidth");
    
    size_t num_elements = size / sizeof(float);
    
    auto result = RunTimedBenchmark(
        "MemoryBandwidth_" + std::to_string(size / (1024 * 1024)) + "MB",
        10, 100,
        [&]() {
          @autoreleasepool {
            id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:src_buffer offset:0 atIndex:0];
            [encoder setBuffer:dst_buffer offset:0 atIndex:1];
            
            MTLSize grid_size = MTLSizeMake(num_elements, 1, 1);
            MTLSize thread_group_size = MTLSizeMake(256, 1, 1);
            
            [encoder dispatchThreads:grid_size
                threadsPerThreadgroup:thread_group_size];
            
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
          }
        },
        size * 2);  // Read + write
    
    ReportBenchmark(result);
  }
#endif
}

TEST_F(MetalPerformanceBenchmark, ComputeIntensiveBenchmark)
{
#ifdef __APPLE__
  // Benchmark compute-intensive operations
  const size_t num_elements = 1024 * 1024;  // 1M elements
  const size_t bytes = num_elements * sizeof(float);
  
  id<MTLBuffer> input_buffer = CreateTestBuffer(device_, bytes);
  id<MTLBuffer> output_buffer = CreateTestBuffer(device_, bytes);
  
  // Kernel with high arithmetic intensity
  NSString* kernel_source = @R"(
    #include <metal_stdlib>
    using namespace metal;
    
    kernel void compute_intensive(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant int& iterations [[buffer(2)]],
        uint index [[thread_position_in_grid]])
    {
        float value = input[index];
        
        // Perform many arithmetic operations
        for (int i = 0; i < iterations; ++i) {
            value = value * 1.01f + 0.1f;
            value = sin(value) * cos(value);
            value = value * value - value;
            value = sqrt(abs(value)) + 0.001f;
        }
        
        output[index] = value;
    }
  )";
  
  auto pipeline = CreateTestComputePipeline(device_, 
      [kernel_source UTF8String], "compute_intensive");
  
  // Test with different arithmetic intensities
  std::vector<int> iteration_counts = {10, 100, 1000};
  
  for (int iterations : iteration_counts) {
    auto result = RunTimedBenchmark(
        "ComputeIntensive_" + std::to_string(iterations) + "_iterations",
        5, 20,
        [&]() {
          @autoreleasepool {
            id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:input_buffer offset:0 atIndex:0];
            [encoder setBuffer:output_buffer offset:0 atIndex:1];
            [encoder setBytes:&iterations length:sizeof(int) atIndex:2];
            
            MTLSize grid_size = MTLSizeMake(num_elements, 1, 1);
            MTLSize thread_group_size = MTLSizeMake(256, 1, 1);
            
            [encoder dispatchThreads:grid_size
                threadsPerThreadgroup:thread_group_size];
            
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
          }
        },
        bytes * 2,
        num_elements * iterations * 10);  // Approximate FLOPS
    
    ReportBenchmark(result);
  }
#endif
}

}}}  // namespace triton::server::test