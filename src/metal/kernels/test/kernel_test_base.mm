#include "kernel_test_base.h"
#include <cmath>
#include <iostream>
#include <iomanip>

namespace triton {
namespace metal {
namespace kernels {
namespace test {

void MetalKernelTest::SetUp() {
    device_ = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device_, nil) << "Failed to create Metal device";
    
    command_queue_ = [device_ newCommandQueue];
    ASSERT_NE(command_queue_, nil) << "Failed to create command queue";
    
    library_ = &MetalKernelLibrary::instance();
    library_->initialize(device_);
    
    rng_.seed(42);  // Fixed seed for reproducibility
}

void MetalKernelTest::TearDown() {
    library_->shutdown();
}

id<MTLBuffer> MetalKernelTest::create_buffer(size_t size, const float* data) {
    id<MTLBuffer> buffer;
    
    if (data) {
        buffer = [device_ newBufferWithBytes:data
                                     length:size
                                    options:MTLResourceStorageModeShared];
    } else {
        buffer = [device_ newBufferWithLength:size
                                     options:MTLResourceStorageModeShared];
    }
    
    return buffer;
}

id<MTLBuffer> MetalKernelTest::create_buffer_random(size_t size, float min_val, float max_val) {
    std::vector<float> data(size / sizeof(float));
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    for (auto& val : data) {
        val = dist(rng_);
    }
    
    return create_buffer(size, data.data());
}

std::vector<float> MetalKernelTest::read_buffer(id<MTLBuffer> buffer, size_t count) {
    const float* ptr = static_cast<const float*>([buffer contents]);
    return std::vector<float>(ptr, ptr + count);
}

void MetalKernelTest::expect_near(const std::vector<float>& actual,
                                 const std::vector<float>& expected,
                                 float tolerance) {
    ASSERT_EQ(actual.size(), expected.size()) 
        << "Size mismatch: actual=" << actual.size() 
        << ", expected=" << expected.size();
    
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance)
            << "Mismatch at index " << i;
    }
}

void MetalKernelTest::expect_equal(const std::vector<int>& actual,
                                   const std::vector<int>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_EQ(actual[i], expected[i])
            << "Mismatch at index " << i;
    }
}

double MetalKernelTest::measure_kernel_time(std::function<void()> kernel_execution,
                                           int num_iterations) {
    // Warm up
    for (int i = 0; i < 10; ++i) {
        kernel_execution();
    }
    
    // Time execution
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        kernel_execution();
    }
    
    // Wait for GPU completion
    id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0 / num_iterations;  // Return average time in ms
}

std::vector<float> MetalKernelTest::reference_gemm(const std::vector<float>& A,
                                                  const std::vector<float>& B,
                                                  size_t M, size_t N, size_t K,
                                                  float alpha, float beta,
                                                  const std::vector<float>& C) {
    std::vector<float> result(M * N);
    
    // Initialize with C if provided
    if (!C.empty()) {
        for (size_t i = 0; i < M * N; ++i) {
            result[i] = beta * C[i];
        }
    } else {
        std::fill(result.begin(), result.end(), 0.0f);
    }
    
    // Compute C = alpha * A * B + beta * C
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            result[m * N + n] += alpha * sum;
        }
    }
    
    return result;
}

std::vector<float> MetalKernelTest::reference_conv2d(const std::vector<float>& input,
                                                    const std::vector<float>& weight,
                                                    const std::vector<float>& bias,
                                                    size_t N, size_t C, size_t H, size_t W,
                                                    size_t K, size_t R, size_t S,
                                                    size_t stride_h, size_t stride_w,
                                                    size_t pad_h, size_t pad_w) {
    size_t P = (H + 2 * pad_h - R) / stride_h + 1;
    size_t Q = (W + 2 * pad_w - S) / stride_w + 1;
    
    std::vector<float> output(N * K * P * Q);
    
    for (size_t n = 0; n < N; ++n) {
        for (size_t k = 0; k < K; ++k) {
            for (size_t p = 0; p < P; ++p) {
                for (size_t q = 0; q < Q; ++q) {
                    float sum = bias.empty() ? 0.0f : bias[k];
                    
                    for (size_t c = 0; c < C; ++c) {
                        for (size_t r = 0; r < R; ++r) {
                            for (size_t s = 0; s < S; ++s) {
                                int h = p * stride_h - pad_h + r;
                                int w = q * stride_w - pad_w + s;
                                
                                if (h >= 0 && h < int(H) && w >= 0 && w < int(W)) {
                                    size_t input_idx = n * C * H * W + c * H * W + h * W + w;
                                    size_t weight_idx = k * C * R * S + c * R * S + r * S + s;
                                    sum += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                    
                    size_t output_idx = n * K * P * Q + k * P * Q + p * Q + q;
                    output[output_idx] = sum;
                }
            }
        }
    }
    
    return output;
}

std::vector<float> MetalKernelTest::reference_relu(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
    
    return output;
}

std::vector<float> MetalKernelTest::reference_sigmoid(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
    
    return output;
}

std::vector<float> MetalKernelTest::reference_softmax(const std::vector<float>& input, 
                                                     size_t axis_size) {
    std::vector<float> output(input.size());
    size_t num_batches = input.size() / axis_size;
    
    for (size_t b = 0; b < num_batches; ++b) {
        size_t offset = b * axis_size;
        
        // Find max
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < axis_size; ++i) {
            max_val = std::max(max_val, input[offset + i]);
        }
        
        // Compute exp and sum
        float sum_exp = 0.0f;
        for (size_t i = 0; i < axis_size; ++i) {
            output[offset + i] = std::exp(input[offset + i] - max_val);
            sum_exp += output[offset + i];
        }
        
        // Normalize
        for (size_t i = 0; i < axis_size; ++i) {
            output[offset + i] /= sum_exp;
        }
    }
    
    return output;
}

// Benchmark implementation
void MetalKernelBenchmark::SetUp() {
    MetalKernelTest::SetUp();
    
    // Enable profiling
    library_->enable_profiling(true);
}

void MetalKernelBenchmark::report_performance(const std::string& kernel_name,
                                             size_t flops,
                                             size_t memory_bytes,
                                             double time_ms) {
    double gflops = (flops / 1e9) / (time_ms / 1000);
    double bandwidth_gbps = (memory_bytes / 1e9) / (time_ms / 1000);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << kernel_name << ": "
              << time_ms << " ms, "
              << gflops << " GFLOPS, "
              << bandwidth_gbps << " GB/s" << std::endl;
}

void MetalKernelBenchmark::run_gemm_benchmark(const BenchmarkConfig& config) {
    for (const auto& size : config.sizes) {
        size_t M = size;
        size_t N = size;
        size_t K = size;
        
        // Create buffers
        auto A = create_buffer_random(M * K * sizeof(float));
        auto B = create_buffer_random(K * N * sizeof(float));
        auto C = create_buffer(M * N * sizeof(float));
        
        // Create tensor descriptors
        std::vector<MetalTensorDescriptor> inputs = {
            MetalTensorDescriptor({M, K}, DataType::FLOAT32),
            MetalTensorDescriptor({K, N}, DataType::FLOAT32)
        };
        std::vector<MetalTensorDescriptor> outputs = {
            MetalTensorDescriptor({M, N}, DataType::FLOAT32)
        };
        
        // Get kernel
        auto kernel = library_->get_kernel(KernelType::GEMM, inputs, outputs);
        auto kernel_config = kernel->suggest_config(inputs, outputs);
        
        // Benchmark
        double time_ms = measure_kernel_time([&]() {
            id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            
            kernel->encode(encoder, {A, B}, {C}, kernel_config);
            
            [encoder endEncoding];
            [command_buffer commit];
        }, config.iterations);
        
        size_t flops = 2 * M * N * K;  // 2 ops per MAC
        size_t memory_bytes = (M * K + K * N + M * N) * sizeof(float);
        
        report_performance("GEMM_" + std::to_string(size), flops, memory_bytes, time_ms);
    }
}

} // namespace test
} // namespace kernels
} // namespace metal
} // namespace triton