#pragma once

#include <Metal/Metal.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <chrono>
#include "../metal_kernel_library.h"

namespace triton {
namespace metal {
namespace kernels {
namespace test {

// Base class for kernel tests
class MetalKernelTest : public ::testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;
    
    // Helper methods for testing
    id<MTLBuffer> create_buffer(size_t size, const float* data = nullptr);
    id<MTLBuffer> create_buffer_random(size_t size, float min_val = -1.0f, float max_val = 1.0f);
    std::vector<float> read_buffer(id<MTLBuffer> buffer, size_t count);
    
    // Comparison helpers
    void expect_near(const std::vector<float>& actual, 
                    const std::vector<float>& expected,
                    float tolerance = 1e-5f);
    
    void expect_equal(const std::vector<int>& actual,
                     const std::vector<int>& expected);
    
    // Performance measurement
    double measure_kernel_time(std::function<void()> kernel_execution,
                              int num_iterations = 100);
    
    // Reference implementations for validation
    std::vector<float> reference_gemm(const std::vector<float>& A,
                                     const std::vector<float>& B,
                                     size_t M, size_t N, size_t K,
                                     float alpha = 1.0f, float beta = 0.0f,
                                     const std::vector<float>& C = {});
    
    std::vector<float> reference_conv2d(const std::vector<float>& input,
                                       const std::vector<float>& weight,
                                       const std::vector<float>& bias,
                                       size_t N, size_t C, size_t H, size_t W,
                                       size_t K, size_t R, size_t S,
                                       size_t stride_h, size_t stride_w,
                                       size_t pad_h, size_t pad_w);
    
    std::vector<float> reference_relu(const std::vector<float>& input);
    std::vector<float> reference_sigmoid(const std::vector<float>& input);
    std::vector<float> reference_softmax(const std::vector<float>& input, size_t axis_size);
    
protected:
    id<MTLDevice> device_;
    id<MTLCommandQueue> command_queue_;
    MetalKernelLibrary* library_;
    
    std::mt19937 rng_;
};

// Macro for easy kernel testing
#define TEST_KERNEL(kernel_type, test_name) \
    TEST_F(MetalKernelTest, kernel_type##_##test_name)

// Performance benchmark base
class MetalKernelBenchmark : public MetalKernelTest {
protected:
    void SetUp() override;
    
    void report_performance(const std::string& kernel_name,
                           size_t flops,
                           size_t memory_bytes,
                           double time_ms);
    
    // Benchmark configurations
    struct BenchmarkConfig {
        std::string name;
        std::vector<size_t> sizes;
        DataType dtype;
        int iterations = 100;
    };
    
    void run_gemm_benchmark(const BenchmarkConfig& config);
    void run_conv2d_benchmark(const BenchmarkConfig& config);
    void run_activation_benchmark(const BenchmarkConfig& config);
};

} // namespace test
} // namespace kernels
} // namespace metal
} // namespace triton