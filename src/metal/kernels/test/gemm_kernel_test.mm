#include "kernel_test_base.h"
#include "../gemm_kernel.h"

namespace triton {
namespace metal {
namespace kernels {
namespace test {

TEST_KERNEL(GEMM, BasicMultiplication) {
    // Test basic 4x4 matrix multiplication
    const size_t M = 4, N = 4, K = 4;
    
    // Create test data
    std::vector<float> A = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    
    std::vector<float> B = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    
    // Create Metal buffers
    auto a_buffer = create_buffer(A.size() * sizeof(float), A.data());
    auto b_buffer = create_buffer(B.size() * sizeof(float), B.data());
    auto c_buffer = create_buffer(M * N * sizeof(float));
    
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
    auto config = kernel->suggest_config(inputs, outputs);
    
    // Execute kernel
    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        kernel->encode(encoder, {a_buffer, b_buffer}, {c_buffer}, config);
        
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    }
    
    // Read results
    auto result = read_buffer(c_buffer, M * N);
    
    // Compute reference
    auto expected = reference_gemm(A, B, M, N, K);
    
    // Compare
    expect_near(result, expected);
}

TEST_KERNEL(GEMM, LargeMatrix) {
    // Test larger matrix multiplication
    const size_t M = 128, N = 256, K = 64;
    
    // Create random test data
    auto a_buffer = create_buffer_random(M * K * sizeof(float));
    auto b_buffer = create_buffer_random(K * N * sizeof(float));
    auto c_buffer = create_buffer(M * N * sizeof(float));
    
    // Read input data for reference computation
    auto A = read_buffer(a_buffer, M * K);
    auto B = read_buffer(b_buffer, K * N);
    
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
    auto config = kernel->suggest_config(inputs, outputs);
    
    // Execute kernel
    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        kernel->encode(encoder, {a_buffer, b_buffer}, {c_buffer}, config);
        
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    }
    
    // Read results
    auto result = read_buffer(c_buffer, M * N);
    
    // Compute reference
    auto expected = reference_gemm(A, B, M, N, K);
    
    // Compare with slightly higher tolerance for larger matrices
    expect_near(result, expected, 1e-4f);
}

TEST_KERNEL(GEMM, AlphaBeta) {
    // Test GEMM with alpha and beta parameters
    const size_t M = 32, N = 32, K = 32;
    const float alpha = 2.0f;
    const float beta = 0.5f;
    
    // Create test data
    auto a_buffer = create_buffer_random(M * K * sizeof(float));
    auto b_buffer = create_buffer_random(K * N * sizeof(float));
    auto c_buffer = create_buffer_random(M * N * sizeof(float));
    
    // Read input data
    auto A = read_buffer(a_buffer, M * K);
    auto B = read_buffer(b_buffer, K * N);
    auto C = read_buffer(c_buffer, M * N);
    
    // Create tensor descriptors
    std::vector<MetalTensorDescriptor> inputs = {
        MetalTensorDescriptor({M, K}, DataType::FLOAT32),
        MetalTensorDescriptor({K, N}, DataType::FLOAT32)
    };
    std::vector<MetalTensorDescriptor> outputs = {
        MetalTensorDescriptor({M, N}, DataType::FLOAT32)
    };
    
    // Get kernel and configure
    auto kernel = std::dynamic_pointer_cast<GEMMKernel>(
        library_->get_kernel(KernelType::GEMM, inputs, outputs));
    kernel->set_alpha(alpha);
    kernel->set_beta(beta);
    
    auto config = kernel->suggest_config(inputs, outputs);
    
    // Execute kernel
    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        kernel->encode(encoder, {a_buffer, b_buffer}, {c_buffer}, config);
        
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    }
    
    // Read results
    auto result = read_buffer(c_buffer, M * N);
    
    // Compute reference
    auto expected = reference_gemm(A, B, M, N, K, alpha, beta, C);
    
    // Compare
    expect_near(result, expected, 1e-4f);
}

// Benchmark test
class GEMMBenchmark : public MetalKernelBenchmark {};

TEST_F(GEMMBenchmark, PerformanceTest) {
    BenchmarkConfig config;
    config.name = "GEMM";
    config.sizes = {256, 512, 1024, 2048};
    config.dtype = DataType::FLOAT32;
    config.iterations = 100;
    
    run_gemm_benchmark(config);
}

} // namespace test
} // namespace kernels
} // namespace metal
} // namespace triton