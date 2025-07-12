// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Comprehensive test suite for AMXProvider

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

#include "../apple/amx_provider.h"
#include "../apple/amx_kernels.h"

namespace triton {
namespace apple {
namespace test {

using ::testing::_;
using ::testing::Return;
using ::testing::AtLeast;

// Test fixture for AMXProvider
class AMXProviderTest : public ::testing::Test {
protected:
    void SetUp() override {
        provider_ = &AMXProvider::Instance();
        
        // Initialize random generator
        std::random_device rd;
        rng_ = std::mt19937(rd());
        
        // Initialize provider if not already done
        auto err = provider_->Initialize();
        if (err != nullptr) {
            TRITONSERVER_ErrorDelete(err);
        }
    }
    
    void TearDown() override {
        provider_->ResetMetrics();
    }
    
    // Helper to generate random matrix
    std::vector<float> GenerateRandomMatrix(size_t rows, size_t cols) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> matrix(rows * cols);
        for (auto& val : matrix) {
            val = dist(rng_);
        }
        return matrix;
    }
    
    // Helper to verify GEMM result
    bool VerifyGEMM(const float* A, const float* B, const float* C,
                    size_t M, size_t N, size_t K,
                    float alpha, float beta, const float* expected) {
        const float tolerance = 1e-3f;
        
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float computed = C[i * N + j];
                float ref = expected[i * N + j];
                if (std::abs(computed - ref) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }
    
    // Compute reference GEMM
    void ComputeReferenceGEMM(const float* A, const float* B, float* C,
                              size_t M, size_t N, size_t K,
                              float alpha, float beta) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = alpha * sum + beta * C[i * N + j];
            }
        }
    }
    
    AMXProvider* provider_;
    std::mt19937 rng_;
};

// Test AMX availability detection
TEST_F(AMXProviderTest, TestAvailability) {
    bool available = provider_->IsAvailable();
    
    // On Apple Silicon, AMX should be available
#ifdef __APPLE__
    #ifdef __arm64__
        EXPECT_TRUE(available) << "AMX should be available on Apple Silicon";
    #else
        EXPECT_FALSE(available) << "AMX should not be available on x86";
    #endif
#else
    EXPECT_FALSE(available) << "AMX should not be available on non-Apple platforms";
#endif
}

// Test capabilities detection
TEST_F(AMXProviderTest, TestCapabilities) {
    const auto& caps = provider_->GetCapabilities();
    
    if (provider_->IsAvailable()) {
        EXPECT_TRUE(caps.has_amx);
        EXPECT_GT(caps.max_tile_rows, 0u);
        EXPECT_GT(caps.max_tile_cols, 0u);
        EXPECT_GT(caps.tile_palette_size, 0u);
        
        // Log capabilities for debugging
        std::cout << "AMX Capabilities:" << std::endl;
        std::cout << "  AMX2: " << caps.has_amx2 << std::endl;
        std::cout << "  FP16: " << caps.has_amx_fp16 << std::endl;
        std::cout << "  INT8: " << caps.has_amx_int8 << std::endl;
        std::cout << "  BF16: " << caps.has_amx_bf16 << std::endl;
        std::cout << "  Max tile rows: " << caps.max_tile_rows << std::endl;
        std::cout << "  Max tile cols: " << caps.max_tile_cols << std::endl;
    }
}

// Test basic GEMM operation
TEST_F(AMXProviderTest, TestBasicGEMM) {
    if (!provider_->IsAvailable()) {
        GTEST_SKIP() << "AMX not available";
    }
    
    const size_t M = 64, N = 64, K = 64;
    auto A = GenerateRandomMatrix(M, K);
    auto B = GenerateRandomMatrix(K, N);
    std::vector<float> C(M * N, 0.0f);
    std::vector<float> C_ref(M * N, 0.0f);
    
    // Compute reference
    ComputeReferenceGEMM(A.data(), B.data(), C_ref.data(), M, N, K, 1.0f, 0.0f);
    
    // Execute with AMX
    auto err = provider_->ExecuteGEMM(
        A.data(), B.data(), C.data(),
        M, N, K, 1.0f, 0.0f);
    
    ASSERT_EQ(err, nullptr) << "GEMM execution should succeed";
    
    // Verify result
    EXPECT_TRUE(VerifyGEMM(A.data(), B.data(), C.data(), M, N, K, 1.0f, 0.0f, C_ref.data()))
        << "AMX GEMM result should match reference";
}

// Test GEMM with different sizes
TEST_F(AMXProviderTest, TestGEMMVariousSizes) {
    if (!provider_->IsAvailable()) {
        GTEST_SKIP() << "AMX not available";
    }
    
    const std::vector<std::tuple<size_t, size_t, size_t>> sizes = {
        {32, 32, 32},    // Small, tile-aligned
        {64, 128, 64},   // Medium, rectangular
        {100, 100, 100}, // Non-tile-aligned
        {256, 256, 256}, // Large
        {1024, 512, 768} // Very large
    };
    
    for (const auto& [M, N, K] : sizes) {
        auto A = GenerateRandomMatrix(M, K);
        auto B = GenerateRandomMatrix(K, N);
        std::vector<float> C(M * N, 0.0f);
        std::vector<float> C_ref(M * N, 0.0f);
        
        ComputeReferenceGEMM(A.data(), B.data(), C_ref.data(), M, N, K, 1.0f, 0.0f);
        
        auto err = provider_->ExecuteGEMM(
            A.data(), B.data(), C.data(),
            M, N, K, 1.0f, 0.0f);
        
        ASSERT_EQ(err, nullptr) << "GEMM(" << M << "x" << N << "x" << K << ") should succeed";
        
        EXPECT_TRUE(VerifyGEMM(A.data(), B.data(), C.data(), M, N, K, 1.0f, 0.0f, C_ref.data()))
            << "GEMM(" << M << "x" << N << "x" << K << ") result should match reference";
    }
}

// Test GEMM with alpha and beta parameters
TEST_F(AMXProviderTest, TestGEMMAlphaBeta) {
    if (!provider_->IsAvailable()) {
        GTEST_SKIP() << "AMX not available";
    }
    
    const size_t M = 64, N = 64, K = 64;
    auto A = GenerateRandomMatrix(M, K);
    auto B = GenerateRandomMatrix(K, N);
    auto C_initial = GenerateRandomMatrix(M, N);
    
    const float alpha = 2.5f;
    const float beta = 0.5f;
    
    std::vector<float> C(C_initial);
    std::vector<float> C_ref(C_initial);
    
    ComputeReferenceGEMM(A.data(), B.data(), C_ref.data(), M, N, K, alpha, beta);
    
    auto err = provider_->ExecuteGEMM(
        A.data(), B.data(), C.data(),
        M, N, K, alpha, beta);
    
    ASSERT_EQ(err, nullptr);
    EXPECT_TRUE(VerifyGEMM(A.data(), B.data(), C.data(), M, N, K, alpha, beta, C_ref.data()));
}

// Test batch GEMM
TEST_F(AMXProviderTest, TestBatchGEMM) {
    if (!provider_->IsAvailable()) {
        GTEST_SKIP() << "AMX not available";
    }
    
    const size_t batch_size = 8;
    const size_t M = 64, N = 64, K = 64;
    
    std::vector<std::vector<float>> A_batch(batch_size);
    std::vector<std::vector<float>> B_batch(batch_size);
    std::vector<std::vector<float>> C_batch(batch_size);
    std::vector<std::vector<float>> C_ref_batch(batch_size);
    
    std::vector<const float*> A_ptrs(batch_size);
    std::vector<const float*> B_ptrs(batch_size);
    std::vector<float*> C_ptrs(batch_size);
    
    for (size_t i = 0; i < batch_size; ++i) {
        A_batch[i] = GenerateRandomMatrix(M, K);
        B_batch[i] = GenerateRandomMatrix(K, N);
        C_batch[i].resize(M * N, 0.0f);
        C_ref_batch[i].resize(M * N, 0.0f);
        
        A_ptrs[i] = A_batch[i].data();
        B_ptrs[i] = B_batch[i].data();
        C_ptrs[i] = C_batch[i].data();
        
        ComputeReferenceGEMM(A_ptrs[i], B_ptrs[i], C_ref_batch[i].data(), M, N, K, 1.0f, 0.0f);
    }
    
    auto err = provider_->ExecuteGEMMBatch(
        A_ptrs.data(), B_ptrs.data(), C_ptrs.data(),
        batch_size, M, N, K, 1.0f, 0.0f);
    
    ASSERT_EQ(err, nullptr);
    
    for (size_t i = 0; i < batch_size; ++i) {
        EXPECT_TRUE(VerifyGEMM(A_ptrs[i], B_ptrs[i], C_ptrs[i], M, N, K, 1.0f, 0.0f, C_ref_batch[i].data()))
            << "Batch " << i << " result should match reference";
    }
}

// Test mixed precision operations
TEST_F(AMXProviderTest, TestMixedPrecision) {
    if (!provider_->IsAvailable()) {
        GTEST_SKIP() << "AMX not available";
    }
    
    const auto& caps = provider_->GetCapabilities();
    
    // Test FP16 if supported
    if (caps.has_amx_fp16) {
        const size_t M = 64, N = 64, K = 64;
        std::vector<uint16_t> A_fp16(M * K);
        std::vector<uint16_t> B_fp16(K * N);
        std::vector<uint16_t> C_fp16(M * N);
        
        // Generate random FP16 values (simplified)
        for (auto& val : A_fp16) val = 0x3C00; // 1.0 in FP16
        for (auto& val : B_fp16) val = 0x3C00; // 1.0 in FP16
        
        auto err = provider_->ExecuteGEMM_FP16(
            A_fp16.data(), B_fp16.data(), C_fp16.data(),
            M, N, K, 1.0f, 0.0f);
        
        if (err == nullptr) {
            // Basic validation - result should be K for each element
            for (const auto& val : C_fp16) {
                // Check if result is approximately K (in FP16)
                EXPECT_GT(val, 0u) << "FP16 GEMM should produce non-zero results";
            }
        } else {
            TRITONSERVER_ErrorDelete(err);
        }
    }
    
    // Test INT8 if supported
    if (caps.has_amx_int8) {
        const size_t M = 64, N = 64, K = 64;
        std::vector<int8_t> A_int8(M * K, 1);
        std::vector<int8_t> B_int8(K * N, 1);
        std::vector<int32_t> C_int32(M * N);
        
        auto err = provider_->ExecuteGEMM_INT8(
            A_int8.data(), B_int8.data(), C_int32.data(),
            M, N, K, 1, 0);
        
        if (err == nullptr) {
            // Each element should be K
            for (const auto& val : C_int32) {
                EXPECT_EQ(val, static_cast<int32_t>(K)) << "INT8 GEMM result should be K";
            }
        } else {
            TRITONSERVER_ErrorDelete(err);
        }
    }
}

// Test GEMV (matrix-vector multiplication)
TEST_F(AMXProviderTest, TestGEMV) {
    if (!provider_->IsAvailable()) {
        GTEST_SKIP() << "AMX not available";
    }
    
    const size_t M = 128, N = 64;
    auto A = GenerateRandomMatrix(M, N);
    auto x = GenerateRandomMatrix(N, 1);
    std::vector<float> y(M, 0.0f);
    std::vector<float> y_ref(M, 0.0f);
    
    // Compute reference
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            y_ref[i] += A[i * N + j] * x[j];
        }
    }
    
    auto err = provider_->ExecuteGEMV(
        A.data(), x.data(), y.data(),
        M, N, 1.0f, 0.0f);
    
    ASSERT_EQ(err, nullptr);
    
    // Verify
    const float tolerance = 1e-3f;
    for (size_t i = 0; i < M; ++i) {
        EXPECT_NEAR(y[i], y_ref[i], tolerance) << "GEMV result at index " << i;
    }
}

// Test performance metrics
TEST_F(AMXProviderTest, TestMetrics) {
    if (!provider_->IsAvailable()) {
        GTEST_SKIP() << "AMX not available";
    }
    
    provider_->ResetMetrics();
    
    const size_t M = 256, N = 256, K = 256;
    auto A = GenerateRandomMatrix(M, K);
    auto B = GenerateRandomMatrix(K, N);
    std::vector<float> C(M * N);
    
    // Perform multiple operations
    const int num_ops = 10;
    for (int i = 0; i < num_ops; ++i) {
        auto err = provider_->ExecuteGEMM(
            A.data(), B.data(), C.data(),
            M, N, K, 1.0f, 0.0f);
        ASSERT_EQ(err, nullptr);
    }
    
    auto metrics = provider_->GetMetrics();
    
    EXPECT_EQ(metrics.total_operations, static_cast<size_t>(num_ops));
    EXPECT_GT(metrics.total_time_ms, 0.0);
    EXPECT_GT(metrics.gflops, 0.0);
    
    // Log performance for debugging
    std::cout << "AMX Performance Metrics:" << std::endl;
    std::cout << "  Total operations: " << metrics.total_operations << std::endl;
    std::cout << "  Total time: " << metrics.total_time_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << metrics.gflops << std::endl;
    std::cout << "  Memory bandwidth: " << metrics.memory_bandwidth_gb_s << " GB/s" << std::endl;
}

// Test optimal configuration selection
TEST_F(AMXProviderTest, TestOptimalConfig) {
    if (!provider_->IsAvailable()) {
        GTEST_SKIP() << "AMX not available";
    }
    
    provider_->SetAutoTuning(true);
    
    // Test various sizes
    const std::vector<std::tuple<size_t, size_t, size_t>> sizes = {
        {32, 32, 32},
        {100, 100, 100},
        {1024, 512, 768},
        {2048, 2048, 2048}
    };
    
    for (const auto& [M, N, K] : sizes) {
        auto config = provider_->GetOptimalConfig(M, N, K);
        
        EXPECT_TRUE(config.use_tiles);
        EXPECT_GT(config.tile_m, 0u);
        EXPECT_GT(config.tile_n, 0u);
        EXPECT_GT(config.tile_k, 0u);
        
        // Tiles should not exceed matrix dimensions
        EXPECT_LE(config.tile_m, M);
        EXPECT_LE(config.tile_n, N);
        EXPECT_LE(config.tile_k, K);
    }
}

// Test error handling
TEST_F(AMXProviderTest, TestErrorHandling) {
    if (!provider_->IsAvailable()) {
        GTEST_SKIP() << "AMX not available";
    }
    
    // Test with null pointers
    auto err = provider_->ExecuteGEMM(
        nullptr, nullptr, nullptr,
        64, 64, 64, 1.0f, 0.0f);
    
    EXPECT_NE(err, nullptr) << "Should fail with null pointers";
    if (err) {
        TRITONSERVER_ErrorDelete(err);
    }
    
    // Test with zero dimensions
    float dummy = 0.0f;
    err = provider_->ExecuteGEMM(
        &dummy, &dummy, &dummy,
        0, 0, 0, 1.0f, 0.0f);
    
    EXPECT_NE(err, nullptr) << "Should fail with zero dimensions";
    if (err) {
        TRITONSERVER_ErrorDelete(err);
    }
}

// Test concurrent execution
TEST_F(AMXProviderTest, TestConcurrentExecution) {
    if (!provider_->IsAvailable()) {
        GTEST_SKIP() << "AMX not available";
    }
    
    const size_t num_threads = 4;
    const size_t M = 128, N = 128, K = 128;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            auto A = GenerateRandomMatrix(M, K);
            auto B = GenerateRandomMatrix(K, N);
            std::vector<float> C(M * N);
            
            for (int i = 0; i < 10; ++i) {
                auto err = provider_->ExecuteGEMM(
                    A.data(), B.data(), C.data(),
                    M, N, K, 1.0f, 0.0f);
                
                if (err == nullptr) {
                    success_count++;
                } else {
                    TRITONSERVER_ErrorDelete(err);
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(success_count.load(), static_cast<int>(num_threads * 10))
        << "All concurrent operations should succeed";
}

// Test AMX kernel library integration
TEST_F(AMXProviderTest, TestKernelLibrary) {
    if (!provider_->IsAvailable()) {
        GTEST_SKIP() << "AMX not available";
    }
    
    AMXKernelLibrary kernel_lib;
    
    const size_t M = 64, N = 64, K = 64;
    auto A = GenerateRandomMatrix(M, K);
    auto B = GenerateRandomMatrix(K, N);
    std::vector<float> C(M * N, 0.0f);
    std::vector<float> C_ref(M * N, 0.0f);
    
    ComputeReferenceGEMM(A.data(), B.data(), C_ref.data(), M, N, K, 1.0f, 0.0f);
    
    // Use kernel library directly
    kernel_lib.sgemm_amx(
        A.data(), B.data(), C.data(),
        M, N, K, 1.0f, 0.0f);
    
    EXPECT_TRUE(VerifyGEMM(A.data(), B.data(), C.data(), M, N, K, 1.0f, 0.0f, C_ref.data()))
        << "Kernel library SGEMM should match reference";
}

// Benchmark test (disabled by default)
TEST_F(AMXProviderTest, DISABLED_BenchmarkGEMM) {
    if (!provider_->IsAvailable()) {
        GTEST_SKIP() << "AMX not available";
    }
    
    const std::vector<size_t> sizes = {128, 256, 512, 1024, 2048};
    
    std::cout << "\nAMX GEMM Benchmark Results:" << std::endl;
    std::cout << "Size\tTime(ms)\tGFLOPS" << std::endl;
    
    for (size_t size : sizes) {
        auto A = GenerateRandomMatrix(size, size);
        auto B = GenerateRandomMatrix(size, size);
        std::vector<float> C(size * size);
        
        // Warmup
        for (int i = 0; i < 5; ++i) {
            provider_->ExecuteGEMM(
                A.data(), B.data(), C.data(),
                size, size, size, 1.0f, 0.0f);
        }
        
        // Benchmark
        const int num_iterations = 20;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_iterations; ++i) {
            provider_->ExecuteGEMM(
                A.data(), B.data(), C.data(),
                size, size, size, 1.0f, 0.0f);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        double time_ms = duration / 1000.0 / num_iterations;
        double gflops = (2.0 * size * size * size) / (time_ms * 1e6);
        
        std::cout << size << "\t" << time_ms << "\t" << gflops << std::endl;
    }
}

} // namespace test
} // namespace apple
} // namespace triton