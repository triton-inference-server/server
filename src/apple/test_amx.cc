// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Test program for AMX provider

#include "amx_provider.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

using namespace triton::apple;

// Helper to generate random matrix
void GenerateRandomMatrix(float* matrix, size_t rows, size_t cols, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    for (size_t i = 0; i < rows * cols; ++i) {
        matrix[i] = dist(gen);
    }
}

// Simple CPU GEMM for verification
void CPUGemm(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, float alpha, float beta) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = alpha * sum + beta * C[m * N + n];
        }
    }
}

// Verify results
bool VerifyResults(const float* expected, const float* actual, size_t size, float tolerance = 1e-3f) {
    for (size_t i = 0; i < size; ++i) {
        float diff = std::abs(expected[i] - actual[i]);
        if (diff > tolerance) {
            std::cout << "Mismatch at index " << i << ": expected " << expected[i] 
                     << ", got " << actual[i] << " (diff: " << diff << ")" << std::endl;
            return false;
        }
    }
    return true;
}

void TestAMXDetection() {
    std::cout << "=== AMX Detection Test ===" << std::endl;
    
    auto& provider = AMXProvider::Instance();
    const auto& caps = provider.GetCapabilities();
    
    std::cout << "AMX Support: " << (caps.has_amx ? "YES" : "NO") << std::endl;
    
    if (caps.has_amx) {
        std::cout << "Capabilities:" << std::endl;
        std::cout << "  AMX Version: " << (caps.has_amx2 ? "AMX2" : "AMX1") << std::endl;
        std::cout << "  Max Tile: " << caps.max_tile_rows << "x" << caps.max_tile_cols << std::endl;
        std::cout << "  FP16: " << (caps.has_amx_fp16 ? "Yes" : "No") << std::endl;
        std::cout << "  INT8: " << (caps.has_amx_int8 ? "Yes" : "No") << std::endl;
        std::cout << "  BF16: " << (caps.has_amx_bf16 ? "Yes" : "No") << std::endl;
        std::cout << "  Tile Palette Size: " << caps.tile_palette_size << std::endl;
    }
    
    std::cout << "\nAMX Info String:\n" << GetAMXInfoString() << std::endl;
    std::cout << "Peak Performance: " << GetAMXPeakGFLOPS() << " GFLOPS" << std::endl;
    std::cout << std::endl;
}

void TestAMXInitialization() {
    std::cout << "=== AMX Initialization Test ===" << std::endl;
    
    auto& provider = AMXProvider::Instance();
    
    auto err = provider.Initialize();
    if (err != nullptr) {
        std::cout << "Initialization failed: " << TRITONSERVER_ErrorMessage(err) << std::endl;
        TRITONSERVER_ErrorDelete(err);
    } else {
        std::cout << "AMX initialized successfully!" << std::endl;
    }
    std::cout << std::endl;
}

void TestSmallGEMM() {
    std::cout << "=== Small GEMM Test (32x32x32) ===" << std::endl;
    
    const size_t M = 32, N = 32, K = 32;
    const float alpha = 1.0f, beta = 0.0f;
    
    // Allocate matrices
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_amx(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);
    
    // Initialize with random data
    GenerateRandomMatrix(A.data(), M, K);
    GenerateRandomMatrix(B.data(), K, N);
    
    auto& provider = AMXProvider::Instance();
    
    // Run AMX GEMM
    auto start = std::chrono::high_resolution_clock::now();
    auto err = provider.ExecuteGEMM(A.data(), B.data(), C_amx.data(), M, N, K, alpha, beta);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (err != nullptr) {
        std::cout << "AMX GEMM failed: " << TRITONSERVER_ErrorMessage(err) << std::endl;
        TRITONSERVER_ErrorDelete(err);
        return;
    }
    
    double amx_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Run CPU GEMM for verification
    start = std::chrono::high_resolution_clock::now();
    CPUGemm(A.data(), B.data(), C_cpu.data(), M, N, K, alpha, beta);
    end = std::chrono::high_resolution_clock::now();
    
    double cpu_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Verify results
    bool passed = VerifyResults(C_cpu.data(), C_amx.data(), M * N);
    
    std::cout << "Verification: " << (passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << "AMX Time: " << amx_time << " ms" << std::endl;
    std::cout << "CPU Time: " << cpu_time << " ms" << std::endl;
    std::cout << "Speedup: " << cpu_time / amx_time << "x" << std::endl;
    
    // Calculate GFLOPS
    double ops = 2.0 * M * N * K;
    double amx_gflops = (ops / 1e9) / (amx_time / 1000.0);
    double cpu_gflops = (ops / 1e9) / (cpu_time / 1000.0);
    
    std::cout << "AMX GFLOPS: " << amx_gflops << std::endl;
    std::cout << "CPU GFLOPS: " << cpu_gflops << std::endl;
    std::cout << std::endl;
}

void TestLargeGEMM() {
    std::cout << "=== Large GEMM Test (1024x1024x1024) ===" << std::endl;
    
    const size_t M = 1024, N = 1024, K = 1024;
    const float alpha = 2.0f, beta = 1.0f;
    
    // Allocate matrices
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);
    
    // Initialize with random data
    GenerateRandomMatrix(A.data(), M, K);
    GenerateRandomMatrix(B.data(), K, N);
    GenerateRandomMatrix(C.data(), M, N);
    
    auto& provider = AMXProvider::Instance();
    
    // Get optimal configuration
    auto config = provider.GetOptimalConfig(M, N, K);
    std::cout << "Optimal config: " << std::endl;
    std::cout << "  Tile size: " << config.tile_m << "x" << config.tile_n << "x" << config.tile_k << std::endl;
    std::cout << "  Op type: " << static_cast<int>(config.op_type) << std::endl;
    
    // Warm up
    provider.ExecuteGEMM(A.data(), B.data(), C.data(), M, N, K, alpha, beta, config);
    
    // Benchmark
    const int num_iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        auto err = provider.ExecuteGEMM(A.data(), B.data(), C.data(), M, N, K, alpha, beta, config);
        if (err != nullptr) {
            std::cout << "AMX GEMM failed: " << TRITONSERVER_ErrorMessage(err) << std::endl;
            TRITONSERVER_ErrorDelete(err);
            return;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_time = total_time / num_iterations;
    
    // Calculate performance
    double ops = 2.0 * M * N * K;
    double gflops = (ops / 1e9) / (avg_time / 1000.0);
    
    std::cout << "Average time: " << avg_time << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "Efficiency: " << (gflops / GetAMXPeakGFLOPS()) * 100 << "%" << std::endl;
    
    // Get metrics
    auto metrics = provider.GetMetrics();
    std::cout << "\nAMX Metrics:" << std::endl;
    std::cout << "  Total operations: " << metrics.total_operations << std::endl;
    std::cout << "  Total time: " << metrics.total_time_ms << " ms" << std::endl;
    std::cout << "  Average GFLOPS: " << metrics.gflops << std::endl;
    std::cout << std::endl;
}

void TestBatchedGEMM() {
    std::cout << "=== Batched GEMM Test ===" << std::endl;
    
    const size_t batch_size = 8;
    const size_t M = 256, N = 256, K = 256;
    
    // Allocate batch of matrices
    std::vector<std::vector<float>> A_batch(batch_size);
    std::vector<std::vector<float>> B_batch(batch_size);
    std::vector<std::vector<float>> C_batch(batch_size);
    
    std::vector<const float*> A_ptrs(batch_size);
    std::vector<const float*> B_ptrs(batch_size);
    std::vector<float*> C_ptrs(batch_size);
    
    for (size_t i = 0; i < batch_size; ++i) {
        A_batch[i].resize(M * K);
        B_batch[i].resize(K * N);
        C_batch[i].resize(M * N, 0.0f);
        
        GenerateRandomMatrix(A_batch[i].data(), M, K);
        GenerateRandomMatrix(B_batch[i].data(), K, N);
        
        A_ptrs[i] = A_batch[i].data();
        B_ptrs[i] = B_batch[i].data();
        C_ptrs[i] = C_batch[i].data();
    }
    
    auto& provider = AMXProvider::Instance();
    
    auto start = std::chrono::high_resolution_clock::now();
    auto err = provider.ExecuteGEMMBatch(
        A_ptrs.data(), B_ptrs.data(), C_ptrs.data(),
        batch_size, M, N, K, 1.0f, 0.0f);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (err != nullptr) {
        std::cout << "Batched GEMM failed: " << TRITONSERVER_ErrorMessage(err) << std::endl;
        TRITONSERVER_ErrorDelete(err);
    } else {
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double ops = 2.0 * batch_size * M * N * K;
        double gflops = (ops / 1e9) / (time_ms / 1000.0);
        
        std::cout << "Batch size: " << batch_size << std::endl;
        std::cout << "Total time: " << time_ms << " ms" << std::endl;
        std::cout << "Time per GEMM: " << time_ms / batch_size << " ms" << std::endl;
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "AMX Provider Test Suite" << std::endl;
    std::cout << "=======================" << std::endl << std::endl;
    
    // Run tests
    TestAMXDetection();
    
    if (AMXProvider::Instance().IsAvailable()) {
        TestAMXInitialization();
        TestSmallGEMM();
        TestLargeGEMM();
        TestBatchedGEMM();
        
        // Reset metrics
        AMXProvider::Instance().ResetMetrics();
    } else {
        std::cout << "AMX is not available on this system. Skipping tests." << std::endl;
    }
    
    return 0;
}