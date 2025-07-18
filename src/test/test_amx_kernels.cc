// Test program for AMX kernel library
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>

#include "../apple/amx_provider.h"

using namespace triton::apple;

// Helper function to generate random data
template<typename T>
void generate_random_data(T* data, size_t size, T min_val = -1, T max_val = 1) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(min_val, max_val);
        for (size_t i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }
    } else if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> dist(min_val, max_val);
        for (size_t i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }
    }
}

// Reference GEMM implementation for validation
void reference_gemm(const float* A, const float* B, float* C,
                   size_t M, size_t N, size_t K,
                   float alpha, float beta) {
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

// Test SGEMM accuracy
bool test_sgemm_accuracy() {
    std::cout << "\n=== Testing SGEMM Accuracy ===" << std::endl;
    
    const size_t M = 128, N = 128, K = 128;
    const float alpha = 1.5f, beta = 0.5f;
    
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_ref(M * N);
    std::vector<float> C_amx(M * N);
    
    // Generate random input data
    generate_random_data(A.data(), A.size());
    generate_random_data(B.data(), B.size());
    generate_random_data(C_ref.data(), C_ref.size());
    std::copy(C_ref.begin(), C_ref.end(), C_amx.begin());
    
    // Compute reference result
    reference_gemm(A.data(), B.data(), C_ref.data(), M, N, K, alpha, beta);
    
    // Compute AMX result
    auto& provider = AMXProvider::Instance();
    auto err = provider.ExecuteGEMM(A.data(), B.data(), C_amx.data(), 
                                   M, N, K, alpha, beta);
    
    if (err != nullptr) {
        std::cerr << "AMX GEMM failed: " << TRITONSERVER_ErrorMessage(err) << std::endl;
        TRITONSERVER_ErrorDelete(err);
        return false;
    }
    
    // Check accuracy
    float max_error = 0.0f;
    for (size_t i = 0; i < M * N; ++i) {
        float error = std::abs(C_ref[i] - C_amx[i]);
        max_error = std::max(max_error, error);
    }
    
    std::cout << "Max error: " << max_error << std::endl;
    bool passed = max_error < 1e-4f;
    std::cout << "Accuracy test: " << (passed ? "PASSED" : "FAILED") << std::endl;
    
    return passed;
}

// Test SGEMM performance
void test_sgemm_performance() {
    std::cout << "\n=== Testing SGEMM Performance ===" << std::endl;
    
    auto& provider = AMXProvider::Instance();
    
    // Test different matrix sizes
    std::vector<size_t> sizes = {64, 128, 256, 512, 1024, 2048};
    
    for (size_t size : sizes) {
        size_t M = size, N = size, K = size;
        
        std::vector<float> A(M * K);
        std::vector<float> B(K * N);
        std::vector<float> C(M * N);
        
        generate_random_data(A.data(), A.size());
        generate_random_data(B.data(), B.size());
        
        // Warmup
        for (int i = 0; i < 3; ++i) {
            provider.ExecuteGEMM(A.data(), B.data(), C.data(), M, N, K);
        }
        
        // Benchmark
        const int num_runs = 10;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; ++i) {
            auto err = provider.ExecuteGEMM(A.data(), B.data(), C.data(), M, N, K);
            if (err != nullptr) {
                std::cerr << "AMX GEMM failed" << std::endl;
                TRITONSERVER_ErrorDelete(err);
                return;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / num_runs;
        
        double gflops = (2.0 * M * N * K) / (time_ms * 1e6);
        
        std::cout << "Size: " << std::setw(4) << size << "x" << std::setw(4) << size 
                  << " | Time: " << std::fixed << std::setprecision(2) << std::setw(8) << time_ms 
                  << " ms | GFLOPS: " << std::setw(8) << gflops << std::endl;
    }
}

// Test batch GEMM
void test_batch_gemm() {
    std::cout << "\n=== Testing Batch GEMM ===" << std::endl;
    
    auto& provider = AMXProvider::Instance();
    
    const size_t batch_size = 8;
    const size_t M = 256, N = 256, K = 256;
    
    // Allocate batch arrays
    std::vector<std::vector<float>> A_batch(batch_size);
    std::vector<std::vector<float>> B_batch(batch_size);
    std::vector<std::vector<float>> C_batch(batch_size);
    
    std::vector<const float*> A_ptrs(batch_size);
    std::vector<const float*> B_ptrs(batch_size);
    std::vector<float*> C_ptrs(batch_size);
    
    for (size_t b = 0; b < batch_size; ++b) {
        A_batch[b].resize(M * K);
        B_batch[b].resize(K * N);
        C_batch[b].resize(M * N);
        
        generate_random_data(A_batch[b].data(), A_batch[b].size());
        generate_random_data(B_batch[b].data(), B_batch[b].size());
        
        A_ptrs[b] = A_batch[b].data();
        B_ptrs[b] = B_batch[b].data();
        C_ptrs[b] = C_batch[b].data();
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    auto err = provider.ExecuteGEMMBatch(A_ptrs.data(), B_ptrs.data(), C_ptrs.data(),
                                        batch_size, M, N, K);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    if (err != nullptr) {
        std::cerr << "Batch GEMM failed: " << TRITONSERVER_ErrorMessage(err) << std::endl;
        TRITONSERVER_ErrorDelete(err);
        return;
    }
    
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gflops = (2.0 * batch_size * M * N * K) / (time_ms * 1e6);
    
    std::cout << "Batch size: " << batch_size << " | Matrices: " << M << "x" << N << "x" << K
              << " | Time: " << time_ms << " ms | GFLOPS: " << gflops << std::endl;
}

// Test FP16 GEMM
void test_fp16_gemm() {
    std::cout << "\n=== Testing FP16 GEMM ===" << std::endl;
    
    auto& provider = AMXProvider::Instance();
    
    if (!provider.GetCapabilities().has_amx_fp16) {
        std::cout << "FP16 not supported on this system" << std::endl;
        return;
    }
    
    const size_t M = 512, N = 512, K = 512;
    
    std::vector<uint16_t> A(M * K);
    std::vector<uint16_t> B(K * N);
    std::vector<uint16_t> C(M * N);
    
    // Generate random FP16 data (simplified)
    for (size_t i = 0; i < A.size(); ++i) {
        A[i] = static_cast<uint16_t>(rand() % 65536);
    }
    for (size_t i = 0; i < B.size(); ++i) {
        B[i] = static_cast<uint16_t>(rand() % 65536);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    auto err = provider.ExecuteGEMM_FP16(A.data(), B.data(), C.data(), M, N, K);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    if (err != nullptr) {
        std::cerr << "FP16 GEMM failed: " << TRITONSERVER_ErrorMessage(err) << std::endl;
        TRITONSERVER_ErrorDelete(err);
        return;
    }
    
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gflops = (2.0 * M * N * K) / (time_ms * 1e6);
    
    std::cout << "FP16 GEMM: " << M << "x" << N << "x" << K 
              << " | Time: " << time_ms << " ms | GFLOPS: " << gflops << std::endl;
}

// Test INT8 GEMM
void test_int8_gemm() {
    std::cout << "\n=== Testing INT8 GEMM ===" << std::endl;
    
    auto& provider = AMXProvider::Instance();
    
    if (!provider.GetCapabilities().has_amx_int8) {
        std::cout << "INT8 not supported on this system" << std::endl;
        return;
    }
    
    const size_t M = 1024, N = 1024, K = 1024;
    
    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(K * N);
    std::vector<int32_t> C(M * N);
    
    generate_random_data(A.data(), A.size(), int8_t(-127), int8_t(127));
    generate_random_data(B.data(), B.size(), int8_t(-127), int8_t(127));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    auto err = provider.ExecuteGEMM_INT8(A.data(), B.data(), C.data(), M, N, K);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    if (err != nullptr) {
        std::cerr << "INT8 GEMM failed: " << TRITONSERVER_ErrorMessage(err) << std::endl;
        TRITONSERVER_ErrorDelete(err);
        return;
    }
    
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gops = (2.0 * M * N * K) / (time_ms * 1e6);
    
    std::cout << "INT8 GEMM: " << M << "x" << N << "x" << K 
              << " | Time: " << time_ms << " ms | GOPS: " << gops << std::endl;
}

// Test activation functions
void test_activations() {
    std::cout << "\n=== Testing Activation Functions ===" << std::endl;
    
    AMXKernelLibrary kernel_lib;
    const size_t size = 1024 * 1024;
    
    std::vector<float> input(size);
    std::vector<float> output(size);
    
    generate_random_data(input.data(), input.size(), -2.0f, 2.0f);
    
    // Test ReLU
    {
        auto start = std::chrono::high_resolution_clock::now();
        kernel_lib.relu_amx(input.data(), output.data(), size);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gb_s = (size * sizeof(float) * 2) / (time_ms * 1e6);
        
        std::cout << "ReLU: " << size << " elements | Time: " << time_ms 
                  << " ms | Bandwidth: " << gb_s << " GB/s" << std::endl;
    }
    
    // Test Sigmoid
    {
        auto start = std::chrono::high_resolution_clock::now();
        kernel_lib.sigmoid_amx(input.data(), output.data(), size);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gb_s = (size * sizeof(float) * 2) / (time_ms * 1e6);
        
        std::cout << "Sigmoid: " << size << " elements | Time: " << time_ms 
                  << " ms | Bandwidth: " << gb_s << " GB/s" << std::endl;
    }
    
    // Test Tanh
    {
        auto start = std::chrono::high_resolution_clock::now();
        kernel_lib.tanh_amx(input.data(), output.data(), size);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gb_s = (size * sizeof(float) * 2) / (time_ms * 1e6);
        
        std::cout << "Tanh: " << size << " elements | Time: " << time_ms 
                  << " ms | Bandwidth: " << gb_s << " GB/s" << std::endl;
    }
}

int main() {
    std::cout << "AMX Kernel Library Test Suite" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Initialize AMX provider
    auto& provider = AMXProvider::Instance();
    auto err = provider.Initialize();
    
    if (err != nullptr) {
        std::cerr << "Failed to initialize AMX: " << TRITONSERVER_ErrorMessage(err) << std::endl;
        TRITONSERVER_ErrorDelete(err);
        return 1;
    }
    
    // Display AMX capabilities
    std::cout << GetAMXInfoString() << std::endl;
    
    // Run tests
    bool accuracy_passed = test_sgemm_accuracy();
    if (!accuracy_passed) {
        std::cerr << "Accuracy test failed, skipping performance tests" << std::endl;
        return 1;
    }
    
    test_sgemm_performance();
    test_batch_gemm();
    test_fp16_gemm();
    test_int8_gemm();
    test_activations();
    
    // Display final metrics
    auto metrics = provider.GetMetrics();
    std::cout << "\n=== Final Metrics ===" << std::endl;
    std::cout << "Total operations: " << metrics.total_operations << std::endl;
    std::cout << "Total time: " << metrics.total_time_ms << " ms" << std::endl;
    std::cout << "Average GFLOPS: " << metrics.gflops << std::endl;
    
    return 0;
}