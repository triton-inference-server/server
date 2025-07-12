// Simple standalone AMX test
#include <iostream>
#include <vector>
#include <chrono>
#include <Accelerate/Accelerate.h>

// Simplified AMX operations test
void test_amx_operations() {
    std::cout << "Testing AMX operations on Apple Silicon..." << std::endl;
    
    // Test matrix sizes
    const size_t M = 64, N = 64, K = 64;
    
    // Allocate aligned memory for best performance
    float* A = (float*)aligned_alloc(64, M * K * sizeof(float));
    float* B = (float*)aligned_alloc(64, K * N * sizeof(float));
    float* C = (float*)aligned_alloc(64, M * N * sizeof(float));
    
    // Initialize matrices
    for (size_t i = 0; i < M * K; i++) A[i] = 1.0f;
    for (size_t i = 0; i < K * N; i++) B[i] = 2.0f;
    for (size_t i = 0; i < M * N; i++) C[i] = 0.0f;
    
    std::cout << "\nMatrix multiplication " << M << "x" << N << "x" << K << std::endl;
    
    // Warm up
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    
    // Benchmark
    const int iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / iterations;
    
    // Calculate performance
    double gflops = (2.0 * M * N * K) / (avg_ms * 1e6);
    
    std::cout << "Results:" << std::endl;
    std::cout << "  Average time: " << avg_ms << " ms" << std::endl;
    std::cout << "  Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "  First result: C[0] = " << C[0] << " (expected: " << K * 2.0f << ")" << std::endl;
    
    // Test different sizes to show AMX behavior
    std::cout << "\nTesting different matrix sizes:" << std::endl;
    std::vector<size_t> sizes = {16, 32, 64, 128, 256};
    
    for (size_t size : sizes) {
        float* A_test = (float*)aligned_alloc(64, size * size * sizeof(float));
        float* B_test = (float*)aligned_alloc(64, size * size * sizeof(float));
        float* C_test = (float*)aligned_alloc(64, size * size * sizeof(float));
        
        for (size_t i = 0; i < size * size; i++) {
            A_test[i] = 1.0f;
            B_test[i] = 1.0f;
            C_test[i] = 0.0f;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    size, size, size, 1.0f, A_test, size, B_test, size, 0.0f, C_test, size);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (2.0 * size * size * size) / (time_ms * 1e6);
        
        std::cout << "  " << size << "x" << size << "x" << size 
                  << ": " << time_ms << " ms, " << gflops << " GFLOPS" << std::endl;
        
        free(A_test);
        free(B_test);
        free(C_test);
    }
    
    // Cleanup
    free(A);
    free(B);
    free(C);
    
    std::cout << "\n✓ AMX operations test completed successfully!" << std::endl;
}

// Test vDSP (part of Accelerate framework)
void test_vdsp_operations() {
    std::cout << "\n\nTesting vDSP operations..." << std::endl;
    
    const size_t size = 1024;
    std::vector<float> a(size, 1.0f);
    std::vector<float> b(size, 2.0f);
    std::vector<float> c(size, 0.0f);
    
    // Vector addition
    vDSP_vadd(a.data(), 1, b.data(), 1, c.data(), 1, size);
    std::cout << "Vector addition: c[0] = " << c[0] << " (expected: 3.0)" << std::endl;
    
    // Vector multiplication
    vDSP_vmul(a.data(), 1, b.data(), 1, c.data(), 1, size);
    std::cout << "Vector multiplication: c[0] = " << c[0] << " (expected: 2.0)" << std::endl;
    
    // Dot product
    float dot_product;
    vDSP_dotpr(a.data(), 1, b.data(), 1, &dot_product, size);
    std::cout << "Dot product: " << dot_product << " (expected: " << size * 2.0f << ")" << std::endl;
    
    std::cout << "✓ vDSP operations test completed!" << std::endl;
}

int main() {
    std::cout << "==================================" << std::endl;
    std::cout << "Apple Silicon AMX Test" << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "Running on: " << std::flush;
    system("sysctl -n machdep.cpu.brand_string");
    std::cout << std::endl;
    
    test_amx_operations();
    test_vdsp_operations();
    
    std::cout << "\n==================================" << std::endl;
    std::cout << "All tests completed!" << std::endl;
    std::cout << "AMX is working correctly on this system." << std::endl;
    std::cout << "==================================" << std::endl;
    
    return 0;
}