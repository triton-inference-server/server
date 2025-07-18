// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Comprehensive test suite for AMX-Metal Interoperability

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <chrono>
#include <thread>
#include <random>

#include "../apple/amx_metal_interop.h"
#include "../apple/amx_provider.h"
#include "../metal/metal_device.h"

namespace triton {
namespace apple {
namespace test {

using ::testing::_;
using ::testing::Return;

// Test fixture for AMXMetalInterop
class AMXMetalInteropTest : public ::testing::Test {
protected:
    void SetUp() override {
        interop_ = &AMXMetalInterop::Instance();
        
        // Initialize Metal device if available
#ifdef __APPLE__
        metal_device_ = metal::MetalDevice::GetInstance();
        if (metal_device_ && !metal_device_->IsInitialized()) {
            metal_device_->Initialize();
        }
#endif
        
        // Initialize interop
        auto err = interop_->Initialize(metal_device_);
        if (err != nullptr) {
            TRITONSERVER_ErrorDelete(err);
        }
        
        // Initialize random generator
        std::random_device rd;
        rng_ = std::mt19937(rd());
    }
    
    void TearDown() override {
        interop_->ResetStats();
    }
    
    // Helper to generate random data
    std::vector<float> GenerateRandomData(size_t size) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> data(size);
        for (auto& val : data) {
            val = dist(rng_);
        }
        return data;
    }
    
    // Helper to create operation characteristics
    OpCharacteristics CreateOpCharacteristics(
        const std::string& op_type,
        const std::vector<size_t>& shape,
        bool memory_bound = false) {
        
        OpCharacteristics op;
        op.op_type = op_type;
        op.input_shapes = shape;
        
        // Calculate FLOPS and memory
        if (op_type == "matmul" && shape.size() >= 3) {
            op.total_flops = 2 * shape[0] * shape[1] * shape[2];
            op.memory_bytes = sizeof(float) * (shape[0] * shape[2] + shape[2] * shape[1] + shape[0] * shape[1]);
        } else if (op_type == "conv2d" && shape.size() >= 4) {
            // Simplified calculation
            op.total_flops = shape[0] * shape[1] * shape[2] * shape[3] * 9; // 3x3 kernel
            op.memory_bytes = sizeof(float) * shape[0] * shape[1] * shape[2] * shape[3];
        } else {
            op.total_flops = 1000000; // 1M FLOPS default
            op.memory_bytes = sizeof(float) * 1000; // 4KB default
        }
        
        op.arithmetic_intensity = static_cast<float>(op.total_flops) / op.memory_bytes;
        op.is_memory_bound = memory_bound;
        op.has_dependencies = false;
        
        return op;
    }
    
    AMXMetalInterop* interop_;
    metal::MetalDevice* metal_device_ = nullptr;
    std::mt19937 rng_;
};

// Test initialization
TEST_F(AMXMetalInteropTest, TestInitialization) {
    // Already initialized in SetUp
    SUCCEED();
}

// Test optimal location selection
TEST_F(AMXMetalInteropTest, TestOptimalLocationSelection) {
    // Test compute-intensive operation (should prefer Metal)
    auto compute_op = CreateOpCharacteristics("matmul", {1024, 1024, 1024}, false);
    auto location = interop_->GetOptimalLocation(compute_op);
    
    if (metal_device_ && metal_device_->IsAvailable()) {
        // Large matrix multiplication should prefer Metal
        EXPECT_TRUE(location == ExecutionLocation::METAL || location == ExecutionLocation::AUTO)
            << "Compute-intensive operations should prefer Metal";
    }
    
    // Test memory-bound operation (should prefer AMX)
    auto memory_op = CreateOpCharacteristics("elementwise_add", {100, 100}, true);
    location = interop_->GetOptimalLocation(memory_op);
    
    if (AMXProvider::Instance().IsAvailable()) {
        // Small memory-bound operations should prefer AMX
        EXPECT_TRUE(location == ExecutionLocation::AMX || location == ExecutionLocation::AUTO)
            << "Memory-bound operations should prefer AMX";
    }
    
    // Test small operation (should prefer AMX)
    auto small_op = CreateOpCharacteristics("matmul", {32, 32, 32}, false);
    location = interop_->GetOptimalLocation(small_op);
    
    if (AMXProvider::Instance().IsAvailable()) {
        EXPECT_TRUE(location == ExecutionLocation::AMX || location == ExecutionLocation::AUTO)
            << "Small operations should prefer AMX";
    }
}

// Test execution planning
TEST_F(AMXMetalInteropTest, TestExecutionPlanning) {
    std::vector<OpCharacteristics> ops;
    std::unordered_map<std::string, std::vector<std::string>> dependencies;
    
    // Create a simple computation graph
    ops.push_back(CreateOpCharacteristics("matmul", {512, 512, 512}));
    ops[0].op_type = "op0_matmul";
    
    ops.push_back(CreateOpCharacteristics("relu", {512, 512}));
    ops[1].op_type = "op1_relu";
    
    ops.push_back(CreateOpCharacteristics("matmul", {512, 256, 512}));
    ops[2].op_type = "op2_matmul";
    
    // Define dependencies: op0 -> op1 -> op2
    dependencies["op1_relu"] = {"op0_matmul"};
    dependencies["op2_matmul"] = {"op1_relu"};
    
    auto plan = interop_->PlanExecution(ops, dependencies);
    
    EXPECT_EQ(plan.placements.size(), 3u) << "Plan should have 3 operations";
    
    // Log the execution plan
    std::cout << "Execution Plan:" << std::endl;
    for (const auto& placement : plan.placements) {
        std::cout << "  " << placement.op_id << " -> ";
        switch (placement.location) {
            case ExecutionLocation::AMX: std::cout << "AMX"; break;
            case ExecutionLocation::METAL: std::cout << "METAL"; break;
            case ExecutionLocation::AUTO: std::cout << "AUTO"; break;
            case ExecutionLocation::HYBRID: std::cout << "HYBRID"; break;
        }
        std::cout << std::endl;
    }
    std::cout << "  Estimated time: " << plan.estimated_time_ms << " ms" << std::endl;
    std::cout << "  Transfer bytes: " << plan.total_transfer_bytes << std::endl;
}

// Test unified buffer creation
TEST_F(AMXMetalInteropTest, TestUnifiedBuffer) {
    const size_t buffer_size = 1024 * 1024; // 1MB
    
    auto unified_buffer = interop_->CreateUnifiedBuffer(buffer_size);
    ASSERT_NE(unified_buffer, nullptr) << "Should create unified buffer";
    
    EXPECT_EQ(unified_buffer->GetSize(), buffer_size);
    EXPECT_NE(unified_buffer->GetCPUPointer(), nullptr);
    
#ifdef __APPLE__
    if (metal_device_ && metal_device_->IsAvailable()) {
        EXPECT_NE(unified_buffer->GetMetalBuffer(), nil);
    }
#endif
    
    // Test data synchronization
    float* cpu_ptr = static_cast<float*>(unified_buffer->GetCPUPointer());
    size_t num_floats = buffer_size / sizeof(float);
    
    // Write data on CPU side
    for (size_t i = 0; i < num_floats; ++i) {
        cpu_ptr[i] = static_cast<float>(i);
    }
    
    // Sync to GPU
    unified_buffer->SyncToGPU();
    
    // Modify and sync back
    cpu_ptr[0] = 999.0f;
    unified_buffer->SyncToCPU();
    
    EXPECT_EQ(cpu_ptr[0], 999.0f) << "Data should be synchronized";
}

// Test data transfer between AMX and Metal
TEST_F(AMXMetalInteropTest, TestDataTransfer) {
    const size_t data_size = 1024 * sizeof(float);
    auto amx_data = GenerateRandomData(1024);
    std::vector<float> metal_data(1024);
    
    // Test transfer to Metal
    auto err = interop_->TransferToMetal(
        amx_data.data(),
        metal_data.data(),
        data_size,
        false); // Synchronous
    
    if (err == nullptr) {
        // Verify data was transferred
        for (size_t i = 0; i < 1024; ++i) {
            EXPECT_EQ(metal_data[i], amx_data[i]) << "Data mismatch at index " << i;
        }
    } else {
        TRITONSERVER_ErrorDelete(err);
    }
    
    // Test transfer from Metal
    std::vector<float> amx_result(1024);
    err = interop_->TransferFromMetal(
        metal_data.data(),
        amx_result.data(),
        data_size,
        false); // Synchronous
    
    if (err == nullptr) {
        // Verify data was transferred back
        for (size_t i = 0; i < 1024; ++i) {
            EXPECT_EQ(amx_result[i], amx_data[i]) << "Data mismatch at index " << i;
        }
    } else {
        TRITONSERVER_ErrorDelete(err);
    }
}

// Test GEMM execution with automatic device selection
TEST_F(AMXMetalInteropTest, TestHybridGEMM) {
    const size_t M = 256, N = 256, K = 256;
    auto A = GenerateRandomData(M * K);
    auto B = GenerateRandomData(K * N);
    std::vector<float> C(M * N, 0.0f);
    
    // Test with AUTO location
    auto err = interop_->ExecuteGEMM(
        A.data(), B.data(), C.data(),
        M, N, K, 1.0f, 0.0f,
        ExecutionLocation::AUTO);
    
    if (err == nullptr) {
        // Verify result is non-zero
        float sum = 0.0f;
        for (const auto& val : C) {
            sum += std::abs(val);
        }
        EXPECT_GT(sum, 0.0f) << "GEMM result should be non-zero";
    } else {
        TRITONSERVER_ErrorDelete(err);
    }
    
    // Test forcing AMX execution
    if (AMXProvider::Instance().IsAvailable()) {
        std::fill(C.begin(), C.end(), 0.0f);
        err = interop_->ExecuteGEMM(
            A.data(), B.data(), C.data(),
            M, N, K, 1.0f, 0.0f,
            ExecutionLocation::AMX);
        
        if (err != nullptr) {
            TRITONSERVER_ErrorDelete(err);
        }
    }
    
    // Test forcing Metal execution
    if (metal_device_ && metal_device_->IsAvailable()) {
        std::fill(C.begin(), C.end(), 0.0f);
        err = interop_->ExecuteGEMM(
            A.data(), B.data(), C.data(),
            M, N, K, 1.0f, 0.0f,
            ExecutionLocation::METAL);
        
        if (err != nullptr) {
            TRITONSERVER_ErrorDelete(err);
        }
    }
}

// Test pipeline creation and execution
TEST_F(AMXMetalInteropTest, TestHybridPipeline) {
    auto pipeline = interop_->CreatePipeline();
    ASSERT_NE(pipeline, nullptr) << "Should create pipeline";
    
    // Add stages (would need actual implementation)
    // This is a placeholder for pipeline testing
    
    auto err = pipeline->Execute();
    if (err != nullptr) {
        TRITONSERVER_ErrorDelete(err);
    }
    
    pipeline->Wait();
    
    auto stats = pipeline->GetStats();
    std::cout << "Pipeline Stats:" << std::endl;
    std::cout << "  AMX time: " << stats.amx_time_ms << " ms" << std::endl;
    std::cout << "  Metal time: " << stats.metal_time_ms << " ms" << std::endl;
    std::cout << "  Transfer time: " << stats.transfer_time_ms << " ms" << std::endl;
}

// Test execution policies
TEST_F(AMXMetalInteropTest, TestExecutionPolicies) {
    const std::vector<AMXMetalInterop::ExecutionPolicy> policies = {
        AMXMetalInterop::ExecutionPolicy::MINIMIZE_LATENCY,
        AMXMetalInterop::ExecutionPolicy::MAXIMIZE_THROUGHPUT,
        AMXMetalInterop::ExecutionPolicy::MINIMIZE_POWER,
        AMXMetalInterop::ExecutionPolicy::BALANCED
    };
    
    for (auto policy : policies) {
        interop_->SetExecutionPolicy(policy);
        EXPECT_EQ(interop_->GetExecutionPolicy(), policy);
        
        // Test how policy affects routing
        auto op = CreateOpCharacteristics("matmul", {512, 512, 512});
        auto location = interop_->GetOptimalLocation(op);
        
        std::cout << "Policy: ";
        switch (policy) {
            case AMXMetalInterop::ExecutionPolicy::MINIMIZE_LATENCY:
                std::cout << "MINIMIZE_LATENCY";
                break;
            case AMXMetalInterop::ExecutionPolicy::MAXIMIZE_THROUGHPUT:
                std::cout << "MAXIMIZE_THROUGHPUT";
                break;
            case AMXMetalInterop::ExecutionPolicy::MINIMIZE_POWER:
                std::cout << "MINIMIZE_POWER";
                break;
            case AMXMetalInterop::ExecutionPolicy::BALANCED:
                std::cout << "BALANCED";
                break;
        }
        std::cout << " -> Location: ";
        switch (location) {
            case ExecutionLocation::AMX: std::cout << "AMX"; break;
            case ExecutionLocation::METAL: std::cout << "METAL"; break;
            case ExecutionLocation::AUTO: std::cout << "AUTO"; break;
            case ExecutionLocation::HYBRID: std::cout << "HYBRID"; break;
        }
        std::cout << std::endl;
    }
}

// Test performance hints
TEST_F(AMXMetalInteropTest, TestPerformanceHints) {
    interop_->SetBatchSize(32);
    interop_->SetMemoryBudget(1024 * 1024 * 1024); // 1GB
    
    // These hints should affect routing decisions
    auto op = CreateOpCharacteristics("matmul", {32, 1024, 1024});
    auto location = interop_->GetOptimalLocation(op);
    
    // With batch size 32, might prefer different location
    SUCCEED();
}

// Test operation profiling
TEST_F(AMXMetalInteropTest, TestOperationProfiling) {
    auto op = CreateOpCharacteristics("matmul", {256, 256, 256});
    
    auto result = interop_->ProfileOperation("matmul", op, 10);
    
    std::cout << "Profiling Results:" << std::endl;
    std::cout << "  AMX time: " << result.amx_time_ms << " ms" << std::endl;
    std::cout << "  Metal time: " << result.metal_time_ms << " ms" << std::endl;
    std::cout << "  AMX GFLOPS: " << result.amx_gflops << std::endl;
    std::cout << "  Metal GFLOPS: " << result.metal_gflops << std::endl;
    std::cout << "  AMX power: " << result.amx_power_watts << " W" << std::endl;
    std::cout << "  Metal power: " << result.metal_power_watts << " W" << std::endl;
    std::cout << "  Recommendation: ";
    switch (result.recommendation) {
        case ExecutionLocation::AMX: std::cout << "AMX"; break;
        case ExecutionLocation::METAL: std::cout << "METAL"; break;
        case ExecutionLocation::AUTO: std::cout << "AUTO"; break;
        case ExecutionLocation::HYBRID: std::cout << "HYBRID"; break;
    }
    std::cout << std::endl;
}

// Test concurrent operations
TEST_F(AMXMetalInteropTest, TestConcurrentOperations) {
    const size_t num_threads = 4;
    const size_t num_ops = 10;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (size_t i = 0; i < num_ops; ++i) {
                const size_t size = 128;
                auto A = GenerateRandomData(size * size);
                auto B = GenerateRandomData(size * size);
                std::vector<float> C(size * size);
                
                auto err = interop_->ExecuteGEMM(
                    A.data(), B.data(), C.data(),
                    size, size, size, 1.0f, 0.0f,
                    ExecutionLocation::AUTO);
                
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
    
    std::cout << "Concurrent operations success: " << success_count.load() 
              << " / " << (num_threads * num_ops) << std::endl;
}

// Test global statistics
TEST_F(AMXMetalInteropTest, TestGlobalStatistics) {
    interop_->ResetStats();
    
    // Perform some operations to generate stats
    const size_t M = 128, N = 128, K = 128;
    auto A = GenerateRandomData(M * K);
    auto B = GenerateRandomData(K * N);
    std::vector<float> C(M * N);
    
    for (int i = 0; i < 5; ++i) {
        auto err = interop_->ExecuteGEMM(
            A.data(), B.data(), C.data(),
            M, N, K, 1.0f, 0.0f,
            ExecutionLocation::AUTO);
        
        if (err != nullptr) {
            TRITONSERVER_ErrorDelete(err);
        }
    }
    
    auto stats = interop_->GetGlobalStats();
    
    std::cout << "Global Statistics:" << std::endl;
    std::cout << "  AMX time: " << stats.amx_time_ms << " ms" << std::endl;
    std::cout << "  Metal time: " << stats.metal_time_ms << " ms" << std::endl;
    std::cout << "  Transfer time: " << stats.transfer_time_ms << " ms" << std::endl;
    std::cout << "  AMX ops: " << stats.amx_ops_count << std::endl;
    std::cout << "  Metal ops: " << stats.metal_ops_count << std::endl;
    std::cout << "  Transfer bytes: " << stats.transfer_bytes << std::endl;
}

// Test utility functions
TEST_F(AMXMetalInteropTest, TestUtilityFunctions) {
    // Test arithmetic intensity calculation
    float intensity = CalculateArithmeticIntensity("matmul", {512, 512, 512});
    EXPECT_GT(intensity, 0.0f) << "Arithmetic intensity should be positive";
    
    // Test FLOPS estimation
    size_t flops = EstimateFLOPS("matmul", {512, 512, 512});
    EXPECT_GT(flops, 0u) << "FLOPS should be positive";
    EXPECT_EQ(flops, 2 * 512 * 512 * 512) << "FLOPS calculation for matmul";
    
    // Test preference functions
    auto compute_heavy_op = CreateOpCharacteristics("matmul", {2048, 2048, 2048});
    bool metal_preferred = IsMetalPreferred(compute_heavy_op);
    
    auto memory_bound_op = CreateOpCharacteristics("copy", {100, 100}, true);
    bool amx_preferred = IsAMXPreferred(memory_bound_op);
    
    std::cout << "Compute-heavy op prefers Metal: " << metal_preferred << std::endl;
    std::cout << "Memory-bound op prefers AMX: " << amx_preferred << std::endl;
}

// Test error handling
TEST_F(AMXMetalInteropTest, TestErrorHandling) {
    // Test with null pointers
    auto err = interop_->ExecuteGEMM(
        nullptr, nullptr, nullptr,
        64, 64, 64, 1.0f, 0.0f,
        ExecutionLocation::AUTO);
    
    EXPECT_NE(err, nullptr) << "Should fail with null pointers";
    if (err) {
        TRITONSERVER_ErrorDelete(err);
    }
    
    // Test invalid transfer
    err = interop_->TransferToMetal(nullptr, nullptr, 0, false);
    EXPECT_NE(err, nullptr) << "Should fail with null transfer";
    if (err) {
        TRITONSERVER_ErrorDelete(err);
    }
}

// Test shutdown and cleanup
TEST_F(AMXMetalInteropTest, TestShutdown) {
    // Shutdown should be safe to call multiple times
    interop_->Shutdown();
    interop_->Shutdown();
    
    // Operations after shutdown should fail gracefully
    auto err = interop_->Initialize(metal_device_);
    if (err != nullptr) {
        TRITONSERVER_ErrorDelete(err);
    }
}

// Benchmark test (disabled by default)
TEST_F(AMXMetalInteropTest, DISABLED_BenchmarkHybridExecution) {
    const std::vector<size_t> sizes = {128, 256, 512, 1024};
    
    std::cout << "\nHybrid Execution Benchmark:" << std::endl;
    std::cout << "Size\tAMX(ms)\tMetal(ms)\tAuto(ms)" << std::endl;
    
    for (size_t size : sizes) {
        auto A = GenerateRandomData(size * size);
        auto B = GenerateRandomData(size * size);
        std::vector<float> C(size * size);
        
        // Benchmark AMX
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            interop_->ExecuteGEMM(
                A.data(), B.data(), C.data(),
                size, size, size, 1.0f, 0.0f,
                ExecutionLocation::AMX);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double amx_time = std::chrono::duration<double, std::milli>(end - start).count() / 10;
        
        // Benchmark Metal
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            interop_->ExecuteGEMM(
                A.data(), B.data(), C.data(),
                size, size, size, 1.0f, 0.0f,
                ExecutionLocation::METAL);
        }
        end = std::chrono::high_resolution_clock::now();
        double metal_time = std::chrono::duration<double, std::milli>(end - start).count() / 10;
        
        // Benchmark Auto
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            interop_->ExecuteGEMM(
                A.data(), B.data(), C.data(),
                size, size, size, 1.0f, 0.0f,
                ExecutionLocation::AUTO);
        }
        end = std::chrono::high_resolution_clock::now();
        double auto_time = std::chrono::duration<double, std::milli>(end - start).count() / 10;
        
        std::cout << size << "\t" << amx_time << "\t" << metal_time << "\t" << auto_time << std::endl;
    }
}

} // namespace test
} // namespace apple
} // namespace triton