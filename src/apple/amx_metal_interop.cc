// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// AMX-Metal Interop Implementation

#include "amx_metal_interop.h"
#include "amx_kernels.h"
#include "../metal/metal_backend_utils.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>

namespace triton {
namespace apple {

// ======================
// AMXMetalInterop Implementation
// ======================

AMXMetalInterop& AMXMetalInterop::Instance() {
    static AMXMetalInterop instance;
    return instance;
}

AMXMetalInterop::AMXMetalInterop() {
    amx_provider_ = &AMXProvider::Instance();
}

AMXMetalInterop::~AMXMetalInterop() {
    Shutdown();
}

TRITONSERVER_Error* AMXMetalInterop::Initialize(metal::MetalDevice* metal_device) {
    // Initialize AMX
    auto err = amx_provider_->Initialize();
    if (err != nullptr) {
        return err;
    }
    
    // Set Metal device
    if (metal_device == nullptr) {
        // Use default Metal device
        metal_device_ = metal::MetalDeviceManager::Instance().GetDefaultDevice();
    } else {
        metal_device_ = metal_device;
    }
    
    if (metal_device_ == nullptr) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            "Failed to initialize Metal device for interop");
    }
    
    // Check for unified memory support
    supports_unified_memory_ = metal_device_->HasUnifiedMemory();
    
    // Start transfer worker thread
    running_ = true;
    transfer_thread_ = std::thread(&AMXMetalInterop::TransferWorker, this);
    
    std::cout << "AMX-Metal Interop initialized" << std::endl;
    std::cout << "  Unified memory: " << (supports_unified_memory_ ? "Yes" : "No") << std::endl;
    
    return nullptr;
}

void AMXMetalInterop::Shutdown() {
    if (running_) {
        running_ = false;
        transfer_cv_.notify_all();
        
        if (transfer_thread_.joinable()) {
            transfer_thread_.join();
        }
    }
}

// ======================
// Execution Planning
// ======================

ExecutionLocation AMXMetalInterop::GetOptimalLocation(
    const OpCharacteristics& op_chars) const {
    
    // Use policy-based routing
    switch (policy_) {
        case ExecutionPolicy::MINIMIZE_LATENCY:
            return RouteBySize(op_chars);
            
        case ExecutionPolicy::MAXIMIZE_THROUGHPUT:
            return ExecutionLocation::HYBRID;
            
        case ExecutionPolicy::MINIMIZE_POWER:
            return RouteByIntensity(op_chars);
            
        case ExecutionPolicy::BALANCED:
        default:
            return RouteByPolicy(op_chars);
    }
}

ExecutionLocation AMXMetalInterop::RouteBySize(const OpCharacteristics& op) const {
    // Small operations are faster on AMX due to lower launch overhead
    const size_t small_threshold = 1024 * 1024;  // 1M elements
    
    size_t total_elements = 1;
    for (size_t dim : op.input_shapes) {
        total_elements *= dim;
    }
    
    if (total_elements < small_threshold) {
        return ExecutionLocation::AMX;
    }
    
    // Large operations benefit from GPU parallelism
    return ExecutionLocation::METAL;
}

ExecutionLocation AMXMetalInterop::RouteByIntensity(const OpCharacteristics& op) const {
    // High arithmetic intensity operations are better on GPU
    // Low arithmetic intensity (memory-bound) can be efficient on AMX
    
    if (op.arithmetic_intensity < 10.0f) {
        // Memory bound - AMX might be better due to lower memory latency
        return ExecutionLocation::AMX;
    } else if (op.arithmetic_intensity > 100.0f) {
        // Compute bound - definitely use GPU
        return ExecutionLocation::METAL;
    }
    
    // In between - consider other factors
    return RouteBySize(op);
}

ExecutionLocation AMXMetalInterop::RouteByPolicy(const OpCharacteristics& op) const {
    // Balanced routing based on multiple factors
    
    // Matrix multiply operations
    if (op.op_type == "matmul" || op.op_type == "gemm") {
        // For GEMM, use size-based routing
        size_t m = op.input_shapes[0];
        size_t n = op.input_shapes[1];
        size_t k = op.input_shapes[2];
        
        if (m * n * k < 32 * 32 * 32) {
            // Very small GEMM - use AMX
            return ExecutionLocation::AMX;
        } else if (m * n * k > 512 * 512 * 512) {
            // Large GEMM - use Metal
            return ExecutionLocation::METAL;
        }
        
        // Medium size - check if it's AMX-friendly (multiple of 32)
        if (amx::kernels::is_amx_friendly_size(m, n, k)) {
            return ExecutionLocation::AMX;
        }
    }
    
    // Convolution operations
    if (op.op_type == "conv2d") {
        // Convolutions are generally better on GPU
        return ExecutionLocation::METAL;
    }
    
    // Element-wise operations
    if (op.op_type == "add" || op.op_type == "mul" || op.op_type == "relu") {
        // Memory bandwidth limited - use device with data
        return ExecutionLocation::AUTO;
    }
    
    // Default to Metal for unknown operations
    return ExecutionLocation::METAL;
}

// ======================
// Memory Management
// ======================

AMXMetalInterop::UnifiedBuffer::UnifiedBuffer(size_t size) : size_(size) {
#ifdef __APPLE__
    // On Apple Silicon, we can create truly unified memory
    auto& interop = AMXMetalInterop::Instance();
    
    if (interop.supports_unified_memory_) {
        // Create Metal buffer with shared storage mode
        metal_buffer_ = [interop.metal_device_->GetDevice() 
            newBufferWithLength:size
            options:MTLResourceStorageModeShared];
        
        if (metal_buffer_) {
            cpu_ptr_ = [metal_buffer_ contents];
            is_unified_ = true;
        }
    }
#endif
    
    // Fallback to separate allocations
    if (!is_unified_) {
        cpu_ptr_ = amx::kernels::allocate_aligned(size, 64);
        
#ifdef __APPLE__
        metal_buffer_ = [AMXMetalInterop::Instance().metal_device_->GetDevice()
            newBufferWithLength:size
            options:MTLResourceStorageModeShared];
#endif
    }
}

AMXMetalInterop::UnifiedBuffer::~UnifiedBuffer() {
    if (!is_unified_ && cpu_ptr_) {
        amx::kernels::free_aligned(cpu_ptr_);
    }
    
#ifdef __APPLE__
    metal_buffer_ = nil;
#endif
}

void AMXMetalInterop::UnifiedBuffer::SyncToGPU() {
    if (!is_unified_) {
#ifdef __APPLE__
        // Copy CPU data to Metal buffer
        memcpy([metal_buffer_ contents], cpu_ptr_, size_);
#endif
    }
    // For unified memory, no sync needed
}

void AMXMetalInterop::UnifiedBuffer::SyncToCPU() {
    if (!is_unified_) {
#ifdef __APPLE__
        // Copy Metal buffer to CPU data
        memcpy(cpu_ptr_, [metal_buffer_ contents], size_);
#endif
    }
    // For unified memory, no sync needed
}

std::shared_ptr<AMXMetalInterop::UnifiedBuffer> 
AMXMetalInterop::CreateUnifiedBuffer(size_t size) {
    return std::make_shared<UnifiedBuffer>(size);
}

// ======================
// Data Transfer
// ======================

TRITONSERVER_Error* AMXMetalInterop::TransferToMetal(
    const void* amx_data,
    void* metal_buffer,
    size_t size,
    bool async) {
    
    if (supports_unified_memory_) {
        // No transfer needed for unified memory
        return nullptr;
    }
    
    if (async) {
        // Queue async transfer
        TransferRequest request;
        request.src = amx_data;
        request.dst = metal_buffer;
        request.size = size;
        request.to_metal = true;
        
        auto future = request.promise.get_future();
        
        {
            std::lock_guard<std::mutex> lock(transfer_mutex_);
            transfer_queue_.push(std::move(request));
        }
        transfer_cv_.notify_one();
        
        return future.get();
    } else {
        // Synchronous transfer
        auto start = std::chrono::high_resolution_clock::now();
        
#ifdef __APPLE__
        id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)metal_buffer;
        memcpy([mtl_buffer contents], amx_data, size);
#else
        memcpy(metal_buffer, amx_data, size);
#endif
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        // Update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            global_stats_.transfer_time_ms += time_ms;
            global_stats_.transfer_bytes += size;
        }
        
        return nullptr;
    }
}

void AMXMetalInterop::TransferWorker() {
    while (running_) {
        std::unique_lock<std::mutex> lock(transfer_mutex_);
        
        transfer_cv_.wait(lock, [this] {
            return !transfer_queue_.empty() || !running_;
        });
        
        if (!running_) break;
        
        while (!transfer_queue_.empty()) {
            auto request = std::move(transfer_queue_.front());
            transfer_queue_.pop();
            lock.unlock();
            
            // Perform transfer
            TRITONSERVER_Error* err = nullptr;
            
            auto start = std::chrono::high_resolution_clock::now();
            
#ifdef __APPLE__
            if (request.to_metal) {
                id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)request.dst;
                memcpy([mtl_buffer contents], request.src, request.size);
            } else {
                id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)request.src;
                memcpy(request.dst, [mtl_buffer contents], request.size);
            }
#else
            memcpy(request.dst, request.src, request.size);
#endif
            
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            
            // Update stats
            {
                std::lock_guard<std::mutex> lock2(stats_mutex_);
                global_stats_.transfer_time_ms += time_ms;
                global_stats_.transfer_bytes += request.size;
            }
            
            request.promise.set_value(err);
            
            lock.lock();
        }
    }
}

// ======================
// Hybrid Execution
// ======================

TRITONSERVER_Error* AMXMetalInterop::ExecuteGEMM(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K,
    float alpha, float beta,
    ExecutionLocation location) {
    
    // Create operation characteristics
    OpCharacteristics op;
    op.op_type = "gemm";
    op.input_shapes = {M, N, K};
    op.total_flops = 2 * M * N * K;
    op.memory_bytes = (M * K + K * N + M * N) * sizeof(float);
    op.arithmetic_intensity = static_cast<float>(op.total_flops) / op.memory_bytes;
    
    // Determine execution location
    if (location == ExecutionLocation::AUTO) {
        location = GetOptimalLocation(op);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    TRITONSERVER_Error* err = nullptr;
    
    switch (location) {
        case ExecutionLocation::AMX: {
            // Execute on AMX
            err = amx_provider_->ExecuteGEMM(A, B, C, M, N, K, alpha, beta);
            
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            
            std::lock_guard<std::mutex> lock(stats_mutex_);
            global_stats_.amx_time_ms += time_ms;
            global_stats_.amx_ops_count++;
            break;
        }
        
        case ExecutionLocation::METAL: {
            // Execute on Metal
            // This would call the Metal GEMM implementation
            // For now, return unsupported
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "Metal GEMM execution not yet implemented in interop");
            break;
        }
        
        case ExecutionLocation::HYBRID: {
            // Split computation across both devices
            // This is complex and would require careful partitioning
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "Hybrid GEMM execution not yet implemented");
            break;
        }
        
        default:
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                "Invalid execution location");
    }
    
    return err;
}

ExecutionStats AMXMetalInterop::GetGlobalStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return global_stats_;
}

void AMXMetalInterop::ResetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    global_stats_ = ExecutionStats{};
}

// ======================
// Utility Functions
// ======================

float CalculateArithmeticIntensity(
    const std::string& op_type,
    const std::vector<size_t>& shape) {
    
    size_t flops = EstimateFLOPS(op_type, shape);
    size_t memory_bytes = 0;
    
    if (op_type == "gemm" && shape.size() >= 3) {
        // GEMM: Read A (M×K), B (K×N), write C (M×N)
        size_t M = shape[0], N = shape[1], K = shape[2];
        memory_bytes = (M * K + K * N + M * N) * sizeof(float);
    } else if (op_type == "conv2d" && shape.size() >= 4) {
        // Convolution: more complex memory pattern
        size_t batch = shape[0], height = shape[1], width = shape[2], channels = shape[3];
        memory_bytes = batch * height * width * channels * sizeof(float) * 3; // Rough estimate
    } else {
        // Default: assume all elements are read/written once
        size_t total_elements = 1;
        for (size_t dim : shape) {
            total_elements *= dim;
        }
        memory_bytes = total_elements * sizeof(float) * 2;
    }
    
    return memory_bytes > 0 ? static_cast<float>(flops) / memory_bytes : 0.0f;
}

size_t EstimateFLOPS(
    const std::string& op_type,
    const std::vector<size_t>& shape) {
    
    if (op_type == "gemm" && shape.size() >= 3) {
        return 2 * shape[0] * shape[1] * shape[2];  // 2 ops per multiply-add
    } else if (op_type == "conv2d" && shape.size() >= 4) {
        // Simplified convolution FLOPS estimate
        return shape[0] * shape[1] * shape[2] * shape[3] * 9 * 2;  // 3x3 kernel
    } else if (op_type == "add" || op_type == "mul") {
        size_t total = 1;
        for (size_t dim : shape) {
            total *= dim;
        }
        return total;
    }
    
    return 0;
}

bool IsAMXPreferred(const OpCharacteristics& op) {
    // AMX is preferred for:
    // 1. Small matrices that fit in cache
    // 2. Operations with specific alignment (multiples of 32)
    // 3. Low arithmetic intensity (memory bound)
    // 4. When minimizing power consumption
    
    if (op.op_type == "gemm") {
        size_t total_size = 1;
        for (size_t dim : op.input_shapes) {
            total_size *= dim;
        }
        
        // Check if it's small enough for AMX
        if (total_size < 256 * 256 * 256) {
            // Check alignment
            bool aligned = true;
            for (size_t dim : op.input_shapes) {
                if (dim % 32 != 0) {
                    aligned = false;
                    break;
                }
            }
            if (aligned) return true;
        }
    }
    
    return op.arithmetic_intensity < 10.0f;
}

bool IsMetalPreferred(const OpCharacteristics& op) {
    // Metal is preferred for:
    // 1. Large parallel workloads
    // 2. High arithmetic intensity
    // 3. Convolutions and other image processing
    // 4. When maximizing throughput
    
    return !IsAMXPreferred(op);
}

} // namespace apple
} // namespace triton