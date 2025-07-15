// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// AMX-Metal Interoperability Layer

#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>
#include <future>

#ifdef __APPLE__
#ifdef __OBJC__
#include <Metal/Metal.h>
#endif
#endif

#include "amx_provider.h"
#include "../metal/metal_device.h"
#include "../metal/metal_memory.h"

namespace triton {
namespace apple {

// Execution location for operations
enum class ExecutionLocation {
    AMX,        // CPU with AMX
    METAL,      // GPU with Metal
    AUTO,       // Automatically decide
    HYBRID      // Split across both
};

// Operation characteristics for routing decisions
struct OpCharacteristics {
    std::string op_type;
    std::vector<size_t> input_shapes;
    size_t total_flops;
    size_t memory_bytes;
    float arithmetic_intensity;  // FLOPS per byte
    bool is_memory_bound;
    bool has_dependencies;
};

// Execution statistics
struct ExecutionStats {
    double amx_time_ms = 0.0;
    double metal_time_ms = 0.0;
    double transfer_time_ms = 0.0;
    size_t amx_ops_count = 0;
    size_t metal_ops_count = 0;
    size_t transfer_bytes = 0;
};

// AMX-Metal Interop Manager
class AMXMetalInterop {
public:
    static AMXMetalInterop& Instance();
    
    // Initialize interop system
    TRITONSERVER_Error* Initialize(
        triton::core::metal::MetalDevice* metal_device = nullptr);
    
    // Shutdown interop system
    void Shutdown();
    
    // ======================
    // Execution Planning
    // ======================
    
    // Determine optimal execution location for an operation
    ExecutionLocation GetOptimalLocation(
        const OpCharacteristics& op_chars) const;
    
    // Create execution plan for a graph of operations
    struct ExecutionPlan {
        struct OpPlacement {
            std::string op_id;
            ExecutionLocation location;
            std::vector<std::string> input_transfers;
            std::vector<std::string> output_transfers;
        };
        
        std::vector<OpPlacement> placements;
        double estimated_time_ms;
        size_t total_transfer_bytes;
    };
    
    ExecutionPlan PlanExecution(
        const std::vector<OpCharacteristics>& ops,
        const std::unordered_map<std::string, std::vector<std::string>>& dependencies);
    
    // ======================
    // Memory Management
    // ======================
    
    // Unified memory buffer that can be accessed from both AMX and Metal
    class UnifiedBuffer {
    public:
        UnifiedBuffer(size_t size);
        ~UnifiedBuffer();
        
        void* GetCPUPointer() { return cpu_ptr_; }
        const void* GetCPUPointer() const { return cpu_ptr_; }
        
#ifdef __APPLE__
#ifdef __OBJC__
        id<MTLBuffer> GetMetalBuffer() { return metal_buffer_; }
#else
        void* GetMetalBuffer() { return metal_buffer_; }
#endif
#endif
        
        size_t GetSize() const { return size_; }
        
        // Synchronization
        void SyncToGPU();
        void SyncToCPU();
        
    private:
        void* cpu_ptr_ = nullptr;
#ifdef __APPLE__
#ifdef __OBJC__
        id<MTLBuffer> metal_buffer_ = nil;
#else
        void* metal_buffer_ = nullptr;
#endif
#endif
        size_t size_ = 0;
        bool is_unified_ = false;
    };
    
    // Create unified buffer
    std::shared_ptr<UnifiedBuffer> CreateUnifiedBuffer(size_t size);
    
    // Transfer data between AMX and Metal
    TRITONSERVER_Error* TransferToMetal(
        const void* amx_data,
        void* metal_buffer,
        size_t size,
        bool async = true);
    
    TRITONSERVER_Error* TransferFromMetal(
        const void* metal_buffer,
        void* amx_data,
        size_t size,
        bool async = true);
    
    // ======================
    // Hybrid Execution
    // ======================
    
    // Execute operation on optimal device
    TRITONSERVER_Error* ExecuteOp(
        const std::string& op_type,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const OpCharacteristics& characteristics,
        ExecutionLocation preferred_location = ExecutionLocation::AUTO);
    
    // Execute GEMM with automatic device selection
    TRITONSERVER_Error* ExecuteGEMM(
        const float* A, const float* B, float* C,
        size_t M, size_t N, size_t K,
        float alpha = 1.0f, float beta = 0.0f,
        ExecutionLocation location = ExecutionLocation::AUTO);
    
    // Execute convolution with automatic device selection
    TRITONSERVER_Error* ExecuteConv2D(
        const float* input,
        const float* kernel,
        float* output,
        const void* params,  // ConvParams struct
        ExecutionLocation location = ExecutionLocation::AUTO);
    
    // ======================
    // Pipeline Management
    // ======================
    
    // Pipeline stage for overlapping computation and transfer
    class PipelineStage {
    public:
        virtual ~PipelineStage() = default;
        virtual TRITONSERVER_Error* Execute() = 0;
        virtual bool IsComplete() const = 0;
        virtual ExecutionLocation GetLocation() const = 0;
    };
    
    // Create pipeline for hybrid execution
    class HybridPipeline {
    public:
        void AddStage(std::unique_ptr<PipelineStage> stage);
        TRITONSERVER_Error* Execute();
        void Wait();
        ExecutionStats GetStats() const;
        
    private:
        std::vector<std::unique_ptr<PipelineStage>> stages_;
        std::vector<std::thread> workers_;
        ExecutionStats stats_;
    };
    
    std::unique_ptr<HybridPipeline> CreatePipeline();
    
    // ======================
    // Performance Optimization
    // ======================
    
    // Set execution policy
    enum class ExecutionPolicy {
        MINIMIZE_LATENCY,      // Prefer faster device
        MAXIMIZE_THROUGHPUT,   // Use both devices
        MINIMIZE_POWER,        // Prefer efficient device
        BALANCED              // Balance all factors
    };
    
    void SetExecutionPolicy(ExecutionPolicy policy) { policy_ = policy; }
    ExecutionPolicy GetExecutionPolicy() const { return policy_; }
    
    // Performance hints
    void SetBatchSize(size_t batch_size) { batch_size_ = batch_size; }
    void SetMemoryBudget(size_t bytes) { memory_budget_ = bytes; }
    
    // Get execution statistics
    ExecutionStats GetGlobalStats() const;
    void ResetStats();
    
    // ======================
    // Profiling Support
    // ======================
    
    // Profile operation on both devices
    struct ProfilingResult {
        double amx_time_ms;
        double metal_time_ms;
        double amx_gflops;
        double metal_gflops;
        double amx_power_watts;
        double metal_power_watts;
        ExecutionLocation recommendation;
    };
    
    ProfilingResult ProfileOperation(
        const std::string& op_type,
        const OpCharacteristics& characteristics,
        int num_iterations = 100);
    
private:
    AMXMetalInterop();
    ~AMXMetalInterop();
    
    // Routing heuristics
    ExecutionLocation RouteBySize(const OpCharacteristics& op) const;
    ExecutionLocation RouteByIntensity(const OpCharacteristics& op) const;
    ExecutionLocation RouteByPolicy(const OpCharacteristics& op) const;
    
    // Cost model for execution planning
    double EstimateExecutionTime(
        const OpCharacteristics& op,
        ExecutionLocation location) const;
    
    double EstimateTransferTime(size_t bytes) const;
    
    // Member variables
    AMXProvider* amx_provider_ = nullptr;
    triton::core::metal::MetalDevice* metal_device_ = nullptr;
    
    ExecutionPolicy policy_ = ExecutionPolicy::BALANCED;
    size_t batch_size_ = 1;
    size_t memory_budget_ = 0;
    
    mutable std::mutex stats_mutex_;
    ExecutionStats global_stats_;
    
    // Unified memory support
    bool supports_unified_memory_ = false;
    size_t unified_memory_alignment_ = 16384;  // 16KB pages
    
    // Transfer queue for async operations
    struct TransferRequest {
        const void* src;
        void* dst;
        size_t size;
        bool to_metal;
        std::promise<TRITONSERVER_Error*> promise;
    };
    
    std::queue<TransferRequest> transfer_queue_;
    std::mutex transfer_mutex_;
    std::condition_variable transfer_cv_;
    std::thread transfer_thread_;
    std::atomic<bool> running_{false};
    
    void TransferWorker();
    
    // Prevent copying
    AMXMetalInterop(const AMXMetalInterop&) = delete;
    AMXMetalInterop& operator=(const AMXMetalInterop&) = delete;
};

// ======================
// Utility Functions
// ======================

// Calculate arithmetic intensity for common operations
float CalculateArithmeticIntensity(
    const std::string& op_type,
    const std::vector<size_t>& shape);

// Estimate FLOPS for operation
size_t EstimateFLOPS(
    const std::string& op_type,
    const std::vector<size_t>& shape);

// Check if operation is better suited for AMX
bool IsAMXPreferred(const OpCharacteristics& op);

// Check if operation is better suited for Metal
bool IsMetalPreferred(const OpCharacteristics& op);

} // namespace apple
} // namespace triton