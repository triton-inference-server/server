#include "gemm_kernel.h"
#include <Foundation/Foundation.h>

namespace triton {
namespace metal {
namespace kernels {

GEMMKernel::GEMMKernel() 
    : MetalKernel(KernelType::GEMM, "gemm") {
}

void GEMMKernel::encode(id<MTLComputeCommandEncoder> encoder,
                       const std::vector<id<MTLBuffer>>& inputs,
                       const std::vector<id<MTLBuffer>>& outputs,
                       const KernelConfig& config) {
    @autoreleasepool {
        // Validate inputs
        if (inputs.size() < 2 || outputs.empty()) {
            throw std::runtime_error("GEMM requires at least 2 inputs and 1 output");
        }
        
        // Get dimensions
        id<MTLBuffer> A = inputs[0];
        id<MTLBuffer> B = inputs[1];
        id<MTLBuffer> C = outputs[0];
        
        // Get dimensions from config or tensor descriptors
        uint M, N, K;
        
        // Check if dimensions are provided in config
        auto m_it = config.int_params.find("M");
        auto n_it = config.int_params.find("N");
        auto k_it = config.int_params.find("K");
        
        if (m_it != config.int_params.end() && 
            n_it != config.int_params.end() && 
            k_it != config.int_params.end()) {
            // Use dimensions from config
            M = m_it->second;
            N = n_it->second;
            K = k_it->second;
        } else {
            // Extract dimensions based on operation: C = alpha * A * B + beta * C
            // A is M x K, B is K x N, C is M x N
            // For simplicity, assume dimensions are passed as buffer metadata
            // In a real implementation, this would come from tensor descriptors
            M = 1024; // Default dimensions for testing
            N = 1024;
            K = 1024;
        }
        
        // Set compute pipeline
        [encoder setComputePipelineState:pipeline_state_];
        
        // Set buffers
        [encoder setBuffer:A offset:0 atIndex:0];
        [encoder setBuffer:B offset:0 atIndex:1];
        [encoder setBuffer:C offset:0 atIndex:2];
        
        // Set constants
        [encoder setBytes:&M length:sizeof(uint) atIndex:3];
        [encoder setBytes:&N length:sizeof(uint) atIndex:4];
        [encoder setBytes:&K length:sizeof(uint) atIndex:5];
        [encoder setBytes:&alpha_ length:sizeof(float) atIndex:6];
        [encoder setBytes:&beta_ length:sizeof(float) atIndex:7];
        
        // Set threadgroup memory if using tiled kernel
        if (config.shared_memory_size > 0) {
            [encoder setThreadgroupMemoryLength:config.shared_memory_size atIndex:0];
            [encoder setThreadgroupMemoryLength:config.shared_memory_size atIndex:1];
        }
        
        // Dispatch threads
        MTLSize gridSize = MTLSizeMake(N, M, 1);
        [encoder dispatchThreads:gridSize 
            threadsPerThreadgroup:config.threadgroup_size];
    }
}

KernelConfig GEMMKernel::suggest_config(const std::vector<MetalTensorDescriptor>& inputs,
                                       const std::vector<MetalTensorDescriptor>& outputs) const {
    KernelConfig config;
    
    // Get matrix dimensions
    const auto& a_shape = inputs[0].shape();
    const auto& b_shape = inputs[1].shape();
    
    size_t M = a_shape[a_shape.size() - 2];
    size_t N = b_shape[b_shape.size() - 1];
    size_t K = a_shape[a_shape.size() - 1];
    
    // Store dimensions in config
    config.int_params["M"] = M;
    config.int_params["N"] = N;
    config.int_params["K"] = K;
    
    // Determine optimal threadgroup size
    if (M * N < 1024) {
        // Small matrices - use simple kernel
        config.threadgroup_size = MTLSizeMake(16, 16, 1);
        config.shared_memory_size = 0;
    } else {
        // Large matrices - use tiled kernel
        const size_t TILE_SIZE = 16;
        config.threadgroup_size = MTLSizeMake(TILE_SIZE, TILE_SIZE, 1);
        config.shared_memory_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    }
    
    // Check if we should use half precision
    if (inputs[0].dtype() == DataType::FLOAT16) {
        config.use_half_precision = true;
    }
    
    // Use SIMD operations on Apple Silicon
    config.use_simd_group = true;
    
    return config;
}

bool GEMMKernel::validate(const std::vector<MetalTensorDescriptor>& inputs,
                         const std::vector<MetalTensorDescriptor>& outputs) const {
    // Check number of inputs/outputs
    if (inputs.size() < 2 || outputs.empty()) {
        return false;
    }
    
    // Get shapes
    const auto& a_shape = inputs[0].shape();
    const auto& b_shape = inputs[1].shape();
    const auto& c_shape = outputs[0].shape();
    
    // Check dimensions
    if (a_shape.size() < 2 || b_shape.size() < 2 || c_shape.size() < 2) {
        return false;
    }
    
    // Check matrix multiplication compatibility
    size_t M = a_shape[a_shape.size() - 2];
    size_t K1 = a_shape[a_shape.size() - 1];
    size_t K2 = b_shape[b_shape.size() - 2];
    size_t N = b_shape[b_shape.size() - 1];
    
    if (K1 != K2) {
        return false;
    }
    
    // Check output shape
    if (c_shape[c_shape.size() - 2] != M || c_shape[c_shape.size() - 1] != N) {
        return false;
    }
    
    // Check data types match
    if (inputs[0].dtype() != inputs[1].dtype() || 
        inputs[0].dtype() != outputs[0].dtype()) {
        return false;
    }
    
    return true;
}

void GEMMKernel::select_optimal_kernel(const MetalTensorDescriptor& a,
                                      const MetalTensorDescriptor& b,
                                      const MetalTensorDescriptor& c) {
    // Logic to select the best kernel variant based on:
    // - Matrix sizes
    // - Data types
    // - Device capabilities
    // - Memory layout
    
    size_t M = a.shape()[a.shape().size() - 2];
    size_t N = b.shape()[b.shape().size() - 1];
    size_t K = a.shape()[a.shape().size() - 1];
    
    if (a.dtype() == DataType::FLOAT16) {
        pipeline_state_ = half_pipeline_;
    } else if (M * N > 1024 * 1024) {
        // Large matrices benefit from SIMD operations
        pipeline_state_ = simdgroup_pipeline_;
    } else if (M * N > 256 * 256) {
        // Medium matrices use tiling
        pipeline_state_ = tiled_pipeline_;
    } else {
        // Small matrices use basic kernel
        pipeline_state_ = basic_pipeline_;
    }
}

// BatchedGEMMKernel implementation
BatchedGEMMKernel::BatchedGEMMKernel()
    : MetalKernel(KernelType::GEMM, "batched_gemm") {
}

void BatchedGEMMKernel::encode(id<MTLComputeCommandEncoder> encoder,
                              const std::vector<id<MTLBuffer>>& inputs,
                              const std::vector<id<MTLBuffer>>& outputs,
                              const KernelConfig& config) {
    @autoreleasepool {
        if (inputs.size() < 2 || outputs.empty()) {
            throw std::runtime_error("Batched GEMM requires at least 2 inputs and 1 output");
        }
        
        id<MTLBuffer> A = inputs[0];
        id<MTLBuffer> B = inputs[1];
        id<MTLBuffer> C = outputs[0];
        
        uint batch_size = config.int_params.at("batch_size");
        uint M = config.int_params.at("M");
        uint N = config.int_params.at("N");
        uint K = config.int_params.at("K");
        
        [encoder setComputePipelineState:pipeline_state_];
        
        [encoder setBuffer:A offset:0 atIndex:0];
        [encoder setBuffer:B offset:0 atIndex:1];
        [encoder setBuffer:C offset:0 atIndex:2];
        [encoder setBytes:&batch_size length:sizeof(uint) atIndex:3];
        [encoder setBytes:&M length:sizeof(uint) atIndex:4];
        [encoder setBytes:&N length:sizeof(uint) atIndex:5];
        [encoder setBytes:&K length:sizeof(uint) atIndex:6];
        [encoder setBytes:&alpha_ length:sizeof(float) atIndex:7];
        [encoder setBytes:&beta_ length:sizeof(float) atIndex:8];
        
        MTLSize gridSize = MTLSizeMake(N, M, batch_size);
        [encoder dispatchThreads:gridSize 
            threadsPerThreadgroup:config.threadgroup_size];
    }
}

KernelConfig BatchedGEMMKernel::suggest_config(const std::vector<MetalTensorDescriptor>& inputs,
                                              const std::vector<MetalTensorDescriptor>& outputs) const {
    KernelConfig config;
    
    const auto& a_shape = inputs[0].shape();
    const auto& b_shape = inputs[1].shape();
    
    size_t batch_size = a_shape[0];
    size_t M = a_shape[a_shape.size() - 2];
    size_t N = b_shape[b_shape.size() - 1];
    size_t K = a_shape[a_shape.size() - 1];
    
    config.int_params["batch_size"] = batch_size;
    config.int_params["M"] = M;
    config.int_params["N"] = N;
    config.int_params["K"] = K;
    
    config.threadgroup_size = MTLSizeMake(16, 16, 1);
    
    return config;
}

bool BatchedGEMMKernel::validate(const std::vector<MetalTensorDescriptor>& inputs,
                                const std::vector<MetalTensorDescriptor>& outputs) const {
    if (inputs.size() < 2 || outputs.empty()) {
        return false;
    }
    
    const auto& a_shape = inputs[0].shape();
    const auto& b_shape = inputs[1].shape();
    const auto& c_shape = outputs[0].shape();
    
    // Check for batch dimension
    if (a_shape.size() < 3 || b_shape.size() < 3 || c_shape.size() < 3) {
        return false;
    }
    
    // Check batch sizes match
    if (a_shape[0] != b_shape[0] || a_shape[0] != c_shape[0]) {
        return false;
    }
    
    // Check matrix dimensions
    size_t M = a_shape[a_shape.size() - 2];
    size_t K1 = a_shape[a_shape.size() - 1];
    size_t K2 = b_shape[b_shape.size() - 2];
    size_t N = b_shape[b_shape.size() - 1];
    
    if (K1 != K2) {
        return false;
    }
    
    if (c_shape[c_shape.size() - 2] != M || c_shape[c_shape.size() - 1] != N) {
        return false;
    }
    
    return true;
}

} // namespace kernels
} // namespace metal
} // namespace triton