#pragma once

#include "metal_kernel_library.h"

namespace triton {
namespace metal {
namespace kernels {

// GEMM kernel implementation
class GEMMKernel : public MetalKernel {
public:
    GEMMKernel();
    ~GEMMKernel() override = default;
    
    // Note: In a full implementation, tensor descriptors would be passed here
    // to provide shape information. Currently dimensions are extracted from config.
    void encode(id<MTLComputeCommandEncoder> encoder,
                const std::vector<id<MTLBuffer>>& inputs,
                const std::vector<id<MTLBuffer>>& outputs,
                const KernelConfig& config) override;
    
    KernelConfig suggest_config(const std::vector<MetalTensorDescriptor>& inputs,
                               const std::vector<MetalTensorDescriptor>& outputs) const override;
    
    bool validate(const std::vector<MetalTensorDescriptor>& inputs,
                 const std::vector<MetalTensorDescriptor>& outputs) const override;
    
    // GEMM-specific configuration
    void set_alpha(float alpha) { alpha_ = alpha; }
    void set_beta(float beta) { beta_ = beta; }
    void set_transpose_a(bool transpose) { transpose_a_ = transpose; }
    void set_transpose_b(bool transpose) { transpose_b_ = transpose; }
    
private:
    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    bool transpose_a_ = false;
    bool transpose_b_ = false;
    
    // Different kernel variants
    id<MTLComputePipelineState> basic_pipeline_;
    id<MTLComputePipelineState> tiled_pipeline_;
    id<MTLComputePipelineState> simdgroup_pipeline_;
    id<MTLComputePipelineState> half_pipeline_;
    
    void select_optimal_kernel(const MetalTensorDescriptor& a,
                              const MetalTensorDescriptor& b,
                              const MetalTensorDescriptor& c);
};

// Batched GEMM kernel
class BatchedGEMMKernel : public MetalKernel {
public:
    BatchedGEMMKernel();
    ~BatchedGEMMKernel() override = default;
    
    void encode(id<MTLComputeCommandEncoder> encoder,
                const std::vector<id<MTLBuffer>>& inputs,
                const std::vector<id<MTLBuffer>>& outputs,
                const KernelConfig& config) override;
    
    KernelConfig suggest_config(const std::vector<MetalTensorDescriptor>& inputs,
                               const std::vector<MetalTensorDescriptor>& outputs) const override;
    
    bool validate(const std::vector<MetalTensorDescriptor>& inputs,
                 const std::vector<MetalTensorDescriptor>& outputs) const override;
    
    void set_alpha(float alpha) { alpha_ = alpha; }
    void set_beta(float beta) { beta_ = beta; }
    
private:
    float alpha_ = 1.0f;
    float beta_ = 0.0f;
};

} // namespace kernels
} // namespace metal
} // namespace triton