#pragma once

#include <Metal/Metal.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <functional>

namespace triton {
namespace metal {
namespace kernels {

// Forward declarations
class MetalKernelCache;
class MetalKernelCompiler;
class MetalTensorDescriptor;

// Kernel type enumeration
enum class KernelType {
    // Math operations
    GEMM,
    GEMV,
    AXPY,
    DOT,
    
    // Neural network operations
    CONV2D,
    CONV3D,
    DEPTHWISE_CONV2D,
    MAXPOOL2D,
    AVGPOOL2D,
    MAXPOOL3D,
    AVGPOOL3D,
    
    // Activations
    RELU,
    RELU6,
    LEAKY_RELU,
    SIGMOID,
    TANH,
    GELU,
    SWISH,
    ELU,
    SOFTMAX,
    
    // Normalization
    BATCH_NORM,
    LAYER_NORM,
    INSTANCE_NORM,
    GROUP_NORM,
    
    // Element-wise operations
    ADD,
    SUB,
    MUL,
    DIV,
    POW,
    EXP,
    LOG,
    SQRT,
    ABS,
    NEG,
    
    // Reduction operations
    REDUCE_SUM,
    REDUCE_MEAN,
    REDUCE_MAX,
    REDUCE_MIN,
    REDUCE_PROD,
    
    // Utility operations
    TRANSPOSE,
    RESHAPE,
    CONCAT,
    SPLIT,
    SLICE,
    PAD,
    CAST,
    
    // Custom
    CUSTOM
};

// Data type enumeration
enum class DataType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT16,
    INT8,
    UINT32,
    UINT16,
    UINT8,
    BOOL
};

// Tensor layout enumeration
enum class TensorLayout {
    NCHW,  // Batch, Channels, Height, Width
    NHWC,  // Batch, Height, Width, Channels
    NCDHW, // Batch, Channels, Depth, Height, Width
    NDHWC, // Batch, Depth, Height, Width, Channels
    NC,    // Batch, Channels (for fully connected)
    NCW,   // Batch, Channels, Width (for 1D conv)
    NWC    // Batch, Width, Channels (for 1D conv)
};

// Kernel configuration
struct KernelConfig {
    // Threadgroup configuration
    MTLSize threadgroup_size;
    MTLSize grid_size;
    
    // Memory configuration
    size_t shared_memory_size = 0;
    
    // Optimization hints
    bool use_simd_group = true;
    bool use_half_precision = false;
    bool use_fused_activation = false;
    
    // Performance tuning
    int cache_hint = 0;
    int memory_order_hint = 0;
    
    // Custom parameters
    std::unordered_map<std::string, float> float_params;
    std::unordered_map<std::string, int> int_params;
};

// Tensor descriptor for Metal kernels
class MetalTensorDescriptor {
public:
    MetalTensorDescriptor() = default;
    MetalTensorDescriptor(const std::vector<size_t>& shape,
                          DataType dtype,
                          TensorLayout layout = TensorLayout::NCHW);
    
    // Getters
    const std::vector<size_t>& shape() const { return shape_; }
    size_t rank() const { return shape_.size(); }
    size_t size() const { return total_size_; }
    size_t bytes() const { return total_size_ * dtype_size_; }
    DataType dtype() const { return dtype_; }
    TensorLayout layout() const { return layout_; }
    
    // Shape utilities
    size_t dim(size_t index) const;
    std::vector<size_t> strides() const;
    
    // Layout conversion
    MetalTensorDescriptor with_layout(TensorLayout new_layout) const;
    
    // Validation
    bool is_contiguous() const;
    bool is_valid() const;
    
private:
    std::vector<size_t> shape_;
    DataType dtype_;
    TensorLayout layout_;
    size_t total_size_;
    size_t dtype_size_;
    
    void compute_size();
    size_t get_dtype_size(DataType dtype) const;
};

// Performance metrics
struct KernelMetrics {
    double execution_time_ms = 0.0;
    double memory_bandwidth_gbps = 0.0;
    double compute_efficiency = 0.0;
    size_t flops = 0;
    size_t memory_reads = 0;
    size_t memory_writes = 0;
    
    void print() const;
};

// Base class for all Metal kernels
class MetalKernel {
public:
    MetalKernel(KernelType type, const std::string& name);
    virtual ~MetalKernel() = default;
    
    // Kernel execution
    virtual void encode(id<MTLComputeCommandEncoder> encoder,
                       const std::vector<id<MTLBuffer>>& inputs,
                       const std::vector<id<MTLBuffer>>& outputs,
                       const KernelConfig& config) = 0;
    
    // Configuration
    virtual KernelConfig suggest_config(const std::vector<MetalTensorDescriptor>& inputs,
                                       const std::vector<MetalTensorDescriptor>& outputs) const = 0;
    
    // Validation
    virtual bool validate(const std::vector<MetalTensorDescriptor>& inputs,
                         const std::vector<MetalTensorDescriptor>& outputs) const = 0;
    
    // Performance estimation
    virtual KernelMetrics estimate_performance(const std::vector<MetalTensorDescriptor>& inputs,
                                              const std::vector<MetalTensorDescriptor>& outputs,
                                              const KernelConfig& config) const;
    
    // Getters
    KernelType type() const { return type_; }
    const std::string& name() const { return name_; }
    id<MTLComputePipelineState> pipeline_state() const { return pipeline_state_; }
    
    // Setters
    void set_pipeline_state(id<MTLComputePipelineState> state) { pipeline_state_ = state; }
    
protected:
    KernelType type_;
    std::string name_;
    id<MTLComputePipelineState> pipeline_state_;
    
    // Helper methods
    MTLSize calculate_threadgroup_size(size_t total_threads, size_t max_threads_per_group) const;
    MTLSize calculate_grid_size(size_t total_threads, MTLSize threadgroup_size) const;
};

// Kernel compiler for runtime compilation
class MetalKernelCompiler {
public:
    MetalKernelCompiler(id<MTLDevice> device);
    ~MetalKernelCompiler();
    
    // Compilation methods
    id<MTLComputePipelineState> compile(const std::string& source,
                                        const std::string& function_name,
                                        const std::unordered_map<std::string, std::string>& defines = {});
    
    id<MTLComputePipelineState> compile_from_file(const std::string& filepath,
                                                  const std::string& function_name,
                                                  const std::unordered_map<std::string, std::string>& defines = {});
    
    // Specialized kernel generation
    std::string generate_gemm_kernel(DataType dtype, bool use_shared_memory, bool use_simd);
    std::string generate_conv2d_kernel(DataType dtype, int kernel_h, int kernel_w, 
                                      int stride_h, int stride_w, int pad_h, int pad_w);
    std::string generate_activation_kernel(KernelType activation, DataType dtype);
    
    // Options
    void set_optimization_level(int level) { optimization_level_ = level; }
    void set_fast_math(bool enabled) { fast_math_ = enabled; }
    
private:
    id<MTLDevice> device_;
    int optimization_level_ = 2;
    bool fast_math_ = true;
    
    MTLCompileOptions* create_compile_options(const std::unordered_map<std::string, std::string>& defines) const;
};

// Kernel cache for compiled kernels
class MetalKernelCache {
public:
    MetalKernelCache(size_t max_cache_size = 1000);
    ~MetalKernelCache();
    
    // Cache operations
    void put(const std::string& key, std::shared_ptr<MetalKernel> kernel);
    std::shared_ptr<MetalKernel> get(const std::string& key) const;
    bool contains(const std::string& key) const;
    void clear();
    
    // Statistics
    size_t size() const { return cache_.size(); }
    size_t hits() const { return hit_count_; }
    size_t misses() const { return miss_count_; }
    double hit_rate() const;
    
private:
    std::unordered_map<std::string, std::shared_ptr<MetalKernel>> cache_;
    size_t max_cache_size_;
    mutable size_t hit_count_ = 0;
    mutable size_t miss_count_ = 0;
    
    void evict_if_needed();
};

// Main kernel library class
class MetalKernelLibrary {
public:
    static MetalKernelLibrary& instance();
    
    // Initialization
    void initialize(id<MTLDevice> device);
    void shutdown();
    
    // Kernel retrieval
    std::shared_ptr<MetalKernel> get_kernel(KernelType type,
                                           const std::vector<MetalTensorDescriptor>& inputs,
                                           const std::vector<MetalTensorDescriptor>& outputs,
                                           const KernelConfig& config = {});
    
    // Custom kernel registration
    void register_custom_kernel(const std::string& name,
                               std::shared_ptr<MetalKernel> kernel);
    
    // Performance profiling
    void enable_profiling(bool enabled) { profiling_enabled_ = enabled; }
    void set_profiling_callback(std::function<void(const std::string&, const KernelMetrics&)> callback);
    
    // Cache management
    MetalKernelCache& cache() { return *cache_; }
    MetalKernelCompiler& compiler() { return *compiler_; }
    
    // Device info
    id<MTLDevice> device() const { return device_; }
    
private:
    MetalKernelLibrary();
    ~MetalKernelLibrary();
    
    // Prevent copying
    MetalKernelLibrary(const MetalKernelLibrary&) = delete;
    MetalKernelLibrary& operator=(const MetalKernelLibrary&) = delete;
    
    id<MTLDevice> device_;
    std::unique_ptr<MetalKernelCache> cache_;
    std::unique_ptr<MetalKernelCompiler> compiler_;
    bool profiling_enabled_ = false;
    std::function<void(const std::string&, const KernelMetrics&)> profiling_callback_;
    
    // Kernel factories
    std::unordered_map<KernelType, std::function<std::shared_ptr<MetalKernel>()>> kernel_factories_;
    
    void register_builtin_kernels();
    std::string generate_cache_key(KernelType type,
                                  const std::vector<MetalTensorDescriptor>& inputs,
                                  const std::vector<MetalTensorDescriptor>& outputs,
                                  const KernelConfig& config) const;
};

// Utility functions
std::string dtype_to_metal_type(DataType dtype);
size_t dtype_size(DataType dtype);
std::string kernel_type_to_string(KernelType type);

} // namespace kernels
} // namespace metal
} // namespace triton