#include "metal_kernel_library.h"
#include <sstream>
#include <iomanip>
#include <fstream>
#include <iostream>

namespace triton {
namespace metal {
namespace kernels {

// MetalTensorDescriptor implementation
MetalTensorDescriptor::MetalTensorDescriptor(const std::vector<size_t>& shape,
                                             DataType dtype,
                                             TensorLayout layout)
    : shape_(shape), dtype_(dtype), layout_(layout) {
    dtype_size_ = get_dtype_size(dtype);
    compute_size();
}

size_t MetalTensorDescriptor::dim(size_t index) const {
    if (index >= shape_.size()) {
        throw std::out_of_range("Dimension index out of range");
    }
    return shape_[index];
}

std::vector<size_t> MetalTensorDescriptor::strides() const {
    std::vector<size_t> strides(shape_.size());
    size_t stride = 1;
    
    for (int i = shape_.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape_[i];
    }
    
    return strides;
}

MetalTensorDescriptor MetalTensorDescriptor::with_layout(TensorLayout new_layout) const {
    MetalTensorDescriptor desc = *this;
    desc.layout_ = new_layout;
    return desc;
}

bool MetalTensorDescriptor::is_contiguous() const {
    // For now, assume all tensors are contiguous
    return true;
}

bool MetalTensorDescriptor::is_valid() const {
    return !shape_.empty() && total_size_ > 0;
}

void MetalTensorDescriptor::compute_size() {
    total_size_ = 1;
    for (size_t dim : shape_) {
        total_size_ *= dim;
    }
}

size_t MetalTensorDescriptor::get_dtype_size(DataType dtype) const {
    switch (dtype) {
        case DataType::FLOAT32:
        case DataType::INT32:
        case DataType::UINT32:
            return 4;
        case DataType::FLOAT16:
        case DataType::INT16:
        case DataType::UINT16:
            return 2;
        case DataType::INT8:
        case DataType::UINT8:
        case DataType::BOOL:
            return 1;
        default:
            return 4;
    }
}

// KernelMetrics implementation
void KernelMetrics::print() const {
    std::cout << "Kernel Metrics:" << std::endl;
    std::cout << "  Execution Time: " << std::fixed << std::setprecision(3) 
              << execution_time_ms << " ms" << std::endl;
    std::cout << "  Memory Bandwidth: " << std::fixed << std::setprecision(2) 
              << memory_bandwidth_gbps << " GB/s" << std::endl;
    std::cout << "  Compute Efficiency: " << std::fixed << std::setprecision(2) 
              << compute_efficiency * 100 << "%" << std::endl;
    std::cout << "  FLOPs: " << flops << std::endl;
    std::cout << "  Memory Reads: " << memory_reads << " bytes" << std::endl;
    std::cout << "  Memory Writes: " << memory_writes << " bytes" << std::endl;
}

// MetalKernel implementation
MetalKernel::MetalKernel(KernelType type, const std::string& name)
    : type_(type), name_(name), pipeline_state_(nil) {}

KernelMetrics MetalKernel::estimate_performance(const std::vector<MetalTensorDescriptor>& inputs,
                                               const std::vector<MetalTensorDescriptor>& outputs,
                                               const KernelConfig& config) const {
    KernelMetrics metrics;
    
    // Calculate memory operations
    for (const auto& input : inputs) {
        metrics.memory_reads += input.bytes();
    }
    for (const auto& output : outputs) {
        metrics.memory_writes += output.bytes();
    }
    
    // Estimate based on kernel type
    switch (type_) {
        case KernelType::GEMM:
            if (inputs.size() >= 2) {
                size_t m = inputs[0].shape()[0];
                size_t k = inputs[0].shape()[1];
                size_t n = inputs[1].shape()[1];
                metrics.flops = 2 * m * n * k;  // 2 ops per MAC
            }
            break;
        case KernelType::CONV2D:
            // Simplified estimation
            if (!outputs.empty()) {
                metrics.flops = outputs[0].size() * 9 * 2;  // Assume 3x3 kernel
            }
            break;
        default:
            // Simple element-wise estimation
            if (!outputs.empty()) {
                metrics.flops = outputs[0].size();
            }
            break;
    }
    
    // Rough performance estimation (placeholder values)
    metrics.execution_time_ms = metrics.flops / 1e9;  // Assume 1 TFLOPS
    double total_memory = metrics.memory_reads + metrics.memory_writes;
    metrics.memory_bandwidth_gbps = (total_memory / 1e9) / (metrics.execution_time_ms / 1000);
    metrics.compute_efficiency = 0.8;  // Placeholder
    
    return metrics;
}

MTLSize MetalKernel::calculate_threadgroup_size(size_t total_threads, size_t max_threads_per_group) const {
    // Simple heuristic for threadgroup size
    size_t threads = std::min(total_threads, max_threads_per_group);
    
    // Try to make it a power of 2
    size_t power_of_2 = 1;
    while (power_of_2 * 2 <= threads) {
        power_of_2 *= 2;
    }
    
    return MTLSizeMake(power_of_2, 1, 1);
}

MTLSize MetalKernel::calculate_grid_size(size_t total_threads, MTLSize threadgroup_size) const {
    size_t groups = (total_threads + threadgroup_size.width - 1) / threadgroup_size.width;
    return MTLSizeMake(groups, 1, 1);
}

// MetalKernelCompiler implementation
MetalKernelCompiler::MetalKernelCompiler(id<MTLDevice> device)
    : device_(device) {}

MetalKernelCompiler::~MetalKernelCompiler() = default;

id<MTLComputePipelineState> MetalKernelCompiler::compile(const std::string& source,
                                                        const std::string& function_name,
                                                        const std::unordered_map<std::string, std::string>& defines) {
    @autoreleasepool {
        NSError* error = nil;
        
        // Create compile options
        MTLCompileOptions* options = create_compile_options(defines);
        
        // Compile the source
        NSString* source_ns = [NSString stringWithUTF8String:source.c_str()];
        id<MTLLibrary> library = [device_ newLibraryWithSource:source_ns
                                                       options:options
                                                         error:&error];
        
        if (error) {
            NSLog(@"Failed to compile Metal kernel: %@", error);
            return nil;
        }
        
        // Get the function
        NSString* function_name_ns = [NSString stringWithUTF8String:function_name.c_str()];
        id<MTLFunction> function = [library newFunctionWithName:function_name_ns];
        
        if (!function) {
            NSLog(@"Failed to find function: %@", function_name_ns);
            return nil;
        }
        
        // Create pipeline state
        id<MTLComputePipelineState> pipeline_state = [device_ newComputePipelineStateWithFunction:function
                                                                                            error:&error];
        
        if (error) {
            NSLog(@"Failed to create pipeline state: %@", error);
            return nil;
        }
        
        return pipeline_state;
    }
}

id<MTLComputePipelineState> MetalKernelCompiler::compile_from_file(const std::string& filepath,
                                                                  const std::string& function_name,
                                                                  const std::unordered_map<std::string, std::string>& defines) {
    // Read file content
    std::ifstream file(filepath);
    if (!file.is_open()) {
        NSLog(@"Failed to open Metal shader file: %s", filepath.c_str());
        return nil;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    
    return compile(buffer.str(), function_name, defines);
}

std::string MetalKernelCompiler::generate_gemm_kernel(DataType dtype, bool use_shared_memory, bool use_simd) {
    std::stringstream ss;
    std::string metal_type = dtype_to_metal_type(dtype);
    
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n\n";
    
    if (use_shared_memory) {
        ss << "kernel void gemm_shared(\n";
        ss << "    device const " << metal_type << "* A [[buffer(0)]],\n";
        ss << "    device const " << metal_type << "* B [[buffer(1)]],\n";
        ss << "    device " << metal_type << "* C [[buffer(2)]],\n";
        ss << "    constant uint& M [[buffer(3)]],\n";
        ss << "    constant uint& N [[buffer(4)]],\n";
        ss << "    constant uint& K [[buffer(5)]],\n";
        ss << "    threadgroup " << metal_type << "* shared_A [[threadgroup(0)]],\n";
        ss << "    threadgroup " << metal_type << "* shared_B [[threadgroup(1)]],\n";
        ss << "    uint2 gid [[thread_position_in_grid]],\n";
        ss << "    uint2 tid [[thread_position_in_threadgroup]],\n";
        ss << "    uint2 tgSize [[threads_per_threadgroup]]) {\n";
        
        ss << "    const uint TILE_SIZE = 16;\n";
        ss << "    const uint row = gid.y;\n";
        ss << "    const uint col = gid.x;\n";
        ss << "    const uint localRow = tid.y;\n";
        ss << "    const uint localCol = tid.x;\n\n";
        
        ss << "    " << metal_type << " sum = 0.0;\n\n";
        
        ss << "    for (uint tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {\n";
        ss << "        // Load tile into shared memory\n";
        ss << "        if (row < M && tile * TILE_SIZE + localCol < K) {\n";
        ss << "            shared_A[localRow * TILE_SIZE + localCol] = A[row * K + tile * TILE_SIZE + localCol];\n";
        ss << "        } else {\n";
        ss << "            shared_A[localRow * TILE_SIZE + localCol] = 0.0;\n";
        ss << "        }\n\n";
        
        ss << "        if (col < N && tile * TILE_SIZE + localRow < K) {\n";
        ss << "            shared_B[localRow * TILE_SIZE + localCol] = B[(tile * TILE_SIZE + localRow) * N + col];\n";
        ss << "        } else {\n";
        ss << "            shared_B[localRow * TILE_SIZE + localCol] = 0.0;\n";
        ss << "        }\n\n";
        
        ss << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n";
        
        ss << "        // Compute partial dot product\n";
        ss << "        for (uint k = 0; k < TILE_SIZE; ++k) {\n";
        ss << "            sum += shared_A[localRow * TILE_SIZE + k] * shared_B[k * TILE_SIZE + localCol];\n";
        ss << "        }\n\n";
        
        ss << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        ss << "    }\n\n";
        
        ss << "    // Write result\n";
        ss << "    if (row < M && col < N) {\n";
        ss << "        C[row * N + col] = sum;\n";
        ss << "    }\n";
        ss << "}\n";
    } else {
        ss << "kernel void gemm_simple(\n";
        ss << "    device const " << metal_type << "* A [[buffer(0)]],\n";
        ss << "    device const " << metal_type << "* B [[buffer(1)]],\n";
        ss << "    device " << metal_type << "* C [[buffer(2)]],\n";
        ss << "    constant uint& M [[buffer(3)]],\n";
        ss << "    constant uint& N [[buffer(4)]],\n";
        ss << "    constant uint& K [[buffer(5)]],\n";
        ss << "    uint2 gid [[thread_position_in_grid]]) {\n";
        
        ss << "    const uint row = gid.y;\n";
        ss << "    const uint col = gid.x;\n\n";
        
        ss << "    if (row >= M || col >= N) return;\n\n";
        
        ss << "    " << metal_type << " sum = 0.0;\n";
        
        if (use_simd) {
            ss << "    // SIMD-optimized loop\n";
            ss << "    const uint simd_width = 4;\n";
            ss << "    for (uint k = 0; k < K - simd_width + 1; k += simd_width) {\n";
            ss << "        " << metal_type << "4 a = " << metal_type << "4(";
            ss << "A[row * K + k], A[row * K + k + 1], ";
            ss << "A[row * K + k + 2], A[row * K + k + 3]);\n";
            ss << "        " << metal_type << "4 b = " << metal_type << "4(";
            ss << "B[k * N + col], B[(k + 1) * N + col], ";
            ss << "B[(k + 2) * N + col], B[(k + 3) * N + col]);\n";
            ss << "        sum += dot(a, b);\n";
            ss << "    }\n";
            ss << "    // Handle remaining elements\n";
            ss << "    for (uint k = K - (K % simd_width); k < K; ++k) {\n";
            ss << "        sum += A[row * K + k] * B[k * N + col];\n";
            ss << "    }\n";
        } else {
            ss << "    for (uint k = 0; k < K; ++k) {\n";
            ss << "        sum += A[row * K + k] * B[k * N + col];\n";
            ss << "    }\n";
        }
        
        ss << "\n    C[row * N + col] = sum;\n";
        ss << "}\n";
    }
    
    return ss.str();
}

std::string MetalKernelCompiler::generate_conv2d_kernel(DataType dtype, int kernel_h, int kernel_w,
                                                       int stride_h, int stride_w, int pad_h, int pad_w) {
    std::stringstream ss;
    std::string metal_type = dtype_to_metal_type(dtype);
    
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n\n";
    
    ss << "kernel void conv2d(\n";
    ss << "    device const " << metal_type << "* input [[buffer(0)]],\n";
    ss << "    device const " << metal_type << "* weight [[buffer(1)]],\n";
    ss << "    device const " << metal_type << "* bias [[buffer(2)]],\n";
    ss << "    device " << metal_type << "* output [[buffer(3)]],\n";
    ss << "    constant uint4& input_shape [[buffer(4)]],  // NCHW\n";
    ss << "    constant uint4& output_shape [[buffer(5)]], // NCHW\n";
    ss << "    uint3 gid [[thread_position_in_grid]]) {\n";
    
    ss << "    const uint batch = gid.z;\n";
    ss << "    const uint out_c = gid.y;\n";
    ss << "    const uint out_x = gid.x;\n\n";
    
    ss << "    const uint in_channels = input_shape.y;\n";
    ss << "    const uint in_height = input_shape.z;\n";
    ss << "    const uint in_width = input_shape.w;\n";
    ss << "    const uint out_channels = output_shape.y;\n";
    ss << "    const uint out_height = output_shape.z;\n";
    ss << "    const uint out_width = output_shape.w;\n\n";
    
    ss << "    if (batch >= output_shape.x || out_c >= out_channels || out_x >= out_width) return;\n\n";
    
    ss << "    const uint out_y = out_x / out_width;\n";
    ss << "    const uint out_x_actual = out_x % out_width;\n\n";
    
    ss << "    " << metal_type << " sum = bias ? bias[out_c] : 0.0;\n\n";
    
    ss << "    for (uint in_c = 0; in_c < in_channels; ++in_c) {\n";
    ss << "        for (int kh = 0; kh < " << kernel_h << "; ++kh) {\n";
    ss << "            for (int kw = 0; kw < " << kernel_w << "; ++kw) {\n";
    ss << "                int in_y = out_y * " << stride_h << " - " << pad_h << " + kh;\n";
    ss << "                int in_x = out_x_actual * " << stride_w << " - " << pad_w << " + kw;\n\n";
    
    ss << "                if (in_y >= 0 && in_y < int(in_height) && in_x >= 0 && in_x < int(in_width)) {\n";
    ss << "                    uint input_idx = batch * in_channels * in_height * in_width +\n";
    ss << "                                     in_c * in_height * in_width +\n";
    ss << "                                     in_y * in_width + in_x;\n";
    ss << "                    uint weight_idx = out_c * in_channels * " << kernel_h << " * " << kernel_w << " +\n";
    ss << "                                      in_c * " << kernel_h << " * " << kernel_w << " +\n";
    ss << "                                      kh * " << kernel_w << " + kw;\n";
    ss << "                    sum += input[input_idx] * weight[weight_idx];\n";
    ss << "                }\n";
    ss << "            }\n";
    ss << "        }\n";
    ss << "    }\n\n";
    
    ss << "    uint output_idx = batch * out_channels * out_height * out_width +\n";
    ss << "                      out_c * out_height * out_width +\n";
    ss << "                      out_y * out_width + out_x_actual;\n";
    ss << "    output[output_idx] = sum;\n";
    ss << "}\n";
    
    return ss.str();
}

std::string MetalKernelCompiler::generate_activation_kernel(KernelType activation, DataType dtype) {
    std::stringstream ss;
    std::string metal_type = dtype_to_metal_type(dtype);
    
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n\n";
    
    std::string func_name;
    std::string operation;
    
    switch (activation) {
        case KernelType::RELU:
            func_name = "relu";
            operation = "max(x, 0.0)";
            break;
        case KernelType::RELU6:
            func_name = "relu6";
            operation = "min(max(x, 0.0), 6.0)";
            break;
        case KernelType::LEAKY_RELU:
            func_name = "leaky_relu";
            operation = "x >= 0 ? x : alpha * x";
            break;
        case KernelType::SIGMOID:
            func_name = "sigmoid";
            operation = "1.0 / (1.0 + exp(-x))";
            break;
        case KernelType::TANH:
            func_name = "tanh_activation";
            operation = "tanh(x)";
            break;
        case KernelType::GELU:
            func_name = "gelu";
            operation = "0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI_F) * (x + 0.044715 * pow(x, 3))))";
            break;
        case KernelType::SWISH:
            func_name = "swish";
            operation = "x / (1.0 + exp(-x))";
            break;
        case KernelType::ELU:
            func_name = "elu";
            operation = "x >= 0 ? x : alpha * (exp(x) - 1.0)";
            break;
        default:
            func_name = "identity";
            operation = "x";
            break;
    }
    
    ss << "kernel void " << func_name << "(\n";
    ss << "    device const " << metal_type << "* input [[buffer(0)]],\n";
    ss << "    device " << metal_type << "* output [[buffer(1)]],\n";
    
    if (activation == KernelType::LEAKY_RELU || activation == KernelType::ELU) {
        ss << "    constant " << metal_type << "& alpha [[buffer(2)]],\n";
    }
    
    ss << "    uint gid [[thread_position_in_grid]]) {\n";
    ss << "    " << metal_type << " x = input[gid];\n";
    ss << "    output[gid] = " << operation << ";\n";
    ss << "}\n";
    
    return ss.str();
}

MTLCompileOptions* MetalKernelCompiler::create_compile_options(const std::unordered_map<std::string, std::string>& defines) const {
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    
    // Set optimization level
    switch (optimization_level_) {
        case 0:
            [options setOptimizationLevel:MTLLibraryOptimizationLevelDefault];
            break;
        case 1:
            [options setOptimizationLevel:MTLLibraryOptimizationLevelSize];
            break;
        case 2:
        default:
            [options setOptimizationLevel:MTLLibraryOptimizationLevelPerformance];
            break;
    }
    
    // Set fast math
    [options setFastMathEnabled:fast_math_];
    
    // Add preprocessor macros
    if (!defines.empty()) {
        NSMutableDictionary* macros = [NSMutableDictionary dictionary];
        for (const auto& [key, value] : defines) {
            NSString* key_ns = [NSString stringWithUTF8String:key.c_str()];
            NSString* value_ns = [NSString stringWithUTF8String:value.c_str()];
            macros[key_ns] = value_ns;
        }
        [options setPreprocessorMacros:macros];
    }
    
    return options;
}

// MetalKernelCache implementation
MetalKernelCache::MetalKernelCache(size_t max_cache_size)
    : max_cache_size_(max_cache_size) {}

MetalKernelCache::~MetalKernelCache() = default;

void MetalKernelCache::put(const std::string& key, std::shared_ptr<MetalKernel> kernel) {
    evict_if_needed();
    cache_[key] = kernel;
}

std::shared_ptr<MetalKernel> MetalKernelCache::get(const std::string& key) const {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        ++hit_count_;
        return it->second;
    }
    ++miss_count_;
    return nullptr;
}

bool MetalKernelCache::contains(const std::string& key) const {
    return cache_.find(key) != cache_.end();
}

void MetalKernelCache::clear() {
    cache_.clear();
    hit_count_ = 0;
    miss_count_ = 0;
}

double MetalKernelCache::hit_rate() const {
    size_t total = hit_count_ + miss_count_;
    return total > 0 ? static_cast<double>(hit_count_) / total : 0.0;
}

void MetalKernelCache::evict_if_needed() {
    // Simple LRU eviction would be implemented here
    // For now, just clear if we exceed the limit
    if (cache_.size() >= max_cache_size_) {
        // Remove the first element (not true LRU)
        cache_.erase(cache_.begin());
    }
}

// MetalKernelLibrary implementation
MetalKernelLibrary& MetalKernelLibrary::instance() {
    static MetalKernelLibrary instance;
    return instance;
}

MetalKernelLibrary::MetalKernelLibrary() = default;

MetalKernelLibrary::~MetalKernelLibrary() = default;

void MetalKernelLibrary::initialize(id<MTLDevice> device) {
    device_ = device;
    cache_ = std::make_unique<MetalKernelCache>();
    compiler_ = std::make_unique<MetalKernelCompiler>(device);
    register_builtin_kernels();
}

void MetalKernelLibrary::shutdown() {
    cache_.reset();
    compiler_.reset();
    device_ = nil;
}

std::shared_ptr<MetalKernel> MetalKernelLibrary::get_kernel(KernelType type,
                                                           const std::vector<MetalTensorDescriptor>& inputs,
                                                           const std::vector<MetalTensorDescriptor>& outputs,
                                                           const KernelConfig& config) {
    // Generate cache key
    std::string cache_key = generate_cache_key(type, inputs, outputs, config);
    
    // Check cache
    auto cached_kernel = cache_->get(cache_key);
    if (cached_kernel) {
        return cached_kernel;
    }
    
    // Create new kernel
    auto factory_it = kernel_factories_.find(type);
    if (factory_it == kernel_factories_.end()) {
        throw std::runtime_error("Kernel type not supported: " + kernel_type_to_string(type));
    }
    
    auto kernel = factory_it->second();
    
    // Validate
    if (!kernel->validate(inputs, outputs)) {
        throw std::runtime_error("Kernel validation failed");
    }
    
    // Cache and return
    cache_->put(cache_key, kernel);
    
    // Profile if enabled
    if (profiling_enabled_ && profiling_callback_) {
        auto metrics = kernel->estimate_performance(inputs, outputs, config);
        profiling_callback_(kernel->name(), metrics);
    }
    
    return kernel;
}

void MetalKernelLibrary::register_custom_kernel(const std::string& name,
                                               std::shared_ptr<MetalKernel> kernel) {
    // For custom kernels, we can register them directly in the cache
    cache_->put("custom_" + name, kernel);
}

void MetalKernelLibrary::set_profiling_callback(std::function<void(const std::string&, const KernelMetrics&)> callback) {
    profiling_callback_ = callback;
}

void MetalKernelLibrary::register_builtin_kernels() {
    // Register kernel factories
    // These will be implemented in separate files for each kernel type
    
    // Note: Actual kernel implementations would be registered here
    // For now, this is a placeholder
}

std::string MetalKernelLibrary::generate_cache_key(KernelType type,
                                                   const std::vector<MetalTensorDescriptor>& inputs,
                                                   const std::vector<MetalTensorDescriptor>& outputs,
                                                   const KernelConfig& config) const {
    std::stringstream ss;
    ss << kernel_type_to_string(type);
    
    // Add input shapes and types
    for (const auto& input : inputs) {
        ss << "_I";
        for (size_t dim : input.shape()) {
            ss << dim << "x";
        }
        ss << static_cast<int>(input.dtype());
    }
    
    // Add output shapes and types
    for (const auto& output : outputs) {
        ss << "_O";
        for (size_t dim : output.shape()) {
            ss << dim << "x";
        }
        ss << static_cast<int>(output.dtype());
    }
    
    // Add config parameters
    ss << "_TG" << config.threadgroup_size.width << "x" << config.threadgroup_size.height;
    ss << "_HP" << config.use_half_precision;
    ss << "_FA" << config.use_fused_activation;
    
    return ss.str();
}

// Utility functions
std::string dtype_to_metal_type(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return "float";
        case DataType::FLOAT16: return "half";
        case DataType::INT32: return "int";
        case DataType::INT16: return "short";
        case DataType::INT8: return "char";
        case DataType::UINT32: return "uint";
        case DataType::UINT16: return "ushort";
        case DataType::UINT8: return "uchar";
        case DataType::BOOL: return "bool";
        default: return "float";
    }
}

size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32:
        case DataType::INT32:
        case DataType::UINT32:
            return 4;
        case DataType::FLOAT16:
        case DataType::INT16:
        case DataType::UINT16:
            return 2;
        case DataType::INT8:
        case DataType::UINT8:
        case DataType::BOOL:
            return 1;
        default:
            return 4;
    }
}

std::string kernel_type_to_string(KernelType type) {
    switch (type) {
        case KernelType::GEMM: return "gemm";
        case KernelType::GEMV: return "gemv";
        case KernelType::CONV2D: return "conv2d";
        case KernelType::CONV3D: return "conv3d";
        case KernelType::RELU: return "relu";
        case KernelType::SIGMOID: return "sigmoid";
        case KernelType::TANH: return "tanh";
        case KernelType::SOFTMAX: return "softmax";
        case KernelType::BATCH_NORM: return "batch_norm";
        case KernelType::LAYER_NORM: return "layer_norm";
        case KernelType::MAXPOOL2D: return "maxpool2d";
        case KernelType::AVGPOOL2D: return "avgpool2d";
        case KernelType::ADD: return "add";
        case KernelType::MUL: return "mul";
        case KernelType::TRANSPOSE: return "transpose";
        case KernelType::RESHAPE: return "reshape";
        case KernelType::REDUCE_SUM: return "reduce_sum";
        case KernelType::REDUCE_MEAN: return "reduce_mean";
        default: return "unknown";
    }
}

} // namespace kernels
} // namespace metal
} // namespace triton