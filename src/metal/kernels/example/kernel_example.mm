#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../metal_kernel_library.h"
#include "../gemm_kernel.h"
#include <iostream>
#include <vector>

using namespace triton::metal::kernels;

void run_gemm_example() {
    @autoreleasepool {
        // Initialize Metal
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal device not found!" << std::endl;
            return;
        }
        
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        
        // Initialize kernel library
        MetalKernelLibrary& library = MetalKernelLibrary::instance();
        library.initialize(device);
        
        // Matrix dimensions
        const size_t M = 1024;
        const size_t N = 1024;
        const size_t K = 1024;
        
        // Create input matrices
        std::vector<float> A(M * K);
        std::vector<float> B(K * N);
        
        // Initialize with some values
        for (size_t i = 0; i < M * K; ++i) {
            A[i] = static_cast<float>(i % 100) / 100.0f;
        }
        for (size_t i = 0; i < K * N; ++i) {
            B[i] = static_cast<float>(i % 50) / 50.0f;
        }
        
        // Create Metal buffers
        id<MTLBuffer> bufferA = [device newBufferWithBytes:A.data()
                                                    length:A.size() * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> bufferB = [device newBufferWithBytes:B.data()
                                                    length:B.size() * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> bufferC = [device newBufferWithLength:M * N * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
        
        // Create tensor descriptors
        std::vector<MetalTensorDescriptor> inputs = {
            MetalTensorDescriptor({M, K}, DataType::FLOAT32),
            MetalTensorDescriptor({K, N}, DataType::FLOAT32)
        };
        std::vector<MetalTensorDescriptor> outputs = {
            MetalTensorDescriptor({M, N}, DataType::FLOAT32)
        };
        
        // Get GEMM kernel
        auto kernel = library.get_kernel(KernelType::GEMM, inputs, outputs);
        
        // Get optimal configuration
        auto config = kernel->suggest_config(inputs, outputs);
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Encode kernel execution
        kernel->encode(encoder, {bufferA, bufferB}, {bufferC}, config);
        
        // Finish encoding
        [encoder endEncoding];
        
        // Commit and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Read results
        float* result = static_cast<float*>([bufferC contents]);
        
        // Print a few results
        std::cout << "GEMM Result (first 10 elements):" << std::endl;
        for (size_t i = 0; i < 10; ++i) {
            std::cout << result[i] << " ";
        }
        std::cout << std::endl;
        
        // Print performance metrics
        auto metrics = kernel->estimate_performance(inputs, outputs, config);
        metrics.print();
        
        // Cleanup
        library.shutdown();
    }
}

void run_conv2d_example() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        
        MetalKernelLibrary& library = MetalKernelLibrary::instance();
        library.initialize(device);
        
        // Conv2D parameters
        const size_t N = 1;   // Batch size
        const size_t C = 3;   // Input channels
        const size_t H = 224; // Height
        const size_t W = 224; // Width
        const size_t K = 64;  // Output channels
        const size_t R = 3;   // Kernel height
        const size_t S = 3;   // Kernel width
        
        // Create input tensor (NCHW layout)
        std::vector<float> input(N * C * H * W, 1.0f);
        
        // Create weight tensor
        std::vector<float> weight(K * C * R * S, 0.1f);
        
        // Create bias
        std::vector<float> bias(K, 0.0f);
        
        // Calculate output dimensions (assuming stride=1, pad=1)
        const size_t P = H; // Output height
        const size_t Q = W; // Output width
        
        // Create Metal buffers
        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input.data()
                                                        length:input.size() * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> weightBuffer = [device newBufferWithBytes:weight.data()
                                                         length:weight.size() * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> biasBuffer = [device newBufferWithBytes:bias.data()
                                                       length:bias.size() * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:N * K * P * Q * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
        
        // Create tensor descriptors
        std::vector<MetalTensorDescriptor> inputs = {
            MetalTensorDescriptor({N, C, H, W}, DataType::FLOAT32, TensorLayout::NCHW),
            MetalTensorDescriptor({K, C, R, S}, DataType::FLOAT32),
            MetalTensorDescriptor({K}, DataType::FLOAT32)
        };
        std::vector<MetalTensorDescriptor> outputs = {
            MetalTensorDescriptor({N, K, P, Q}, DataType::FLOAT32, TensorLayout::NCHW)
        };
        
        // Configure convolution parameters
        KernelConfig config;
        config.int_params["stride_h"] = 1;
        config.int_params["stride_w"] = 1;
        config.int_params["pad_h"] = 1;
        config.int_params["pad_w"] = 1;
        
        // Get Conv2D kernel
        auto kernel = library.get_kernel(KernelType::CONV2D, inputs, outputs, config);
        
        // Execute kernel
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        kernel->encode(encoder, {inputBuffer, weightBuffer, biasBuffer}, {outputBuffer}, config);
        
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        std::cout << "Conv2D completed successfully!" << std::endl;
        
        library.shutdown();
    }
}

void run_activation_chain_example() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        
        MetalKernelLibrary& library = MetalKernelLibrary::instance();
        library.initialize(device);
        
        // Enable profiling
        library.enable_profiling(true);
        library.set_profiling_callback([](const std::string& name, const KernelMetrics& metrics) {
            std::cout << "Kernel: " << name << std::endl;
            metrics.print();
            std::cout << std::endl;
        });
        
        // Create test tensor
        const size_t size = 1024 * 1024;
        std::vector<float> data(size);
        
        // Initialize with random values
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        }
        
        // Create buffers
        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:data.data()
                                                        length:data.size() * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> tempBuffer = [device newBufferWithLength:size * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:size * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
        
        // Create tensor descriptor
        MetalTensorDescriptor tensorDesc({size}, DataType::FLOAT32);
        
        // Chain of operations: Input -> ReLU -> Sigmoid -> Output
        
        // 1. ReLU
        {
            auto relu = library.get_kernel(KernelType::RELU, {tensorDesc}, {tensorDesc});
            auto config = relu->suggest_config({tensorDesc}, {tensorDesc});
            
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            
            relu->encode(encoder, {inputBuffer}, {tempBuffer}, config);
            
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }
        
        // 2. Sigmoid
        {
            auto sigmoid = library.get_kernel(KernelType::SIGMOID, {tensorDesc}, {tensorDesc});
            auto config = sigmoid->suggest_config({tensorDesc}, {tensorDesc});
            
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            
            sigmoid->encode(encoder, {tempBuffer}, {outputBuffer}, config);
            
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }
        
        // Read results
        float* result = static_cast<float*>([outputBuffer contents]);
        
        std::cout << "Activation chain result (first 10 elements):" << std::endl;
        for (size_t i = 0; i < 10; ++i) {
            std::cout << "Input: " << data[i] 
                     << " -> Output: " << result[i] << std::endl;
        }
        
        library.shutdown();
    }
}

int main(int argc, const char * argv[]) {
    std::cout << "=== Metal Kernel Library Examples ===" << std::endl;
    
    std::cout << "\n1. Running GEMM example..." << std::endl;
    run_gemm_example();
    
    std::cout << "\n2. Running Conv2D example..." << std::endl;
    run_conv2d_example();
    
    std::cout << "\n3. Running activation chain example..." << std::endl;
    run_activation_chain_example();
    
    std::cout << "\nAll examples completed!" << std::endl;
    
    return 0;
}