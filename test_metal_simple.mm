// Simple Metal GPU test
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <iostream>
#import <vector>

void test_metal_device() {
    std::cout << "Testing Metal GPU..." << std::endl;
    
    // Get default Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cout << "✗ No Metal device found!" << std::endl;
        return;
    }
    
    std::cout << "✓ Metal device found: " << [device.name UTF8String] << std::endl;
    std::cout << "  Unified memory: " << (device.hasUnifiedMemory ? "Yes" : "No") << std::endl;
    std::cout << "  Max buffer size: " << (device.maxBufferLength / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    
    // Test simple compute
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    // Create buffers
    const size_t arrayLength = 1024;
    const size_t bufferSize = arrayLength * sizeof(float);
    
    id<MTLBuffer> bufferA = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferC = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    
    // Initialize data
    float* a = (float*)bufferA.contents;
    float* b = (float*)bufferB.contents;
    
    for (size_t i = 0; i < arrayLength; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    
    // Simple compute shader for vector addition
    NSString* kernelSource = @"#include <metal_stdlib>\n"
                            "using namespace metal;\n"
                            "kernel void add_vectors(device float* a [[buffer(0)]],\n"
                            "                       device float* b [[buffer(1)]],\n"
                            "                       device float* c [[buffer(2)]],\n"
                            "                       uint id [[thread_position_in_grid]]) {\n"
                            "    c[id] = a[id] + b[id];\n"
                            "}";
    
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:kernelSource options:nil error:&error];
    if (error) {
        std::cout << "✗ Failed to create library: " << [[error localizedDescription] UTF8String] << std::endl;
        return;
    }
    
    id<MTLFunction> addFunction = [library newFunctionWithName:@"add_vectors"];
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:addFunction error:&error];
    
    if (error) {
        std::cout << "✗ Failed to create pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
        return;
    }
    
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipelineState];
    [encoder setBuffer:bufferA offset:0 atIndex:0];
    [encoder setBuffer:bufferB offset:0 atIndex:1];
    [encoder setBuffer:bufferC offset:0 atIndex:2];
    
    MTLSize gridSize = MTLSizeMake(arrayLength, 1, 1);
    NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > arrayLength) {
        threadGroupSize = arrayLength;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Check results
    float* c = (float*)bufferC.contents;
    std::cout << "  Vector addition result: c[0] = " << c[0] << " (expected: 3.0)" << std::endl;
    
    // Test matrix multiplication with MPS
    std::cout << "\nTesting Metal matrix multiplication..." << std::endl;
    
    const size_t M = 128, N = 128, K = 128;
    
    id<MTLBuffer> matrixA = [device newBufferWithLength:M*K*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> matrixB = [device newBufferWithLength:K*N*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> matrixC = [device newBufferWithLength:M*N*sizeof(float) options:MTLResourceStorageModeShared];
    
    // Initialize matrices
    float* matA = (float*)matrixA.contents;
    float* matB = (float*)matrixB.contents;
    
    for (size_t i = 0; i < M*K; i++) matA[i] = 1.0f;
    for (size_t i = 0; i < K*N; i++) matB[i] = 2.0f;
    
    // Create matrix descriptors
    MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                        columns:K
                                                                       rowBytes:K*sizeof(float)
                                                                       dataType:MPSDataTypeFloat32];
    
    MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                        columns:N
                                                                       rowBytes:N*sizeof(float)
                                                                       dataType:MPSDataTypeFloat32];
    
    MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                        columns:N
                                                                       rowBytes:N*sizeof(float)
                                                                       dataType:MPSDataTypeFloat32];
    
    MPSMatrix* mpsMatrixA = [[MPSMatrix alloc] initWithBuffer:matrixA descriptor:descA];
    MPSMatrix* mpsMatrixB = [[MPSMatrix alloc] initWithBuffer:matrixB descriptor:descB];
    MPSMatrix* mpsMatrixC = [[MPSMatrix alloc] initWithBuffer:matrixC descriptor:descC];
    
    // Create GEMM operation
    MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                         transposeLeft:NO
                                                                        transposeRight:NO
                                                                            resultRows:M
                                                                         resultColumns:N
                                                                       interiorColumns:K
                                                                                 alpha:1.0
                                                                                  beta:0.0];
    
    // Execute
    id<MTLCommandBuffer> gemmCommandBuffer = [commandQueue commandBuffer];
    [matmul encodeToCommandBuffer:gemmCommandBuffer
                       leftMatrix:mpsMatrixA
                      rightMatrix:mpsMatrixB
                     resultMatrix:mpsMatrixC];
    
    [gemmCommandBuffer commit];
    [gemmCommandBuffer waitUntilCompleted];
    
    float* matC = (float*)matrixC.contents;
    std::cout << "  Matrix multiplication result: C[0] = " << matC[0] << " (expected: " << K*2.0f << ")" << std::endl;
    
    std::cout << "\n✓ Metal GPU test completed successfully!" << std::endl;
}

int main() {
    @autoreleasepool {
        std::cout << "==================================" << std::endl;
        std::cout << "Metal GPU Test" << std::endl;
        std::cout << "==================================" << std::endl;
        
        test_metal_device();
        
        std::cout << "==================================" << std::endl;
    }
    return 0;
}