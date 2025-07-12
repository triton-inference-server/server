// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "mps_engine.h"
#include "mps_memory_manager.h"
#include <iostream>
#include <cstring>

namespace triton { namespace backend { namespace metal_mps {

// Helper to calculate total elements from shape
static size_t
CalculateElementCount(const std::vector<int64_t>& shape)
{
  size_t count = 1;
  for (auto dim : shape) {
    count *= dim;
  }
  return count;
}

// Helper to get data type size
static size_t
GetDataTypeSize(TRITONSERVER_DataType dtype)
{
  switch (dtype) {
    case TRITONSERVER_TYPE_BOOL:
    case TRITONSERVER_TYPE_UINT8:
    case TRITONSERVER_TYPE_INT8:
      return 1;
    case TRITONSERVER_TYPE_UINT16:
    case TRITONSERVER_TYPE_INT16:
    case TRITONSERVER_TYPE_FP16:
      return 2;
    case TRITONSERVER_TYPE_UINT32:
    case TRITONSERVER_TYPE_INT32:
    case TRITONSERVER_TYPE_FP32:
      return 4;
    case TRITONSERVER_TYPE_UINT64:
    case TRITONSERVER_TYPE_INT64:
    case TRITONSERVER_TYPE_FP64:
      return 8;
    default:
      return 0;
  }
}

MPSEngine::MPSEngine()
    : initialized_(false), device_id_(0), device_(nullptr), 
      command_queue_(nullptr), execution_descriptor_(nullptr)
{
}

MPSEngine::~MPSEngine()
{
  @autoreleasepool {
    if (execution_descriptor_) {
      [(__bridge_transfer MPSGraphExecutionDescriptor*)execution_descriptor_ release];
    }
    if (command_queue_) {
      [(__bridge_transfer id<MTLCommandQueue>)command_queue_ release];
    }
    if (device_) {
      [(__bridge_transfer id<MTLDevice>)device_ release];
    }
  }
}

TRITONSERVER_Error*
MPSEngine::Initialize(int device_id)
{
  @autoreleasepool {
    device_id_ = device_id;
    
    // Get Metal device
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (devices.count == 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNAVAILABLE,
          "No Metal devices found");
    }
    
    if (device_id >= devices.count) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "Invalid device ID");
    }
    
    id<MTLDevice> device = devices[device_id];
    device_ = (__bridge_retained void*)device;
    
    // Create command queue
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "Failed to create Metal command queue");
    }
    command_queue_ = (__bridge_retained void*)queue;
    
    // Create memory manager
    memory_manager_ = std::make_unique<MPSMemoryManager>(device);
    
    // Create execution descriptor (reused across executions)
    MPSGraphExecutionDescriptor* execDesc = [[MPSGraphExecutionDescriptor alloc] init];
    execDesc.waitUntilCompleted = YES;  // Synchronous execution for simplicity
    execution_descriptor_ = (__bridge_retained void*)execDesc;
    
    initialized_ = true;
    
    // Log device info
    NSString* deviceName = [device name];
    std::cout << "MPS Engine initialized with device: " << [deviceName UTF8String] << std::endl;
    
    return nullptr;  // success
  }
}

std::string
MPSEngine::GetDeviceName() const
{
  if (!device_) {
    return "Unknown";
  }
  
  @autoreleasepool {
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
    return std::string([[device name] UTF8String]);
  }
}

uint64_t
MPSEngine::GetDeviceMemory() const
{
  if (!device_) {
    return 0;
  }
  
  @autoreleasepool {
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
    // Note: recommendedMaxWorkingSetSize is available on macOS 10.12+
    if ([device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
      return [device recommendedMaxWorkingSetSize];
    }
    return 0;
  }
}

TRITONSERVER_Error*
MPSEngine::Execute(
    MPSModel* model,
    const std::vector<MPSTensor>& inputs,
    std::vector<MPSTensor>& outputs)
{
  if (!initialized_) {
    RETURN_IF_ERROR(Initialize(device_id_));
  }
  
  @autoreleasepool {
    // Get the executable
    MPSGraphExecutable* executable = (__bridge MPSGraphExecutable*)model->GetExecutable();
    if (!executable) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "Model executable is null");
    }
    
    // Prepare input data
    std::unordered_map<std::string, void*> input_data;
    RETURN_IF_ERROR(PrepareInputs(model, inputs, input_data));
    
    // Create input feeds dictionary
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = 
        [NSMutableDictionary dictionary];
    
    for (const auto& input : inputs) {
      // Get the input tensor from the model
      MPSGraphTensor* inputTensor = (__bridge MPSGraphTensor*)model->GetInputTensor(input.name);
      if (!inputTensor) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            ("Input tensor not found: " + input.name).c_str());
      }
      
      // Create MPSGraphTensorData
      NSMutableArray<NSNumber*>* shape = [NSMutableArray array];
      for (auto dim : input.shape) {
        [shape addObject:@(dim)];
      }
      
      // Get Metal buffer from memory manager
      id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)memory_manager_->GetBuffer(input.data, input.byte_size);
      
      MPSGraphTensorData* tensorData = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                                               shape:shape
                                                                            dataType:MPSDataTypeFloat32];
      
      feeds[inputTensor] = tensorData;
    }
    
    // Create output buffers
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = 
        [NSMutableDictionary dictionary];
    
    for (const auto& outputSpec : model->GetOutputs()) {
      MPSGraphTensor* outputTensor = (__bridge MPSGraphTensor*)model->GetOutputTensor(outputSpec.name);
      if (!outputTensor) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("Output tensor not found: " + outputSpec.name).c_str());
      }
      
      // Calculate output size
      size_t elementCount = CalculateElementCount(outputSpec.shape);
      size_t byteSize = elementCount * GetDataTypeSize(outputSpec.datatype);
      
      // Allocate output buffer
      id<MTLBuffer> outputBuffer = (__bridge id<MTLBuffer>)memory_manager_->AllocateBuffer(byteSize);
      
      NSMutableArray<NSNumber*>* shape = [NSMutableArray array];
      for (auto dim : outputSpec.shape) {
        [shape addObject:@(dim)];
      }
      
      MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:outputBuffer
                                                                               shape:shape
                                                                            dataType:MPSDataTypeFloat32];
      
      results[outputTensor] = outputData;
    }
    
    // Execute the graph
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)command_queue_;
    MPSGraphExecutionDescriptor* execDesc = (__bridge MPSGraphExecutionDescriptor*)execution_descriptor_;
    
    NSError* error = nil;
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* executionResults = 
        [executable runAsyncWithMTLCommandQueue:queue
                                    inputsArray:@[feeds]
                                  resultsArray:@[results]
                            executionDescriptor:execDesc
                                          error:&error];
    
    if (error) {
      NSString* errorDesc = [error localizedDescription];
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          [errorDesc UTF8String]);
    }
    
    // Wait for completion (since we set waitUntilCompleted = YES)
    
    // Collect outputs
    std::unordered_map<std::string, void*> output_data;
    for (const auto& outputSpec : model->GetOutputs()) {
      MPSGraphTensor* outputTensor = (__bridge MPSGraphTensor*)model->GetOutputTensor(outputSpec.name);
      MPSGraphTensorData* outputTensorData = results[outputTensor];
      
      if (outputTensorData) {
        output_data[outputSpec.name] = (__bridge void*)outputTensorData.mpsndarray.buffer;
      }
    }
    
    RETURN_IF_ERROR(CollectOutputs(model, output_data, outputs));
    
    return nullptr;  // success
  }
}

TRITONSERVER_Error*
MPSEngine::PrepareInputs(
    MPSModel* model,
    const std::vector<MPSTensor>& inputs,
    std::unordered_map<std::string, void*>& input_data)
{
  for (const auto& input : inputs) {
    // Validate input exists in model
    bool found = false;
    for (const auto& modelInput : model->GetInputs()) {
      if (modelInput.name == input.name) {
        found = true;
        
        // Validate data type
        if (modelInput.datatype != input.datatype) {
          // Perform type conversion if needed
          size_t elementCount = CalculateElementCount(input.shape);
          void* convertedData = ConvertDataType(
              input.data, input.datatype, modelInput.datatype, elementCount);
          if (!convertedData) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                ("Data type conversion not supported: " + input.name).c_str());
          }
          input_data[input.name] = convertedData;
        } else {
          input_data[input.name] = input.data;
        }
        break;
      }
    }
    
    if (!found) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          ("Input not found in model: " + input.name).c_str());
    }
  }
  
  return nullptr;  // success
}

TRITONSERVER_Error*
MPSEngine::CollectOutputs(
    MPSModel* model,
    const std::unordered_map<std::string, void*>& output_data,
    std::vector<MPSTensor>& outputs)
{
  outputs.clear();
  
  for (const auto& outputSpec : model->GetOutputs()) {
    auto it = output_data.find(outputSpec.name);
    if (it == output_data.end()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("Output data not found: " + outputSpec.name).c_str());
    }
    
    MPSTensor output;
    output.name = outputSpec.name;
    output.datatype = outputSpec.datatype;
    output.shape = outputSpec.shape;
    
    // Calculate byte size
    size_t elementCount = CalculateElementCount(output.shape);
    output.byte_size = elementCount * GetDataTypeSize(output.datatype);
    
    // Allocate CPU memory for output
    output.data = malloc(output.byte_size);
    if (!output.data) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "Failed to allocate output memory");
    }
    
    // Copy data from Metal buffer
    @autoreleasepool {
      id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)it->second;
      memcpy(output.data, [buffer contents], output.byte_size);
    }
    
    outputs.push_back(output);
  }
  
  return nullptr;  // success
}

void*
MPSEngine::ConvertDataType(
    const void* src_data,
    TRITONSERVER_DataType src_type,
    TRITONSERVER_DataType dst_type,
    size_t element_count)
{
  // Simple conversion implementation
  // In production, this would handle all type conversions
  
  if (src_type == dst_type) {
    return const_cast<void*>(src_data);
  }
  
  // Example: FP64 to FP32 conversion
  if (src_type == TRITONSERVER_TYPE_FP64 && dst_type == TRITONSERVER_TYPE_FP32) {
    float* dst = (float*)malloc(element_count * sizeof(float));
    const double* src = (const double*)src_data;
    for (size_t i = 0; i < element_count; i++) {
      dst[i] = (float)src[i];
    }
    return dst;
  }
  
  // Add more conversions as needed
  
  return nullptr;  // Conversion not supported
}

}}}  // namespace triton::backend::metal_mps