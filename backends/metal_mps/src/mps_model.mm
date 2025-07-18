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

#include "mps_model.h"
#include "mps_graph_builder.h"
#include <fstream>
#include <sstream>

namespace triton { namespace backend { namespace metal_mps {

// Helper function to convert Triton data type to MPSDataType
static MPSDataType
ConvertToMPSDataType(TRITONSERVER_DataType dtype)
{
  switch (dtype) {
    case TRITONSERVER_TYPE_BOOL:
      return MPSDataTypeBool;
    case TRITONSERVER_TYPE_UINT8:
      return MPSDataTypeUInt8;
    case TRITONSERVER_TYPE_INT8:
      return MPSDataTypeInt8;
    case TRITONSERVER_TYPE_INT16:
      return MPSDataTypeInt16;
    case TRITONSERVER_TYPE_INT32:
      return MPSDataTypeInt32;
    case TRITONSERVER_TYPE_INT64:
      return MPSDataTypeInt64;
    case TRITONSERVER_TYPE_FP16:
      return MPSDataTypeFloat16;
    case TRITONSERVER_TYPE_FP32:
      return MPSDataTypeFloat32;
    default:
      return MPSDataTypeInvalid;
  }
}

// Helper to create shape descriptor
static MPSGraphTensorDescriptor*
CreateTensorDescriptor(const std::vector<int64_t>& shape, TRITONSERVER_DataType dtype)
{
  NSMutableArray<NSNumber*>* mpsShape = [NSMutableArray array];
  for (auto dim : shape) {
    [mpsShape addObject:@(dim)];
  }
  
  MPSDataType mpsDataType = ConvertToMPSDataType(dtype);
  if (mpsDataType == MPSDataTypeInvalid) {
    return nil;
  }
  
  return [MPSGraphTensorDescriptor descriptorWithDataType:mpsDataType
                                                     shape:mpsShape];
}

MPSModel::MPSModel()
    : name_("mps_model"), max_batch_size_(1), graph_(nullptr), executable_(nullptr)
{
}

MPSModel::~MPSModel()
{
  @autoreleasepool {
    if (executable_) {
      [(__bridge_transfer MPSGraphExecutable*)executable_ release];
    }
    if (graph_) {
      [(__bridge_transfer MPSGraph*)graph_ release];
    }
  }
}

TRITONSERVER_Error*
MPSModel::LoadFromFile(const std::string& path)
{
  // Determine file format
  std::string extension = path.substr(path.find_last_of(".") + 1);
  
  if (extension == "onnx") {
    return LoadONNXModel(path);
  } else if (extension == "mps") {
    // For now, .mps format is the same as ONNX
    // In the future, this could be a custom serialized format
    return LoadONNXModel(path);
  } else {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "Unsupported model format. Only .onnx and .mps are supported");
  }
}

TRITONSERVER_Error*
MPSModel::LoadONNXModel(const std::string& path)
{
  @autoreleasepool {
    // For this implementation, we'll create a simple example model
    // In a real implementation, you would parse the ONNX file
    // and build the graph accordingly
    
    // Create MPS graph
    MPSGraph* mpsGraph = [[MPSGraph alloc] init];
    graph_ = (__bridge_retained void*)mpsGraph;
    
    // Example: Create a simple model with Conv2D -> ReLU -> MaxPool
    // This is a placeholder - real implementation would parse ONNX
    
    // Define input
    MPSIOSpec inputSpec;
    inputSpec.name = "input";
    inputSpec.datatype = TRITONSERVER_TYPE_FP32;
    inputSpec.shape = {1, 3, 224, 224};  // NCHW format
    inputSpec.is_shape_tensor = false;
    inputs_.push_back(inputSpec);
    
    // Create input placeholder
    MPSGraphTensorDescriptor* inputDesc = CreateTensorDescriptor(inputSpec.shape, inputSpec.datatype);
    MPSGraphTensor* inputTensor = [mpsGraph placeholderWithDescriptor:inputDesc
                                                                  name:@"input"];
    input_tensors_[inputSpec.name] = (__bridge_retained void*)inputTensor;
    
    // Example: Add a convolution layer
    // Weight shape: [output_channels, input_channels, height, width]
    std::vector<int64_t> weightShape = {64, 3, 7, 7};
    MPSGraphTensorDescriptor* weightDesc = CreateTensorDescriptor(weightShape, TRITONSERVER_TYPE_FP32);
    
    // Create weight data (normally loaded from model file)
    size_t weightSize = 64 * 3 * 7 * 7;
    NSMutableData* weightData = [NSMutableData dataWithLength:weightSize * sizeof(float)];
    float* weightPtr = (float*)weightData.mutableBytes;
    // Initialize with small random values
    for (size_t i = 0; i < weightSize; i++) {
      weightPtr[i] = (float)(rand() % 100) / 1000.0f - 0.05f;
    }
    
    MPSGraphTensor* weightTensor = [mpsGraph constantWithData:weightData
                                                        shape:@[@64, @3, @7, @7]
                                                     dataType:MPSDataTypeFloat32];
    
    // Create bias
    NSMutableData* biasData = [NSMutableData dataWithLength:64 * sizeof(float)];
    float* biasPtr = (float*)biasData.mutableBytes;
    for (int i = 0; i < 64; i++) {
      biasPtr[i] = 0.0f;
    }
    MPSGraphTensor* biasTensor = [mpsGraph constantWithData:biasData
                                                      shape:@[@64]
                                                   dataType:MPSDataTypeFloat32];
    
    // Create convolution descriptor
    MPSGraphConvolution2DOpDescriptor* convDesc = [[MPSGraphConvolution2DOpDescriptor alloc] init];
    convDesc.strideInX = 2;
    convDesc.strideInY = 2;
    convDesc.paddingLeft = 3;
    convDesc.paddingRight = 3;
    convDesc.paddingTop = 3;
    convDesc.paddingBottom = 3;
    convDesc.dataLayout = MPSGraphTensorNamedDataLayoutNCHW;
    convDesc.weightsLayout = MPSGraphTensorNamedDataLayoutOIHW;
    
    // Apply convolution
    MPSGraphTensor* convOutput = [mpsGraph convolution2DWithSourceTensor:inputTensor
                                                           weightsTensor:weightTensor
                                                              descriptor:convDesc
                                                                    name:@"conv1"];
    
    // Add bias
    MPSGraphTensor* convWithBias = [mpsGraph additionWithPrimaryTensor:convOutput
                                                       secondaryTensor:biasTensor
                                                                  name:@"conv1_bias"];
    
    // Apply ReLU
    MPSGraphTensor* reluOutput = [mpsGraph reLUWithTensor:convWithBias
                                                      name:@"relu1"];
    
    // Max pooling
    MPSGraphPooling2DOpDescriptor* poolDesc = [[MPSGraphPooling2DOpDescriptor alloc] init];
    poolDesc.kernelWidth = 3;
    poolDesc.kernelHeight = 3;
    poolDesc.strideInX = 2;
    poolDesc.strideInY = 2;
    poolDesc.paddingLeft = 1;
    poolDesc.paddingRight = 1;
    poolDesc.paddingTop = 1;
    poolDesc.paddingBottom = 1;
    poolDesc.dataLayout = MPSGraphTensorNamedDataLayoutNCHW;
    
    MPSGraphTensor* poolOutput = [mpsGraph maxPooling2DWithSourceTensor:reluOutput
                                                             descriptor:poolDesc
                                                                   name:@"maxpool1"];
    
    // Define output
    MPSIOSpec outputSpec;
    outputSpec.name = "output";
    outputSpec.datatype = TRITONSERVER_TYPE_FP32;
    // Calculate output shape (simplified)
    outputSpec.shape = {1, 64, 56, 56};
    outputSpec.is_shape_tensor = false;
    outputs_.push_back(outputSpec);
    
    output_tensors_[outputSpec.name] = (__bridge_retained void*)poolOutput;
    
    // Compile the graph
    MPSGraphDevice* device = [MPSGraphDevice deviceWithMTLDevice:MTLCreateSystemDefaultDevice()];
    
    MPSGraphCompilationDescriptor* compilationDesc = [[MPSGraphCompilationDescriptor alloc] init];
    compilationDesc.optimizationLevel = MPSGraphOptimizationLevel1;
    
    NSError* error = nil;
    MPSGraphExecutable* exec = [mpsGraph compileWithDevice:device
                                         inputTensors:@{@"input": inputTensor}
                                       targetTensors:@{@"output": poolOutput}
                                        targetOperations:nil
                                compilationDescriptor:compilationDesc
                                                error:&error];
    
    if (error) {
      NSString* errorDesc = [error localizedDescription];
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          [errorDesc UTF8String]);
    }
    
    executable_ = (__bridge_retained void*)exec;
    
    return nullptr;  // success
  }
}

TRITONSERVER_Error*
MPSModel::BuildMPSGraph()
{
  // This would be called by LoadONNXModel after parsing
  // For now, it's integrated into LoadONNXModel
  return nullptr;
}

void*
MPSModel::GetInputTensor(const std::string& name) const
{
  auto it = input_tensors_.find(name);
  if (it != input_tensors_.end()) {
    return it->second;
  }
  return nullptr;
}

void*
MPSModel::GetOutputTensor(const std::string& name) const
{
  auto it = output_tensors_.find(name);
  if (it != output_tensors_.end()) {
    return it->second;
  }
  return nullptr;
}

}}}  // namespace triton::backend::metal_mps