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
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "mps_graph_builder.h"
#include <iostream>

namespace triton { namespace backend { namespace metal_mps {

// Helper to convert data type
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

// Helper to create shape array
static NSArray<NSNumber*>*
CreateShapeArray(const std::vector<int64_t>& shape)
{
  NSMutableArray<NSNumber*>* array = [NSMutableArray array];
  for (auto dim : shape) {
    [array addObject:@(dim)];
  }
  return array;
}

MPSGraphBuilder::MPSGraphBuilder()
    : graph_(nullptr), finalized_(false)
{
}

MPSGraphBuilder::~MPSGraphBuilder()
{
  @autoreleasepool {
    if (graph_) {
      [(__bridge_transfer MPSGraph*)graph_ release];
    }
  }
}

void
MPSGraphBuilder::BeginGraph()
{
  @autoreleasepool {
    if (graph_) {
      [(__bridge_transfer MPSGraph*)graph_ release];
    }
    
    MPSGraph* mpsGraph = [[MPSGraph alloc] init];
    graph_ = (__bridge_retained void*)mpsGraph;
    
    tensor_map_.clear();
    nodes_.clear();
    finalized_ = false;
  }
}

TRITONSERVER_Error*
MPSGraphBuilder::AddInput(
    const std::string& name,
    const std::vector<int64_t>& shape,
    TRITONSERVER_DataType datatype)
{
  if (!graph_) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Graph not initialized. Call BeginGraph() first");
  }
  
  @autoreleasepool {
    MPSGraph* mpsGraph = (__bridge MPSGraph*)graph_;
    
    MPSDataType mpsDataType = ConvertToMPSDataType(datatype);
    if (mpsDataType == MPSDataTypeInvalid) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "Unsupported data type for input");
    }
    
    NSArray<NSNumber*>* mpsShape = CreateShapeArray(shape);
    MPSGraphTensor* tensor = [mpsGraph placeholderWithShape:mpsShape
                                                    dataType:mpsDataType
                                                        name:[NSString stringWithUTF8String:name.c_str()]];
    
    tensor_map_[name] = (__bridge_retained void*)tensor;
    
    // Record node
    MPSGraphNode node;
    node.name = name;
    node.op_type = MPSOpType::ADD;  // Placeholder op type
    node.outputs.push_back(name);
    node.mps_operation = (__bridge_retained void*)tensor;
    nodes_.push_back(node);
    
    return nullptr;  // success
  }
}

TRITONSERVER_Error*
MPSGraphBuilder::AddConstant(
    const std::string& name,
    const std::vector<int64_t>& shape,
    TRITONSERVER_DataType datatype,
    const void* data)
{
  if (!graph_) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Graph not initialized. Call BeginGraph() first");
  }
  
  @autoreleasepool {
    MPSGraph* mpsGraph = (__bridge MPSGraph*)graph_;
    
    MPSDataType mpsDataType = ConvertToMPSDataType(datatype);
    if (mpsDataType == MPSDataTypeInvalid) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "Unsupported data type for constant");
    }
    
    // Calculate data size
    size_t elementCount = 1;
    for (auto dim : shape) {
      elementCount *= dim;
    }
    
    size_t dataSize = 0;
    switch (datatype) {
      case TRITONSERVER_TYPE_BOOL:
      case TRITONSERVER_TYPE_UINT8:
      case TRITONSERVER_TYPE_INT8:
        dataSize = elementCount;
        break;
      case TRITONSERVER_TYPE_UINT16:
      case TRITONSERVER_TYPE_INT16:
      case TRITONSERVER_TYPE_FP16:
        dataSize = elementCount * 2;
        break;
      case TRITONSERVER_TYPE_UINT32:
      case TRITONSERVER_TYPE_INT32:
      case TRITONSERVER_TYPE_FP32:
        dataSize = elementCount * 4;
        break;
      case TRITONSERVER_TYPE_UINT64:
      case TRITONSERVER_TYPE_INT64:
      case TRITONSERVER_TYPE_FP64:
        dataSize = elementCount * 8;
        break;
      default:
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            "Unsupported data type size");
    }
    
    NSData* nsData = [NSData dataWithBytes:data length:dataSize];
    NSArray<NSNumber*>* mpsShape = CreateShapeArray(shape);
    
    MPSGraphTensor* tensor = [mpsGraph constantWithData:nsData
                                                  shape:mpsShape
                                               dataType:mpsDataType];
    
    tensor_map_[name] = (__bridge_retained void*)tensor;
    
    return nullptr;  // success
  }
}

TRITONSERVER_Error*
MPSGraphBuilder::AddOperation(
    const std::string& name,
    MPSOpType op_type,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const MPSOpAttributes& attributes)
{
  if (!graph_) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Graph not initialized. Call BeginGraph() first");
  }
  
  @autoreleasepool {
    MPSGraph* mpsGraph = (__bridge MPSGraph*)graph_;
    
    // Get input tensors
    std::vector<MPSGraphTensor*> inputTensors;
    for (const auto& inputName : inputs) {
      auto it = tensor_map_.find(inputName);
      if (it == tensor_map_.end()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            ("Input tensor not found: " + inputName).c_str());
      }
      inputTensors.push_back((__bridge MPSGraphTensor*)it->second);
    }
    
    MPSGraphTensor* outputTensor = nil;
    
    // Create operation based on type
    switch (op_type) {
      case MPSOpType::ADD:
        if (inputTensors.size() != 2) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              "ADD operation requires exactly 2 inputs");
        }
        outputTensor = [mpsGraph additionWithPrimaryTensor:inputTensors[0]
                                           secondaryTensor:inputTensors[1]
                                                      name:[NSString stringWithUTF8String:name.c_str()]];
        break;
        
      case MPSOpType::RELU:
        if (inputTensors.size() != 1) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              "RELU operation requires exactly 1 input");
        }
        outputTensor = [mpsGraph reLUWithTensor:inputTensors[0]
                                           name:[NSString stringWithUTF8String:name.c_str()]];
        break;
        
      case MPSOpType::CONV2D:
        outputTensor = (__bridge MPSGraphTensor*)CreateConvolution(
            (__bridge void*)inputTensors[0],
            inputTensors.size() > 1 ? (__bridge void*)inputTensors[1] : nullptr,
            inputTensors.size() > 2 ? (__bridge void*)inputTensors[2] : nullptr,
            attributes);
        break;
        
      case MPSOpType::MAXPOOL2D:
        outputTensor = (__bridge MPSGraphTensor*)CreatePooling(
            (__bridge void*)inputTensors[0],
            MPSOpType::MAXPOOL2D,
            attributes);
        break;
        
      case MPSOpType::MATMUL:
        if (inputTensors.size() != 2) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              "MATMUL operation requires exactly 2 inputs");
        }
        outputTensor = (__bridge MPSGraphTensor*)CreateMatMul(
            (__bridge void*)inputTensors[0],
            (__bridge void*)inputTensors[1],
            attributes);
        break;
        
      case MPSOpType::RESHAPE:
        if (inputTensors.size() != 1) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              "RESHAPE operation requires exactly 1 input");
        }
        outputTensor = (__bridge MPSGraphTensor*)CreateReshape(
            (__bridge void*)inputTensors[0],
            attributes.shape);
        break;
        
      case MPSOpType::TRANSPOSE:
        if (inputTensors.size() != 1) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              "TRANSPOSE operation requires exactly 1 input");
        }
        outputTensor = (__bridge MPSGraphTensor*)CreateTranspose(
            (__bridge void*)inputTensors[0],
            attributes.perm);
        break;
        
      default:
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            "Unsupported operation type");
    }
    
    if (!outputTensor) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "Failed to create operation");
    }
    
    // Store output tensor
    if (!outputs.empty()) {
      tensor_map_[outputs[0]] = (__bridge_retained void*)outputTensor;
    }
    
    // Record node
    MPSGraphNode node;
    node.name = name;
    node.op_type = op_type;
    node.inputs = inputs;
    node.outputs = outputs;
    node.attributes = attributes;
    node.mps_operation = (__bridge_retained void*)outputTensor;
    nodes_.push_back(node);
    
    return nullptr;  // success
  }
}

void*
MPSGraphBuilder::CreateConvolution(
    void* input_tensor,
    void* weight_tensor,
    void* bias_tensor,
    const MPSOpAttributes& attrs)
{
  @autoreleasepool {
    MPSGraph* mpsGraph = (__bridge MPSGraph*)graph_;
    MPSGraphTensor* input = (__bridge MPSGraphTensor*)input_tensor;
    MPSGraphTensor* weights = (__bridge MPSGraphTensor*)weight_tensor;
    
    MPSGraphConvolution2DOpDescriptor* desc = [[MPSGraphConvolution2DOpDescriptor alloc] init];
    
    if (attrs.strides.size() >= 2) {
      desc.strideInX = attrs.strides[1];
      desc.strideInY = attrs.strides[0];
    }
    
    if (attrs.padding.size() >= 4) {
      desc.paddingLeft = attrs.padding[1];
      desc.paddingRight = attrs.padding[3];
      desc.paddingTop = attrs.padding[0];
      desc.paddingBottom = attrs.padding[2];
    }
    
    if (attrs.dilations.size() >= 2) {
      desc.dilationRateInX = attrs.dilations[1];
      desc.dilationRateInY = attrs.dilations[0];
    }
    
    desc.groups = attrs.groups;
    desc.dataLayout = MPSGraphTensorNamedDataLayoutNCHW;
    desc.weightsLayout = MPSGraphTensorNamedDataLayoutOIHW;
    
    MPSGraphTensor* conv = [mpsGraph convolution2DWithSourceTensor:input
                                                     weightsTensor:weights
                                                        descriptor:desc
                                                              name:nil];
    
    if (bias_tensor) {
      MPSGraphTensor* bias = (__bridge MPSGraphTensor*)bias_tensor;
      conv = [mpsGraph additionWithPrimaryTensor:conv
                                secondaryTensor:bias
                                           name:nil];
    }
    
    return (__bridge_retained void*)conv;
  }
}

void*
MPSGraphBuilder::CreatePooling(
    void* input_tensor,
    MPSOpType pool_type,
    const MPSOpAttributes& attrs)
{
  @autoreleasepool {
    MPSGraph* mpsGraph = (__bridge MPSGraph*)graph_;
    MPSGraphTensor* input = (__bridge MPSGraphTensor*)input_tensor;
    
    MPSGraphPooling2DOpDescriptor* desc = [[MPSGraphPooling2DOpDescriptor alloc] init];
    
    if (attrs.kernel_size.size() >= 2) {
      desc.kernelWidth = attrs.kernel_size[1];
      desc.kernelHeight = attrs.kernel_size[0];
    }
    
    if (attrs.strides.size() >= 2) {
      desc.strideInX = attrs.strides[1];
      desc.strideInY = attrs.strides[0];
    }
    
    if (attrs.padding.size() >= 4) {
      desc.paddingLeft = attrs.padding[1];
      desc.paddingRight = attrs.padding[3];
      desc.paddingTop = attrs.padding[0];
      desc.paddingBottom = attrs.padding[2];
    }
    
    desc.dataLayout = MPSGraphTensorNamedDataLayoutNCHW;
    
    MPSGraphTensor* output = nil;
    if (pool_type == MPSOpType::MAXPOOL2D) {
      output = [mpsGraph maxPooling2DWithSourceTensor:input
                                           descriptor:desc
                                                 name:nil];
    } else if (pool_type == MPSOpType::AVGPOOL2D) {
      output = [mpsGraph avgPooling2DWithSourceTensor:input
                                           descriptor:desc
                                                 name:nil];
    }
    
    return (__bridge_retained void*)output;
  }
}

void*
MPSGraphBuilder::CreateMatMul(
    void* a_tensor,
    void* b_tensor,
    const MPSOpAttributes& attrs)
{
  @autoreleasepool {
    MPSGraph* mpsGraph = (__bridge MPSGraph*)graph_;
    MPSGraphTensor* a = (__bridge MPSGraphTensor*)a_tensor;
    MPSGraphTensor* b = (__bridge MPSGraphTensor*)b_tensor;
    
    MPSGraphTensor* output = [mpsGraph matrixMultiplicationWithPrimaryTensor:a
                                                            secondaryTensor:b
                                                                       name:nil];
    
    return (__bridge_retained void*)output;
  }
}

void*
MPSGraphBuilder::CreateReshape(
    void* input_tensor,
    const std::vector<int64_t>& shape)
{
  @autoreleasepool {
    MPSGraph* mpsGraph = (__bridge MPSGraph*)graph_;
    MPSGraphTensor* input = (__bridge MPSGraphTensor*)input_tensor;
    
    NSArray<NSNumber*>* newShape = CreateShapeArray(shape);
    MPSGraphTensor* output = [mpsGraph reshapeTensor:input
                                            withShape:newShape
                                                 name:nil];
    
    return (__bridge_retained void*)output;
  }
}

void*
MPSGraphBuilder::CreateTranspose(
    void* input_tensor,
    const std::vector<int>& perm)
{
  @autoreleasepool {
    MPSGraph* mpsGraph = (__bridge MPSGraph*)graph_;
    MPSGraphTensor* input = (__bridge MPSGraphTensor*)input_tensor;
    
    NSMutableArray<NSNumber*>* permArray = [NSMutableArray array];
    for (auto p : perm) {
      [permArray addObject:@(p)];
    }
    
    MPSGraphTensor* output = [mpsGraph transposeTensor:input
                                              dimension:permArray[0].intValue
                                          withDimension:permArray[1].intValue
                                                   name:nil];
    
    return (__bridge_retained void*)output;
  }
}

TRITONSERVER_Error*
MPSGraphBuilder::FinalizeGraph(
    void** graph,
    std::unordered_map<std::string, void*>& tensors)
{
  if (!graph_) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Graph not initialized");
  }
  
  *graph = graph_;
  tensors = tensor_map_;
  finalized_ = true;
  
  // Transfer ownership
  graph_ = nullptr;
  
  return nullptr;  // success
}

TRITONSERVER_Error*
MPSGraphBuilder::OptimizeGraph()
{
  // Graph optimization would be implemented here
  // MPS framework handles most optimizations automatically
  return nullptr;  // success
}

}}}  // namespace triton::backend::metal_mps