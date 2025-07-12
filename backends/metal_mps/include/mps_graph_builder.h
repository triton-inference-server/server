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

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

#include "triton/core/tritonserver.h"

#ifdef __OBJC__
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#else
// Forward declarations for C++ files
typedef struct MPSGraph MPSGraph;
typedef struct MPSGraphTensor MPSGraphTensor;
#endif

namespace triton { namespace backend { namespace metal_mps {

// Operation types supported
enum class MPSOpType {
  // Tensor operations
  ADD,
  SUB,
  MUL,
  DIV,
  
  // Neural network layers
  CONV2D,
  MAXPOOL2D,
  AVGPOOL2D,
  BATCHNORM,
  
  // Activation functions
  RELU,
  SIGMOID,
  TANH,
  SOFTMAX,
  
  // Matrix operations
  MATMUL,
  GEMM,
  
  // Shape operations
  RESHAPE,
  TRANSPOSE,
  CONCAT,
  SPLIT,
  
  // Reduction operations
  REDUCE_MEAN,
  REDUCE_SUM,
  REDUCE_MAX,
  REDUCE_MIN
};

// Operation attributes
struct MPSOpAttributes {
  // Convolution attributes
  std::vector<int> strides;
  std::vector<int> padding;
  std::vector<int> dilations;
  int groups = 1;
  
  // Pooling attributes
  std::vector<int> kernel_size;
  
  // Transpose attributes
  std::vector<int> perm;
  
  // Reshape attributes
  std::vector<int64_t> shape;
  
  // Reduce attributes
  std::vector<int> axes;
  bool keepdims = false;
  
  // Activation attributes
  float alpha = 0.0f;
  float beta = 0.0f;
};

// Graph node representation
struct MPSGraphNode {
  std::string name;
  MPSOpType op_type;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  MPSOpAttributes attributes;
  void* mps_operation;  // MPSGraphOperation* or MPSGraphTensor*
};

// MPS Graph Builder - constructs MPS graphs from model definitions
class MPSGraphBuilder {
 public:
  MPSGraphBuilder();
  ~MPSGraphBuilder();

  // Initialize with a new graph
  void BeginGraph();
  
  // Add operations
  TRITONSERVER_Error* AddInput(
      const std::string& name,
      const std::vector<int64_t>& shape,
      TRITONSERVER_DataType datatype);
  
  TRITONSERVER_Error* AddConstant(
      const std::string& name,
      const std::vector<int64_t>& shape,
      TRITONSERVER_DataType datatype,
      const void* data);
  
  TRITONSERVER_Error* AddOperation(
      const std::string& name,
      MPSOpType op_type,
      const std::vector<std::string>& inputs,
      const std::vector<std::string>& outputs,
      const MPSOpAttributes& attributes);
  
  // Finalize and get the graph
  TRITONSERVER_Error* FinalizeGraph(
      void** graph,  // MPSGraph**
      std::unordered_map<std::string, void*>& tensors);  // name -> MPSGraphTensor*

  // Graph optimization
  TRITONSERVER_Error* OptimizeGraph();

 private:
  // Helper methods for creating specific operations
  void* CreateConvolution(
      void* input_tensor,
      void* weight_tensor,
      void* bias_tensor,
      const MPSOpAttributes& attrs);
  
  void* CreatePooling(
      void* input_tensor,
      MPSOpType pool_type,
      const MPSOpAttributes& attrs);
  
  void* CreateActivation(
      void* input_tensor,
      MPSOpType activation_type,
      const MPSOpAttributes& attrs);
  
  void* CreateMatMul(
      void* a_tensor,
      void* b_tensor,
      const MPSOpAttributes& attrs);
  
  void* CreateElementwise(
      void* a_tensor,
      void* b_tensor,
      MPSOpType op_type);
  
  void* CreateReshape(
      void* input_tensor,
      const std::vector<int64_t>& shape);
  
  void* CreateTranspose(
      void* input_tensor,
      const std::vector<int>& perm);
  
  void* CreateReduce(
      void* input_tensor,
      MPSOpType reduce_type,
      const MPSOpAttributes& attrs);

  // Graph state
  void* graph_;  // MPSGraph*
  std::unordered_map<std::string, void*> tensor_map_;  // name -> MPSGraphTensor*
  std::vector<MPSGraphNode> nodes_;
  bool finalized_;
};

}}}  // namespace triton::backend::metal_mps