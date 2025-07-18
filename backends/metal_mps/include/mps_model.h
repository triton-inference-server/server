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
#include <memory>
#include <unordered_map>

#include "triton/core/tritonserver.h"

#ifdef __OBJC__
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#else
// Forward declarations for C++ files
typedef struct MPSGraph MPSGraph;
typedef struct MPSGraphTensor MPSGraphTensor;
typedef struct MPSGraphOperation MPSGraphOperation;
typedef struct MPSGraphExecutable MPSGraphExecutable;
#endif

namespace triton { namespace backend { namespace metal_mps {

// Tensor information structure
struct MPSTensor {
  std::string name;
  TRITONSERVER_DataType datatype;
  std::vector<int64_t> shape;
  void* data;
  uint64_t byte_size;
};

// Model input/output specification
struct MPSIOSpec {
  std::string name;
  TRITONSERVER_DataType datatype;
  std::vector<int64_t> shape;
  bool is_shape_tensor;
};

// MPS Model class - represents a loaded model
class MPSModel {
 public:
  MPSModel();
  ~MPSModel();

  // Load model from file
  TRITONSERVER_Error* LoadFromFile(const std::string& path);

  // Get model information
  const std::vector<MPSIOSpec>& GetInputs() const { return inputs_; }
  const std::vector<MPSIOSpec>& GetOutputs() const { return outputs_; }
  const std::string& GetName() const { return name_; }
  int64_t GetMaxBatchSize() const { return max_batch_size_; }

  // Get the compiled executable (for execution)
  void* GetExecutable() const { return executable_; }

  // Get input/output tensors
  void* GetInputTensor(const std::string& name) const;
  void* GetOutputTensor(const std::string& name) const;

 private:
  // Load ONNX model
  TRITONSERVER_Error* LoadONNXModel(const std::string& path);

  // Build MPS graph from loaded model
  TRITONSERVER_Error* BuildMPSGraph();

  std::string name_;
  int64_t max_batch_size_;
  std::vector<MPSIOSpec> inputs_;
  std::vector<MPSIOSpec> outputs_;

  // MPS graph objects (stored as void* to avoid Obj-C in header)
  void* graph_;  // MPSGraph*
  void* executable_;  // MPSGraphExecutable*
  
  // Maps for quick lookup
  std::unordered_map<std::string, void*> input_tensors_;  // name -> MPSGraphTensor*
  std::unordered_map<std::string, void*> output_tensors_;  // name -> MPSGraphTensor*
  
  // Model weights and parameters
  std::unordered_map<std::string, MPSTensor> parameters_;
};

}}}  // namespace triton::backend::metal_mps