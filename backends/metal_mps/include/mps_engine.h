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

#include <memory>
#include <vector>
#include <unordered_map>

#include "triton/core/tritonserver.h"
#include "mps_model.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#else
// Forward declarations for C++ files
typedef struct id<MTLDevice> MTLDevice;
typedef struct id<MTLCommandQueue> MTLCommandQueue;
typedef struct id<MTLBuffer> MTLBuffer;
typedef struct MPSGraphExecutionDescriptor MPSGraphExecutionDescriptor;
#endif

namespace triton { namespace backend { namespace metal_mps {

// Forward declaration
class MPSMemoryManager;

// MPS Execution Engine - handles model execution
class MPSEngine {
 public:
  MPSEngine();
  ~MPSEngine();

  // Initialize the engine
  TRITONSERVER_Error* Initialize(int device_id = 0);

  // Execute a model with given inputs
  TRITONSERVER_Error* Execute(
      MPSModel* model,
      const std::vector<MPSTensor>& inputs,
      std::vector<MPSTensor>& outputs);

  // Get device information
  std::string GetDeviceName() const;
  uint64_t GetDeviceMemory() const;

 private:
  // Prepare input tensors for execution
  TRITONSERVER_Error* PrepareInputs(
      MPSModel* model,
      const std::vector<MPSTensor>& inputs,
      std::unordered_map<std::string, void*>& input_data);

  // Collect output tensors after execution
  TRITONSERVER_Error* CollectOutputs(
      MPSModel* model,
      const std::unordered_map<std::string, void*>& output_data,
      std::vector<MPSTensor>& outputs);

  // Convert data types between Triton and MPS
  void* ConvertDataType(
      const void* src_data,
      TRITONSERVER_DataType src_type,
      TRITONSERVER_DataType dst_type,
      size_t element_count);

  bool initialized_;
  int device_id_;

  // Metal objects (stored as void* to avoid Obj-C in header)
  void* device_;  // id<MTLDevice>
  void* command_queue_;  // id<MTLCommandQueue>
  
  // Memory manager
  std::unique_ptr<MPSMemoryManager> memory_manager_;

  // Execution descriptor (reused across executions)
  void* execution_descriptor_;  // MPSGraphExecutionDescriptor*
};

}}}  // namespace triton::backend::metal_mps