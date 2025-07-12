// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifdef TRITON_ENABLE_METAL

#include "metal_backend_unified_integration.h"
#include <numeric>

namespace triton { namespace backend { namespace metal {

//
// UnifiedMetalBackend Implementation
//

UnifiedMetalBackend::UnifiedMetalBackend(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model), unified_memory_initialized_(false)
{
}

UnifiedMetalBackend::~UnifiedMetalBackend()
{
  // Cleanup is handled by UnifiedMemoryOptimizer singleton
}

TRITONSERVER_Error*
UnifiedMetalBackend::InitializeUnifiedMemory(const UnifiedBackendConfig& config)
{
  config_ = config;
  
  // Initialize unified memory optimizer
  triton::core::UnifiedMemoryConfig um_config;
  um_config.enable_auto_placement = config.enable_unified_memory;
  um_config.enable_zero_copy = config.enable_unified_memory;
  um_config.enable_prefetching = true;
  um_config.enable_pressure_adaptation = true;
  um_config.enable_numa_optimization = true;
  
  auto status = triton::core::UnifiedMemoryOptimizer::Initialize(um_config);
  if (!status.IsOk()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("Failed to initialize unified memory optimizer: " + status.Message()).c_str());
  }
  
  if (config.enable_profiling) {
    triton::core::UnifiedMemoryOptimizer::EnableProfiling(true);
  }
  
  unified_memory_initialized_ = true;
  return nullptr;  // success
}

TRITONSERVER_Error*
UnifiedMetalBackend::CreateOptimizedInputTensor(
    const std::string& name,
    TRITONSERVER_DataType datatype,
    const std::vector<int64_t>& shape,
    std::unique_ptr<triton::core::MetalBuffer>& buffer,
    bool zero_copy_if_possible)
{
  if (!unified_memory_initialized_) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Unified memory not initialized");
  }
  
  size_t tensor_size = CalculateTensorSize(datatype, shape);
  auto pattern = GetTensorPattern(name);
  
  // For inputs, we often want CPU-dominant access
  if (pattern == triton::core::UnifiedMemoryPattern::UNKNOWN) {
    pattern = triton::core::UnifiedMemoryPattern::CPU_DOMINANT;
  }
  
  auto status = triton::core::UnifiedMemoryOptimizer::AllocateOptimized(
      buffer, tensor_size, pattern);
  
  if (!status.IsOk()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("Failed to create optimized input tensor: " + status.Message()).c_str());
  }
  
  return nullptr;  // success
}

TRITONSERVER_Error*
UnifiedMetalBackend::CreateOptimizedOutputTensor(
    const std::string& name,
    TRITONSERVER_DataType datatype,
    const std::vector<int64_t>& shape,
    std::unique_ptr<triton::core::MetalBuffer>& buffer)
{
  if (!unified_memory_initialized_) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Unified memory not initialized");
  }
  
  size_t tensor_size = CalculateTensorSize(datatype, shape);
  auto pattern = GetTensorPattern(name);
  
  // For outputs, default to GPU-dominant if not specified
  if (pattern == triton::core::UnifiedMemoryPattern::UNKNOWN) {
    pattern = triton::core::UnifiedMemoryPattern::GPU_DOMINANT;
  }
  
  auto status = triton::core::UnifiedMemoryOptimizer::AllocateOptimized(
      buffer, tensor_size, pattern);
  
  if (!status.IsOk()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("Failed to create optimized output tensor: " + status.Message()).c_str());
  }
  
  return nullptr;  // success
}

TRITONSERVER_Error*
UnifiedMetalBackend::GetZeroCopyInputTensor(
    TRITONBACKEND_Request* request,
    const std::string& name,
    std::unique_ptr<triton::core::ZeroCopyTensor>& tensor)
{
  const void* input_buffer;
  size_t input_byte_size;
  TRITONSERVER_MemoryType input_memory_type;
  int64_t input_memory_id;
  
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputByName(
      request, name.c_str(), &input_buffer, &input_byte_size,
      &input_memory_type, &input_memory_id));
  
  // Get shape information
  TRITONBACKEND_Input* input;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInput(
      request, name.c_str(), &input));
  
  TRITONSERVER_DataType datatype;
  const int64_t* shape_ptr;
  uint32_t dims_count;
  uint64_t byte_size;
  
  RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
      input, nullptr, &datatype, &shape_ptr, &dims_count, &byte_size, nullptr));
  
  std::vector<int64_t> shape(shape_ptr, shape_ptr + dims_count);
  
  // Check if we can use zero-copy
  if (utils::CanUseZeroCopy(input_byte_size, input_memory_type, config_)) {
    // Create zero-copy tensor
    auto status = triton::core::ZeroCopyTensor::CreateFromCPUMemory(
        tensor, const_cast<void*>(input_buffer), input_byte_size, shape, datatype);
    
    if (!status.IsOk()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("Failed to create zero-copy tensor: " + status.Message()).c_str());
    }
    
    // Record the eliminated transfer
    triton::core::TransferEliminationTracker::RecordEliminatedTransfer(
        input_byte_size, true);  // CPU to GPU transfer eliminated
  } else {
    // Fall back to regular copy
    std::unique_ptr<triton::core::MetalBuffer> buffer;
    RETURN_IF_ERROR(CreateOptimizedInputTensor(
        name, datatype, shape, buffer, false));
    
    // Copy data to buffer
    auto copy_status = buffer->CopyFromHost(input_buffer, input_byte_size);
    if (!copy_status.IsOk()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("Failed to copy input data: " + copy_status.Message()).c_str());
    }
    
    // Create tensor wrapper
    auto status = triton::core::ZeroCopyTensor::CreateFromGPUMemory(
        tensor, buffer->Data(), input_byte_size, shape, datatype);
    
    if (!status.IsOk()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("Failed to create tensor from GPU memory: " + status.Message()).c_str());
    }
  }
  
  return nullptr;  // success
}

TRITONSERVER_Error*
UnifiedMetalBackend::SetZeroCopyOutputTensor(
    TRITONBACKEND_Response* response,
    const std::string& name,
    std::unique_ptr<triton::core::ZeroCopyTensor> tensor)
{
  TRITONBACKEND_Output* output;
  RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
      response, &output, name.c_str(), tensor->DataType(),
      tensor->Shape().data(), tensor->Shape().size()));
  
  void* output_buffer;
  TRITONSERVER_MemoryType output_memory_type;
  int64_t output_memory_id;
  
  RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(
      output, &output_buffer, tensor->Size(),
      &output_memory_type, &output_memory_id));
  
  // Check if we can avoid copy
  if (tensor->IsCPU() && output_memory_type == TRITONSERVER_MEMORY_CPU) {
    // Both are CPU memory, check if we can use the same buffer
    if (output_buffer == tensor->Data()) {
      // Zero-copy achieved!
      triton::core::TransferEliminationTracker::RecordEliminatedTransfer(
          tensor->Size(), false);  // GPU to CPU transfer eliminated
      return nullptr;
    }
  }
  
  // Need to copy data
  if (tensor->IsGPU() && output_memory_type == TRITONSERVER_MEMORY_CPU) {
    // Copy from GPU to CPU
    auto* metal_buffer = tensor->GetMetalBuffer();
    if (metal_buffer) {
      auto status = metal_buffer->CopyToHost(output_buffer, tensor->Size());
      if (!status.IsOk()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("Failed to copy output data: " + status.Message()).c_str());
      }
    }
  } else {
    // Direct memory copy for CPU to CPU
    std::memcpy(output_buffer, tensor->Data(), tensor->Size());
  }
  
  return nullptr;  // success
}

TRITONSERVER_Error*
UnifiedMetalBackend::BatchAllocateTensors(
    const std::vector<std::string>& names,
    const std::vector<TRITONSERVER_DataType>& datatypes,
    const std::vector<std::vector<int64_t>>& shapes,
    std::vector<std::unique_ptr<triton::core::MetalBuffer>>& buffers)
{
  if (!unified_memory_initialized_) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Unified memory not initialized");
  }
  
  if (names.size() != datatypes.size() || names.size() != shapes.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Mismatched tensor specification sizes");
  }
  
  std::vector<size_t> sizes;
  sizes.reserve(names.size());
  
  // Calculate sizes and determine common pattern
  std::unordered_map<triton::core::UnifiedMemoryPattern, int> pattern_counts;
  for (size_t i = 0; i < names.size(); ++i) {
    sizes.push_back(CalculateTensorSize(datatypes[i], shapes[i]));
    auto pattern = GetTensorPattern(names[i]);
    pattern_counts[pattern]++;
  }
  
  // Use most common pattern for batch allocation
  triton::core::UnifiedMemoryPattern batch_pattern = 
      triton::core::UnifiedMemoryPattern::BALANCED;
  int max_count = 0;
  for (const auto& [pattern, count] : pattern_counts) {
    if (count > max_count) {
      max_count = count;
      batch_pattern = pattern;
    }
  }
  
  auto status = triton::core::UnifiedMemoryOptimizer::BatchAllocate(
      buffers, sizes, batch_pattern);
  
  if (!status.IsOk()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("Failed to batch allocate tensors: " + status.Message()).c_str());
  }
  
  return nullptr;  // success
}

void
UnifiedMetalBackend::GetMemoryStatistics(
    size_t& total_allocated,
    size_t& unified_memory_used,
    size_t& transfers_eliminated)
{
  if (!unified_memory_initialized_) {
    total_allocated = 0;
    unified_memory_used = 0;
    transfers_eliminated = 0;
    return;
  }
  
  std::unordered_map<triton::core::UnifiedMemoryPattern, size_t> pattern_distribution;
  triton::core::UnifiedMemoryOptimizer::GetMemoryStats(
      total_allocated, unified_memory_used, transfers_eliminated, pattern_distribution);
}

triton::core::UnifiedMemoryPattern
UnifiedMetalBackend::GetTensorPattern(const std::string& tensor_name)
{
  auto it = config_.tensor_patterns.find(tensor_name);
  if (it != config_.tensor_patterns.end()) {
    return it->second;
  }
  
  // Check for common patterns
  if (tensor_name.find("input") != std::string::npos) {
    return triton::core::UnifiedMemoryPattern::CPU_DOMINANT;
  } else if (tensor_name.find("output") != std::string::npos) {
    return triton::core::UnifiedMemoryPattern::GPU_DOMINANT;
  } else if (tensor_name.find("weight") != std::string::npos ||
             tensor_name.find("bias") != std::string::npos) {
    return triton::core::UnifiedMemoryPattern::GPU_DOMINANT;
  }
  
  return triton::core::UnifiedMemoryPattern::UNKNOWN;
}

size_t
UnifiedMetalBackend::CalculateTensorSize(
    TRITONSERVER_DataType datatype,
    const std::vector<int64_t>& shape)
{
  size_t element_size = 0;
  
  switch (datatype) {
    case TRITONSERVER_TYPE_BOOL:
    case TRITONSERVER_TYPE_UINT8:
    case TRITONSERVER_TYPE_INT8:
      element_size = 1;
      break;
    case TRITONSERVER_TYPE_UINT16:
    case TRITONSERVER_TYPE_INT16:
    case TRITONSERVER_TYPE_FP16:
    case TRITONSERVER_TYPE_BF16:
      element_size = 2;
      break;
    case TRITONSERVER_TYPE_UINT32:
    case TRITONSERVER_TYPE_INT32:
    case TRITONSERVER_TYPE_FP32:
      element_size = 4;
      break;
    case TRITONSERVER_TYPE_UINT64:
    case TRITONSERVER_TYPE_INT64:
    case TRITONSERVER_TYPE_FP64:
      element_size = 8;
      break;
    default:
      return 0;
  }
  
  size_t num_elements = 1;
  for (auto dim : shape) {
    num_elements *= static_cast<size_t>(dim);
  }
  
  return element_size * num_elements;
}

//
// Example PyTorch Backend Implementation
//

TRITONSERVER_Error*
UnifiedPyTorchBackend::Execute(
    TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  std::vector<TRITONBACKEND_Request*> request_vec(requests, requests + request_count);
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  
  // Create responses
  for (uint32_t i = 0; i < request_count; ++i) {
    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, requests[i]));
    responses.push_back(response);
  }
  
  // Process batch with optimizations
  auto err = ProcessBatchOptimized(request_vec, responses);
  
  // Send responses
  for (auto* response : responses) {
    if (err != nullptr) {
      TRITONBACKEND_ResponseSetError(response, err);
    }
    TRITONBACKEND_ResponseSend(
        response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err);
  }
  
  return err;
}

TRITONSERVER_Error*
UnifiedPyTorchBackend::ProcessBatchOptimized(
    const std::vector<TRITONBACKEND_Request*>& requests,
    std::vector<TRITONBACKEND_Response*>& responses)
{
  // Example: Process image classification batch
  const std::string input_name = "input";
  const std::string output_name = "output";
  
  // Collect all input tensors with zero-copy when possible
  std::vector<std::unique_ptr<triton::core::ZeroCopyTensor>> input_tensors;
  
  for (auto* request : requests) {
    std::unique_ptr<triton::core::ZeroCopyTensor> tensor;
    RETURN_IF_ERROR(GetZeroCopyInputTensor(request, input_name, tensor));
    input_tensors.push_back(std::move(tensor));
  }
  
  // Batch allocate output tensors
  std::vector<std::string> output_names(requests.size(), output_name);
  std::vector<TRITONSERVER_DataType> output_types(requests.size(), TRITONSERVER_TYPE_FP32);
  std::vector<std::vector<int64_t>> output_shapes;
  
  // Assume output shape is [batch_size, num_classes]
  const int64_t num_classes = 1000;
  for (size_t i = 0; i < requests.size(); ++i) {
    output_shapes.push_back({1, num_classes});
  }
  
  std::vector<std::unique_ptr<triton::core::MetalBuffer>> output_buffers;
  RETURN_IF_ERROR(BatchAllocateTensors(
      output_names, output_types, output_shapes, output_buffers));
  
  // Simulate model execution with unified memory
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    // Track memory access
    BackendMemoryAccessTracker input_access(
        input_tensors[i]->Data(), input_tensors[i]->Size(), true, input_name);
    
    BackendMemoryAccessTracker output_access(
        output_buffers[i]->Data(), output_buffers[i]->Size(), false, output_name);
    
    // Simulate inference
    // In real implementation, this would call PyTorch or other framework
    float* output_data = static_cast<float*>(output_buffers[i]->Data());
    for (int j = 0; j < num_classes; ++j) {
      output_data[j] = static_cast<float>(j) / num_classes;
    }
    
    // Create output tensor and set response
    std::unique_ptr<triton::core::ZeroCopyTensor> output_tensor;
    auto status = triton::core::ZeroCopyTensor::CreateFromGPUMemory(
        output_tensor, output_buffers[i]->Data(),
        output_buffers[i]->Size(), output_shapes[i], TRITONSERVER_TYPE_FP32);
    
    if (!status.IsOk()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("Failed to create output tensor: " + status.Message()).c_str());
    }
    
    RETURN_IF_ERROR(SetZeroCopyOutputTensor(
        responses[i], output_name, std::move(output_tensor)));
  }
  
  // Log memory statistics periodically
  static size_t execution_count = 0;
  if (++execution_count % 100 == 0) {
    size_t total_allocated, unified_memory_used, transfers_eliminated;
    GetMemoryStatistics(total_allocated, unified_memory_used, transfers_eliminated);
    
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Unified Memory Stats: Total=") +
         std::to_string(total_allocated / (1024*1024)) + "MB, " +
         "Unified=" + std::to_string(unified_memory_used / (1024*1024)) + "MB, " +
         "Transfers Eliminated=" + std::to_string(transfers_eliminated)).c_str());
  }
  
  return nullptr;  // success
}

}}}  // namespace triton::backend::metal

#endif  // TRITON_ENABLE_METAL