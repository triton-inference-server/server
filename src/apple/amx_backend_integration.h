// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// AMX Backend Integration for Triton

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"
#include "amx_provider.h"

namespace triton {
namespace apple {

// AMX operation registry
class AMXOpRegistry {
public:
    // Operation handler function
    using OpHandler = std::function<TRITONSERVER_Error*(
        const std::vector<TRITONBACKEND_Input*>& inputs,
        const std::vector<TRITONBACKEND_Output*>& outputs,
        const std::unordered_map<std::string, std::string>& params)>;
    
    static AMXOpRegistry& Instance();
    
    // Register an operation
    void RegisterOp(const std::string& op_name, OpHandler handler);
    
    // Get operation handler
    OpHandler GetOp(const std::string& op_name) const;
    
    // Check if operation is supported
    bool IsSupported(const std::string& op_name) const;
    
    // List all supported operations
    std::vector<std::string> ListOps() const;
    
private:
    AMXOpRegistry();
    std::unordered_map<std::string, OpHandler> ops_;
};

// AMX backend instance
class AMXBackendInstance {
public:
    AMXBackendInstance(const std::string& name, const std::string& version);
    ~AMXBackendInstance();
    
    // Execute inference request
    TRITONSERVER_Error* Execute(
        TRITONBACKEND_Request* request,
        const std::vector<TRITONBACKEND_Response*>& responses);
    
    // Get backend statistics
    void GetStatistics(TRITONBACKEND_Stat** stats, uint32_t* stat_count);
    
private:
    std::string name_;
    std::string version_;
    AMXProvider* amx_provider_;
    
    // Helper methods
    TRITONSERVER_Error* ProcessTensor(
        TRITONBACKEND_Input* input,
        TRITONBACKEND_Output* output);
    
    TRITONSERVER_Error* GetInputTensor(
        TRITONBACKEND_Request* request,
        const std::string& name,
        const void** data,
        size_t* byte_size,
        TRITONSERVER_DataType* dtype,
        const int64_t** shape,
        uint32_t* dims_count);
};

// AMX-accelerated operations
namespace AMXOps {

// Matrix operations
TRITONSERVER_Error* MatMul(
    const std::vector<TRITONBACKEND_Input*>& inputs,
    const std::vector<TRITONBACKEND_Output*>& outputs,
    const std::unordered_map<std::string, std::string>& params);

TRITONSERVER_Error* BatchMatMul(
    const std::vector<TRITONBACKEND_Input*>& inputs,
    const std::vector<TRITONBACKEND_Output*>& outputs,
    const std::unordered_map<std::string, std::string>& params);

// Convolution operations
TRITONSERVER_Error* Conv2D(
    const std::vector<TRITONBACKEND_Input*>& inputs,
    const std::vector<TRITONBACKEND_Output*>& outputs,
    const std::unordered_map<std::string, std::string>& params);

TRITONSERVER_Error* DepthwiseConv2D(
    const std::vector<TRITONBACKEND_Input*>& inputs,
    const std::vector<TRITONBACKEND_Output*>& outputs,
    const std::unordered_map<std::string, std::string>& params);

// Pooling operations
TRITONSERVER_Error* MaxPool2D(
    const std::vector<TRITONBACKEND_Input*>& inputs,
    const std::vector<TRITONBACKEND_Output*>& outputs,
    const std::unordered_map<std::string, std::string>& params);

TRITONSERVER_Error* AvgPool2D(
    const std::vector<TRITONBACKEND_Input*>& inputs,
    const std::vector<TRITONBACKEND_Output*>& outputs,
    const std::unordered_map<std::string, std::string>& params);

// Activation functions
TRITONSERVER_Error* ReLU(
    const std::vector<TRITONBACKEND_Input*>& inputs,
    const std::vector<TRITONBACKEND_Output*>& outputs,
    const std::unordered_map<std::string, std::string>& params);

TRITONSERVER_Error* Sigmoid(
    const std::vector<TRITONBACKEND_Input*>& inputs,
    const std::vector<TRITONBACKEND_Output*>& outputs,
    const std::unordered_map<std::string, std::string>& params);

TRITONSERVER_Error* Tanh(
    const std::vector<TRITONBACKEND_Input*>& inputs,
    const std::vector<TRITONBACKEND_Output*>& outputs,
    const std::unordered_map<std::string, std::string>& params);

// Normalization operations
TRITONSERVER_Error* BatchNorm(
    const std::vector<TRITONBACKEND_Input*>& inputs,
    const std::vector<TRITONBACKEND_Output*>& outputs,
    const std::unordered_map<std::string, std::string>& params);

TRITONSERVER_Error* LayerNorm(
    const std::vector<TRITONBACKEND_Input*>& inputs,
    const std::vector<TRITONBACKEND_Output*>& outputs,
    const std::unordered_map<std::string, std::string>& params);

// Transformer operations
TRITONSERVER_Error* MultiHeadAttention(
    const std::vector<TRITONBACKEND_Input*>& inputs,
    const std::vector<TRITONBACKEND_Output*>& outputs,
    const std::unordered_map<std::string, std::string>& params);

TRITONSERVER_Error* FeedForward(
    const std::vector<TRITONBACKEND_Input*>& inputs,
    const std::vector<TRITONBACKEND_Output*>& outputs,
    const std::unordered_map<std::string, std::string>& params);

} // namespace AMXOps

// Helper utilities
namespace AMXUtils {

// Convert Triton data type to AMX data type
TRITONSERVER_Error* ConvertDataType(
    TRITONSERVER_DataType triton_type,
    amx::DataType& amx_type);

// Get optimal AMX configuration for operation
AMXConfig GetOptimalConfig(
    const std::string& op_name,
    const std::vector<int64_t>& input_shape,
    TRITONSERVER_DataType dtype);

// Validate tensor for AMX operation
TRITONSERVER_Error* ValidateTensor(
    const int64_t* shape,
    uint32_t dims_count,
    TRITONSERVER_DataType dtype,
    const std::string& tensor_name);

// Memory alignment check
bool IsMemoryAligned(const void* ptr, size_t alignment = 64);

// Pad tensor for AMX tile requirements
TRITONSERVER_Error* PadTensor(
    const void* input,
    void* output,
    const int64_t* shape,
    uint32_t dims_count,
    size_t element_size,
    size_t tile_size = 32);

} // namespace AMXUtils

// Integration with Metal backend
class AMXMetalInterop {
public:
    // Check if operation should use AMX or Metal
    static bool ShouldUseAMX(
        const std::string& op_name,
        const std::vector<int64_t>& shape,
        TRITONSERVER_DataType dtype);
    
    // Transfer data between AMX and Metal
    static TRITONSERVER_Error* TransferToMetal(
        const void* amx_data,
        void* metal_buffer,
        size_t byte_size);
    
    static TRITONSERVER_Error* TransferFromMetal(
        const void* metal_buffer,
        void* amx_data,
        size_t byte_size);
    
    // Hybrid execution planning
    struct ExecutionPlan {
        std::vector<std::string> amx_ops;
        std::vector<std::string> metal_ops;
        std::vector<std::pair<size_t, size_t>> transfer_points;
    };
    
    static ExecutionPlan PlanHybridExecution(
        const std::vector<std::string>& op_sequence,
        const std::vector<std::vector<int64_t>>& shapes);
};

} // namespace apple
} // namespace triton