// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#define TRTIS_CUSTOM_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define TRTIS_CUSTOM_EXPORT __attribute__((__visibility__("default")))
#else
#define TRTIS_CUSTOM_EXPORT
#endif

/// GPU device number that indicates that no GPU is available for a
/// context. In CustomInitializeData this value is used for
/// 'gpu_device_id' to indicate that the model must execute on the
/// CPU.
#define CUSTOM_NO_GPU_DEVICE -1

/// The number of server parameters provided to custom backend for
/// initialization. This must keep aligned with CustomServerParameter
/// enum values.
#define CUSTOM_SERVER_PARAMETER_CNT 2

/// Types of memory recognized by Triton and custom backend.
typedef enum custom_memorytype_enum {
  CUSTOM_MEMORY_CPU,
  CUSTOM_MEMORY_GPU,
  CUSTOM_MEMORY_CPU_PINNED
} CustomMemoryType;

/// The server parameter values provided to custom backends. New
/// values must be added using the next greater integer value and
/// CUSTOM_SERVER_PARAMETER_CNT must be updated to match.
typedef enum custom_serverparamkind_enum {
  /// The inference server version.
  INFERENCE_SERVER_VERSION = 0,

  /// The absolute path to the root directory of the model repository.
  MODEL_REPOSITORY_PATH = 1
} CustomServerParameter;

// The initialization information provided to a custom backend when it
// is created.
typedef struct custom_initdata_struct {
  /// The name of this instance of the custom backend. Instance names
  /// are unique.
  const char* instance_name;

  /// Serialized representation of the model configuration. This
  /// serialization is owned by the caller and must be copied if a
  /// persistent copy of required by the custom backend.
  const char* serialized_model_config;

  /// The size of 'serialized_model_config', in bytes.
  size_t serialized_model_config_size;

  /// The GPU device ID to initialize for, or CUSTOM_NO_GPU_DEVICE if
  /// should initialize for CPU.
  int gpu_device_id;

  /// The number of server parameters (i.e. the length of
  /// 'server_parameters').
  size_t server_parameter_cnt;

  /// The server parameter values as null-terminated strings, indexed
  /// by CustomServerParameter. This strings are owned by the caller
  /// and must be copied if a persistent copy of required by the
  /// custom backend.
  const char** server_parameters;
} CustomInitializeData;

/// A payload represents the input tensors and the required output
/// needed for execution in the backend.
typedef struct custom_payload_struct {
  /// The size of the batch represented by this payload.
  uint32_t batch_size;

  /// The number of inputs included in this payload.
  uint32_t input_cnt;

  /// For each of the 'input_cnt' inputs, the name of the input as a
  /// null-terminated string.
  const char** input_names;

  /// For each of the 'input_cnt' inputs, the number of dimensions in
  /// the input's shape, not including the batch dimension.
  const size_t* input_shape_dim_cnts;

  /// For each of the 'input_cnt' inputs, the shape of the input, not
  /// including the batch dimension.
  const int64_t** input_shape_dims;

  /// The number of outputs that must be computed for this
  /// payload. Can be 0 to indicate that no outputs are required from
  /// the backend.
  uint32_t output_cnt;

  /// For each of the 'output_cnt' outputs, the name of the output as
  /// a null-terminated string.  Each name must be one of the names
  /// from the model configuration, but all outputs do not need to be
  /// computed.
  const char** required_output_names;

  /// The context to use with CustomGetNextInput callback function to
  /// get the input tensor values for this payload.
  void* input_context;

  /// The context to use with CustomGetOutput callback function to get
  /// the buffer for output tensor values for this payload.
  void* output_context;

  /// The error code indicating success or failure from execution. A
  /// value of 0 (zero) indicates success, all other values indicate
  /// failure and are backend defined.
  int error_code;
} CustomPayload;

/// Type for the CustomGetNextInput callback function.
///
/// This callback function is provided in the call to ComputeExecute
/// and is used to get the value of the input tensors. Each call to
/// this function returns a contiguous block of the input tensor
/// value. The entire tensor value may be in multiple non-contiguous
/// blocks and so this function must be called multiple times until
/// 'content' returns nullptr. This callback function is not thread
/// safe.
///
/// \param input_context The input context provided in call to
/// CustomExecute.
/// \param name The name of the input tensor.
/// \param content Returns a pointer to the next contiguous block of
/// content for the named input. Returns nullptr if there is no more
/// content for the input.
/// \param content_byte_size Acts as both input and output. On input
/// gives the maximum size expected for 'content'. Returns the actual
/// size, in bytes, of 'content'.
/// \return false if error, true if success.
typedef bool (*CustomGetNextInputFn_t)(
    void* input_context, const char* name, const void** content,
    uint64_t* content_byte_size);

/// Type for the CustomGetOutput callback function.
///
/// This callback function is provided in the call to ComputeExecute
/// and is used to report the shape of an output and to get the
/// buffers to store the output tensor values. This callback funtion
/// is not thread safe.
///
/// \param output_context The output context provided in call to
/// CustomExecute.
/// \param name The name of the output tensor.
/// \param shape_dim_cnt The number of dimensions in the output shape.
/// \param shape_dims The dimensions of the output shape.
/// \param content_byte_size The size, in bytes, of the output tensor.
/// \param content Returns a pointer to a buffer where the output for
/// the tensor should be copied. If nullptr and function returns true
/// (no error), then the output should not be written and the backend
/// should continue to the next output. If non-nullptr, the size of
/// the buffer will be large enough to hold 'content_byte_size' bytes.
/// \return false if error, true if success.
typedef bool (*CustomGetOutputFn_t)(
    void* output_context, const char* name, size_t shape_dim_cnt,
    int64_t* shape_dims, uint64_t content_byte_size, void** content);

/// Type for the CustomVersion function.
typedef uint32_t (*CustomVersionFn_t)();

/// Type for the CustomInitialize function.
typedef int (*CustomInitializeFn_t)(const CustomInitializeData*, void**);

/// Type for the CustomFinalize function.
typedef int (*CustomFinalizeFn_t)(void*);

/// Type for the CustomErrorString function.
typedef char* (*CustomErrorStringFn_t)(void*, int);

/// Type for the CustomExecute function.
typedef int (*CustomExecuteFn_t)(
    void*, uint32_t, CustomPayload*, CustomGetNextInputFn_t,
    CustomGetOutputFn_t);

/// See CustomGetNextInputFn_t. This callback funtion is not thread
/// safe.
///
/// \param memory_type Acts as both input and output. On input
/// gives the buffer memory type preferred by the function caller.
/// Returns the actual memory type of 'content'.
/// \param memory_type_id Acts as both input and output. On input
/// gives the buffer memory type id preferred by the function caller.
/// Returns the actual memory type id of 'content'.
typedef bool (*CustomGetNextInputV2Fn_t)(
    void* input_context, const char* name, const void** content,
    uint64_t* content_byte_size, CustomMemoryType* memory_type,
    int64_t* memory_type_id);

/// See CustomGetOutputFn_t. This callback funtion is not thread safe.
///
/// \param memory_type Acts as both input and output. On input
/// gives the buffer memory type preferred by the function caller.
/// Returns the actual memory type of 'content'.
/// \param memory_type_id Acts as both input and output. On input
/// gives the buffer memory type id preferred by the function caller.
/// Returns the actual memory type id of 'content'.
typedef bool (*CustomGetOutputV2Fn_t)(
    void* output_context, const char* name, size_t shape_dim_cnt,
    int64_t* shape_dims, uint64_t content_byte_size, void** content,
    CustomMemoryType* memory_type, int64_t* memory_type_id);

/// Type for the CustomExecuteV2 function.
typedef int (*CustomExecuteV2Fn_t)(
    void*, uint32_t, CustomPayload*, CustomGetNextInputV2Fn_t,
    CustomGetOutputV2Fn_t);

/// Get the custom version. For a custom backend that doesn't define this entry
/// point, the inference server will assume the backend version is 1. The
/// currently supported versions are defined below, returning any other version
/// is an error:
///
/// Version 1: Input and output tensors must be communicated via system memory
/// (i.e. CPU memory). The CustomExecute function must be defined and
/// CustomGetNextInputFn_t and CustomGetOutputFn_t define the function signature
/// for the input and output callbacks.
///
/// Version 2: Input and output tensors may be communicated by both system
/// memory and GPU memory. The CustomExecuteV2 function must be defined and
/// CustomGetNextInputV2Fn_t and CustomGetOutputV2Fn_t define the function
/// signature for the input and output callbacks.
///
/// \return the custom version.
TRTIS_CUSTOM_EXPORT uint32_t CustomVersion();

/// Initialize the custom backend for a given model configuration and
/// get the associated custom context.
///
/// \param data The CustomInitializeData provided for initialization.
/// \param custom_context Returns the opaque handle to the custom
/// state associated with this initialization. Returns nullptr if
/// no context associated with the initialization. Note that in the case
/// of initialization failure, a context state needs to be returned to retrieve
/// error string for a given code.
/// \return An error code. Zero indicates success, all other values
/// indicate failure. Use CustomErrorString to get the error string
/// for an error code.
TRTIS_CUSTOM_EXPORT int CustomInitialize(
    const CustomInitializeData* data, void** custom_context);

/// Finalize a custom context. All state associated with the context
/// should be freed.
///
/// \param custom_context The custom state associated with context
/// that should be freed. Can be nullptr if no custom state.
/// \return An error code. Zero indicates success, all other values
/// indicate failure. Use CustomErrorString to get the error string
/// for an error code.
TRTIS_CUSTOM_EXPORT int CustomFinalize(void* custom_context);

/// Get the string for an error code.
///
/// \param custom_context The custom state associated with the error
/// code. Can be nullptr if no custom state.
/// \param errcode The error code.
/// \return The error code string, or nullptr if the error code has no
/// string representation.
TRTIS_CUSTOM_EXPORT const char* CustomErrorString(
    void* custom_context, int errcode);

/// Execute the custom model using the version 1 implementation of the execute
/// interface. This function must be defined when the custom backend returns 1
/// from CustomVersion (or when CustomVersion is not defined)
///
/// \param custom_context The custom state associated with the context
/// that should execute. Can be nullptr if no custom state.
/// \param payload_cnt The number of payloads to execute.
/// \param payloads The payloads to execute.
/// \param input_fn The callback function to get tensor input (see
/// CustomGetNextInputFn_t).
/// \param output_fn The callback function to get buffer for tensor
/// output (see CustomGetOutputFn_t).
/// \return An error code. Zero indicates success, all other values
/// indicate failure. Use CustomErrorString to get the error string
/// for an error code.
TRTIS_CUSTOM_EXPORT int CustomExecute(
    void* custom_context, uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn);

/// Execute the custom model using the version 2 implementation of the execute
/// interface. This function must be defined when the custom backend returns 2
/// from CustomVersion. See CustomExecute for description of the parameters.
TRTIS_CUSTOM_EXPORT int CustomExecuteV2(
    void* custom_context, uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputV2Fn_t input_fn, CustomGetOutputV2Fn_t output_fn);

#ifdef __cplusplus
}
#endif
