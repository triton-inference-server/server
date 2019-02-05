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

/// GPU device number that indicates that no GPU is available for a
/// context. In CustomInitialize this value is used for 'gpu_device_id' to
/// indicate that the model must execute on the CPU.
#define CUSTOM_NO_GPU_DEVICE -1

// A payload represents the input tensors and the required output
// needed for execution in the backend.
typedef struct custom_payload_struct {
  // The size of the batch represented by this payload.
  uint32_t batch_size;

  // The number of inputs included in this payload.
  uint32_t input_cnt;

  // The 'input_cnt' names of the inputs included in this payload.
  const char** input_names;

  // For each of the 'input_cnt' inputs, the number of dimensions in
  // the input's shape, not including the batch dimension.
  const size_t* input_shape_dim_cnts;

  // For each of the 'input_cnt' inputs, the shape of the input, not
  // including the batch dimension.
  const int64_t** input_shape_dims;

  // The number of outputs that must be computed for this payload. Can
  // be 0 to indicate that no outputs are required from the backend.
  uint32_t output_cnt;

  // The 'output_cnt' names of the outputs that must be computed for
  // this payload. Each name must be one of the names from the model
  // configuration, but all outputs do not need to be computed.
  const char** required_output_names;

  // The context to use with CustomGetNextInput callback function to
  // get the input tensor values for this payload.
  void* input_context;

  // The context to use with CustomGetOutput callback function to get
  // the buffer for output tensor values for this payload.
  void* output_context;

  // The error code indicating success or failure from execution. A
  // value of 0 (zero) indicates success, all other values indicate
  // failure and are backend defined.
  int error_code;
} CustomPayload;

/// Type for the CustomGetNextInput callback function.
///
/// This callback function is provided in the call to ComputeExecute
/// and is used to get the value of the input tensors. Each call to
/// this function returns a contiguous block of the input tensor
/// value. The entire tensor value may be in multiple non-contiguous
/// blocks and so this function must be called multiple times until
/// 'content' returns nullptr.
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
/// and is used to get the buffers to store the output tensor
/// values.
///
/// \param output_context The output context provided in call to
/// CustomExecute.
/// \param name The name of the output tensor.
/// \param shape_dim_cnt The number of dimensions in the output shape.
/// \param shape_dims The dimensions of the output shape.
/// \param content_byte_size The size, in bytes, of the output tensor.
/// \param content Returns a pointer to a buffer where the output for
/// the tensor should be copied. The size of the buffer will be large
/// enough to hold 'content_byte_size' bytes.
/// \return false if error, true if success.
typedef bool (*CustomGetOutputFn_t)(
    void* output_context, const char* name, size_t shape_dim_cnt,
    int64_t* shape_dims, uint64_t content_byte_size, void** content);

/// Type for the CustomInitialize function.
typedef int (*CustomInitializeFn_t)(const char*, size_t, int, void**);

/// Type for the CustomFinalize function.
typedef int (*CustomFinalizeFn_t)(void*);

/// Type for the CustomErrorString function.
typedef char* (*CustomErrorStringFn_t)(void*, int);

/// Type for the CustomExecute function.
typedef int (*CustomExecuteFn_t)(
    void*, uint32_t, CustomPayload*, CustomGetNextInputFn_t,
    CustomGetOutputFn_t);

/// Initialize the custom shared library for a given model
/// configuration and get the associated custom context.
///
/// \param serialized_model_config Serialized representation of the
/// model configuration to use for initialization. This serialization
/// is owned by the caller and so must be copied if a persistent
/// copy of required by the shared library.
/// \param serialized_model_config_size The size of serialized_model_config,
/// in bytes.
/// \param gpu_device_id The GPU device ID to initialize for, or
/// CUSTOM_NO_GPU_DEVICE if should initialize for CPU.
/// \param custom_context Returns the opaque handle to the custom
/// state associated with this initialization. Returns nullptr if
/// no context associated with the initialization.
/// \return An error code. Zero indicates success, all other values
/// indicate failure. Use CustomErrorString to get the error string
/// for an error code.
int CustomInitialize(
    const char* serialized_model_config, size_t serialized_model_config_size,
    int gpu_device_id, void** custom_context);

/// Finalize a custom context. All state associated with the context
/// should be freed.
///
/// \param custom_context The custom state associated with context
/// that should be freed. Can be nullptr if no custom state.
/// \return An error code. Zero indicates success, all other values
/// indicate failure. Use CustomErrorString to get the error string
/// for an error code.
int CustomFinalize(void* custom_context);

/// Get the string for an error code.
///
/// \param custom_context The custom state associated with the error
/// code. Can be nullptr if no custom state.
/// \param errcode The error code.
/// \return The error code string, or nullptr if the error code has no
/// string representation.
const char* CustomErrorString(void* custom_context, int errcode);

/// Execute the custom model.
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
int CustomExecute(
    void* custom_context, uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn);

#ifdef __cplusplus
}
#endif
