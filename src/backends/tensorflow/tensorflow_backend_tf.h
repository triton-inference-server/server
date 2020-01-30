// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include <map>
#include <vector>

// To avoid namespace and protobuf collision between TRTIS and
// TensorFlow, we keep TensorFlow interface isolated to
// tensorflow_backend_tf. We use a strict C interface to avoid any ABI
// problems since we don't know how TF is built.

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#define TRTISTF_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define TRTISTF_EXPORT __attribute__((__visibility__("default")))
#else
#define TRTISTF_EXPORT
#endif

// GPU device number that indicates that no gpu is available.
#define TRTISTF_NO_GPU_DEVICE -1

// GPU device number that indicates TRTIS should do nothing to control
// the device alloaction for the network and let Tensorflow handle it.
#define TRTISTF_MODEL_DEVICE -2

// Max batch size value that indicates batching is not supported.
#define TRTISTF_NO_BATCHING 0

// Error reporting. A NULL TRTISTF_Error indicates no error, otherwise
// the error is indicated by the 'msg_'.
typedef struct {
  // The error message as a null-terminated string.
  char* msg_;
} TRTISTF_Error;

// Delete an error.
TRTISTF_EXPORT void TRTISTF_ErrorDelete(TRTISTF_Error* error);

// Input or output datatype. Protobufs can't cross the TRTISTF
// boundary so need to have this non-protobuf definition.
typedef enum {
  TRTISTF_TYPE_INVALID,
  TRTISTF_TYPE_BOOL,
  TRTISTF_TYPE_UINT8,
  TRTISTF_TYPE_UINT16,
  TRTISTF_TYPE_UINT32,
  TRTISTF_TYPE_UINT64,
  TRTISTF_TYPE_INT8,
  TRTISTF_TYPE_INT16,
  TRTISTF_TYPE_INT32,
  TRTISTF_TYPE_INT64,
  TRTISTF_TYPE_FP16,
  TRTISTF_TYPE_FP32,
  TRTISTF_TYPE_FP64,
  TRTISTF_TYPE_STRING
} TRTISTF_DataType;

typedef enum {
  TRTISTF_MODE_FP32,
  TRTISTF_MODE_FP16,
  TRTISTF_MODE_INT8,
} TRTISTF_TFTRTPrecisionMode;

// Config for TF-TRT optimization if specified
typedef struct {
  bool is_dynamic_op_;
  int64_t max_batch_size_;
  int64_t max_workspace_size_bytes_;
  TRTISTF_TFTRTPrecisionMode precision_mode_;
  int64_t minimum_segment_size_;
  int64_t max_cached_engines_;
} TRTISTF_TFTRTConfig;

// A shape
typedef struct {
  // Number of dimensions in the shape
  size_t rank_;

  // The size of each dimension. -1 indicates variables-sized
  // dimension
  int64_t* dims_;
} TRTISTF_Shape;

// Information about an input or output
typedef struct {
  // Name as null-terminated string
  char* name_;

  // Name in the model itself as null-terminated string. May be null
  // if the in-model name is the same as 'name_'
  char* inmodel_name_;

  // The data-type
  TRTISTF_DataType data_type_;

  // The shape
  TRTISTF_Shape* shape_;
} TRTISTF_IO;

// List of I/O information
typedef struct trtistf_iolist_struct {
  TRTISTF_IO* io_;
  struct trtistf_iolist_struct* next_;
} TRTISTF_IOList;

//
// Tensor
//

// Opaque handle to a tensor
struct TRTISTF_Tensor;

// List of tensors
typedef struct trtistf_tensorlist_struct {
  TRTISTF_Tensor* tensor_;
  struct trtistf_tensorlist_struct* next_;
} TRTISTF_TensorList;

// Create an new tensor list. Ownership of 'tensor' passes to the
// list.
TRTISTF_EXPORT TRTISTF_TensorList* TRTISTF_TensorListNew(
    TRTISTF_Tensor* tensor, TRTISTF_TensorList* next);

// Delete a list of tensors. Any tensors contained in the list are
// also deleted.
TRTISTF_EXPORT void TRTISTF_TensorListDelete(TRTISTF_TensorList* list);


// Create a new tensor with a given name, type and shape. 'shape_dims'
// must be nullptr if shape_rank is 0. If a tensor is intended to be used as
// GPU input for model that supports GPU I/O (see TRTISTF_ModelMakeCallable),
// 'tf_gpu_id' must be the same as the model's device id. Otherwise, negative
// value should be provided. Note that a tensor may be created on CPU if
// the data type is not supported for GPU tensor.
// Return nullptr if failed to create the tensor.
TRTISTF_EXPORT TRTISTF_Tensor* TRTISTF_TensorNew(
    const char* name, TRTISTF_DataType dtype, size_t shape_rank,
    int64_t* shape_dims, int tf_gpu_id);

// Return a tensor's datatype.
TRTISTF_EXPORT TRTISTF_DataType TRTISTF_TensorDataType(TRTISTF_Tensor* tensor);

// Return the size of a tensor datatype, in bytes.
TRTISTF_EXPORT int64_t TRTISTF_TensorDataTypeByteSize(TRTISTF_Tensor* tensor);

// Return the shape of the tensor. The shape is owned by the tensor
// and should not be modified or freed by the caller.
TRTISTF_EXPORT
TRTISTF_Shape* TRTISTF_TensorShape(TRTISTF_Tensor* tensor);

// Get the base of the tensor data. Defined only for non-string
// types.. bad things might happen if called for string type tensor.
TRTISTF_EXPORT char* TRTISTF_TensorData(TRTISTF_Tensor* tensor);

// Check whether the memory type of the tensor data is GPU.
TRTISTF_EXPORT bool TRTISTF_TensorIsGPUTensor(TRTISTF_Tensor* tensor);

// Get the size, in bytes, of the tensor data. Defined only for
// non-string types.. bad things might happen if called for string
// type tensor.
TRTISTF_EXPORT size_t TRTISTF_TensorDataByteSize(TRTISTF_Tensor* tensor);

// Get a string at a specified index within a tensor. Defined only for
// string type.. bad things might happen if called for non-string type
// tensor. The returned string is owned by the Tensor and must be
// copied if the caller requires ownership. Additionally returns the
// 'length' of the string.
TRTISTF_EXPORT const char* TRTISTF_TensorString(
    TRTISTF_Tensor* tensor, size_t idx, size_t* length);

// Set a string at a specified index within a tensor. Defined only for
// string type.. bad things might happen if called for non-string type
// tensor. The provided string is copied by the tensor so the caller
// retains ownership of 'str'. 'str' may be NULL to indicate that the
// string should be set to empty. 'length' denotes the size of the
// character sequence to copy into the string within the tensor.
TRTISTF_EXPORT void TRTISTF_TensorSetString(
    TRTISTF_Tensor* tensor, size_t idx, const char* str, size_t length);

//
// Model
//

// Opaque handle to a model
struct TRTISTF_Model;

// Create a GraphDef model.
TRTISTF_EXPORT TRTISTF_Error* TRTISTF_ModelCreateFromGraphDef(
    TRTISTF_Model** trtistf_model, const char* model_name,
    const char* model_path, const int device_id, const bool has_graph_level,
    const int graph_level, const bool allow_gpu_memory_growth,
    const float per_process_gpu_memory_fraction,
    const bool allow_soft_placement,
    const std::map<int, std::vector<float>>& memory_limit_mb,
    const TRTISTF_TFTRTConfig* tftrt_config);

// Create a SavedModel model.
TRTISTF_EXPORT TRTISTF_Error* TRTISTF_ModelCreateFromSavedModel(
    TRTISTF_Model** trtistf_model, const char* model_name,
    const char* model_path, const int device_id, const bool has_graph_level,
    const int graph_level, const bool allow_gpu_memory_growth,
    const float per_process_gpu_memory_fraction,
    const bool allow_soft_placement,
    const std::map<int, std::vector<float>>& memory_limit_mb,
    const TRTISTF_TFTRTConfig* tftrt_config);

// Delete a model.
TRTISTF_EXPORT void TRTISTF_ModelDelete(TRTISTF_Model* model);

// Create a Callable for the model so that the inputs will be assumed to be from
// GPU while the outputs will be produced on GPU. The Callable will assume the
// inputs are on the same TF device (vGPU) as the model session.
// Note that depending on the data type, GPU tensor may not be supported,
// in such case, the callable will expect those unsupported I/Os to be on CPU.
TRTISTF_Error* TRTISTF_ModelMakeCallable(
    TRTISTF_Model* model, const char** input_names,
    const TRTISTF_DataType* input_types, const size_t num_inputs,
    const char** output_names, const TRTISTF_DataType* output_types,
    const size_t num_outputs);

// Get information about a model inputs. The returned list is owned by
// the model and should not be modified or freed by the caller.
TRTISTF_EXPORT TRTISTF_IOList* TRTISTF_ModelInputs(TRTISTF_Model* model);

// Get information about a model outputs. The returned list is owned
// by the model and should not be modified or freed by the caller.
TRTISTF_EXPORT TRTISTF_IOList* TRTISTF_ModelOutputs(TRTISTF_Model* model);

// Run a model using the provides input tensors to produce the named
// outputs. Ownership of the 'input_tensors' is passed to the model
// and the caller must not access (or free) it after this
// call. 'output_tensors' returns the outputs in the same order as
// 'output_names'. The caller must free 'output_tensors' by calling
// TRTISTF_TensorListDelete.
TRTISTF_EXPORT TRTISTF_Error* TRTISTF_ModelRun(
    TRTISTF_Model* model, TRTISTF_TensorList* input_tensors, size_t num_outputs,
    const char** output_names, TRTISTF_TensorList** output_tensors);

#ifdef __cplusplus
}  // extern "C"
#endif
