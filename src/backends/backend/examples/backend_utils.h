// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <time.h>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "src/backends/backend/tritonbackend.h"

#define TRITONJSON_STATUSTYPE TRITONSERVER_Error*
#define TRITONJSON_STATUSRETURN(M) \
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (M).c_str())
#define TRITONJSON_STATUSSUCCESS nullptr
#include "src/core/json.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace nvidia { namespace inferenceserver { namespace backend {

#define IGNORE_ERROR(X)                   \
  do {                                    \
    TRITONSERVER_Error* ie_err__ = (X);   \
    if (ie_err__ != nullptr) {            \
      TRITONSERVER_ErrorDelete(ie_err__); \
    }                                     \
  } while (false)

#define LOG_IF_ERROR(X, MSG)                                                   \
  do {                                                                         \
    TRITONSERVER_Error* lie_err__ = (X);                                       \
    if (lie_err__ != nullptr) {                                                \
      IGNORE_ERROR(TRITONSERVER_LogMessage(                                    \
          TRITONSERVER_LOG_INFO, __FILE__, __LINE__,                           \
          (std::string(MSG) + ": " + TRITONSERVER_ErrorCodeString(lie_err__) + \
           " - " + TRITONSERVER_ErrorMessage(lie_err__))                       \
              .c_str()));                                                      \
      TRITONSERVER_ErrorDelete(lie_err__);                                     \
    }                                                                          \
  } while (false)

#define LOG_MESSAGE(LEVEL, MSG)                                  \
  do {                                                           \
    LOG_IF_ERROR(                                                \
        TRITONSERVER_LogMessage(LEVEL, __FILE__, __LINE__, MSG), \
        ("failed to log message: "));                            \
  } while (false)


#define RETURN_ERROR_IF_FALSE(P, C, MSG)              \
  do {                                                \
    if (!(P)) {                                       \
      return TRITONSERVER_ErrorNew(C, (MSG).c_str()); \
    }                                                 \
  } while (false)

#define RETURN_ERROR_IF_TRUE(P, C, MSG)               \
  do {                                                \
    if ((P)) {                                        \
      return TRITONSERVER_ErrorNew(C, (MSG).c_str()); \
    }                                                 \
  } while (false)

#define RETURN_IF_ERROR(X)               \
  do {                                   \
    TRITONSERVER_Error* rie_err__ = (X); \
    if (rie_err__ != nullptr) {          \
      return rie_err__;                  \
    }                                    \
  } while (false)

#define RESPOND_AND_SET_NULL_IF_ERROR(RESPONSE_PTR, X)               \
  do {                                                               \
    TRITONSERVER_Error* rarie_err__ = (X);                           \
    if (rarie_err__ != nullptr) {                                    \
      if (*RESPONSE_PTR != nullptr) {                                \
        LOG_IF_ERROR(                                                \
            TRITONBACKEND_ResponseSend(                              \
                *RESPONSE_PTR, TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                rarie_err__),                                        \
            "failed to send error response");                        \
        *RESPONSE_PTR = nullptr;                                     \
      }                                                              \
      TRITONSERVER_ErrorDelete(rarie_err__);                         \
    }                                                                \
  } while (false)

#ifdef TRITON_ENABLE_STATS
#define TIMESPEC_TO_NANOS(TS) ((TS).tv_sec * 1000000000 + (TS).tv_nsec)
#define SET_TIMESTAMP(TS_NS)             \
  {                                      \
    struct timespec ts;                  \
    clock_gettime(CLOCK_MONOTONIC, &ts); \
    TS_NS = TIMESPEC_TO_NANOS(ts);       \
  }
#define DECL_TIMESTAMP(TS_NS) \
  uint64_t TS_NS;             \
  SET_TIMESTAMP(TS_NS);
#else
#define DECL_TIMESTAMP(TS_NS)
#define SET_TIMESTAMP(TS_NS)
#endif  // TRITON_ENABLE_STATS

#ifndef TRITON_ENABLE_GPU
using cudaStream_t = void*;
#endif  // !TRITON_ENABLE_GPU

/// Convenience deleter for TRITONBACKEND_ResponseFactory.
struct ResponseFactoryDeleter {
  void operator()(TRITONBACKEND_ResponseFactory* f)
  {
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseFactoryDelete(f),
        "failed deleting response factory");
  }
};

//
// BlockingQueue
//
// A blocking queue is useful for communicating between multiple
// threads within a backend. Multiple threads are often used to
// implement model instances.
///
template <typename T>
class BlockingQueue {
 public:
  bool WaitNotEmpty() const
  {
    std::unique_lock<std::mutex> lk(mu_);
    if (queue_.empty()) {
      cv_.wait(lk, [this] { return !queue_.empty(); });
    }
    return true;
  }

  bool Empty() const
  {
    std::lock_guard<std::mutex> lk(mu_);
    return queue_.empty();
  }

  T Pop()
  {
    std::unique_lock<std::mutex> lk(mu_);
    if (queue_.empty()) {
      cv_.wait(lk, [this] { return !queue_.empty(); });
    }
    auto res = std::move(queue_.front());
    queue_.pop_front();
    return res;
  }

  void Push(const T& value)
  {
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.emplace_back(value);
    }
    cv_.notify_one();
  }

  void Push(T&& value)
  {
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.emplace_back(std::move(value));
    }
    cv_.notify_one();
  }

 private:
  mutable std::mutex mu_;
  mutable std::condition_variable cv_;
  std::deque<T> queue_;
};

/// The value for a dimension in a shape that indicates that that
/// dimension can take on any size.
constexpr int WILDCARD_DIM = -1;

constexpr char kTensorRTExecutionAccelerator[] = "tensorrt";
constexpr char kOpenVINOExecutionAccelerator[] = "openvino";
constexpr char kGPUIOExecutionAccelerator[] = "gpu_io";
constexpr char kAutoMixedPrecisionExecutionAccelerator[] =
    "auto_mixed_precision";

/// Parse an array in a JSON object into the corresponding shape. The
/// array must be composed of integers.
///
/// \param io The JSON object containing the member array.
/// \param name The name of the array member in the JSON object.
/// \param shape Returns the shape.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* ParseShape(
    TritonJson::Value& io, const std::string& name,
    std::vector<int64_t>* shape);

/// Return the string representation of a shape.
///
/// \param dims The shape dimensions.
/// \param dims_count The number of dimensions.
/// \return The string representation.
std::string ShapeToString(const int64_t* dims, const size_t dims_count);

/// Return the string representation of a shape.
///
/// \param shape The shape as a vector of dimensions.
/// \return The string representation.
std::string ShapeToString(const std::vector<int64_t>& shape);

/// Return the number of elements of a shape.
///
/// \param dims The shape dimensions.
/// \param dims_count The number of dimensions.
/// \return The number of elements.
int64_t GetElementCount(const int64_t* dims, const size_t dims_count);

/// Return the number of elements of a shape.
///
/// \param shape The shape as a vector of dimensions.
/// \return The number of elements.
int64_t GetElementCount(const std::vector<int64_t>& shape);

/// Get the size, in bytes, of a tensor based on datatype and
/// shape.
/// \param dtype The data-type.
/// \param dims The shape.
/// \return The size, in bytes, of the corresponding tensor, or -1 if
/// unable to determine the size.
int64_t GetByteSize(
    const TRITONSERVER_DataType& dtype, const std::vector<int64_t>& dims);

/// Get an input tensor's contents into a buffer.
///
/// \param request The inference request.
/// \param input_name The name of the input buffer.
/// \param buffer The buffer where the input tensor content is copied into.
/// \param buffer_byte_size Acts as both input and output. On input
/// gives the size of 'buffer', in bytes. The function will fail if
/// the buffer is not large enough to hold the input tensor
/// contents. Returns the size of the input tensor data returned in
/// 'buffer'.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* ReadInputTensor(
    TRITONBACKEND_Request* request, const std::string& input_name, char* buffer,
    size_t* buffer_byte_size);

/// Validate that an input matches one of the allowed input names.
/// \param io The model input.
/// \param allowed The set of allowed input names.
/// \return The error status. A non-OK status indicates the input
/// is not valid.
TRITONSERVER_Error* CheckAllowedModelInput(
    TritonJson::Value& io, const std::set<std::string>& allowed);

/// Validate that an output matches one of the allowed output names.
/// \param io The model output.
/// \param allowed The set of allowed output names.
/// \return The error status. A non-OK status indicates the output
/// is not valid.
TRITONSERVER_Error* CheckAllowedModelOutput(
    TritonJson::Value& io, const std::set<std::string>& allowed);

/// Get the tensor name, false value, and true value for a boolean
/// sequence batcher control kind. If 'required' is true then must
/// find a tensor for the control. If 'required' is false, return
/// 'tensor_name' as empty-string if the control is not mapped to any
/// tensor.
///
/// \param batcher The JSON object of the sequence batcher.
/// \param model_name The name of the model.
/// \param control_kind The kind of control tensor to look for.
/// \param required Whether the tensor must be specified.
/// \param tensor_name Returns the name of the tensor.
/// \param tensor_datatype Returns the data type of the tensor.
/// \param fp32_false_value Returns the float value for false if
/// the tensor type is FP32.
/// \param fp32_true_value Returns the float value for true if
/// the tensor type is FP32.
/// \param int32_false_value Returns the int value for false if
/// the tensor type is FP32.
/// \param int32_true_value Returns the int value for true if
/// the tensor type is FP32.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* GetBooleanSequenceControlProperties(
    TritonJson::Value& batcher, const std::string& model_name,
    const std::string& control_kind, const bool required,
    std::string* tensor_name, std::string* tensor_datatype,
    float* fp32_false_value, float* fp32_true_value, int32_t* int32_false_value,
    int32_t* int32_true_value);

/// Get the tensor name and datatype for a non-boolean sequence
/// batcher control kind. If 'required' is true then must find a
/// tensor for the control. If 'required' is false, return
/// 'tensor_name' as empty-string if the control is not mapped to any
/// tensor. 'tensor_datatype' returns the required datatype for the
/// control.
///
/// \param batcher The JSON object of the sequence batcher.
/// \param model_name The name of the model.
/// \param control_kind The kind of control tensor to look for.
/// \param required Whether the tensor must be specified.
/// \param tensor_name Returns the name of the tensor.
/// \param tensor_datatype Returns the data type of the tensor.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* GetTypedSequenceControlProperties(
    TritonJson::Value& batcher, const std::string& model_name,
    const std::string& control_kind, const bool required,
    std::string* tensor_name, std::string* tensor_datatype);

/// Copy buffer from 'src' to 'dst' for given 'byte_size'. The buffer location
/// is identified by the memory type and id, and the corresponding copy will be
/// initiated.
/// \param msg The message to be prepended in error message.
/// \param src_memory_type The memory type of the source buffer.
/// \param src_memory_type_id The memory type id of the source buffer.
/// \param dst_memory_type The memory type of the destination buffer.
/// \param dst_memory_type_id The memory type id of the destination buffer.
/// \param byte_size The byte size of the source buffer.
/// \param src The pointer to the source buffer.
/// \param dst The pointer to the destination buffer.
/// \param cuda_stream specifies the stream to be associated with, and 0 can be
/// passed for default stream.
/// \param cuda_used returns whether a CUDA memory copy is initiated. If true,
/// the caller should synchronize on the given 'cuda_stream' to ensure data copy
/// is completed.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* CopyBuffer(
    const std::string& msg, const TRITONSERVER_MemoryType src_memory_type,
    const int64_t src_memory_type_id,
    const TRITONSERVER_MemoryType dst_memory_type,
    const int64_t dst_memory_type_id, const size_t byte_size, const void* src,
    void* dst, cudaStream_t cuda_stream, bool* cuda_used);

/// Returns the content in the model version path and the path to the content as
/// key-value pair.
/// \param model_repository_path The path to the model repository.
/// \param version The version of the model.
/// \param ignore_directories Whether the directories will be ignored.
/// \param ignore_files Whether the files will be ignored.
/// \param model_paths Returns the content in the model version path and
/// the path to the content.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* ModelPaths(
    const char* model_repository_path, uint64_t version,
    const bool ignore_directories, const bool ignore_files,
    std::unordered_map<std::string, std::string>* model_paths);

}}}  // namespace nvidia::inferenceserver::backend
