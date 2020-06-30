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
#include <string>
#include <vector>
#include "src/backends/backend/tritonbackend.h"

#define TRITONJSON_STATUSTYPE TRITONSERVER_Error*
#define TRITONJSON_STATUSRETURN(M) \
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (M).c_str())
#define TRITONJSON_STATUSSUCCESS nullptr
#include "src/core/json.h"

namespace nvidia { namespace inferenceserver { namespace backend {

#define LOG_IF_ERROR(X, MSG)                                                   \
  do {                                                                         \
    TRITONSERVER_Error* lie_err__ = (X);                                       \
    if (lie_err__ != nullptr) {                                                \
      TRITONSERVER_LogMessage(                                                 \
          TRITONSERVER_LOG_INFO, __FILE__, __LINE__,                           \
          (std::string(MSG) + ": " + TRITONSERVER_ErrorCodeString(lie_err__) + \
           " - " + TRITONSERVER_ErrorMessage(lie_err__))                       \
              .c_str());                                                       \
      TRITONSERVER_ErrorDelete(lie_err__);                                     \
    }                                                                          \
  } while (false)

#define RETURN_ERROR_IF_FALSE(P, C, MSG)              \
  do {                                                \
    if (!(P)) {                                       \
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

/// Convenience deleter for TRITONBACKEND_ResponseFactory.
struct ResponseFactoryDeleter {
  void operator()(TRITONBACKEND_ResponseFactory* f)
  {
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseFactoryDelete(f),
        "failed deleting response factory");
  }
};

///
/// InstanceProperties
///
/// Configuration information for a model instance.
///
struct InstanceProperties {
  enum class Kind { CPU, GPU };

  InstanceProperties(const size_t i, const Kind k, const int d)
      : id_(i), kind_(k), device_id_(d)
  {
  }
  std::string AsString() const;

  size_t id_;

  // For CPU device_id_ is always 0. For GPU device_id_ indicates the
  // GPU device to be used by the instance.
  Kind kind_;
  int device_id_;
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

/// Parse model configuration and extra the model instances that
/// should be implemented for the specified instance groups.
///
/// \param model_config The model configuration.
/// \param instances Returns the model instance information.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* ParseInstanceGroups(
    TritonJson::Value& model_config,
    std::vector<InstanceProperties>* instances);

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

}}}  // namespace nvidia::inferenceserver::backend
