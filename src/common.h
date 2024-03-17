// Copyright 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "triton/core/tritonserver.h"

namespace triton { namespace server {

constexpr char kInferHeaderContentLengthHTTPHeader[] =
    "Inference-Header-Content-Length";
constexpr char kAcceptEncodingHTTPHeader[] = "Accept-Encoding";
constexpr char kContentEncodingHTTPHeader[] = "Content-Encoding";
constexpr char kContentTypeHeader[] = "Content-Type";
constexpr char kContentLengthHeader[] = "Content-Length";

constexpr int MAX_GRPC_MESSAGE_SIZE = INT32_MAX;

/// The value for a dimension in a shape that indicates that that
/// dimension can take on any size.
constexpr int WILDCARD_DIM = -1;

/// Request parameter keys that start with a "triton_" prefix for internal use
const std::vector<std::string> TRITON_RESERVED_REQUEST_PARAMS{
    "triton_enable_empty_final_response"};

#define RETURN_IF_ERR(X)             \
  do {                               \
    TRITONSERVER_Error* err__ = (X); \
    if (err__ != nullptr) {          \
      return err__;                  \
    }                                \
  } while (false)

#define RETURN_MSG_IF_ERR(X, MSG)                                      \
  do {                                                                 \
    TRITONSERVER_Error* err__ = (X);                                   \
    if (err__ != nullptr) {                                            \
      auto new_err = TRITONSERVER_ErrorNew(                            \
          TRITONSERVER_ErrorCode(err__),                               \
          (std::string(MSG) + ": " + TRITONSERVER_ErrorMessage(err__)) \
              .c_str());                                               \
      TRITONSERVER_ErrorDelete(err__);                                 \
      return new_err;                                                  \
    }                                                                  \
  } while (false)

#define GOTO_IF_ERR(X, T)            \
  do {                               \
    TRITONSERVER_Error* err__ = (X); \
    if (err__ != nullptr) {          \
      goto T;                        \
    }                                \
  } while (false)

#define FAIL(MSG)                                 \
  do {                                            \
    std::cerr << "error: " << (MSG) << std::endl; \
    exit(1);                                      \
  } while (false)

#define FAIL_IF_ERR(X, MSG)                                       \
  do {                                                            \
    TRITONSERVER_Error* err__ = (X);                              \
    if (err__ != nullptr) {                                       \
      std::cerr << "error: " << (MSG) << ": "                     \
                << TRITONSERVER_ErrorCodeString(err__) << " - "   \
                << TRITONSERVER_ErrorMessage(err__) << std::endl; \
      TRITONSERVER_ErrorDelete(err__);                            \
      exit(1);                                                    \
    }                                                             \
  } while (false)

#define THROW_IF_ERR(EX_TYPE, X, MSG)                                     \
  do {                                                                    \
    TRITONSERVER_Error* err__ = (X);                                      \
    if (err__ != nullptr) {                                               \
      auto ex__ = (EX_TYPE)(std::string("error: ") + (MSG) + ": " +       \
                            TRITONSERVER_ErrorCodeString(err__) + " - " + \
                            TRITONSERVER_ErrorMessage(err__));            \
      TRITONSERVER_ErrorDelete(err__);                                    \
      throw ex__;                                                         \
    }                                                                     \
  } while (false)

#define IGNORE_ERR(X)                  \
  do {                                 \
    TRITONSERVER_Error* err__ = (X);   \
    if (err__ != nullptr) {            \
      TRITONSERVER_ErrorDelete(err__); \
    }                                  \
  } while (false)

#ifdef TRITON_ENABLE_GPU
#define FAIL_IF_CUDA_ERR(X, MSG)                                           \
  do {                                                                     \
    cudaError_t err__ = (X);                                               \
    if (err__ != cudaSuccess) {                                            \
      std::cerr << "error: " << (MSG) << ": " << cudaGetErrorString(err__) \
                << std::endl;                                              \
      exit(1);                                                             \
    }                                                                      \
  } while (false)
#endif  // TRITON_ENABLE_GPU

/// Get the integral version from a string, or fail if string does not
/// represent a valid version.
///
/// \param version_string The string version.
/// \param version Returns the integral version.
/// \return The error status. Failure if 'version_string' doesn't
/// convert to valid version.
TRITONSERVER_Error* GetModelVersionFromString(
    const std::string& version_string, int64_t* version);

/// Get the value of the environment variable, or default value if not set
///
/// \param variable_name The name of the environment variable.
/// \param default_value The default value.
/// \return The environment variable or the default value if not set.
std::string GetEnvironmentVariableOrDefault(
    const std::string& variable_name, const std::string& default_value);

/// Get the number of elements in a shape.
///
/// \param dims The shape.
/// \return The number of elements, or -1 if the number of elements
/// cannot be determined because the shape contains one or more
/// wildcard dimensions.
int64_t GetElementCount(const std::vector<int64_t>& dims);

/// Returns if 'vec' contains 'str'.
///
/// \param vec The vector of strings to search.
/// \param str The string to lookup.
/// \return True if the str is found, false otherwise.
bool Contains(const std::vector<std::string>& vec, const std::string& str);

/// Joins container of strings into a single string delimited by
/// 'delim'.
///
/// \param container The container of strings to join.
/// \param delim The delimiter to join with.
/// \return The joint string.
template <class T>
std::string
Join(const T& container, const std::string& delim)
{
  if (container.empty()) {
    return "";
  }
  std::stringstream ss;
  ss << container[0];
  for (size_t i = 1; i < container.size(); ++i) {
    ss << delim << container[i];
  }
  return ss.str();
}

}}  // namespace triton::server
