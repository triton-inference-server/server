// Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string>

#include "triton/core/tritonserver.h"

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
      return TRITONSERVER_ErrorNew(                                    \
          TRITONSERVER_ErrorCode(err__),                               \
          (std::string(MSG) + ": " + TRITONSERVER_ErrorMessage(err__)) \
              .c_str());                                               \
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
