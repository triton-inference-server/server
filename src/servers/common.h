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

#include <iostream>
#include "src/core/api.pb.h"
#include "src/core/request_status.pb.h"
#include "src/core/trtserver.h"

#ifdef TRTIS_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRTIS_ENABLE_GPU

namespace nvidia { namespace inferenceserver {

#define FAIL(MSG)                                 \
  do {                                            \
    std::cerr << "error: " << (MSG) << std::endl; \
    exit(1);                                      \
  } while (false)

#define FAIL_IF_ERR(X, MSG)                                    \
  do {                                                         \
    TRTSERVER_Error* err__ = (X);                              \
    if (err__ != nullptr) {                                    \
      std::cerr << "error: " << (MSG) << ": "                  \
                << TRTSERVER_ErrorCodeString(err__) << " - "   \
                << TRTSERVER_ErrorMessage(err__) << std::endl; \
      TRTSERVER_ErrorDelete(err__);                            \
      exit(1);                                                 \
    }                                                          \
  } while (false)

#define LOG_IF_ERR(X, MSG)                                     \
  do {                                                         \
    TRTSERVER_Error* err__ = (X);                              \
    if (err__ != nullptr) {                                    \
      std::cerr << "error: " << (MSG) << ": "                  \
                << TRTSERVER_ErrorCodeString(err__) << " - "   \
                << TRTSERVER_ErrorMessage(err__) << std::endl; \
      TRTSERVER_ErrorDelete(err__);                            \
    }                                                          \
  } while (false)

#define RETURN_IF_ERR(X)          \
  do {                            \
    TRTSERVER_Error* err__ = (X); \
    if (err__ != nullptr) {       \
      return err__;               \
    }                             \
  } while (false)

#ifdef TRTIS_ENABLE_GPU
#define FAIL_IF_CUDA_ERR(X, MSG)                                           \
  do {                                                                     \
    cudaError_t err__ = (X);                                               \
    if (err__ != cudaSuccess) {                                            \
      std::cerr << "error: " << (MSG) << ": " << cudaGetErrorString(err__) \
                << std::endl;                                              \
      exit(1);                                                             \
    }                                                                      \
  } while (false)
#endif  // TRTIS_ENABLE_GPU

//
// RequestStatusUtil
//
// Utilities for creating and using RequestStatus
//
class RequestStatusUtil {
 public:
  // Create RequestStatus from a TRTSERVER error.
  static void Create(
      RequestStatus* status, TRTSERVER_Error* err, uint64_t request_id,
      const std::string& server_id);
  // Create a RequestStatus object from a code and optional message.
  static void Create(
      RequestStatus* status, uint64_t request_id, const std::string& server_id,
      RequestStatusCode code, const std::string& msg);
  static void Create(
      RequestStatus* status, uint64_t request_id, const std::string& server_id,
      RequestStatusCode code);

  // Return the RequestStatusCode for a TRTSERVER error code.
  static RequestStatusCode CodeToStatus(TRTSERVER_Error_Code code);

  // Return a unique request ID
  static uint64_t NextUniqueRequestId();
};

TRTSERVER_Error* SetTRTSERVER_InferenceRequestOptions(
    TRTSERVER_InferenceRequestOptions* request_options,
    InferRequestHeader request_header_protobuf);

}}  // namespace nvidia::inferenceserver
