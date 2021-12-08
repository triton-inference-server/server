// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"

namespace triton { namespace backend { namespace query {


// Query backend that is solely used with unit-testing query functionality
// in both the server API and backend API.
//
// The backend will call the backend query API in setting below:
// name, byte_size, memory_type, memory_type_id (refer backend API for detail)
// "OUTPUT0", nullptr, CPU_PINNED, 1
// "OUTPUT1", nullptr, CPU_PINNED, 1
// Then it will call the alloc function (TRITONBACKEND_OutputBuffer) with
// the returned value accordingly. If 'byte_size' is nullptr, it creates the
// outputs with UINT8 type and shape [2].
// The backend will read environment variables for different query behavior
// 'TEST_ANONYMOUS': the backend will call the query API only once with 'name'
//                   set to nullptr
// 'TEST_BYTE_SIZE': the backend will call the query API once with 'byte_size'
//                   set to the variable value, and the outputs will be created
//                   with UINT8 and shape [byte_size]. If 'TEST_ANONYMOUS' is
//                   also specified, the outputs will have shape [byte_size / 2]
// 'TEST_FAIL_WITH_QUERY_RESULT' : the query results will be formatted to string
//                                 and returned as error message.

#define RESPOND_IF_ERROR(RESPONSE, X)                                   \
  do {                                                                  \
    if (RESPONSE != nullptr) {                                          \
      TRITONSERVER_Error* err__ = (X);                                  \
      if (err__ != nullptr) {                                           \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                RESPONSE, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err__), \
            "failed to send error response");                           \
        TRITONSERVER_ErrorDelete(err__);                                \
      }                                                                 \
    }                                                                   \
  } while (false)

extern "C" {

// Implementing TRITONBACKEND_ModelInstanceExecute is required.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Read environment variables
  const char* anonymous_str = getenv("TEST_ANONYMOUS");
  const char* byte_size_str = getenv("TEST_BYTE_SIZE");
  const char* fail_str = getenv("TEST_FAIL_WITH_QUERY_RESULT");
  bool anonymous = (anonymous_str != nullptr);
  size_t byte_size = 2;
  size_t query_byte_size = byte_size;
  size_t* byte_size_ptr = nullptr;
  if (byte_size_str != nullptr) {
    byte_size = atoi(byte_size_str);
    query_byte_size = byte_size;
    if (anonymous) {
      byte_size /= 2;
    }
    byte_size_ptr = &query_byte_size;
  }

  for (uint32_t r = 0; r < request_count; ++r) {
    std::string log_message;

    TRITONBACKEND_Request* request = requests[r];
    TRITONBACKEND_Response* response = nullptr;

    // Query before creating output
    std::vector<const char*> names;
    if (anonymous) {
      names.emplace_back(nullptr);
    } else {
      names = {"OUTPUT0", "OUTPUT1"};
    }
    std::vector<TRITONSERVER_MemoryType> types{TRITONSERVER_MEMORY_CPU_PINNED,
                                               TRITONSERVER_MEMORY_CPU_PINNED};
    std::vector<int64_t> type_ids{1, 1};
    for (size_t i = 0; i < names.size(); ++i) {
      auto err = TRITONBACKEND_RequestOutputBufferProperties(
          request, names[i], byte_size_ptr, &types[i], &type_ids[i]);
      if (err != nullptr) {
        RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
        RESPOND_IF_ERROR(response, err);
        break;
      }
      if (fail_str != nullptr) {
        log_message += ((names[i] == nullptr) ? "NULL" : names[i]);
        switch (types[i]) {
          case TRITONSERVER_MEMORY_CPU:
            log_message += " CPU ";
            break;
          case TRITONSERVER_MEMORY_CPU_PINNED:
            log_message += " CPU_PINNED ";
            break;
          case TRITONSERVER_MEMORY_GPU:
            log_message += " GPU ";
            break;
        }
        log_message += (std::to_string(type_ids[i]) + "; ");
      }
    }

    // If response is not nullptr, some error is returned from query API and
    // the response has been sent
    if (response == nullptr) {
      if (names.size() == 1) {
        names = {"OUTPUT0", "OUTPUT1"};
        types[1] = types[0];
        type_ids[1] = type_ids[0];
      }
      std::vector<int64_t> shape{(int64_t)byte_size};

      TRITONBACKEND_Response* response;
      RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
      TRITONSERVER_Error* err = nullptr;
      if (fail_str == nullptr) {
        for (size_t i = 0; i < names.size(); ++i) {
          TRITONBACKEND_Output* output;
          err = TRITONBACKEND_ResponseOutput(
              response, &output, names[i], TRITONSERVER_TYPE_UINT8,
              shape.data(), 1);
          if (err != nullptr) {
            break;
          }
          void* output_buffer;
          err = TRITONBACKEND_OutputBuffer(
              output, &output_buffer, byte_size, &types[i], &type_ids[i]);
          if (err != nullptr) {
            break;
          }
          // Do nothing with the buffer as we don't care
        }
      } else {
        // Use an uncommon error code
        err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNKNOWN, log_message.c_str());
      }

      TRITONBACKEND_ResponseSend(
          response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err);
    }

    TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL);
  }

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::query
