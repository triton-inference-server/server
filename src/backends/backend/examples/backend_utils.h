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

#include <iostream>
#include <string>
#include <vector>
#include "src/backends/backend/tritonbackend.h"

#define TRITONJSON_STATUSTYPE TRITONSERVER_Error*
#define TRITONJSON_STATUSRETURN(M) \
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (M).c_str())
#define TRITONJSON_STATUSSUCCESS nullptr
#include "src/core/json.h"

namespace nvidia { namespace inferenceserver { namespace backend {

#define LOG_IF_ERROR(X, MSG)                                               \
  do {                                                                     \
    TRITONSERVER_Error* err__ = (X);                                       \
    if (err__ != nullptr) {                                                \
      TRITONSERVER_LogMessage(                                             \
          TRITONSERVER_LOG_INFO, __FILE__, __LINE__,                       \
          (std::string(MSG) + ": " + TRITONSERVER_ErrorCodeString(err__) + \
           " - " + TRITONSERVER_ErrorMessage(err__))                       \
              .c_str());                                                   \
      TRITONSERVER_ErrorDelete(err__);                                     \
    }                                                                      \
  } while (false)

#define RETURN_ERROR_IF_FALSE(P, C, MSG)              \
  do {                                                \
    if (!(P)) {                                       \
      return TRITONSERVER_ErrorNew(C, (MSG).c_str()); \
    }                                                 \
  } while (false)

#define RETURN_IF_ERROR(X)           \
  do {                               \
    TRITONSERVER_Error* err__ = (X); \
    if (err__ != nullptr) {          \
      return err__;                  \
    }                                \
  } while (false)


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

}}}  // namespace nvidia::inferenceserver::backend
