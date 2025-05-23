// Copyright 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common.h"

#include <algorithm>
#include <climits>
#include <iterator>

#include "restricted_features.h"
#include "triton/core/tritonserver.h"

extern "C" {
#include <b64/cdecode.h>
}

namespace triton { namespace server {

TRITONSERVER_Error*
GetModelVersionFromString(const std::string& version_string, int64_t* version)
{
  if (version_string.empty()) {
    *version = -1;
    return nullptr;  // success
  }

  try {
    *version = std::stol(version_string);
  }
  catch (std::exception& e) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "failed to get model version from specified version string '" +
            version_string + "' (details: " + e.what() +
            "), version should be an integral value > 0")
            .c_str());
  }

  if (*version < 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "invalid model version specified '" + version_string +
            "' , version should be an integral value > 0")
            .c_str());
  }

  return nullptr;  // success
}

std::string
GetEnvironmentVariableOrDefault(
    const std::string& variable_name, const std::string& default_value)
{
  const char* value = getenv(variable_name.c_str());
  return value ? value : default_value;
}

std::string
ShapeToString(const int64_t* dims, const size_t dims_count)
{
  bool first = true;

  std::string str("[");
  for (size_t i = 0; i < dims_count; ++i) {
    const int64_t dim = dims[i];
    if (!first) {
      str += ",";
    }
    str += std::to_string(dim);
    first = false;
  }

  str += "]";
  return str;
}

std::string
ShapeToString(const std::vector<int64_t>& shape)
{
  return ShapeToString(shape.data(), shape.size());
}

int64_t
GetElementCount(const std::vector<int64_t>& dims)
{
  bool first = true;
  int64_t cnt = 0;
  for (auto dim : dims) {
    if (dim == WILDCARD_DIM) {
      return -1;
    } else if (dim < 0) {  // invalid dim
      return -2;
    } else if (dim == 0) {
      return 0;
    }

    if (first) {
      cnt = dim;
      first = false;
    } else {
      // Check for overflow before multiplication
      if (cnt > (INT64_MAX / dim)) {
        return -3;
      }
      cnt *= dim;
    }
  }

  return cnt;
}

bool
Contains(const std::vector<std::string>& vec, const std::string& str)
{
  return std::find(vec.begin(), vec.end(), str) != vec.end();
}

TRITONSERVER_Error*
DecodeBase64(
    const char* input, size_t input_len, std::vector<char>& decoded_data,
    size_t& decoded_size, const std::string& name)
{
  if (input_len > static_cast<size_t>(INT_MAX)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        ("'" + name + "' exceeds the maximum allowed data size limit INT_MAX")
            .c_str());
  }

  // The decoded size cannot be larger than the input
  decoded_data.resize(input_len + 1);
  base64_decodestate state;
  base64_init_decodestate(&state);

  decoded_size =
      base64_decode_block(input, input_len, decoded_data.data(), &state);

  return nullptr;
}

}}  // namespace triton::server
