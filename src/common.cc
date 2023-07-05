// Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <iterator>

#include "triton/core/tritonserver.h"

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

int64_t
GetElementCount(const std::vector<int64_t>& dims)
{
  bool first = true;
  int64_t cnt = 0;
  for (auto dim : dims) {
    if (dim == WILDCARD_DIM) {
      return -1;
    }

    if (first) {
      cnt = dim;
      first = false;
    } else {
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

std::string
Join(const std::vector<std::string>& vec, const std::string& delim)
{
  std::stringstream ss;
  std::copy(
      vec.begin(), vec.end(),
      std::ostream_iterator<std::string>(ss, delim.c_str()));
  return ss.str();
}


uint16_t
bswap(uint16_t input)
{
  return __builtin_bswap16(input);
}
uint32_t
bswap(uint32_t input)
{
  return __builtin_bswap32(input);
}
uint64_t
bswap(uint64_t input)
{
  return __builtin_bswap64(input);
}

template <typename T>
TRITONSERVER_Error*
SwapEndian(const char* buffer, size_t size)
{
  if (size % sizeof(T)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Number of bytes is not multiple of type size. Can't swap endianness.");
  }
  for (; size > 0; size -= sizeof(T), buffer += sizeof(T)) {
    T* temp = const_cast<T*>(reinterpret_cast<const T*>(buffer));
    *temp = bswap(*temp);
  }
  return nullptr;
}

void
LittleEndianToHost(
    struct evbuffer_iovec* buffers, int buffer_index, int total_buffers,
    int byte_size, TRITONSERVER_DataType datatype)
{
  size_t offset = 0;
  while ((byte_size > 0) && (buffer_index < total_buffers)) {
    char* base = static_cast<char*>(buffers[buffer_index].iov_base) + offset;
    size_t base_size = std::min(byte_size, buffers[buffer_index].iov_len);
    SwapEndian(datatype, base, base_size, offset);
  }
}


template <typename T>
TRITONSERVER_Error*
SwapEndian(
    const char* buffer, size_t size, const char* next_buffer, size_t next_size,
    size_t* offset, T* last_value)
{
  size_t left_over = size % sizeof(T);
  size_t count = size / sizeof(T);
  size_t remaining = sizeof(T) - left_over;
  *offset = remaining;
  if ((remaining) && (remaining > next_size)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Number of bytes is not multiple of type size. Can't swap endianness.");
  }

  for (; count > 0; --count, buffer += sizeof(T)) {
    T* temp = const_cast<T*>(reinterpret_cast<const T*>(buffer));
    *temp = bswap(*temp);
    *last_value = *temp;
  }

  if (remaining) {
    T temp;
    char* temp_buffer = reinterpret_cast<char*> & temp;
    for (uint32 i = 0; i < left_over; ++i, ++temp_buffer, ++buffer) {
      *temp_buffer = *buffer;
    }
    for (uint32 i = 0; i < remaining; ++i, ++temp_buffer, ++next_buffer) {
      *temp_buffer = *next_buffer;
    }

    temp = bswap(temp);

    *last_value = temp;

    for (uint32 i = 0, buffer -= left_over; i < left;
         ++i, ++temp_buffer, ++buffer) {
      *const_cast<char*>(buffer) = *temp_buffer;
    }

    for (uint32 i = 0, next_buffer -= remaining; i < remaining;
         ++i, ++temp_buffer, ++next_buffer) {
      *const_cast<char*>(next_buffer) = *temp_buffer;
    }
  }
  return nullptr;
}

TRITONSERVER_Error*
SwapEndian(TRITONSERVER_DataType datatype, std::string& serialized)
{
  return SwapEndian(datatype, serialized.c_str(), serialized.length());
}

  TRITONSERVER_Error* SwapEndian(
      TRITONSERVER_DataType datatype, const char* buffer, size_t size)
  {
    switch (datatype) {
      case TRITONSERVER_TYPE_UINT16:
      case TRITONSERVER_TYPE_INT16:
      case TRITONSERVER_TYPE_FP16: {
        return SwapEndian<uint16_t>(buffer, size);
      }
      case TRITONSERVER_TYPE_UINT32:
      case TRITONSERVER_TYPE_INT32:
      case TRITONSERVER_TYPE_FP32: {
        return SwapEndian<uint32_t>(buffer, size);
      }
      case TRITONSERVER_TYPE_UINT64:
      case TRITONSERVER_TYPE_INT64:
      case TRITONSERVER_TYPE_FP64: {
        return SwapEndian<uint64_t>(buffer, size);
      }
      default: {
      }
    }
    return nullptr;
  }


  TRITONSERVER_Error*
  SwapEndian(
      TRITONSERVER_DataType datatype, const char* buffer, size_t size,
      const char* next_buffer, size_t next_size, size_t* offset,
      void* last_value)
  {
    switch (datatype) {
      case TRITONSERVER_TYPE_UINT16:
      case TRITONSERVER_TYPE_INT16:
      case TRITONSERVER_TYPE_FP16: {
        return SwapEndian<uint16_t>(
            buffer, size, next_buffer, next_size, offset,
            reinterpret_cast<uint16_t*> last_value);
      }
      case TRITONSERVER_TYPE_UINT32:
      case TRITONSERVER_TYPE_INT32:
      case TRITONSERVER_TYPE_FP32: {
        return SwapEndian<uint32_t>(
            buffer, size, next_buffer, next_size, offset,
            reinterpret_cast<uint32_t*> last_value);
      }
      case TRITONSERVER_TYPE_UINT64:
      case TRITONSERVER_TYPE_INT64:
      case TRITONSERVER_TYPE_FP64: {
        return SwapEndian<uint64_t>(
            buffer, size, next_buffer, next_size, offset,
            reinterpret_cast<uint64_t*> last_value);
      }
      default: {
      }
    }
    return nullptr;
  }

  TRITONSERVER_Error* SwapEndian(
      TRITONSERVER_DataType datatype, const char* buffer, size_t size,
      const char* next_buffer, size_t next_size)
  {
  }
}
}  // namespace server
