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

#ifdef TRITON_BIG_ENDIAN

#define TRITONJSON_STATUSTYPE TRITONSERVER_Error*
#define TRITONJSON_STATUSRETURN(M) \
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (M).c_str())
#define TRITONJSON_STATUSSUCCESS nullptr
#include "triton/common/triton_json.h"

#endif

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

#ifdef TRITON_BIG_ENDIAN

uint16_t
bswap(const uint16_t input)
{
  return __builtin_bswap16(input);
}
uint32_t
bswap(const uint32_t input)
{
  return __builtin_bswap32(input);
}

uint32_t
bswap_length(
    const uint32_t length, const bool host_to_little_endian,
    uint32_t* host_length)
{
  uint32_t swapped = bswap(length);
  if (host_to_little_endian) {
    *host_length = length;
  } else {
    *host_length = swapped;
  }
  return swapped;
}

uint64_t
bswap(const uint64_t input)
{
  return __builtin_bswap64(input);
}

template <typename T>
void
SwapEndian(char* buffer, size_t size)
{
  for (size_t offset = 0; offset < size;
       offset += sizeof(T), buffer += sizeof(T)) {
    T* temp = reinterpret_cast<T*>(buffer);
    *temp = bswap(*temp);
  }
}

void
bswap_length_partial(
    char* buffer, size_t size, const bool host_to_little_endian,
    uint32_t* host_length, std::vector<char*>& partial_result)
{
  *host_length = 0;
  
  if (partial_result.size()) {
    size_t offset = 0;
    for (; (partial_result.size() < sizeof(uint32_t)) && (offset < size);
         ++offset, ++buffer) {
      partial_result.emplace_back(buffer);
    }
    if (partial_result.size() == sizeof(uint32_t)) {
      char* host_length_buffer = reinterpret_cast<char*>(host_length);
      for (size_t left_index = 0, right_index = sizeof(uint32_t) - 1;
           left_index < right_index; left_index += 1, right_index -= 1) {
        uint8_t temp = *partial_result[right_index];
        *partial_result[right_index] = *partial_result[left_index];
        *partial_result[left_index] = temp;
        if (host_to_little_endian) {
          host_length_buffer[right_index] = *partial_result[left_index];
          host_length_buffer[left_index] = *partial_result[right_index];
        } else {
          host_length_buffer[right_index] = *partial_result[right_index];
          host_length_buffer[left_index] = *partial_result[left_index];
        }
      }
      partial_result.clear();
      return;
    }
    return;
  }

  if (size >= sizeof(uint32_t)) {
    uint32_t* value = reinterpret_cast<uint32_t*>(buffer);
    *value = bswap_length(*value, host_to_little_endian, host_length);
    return;
  }

  for (; size; --size, ++buffer) {
    partial_result.emplace_back(buffer);
  }
}

template <typename T>
void
SwapEndian(char* buffer, size_t size, std::vector<char*>& partial_result)
{
  size_t offset = 0;

  if (partial_result.size()) {
    for (; (partial_result.size() < sizeof(T)) && (offset < size);
         ++offset, ++buffer) {
      partial_result.emplace_back(buffer);
    }
    if (partial_result.size() == sizeof(T)) {
      for (size_t left_index = 0, right_index = sizeof(T) - 1;
           left_index < right_index; left_index += 1, right_index -= 1) {
        uint8_t temp = *partial_result[right_index];
        *partial_result[right_index] = *partial_result[left_index];
        *partial_result[left_index] = temp;
      }
      partial_result.clear();
    }
  }

  T* value = reinterpret_cast<T*>(buffer);
  for (; size - offset >= sizeof(T); offset += sizeof(T), ++value) {
    *value = bswap(*value);
  }

  buffer = reinterpret_cast<char*>(value);

  for (; offset < size; ++offset, ++buffer) {
    partial_result.emplace_back(buffer);
  }
}


void
SwapEndian(
    TRITONSERVER_DataType datatype, char* buffer, size_t size,
    bool host_to_little_endian)
{
  switch (datatype) {
    case TRITONSERVER_TYPE_UINT16:
    case TRITONSERVER_TYPE_INT16:
    case TRITONSERVER_TYPE_FP16: {
      SwapEndian<uint16_t>(buffer, size);
      break;
    }
    case TRITONSERVER_TYPE_UINT32:
    case TRITONSERVER_TYPE_INT32:
    case TRITONSERVER_TYPE_FP32: {
      SwapEndian<uint32_t>(buffer, size);
      break;
    }
    case TRITONSERVER_TYPE_UINT64:
    case TRITONSERVER_TYPE_INT64:
    case TRITONSERVER_TYPE_FP64: {
      SwapEndian<uint64_t>(buffer, size);
      break;
    }
    case TRITONSERVER_TYPE_BYTES: {
      size_t offset = 0;
      uint32_t host_len;
      while (offset < size) {
        uint32_t* temp = reinterpret_cast<uint32_t*>(buffer + offset);
        *temp = bswap_length(*temp, host_to_little_endian, &host_len);
        offset += sizeof(uint32_t) + host_len;
      }
      break;
    }
    default: {
    }
  }
}

void
SwapEndian(
    TRITONSERVER_DataType datatype, char* buffer, size_t size,
    bool host_to_little_endian, std::vector<char*>& partial_result,
    size_t& offset)
{
  switch (datatype) {
    case TRITONSERVER_TYPE_UINT16:
    case TRITONSERVER_TYPE_INT16:
    case TRITONSERVER_TYPE_FP16: {
      SwapEndian<uint16_t>(buffer, size, partial_result);
      break;
    }
    case TRITONSERVER_TYPE_UINT32:
    case TRITONSERVER_TYPE_INT32:
    case TRITONSERVER_TYPE_FP32: {
      SwapEndian<uint32_t>(buffer, size, partial_result);
      break;
    }
    case TRITONSERVER_TYPE_UINT64:
    case TRITONSERVER_TYPE_INT64:
    case TRITONSERVER_TYPE_FP64: {
      SwapEndian<uint64_t>(buffer, size, partial_result);
      break;
    }
    case TRITONSERVER_TYPE_BYTES: {
      uint32_t host_len=0;
      while (offset < size) {
	size_t incoming_partial_result_size = partial_result.size();
        bswap_length_partial(
            buffer + offset, size - offset, host_to_little_endian, &host_len,
            partial_result);
        if (partial_result.size() == 0) {
          offset += host_len + sizeof(uint32_t)-incoming_partial_result_size;
        } else {
          offset = size;
        }
      }
      offset -= size;
      break;
    }
    default: {
    }
  }
}

void
HostToLittleEndian(TRITONSERVER_DataType datatype, char* base, size_t byte_size)
{
  SwapEndian(datatype, base, byte_size, true);
}

void
LittleEndianToHost(TRITONSERVER_DataType datatype, char* base, size_t byte_size)
{
  SwapEndian(datatype, base, byte_size, false);
}

void
LittleEndianToHost(
    TRITONSERVER_DataType datatype, char* base, size_t byte_size,
    std::vector<char*>& partial_result, size_t& offset)
{
  SwapEndian(datatype, base, byte_size, false, partial_result, offset);
}

TRITONSERVER_DataType
GetDataTypeForRawInput(
    TRITONSERVER_Server* server, const std::string& model_name,
    const int64_t model_version)
{
  TRITONSERVER_Message* message = nullptr;
  triton::common::TritonJson::Value document;
  triton::common::TritonJson::Value inputs;
  triton::common::TritonJson::Value input;
  const char* datatype = "";
  size_t datatype_length = 0;
  const char* buffer;
  size_t byte_size;

  RETURN_VAL_IF_ERR(
      TRITONSERVER_ServerModelMetadata(
          server, model_name.c_str(), model_version, &message),
      TRITONSERVER_TYPE_INVALID);

  RETURN_VAL_IF_ERR(
      TRITONSERVER_MessageSerializeToJson(message, &buffer, &byte_size),
      TRITONSERVER_TYPE_INVALID);

  RETURN_VAL_IF_ERR(
      document.Parse(buffer, byte_size), TRITONSERVER_TYPE_INVALID);

  RETURN_VAL_IF_ERR(
      document.MemberAsArray("inputs", &inputs), TRITONSERVER_TYPE_INVALID);

  RETURN_VAL_IF_ERR(inputs.IndexAsObject(0, &input), TRITONSERVER_TYPE_INVALID);

  RETURN_VAL_IF_ERR(
      input.MemberAsString("datatype", &datatype, &datatype_length),
      TRITONSERVER_TYPE_INVALID);

  return TRITONSERVER_StringToDataType(datatype);
}

#endif  // TRITON_BIG_ENDIAN

}}  // namespace triton::server
