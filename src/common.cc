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

/// Returns uint16_t input with byte order swapped
///
/// \param[in] input 
/// \return input with byte order swapped
///
uint16_t
bswap(const uint16_t input)
{
  return __builtin_bswap16(input);
}

/// Returns uint32_t input with byte order swapped
///
/// \param[in] input
/// \return input with byte order swapped
///
uint32_t
bswap(const uint32_t input)
{
  return __builtin_bswap32(input);
}

/// Returns uint64_t input with byte order swapped
///
/// \param[in] input
/// \return input with byte order swapped
///
uint64_t
bswap(const uint64_t input)
{
  return __builtin_bswap64(input);
}

/// Returns uint32_t length with byte order swapped and
/// returns length in host byte order.
///
/// \param[in] length
/// \param[in] host_byte_order whether original length is in host byte order
/// \param[out] host_length length in host byte order
/// \return length with byte order swapped
///
uint32_t
bswap_length(
    const uint32_t length, const bool host_byte_order, uint32_t* host_length)
{
  uint32_t swapped = bswap(length);
  *host_length = host_byte_order ? length : swapped;
  return swapped;
}

/// Swaps length byte order in place with support for non coniguous
/// arrays and returns length in host byte order. For
/// non-contiguous arrays the initial call must pass an empty partial result.
///
/// \param[in, out] base pointer to array with length
/// \param[in] byte_size size of array
/// \param[in] host_byte_order whether original length is in host byte order
/// \param[out] host_length length in host byte order
/// \param[in, out] partial_result vector for partial results.
///
void
bswap_length_partial(
    char* base, size_t byte_size, const bool host_byte_order,
    uint32_t* host_length, std::vector<char*>& partial_result)
{
  *host_length = 0;

  if (partial_result.size()) {
    for (size_t offset = 0,
                remaining = sizeof(uint32_t) - partial_result.size();
         (remaining > 0) && (offset < byte_size);
         ++offset, ++base, --remaining) {
      partial_result.emplace_back(base);
    }
    if (partial_result.size() == sizeof(uint32_t)) {
      char* host_length_buffer = reinterpret_cast<char*>(host_length);
      for (size_t left_index = 0, right_index = sizeof(uint32_t) - 1;
           left_index < right_index; ++left_index, --right_index) {
        char temp = *partial_result[right_index];
        *partial_result[right_index] = *partial_result[left_index];
        *partial_result[left_index] = temp;
        if (host_byte_order) {
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

  if (byte_size >= sizeof(uint32_t)) {
    uint32_t* value = reinterpret_cast<uint32_t*>(base);
    *value = bswap_length(*value, host_byte_order, host_length);
    return;
  }

  for (; byte_size; --byte_size, ++base) {
    partial_result.emplace_back(base);
  }
}

/// Swaps byte order in place
///
/// \param[in, out] base pointer to array of data type elements to convert
/// \param[in] byte_size size of array in bytes
///
template <typename T>
void
SwapEndian(char* base, size_t byte_size)
{
  T* value = reinterpret_cast<T*>(base);
  for (size_t offset = 0; offset < byte_size; offset += sizeof(T), ++value) {
    *value = bswap(*value);
  }
}

/// Swaps byte order in place with support for non-contiguous arrays. For
/// non-contiguous arrays the initial call must pass an empty partial result.
///
/// \param[in, out] base pointer to array of data type elements to convert
/// \param[in] byte_size size of array in bytes
/// \param[in, out] partial_result vector to store partial results
///
template <typename T>
void
SwapEndian(char* base, size_t byte_size, std::vector<char*>& partial_result)
{
  size_t offset = 0;

  if (partial_result.size()) {
    for (size_t remaining = sizeof(T) - partial_result.size();
         (remaining > 0) && (offset < byte_size);
         ++offset, ++base, --remaining) {
      partial_result.emplace_back(base);
    }
    if (partial_result.size() == sizeof(T)) {
      for (size_t left_index = 0, right_index = sizeof(T) - 1;
           left_index < right_index; ++left_index, --right_index) {
        char temp = *partial_result[right_index];
        *partial_result[right_index] = *partial_result[left_index];
        *partial_result[left_index] = temp;
      }
      partial_result.clear();
    }
  }

  T* value = reinterpret_cast<T*>(base);
  for (; byte_size - offset >= sizeof(T); offset += sizeof(T), ++value) {
    *value = bswap(*value);
  }

  base = reinterpret_cast<char*>(value);
  for (; offset < byte_size; ++offset, ++base) {
    partial_result.emplace_back(base);
  }
}


/// Swaps byte order in place.
///
/// \param[in] datatype data type of array
/// \param[in, out] base pointer to array of data type elements to convert
/// \param[in] byte_size size of array in bytes
/// \param[in] host_byte_order whether original array is in host byte
/// order
///
void
SwapEndian(
    TRITONSERVER_DataType datatype, char* base, size_t byte_size,
    bool host_byte_order)
{
  switch (datatype) {
    case TRITONSERVER_TYPE_UINT16:
    case TRITONSERVER_TYPE_INT16:
    case TRITONSERVER_TYPE_FP16: {
      SwapEndian<uint16_t>(base, byte_size);
      break;
    }
    case TRITONSERVER_TYPE_UINT32:
    case TRITONSERVER_TYPE_INT32:
    case TRITONSERVER_TYPE_FP32: {
      SwapEndian<uint32_t>(base, byte_size);
      break;
    }
    case TRITONSERVER_TYPE_UINT64:
    case TRITONSERVER_TYPE_INT64:
    case TRITONSERVER_TYPE_FP64: {
      SwapEndian<uint64_t>(base, byte_size);
      break;
    }
    case TRITONSERVER_TYPE_BYTES: {
      size_t next_offset = 0;
      uint32_t host_length = 0;
      while (next_offset < byte_size) {
        uint32_t* length = reinterpret_cast<uint32_t*>(base + next_offset);
        *length = bswap_length(*length, host_byte_order, &host_length);
        next_offset += sizeof(uint32_t) + host_length;
      }
      break;
    }
    default: {
    }
  }
}

/// Swaps byte order in place with support for non-contiguous arrays. For
/// non-contiguous arrays the initial call must pass an empty partial result and
/// 0 for the initial next_offset. Subsequent calls should pass the returned
/// values without modification.
///
/// \param[in] datatype data type of array
/// \param[in, out] base pointer to array of data type elements to convert
/// \param[in] byte_size size of array in bytes
/// \param[in] host_byte_order whether original array is in host byte
/// order
/// \param[in, out] partial_result vector to store partial results
/// \param[in, out] next_offset intermediate value used for storing next offset
/// for BYTES data type
///
void
SwapEndian(
    TRITONSERVER_DataType datatype, char* base, size_t byte_size,
    bool host_byte_order, std::vector<char*>& partial_result,
    size_t& next_offset)
{
  switch (datatype) {
    case TRITONSERVER_TYPE_UINT16:
    case TRITONSERVER_TYPE_INT16:
    case TRITONSERVER_TYPE_FP16: {
      SwapEndian<uint16_t>(base, byte_size, partial_result);
      break;
    }
    case TRITONSERVER_TYPE_UINT32:
    case TRITONSERVER_TYPE_INT32:
    case TRITONSERVER_TYPE_FP32: {
      SwapEndian<uint32_t>(base, byte_size, partial_result);
      break;
    }
    case TRITONSERVER_TYPE_UINT64:
    case TRITONSERVER_TYPE_INT64:
    case TRITONSERVER_TYPE_FP64: {
      SwapEndian<uint64_t>(base, byte_size, partial_result);
      break;
    }
    case TRITONSERVER_TYPE_BYTES: {
      uint32_t host_length = 0;
      while (next_offset < byte_size) {
        size_t partial_offset = partial_result.size();
        bswap_length_partial(
            base + next_offset, byte_size - next_offset, host_byte_order,
            &host_length, partial_result);
        if (partial_result.size() == 0) {
          next_offset += sizeof(uint32_t) - partial_offset + host_length;
        } else {
          next_offset = 0;
          return;
        }
      }
      next_offset -= byte_size;
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
    std::vector<char*>& partial_result, size_t& next_offset)
{
  SwapEndian(datatype, base, byte_size, false, partial_result, next_offset);
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
