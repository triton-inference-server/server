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

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <string>

#include "src/clients/c++/experimental_api_v2/perf_client/perf_utils.h"

ProtocolType
ParseProtocol(const std::string& str)
{
  std::string protocol(str);
  std::transform(protocol.begin(), protocol.end(), protocol.begin(), ::tolower);
  if (protocol == "http") {
    return ProtocolType::HTTP;
  } else if (protocol == "grpc") {
    return ProtocolType::GRPC;
  }
  return ProtocolType::UNKNOWN;
}

nic::Error
ReadFile(const std::string& path, std::vector<char>* contents)
{
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in) {
    return nic::Error("failed to open file '" + path + "'");
  }

  in.seekg(0, std::ios::end);

  int file_size = in.tellg();
  if (file_size > 0) {
    contents->resize(file_size);
    in.seekg(0, std::ios::beg);
    in.read(&(*contents)[0], contents->size());
  }

  in.close();

  // If size is invalid, report after ifstream is closed
  if (file_size < 0) {
    return nic::Error("failed to get size for file '" + path + "'");
  } else if (file_size == 0) {
    return nic::Error("file '" + path + "' is empty");
  }

  return nic::Error::Success;
}

nic::Error
ReadTextFile(const std::string& path, std::vector<std::string>* contents)
{
  std::ifstream in(path);
  if (!in) {
    return nic::Error("failed to open file '" + path + "'");
  }

  std::string current_string;
  while (std::getline(in, current_string)) {
    contents->push_back(current_string);
  }
  in.close();

  if (contents->size() == 0) {
    return nic::Error("file '" + path + "' is empty");
  }
  return nic::Error::Success;
}

nic::Error
ReadTimeIntervalsFile(
    const std::string& path, std::vector<std::chrono::nanoseconds>* contents)
{
  std::ifstream in(path);
  if (!in) {
    return nic::Error("failed to open file '" + path + "'");
  }

  std::string current_string;
  while (std::getline(in, current_string)) {
    std::chrono::nanoseconds curent_time_interval_ns(
        std::stol(current_string) * 1000);
    contents->push_back(curent_time_interval_ns);
  }
  in.close();

  if (contents->size() == 0) {
    return nic::Error("file '" + path + "' is empty");
  }
  return nic::Error::Success;
}

bool
IsDirectory(const std::string& path)
{
  struct stat s;
  if (stat(path.c_str(), &s) == 0 && (s.st_mode & S_IFDIR)) {
    return true;
  } else {
    return false;
  }
}

bool
IsFile(const std::string& complete_path)
{
  struct stat s;
  if (stat(complete_path.c_str(), &s) == 0 && (s.st_mode & S_IFREG)) {
    return true;
  } else {
    return false;
  }
}

int64_t
ByteSize(const std::vector<int64_t>& shape, const std::string& datatype)
{
  int one_element_size;
  if ((datatype.compare("BOOL") == 0) || (datatype.compare("INT8") == 0) ||
      (datatype.compare("UINT8") == 0)) {
    one_element_size = 1;
  } else if (
      (datatype.compare("INT16") == 0) || (datatype.compare("UINT16") == 0) ||
      (datatype.compare("FP16") == 0)) {
    one_element_size = 2;
  } else if (
      (datatype.compare("INT32") == 0) || (datatype.compare("UINT32") == 0) ||
      (datatype.compare("FP32") == 0)) {
    one_element_size = 4;
  } else if (
      (datatype.compare("INT64") == 0) || (datatype.compare("UINT64") == 0) ||
      (datatype.compare("FP64") == 0)) {
    one_element_size = 8;
  } else {
    return -1;
  }

  return (one_element_size * ElementCount(shape));
}

int64_t
ElementCount(const std::vector<int64_t>& shape)
{
  int64_t count = 1;
  for (const auto dim : shape) {
    count *= dim;
  }
  return count;
}


void
SerializeStringTensor(
    std::vector<std::string> string_tensor, std::vector<char>* serialized_data)
{
  std::string serialized = "";
  for (auto s : string_tensor) {
    uint32_t len = s.size();
    serialized.append(reinterpret_cast<const char*>(&len), sizeof(uint32_t));
    serialized.append(s);
  }

  std::copy(
      serialized.begin(), serialized.end(),
      std::back_inserter(*serialized_data));
}


nic::Error
SerializeExplicitTensor(
    const rapidjson::Value& tensor, const std::string& dt,
    std::vector<char>* decoded_data)
{
  if (dt.compare("BYTES") == 0) {
    std::string serialized = "";
    for (const auto& value : tensor.GetArray()) {
      if (!value.IsString()) {
        return nic::Error("unable to find string data in json");
      }
      std::string element(value.GetString());
      uint32_t len = element.size();
      serialized.append(reinterpret_cast<const char*>(&len), sizeof(uint32_t));
      serialized.append(element);
    }
    std::copy(
        serialized.begin(), serialized.end(),
        std::back_inserter(*decoded_data));
  } else {
    for (const auto& value : tensor.GetArray()) {
      if (dt.compare("BOOL") == 0) {
        if (!value.IsBool()) {
          return nic::Error("unable to find bool data in json");
        }
        bool element(value.GetBool());
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(bool));
      } else if (dt.compare("UINT8") == 0) {
        if (!value.IsUint()) {
          return nic::Error("unable to find uint8_t data in json");
        }
        uint8_t element(static_cast<uint8_t>(value.GetUint()));
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(uint8_t));
      } else if (dt.compare("INT8") == 0) {
        if (!value.IsInt()) {
          return nic::Error("unable to find int8_t data in json");
        }
        int8_t element(static_cast<int8_t>(value.GetInt()));
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(int8_t));
      } else if (dt.compare("UINT16") == 0) {
        if (!value.IsUint()) {
          return nic::Error("unable to find uint16_t data in json");
        }
        uint16_t element(static_cast<uint16_t>(value.GetUint()));
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(uint16_t));
      } else if (dt.compare("INT16") == 0) {
        if (!value.IsInt()) {
          return nic::Error("unable to find int16_t data in json");
        }
        int16_t element(static_cast<int16_t>(value.GetInt()));
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(int16_t));
      } else if (dt.compare("FP16") == 0) {
        return nic::Error(
            "Can not use explicit tensor description for fp16 datatype");
      } else if (dt.compare("UINT32") == 0) {
        if (!value.IsUint()) {
          return nic::Error("unable to find uint32_t data in json");
        }
        uint32_t element(value.GetUint());
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(uint32_t));
      } else if (dt.compare("INT32") == 0) {
        if (!value.IsInt()) {
          return nic::Error("unable to find int32_t data in json");
        }
        int32_t element(value.GetInt());
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(int32_t));
      } else if (dt.compare("FP32") == 0) {
        if (!value.IsDouble()) {
          return nic::Error("unable to find float data in json");
        }
        float element(value.GetFloat());
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(float));
      } else if (dt.compare("UINT64") == 0) {
        if (!value.IsUint64()) {
          return nic::Error("unable to find uint64_t data in json");
        }
        uint64_t element(value.GetUint64());
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(uint64_t));
      } else if (dt.compare("INT64") == 0) {
        if (!value.IsInt64()) {
          return nic::Error("unable to find int64_t data in json");
        }
        int64_t element(value.GetInt64());
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(int64_t));
      } else if (dt.compare("FP64") == 0) {
        if (!value.IsDouble()) {
          return nic::Error("unable to find fp64 data in json");
        }
        double element(value.GetDouble());
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(double));
      }
    }
  }
  return nic::Error::Success;
}

std::string
GetRandomString(const int string_length)
{
  std::mt19937_64 gen{std::random_device()()};
  std::uniform_int_distribution<size_t> dist{0, character_set.length() - 1};
  std::string random_string;
  std::generate_n(std::back_inserter(random_string), string_length, [&] {
    return character_set[dist(gen)];
  });
  return random_string;
}

std::string
ShapeVecToString(const std::vector<int64_t> shape_vec)
{
  bool first = true;
  std::string str("[");
  for (const auto& value : shape_vec) {
    if (!first) {
      str += ",";
    }
    str += std::to_string(value);
    first = false;
  }

  str += "]";
  return str;
}

std::string
ShapeTensorValuesToString(const int* data_ptr, const int count)
{
  bool first = true;
  std::string str("[");
  for (int i = 0; i < count; i++) {
    if (!first) {
      str += ",";
    }
    str += std::to_string(*(data_ptr + i));
    first = false;
  }

  str += "]";
  return str;
}


template <>
std::function<std::chrono::nanoseconds(std::mt19937&)>
ScheduleDistribution<Distribution::POISSON>(const double request_rate)
{
  std::exponential_distribution<> dist =
      std::exponential_distribution<>(request_rate);
  return [dist](std::mt19937& gen) mutable {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(dist(gen)));
  };
}

template <>
std::function<std::chrono::nanoseconds(std::mt19937&)>
ScheduleDistribution<Distribution::CONSTANT>(const double request_rate)
{
  std::chrono::nanoseconds period =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(1.0 / request_rate));
  return [period](std::mt19937& /*gen*/) { return period; };
}
