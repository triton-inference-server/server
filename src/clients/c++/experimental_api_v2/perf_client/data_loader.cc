// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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


#include "src/clients/c++/experimental_api_v2/perf_client/data_loader.h"
#include <b64/decode.h>
#include "src/core/model_config.h"

#include <fstream>
#include "rapidjson/filereadstream.h"

DataLoader::DataLoader(const size_t batch_size)
    : batch_size_(batch_size), data_stream_cnt_(0)
{
}

nic::Error
DataLoader::ReadDataFromDir(
    std::shared_ptr<ModelTensorMap> inputs, const std::string& data_directory)
{
  // Directory structure supports only a single data stream and step
  data_stream_cnt_ = 1;
  step_num_.push_back(1);

  for (const auto& input : *inputs) {
    if (input.second.datatype_.compare("BYTES") != 0) {
      const auto file_path = data_directory + "/" + input.second.name_;
      std::string key_name(
          input.second.name_ + "_" + std::to_string(0) + "_" +
          std::to_string(0));
      auto it = input_data_.emplace(key_name, std::vector<char>()).first;
      RETURN_IF_ERROR(ReadFile(file_path, &it->second));
    } else {
      const auto file_path = data_directory + "/" + input.second.name_;
      std::vector<std::string> input_string_data;
      RETURN_IF_ERROR(ReadTextFile(file_path, &input_string_data));
      std::string key_name(
          input.second.name_ + "_" + std::to_string(0) + "_" +
          std::to_string(0));
      auto it = input_data_.emplace(key_name, std::vector<char>()).first;
      SerializeStringTensor(input_string_data, &it->second);
    }
  }
  return nic::Error::Success;
}

nic::Error
DataLoader::ReadDataFromJSON(
    std::shared_ptr<ModelTensorMap> inputs, const std::string& json_file)
{
  FILE* data_file = fopen(json_file.c_str(), "r");
  if (data_file == nullptr) {
    return nic::Error("failed to open file for reading input data");
  }

  char readBuffer[65536];
  rapidjson::FileReadStream fs(data_file, readBuffer, sizeof(readBuffer));

  rapidjson::Document d{};
  d.ParseStream(fs);

  if (d.HasParseError()) {
    std::cerr << "Error  : " << d.GetParseError() << '\n'
              << "Offset : " << d.GetErrorOffset() << '\n';
    return nic::Error(
        "failed to parse the specified json file for reading inputs");
  }

  if (!d.HasMember("data")) {
    return nic::Error("The json file doesn't contain data field");
  }

  const rapidjson::Value& streams = d["data"];
  int count = streams.Size();

  data_stream_cnt_ += count;
  int offset = step_num_.size();
  for (size_t i = offset; i < data_stream_cnt_; i++) {
    const rapidjson::Value& steps = streams[i - offset];
    if (steps.IsArray()) {
      step_num_.push_back(steps.Size());
      for (size_t k = 0; k < step_num_[i]; k++) {
        RETURN_IF_ERROR(ReadInputTensorData(steps[k], inputs, i, k));
      }
    } else {
      // There is no nesting of tensors, hence, will interpret streams as steps
      // and add the tensors to a single stream '0'.
      int offset = 0;
      if (step_num_.empty()) {
        step_num_.push_back(count);
      } else {
        offset = step_num_[0];
        step_num_[0] += (count);
      }
      data_stream_cnt_ = 1;
      for (size_t k = offset; k < step_num_[0]; k++) {
        RETURN_IF_ERROR(ReadInputTensorData(streams[k - offset], inputs, 0, k));
      }
      break;
    }
  }

  max_non_sequence_step_id_ = std::max(1, (int)(step_num_[0] / batch_size_));

  fclose(data_file);
  return nic::Error::Success;
}

nic::Error
DataLoader::GenerateData(
    std::shared_ptr<ModelTensorMap> inputs, const bool zero_input,
    const size_t string_length, const std::string& string_data)
{
  // Data generation supports only a single data stream and step
  // Not supported for inputs with dynamic shapes
  data_stream_cnt_ = 1;
  step_num_.push_back(1);

  // Validate the absence of shape tensors
  for (const auto& input : *inputs) {
    if (input.second.is_shape_tensor_) {
      return nic::Error(
          "can not generate data for shape tensor '" + input.second.name_ +
          "', user-provided data is needed.");
    }
  }

  uint64_t max_input_byte_size = 0;
  for (const auto& input : *inputs) {
    if (input.second.datatype_.compare("BYTES") != 0) {
      int64_t byte_size = ByteSize(input.second.shape_, input.second.datatype_);
      if (byte_size < 0) {
        return nic::Error(
            "input " + input.second.name_ +
            " contains dynamic shape, provide shapes to send along with the "
            "request");
      }
      max_input_byte_size = std::max(max_input_byte_size, (size_t)byte_size);
    } else {
      // Generate string input and store it into map
      std::vector<std::string> input_string_data;
      size_t batch1_num_strings = ElementCount(input.second.shape_);
      input_string_data.resize(batch1_num_strings);
      if (!string_data.empty()) {
        for (size_t i = 0; i < batch1_num_strings; i++) {
          input_string_data[i] = string_data;
        }
      } else {
        for (size_t i = 0; i < batch1_num_strings; i++) {
          input_string_data[i] = GetRandomString(string_length);
        }
      }

      std::string key_name(
          input.second.name_ + "_" + std::to_string(0) + "_" +
          std::to_string(0));
      auto it = input_data_.emplace(key_name, std::vector<char>()).first;
      SerializeStringTensor(input_string_data, &it->second);
    }
  }

  // Create a zero or randomly (as indicated by zero_input)
  // initialized buffer that is large enough to provide the largest
  // needed input. We (re)use this buffer for all non-string input values.
  if (max_input_byte_size > 0) {
    if (zero_input) {
      input_buf_.resize(max_input_byte_size, 0);
    } else {
      input_buf_.resize(max_input_byte_size);
      for (auto& byte : input_buf_) {
        byte = rand();
      }
    }
  }

  return nic::Error::Success;
}

nic::Error
DataLoader::GetInputData(
    const ModelTensor& input, const int stream_id, const int step_id,
    const uint8_t** data_ptr, size_t* batch1_size)
{
  // If json data is available then try to retrieve the data from there
  if (!input_data_.empty()) {
    // validate if the indices conform to the vector sizes
    if (stream_id < 0 || stream_id >= (int)data_stream_cnt_) {
      return nic::Error(
          "stream_id for retrieving the data should be less than " +
          std::to_string(data_stream_cnt_) + ", got " +
          std::to_string(stream_id));
    }
    if (step_id < 0 || step_id >= (int)step_num_[stream_id]) {
      return nic::Error(
          "step_id for retrieving the data should be less than " +
          std::to_string(step_num_[stream_id]) + ", got " +
          std::to_string(step_id));
    }
    std::string key_name(
        input.name_ + "_" + std::to_string(stream_id) + "_" +
        std::to_string(step_id));
    // Get the data and the corresponding byte-size
    auto it = input_data_.find(key_name);
    if (it != input_data_.end()) {
      if (input.datatype_.compare("BYTES") != 0) {
        *batch1_size = it->second.size();
      } else {
        std::vector<char>* string_data;
        string_data = &it->second;
        *batch1_size = string_data->size();
      }
      *data_ptr = (const uint8_t*)&((it->second)[0]);
    } else {
      return nic::Error(
          "unable to find data for input '" + input.name_ +
          "' in provided data.");
    }
  } else if (
      (input.datatype_.compare("BYTES") != 0) && (input_buf_.size() != 0)) {
    int64_t byte_size = ByteSize(input.shape_, input.datatype_);
    if (byte_size < 0) {
      return nic::Error(
          "failed to get correct byte size for '" + input.name_ + "'.");
    }
    *batch1_size = (size_t)byte_size;
    *data_ptr = &input_buf_[0];
  } else {
    return nic::Error("unable to find data for input '" + input.name_ + "'.");
  }
  return nic::Error::Success;
}

nic::Error
DataLoader::GetInputShape(
    const ModelTensor& input, const int stream_id, const int step_id,
    const std::vector<int64_t>** provided_shape)
{
  std::string key_name(
      input.name_ + "_" + std::to_string(stream_id) + "_" +
      std::to_string(step_id));

  // Prefer the values read from file over the ones provided from
  // CLI
  auto it = input_shapes_.find(key_name);
  if (it != input_shapes_.end()) {
    *provided_shape = &it->second;
  } else {
    *provided_shape = &input.shape_;
  }
  return nic::Error::Success;
}

nic::Error
DataLoader::ReadInputTensorData(
    const rapidjson::Value& step, std::shared_ptr<ModelTensorMap> inputs,
    int stream_index, int step_index)
{
  for (const auto& input : *inputs) {
    if (step.HasMember(input.first.c_str())) {
      std::string key_name(
          input.first + "_" + std::to_string(stream_index) + "_" +
          std::to_string(step_index));

      auto it = input_data_.emplace(key_name, std::vector<char>()).first;

      const rapidjson::Value& tensor = step[(input.first).c_str()];

      const rapidjson::Value* content;

      if (tensor.IsArray()) {
        content = &tensor;
      } else if (tensor.HasMember("b64")) {
        content = &tensor;
      } else {
        // Populate the shape values first if available
        if (tensor.HasMember("shape")) {
          auto shape_it =
              input_shapes_.emplace(key_name, std::vector<int64_t>()).first;
          for (const auto& value : tensor["shape"].GetArray()) {
            if (!value.IsInt()) {
              return nic::Error("shape values must be integers.");
            }
            shape_it->second.push_back(value.GetInt());
          }
          const auto& dims = shape_it->second;
        }

        if (!tensor.HasMember("content")) {
          return nic::Error(
              "missing content field. ( Location stream id: " +
              std::to_string(stream_index) +
              ", step id: " + std::to_string(step_index) + ")");
        }

        content = &tensor["content"];
      }


      if (content->IsArray()) {
        RETURN_IF_ERROR(SerializeExplicitTensor(
            *content, input.second.datatype_, &it->second));
      } else {
        if (content->HasMember("b64")) {
          if ((*content)["b64"].IsString()) {
            const std::string& encoded = (*content)["b64"].GetString();
            it->second.resize(encoded.length());
            base64::decoder D;
            int size =
                D.decode(encoded.c_str(), encoded.length(), &it->second[0]);
            it->second.resize(size);

            int64_t batch1_byte = ByteSize(input.second.shape_, input.second.datatype_);;
            if (batch1_byte > 0 && (size_t)batch1_byte != it->second.size()) {
              return nic::Error(
                  "mismatch in the data provided. "
                  "Expected: " +
                  std::to_string(batch1_byte) +
                  " bytes, Got: " + std::to_string(it->second.size()) +
                  " bytes ( Location stream id: " +
                  std::to_string(stream_index) +
                  ", step id: " + std::to_string(step_index) + ")");
            }
          } else {
            return nic::Error(
                "the value of b64 field should be of type string ( "
                "Location stream id: " +
                std::to_string(stream_index) +
                ", step id: " + std::to_string(step_index) + ")");
          }
        } else {
          return nic::Error(
              "The input values are not supported. Expected an array or "
              "b64 string ( Location stream id: " +
              std::to_string(stream_index) +
              ", step id: " + std::to_string(step_index) + ")");
        }
      }
    } else {
      return nic::Error(
          "missing input " + input.first +
          " ( Location stream id: " + std::to_string(stream_index) +
          ", step id: " + std::to_string(step_index) + ")");
    }
  }

  return nic::Error::Success;
}
