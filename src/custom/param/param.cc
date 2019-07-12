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

#include <string>

#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/custom/sdk/custom_instance.h"

#define LOG_ERROR std::cerr
#define LOG_INFO std::cout

// This custom backend returns system and configuration parameter
// values every time it is executed.
//
// A single int32, shape [ 1 ] input must be provided. A single output
// is produced that is a variable-sized vector of strings. The output
// strings return the input value and the parameter values.

namespace nvidia { namespace inferenceserver { namespace custom {
namespace param {

// Integer error codes. TRTIS requires that success must be 0. All
// other codes are interpreted by TRTIS as failures.
enum ErrorCodes {
  kSuccess = 0,
  kUnknown,
  kInvalidModelConfig,
  kBatching,
  kInput,
  kInputContents,
  kOutput,
  kOutputBuffer
};

// Context object. All state must be kept in this object.
class Context : public CustomInstance {
 public:
  Context(
      const std::string& instance_name, const ModelConfig& config,
      const int gpu_device, const size_t server_parameter_cnt,
      const char** server_parameters);

  // Initialize the context. Validate that the model configuration is
  // something that we can handle.
  int Init();

  // Perform custom execution on the payloads.
  int Execute(
      const uint32_t payload_cnt, CustomPayload* payloads,
      CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn);

 private:
  // The server parameter values.
  std::vector<std::string> server_params_;

  const int kBatching = RegisterError("batching not supported");
  const int kInput =
      RegisterError("expected single int32 input with shape [ 1 ]");
  const int kInputContents = RegisterError("unable to get input tensor values");
  const int kOutput =
      RegisterError("expected single output, variable-size vector of string");
  const int kOutputBuffer =
      RegisterError("unable to get buffer for output tensor values");
};

Context::Context(
    const std::string& instance_name, const ModelConfig& model_config,
    const int gpu_device, const size_t server_parameter_cnt,
    const char** server_parameters)
    : CustomInstance(instance_name, model_config, gpu_device)
{
  // Must make a copy of server_parameters since we don't own those
  // strings.
  for (size_t i = 0; i < server_parameter_cnt; ++i) {
    server_params_.push_back(server_parameters[i]);
  }
}

int
Context::Init()
{
  // Batching is not supported...
  if (model_config_.max_batch_size() != 0) {
    return kBatching;
  }

  // There must be a single int32, shape [ 1 ] input.
  if (model_config_.input_size() != 1) {
    return kInput;
  }
  if (model_config_.input(0).dims_size() != 1) {
    return kInput;
  }
  if (model_config_.input(0).dims(0) != 1) {
    return kInput;
  }
  if (model_config_.input(0).data_type() != DataType::TYPE_INT32) {
    return kInput;
  }

  // There must be a single output that allows a variable-length
  // vector of strings.
  if (model_config_.output_size() != 1) {
    return kOutput;
  }
  if (model_config_.output(0).dims_size() != 1) {
    return kOutput;
  }
  if (model_config_.output(0).dims(0) != -1) {
    return kOutput;
  }

  if (model_config_.output(0).data_type() != DataType::TYPE_STRING) {
    return kOutput;
  }

  return kSuccess;
}

int
Context::Execute(
    const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
{
  // Batching is not supported so we never expect to see more that a
  // single payload with batch-size 1.
  if ((payload_cnt != 1) || (payloads[0].batch_size != 1)) {
    return kUnknown;
  }

  // If output wasn't requested just do nothing.
  if (payloads[0].output_cnt == 0) {
    return kSuccess;
  }

  const char* output_name = payloads[0].required_output_names[0];

  // Always expect 1 input... We could get the input name from the
  // model configuration during init time but we can als read it from
  // the payload as we do here.
  if (payloads[0].input_cnt != 1) {
    return kUnknown;
  }

  const char* input_name = payloads[0].input_names[0];

  // The output is a vector of strings, with one element for the input
  // and one element for each parameter from the system and model
  // configuration. TRTIS requires that each string be represented in
  // the output buffer by a 4-byte length followed by the string
  // itself, with no terminating null.
  size_t output_cnt = 0;
  std::string output;

  // Read the input value and convert it to a string in the output.
  {
    const void* content;
    uint64_t content_byte_size = 64;
    if (!input_fn(
            payloads[0].input_context, input_name, &content,
            &content_byte_size)) {
      return kInputContents;
    }

    // If 'content' returns nullptr or if the content is not the
    // expected size, then something went wrong.
    if ((content == nullptr) ||
        (content_byte_size != GetDataTypeByteSize(DataType::TYPE_INT32))) {
      return kInputContents;
    }

    const int32_t* icontent = reinterpret_cast<const int32_t*>(content);

    const std::string key = std::string(input_name) + "=";
    const std::string value = std::to_string(*icontent);
    uint32_t byte_size = key.size() + value.size();
    output.append(reinterpret_cast<const char*>(&byte_size), 4);
    output.append(key);
    output.append(value);
    output_cnt++;
  }

  size_t sparam_cnt = 0;
  for (const auto& value : server_params_) {
    const std::string key = "server_" + std::to_string(sparam_cnt) + "=";
    uint32_t byte_size = key.size() + value.size();
    output.append(reinterpret_cast<const char*>(&byte_size), 4);
    output.append(key);
    output.append(value);
    sparam_cnt++;
    output_cnt++;
  }

  for (const auto& pr : model_config_.parameters()) {
    std::string key = pr.first + "=";
    const std::string& value = pr.second.string_value();
    uint32_t byte_size = key.size() + value.size();
    output.append(reinterpret_cast<const char*>(&byte_size), 4);
    output.append(key);
    output.append(value);
    output_cnt++;
  }

  std::vector<int64_t> output_shape;
  output_shape.push_back(output_cnt);

  void* obuffer;
  if (!output_fn(
          payloads[0].output_context, output_name, output_shape.size(),
          &output_shape[0], output.size(), &obuffer)) {
    return kOutputBuffer;
  }

  // If there is no error but the 'obuffer' is returned as nullptr,
  // then skip writing this output.
  if (obuffer != nullptr) {
    memcpy(obuffer, output.data(), output.size());
  }

  return kSuccess;
}

}  // namespace param

// Creates a new param context instance
int
CustomInstance::Create(
    CustomInstance** instance, const std::string& name,
    const ModelConfig& model_config, int gpu_device,
    const CustomInitializeData* data)
{
  param::Context* context = new param::Context(
      name, model_config, gpu_device, data->server_parameter_cnt,
      data->server_parameters);

  *instance = context;

  if (context == nullptr) {
    return ErrorCodes::CreationFailure;
  }

  return context->Init();
}

}}}  // namespace nvidia::inferenceserver::custom
