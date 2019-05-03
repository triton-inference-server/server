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
#include "src/backends/custom/custom.h"
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"

#define LOG_ERROR std::cerr
#define LOG_INFO std::cout

// This custom backend returns system and configuration parameter
// values every time it is executed.
//
// Inputs are ignored but here must be a single output that is a
// variable-sized vector of strings. The output is used to return the
// parameter values.

namespace nvidia { namespace inferenceserver { namespace custom {
namespace param {

// Integer error codes. TRTIS requires that success must be 0. All
// other codes are interpreted by TRTIS as failures.
enum ErrorCodes {
  kSuccess = 0,
  kUnknown,
  kInvalidModelConfig,
  kBatching,
  kOutput,
  kOutputBuffer
};

// Context object. All state must be kept in this object.
class Context {
 public:
  Context(
      const std::string& instance_name, const ModelConfig& config,
      const size_t server_parameter_cnt, const char** server_parameters);

  // Initialize the context. Validate that the model configuration,
  // etc. is something that we can handle.
  int Init();

  // Perform custom execution on the payloads.
  int Execute(
      const uint32_t payload_cnt, CustomPayload* payloads,
      CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn);

 private:
  // The name of this instance of the backend.
  const std::string instance_name_;

  // The model configuration.
  const ModelConfig model_config_;

  // The server parameter values.
  std::vector<std::string> server_params_;
};

Context::Context(
    const std::string& instance_name, const ModelConfig& model_config,
    const size_t server_parameter_cnt, const char** server_parameters)
    : instance_name_(instance_name), model_config_(model_config)
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
  // Batching is not supported..
  if (model_config_.max_batch_size() != 0) {
    return kBatching;
  }

  // Don't care how many inputs there are but there must be a single
  // output that allows a variable-length vector of strings.
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
  // Batching is not supported so there must be a single payload of
  // batch-size 1.
  if ((payload_cnt != 1) || (payloads[0].batch_size != 1)) {
    return kUnknown;
  }

  // If output wasn't requested just do nothing.
  if (payloads[0].output_cnt == 0) {
    return kSuccess;
  }

  const char* output_name = payloads[0].required_output_names[0];

  // The output is a vector of strings, with one element for each
  // parameter from the system and model configuration. Each string is
  // represented in the output buffer by a 4-byte length followed by
  // the string itself, with no terminating null.
  size_t param_cnt = 0;
  std::string output;

  for (const auto& value : server_params_) {
    const std::string key = "server_" + std::to_string(param_cnt) + "=";
    uint32_t byte_size = key.size() + value.size();
    output.append(reinterpret_cast<const char*>(&byte_size), 4);
    output.append(key);
    output.append(value);

    param_cnt++;
  }

  for (const auto& pr : model_config_.parameters()) {
    std::string key = pr.first + "=";
    const std::string& value = pr.second.string_value();
    uint32_t byte_size = key.size() + value.size();
    output.append(reinterpret_cast<const char*>(&byte_size), 4);
    output.append(key);
    output.append(value);

    param_cnt++;
  }

  std::vector<int64_t> output_shape;
  output_shape.push_back(param_cnt);

  void* obuffer;
  if (!output_fn(
          payloads[0].output_context, output_name, output_shape.size(),
          &output_shape[0], output.size(), &obuffer)) {
    return kOutputBuffer;
  }

  // If no error but the 'obuffer' is returned as nullptr, then
  // skip writing this output.
  if (obuffer != nullptr) {
    memcpy(obuffer, output.data(), output.size());
  }

  return kSuccess;
}

/////////////

extern "C" {

int
CustomInitialize(const CustomInitializeData* data, void** custom_context)
{
  // Convert the serialized model config to a ModelConfig object.
  ModelConfig model_config;
  if (!model_config.ParseFromString(std::string(
          data->serialized_model_config, data->serialized_model_config_size))) {
    return kInvalidModelConfig;
  }

  // Create the context and validate that the model configuration is
  // something that we can handle.
  Context* context = new Context(
      std::string(data->instance_name), model_config,
      data->server_parameter_cnt, data->server_parameters);
  int err = context->Init();
  if (err != kSuccess) {
    return err;
  }

  *custom_context = static_cast<void*>(context);

  return kSuccess;
}

int
CustomFinalize(void* custom_context)
{
  if (custom_context != nullptr) {
    Context* context = static_cast<Context*>(custom_context);
    delete context;
  }

  return kSuccess;
}

const char*
CustomErrorString(void* custom_context, int errcode)
{
  switch (errcode) {
    case kSuccess:
      return "success";
    case kInvalidModelConfig:
      return "invalid model configuration";
    case kBatching:
      return "batching not supported";
    case kOutput:
      return "expected single output, variable-size vector of string";
    case kOutputBuffer:
      return "unable to get buffer for output tensor values";
    default:
      break;
  }

  return "unknown error";
}

int
CustomExecute(
    void* custom_context, const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
{
  if (custom_context == nullptr) {
    return kUnknown;
  }

  Context* context = static_cast<Context*>(custom_context);
  return context->Execute(payload_cnt, payloads, input_fn, output_fn);
}

}  // extern "C"

}}}}  // namespace nvidia::inferenceserver::custom::param
