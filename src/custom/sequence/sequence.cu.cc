// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/servables/custom/custom.h"

// This custom backend takes two one-element input tensors, one an
// INT32 control value and one an INT32 value input; and produces a
// one-element output tensor. The input tensors must be named "CONTROL"
// and "INPUT". The output tensor must be named "OUTPUT".
//
// The backend maintains an INT32 accumulator which is updated based
// on the control value:
//
//   0: Add value input to accumulator.
//   1: Set accumulator equal to value input.
//   2: Ignore value input, do not change accumulator value.
//
// In all cases the accumulator is returned by the output.
//

namespace nvidia { namespace inferenceserver { namespace custom {
namespace sequence {

// Integer error codes. TRTIS requires that success must be 0. All
// other codes are interpreted by TRTIS as failures.
enum ErrorCodes {
  kSuccess,
  kUnknown,
  kInvalidModelConfig,
  kGpuNotSupported,
  kInputOutput,
  kInputName,
  kOutputName,
  kInputOutputDataType,
  kInputContents,
  kInputSize,
  kOutputBuffer,
  kUnknownControl,
  kBatchTooBig,
  kTimesteps
};

// Context object. All state must be kept in this object.
class Context {
 public:
  Context(const ModelConfig& config, const int gpu_device);
  ~Context();

  // Initialize the context. Validate that the model configuration,
  // etc. is something that we can handle.
  int Init();

  // Perform custom execution on the payloads.
  int Execute(
      const uint32_t payload_cnt, CustomPayload* payloads,
      CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn);

 private:
  int GetInputTensor(
      CustomGetNextInputFn_t input_fn, void* input_context, const char* name,
      const size_t expected_byte_size, std::vector<int32_t>* input);

  // The model configuration.
  const ModelConfig model_config_;

  // The GPU device ID to execute on or CUSTOM_NO_GPU_DEVICE if should
  // execute on CPU.
  const int gpu_device_;

  // Accumulators maintained by this context, one for each batch slot.
  std::vector<int32_t> accumulator_;
};

Context::Context(const ModelConfig& model_config, const int gpu_device)
    : model_config_(model_config), gpu_device_(gpu_device)
{
  accumulator_.resize(std::max(1, model_config_.max_batch_size()));
}

Context::~Context() {}

int
Context::Init()
{
  // Execution on GPUs not supported since only a trivial amount of
  // computation is required.
  if (gpu_device_ != CUSTOM_NO_GPU_DEVICE) {
    return kGpuNotSupported;
  }

  // There must be two inputs INT32 inputs with shape [1]. The inputs
  // must be named CONTROL and INPUT.
  if (model_config_.input_size() != 2) {
    return kInputOutput;
  }
  if ((model_config_.input(0).dims().size() != 1) ||
      (model_config_.input(0).dims(0) != 1)) {
    return kInputOutput;
  }
  if ((model_config_.input(1).dims().size() != 1) ||
      (model_config_.input(1).dims(0) != 1)) {
    return kInputOutput;
  }
  if ((model_config_.input(0).data_type() != DataType::TYPE_INT32) ||
      (model_config_.input(1).data_type() != DataType::TYPE_INT32)) {
    return kInputOutputDataType;
  }
  if ((model_config_.input(0).name() != "CONTROL") ||
      (model_config_.input(1).name() != "INPUT")) {
    return kInputName;
  }

  // There must be one INT32 output with shape [1]. The output must be
  // named OUTPUT.
  if (model_config_.output_size() != 1) {
    return kInputOutput;
  }
  if ((model_config_.output(0).dims().size() != 1) ||
      (model_config_.output(0).dims(0) != 1)) {
    return kInputOutput;
  }
  if (model_config_.output(0).data_type() != DataType::TYPE_INT32) {
    return kInputOutputDataType;
  }
  if (model_config_.output(0).name() != "OUTPUT") {
    return kOutputName;
  }

  return kSuccess;
}

int
Context::GetInputTensor(
    CustomGetNextInputFn_t input_fn, void* input_context, const char* name,
    const size_t expected_byte_size, std::vector<int32_t>* input)
{
  // The values for an input tensor are not necessarily in one
  // contiguous chunk, so we copy the chunks into 'input' vector. A
  // more performant solution would attempt to use the input tensors
  // in-place instead of having this copy.
  uint64_t total_content_byte_size = 0;

  while (true) {
    const void* content;
    uint64_t content_byte_size = expected_byte_size;
    if (!input_fn(input_context, name, &content, &content_byte_size)) {
      return kInputContents;
    }

    // If 'content' returns nullptr we have all the input.
    if (content == nullptr) {
      break;
    }

    // If the total amount of content received exceeds what we expect
    // then something is wrong.
    total_content_byte_size += content_byte_size;
    if (total_content_byte_size > expected_byte_size) {
      return kInputSize;
    }

    size_t content_elements = content_byte_size / sizeof(int32_t);
    input->insert(
        input->end(), static_cast<const int32_t*>(content),
        static_cast<const int32_t*>(content) + content_elements);
  }

  // Make sure we end up with exactly the amount of input we expect.
  if (total_content_byte_size != expected_byte_size) {
    return kInputSize;
  }

  return kSuccess;
}

int
Context::Execute(
    const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
{
  LOG_VERBOSE(1) << "Sequence executing " << payload_cnt << " payloads";

  // Each payload represents different sequence, which corresponds to
  // the accumulator at the same index. Each payload must have
  // batch-size 1 inputs which is the next timestep for that
  // sequence. The total number of payloads will not exceed the
  // max-batch-size specified in the model configuration.
  int err;

  if (payload_cnt > accumulator_.size()) {
    return kBatchTooBig;
  }

  for (uint32_t pidx = 0; pidx < payload_cnt; ++pidx) {
    CustomPayload& payload = payloads[pidx];
    if (payload.batch_size != 1) {
      payload.error_code = kTimesteps;
      continue;
    }

    const size_t batch1_byte_size = GetDataTypeByteSize(TYPE_INT32);

    // Get the input tensors.
    std::vector<int32_t> control;
    err = GetInputTensor(
        input_fn, payload.input_context, "CONTROL", batch1_byte_size, &control);
    if (err != kSuccess) {
      payload.error_code = err;
      continue;
    }

    std::vector<int32_t> input;
    err = GetInputTensor(
        input_fn, payload.input_context, "INPUT", batch1_byte_size, &input);
    if (err != kSuccess) {
      payload.error_code = err;
      continue;
    }

    // Update the accumulator value based on CONTROL and calculate
    // the output value.
    int32_t output;
    switch (control[0]) {
      case 0:  // Update accumulator.
        accumulator_[pidx] += input[0];
        break;
      case 1:  // Set accumulator.
        accumulator_[pidx] = input[0];
        break;
      case 2:  // Don't update accumulator.
        break;
      default:
        payload.error_code = kUnknownControl;
        break;
    }

    output = accumulator_[pidx];

    // If the output is requested, copy the calculated output value
    // into the output buffer.
    if ((payload.error_code == 0) && (payload.output_cnt > 0)) {
      const char* output_name = payload.required_output_names[0];

      // The output shape is [1, 1] if the model configuration
      // supports batching, or just [1] if the model configuration
      // does not support batching.
      std::vector<int64_t> shape;
      if (model_config_.max_batch_size() != 0) {
        shape.push_back(1);
      }
      shape.push_back(1);

      void* obuffer;
      if (!output_fn(
              payload.output_context, output_name, shape.size(), &shape[0],
              batch1_byte_size, &obuffer)) {
        payload.error_code = kOutputBuffer;
        continue;
      }

      memcpy(obuffer, &output, batch1_byte_size);
    }
  }

  return kSuccess;
}

/////////////

extern "C" {

int
CustomInitialize(
    const char* serialized_model_config, size_t serialized_model_config_size,
    int gpu_device_id, void** custom_context)
{
  // Convert the serialized model config to a ModelConfig object.
  ModelConfig model_config;
  if (!model_config.ParseFromString(
          std::string(serialized_model_config, serialized_model_config_size))) {
    return kInvalidModelConfig;
  }

  // Create the context and validate that the model configuration is
  // something that we can handle.
  Context* context = new Context(model_config, gpu_device_id);
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
    case kGpuNotSupported:
      return "execution on GPU not supported";
    case kInputOutput:
      return "model must have two inputs and one output with shape [1]";
    case kInputName:
      return "model inputs must be named 'INPUT0' and 'INPUT1'";
    case kOutputName:
      return "model output must be named 'OUTPUT'";
    case kInputOutputDataType:
      return "model inputs and output must have TYPE_INT32 data-type";
    case kInputContents:
      return "unable to get input tensor values";
    case kInputSize:
      return "unexpected size for input tensor";
    case kOutputBuffer:
      return "unable to get buffer for output tensor values";
    case kUnknownControl:
      return "unsupported value for 'CONTROL' input";
    case kBatchTooBig:
      return "unable to execute batch larger than max-batch-size";
    case kTimesteps:
      return "unable to execute more than 1 timestep at a time";
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

}}}}  // namespace nvidia::inferenceserver::custom::sequence
