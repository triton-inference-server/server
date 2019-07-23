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

#include <chrono>
#include <string>
#include <thread>

#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/custom/sdk/custom_instance.h"

#define LOG_ERROR std::cerr
#define LOG_INFO std::cout

// This custom backend takes three one-element input tensors, two
// INT32 control values and one an INT32 value input; and produces a
// one-element output tensor. The input tensors must be named "START",
// "READY" and "INPUT". The output tensor must be named "OUTPUT".
//
// The backend maintains an INT32 accumulator which is updated based
// on the control values in "START" and "READY":
//
//   READY=0, START=x: Ignore value input, do not change accumulator value.
//   READY=1, START=1: Start accumulating. Set accumulator equal to value input.
//   READY=1, START=0: Add value input to accumulator.
//
// When READY=1, the accumulator is returned in the output.
//

namespace nvidia { namespace inferenceserver { namespace custom {
namespace sequence {

// Context object. All state must be kept in this object.
class Context : public CustomInstance {
 public:
  Context(
      const std::string& instance_name, const ModelConfig& config,
      const int gpu_device);
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
      const size_t expected_byte_size, std::vector<uint8_t>* input);

  // Delay to introduce into execution, in milliseconds.
  int execute_delay_ms_;

  // Accumulators maintained by this context, one for each batch slot.
  std::vector<int32_t> accumulator_;

  // Local error codes
  const int kGpuNotSupported = RegisterError("execution on GPU not supported");
  const int kSequenceBatcher =
      RegisterError("model configuration must configure sequence batcher");
  const int kModelControl = RegisterError(
      "'START' and 'READY' must be configured as the control inputs");
  const int kInputOutput =
      RegisterError("model must have two inputs and one output with shape [1]");
  const int kInputName =
      RegisterError("model inputs must be named 'INPUT0' and 'INPUT1'");
  const int kOutputName = RegisterError("model output must be named 'OUTPUT'");
  const int kInputOutputDataType =
      RegisterError("model inputs and output must have TYPE_INT32 data-type");
  const int kInputContents = RegisterError("unable to get input tensor values");
  const int kInputSize = RegisterError("unexpected size for input tensor");
  const int kOutputBuffer =
      RegisterError("unable to get buffer for output tensor values");
  const int kBatchTooBig =
      RegisterError("unable to execute batch larger than max-batch-size");
  const int kTimesteps =
      RegisterError("unable to execute more than 1 timestep at a time");
};

Context::Context(
    const std::string& instance_name, const ModelConfig& model_config,
    const int gpu_device)
    : CustomInstance(instance_name, model_config, gpu_device),
      execute_delay_ms_(0)
{
  if (model_config_.parameters_size() > 0) {
    const auto itr = model_config_.parameters().find("execute_delay_ms");
    if (itr != model_config_.parameters().end()) {
      execute_delay_ms_ = std::stoi(itr->second.string_value());
    }
  }

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

  // The model configuration must specify the sequence batcher and
  // must use the START and READY input to indicate control values.
  if (!model_config_.has_sequence_batching()) {
    return kSequenceBatcher;
  }

  auto& batcher = model_config_.sequence_batching();
  if (batcher.control_input_size() != 2) {
    return kModelControl;
  }
  if (!(((batcher.control_input(0).name() == "START") &&
         (batcher.control_input(1).name() == "READY")) ||
        ((batcher.control_input(0).name() == "READY") &&
         (batcher.control_input(1).name() == "START")))) {
    return kModelControl;
  }

  // There must be one INT32 input called INPUT defined in the model
  // configuration with shape [1].
  if (model_config_.input_size() != 1) {
    return kInputOutput;
  }
  if ((model_config_.input(0).dims().size() != 1) ||
      (model_config_.input(0).dims(0) != 1)) {
    return kInputOutput;
  }
  if (model_config_.input(0).data_type() != DataType::TYPE_INT32) {
    return kInputOutputDataType;
  }
  if (model_config_.input(0).name() != "INPUT") {
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

  return ErrorCodes::Success;
}

int
Context::GetInputTensor(
    CustomGetNextInputFn_t input_fn, void* input_context, const char* name,
    const size_t expected_byte_size, std::vector<uint8_t>* input)
{
  // The values for an input tensor are not necessarily in one
  // contiguous chunk, so we copy the chunks into 'input' vector. A
  // more performant solution would attempt to use the input tensors
  // in-place instead of having this copy.
  uint64_t total_content_byte_size = 0;

  while (true) {
    const void* content;
    uint64_t content_byte_size = expected_byte_size - total_content_byte_size;
    if (!input_fn(input_context, name, &content, &content_byte_size)) {
      return kInputContents;
    }

    // If 'content' returns nullptr we have all the input.
    if (content == nullptr) {
      break;
    }

    std::cout << std::string(name) << ": size " << content_byte_size << ", "
              << (reinterpret_cast<const int32_t*>(content)[0]) << std::endl;

    // If the total amount of content received exceeds what we expect
    // then something is wrong.
    total_content_byte_size += content_byte_size;
    if (total_content_byte_size > expected_byte_size) {
      return kInputSize;
    }

    input->insert(
        input->end(), static_cast<const uint8_t*>(content),
        static_cast<const uint8_t*>(content) + content_byte_size);
  }

  // Make sure we end up with exactly the amount of input we expect.
  if (total_content_byte_size != expected_byte_size) {
    return kInputSize;
  }

  return ErrorCodes::Success;
}

int
Context::Execute(
    const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
{
  std::cout << "Sequence executing " << payload_cnt << " payloads" << std::endl;

  // Each payload represents different sequence, which corresponds to
  // the accumulator at the same index. Each payload must have
  // batch-size 1 inputs which is the next timestep for that
  // sequence. The total number of payloads will not exceed the
  // max-batch-size specified in the model configuration.
  int err;

  if (payload_cnt > accumulator_.size()) {
    return kBatchTooBig;
  }

  // Delay if requested...
  if (execute_delay_ms_ > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(execute_delay_ms_));
  }

  for (uint32_t pidx = 0; pidx < payload_cnt; ++pidx) {
    CustomPayload& payload = payloads[pidx];
    if (payload.batch_size != 1) {
      payload.error_code = kTimesteps;
      continue;
    }

    const size_t batch1_byte_size = GetDataTypeByteSize(TYPE_INT32);

    // Get the input tensors.
    std::vector<uint8_t> start_buffer, ready_buffer, input_buffer;
    err = GetInputTensor(
        input_fn, payload.input_context, "START", batch1_byte_size,
        &start_buffer);
    if (err != ErrorCodes::Success) {
      payload.error_code = err;
      continue;
    }

    err = GetInputTensor(
        input_fn, payload.input_context, "READY", batch1_byte_size,
        &ready_buffer);
    if (err != ErrorCodes::Success) {
      payload.error_code = err;
      continue;
    }

    err = GetInputTensor(
        input_fn, payload.input_context, "INPUT", batch1_byte_size,
        &input_buffer);
    if (err != ErrorCodes::Success) {
      payload.error_code = err;
      continue;
    }

    int32_t* start = reinterpret_cast<int32_t*>(&start_buffer[0]);
    int32_t* ready = reinterpret_cast<int32_t*>(&ready_buffer[0]);
    int32_t* input = reinterpret_cast<int32_t*>(&input_buffer[0]);

    // Update the accumulator value based on START/READY and calculate
    // the output value.
    if (ready[0] != 0) {
      if (start[0] == 0) {
        // Update accumulator.
        accumulator_[pidx] += input[0];
      } else {
        // Set accumulator.
        accumulator_[pidx] = input[0];
      }

      const int32_t output = accumulator_[pidx];

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

        // If no error but the 'obuffer' is returned as nullptr, then
        // skip writing this output.
        if (obuffer != nullptr) {
          memcpy(obuffer, &output, batch1_byte_size);
        }
      }
    }
  }

  return ErrorCodes::Success;
}

}  // namespace sequence

// Creates a new sequence context instance
int
CustomInstance::Create(
    CustomInstance** instance, const std::string& name,
    const ModelConfig& model_config, int gpu_device,
    const CustomInitializeData* data)
{
  sequence::Context* context =
      new sequence::Context(name, model_config, gpu_device);

  *instance = context;

  if (context == nullptr) {
    return ErrorCodes::CreationFailure;
  }

  return context->Init();
}

}}}  // namespace nvidia::inferenceserver::custom
