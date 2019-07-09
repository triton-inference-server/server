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

#include <chrono>
#include <string>
#include <thread>
#include "src/backends/custom/custom.h"
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/custom/sdk/custom_instance.h"

#define LOG_ERROR std::cerr
#define LOG_INFO std::cout

// This custom backend takes any number of inputs and copies each
// input to the corresponding output. The number of inputs must equal
// the number of outputs and each pair must be named
// INPUTn/OUTPUTn. The datatype and size of each input must match the
// corresponding output.
//

namespace nvidia { namespace inferenceserver { namespace custom {
namespace identity {

// Integer error codes. TRTIS requires that success must be 0. All
// other codes are interpreted by TRTIS as failures.
enum ContextErrorCodes : int {
  kGpuNotSupported = kNumErrorCodes,
  kInputOutput,
  kInputOutputName,
  kInputOutputDataType,
  kInputContents,
  kInputSize,
  kRequestOutput,
  kOutputBuffer
};

// Context object. All state must be kept in this object.
class Context : public CustomInstance {
 public:
  Context(
      const std::string& instance_name, const ModelConfig& config,
      const int gpu_device);
  ~Context() = default;

  // Validate the model configuration for the derived backend instance
  int Init();

  int Execute(
      const uint32_t payload_cnt, CustomPayload* payloads,
      CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn) override;

 private:
  struct CopyInfo {
    std::string input_name_;
    DataType datatype_;
  };

  // Map from output name to information needed to copy input into
  // that output.
  std::unordered_map<std::string, CopyInfo> copy_map_;

  // An overridable method to add error strings for additional custom errors
  const char* CustomErrorString(int errcode) override;
};

Context::Context(
    const std::string& instance_name, const ModelConfig& model_config,
    const int gpu_device)
    : CustomInstance(instance_name, model_config, gpu_device)
{
}

int
Context::Init()
{
  // Execution on GPUs not supported since only a trivial amount of
  // computation is required.
  if (gpu_device_ != CUSTOM_NO_GPU_DEVICE) {
    return kGpuNotSupported;
  }

  // Equal number of inputs and outputs.
  if (model_config_.input_size() != model_config_.output_size()) {
    return kInputOutput;
  }

  // Name, datatypes, and shape must be equal across input/output
  // pairs. Use reshape dimensions if specified...
  for (int i = 0; i < model_config_.input_size(); ++i) {
    const DimsList& input_shape = (model_config_.input(i).has_reshape())
                                      ? model_config_.input(i).reshape().shape()
                                      : model_config_.input(i).dims();
    const DimsList& output_shape =
        (model_config_.output(i).has_reshape())
            ? model_config_.output(i).reshape().shape()
            : model_config_.output(i).dims();

    if (!CompareDims(input_shape, output_shape)) {
      return kInputOutput;
    }
    if (model_config_.input(i).data_type() !=
        model_config_.output(i).data_type()) {
      return kInputOutputDataType;
    }
    if ((model_config_.input(i).name().substr(0, 5) != "INPUT") ||
        (model_config_.output(i).name().substr(0, 6) != "OUTPUT") ||
        (model_config_.input(i).name().substr(5) !=
         model_config_.output(i).name().substr(6))) {
      return kInputOutputName;
    }

    copy_map_[model_config_.output(i).name()] = CopyInfo{
        model_config_.input(i).name(), model_config_.output(i).data_type()};
  }

  return kSuccess;
}

int
Context::Execute(
    const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
{
  for (uint32_t pidx = 0; pidx < payload_cnt; ++pidx) {
    CustomPayload& payload = payloads[pidx];

    for (uint32_t output_idx = 0; output_idx < payload.output_cnt;
         ++output_idx) {
      if (payload.error_code != 0) {
        break;
      }

      const char* output_cname = payload.required_output_names[output_idx];
      const auto itr = copy_map_.find(output_cname);
      if (itr == copy_map_.end()) {
        payload.error_code = kRequestOutput;
        break;
      }

      const std::string& input_name = itr->second.input_name_;
      const DataType datatype = itr->second.datatype_;

      std::vector<int64_t> shape;
      if (model_config_.max_batch_size() != 0) {
        shape.push_back(payload.batch_size);
      }
      for (uint32_t input_idx = 0; input_idx < payload.input_cnt; ++input_idx) {
        if (!strcmp(payload.input_names[input_idx], input_name.c_str())) {
          shape.insert(
              shape.end(), payload.input_shape_dims[input_idx],
              payload.input_shape_dims[input_idx] +
                  payload.input_shape_dim_cnts[input_idx]);
          break;
        }
      }

      const int64_t batchn_byte_size = GetByteSize(datatype, shape);
      if (batchn_byte_size < 0) {
        payload.error_code = kOutputBuffer;
        break;
      }

      void* obuffer;
      if (!output_fn(
              payload.output_context, output_cname, shape.size(), &shape[0],
              batchn_byte_size, &obuffer)) {
        payload.error_code = kOutputBuffer;
        break;
      }

      // If no error but the 'obuffer' is returned as nullptr, then
      // skip writing this output.
      if (obuffer == nullptr) {
        continue;
      }

      char* output_buffer = reinterpret_cast<char*>(obuffer);

      uint64_t total_byte_size = 0;
      while (true) {
        const void* content;
        uint64_t content_byte_size = 128 * 1024;
        if (!input_fn(
                payload.input_context, input_name.c_str(), &content,
                &content_byte_size)) {
          payload.error_code = kInputContents;
          break;
        }

        // If 'content' returns nullptr we have all the input.
        if (content == nullptr) {
          break;
        }

        memcpy(output_buffer + total_byte_size, content, content_byte_size);
        total_byte_size += content_byte_size;
      }
    }
  }

  return kSuccess;
}

const char*
Context::CustomErrorString(int errcode)
{
  switch (errcode) {
    case kGpuNotSupported:
      return "execution on GPU not supported";
    case kInputOutput:
      return "model must have equal input/output pairs with matching shape";
    case kInputOutputName:
      return "model input/output pairs must be named 'INPUTn' and 'OUTPUTn'";
    case kInputOutputDataType:
      return "model input/output pairs must have same data-type";
    case kInputContents:
      return "unable to get input tensor values";
    case kInputSize:
      return "unexpected size for input tensor";
    case kRequestOutput:
      return "inference request for unknown output";
    case kOutputBuffer:
      return "unable to get buffer for output tensor values";
    default:
      break;
  }

  return "unknown error";
}

}  // namespace identity

// Creates a new Indentiy context instance
int
CustomInstance::Create(
    CustomInstance** instance, const std::string& name,
    const ModelConfig& model_config, int gpu_device,
    const CustomInitializeData* data)
{
  identity::Context* ctx =
      new identity::Context(name, model_config, gpu_device);

  *instance = ctx;

  if (ctx == nullptr) {
    return kCreationFailure;
  }

  return ctx->Init();
}

}}}  // namespace nvidia::inferenceserver::custom
