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
#include "cuda/include/cuda.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/core/model_config_cuda.h"
#include "src/custom/addsub/kernel.h"
#include "src/servables/custom/custom.h"

// This custom backend takes two input tensors (any shape but must
// have the same shape) and produces two output tensors (with same
// shape as the inputs). All tensors must be the same data-type,
// either INT32 or FP32. The input tensors must be named "INPUT0" and
// "INPUT1". The output tensors must be named "OUTPUT0" and
// "OUTPUT1". This backend does element-wise operation to produce:
//
//   OUTPUT0 = INPUT0 + INPUT1
//   OUTPUT1 = INPUT0 - INPUT1
//

namespace nvidia { namespace inferenceserver { namespace custom {
namespace addsub {

// Integer error codes. TRTIS requires that success must be 0. All
// other codes are interpreted by TRTIS as failures.
enum ErrorCodes {
  kSuccess,
  kUnknown,
  kInvalidModelConfig,
  kGpuNotSupported,
  kInputOutputShape,
  kInputName,
  kOutputName,
  kInputOutputDataType,
  kInputContents,
  kInputSize,
  kOutputBuffer,
  kCudaMalloc,
  kCudaMemcpy,
  kCudaStream
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
  int FreeCudaBuffers();
  int AllocateCudaBuffers(size_t byte_size);

  int GetInputTensorCPU(
      CustomGetNextInputFn_t input_fn, void* input_context, const char* name,
      const size_t expected_byte_size, std::vector<uint8_t>* input);
  int GetInputTensorGPU(
      CustomGetNextInputFn_t input_fn, void* input_context, const char* name,
      const size_t expected_byte_size, uint8_t* input);

  int ExecuteCPU(
      const uint32_t payload_cnt, CustomPayload* payloads,
      CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn);
  int ExecuteGPU(
      const uint32_t payload_cnt, CustomPayload* payloads,
      CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn);

  // The model configuration.
  const ModelConfig model_config_;

  // The GPU device ID to execute on or CUSTOM_NO_GPU_DEVICE if should
  // execute on CPU.
  const int gpu_device_;

  // The data-type of the input and output tensors. Must be either
  // INT32 or FP32.
  DataType datatype_;

  // CUDA memory buffers for input and output tensors.
  size_t cuda_buffer_byte_size_;
  uint8_t* cuda_input0_;
  uint8_t* cuda_input1_;
  uint8_t* cuda_output_;

  // The contexts executing on a GPU, the CUDA stream to use for the
  // execution.
  cudaStream_t stream_;
};

Context::Context(const ModelConfig& model_config, const int gpu_device)
    : model_config_(model_config), gpu_device_(gpu_device),
      datatype_(DataType::TYPE_INVALID), cuda_buffer_byte_size_(0),
      cuda_input0_(nullptr), cuda_input1_(nullptr), cuda_output_(nullptr),
      stream_(nullptr)
{
}

Context::~Context()
{
  FreeCudaBuffers();

  if (stream_ != nullptr) {
    cudaError_t cuerr = cudaStreamDestroy(stream_);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "Failed to destroy cuda stream: "
                << cudaGetErrorString(cuerr);
    }
    stream_ = nullptr;
  }
}

int
Context::FreeCudaBuffers()
{
  if (cuda_input0_ != nullptr) {
    cudaError_t cuerr = cudaFree(cuda_input0_);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "Failed to free cuda memory: " << cudaGetErrorString(cuerr);
    }
    cuda_input0_ = nullptr;
  }
  if (cuda_input1_ != nullptr) {
    cudaError_t cuerr = cudaFree(cuda_input1_);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "Failed to free cuda memory: " << cudaGetErrorString(cuerr);
    }
    cuda_input1_ = nullptr;
  }
  if (cuda_output_ != nullptr) {
    cudaError_t cuerr = cudaFree(cuda_output_);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "Failed to free cuda memory: " << cudaGetErrorString(cuerr);
    }
    cuda_output_ = nullptr;
  }

  cuda_buffer_byte_size_ = 0;
  return kSuccess;
}

int
Context::AllocateCudaBuffers(size_t byte_size)
{
  cudaError_t cuerr;

  FreeCudaBuffers();

  // Allocate GPU memory buffers large enough for each input and
  // output. For performance we allocate once during initialization
  // instead of doing it each time we execute.
  cuerr = cudaMalloc(&cuda_input0_, byte_size);
  if (cuerr != cudaSuccess) {
    LOG_ERROR << "unable to allocate memory for addsub: "
              << cudaGetErrorString(cuerr);
    return kCudaMalloc;
  }
  cuerr = cudaMalloc(&cuda_input1_, byte_size);
  if (cuerr != cudaSuccess) {
    LOG_ERROR << "unable to allocate memory for addsub: "
              << cudaGetErrorString(cuerr);
    return kCudaMalloc;
  }
  cuerr = cudaMalloc(&cuda_output_, byte_size);
  if (cuerr != cudaSuccess) {
    LOG_ERROR << "unable to allocate memory for addsub: "
              << cudaGetErrorString(cuerr);
    return kCudaMalloc;
  }

  cuda_buffer_byte_size_ = byte_size;
  return kSuccess;
}

int
Context::Init()
{
  // There must be two inputs that have the same shape. The shape can
  // be anything (including having wildcard, -1, dimensions) since we
  // are just going to do an element-wise add and an element-wise
  // subtract. The input data-type must be INT32 or FP32. The inputs
  // must be named INPUT0 and INPUT1.
  if (model_config_.input_size() != 2) {
    return kInputOutputShape;
  }
  if (!CompareDims(
          model_config_.input(0).dims(), model_config_.input(1).dims())) {
    return kInputOutputShape;
  }

  datatype_ = model_config_.input(0).data_type();
  if (((datatype_ != DataType::TYPE_INT32) &&
       (datatype_ != DataType::TYPE_FP32)) ||
      (model_config_.input(1).data_type() != datatype_)) {
    return kInputOutputDataType;
  }
  if ((model_config_.input(0).name() != "INPUT0") ||
      (model_config_.input(1).name() != "INPUT1")) {
    return kInputName;
  }

  // There must be two outputs that have the same shape as the
  // inputs. The output data-type must be the same as the input
  // data-type. The outputs must be named OUTPUT0 and OUTPUT1.
  if (model_config_.output_size() != 2) {
    return kInputOutputShape;
  }
  if (!CompareDims(
          model_config_.output(0).dims(), model_config_.output(1).dims()) ||
      !CompareDims(
          model_config_.output(0).dims(), model_config_.input(0).dims())) {
    return kInputOutputShape;
  }
  if ((model_config_.output(0).data_type() != datatype_) ||
      (model_config_.output(1).data_type() != datatype_)) {
    return kInputOutputDataType;
  }
  if ((model_config_.output(0).name() != "OUTPUT0") ||
      (model_config_.output(1).name() != "OUTPUT1")) {
    return kOutputName;
  }

  // Additional initialization if executing on the GPU...
  if (gpu_device_ != CUSTOM_NO_GPU_DEVICE) {
    // Create a CUDA stream for this context so that it executes
    // independently of other instances of this backend.
    const int cuda_stream_priority =
        GetCudaStreamPriority(model_config_.optimization().priority());
    cudaError_t cuerr = cudaStreamCreateWithPriority(
        &stream_, cudaStreamDefault, cuda_stream_priority);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "unable to create stream for addsub backend: "
                << cudaGetErrorString(cuerr);
      return kCudaStream;
    }
  }

  return kSuccess;
}

namespace {

template <typename T>
void
AddForType(uint64_t cnt, uint8_t* in0, uint8_t* in1, uint8_t* out)
{
  T* output = reinterpret_cast<T*>(out);
  T* input0 = reinterpret_cast<T*>(in0);
  T* input1 = reinterpret_cast<T*>(in1);
  for (uint64_t i = 0; i < cnt; ++i) {
    output[i] = input0[i] + input1[i];
  }
}

template <typename T>
void
SubForType(uint64_t cnt, uint8_t* in0, uint8_t* in1, uint8_t* out)
{
  T* output = reinterpret_cast<T*>(out);
  T* input0 = reinterpret_cast<T*>(in0);
  T* input1 = reinterpret_cast<T*>(in1);
  for (uint64_t i = 0; i < cnt; ++i) {
    output[i] = input0[i] - input1[i];
  }
}

}  // namespace

int
Context::GetInputTensorCPU(
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

    size_t content_elements = content_byte_size / sizeof(uint8_t);
    input->insert(
        input->end(), static_cast<const uint8_t*>(content),
        static_cast<const uint8_t*>(content) + content_elements);
  }

  // Make sure we end up with exactly the amount of input we expect.
  if (total_content_byte_size != expected_byte_size) {
    return kInputSize;
  }

  return kSuccess;
}

int
Context::ExecuteCPU(
    const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
{
  // Each payload represents a related set of inputs and required
  // outputs. Each payload may have a different batch size. The total
  // batch-size of all payloads will not exceed the max-batch-size
  // specified in the model configuration.
  if (payload_cnt == 0) {
    return kSuccess;
  }

  // For performance, we would typically execute all payloads together
  // as a single batch by first gathering the inputs from across the
  // payloads and then scattering the outputs across the payloads.
  // Here, for simplicity and clarity, we instead process each payload
  // separately.

  // Make sure all inputs have the same shape. We need to do this
  // check for every request to support variable-size input tensors
  // (otherwise the checks of the model configuration in Init() would
  // be sufficient). The scheduler will ensure that all payloads have
  // consistent shape for all inputs so we only need to check that the
  // first payload INPUT0 and INPUT1 are the same shape.

  if (payloads[0].input_cnt != 2) {
    // Should never hit this case since inference server will ensure
    // correct number of inputs...
    return kInputOutputShape;
  }

  std::vector<int64_t> shape(
      payloads[0].input_shape_dims[0],
      payloads[0].input_shape_dims[0] + payloads[0].input_shape_dim_cnts[0]);
  std::vector<int64_t> shape1(
      payloads[0].input_shape_dims[1],
      payloads[0].input_shape_dims[1] + payloads[0].input_shape_dim_cnts[1]);
  if (shape != shape1) {
    return kInputOutputShape;
  }

  const uint64_t batch1_element_count = GetElementCount(shape);
  const uint64_t batch1_byte_size =
      batch1_element_count * GetDataTypeByteSize(datatype_);

  int err;
  for (uint32_t pidx = 0; pidx < payload_cnt; ++pidx) {
    CustomPayload& payload = payloads[pidx];

    // For this payload the expected size of the input and output
    // tensors is determined by the batch-size of this payload.
    const uint64_t batchn_element_count =
        payload.batch_size * batch1_element_count;
    const uint64_t batchn_byte_size = payload.batch_size * batch1_byte_size;

    // Get the input tensors.
    std::vector<uint8_t> input0;
    err = GetInputTensorCPU(
        input_fn, payload.input_context, "INPUT0", batchn_byte_size, &input0);
    if (err != kSuccess) {
      payload.error_code = err;
      continue;
    }

    std::vector<uint8_t> input1;
    err = GetInputTensorCPU(
        input_fn, payload.input_context, "INPUT1", batchn_byte_size, &input1);
    if (err != kSuccess) {
      payload.error_code = err;
      continue;
    }

    // The output shape is [payload-batch-size, shape] if the model
    // configuration supports batching, or just [shape] if the
    // model configuration does not support batching.
    std::vector<int64_t> output_shape;
    if (model_config_.max_batch_size() != 0) {
      output_shape.push_back(payload.batch_size);
    }
    output_shape.insert(output_shape.end(), shape.begin(), shape.end());

    // For each requested output get the buffer to hold the output
    // values and calculate the sum/difference directly into that
    // buffer.
    for (uint32_t oidx = 0; oidx < payload.output_cnt; ++oidx) {
      const char* output_name = payload.required_output_names[oidx];

      void* obuffer;
      if (!output_fn(
              payload.output_context, output_name, output_shape.size(),
              &output_shape[0], batchn_byte_size, &obuffer)) {
        payload.error_code = kOutputBuffer;
        break;
      }

      if (!strncmp(output_name, "OUTPUT0", strlen("OUTPUT0"))) {
        if (datatype_ == DataType::TYPE_INT32) {
          AddForType<int32_t>(
              batchn_element_count, &input0[0], &input1[0],
              reinterpret_cast<uint8_t*>(obuffer));
        } else {
          AddForType<float>(
              batchn_element_count, &input0[0], &input1[0],
              reinterpret_cast<uint8_t*>(obuffer));
        }
      } else {
        if (datatype_ == DataType::TYPE_INT32) {
          SubForType<int32_t>(
              batchn_element_count, &input0[0], &input1[0],
              reinterpret_cast<uint8_t*>(obuffer));
        } else {
          SubForType<float>(
              batchn_element_count, &input0[0], &input1[0],
              reinterpret_cast<uint8_t*>(obuffer));
        }
      }
    }
  }

  return kSuccess;
}

int
Context::GetInputTensorGPU(
    CustomGetNextInputFn_t input_fn, void* input_context, const char* name,
    const size_t expected_byte_size, uint8_t* input)
{
  // The values for an input tensor are not necessarily in one
  // contiguous chunk, so we copy the chunks into 'input', which
  // points to CUDA memory.
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
    if ((total_content_byte_size + content_byte_size) > expected_byte_size) {
      return kInputSize;
    }

    cudaError_t cuerr = cudaMemcpyAsync(
        reinterpret_cast<char*>(input) + total_content_byte_size, content,
        content_byte_size, cudaMemcpyHostToDevice, stream_);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "failed to copy input values to GPU for addsub: "
                << cudaGetErrorString(cuerr);
      return kCudaMemcpy;
    }

    total_content_byte_size += content_byte_size;
  }

  // Make sure we end up with exactly the amount of input we expect.
  if (total_content_byte_size != expected_byte_size) {
    return kInputSize;
  }

  return kSuccess;
}

int
Context::ExecuteGPU(
    const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
{
  // Each payload represents a related set of inputs and required
  // outputs. Each payload may have a different batch size. The total
  // batch-size of all payloads will not exceed the max-batch-size
  // specified in the model configuration.
  if (payload_cnt == 0) {
    return kSuccess;
  }

  cudaSetDevice(gpu_device_);

  // For performance, we would typically execute all payloads together
  // as a single batch by first gathering the inputs from across the
  // payloads and then scattering the outputs across the payloads.
  // Here, for simplicity and clarity, we instead process each payload
  // separately.

  // Make sure all inputs have the same shape. We need to do this
  // check for every request to support variable-size input tensors
  // (otherwise the checks of the model configuration in Init() would
  // be sufficient). The scheduler will ensure that all payloads have
  // consistent shape for all inputs so we only need to check that the
  // first payload INPUT0 and INPUT1 are the same shape.

  if (payloads[0].input_cnt != 2) {
    // Should never hit this case since inference server will ensure
    // correct number of inputs...
    return kInputOutputShape;
  }

  std::vector<int64_t> shape(
      payloads[0].input_shape_dims[0],
      payloads[0].input_shape_dims[0] + payloads[0].input_shape_dim_cnts[0]);
  std::vector<int64_t> shape1(
      payloads[0].input_shape_dims[1],
      payloads[0].input_shape_dims[1] + payloads[0].input_shape_dim_cnts[1]);
  if (shape != shape1) {
    return kInputOutputShape;
  }

  const uint64_t batch1_element_count = GetElementCount(shape);
  const uint64_t batch1_byte_size =
      batch1_element_count * GetDataTypeByteSize(datatype_);

  int err;
  for (uint32_t pidx = 0; pidx < payload_cnt; ++pidx) {
    CustomPayload& payload = payloads[pidx];

    // For this payload the expected size of the input and output
    // tensors is determined by the batch-size of this payload.
    const uint64_t batchn_element_count =
        payload.batch_size * batch1_element_count;
    const uint64_t batchn_byte_size = payload.batch_size * batch1_byte_size;

    // Make sure the CUDA memory buffers are large enough for this
    // payload. If not increase their size.
    if (batchn_byte_size > cuda_buffer_byte_size_) {
      FreeCudaBuffers();
      err = AllocateCudaBuffers(batchn_byte_size);
      if (err != kSuccess) {
        payload.error_code = err;
        continue;
      }
    }

    // Copy the input tensors into the appropriate CUDA memory buffer.
    err = GetInputTensorGPU(
        input_fn, payload.input_context, "INPUT0", batchn_byte_size,
        cuda_input0_);
    if (err != kSuccess) {
      payload.error_code = err;
      continue;
    }

    err = GetInputTensorGPU(
        input_fn, payload.input_context, "INPUT1", batchn_byte_size,
        cuda_input1_);
    if (err != kSuccess) {
      payload.error_code = err;
      continue;
    }

    // The output shape is [payload-batch-size, shape] if the model
    // configuration supports batching, or just [shape] if the
    // model configuration does not support batching.
    std::vector<int64_t> output_shape;
    if (model_config_.max_batch_size() != 0) {
      output_shape.push_back(payload.batch_size);
    }
    output_shape.insert(output_shape.end(), shape.begin(), shape.end());

    // For each requested output calculate the sum/difference directly
    // into the CUDA output buffer and then copy out.
    for (uint32_t oidx = 0; oidx < payload.output_cnt; ++oidx) {
      const char* output_name = payload.required_output_names[oidx];

      void* obuffer;
      if (!output_fn(
              payload.output_context, output_name, output_shape.size(),
              &output_shape[0], batchn_byte_size, &obuffer)) {
        payload.error_code = kOutputBuffer;
        break;
      }

      int block_size = 1024;
      int grid_size = (batchn_element_count + block_size - 1) / block_size;
      if (!strncmp(output_name, "OUTPUT0", strlen("OUTPUT0"))) {
        if (datatype_ == DataType::TYPE_INT32) {
          VecAddInt32<<<grid_size, block_size, 0, stream_>>>(
              reinterpret_cast<int32_t*>(cuda_input0_),
              reinterpret_cast<int32_t*>(cuda_input1_),
              reinterpret_cast<int32_t*>(cuda_output_), batchn_element_count);
        } else {
          VecAddFp32<<<grid_size, block_size, 0, stream_>>>(
              reinterpret_cast<float*>(cuda_input0_),
              reinterpret_cast<float*>(cuda_input1_),
              reinterpret_cast<float*>(cuda_output_), batchn_element_count);
        }
      } else {
        if (datatype_ == DataType::TYPE_INT32) {
          VecSubInt32<<<grid_size, block_size, 0, stream_>>>(
              reinterpret_cast<int32_t*>(cuda_input0_),
              reinterpret_cast<int32_t*>(cuda_input1_),
              reinterpret_cast<int32_t*>(cuda_output_), batchn_element_count);
        } else {
          VecSubFp32<<<grid_size, block_size, 0, stream_>>>(
              reinterpret_cast<float*>(cuda_input0_),
              reinterpret_cast<float*>(cuda_input1_),
              reinterpret_cast<float*>(cuda_output_), batchn_element_count);
        }
      }

      cudaError_t cuerr = cudaMemcpyAsync(
          obuffer, cuda_output_, batchn_byte_size, cudaMemcpyDeviceToHost,
          stream_);
      if (cuerr != cudaSuccess) {
        LOG_ERROR << "failed to copy output values from GPU for addsub: "
                  << cudaGetErrorString(cuerr);
        payload.error_code = kCudaMemcpy;
        break;
      }
    }
  }

  // Wait for all compute and memcpy to complete before returning.
  cudaStreamSynchronize(stream_);

  return kSuccess;
}

int
Context::Execute(
    const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
{
  if (gpu_device_ == CUSTOM_NO_GPU_DEVICE) {
    return ExecuteCPU(payload_cnt, payloads, input_fn, output_fn);
  } else {
    return ExecuteGPU(payload_cnt, payloads, input_fn, output_fn);
  }
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
    case kInputOutputShape:
      return "model must have two inputs and two outputs with the same shape";
    case kInputName:
      return "model inputs must be named 'INPUT0' and 'INPUT1'";
    case kOutputName:
      return "model outputs must be named 'OUTPUT0' and 'OUTPUT1'";
    case kInputOutputDataType:
      return "model inputs and outputs must have TYPE_INT32 or TYPE_FP32 "
             "data-type";
    case kInputContents:
      return "unable to get input tensor values";
    case kInputSize:
      return "unexpected size for input tensor";
    case kOutputBuffer:
      return "unable to get buffer for output tensor values";
    case kCudaMalloc:
      return "cudaMalloc failed";
    case kCudaMemcpy:
      return "cudaMemcpy failed";
    case kCudaStream:
      return "failed to create CUDA stream";
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

}}}}  // namespace nvidia::inferenceserver::custom::addsub
