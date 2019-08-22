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

#include "src/backends/pytorch/libtorch_backend.h"

#include <stdint.h>
#include <exception>
#include <memory>
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config_cuda.h"
#include "src/core/model_config_utils.h"
#include "src/core/provider.h"
#include "src/core/server_status.h"

#ifdef TRTIS_ENABLE_GPU
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>
#endif  // TRTIS_ENABLE_GPU

namespace nvidia { namespace inferenceserver {

LibTorchBackend::Context::Context(
    const std::string& name, const int gpu_device, const int max_batch_size)
    : BackendContext(name, gpu_device, max_batch_size),
      device_(torch::Device(torch::kCPU))
{
}

LibTorchBackend::Context::~Context()
{
  torch_model_.reset();
#ifdef TRTIS_ENABLE_GPU
  c10::cuda::CUDACachingAllocator::emptyCache();
#endif  // TRTIS_ENABLE_GPU
  LOG_VERBOSE(1) << "~LibTorchBackend::Context ";
}

std::pair<bool, torch::ScalarType>
ConvertDataTypeToTorchType(const DataType& dtype)
{
  torch::ScalarType type = torch::kInt;
  switch (dtype) {
    case TYPE_BOOL:
      type = torch::kBool;
      break;
    case TYPE_UINT8:
      type = torch::kByte;
      break;
    case TYPE_INT8:
      type = torch::kChar;
      break;
    case TYPE_INT16:
      type = torch::kShort;
      break;
    case TYPE_INT32:
      type = torch::kInt;
      break;
    case TYPE_INT64:
      type = torch::kLong;
      break;
    case TYPE_FP16:
      type = torch::kHalf;
      break;
    case TYPE_FP32:
      type = torch::kFloat;
      break;
    case TYPE_FP64:
      type = torch::kDouble;
      break;
    case TYPE_UINT16:
    case TYPE_UINT32:
    case TYPE_UINT64:
    case TYPE_STRING:
    default:
      return std::make_pair(false, type);
  }

  return std::make_pair(true, type);
}

DataType
ConvertTorchTypeToDataType(const torch::ScalarType& ttype)
{
  switch (ttype) {
    case torch::kBool:
      return TYPE_BOOL;
    case torch::kByte:
      return TYPE_UINT8;
    case torch::kChar:
      return TYPE_INT8;
    case torch::kShort:
      return TYPE_INT16;
    case torch::kInt:
      return TYPE_INT32;
    case torch::kLong:
      return TYPE_INT64;
    case torch::kHalf:
      return TYPE_FP16;
    case torch::kFloat:
      return TYPE_FP32;
    case torch::kDouble:
      return TYPE_FP64;
    default:
      return TYPE_FP32;
  }
}

Status
LibTorchBackend::Init(const std::string& path, const ModelConfig& config)
{
  RETURN_IF_ERROR(ValidateModelConfig(config, kPyTorchLibTorchPlatform));
  RETURN_IF_ERROR(SetModelConfig(path, config));

  return Status::Success;
}

Status
LibTorchBackend::CreateExecutionContexts(
    const std::unordered_map<std::string, std::string>& models)
{
  uint32_t total_context_cnt = 0;

  // Create a context for each instance.
  for (const auto& group : Config().instance_group()) {
    for (int c = 0; c < group.count(); c++) {
      if (group.kind() == ModelInstanceGroup::KIND_CPU) {
        const std::string instance_name =
            group.name() + "_" + std::to_string(c) + "_cpu";
        RETURN_IF_ERROR(CreateExecutionContext(
            instance_name, Context::NO_GPU_DEVICE, models));
        total_context_cnt++;
      } else {
        for (int gpu_device : group.gpus()) {
          const std::string instance_name = group.name() + "_" +
                                            std::to_string(c) + "_gpu" +
                                            std::to_string(gpu_device);
          RETURN_IF_ERROR(
              CreateExecutionContext(instance_name, gpu_device, models));
          total_context_cnt++;
        }
      }
    }
  }

  // Create a scheduler with one thread for each context available for
  // this model. Each runner is exclusively tied to the context.
  RETURN_IF_ERROR(SetConfiguredScheduler(
      total_context_cnt,
      [](uint32_t runner_idx) -> Status { return Status::Success; },
      [this](
          uint32_t runner_idx, std::vector<Scheduler::Payload>* payloads,
          std::function<void(Status)> func) {
        Run(runner_idx, payloads, func);
      }));

  return Status::Success;
}

Status
LibTorchBackend::CreateExecutionContext(
    const std::string& instance_name, const int gpu_device,
    const std::unordered_map<std::string, std::string>& models)
{
  // For a GPU context, determine the model file to use for device
  // compute capability. CPU always uses the default model file.
  std::string cc;
  std::string cc_model_filename;
  if (gpu_device == Context::NO_GPU_DEVICE) {
    cc_model_filename = Config().default_model_filename();
  } else {
#ifdef TRTIS_ENABLE_GPU
    cudaDeviceProp cuprops;
    cudaError_t cuerr = cudaGetDeviceProperties(&cuprops, gpu_device);
    if (cuerr != cudaSuccess) {
      return Status(
          RequestStatusCode::INTERNAL,
          "unable to get CUDA device properties for " + Name() + ": " +
              cudaGetErrorString(cuerr));
    }

    cc = std::to_string(cuprops.major) + "." + std::to_string(cuprops.minor);
    const auto& cc_itr = Config().cc_model_filenames().find(cc);
    cc_model_filename = (cc_itr == Config().cc_model_filenames().end())
                            ? Config().default_model_filename()
                            : cc_itr->second;
#else
    return Status(RequestStatusCode::INTERNAL, "GPU instances not supported");
#endif  // TRTIS_ENABLE_GPU
  }

  const auto& lp_itr = models.find(cc_model_filename);
  if (lp_itr == models.end()) {
    return Status(
        RequestStatusCode::INTERNAL, "unable to find LibTorch model '" +
                                         cc_model_filename + "' for " + Name());
  }

  if (gpu_device == Context::NO_GPU_DEVICE) {
    LOG_INFO << "Creating instance " << instance_name << " on CPU using "
             << cc_model_filename;
  } else {
    LOG_INFO << "Creating instance " << instance_name << " on GPU "
             << gpu_device << " (" << cc << ") using " << cc_model_filename;
  }

  // Max batch size. A value of 0 in the config becomes NO_BATCHING.
  const int mbs = (Config().max_batch_size() <= 0) ? Context::NO_BATCHING
                                                   : Config().max_batch_size();

  contexts_.emplace_back(new Context(instance_name, gpu_device, mbs));
  Context* context = contexts_.back().get();

  RETURN_IF_ERROR(context->CreateCudaStream());

  if (gpu_device == Context::NO_GPU_DEVICE) {
    context->device_ = torch::Device(torch::kCPU);
  } else {
    context->device_ = torch::Device(torch::kCUDA, gpu_device);
  }

  try {
    // lp_itr->second is the torch model serialized to string
    std::istringstream model_stream(lp_itr->second);
    context->torch_model_ = torch::jit::load(model_stream, context->device_);
  }
  catch (const std::exception& ex) {
    return Status(
        RequestStatusCode::INTERNAL, "load failed for libtorch model -> '" +
                                         Config().name() + "': " + ex.what());
  }

  RETURN_IF_ERROR(context->ValidateInputs(Config().input()));
  RETURN_IF_ERROR(context->ValidateOutputs(Config().output()));
  return Status::Success;
}

Status
LibTorchBackend::Context::ValidateInputs(
    const ::google::protobuf::RepeatedPtrField<ModelInput>& ios)
{
  std::string deliminator = "__";
  int ip_index;

  for (const auto& io : ios) {
    const auto pr = ConvertDataTypeToTorchType(io.data_type());
    if (!pr.first) {
      return Status(
          RequestStatusCode::INTERNAL,
          "unsupported datatype " + DataType_Name(io.data_type()) +
              " for input '" + io.name() + "' for model '" + name_ + "'");
    } else {
      const std::string& name = io.name();
      try {
        int start_pos = name.find(deliminator);
        if (start_pos == -1) {
          throw std::invalid_argument(
              "Input '" + name +
              "' does not follow naming convention i.e. <name>__<index>.");
        }
        ip_index = std::atoi(name.substr(start_pos + 2).c_str());
      }
      catch (std::exception& ex) {
        return Status(
            RequestStatusCode::INTERNAL,
            "Input '" + name +
                "' does not follow naming convention i.e. <name>__<index>.");
      }
      input_index_map_[name] = ip_index;
    }
  }

  return Status::Success;
}


Status
LibTorchBackend::Context::ValidateOutputs(
    const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios)
{
  std::string deliminator = "__";
  int op_index;

  for (const auto& io : ios) {
    const auto pr = ConvertDataTypeToTorchType(io.data_type());
    if (!pr.first) {
      return Status(
          RequestStatusCode::INTERNAL,
          "unsupported datatype " + DataType_Name(io.data_type()) +
              " for output '" + io.name() + "' for model '" + name_ + "'");
    } else {
      const std::string& name = io.name();
      try {
        int start_pos = name.find(deliminator);
        if (start_pos == -1) {
          throw std::invalid_argument(
              "Output '" + name +
              "' does not follow naming convention i.e. <name>__<index>.");
        }
        op_index = std::atoi(name.substr(start_pos + 2).c_str());
      }
      catch (std::exception& ex) {
        return Status(
            RequestStatusCode::INTERNAL,
            "Output '" + name +
                "' does not follow naming convention i.e. <name>__<index>.");
      }
      output_index_map_[name] = op_index;
    }
  }

  return Status::Success;
}

void
LibTorchBackend::Run(
    uint32_t runner_idx, std::vector<Scheduler::Payload>* payloads,
    std::function<void(Status)> OnCompleteQueuedPayloads)
{
  // Each runner executes using the corresponding context...
  if (runner_idx >= contexts_.size()) {
    OnCompleteQueuedPayloads(Status(
        RequestStatusCode::INTERNAL,
        "unexpected runner index" + std::to_string(runner_idx) +
            ", max allowed " + std::to_string(contexts_.size())));
    return;
  }

  std::vector<ModelInferStats::ScopedTimer> compute_timers;
  for (auto& payload : *payloads) {
    // Stop queue timer when the payload is scheduled to run
    if (payload.queue_timer_ != nullptr) {
      payload.queue_timer_.reset();
    }

    if (payload.stats_ != nullptr) {
      compute_timers.emplace_back();
      payload.stats_->StartComputeTimer(&compute_timers.back());
      payload.stats_->SetGPUDevice(contexts_[runner_idx]->gpu_device_);
    }
  }

  Status status = contexts_[runner_idx]->Run(this, payloads);
  // reset compute timers before calling OnComplete function
  compute_timers.clear();

  OnCompleteQueuedPayloads(status);
}

Status
LibTorchBackend::Context::SetFixedSizedInputTensor(
    std::vector<torch::jit::IValue>* inputs_, const std::string& name,
    const int& ip_index, const std::vector<int64_t>& shape,
    const DataType dtype, const size_t batch1_byte_size,
    const size_t total_byte_size, std::vector<Scheduler::Payload>* payloads,
    std::vector<std::unique_ptr<AllocatedSystemMemory>>* input_buffers,
    bool* cuda_copy)
{
  // The entire input tensor must be delivered as a single
  // contiguous chunk so create a buffer large enough to hold the
  // entire dynamic batched input.
  auto memory_type = (gpu_device_ == NO_GPU_DEVICE) ? TRTSERVER_MEMORY_CPU
                                                    : TRTSERVER_MEMORY_GPU;
  input_buffers->emplace_back();
  input_buffers->back().reset(
      new AllocatedSystemMemory(total_byte_size, memory_type));
  char* buffer = input_buffers->back()->MutableBuffer(&memory_type);

  // Visit the payloads in order and copy the input tensors to 'buffer'.
  std::vector<size_t> expected_byte_sizes;
  for (auto& payload : *payloads) {
    const InferRequestHeader& request_header =
        payload.request_provider_->RequestHeader();
    expected_byte_sizes.push_back(
        request_header.batch_size() * batch1_byte_size);
  }

  *cuda_copy |=
      SetInputBuffer(name, expected_byte_sizes, payloads, memory_type, buffer);

  RETURN_IF_ERROR(SetInputTensor(
      inputs_, name, ip_index, shape, dtype, static_cast<char*>(buffer),
      total_byte_size, memory_type));
  return Status::Success;
}

Status
LibTorchBackend::Context::SetInputTensor(
    std::vector<torch::jit::IValue>* inputs_, const std::string& name,
    const int& ip_index, const std::vector<int64_t>& shape,
    const DataType dtype, char* content, const size_t byte_size,
    const TRTSERVER_Memory_Type memory_type)
{
  const auto pr = ConvertDataTypeToTorchType(dtype);
  if (!pr.first) {
    return Status(
        RequestStatusCode::INTERNAL, "Failed to convert DataType '" +
                                         DataType_Name(dtype) +
                                         "' to Torch datatype");
  }
  torch::TensorOptions options{pr.second};
  auto updated_options = options.device(
      (memory_type == TRTSERVER_MEMORY_CPU) ? torch::kCPU : torch::kCUDA);
  torch::Tensor input_tensor =
      torch::from_blob(content, shape, updated_options);
  input_tensor = input_tensor.to(device_);

  if (input_tensor.nbytes() != byte_size) {
    return Status(
        RequestStatusCode::INTERNAL,
        "unexpected size " + std::to_string(byte_size) +
            " for inference input '" + name + "', expecting " +
            std::to_string(input_tensor.nbytes()));
  }
  (*inputs_)[ip_index] = input_tensor;

  return Status::Success;
}

Status
LibTorchBackend::Context::ReadFixedSizedOutputTensor(
    std::vector<torch::Tensor>* outputs_, const std::string& name,
    const int& op_index, const DataType dtype, const size_t dtype_byte_size,
    const size_t total_batch_size, const DimsList& dims,
    std::vector<Scheduler::Payload>* payloads, bool* cuda_copy)
{
  std::vector<int64_t> content_shape;
  void* content = nullptr;
  size_t byte_size = 0;
  RETURN_IF_ERROR(GetOutputTensor(
      outputs_, op_index, name, dtype, &content, &byte_size, &content_shape));

  // verify shape of output matches shape from model config
  const int batch_offset = ((max_batch_size_ == NO_BATCHING) ? 0 : 1);

  for (int i = 0; i < dims.size(); i++) {
    if (dims[i] != -1) {
      if (dims[i] != content_shape[i + batch_offset]) {
        return Status(
            RequestStatusCode::INVALID_ARG,
            "unexpected shape for output '" + name +
                "', model configuration shape is " +
                DimsListToString(content_shape) + ", inference shape is " +
                DimsListToString(dims));
      }
    }
  }

  const size_t total_byte_size =
      GetElementCount(content_shape) * dtype_byte_size;
  const size_t batch1_byte_size = total_byte_size / total_batch_size;

  if (byte_size != total_byte_size) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "unexpected size for output '" + name + "', byte-size " +
            std::to_string(byte_size) + " does not equal " +
            std::to_string(total_batch_size) + " * " +
            std::to_string(batch1_byte_size));
  }

  auto content_memory_type =
      (device_ == torch::kCPU) ? TRTSERVER_MEMORY_CPU : TRTSERVER_MEMORY_GPU;
  *cuda_copy |= SetFixedSizeOutputBuffer(
      name, batch1_byte_size, (char*)content, content_shape,
      content_memory_type, payloads);

  return Status::Success;
}

Status
LibTorchBackend::Context::GetOutputTensor(
    std::vector<torch::Tensor>* outputs_, const int& op_index,
    const std::string& name, const DataType dtype, void** content,
    size_t* byte_size, std::vector<int64_t>* content_shape)
{
  try {
    torch::Tensor output_flat = (*outputs_)[op_index].flatten();

    // verify output datatype matches datatype from model config
    DataType rec_dtype = ConvertTorchTypeToDataType(output_flat.scalar_type());
    if (dtype != rec_dtype) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "unexpected datatype " + DataType_Name(rec_dtype) +
              " for inference output '" + name + "', expecting " +
              DataType_Name(dtype));
    }

    *byte_size = output_flat.nbytes();

    // Copy output into buffer
    *content = output_flat.data_ptr();

    //  Set content shape
    auto shape = (*outputs_)[op_index].sizes();
    for (auto itr = shape.begin(); itr != shape.end(); itr++) {
      content_shape->push_back(*itr);
    }
  }
  catch (std::exception& ex) {
    return Status(RequestStatusCode::INTERNAL, "failed to get LibTorch output");
  }

  return Status::Success;
}

Status
LibTorchBackend::Context::SetInput(
    std::vector<torch::jit::IValue>* inputs_, const std::string& name,
    const int& ip_index, const DataType datatype, const DimsList& dims,
    const size_t total_batch_size, std::vector<Scheduler::Payload>* payloads,
    std::vector<std::unique_ptr<AllocatedSystemMemory>>* input_buffers,
    bool* cuda_copy)
{
  // Get the shape of the input. The provider has already checked that
  // the request shape is valid so don't need to do it here.
  std::vector<int64_t> shape;

  // If model supports batching then prepend the batch dimension
  // onto the input shape.
  if (max_batch_size_ != NO_BATCHING) {
    shape.push_back(total_batch_size);
  }

  size_t batch1_element_cnt = 1;
  for (auto dim : dims) {
    shape.push_back(dim);
    batch1_element_cnt *= dim;
  }

  // Checked at initialization time to make sure that STRING is not
  // being used for an input, so can just assume fixed-sized here.
  const size_t batch1_byte_size =
      batch1_element_cnt * GetDataTypeByteSize(datatype);
  const size_t total_byte_size = total_batch_size * batch1_byte_size;

  return SetFixedSizedInputTensor(
      inputs_, name, ip_index, shape, datatype, batch1_byte_size,
      total_byte_size, payloads, input_buffers, cuda_copy);
}

Status
LibTorchBackend::Context::Run(
    const LibTorchBackend* base, std::vector<Scheduler::Payload>* payloads)
{
  LOG_VERBOSE(1) << "Running " << name_ << " with " << payloads->size()
                 << " request payloads";

  std::shared_ptr<InferRequestProvider> input_request_provider;

  // For each request in 'payloads' collect the total batch size for
  // this inference execution. The batch-size, number of inputs, and
  // size of each input has already been checked by each payloads
  // request provider so don't need to do that here.
  size_t total_batch_size = 0;
  for (auto& payload : *payloads) {
    if (!payload.status_.IsOk()) {
      return Status(
          RequestStatusCode::INTERNAL,
          "unexpected payload with non-OK status given to runner for '" +
              name_ + "'");
    }

    total_batch_size += payload.request_provider_->RequestHeader().batch_size();

    // All payloads must have equally-sized input tensors so use any
    // payload as the representative for the input tensors.
    input_request_provider = payload.request_provider_;
  }

  // If there are no valid payloads then no need to run the
  // inference. The payloads will have their error status set so can
  // just return.
  if (total_batch_size == 0) {
    return Status::Success;
  }

  // total_batch_size can be 1 for models that don't support batching
  // (i.e. max_batch_size_ == 0).
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size_)) {
    return Status(
        RequestStatusCode::INTERNAL,
        "dynamic batch size " + std::to_string(total_batch_size) + " for '" +
            name_ + "', max allowed is " + std::to_string(max_batch_size_));
  }

  // Additional inputs added to the provider...
  const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
      input_override_map = input_request_provider->GetInputOverride();

  // Hold reference to each buffer of input data to that it stays
  // until the inference has completed.
  std::vector<std::unique_ptr<AllocatedSystemMemory>> input_buffers;

  size_t overide_inputs = 0;
  if (input_override_map != nullptr) {
    overide_inputs = input_override_map->size();
  }

  // Store input and output tensors
  std::vector<torch::jit::IValue> inputs_(
      input_request_provider->RequestHeader().input().size() + overide_inputs);
  std::vector<torch::Tensor> outputs_;

  // Inputs from the request...
  bool cuda_copy = false;
  for (const auto& input : input_request_provider->RequestHeader().input()) {
    const std::string& name = input.name();
    int ip_index = input_index_map_[name];
    const ModelInput* input_config;
    RETURN_IF_ERROR(base->GetInput(name, &input_config));

    RETURN_IF_ERROR(SetInput(
        &inputs_, name, ip_index, input_config->data_type(), input.dims(),
        total_batch_size, payloads, &input_buffers, &cuda_copy));
  }

  std::string deliminator = "__";
  int ip_index;

  if (input_override_map != nullptr) {
    for (const auto& pr : *input_override_map) {
      const std::string& name = pr.first;
      LOG_VERBOSE(1) << "Processing extra input: " << name;
      const std::shared_ptr<InferRequestProvider::InputOverride>& override =
          pr.second;
      try {
        int start_pos = name.find(deliminator);
        if (start_pos == -1) {
          throw std::invalid_argument(
              "Input '" + name +
              "' does not follow naming convention i.e. <name>__<index>.");
        }
        ip_index = std::atoi(name.substr(start_pos + 2).c_str());
      }
      catch (std::exception& ex) {
        return Status(
            RequestStatusCode::INTERNAL,
            "Input '" + name +
                "' does not follow naming convention i.e. <name>__<index>.");
      }
      input_index_map_[name] = ip_index;
      RETURN_IF_ERROR(SetInput(
          &inputs_, name, ip_index, override->datatype_, override->dims_,
          total_batch_size, payloads, &input_buffers, &cuda_copy));
    }
  }
#ifdef TRTIS_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif  // TRTIS_ENABLE_GPU

  // Run...
  RETURN_IF_ERROR(Execute(&inputs_, &outputs_));

  // verify output indices are valid with number of outputs after execution
  for (const auto& output : base->Config().output()) {
    int op_index = output_index_map_[output.name()];
    int max_index = outputs_.size() - 1;
    if ((op_index < 0) || (op_index > max_index)) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "The output " + output.name() +
              " in the model config refers to an output index which doesn't "
              "exist. This model has " +
              std::to_string(max_index + 1) + " outputs");
    }
  }

  // Prepare set of Outputs requested for
  std::set<std::string> required_outputs;
  for (auto& payload : *payloads) {
    const InferRequestHeader& request_header =
        payload.request_provider_->RequestHeader();
    for (const auto& output : request_header.output()) {
      required_outputs.insert(output.name());
    }
  }

  // Ensure outputs have the expected size and copy it to the payload responses.
  cuda_copy = false;
  for (const auto& name : required_outputs) {
    int op_index = output_index_map_[name];
    const ModelOutput* output_config;
    RETURN_IF_ERROR(base->GetOutput(name, &output_config));

    const DataType dtype = output_config->data_type();

    // If a reshape is provided for the output then use that when
    // validating that the model matches what is expected.
    const DimsList& output_dims = (output_config->has_reshape())
                                      ? output_config->reshape().shape()
                                      : output_config->dims();

    // Checked at initialization time to make sure that STRING is not
    // being used for an output, so can just assume fixed-sized here.
    RETURN_IF_ERROR(ReadFixedSizedOutputTensor(
        &outputs_, name, op_index, dtype,
        GetDataTypeByteSize(output_config->data_type()), total_batch_size,
        output_dims, payloads, &cuda_copy));
  }

#ifdef TRTIS_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif  // TRTIS_ENABLE_GPU

  return Status::Success;
}

Status
LibTorchBackend::Context::Execute(
    std::vector<torch::jit::IValue>* inputs_,
    std::vector<torch::Tensor>* outputs_)
{
  torch::jit::IValue model_outputs_;

  try {
    model_outputs_ = torch_model_->forward(*inputs_);
    auto model_outputs_tuple = model_outputs_.toTuple();
    for (auto& m_op : model_outputs_tuple->elements()) {
      outputs_->push_back(m_op.toTensor());
    }
  }
  catch (std::exception& ex) {
    try {
      auto model_output_tensor = model_outputs_.toTensor();
      outputs_->push_back(model_output_tensor);
    }
    catch (std::exception& exx) {
      LOG_VERBOSE(1) << ex.what();
      return Status(
          RequestStatusCode::INTERNAL, "failed to run model '" + name_);
    }
  }

  return Status::Success;
}

std::ostream&
operator<<(std::ostream& out, const LibTorchBackend& pb)
{
  out << "name=" << pb.Name() << std::endl;
  out << "contexts:" << std::endl;
  for (const auto& context : pb.contexts_) {
    out << "  name=" << context->name_ << ", gpu="
        << ((context->gpu_device_ == LibTorchBackend::Context::NO_GPU_DEVICE)
                ? "<none>"
                : std::to_string(context->gpu_device_));
  }

  return out;
}

}}  // namespace nvidia::inferenceserver
