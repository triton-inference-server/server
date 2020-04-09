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

#include "src/backends/pytorch/libtorch_backend.h"

#include <stdint.h>
#include <exception>
#include <memory>
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config_cuda.h"
#include "src/core/model_config_utils.h"
#include "src/core/server_status.h"

#ifdef TRTIS_ENABLE_GPU
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>
#include "src/core/cuda_utils.h"
#endif  // TRTIS_ENABLE_GPU

namespace nvidia { namespace inferenceserver {

struct LibTorchBackend::Context::InputMetaData {
  std::string name_;
  std::vector<int64_t> shape_;
  torch::ScalarType torch_type_;
  std::unique_ptr<AllocatedMemory> input_buffer_;
};

LibTorchBackend::Context::Context(
    const std::string& name, const int gpu_device, const int max_batch_size,
    const bool enable_pinned_input, const bool enable_pinned_output)
    : BackendContext(
          name, gpu_device, max_batch_size, enable_pinned_input,
          enable_pinned_output),
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
      },
      [this](
          uint32_t runner_idx, const InferenceRequest::Input& input,
          const Scheduler::Payload& payload,
          std::vector<int64_t>* shape) -> Status { return Status::Success; }));

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
          Status::Code::INTERNAL, "unable to get CUDA device properties for " +
                                      Name() + ": " +
                                      cudaGetErrorString(cuerr));
    }

    cc = std::to_string(cuprops.major) + "." + std::to_string(cuprops.minor);
    const auto& cc_itr = Config().cc_model_filenames().find(cc);
    cc_model_filename = (cc_itr == Config().cc_model_filenames().end())
                            ? Config().default_model_filename()
                            : cc_itr->second;
#else
    return Status(Status::Code::INTERNAL, "GPU instances not supported");
#endif  // TRTIS_ENABLE_GPU
  }

  const auto& lp_itr = models.find(cc_model_filename);
  if (lp_itr == models.end()) {
    return Status(
        Status::Code::INTERNAL, "unable to find LibTorch model '" +
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
  const bool pinned_input =
      Config().optimization().input_pinned_memory().enable();
  const bool pinned_output =
      Config().optimization().output_pinned_memory().enable();

  contexts_.emplace_back(
      new Context(instance_name, gpu_device, mbs, pinned_input, pinned_output));
  Context* context = static_cast<Context*>(contexts_.back().get());

  RETURN_IF_ERROR(context->CreateCudaStream());

  if (gpu_device == Context::NO_GPU_DEVICE) {
    context->device_ = torch::Device(torch::kCPU);
  } else {
    context->device_ = torch::Device(torch::kCUDA, gpu_device);
  }

  try {
    // lp_itr->second is the torch model serialized to string
    std::istringstream model_stream(lp_itr->second);
    context->torch_model_ = std::make_shared<torch::jit::script::Module>(
        torch::jit::load(model_stream, context->device_));
  }
  catch (const std::exception& ex) {
    return Status(
        Status::Code::INTERNAL, "load failed for libtorch model -> '" +
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
          Status::Code::INTERNAL,
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
            Status::Code::INTERNAL,
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
          Status::Code::INTERNAL,
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
            Status::Code::INTERNAL,
            "Output '" + name +
                "' does not follow naming convention i.e. <name>__<index>.");
      }
      output_index_map_[name] = op_index;
    }
  }

  return Status::Success;
}

Status
LibTorchBackend::Context::SetInputTensor(
    const InputMetaData& meta_data, torch::jit::IValue* tensor)
{
  TRTSERVER_Memory_Type memory_type;
  int64_t memory_type_id;
  size_t total_byte_size = meta_data.input_buffer_->TotalByteSize();
  char* buffer =
      meta_data.input_buffer_->MutableBuffer(&memory_type, &memory_type_id);

  torch::TensorOptions options{meta_data.torch_type_};
  auto updated_options = (memory_type == TRTSERVER_MEMORY_GPU)
                             ? options.device(torch::kCUDA, memory_type_id)
                             : options.device(torch::kCPU);
  torch::Tensor input_tensor =
      torch::from_blob(buffer, meta_data.shape_, updated_options);

  if (input_tensor.nbytes() != total_byte_size) {
    return Status(
        Status::Code::INTERNAL,
        "unexpected size " + std::to_string(total_byte_size) +
            " for inference input '" + meta_data.name_ + "', expecting " +
            std::to_string(input_tensor.nbytes()));
  }
  *tensor = input_tensor;

  return Status::Success;
}

Status
LibTorchBackend::Context::ReadFixedSizedOutputTensor(
    std::vector<torch::Tensor>* outputs_, const std::string& name,
    const int& op_index, const DataType dtype, const size_t dtype_byte_size,
    const size_t total_batch_size, const DimsList& dims,
    std::vector<Scheduler::Payload>* payloads, OutputInfo* output,
    bool* cuda_copy)
{
  size_t byte_size = 0;
  RETURN_IF_ERROR(GetOutputTensor(
      outputs_, op_index, name, dtype, &output->output_buffer_, &byte_size,
      &output->output_shape_));

  // verify shape of output matches shape from model config
  RETURN_IF_ERROR(CompareOutputDims(
      name, output->output_shape_, dims,
      max_batch_size_ != NO_BATCHING /* supports_batching */));

  const size_t total_byte_size =
      GetElementCount(output->output_shape_) * dtype_byte_size;
  const size_t batch1_byte_size = total_byte_size / total_batch_size;

  if (byte_size != total_byte_size) {
    return Status(
        Status::Code::INVALID_ARG,
        "unexpected size for output '" + name + "', byte-size " +
            std::to_string(byte_size) + " does not equal " +
            std::to_string(total_batch_size) + " * " +
            std::to_string(batch1_byte_size));
  }

  output->memory_type_ =
      (device_ == torch::kCPU) ? TRTSERVER_MEMORY_CPU : TRTSERVER_MEMORY_GPU;
  output->memory_type_id_ = (device_ == torch::kCPU) ? 0 : gpu_device_;
  *cuda_copy |=
      SetFixedSizeOutputBuffer(name, batch1_byte_size, output, payloads);

  return Status::Success;
}

Status
LibTorchBackend::Context::GetOutputTensor(
    std::vector<torch::Tensor>* outputs_, const int& op_index,
    const std::string& name, const DataType dtype, const char** content,
    size_t* byte_size, std::vector<int64_t>* content_shape)
{
  try {
    torch::Tensor output_flat = (*outputs_)[op_index].contiguous().flatten();

    // verify output datatype matches datatype from model config
    DataType rec_dtype = ConvertTorchTypeToDataType(output_flat.scalar_type());
    if (dtype != rec_dtype) {
      return Status(
          Status::Code::INVALID_ARG,
          "unexpected datatype " + DataType_Name(rec_dtype) +
              " for inference output '" + name + "', expecting " +
              DataType_Name(dtype));
    }

    *byte_size = output_flat.nbytes();
    *content = static_cast<const char*>(output_flat.data_ptr());

    //  Set content shape
    auto shape = (*outputs_)[op_index].sizes();
    for (auto itr = shape.begin(); itr != shape.end(); itr++) {
      content_shape->push_back(*itr);
    }
  }
  catch (std::exception& ex) {
    return Status(Status::Code::INTERNAL, "failed to get LibTorch output");
  }

  return Status::Success;
}

Status
LibTorchBackend::Context::SetInputMetaData(
    const std::string& name, const DataType datatype,
    const std::vector<int64_t>& dims, const size_t total_batch_size,
    std::vector<Scheduler::Payload>* payloads, std::vector<InputInfo>* inputs,
    InputMetaData* meta_data, bool* cuda_copy)
{
  meta_data->name_ = name;
  // Get the shape of the input. The request normalizer has already
  // checked that the request shape is valid so don't need to do it
  // here.
  meta_data->shape_.clear();

  // If model supports batching then prepend the batch dimension
  // onto the input shape.
  if (max_batch_size_ != NO_BATCHING) {
    meta_data->shape_.push_back(total_batch_size);
  }

  size_t batch1_element_cnt = 1;
  for (auto dim : dims) {
    meta_data->shape_.push_back(dim);
    batch1_element_cnt *= dim;
  }

  const auto pr = ConvertDataTypeToTorchType(datatype);
  if (!pr.first) {
    return Status(
        Status::Code::INTERNAL, "Failed to convert DataType '" +
                                    DataType_Name(datatype) +
                                    "' to Torch datatype");
  }
  meta_data->torch_type_ = pr.second;

  // Checked at initialization time to make sure that STRING is not
  // being used for an input, so can just assume fixed-sized here.
  const size_t batch1_byte_size =
      batch1_element_cnt * GetDataTypeByteSize(datatype);
  const auto total_byte_size = total_batch_size * batch1_byte_size;

  inputs->emplace_back();

  return SetFixedSizedInputBuffer(
      name, batch1_byte_size, total_byte_size, payloads, &inputs->back(),
      meta_data, cuda_copy);
}

Status
LibTorchBackend::Context::SetFixedSizedInputBuffer(
    const std::string& name, const size_t batch1_byte_size,
    const size_t total_byte_size, std::vector<Scheduler::Payload>* payloads,
    InputInfo* input, InputMetaData* meta_data, bool* cuda_copy)
{
  // The entire input tensor must be delivered as a single
  // contiguous chunk so create a buffer large enough to hold the
  // entire dynamic batched input.
  auto memory_type = (gpu_device_ == NO_GPU_DEVICE)
                         ? TRTSERVER_MEMORY_CPU_PINNED
                         : TRTSERVER_MEMORY_GPU;
  int64_t memory_type_id = (gpu_device_ == NO_GPU_DEVICE) ? 0 : gpu_device_;
  meta_data->input_buffer_.reset(
      new AllocatedMemory(total_byte_size, memory_type, memory_type_id));
  input->input_buffer_ = meta_data->input_buffer_->MutableBuffer(
      &input->memory_type_, &input->memory_type_id_);

  // Visit the payloads in order and copy the input tensors to 'buffer'.
  std::vector<size_t> expected_byte_sizes;
  for (auto& payload : *payloads) {
    const auto& irequest = payload.request_;
    expected_byte_sizes.push_back(irequest->BatchSize() * batch1_byte_size);
  }

  *cuda_copy |= SetInputBuffer(name, expected_byte_sizes, payloads, input);

  return Status::Success;
}

Status
LibTorchBackend::Context::Run(
    const InferenceBackend* base, std::vector<Scheduler::Payload>* payloads)
{
  LOG_VERBOSE(1) << "Running " << name_ << " with " << payloads->size()
                 << " request payloads";

  const InferenceRequest* repr_input_request = nullptr;

  // For each request in 'payloads' collect the total batch size for
  // this inference execution. The batch-size, number of inputs, and
  // size of each input has already been checked by each payloads
  // request provider so don't need to do that here.
  size_t total_batch_size = 0;
  for (auto& payload : *payloads) {
    if (!payload.status_.IsOk()) {
      return Status(
          Status::Code::INTERNAL,
          "unexpected payload with non-OK status given to runner for '" +
              name_ + "'");
    }

    total_batch_size += payload.request_->BatchSize();

    // All payloads must have equally-sized input tensors so use any
    // payload as the representative for the input tensors.
    repr_input_request = payload.request_.get();
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
        Status::Code::INTERNAL,
        "dynamic batch size " + std::to_string(total_batch_size) + " for '" +
            name_ + "', max allowed is " + std::to_string(max_batch_size_));
  }

  size_t input_count = repr_input_request->ImmutableInputs().size();

  // Hold reference to each buffer of input data so that it stays
  // until the inference has completed.
  std::vector<InputMetaData> input_meta_data(input_count);

  // Store input and output tensors
  std::vector<torch::jit::IValue> inputs_(input_count);
  std::vector<torch::Tensor> outputs_;
  std::vector<InputInfo> inputs;

  // Collect input metadata. FIXME override inputs from controls
  // should be known from the model configuration at load time and so
  // they should be processed then to initialze
  // input_index_map_. Since they are not we do it here for every
  // request which is unnecessary perf overhead.
  bool cuda_copy = false;
  for (const auto& pr : repr_input_request->ImmutableInputs()) {
    const InferenceRequest::Input* input = pr.second;
    const std::string& name = input->Name();
    int ip_index;

    const auto& itr = input_index_map_.find(name);
    if (itr != input_index_map_.end()) {
      ip_index = itr->second;
    } else {
      static const std::string deliminator = "__";

      LOG_VERBOSE(1) << "Processing override input: " << name;

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
            Status::Code::INTERNAL,
            "Input '" + name +
                "' does not follow naming convention i.e. <name>__<index>.");
      }

      input_index_map_[name] = ip_index;
    }

    RETURN_IF_ERROR(SetInputMetaData(
        name, input->DType(), input->Shape(), total_batch_size, payloads,
        &inputs, &(input_meta_data[ip_index]), &cuda_copy));
  }

#ifdef TRTIS_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
  cuda_copy = false;
  for (auto& input : inputs) {
    for (auto& indirect_buffer : input.indirect_buffers_) {
      bool cuda_used;
      TRTSERVER_Memory_Type buffer_memory_type;
      int64_t buffer_memory_id;
      size_t buffer_byte_size;
      auto buffer =
          std::get<0>(indirect_buffer)
              ->BufferAt(
                  0, &buffer_byte_size, &buffer_memory_type, &buffer_memory_id);
      auto status = CopyBuffer(
          "indirect buffer", buffer_memory_type, buffer_memory_id,
          input.memory_type_, input.memory_type_id_, buffer_byte_size, buffer,
          input.input_buffer_ + std::get<1>(indirect_buffer), stream_,
          &cuda_used);
      if (!status.IsOk()) {
        for (const auto& payload_idx : std::get<2>(indirect_buffer)) {
          (*payloads)[payload_idx].status_ = status;
        }
      } else {
        cuda_copy |= cuda_used;
      }
    }
  }
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif  // TRTIS_ENABLE_GPU

  for (size_t i = 0; i < inputs_.size(); i++) {
    SetInputTensor(input_meta_data[i], &(inputs_[i]));
  }

#ifdef TRTIS_ENABLE_STATS
  for (auto& payload : *payloads) {
    if (payload.stats_ != nullptr) {
      payload.stats_->CaptureTimestamp(
          ModelInferStats::TimestampKind::kComputeInputEnd);
    }
  }
#endif  // TRTIS_ENABLE_STATS

  // Run...
  RETURN_IF_ERROR(Execute(&inputs_, &outputs_));

#ifdef TRTIS_ENABLE_STATS
  for (auto& payload : *payloads) {
    if (payload.stats_ != nullptr) {
      payload.stats_->CaptureTimestamp(
          ModelInferStats::TimestampKind::kComputeOutputStart);
    }
  }
#endif  // TRTIS_ENABLE_STATS

  // verify output indices are valid with number of outputs after execution
  for (const auto& output : base->Config().output()) {
    int op_index = output_index_map_[output.name()];
    int max_index = outputs_.size() - 1;
    if ((op_index < 0) || (op_index > max_index)) {
      return Status(
          Status::Code::INVALID_ARG,
          "The output " + output.name() +
              " in the model configuration refers to an output index which "
              "doesn't exist. This model has " +
              std::to_string(max_index + 1) + " outputs");
    }
  }

  // Prepare set of Outputs requested for
  std::set<std::string> required_outputs;
  for (auto& payload : *payloads) {
    const auto& irequest = payload.request_;
    for (const auto& pr : irequest->RequestedOutputs()) {
      required_outputs.insert(pr.first);
    }
  }

  // Ensure outputs have the expected size and copy it to the payload responses.
  std::vector<OutputInfo> outputs;
  cuda_copy = false;
  for (const auto& name : required_outputs) {
    outputs.emplace_back();
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
        output_dims, payloads, &outputs.back(), &cuda_copy));
  }

#ifdef TRTIS_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
  cuda_copy = false;
  for (auto& output : outputs) {
    for (auto& indirect_buffer : output.indirect_buffers_) {
      bool cuda_used;
      TRTSERVER_Memory_Type src_memory_type;
      int64_t src_memory_type_id;
      // placeholder, copy byte size is determined by dst_byte_size
      size_t src_byte_size;
      auto src = indirect_buffer.first->BufferAt(
          0, &src_byte_size, &src_memory_type, &src_memory_type_id);
      TRTSERVER_Memory_Type dst_memory_type;
      int64_t dst_memory_type_id;
      for (auto& payload_output : indirect_buffer.second) {
        char* dst = payload_output.second->MutableBuffer(
            &dst_memory_type, &dst_memory_type_id);
        auto dst_byte_size = payload_output.second->TotalByteSize();
        (*payloads)[payload_output.first].status_ = CopyBuffer(
            "indirect buffer", src_memory_type, src_memory_type_id,
            dst_memory_type, dst_memory_type_id, dst_byte_size, src, dst,
            stream_, &cuda_used);
        cuda_copy |= cuda_used;
        src += dst_byte_size;
      }
    }
  }
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
      return Status(Status::Code::INTERNAL, "failed to run model '" + name_);
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
