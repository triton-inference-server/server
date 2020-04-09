// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "src/backends/caffe2/netdef_backend.h"

#include <stdint.h>
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/model_config_utils.h"
#include "src/core/server_status.h"

#ifdef TRTIS_ENABLE_GPU
#include <cuda_runtime_api.h>
#include "src/core/cuda_utils.h"
#endif  // TRTIS_ENABLE_GPU

namespace nvidia { namespace inferenceserver {

namespace {

// Convert model datatype to non-protobuf equivalent datatype required
// by Caffe2Workspace.
Caffe2Workspace::DataType
ConvertDataType(DataType dtype)
{
  switch (dtype) {
    case DataType::TYPE_INVALID:
      return Caffe2Workspace::DataType::TYPE_INVALID;
    case DataType::TYPE_BOOL:
      return Caffe2Workspace::DataType::TYPE_BOOL;
    case DataType::TYPE_UINT8:
      return Caffe2Workspace::DataType::TYPE_UINT8;
    case DataType::TYPE_UINT16:
      return Caffe2Workspace::DataType::TYPE_UINT16;
    case DataType::TYPE_UINT32:
      return Caffe2Workspace::DataType::TYPE_UINT32;
    case DataType::TYPE_UINT64:
      return Caffe2Workspace::DataType::TYPE_UINT64;
    case DataType::TYPE_INT8:
      return Caffe2Workspace::DataType::TYPE_INT8;
    case DataType::TYPE_INT16:
      return Caffe2Workspace::DataType::TYPE_INT16;
    case DataType::TYPE_INT32:
      return Caffe2Workspace::DataType::TYPE_INT32;
    case DataType::TYPE_INT64:
      return Caffe2Workspace::DataType::TYPE_INT64;
    case DataType::TYPE_FP16:
      return Caffe2Workspace::DataType::TYPE_FP16;
    case DataType::TYPE_FP32:
      return Caffe2Workspace::DataType::TYPE_FP32;
    case DataType::TYPE_FP64:
      return Caffe2Workspace::DataType::TYPE_FP64;
    default:
      break;
  }

  return Caffe2Workspace::DataType::TYPE_INVALID;
}

}  // namespace


NetDefBackend::Context::Context(
    const std::string& name, const int gpu_device, const int max_batch_size,
    const bool enable_pinned_input, const bool enable_pinned_output)
    : BackendContext(
          name, gpu_device, max_batch_size, enable_pinned_input,
          enable_pinned_output)
{
}

NetDefBackend::Context::~Context()
{
  LOG_VERBOSE(1) << "~NetDefBackend::Context ";
}

Status
NetDefBackend::CreateExecutionContexts(
    const std::unordered_map<std::string, std::vector<char>>& models)
{
  uint32_t total_context_cnt = 0;

  // Create a workspace for each instance.
  //
  // TODO [DLIS-52] Can this be optimized by sharing a workspace
  // (across all instances?).
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

  LOG_VERBOSE(1) << "netdef backend for " << Name() << std::endl << *this;
  return Status::Success;
}

Status
NetDefBackend::CreateExecutionContext(
    const std::string& instance_name, const int gpu_device,
    const std::unordered_map<std::string, std::vector<char>>& models)
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

  const auto& mn_itr = models.find(cc_model_filename);
  if (mn_itr == models.end()) {
    return Status(
        Status::Code::INTERNAL, "unable to find NetDef model '" +
                                    cc_model_filename + "' for " + Name());
  }

  // NetDef also requires an init network, the name of which is always
  // derived from 'cc_model_filename'.
  const std::string& cc_init_filename =
      kCaffe2NetDefInitFilenamePrefix + cc_model_filename;
  const auto& imn_itr = models.find(cc_init_filename);
  if (imn_itr == models.end()) {
    return Status(
        Status::Code::INTERNAL, "unable to find NetDef initialization model '" +
                                    cc_init_filename + "' for " + Name());
  }

  if (gpu_device == Context::NO_GPU_DEVICE) {
    LOG_INFO << "Creating instance " << instance_name << " on CPU using "
             << cc_init_filename << " and " << cc_model_filename;
  } else {
    LOG_INFO << "Creating instance " << instance_name << " on GPU "
             << gpu_device << " (" << cc << ") using " << cc_init_filename
             << " and " << cc_model_filename;
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

  // Extract input and output names from the config...
  std::vector<std::string> input_names;
  for (const auto& io : Config().input()) {
    input_names.push_back(io.name());
  }
  std::vector<std::string> output_names;
  for (const auto& io : Config().output()) {
    output_names.push_back(io.name());
  }

  // If this is a sequence model then make sure the require control
  // inputs are available in the model.
  if (Config().has_sequence_batching()) {
    RETURN_IF_ERROR(ValidateBooleanSequenceControl(
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_START, &input_names,
        false /* required */));
    RETURN_IF_ERROR(ValidateBooleanSequenceControl(
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_END, &input_names,
        false /* required */));
    RETURN_IF_ERROR(ValidateBooleanSequenceControl(
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_READY, &input_names,
        false /* required */));
    RETURN_IF_ERROR(ValidateTypedSequenceControl(
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_CORRID, &input_names,
        false /* required */));
  }

  try {
    // Create a Caffe2 workspace. We can't cross the raw protobuf
    // across this boundary (since Caffe2 build may use a different
    // protobuf).
    Caffe2Workspace* c2ws;
    Caffe2Workspace::Error err = Caffe2WorkspaceCreate(
        &c2ws, Config().name(), Config().max_batch_size(), input_names,
        output_names, gpu_device, imn_itr->second, mn_itr->second);
    if (!err.IsOk()) {
      return Status(Status::Code::INTERNAL, err.Message());
    }

    context->workspace_.reset(c2ws);
  }
  catch (const std::exception& ex) {
    return Status(
        Status::Code::INTERNAL,
        "load failed for '" + Config().name() + "': " + ex.what());
  }

  RETURN_IF_ERROR(context->ValidateInputs(Config().input()));
  RETURN_IF_ERROR(context->ValidateOutputs(Config().output()));

  return Status::Success;
}

Status
NetDefBackend::ValidateBooleanSequenceControl(
    const ModelSequenceBatching::Control::Kind control_kind,
    std::vector<std::string>* input_names, bool required)
{
  std::string tensor_name;
  RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
      Config().sequence_batching(), Name(), control_kind, required,
      &tensor_name, nullptr, nullptr, nullptr, nullptr, nullptr));
  if (!tensor_name.empty()) {
    input_names->push_back(tensor_name);
  }

  return Status::Success;
}

Status
NetDefBackend::ValidateTypedSequenceControl(
    const ModelSequenceBatching::Control::Kind control_kind,
    std::vector<std::string>* input_names, bool required)
{
  std::string tensor_name;
  RETURN_IF_ERROR(GetTypedSequenceControlProperties(
      Config().sequence_batching(), Name(), control_kind, required,
      &tensor_name, nullptr));
  if (!tensor_name.empty()) {
    input_names->push_back(tensor_name);
  }

  return Status::Success;
}

Status
NetDefBackend::Context::ValidateInputs(
    const ::google::protobuf::RepeatedPtrField<ModelInput>& ios)
{
  for (const auto& io : ios) {
    // For now, skipping the check if potential names is empty
    if (!workspace_->PotentialInputNames().empty()) {
      RETURN_IF_ERROR(
          CheckAllowedModelInput(io, workspace_->PotentialInputNames()));
    }

    if (ConvertDataType(io.data_type()) ==
        Caffe2Workspace::DataType::TYPE_INVALID) {
      return Status(
          Status::Code::INTERNAL,
          "unsupported datatype " + DataType_Name(io.data_type()) +
              " for input '" + io.name() + "' for model '" + name_ + "'");
    }
  }

  return Status::Success;
}


Status
NetDefBackend::Context::ValidateOutputs(
    const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios)
{
  for (const auto& io : ios) {
    // For now, skipping the check if potential names is empty
    if (!workspace_->PotentialOutputNames().empty()) {
      RETURN_IF_ERROR(
          CheckAllowedModelOutput(io, workspace_->PotentialOutputNames()));
    }

    if (ConvertDataType(io.data_type()) ==
        Caffe2Workspace::DataType::TYPE_INVALID) {
      return Status(
          Status::Code::INTERNAL,
          "unsupported datatype " + DataType_Name(io.data_type()) +
              " for output '" + io.name() + "' for model '" + name_ + "'");
    }
  }

  return Status::Success;
}

Status
NetDefBackend::Context::SetFixedSizedInputTensor(
    const std::string& name, const std::vector<int64_t>& shape,
    const Caffe2Workspace::DataType dtype, const size_t batch1_byte_size,
    const size_t total_byte_size, std::vector<Scheduler::Payload>* payloads,
    InputInfo* input, bool* cuda_copy)
{
  // Visit the payloads in order and copy the input tensors to
  // 'buffer'.
  std::vector<size_t> expected_byte_sizes;
  for (auto& payload : *payloads) {
    const auto& irequest = payload.request_;
    expected_byte_sizes.push_back(irequest->BatchSize() * batch1_byte_size);
  }

  *cuda_copy |= SetInputBuffer(name, expected_byte_sizes, payloads, input);

  Caffe2Workspace::Error err = workspace_->SetInputTensor(
      name, shape, dtype, static_cast<const char*>(input->input_buffer_),
      total_byte_size);
  if (!err.IsOk()) {
    return Status(Status::Code::INTERNAL, err.Message());
  }

  return Status::Success;
}

Status
NetDefBackend::Context::ReadFixedSizedOutputTensor(
    const std::string& name, const Caffe2Workspace::DataType dtype,
    const size_t dtype_byte_size, const size_t total_batch_size,
    const DimsList& dims, std::vector<Scheduler::Payload>* payloads,
    OutputInfo* output, bool* cuda_copy)
{
  // [TODO] use the following statement. Right now we always create
  // netdef workspace with inputs / outputs on CPU node
  // auto content_memory_type = (gpu_device_ == NO_GPU_DEVICE)
  //                                ? TRTSERVER_MEMORY_CPU
  //                                : TRTSERVER_MEMORY_GPU;
  output->memory_type_ = TRTSERVER_MEMORY_CPU;
  output->memory_type_id_ = 0;
  size_t byte_size = 0;
  Caffe2Workspace::Error err = workspace_->GetOutputTensor(
      name, dtype, &output->output_buffer_, &byte_size, &output->output_shape_);
  if (!err.IsOk()) {
    return Status(Status::Code::INTERNAL, err.Message());
  }

  // verify shape of output matches shape from model config
  RETURN_IF_ERROR(CompareOutputDims(
      name, output->output_shape_, dims,
      max_batch_size_ != NO_BATCHING /* supports_batching */));

  const size_t total_byte_size =
      GetElementCount(output->output_shape_) * dtype_byte_size;
  const size_t batch1_byte_size = total_byte_size / total_batch_size;

  if (byte_size != total_byte_size) {
    return Status(
        Status::Code::INTERNAL,
        "unexpected size for output '" + name + "', byte-size " +
            std::to_string(byte_size) + " does not equal " +
            std::to_string(total_batch_size) + " * " +
            std::to_string(batch1_byte_size));
  }

  *cuda_copy |=
      SetFixedSizeOutputBuffer(name, batch1_byte_size, output, payloads);
  return Status::Success;
}

Status
NetDefBackend::Context::SetInput(
    const std::string& name, const DataType datatype,
    const std::vector<int64_t>& dims, const size_t total_batch_size,
    std::vector<Scheduler::Payload>* payloads,
    std::vector<std::unique_ptr<AllocatedMemory>>* input_buffers,
    std::vector<InputInfo>* inputs, bool* cuda_copy)
{
  // Get the shape of the input. Request normalize already checked
  // that the request shape is valid so don't need to do it here.
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
  const Caffe2Workspace::DataType dtype = ConvertDataType(datatype);
  const size_t batch1_byte_size =
      batch1_element_cnt * GetDataTypeByteSize(datatype);
  const size_t total_byte_size = total_batch_size * batch1_byte_size;

  // The entire input tensor must be delivered as a single
  // contiguous chunk so create a buffer large enough to hold the
  // entire dynamic batched input.
  input_buffers->emplace_back(
      new AllocatedMemory(total_byte_size, TRTSERVER_MEMORY_CPU_PINNED, 0));
  inputs->emplace_back();
  auto& input = inputs->back();
  input.input_buffer_ = input_buffers->back()->MutableBuffer(
      &input.memory_type_, &input.memory_type_id_);

  return SetFixedSizedInputTensor(
      name, shape, dtype, batch1_byte_size, total_byte_size, payloads, &input,
      cuda_copy);
}

Status
NetDefBackend::Context::Run(
    const InferenceBackend* base, std::vector<Scheduler::Payload>* payloads)
{
  LOG_VERBOSE(1) << "Running " << name_ << " with " << payloads->size()
                 << " request payloads";

  const InferenceRequest* repr_input_request = nullptr;

  // For each request in 'payloads' collect the total batch size for
  // this inference execution. The batch-size, number of inputs, and
  // size of each input has already been checked by each request
  // normalizer so don't need to do that here.
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

  // Hold reference to each buffer of input data to that it stays
  // until the inference has completed.
  std::vector<std::unique_ptr<AllocatedMemory>> input_buffers;
  std::vector<InputInfo> inputs;

  // Create a tensor for each input sized correctly for the total
  // payload batch size. Concatenate input values from each payload
  // into the corresponding tensor.

  // Inputs from the request...
  bool cuda_copy = false;
  for (const auto& pr : repr_input_request->ImmutableInputs()) {
    const InferenceRequest::Input* input = pr.second;
    const std::string& name = input->Name();

    RETURN_IF_ERROR(SetInput(
        name, input->DType(), input->Shape(), total_batch_size, payloads,
        &input_buffers, &inputs, &cuda_copy));
  }

#ifdef TRTIS_ENABLE_GPU
  // Two pass synchronization, one to make sure indirect buffers are filled if
  // any, the other to make sure the input buffer for execution is ready.
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

#ifdef TRTIS_ENABLE_STATS
  for (auto& payload : *payloads) {
    if (payload.stats_ != nullptr) {
      payload.stats_->CaptureTimestamp(
          ModelInferStats::TimestampKind::kComputeInputEnd);
    }
  }
#endif  // TRTIS_ENABLE_STATS

  // Run...
  Caffe2Workspace::Error err = workspace_->Run();
  if (!err.IsOk()) {
    return Status(Status::Code::INTERNAL, err.Message());
  }

#ifdef TRTIS_ENABLE_STATS
  for (auto& payload : *payloads) {
    if (payload.stats_ != nullptr) {
      payload.stats_->CaptureTimestamp(
          ModelInferStats::TimestampKind::kComputeOutputStart);
    }
  }
#endif  // TRTIS_ENABLE_STATS

  std::vector<OutputInfo> outputs;
  // Make sure each output is of the expected size and copy it into
  // the payload responses.
  cuda_copy = false;
  for (const auto& output : base->Config().output()) {
    const std::string& name = output.name();

    const ModelOutput* output_config;
    RETURN_IF_ERROR(base->GetOutput(name, &output_config));

    // Checked at initialization time to make sure that STRING is not
    // being used for an output, so can just assume fixed-sized here.
    const Caffe2Workspace::DataType dtype =
        ConvertDataType(output_config->data_type());

    const DimsList& output_dims = (output_config->has_reshape())
                                      ? output_config->reshape().shape()
                                      : output_config->dims();

    outputs.emplace_back();
    RETURN_IF_ERROR(ReadFixedSizedOutputTensor(
        name, dtype, GetDataTypeByteSize(output_config->data_type()),
        total_batch_size, output_dims, payloads, &outputs.back(), &cuda_copy));
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

std::ostream&
operator<<(std::ostream& out, const NetDefBackend& pb)
{
  out << "name=" << pb.Name() << std::endl;
  out << "contexts:" << std::endl;
  for (const auto& context : pb.contexts_) {
    out << "  name=" << context->name_ << ", gpu="
        << ((context->gpu_device_ == NetDefBackend::Context::NO_GPU_DEVICE)
                ? "<none>"
                : std::to_string(context->gpu_device_));
  }

  return out;
}

}}  // namespace nvidia::inferenceserver
