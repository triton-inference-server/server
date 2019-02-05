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

#include "src/servables/caffe2/netdef_bundle.h"

#include <NvInfer.h>
#include <stdint.h>
#include "cuda/include/cuda_runtime_api.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/provider.h"
#include "src/core/server_status.h"
#include "src/core/utils.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/lib/io/path.h"

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


NetDefBundle::Context::Context(
    const std::string& name, const int gpu_device, const int max_batch_size)
    : name_(name), gpu_device_(gpu_device), max_batch_size_(max_batch_size)
{
}

NetDefBundle::Context::Context(Context&& o)
    : name_(std::move(o.name_)), gpu_device_(o.gpu_device_),
      max_batch_size_(o.max_batch_size_)
{
  o.gpu_device_ = NO_GPU_DEVICE;
  o.max_batch_size_ = NO_BATCHING;
  workspace_.swap(o.workspace_);
}

NetDefBundle::Context::~Context()
{
  LOG_VERBOSE(1) << "~NetDefBundle::Context ";
}

tensorflow::Status
NetDefBundle::Init(
    const tensorflow::StringPiece& path, const ModelConfig& config)
{
  TF_RETURN_IF_ERROR(ValidateModelConfig(config, kCaffe2NetDefPlatform));
  TF_RETURN_IF_ERROR(SetModelConfig(path, config));

  return tensorflow::Status::OK();
}

tensorflow::Status
NetDefBundle::CreateExecutionContexts(
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
        TF_RETURN_IF_ERROR(CreateExecutionContext(
            instance_name, Context::NO_GPU_DEVICE, models));
        total_context_cnt++;
      } else {
        for (int gpu_device : group.gpus()) {
          const std::string instance_name = group.name() + "_" +
                                            std::to_string(c) + "_gpu" +
                                            std::to_string(gpu_device);
          TF_RETURN_IF_ERROR(
              CreateExecutionContext(instance_name, gpu_device, models));
          total_context_cnt++;
        }
      }
    }
  }

  // Create a scheduler with one thread for each context available for
  // this model. Each runner is exclusively tied to the context.
  TF_RETURN_IF_ERROR(SetConfiguredScheduler(
      total_context_cnt,
      [this](
          uint32_t runner_idx, std::vector<Scheduler::Payload>* payloads,
          std::function<void(tensorflow::Status)> func) {
        Run(runner_idx, payloads, func);
      }));

  LOG_VERBOSE(1) << "netdef bundle for " << Name() << std::endl << *this;
  return tensorflow::Status::OK();
}

tensorflow::Status
NetDefBundle::CreateExecutionContext(
    const std::string& instance_name, const int gpu_device,
    const std::unordered_map<std::string, std::vector<char>>& models)
{
  cudaError_t cuerr;

  // For a GPU context, determine the model file to use for device
  // compute capability. CPU always uses the default model file.
  std::string cc;
  std::string cc_model_filename;
  if (gpu_device == Context::NO_GPU_DEVICE) {
    cc_model_filename = Config().default_model_filename();
  } else {
    cudaDeviceProp cuprops;
    cuerr = cudaGetDeviceProperties(&cuprops, gpu_device);
    if (cuerr != cudaSuccess) {
      return tensorflow::errors::Internal(
          "unable to get CUDA device properties for ", Name(), ": ",
          cudaGetErrorString(cuerr));
    }

    cc = std::to_string(cuprops.major) + "." + std::to_string(cuprops.minor);
    const auto& cc_itr = Config().cc_model_filenames().find(cc);
    cc_model_filename = (cc_itr == Config().cc_model_filenames().end())
                            ? Config().default_model_filename()
                            : cc_itr->second;
  }


  const auto& mn_itr = models.find(cc_model_filename);
  if (mn_itr == models.end()) {
    return tensorflow::errors::Internal(
        "unable to find NetDef model '", cc_model_filename, "' for ", Name());
  }

  // NetDef also requires an init network, the name of which is always
  // derived from 'cc_model_filename'.
  const std::string& cc_init_filename =
      kCaffe2NetDefInitFilenamePrefix + cc_model_filename;
  const auto& imn_itr = models.find(cc_init_filename);
  if (imn_itr == models.end()) {
    return tensorflow::errors::Internal(
        "unable to find NetDef initialization model '", cc_init_filename,
        "' for ", Name());
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

  contexts_.emplace_back(instance_name, gpu_device, mbs);
  Context& context = contexts_.back();

  // Extract input and output names from the config and use them to
  // create a Caffe2 workspace. We can't cross the raw protobuf across
  // this boundary (since Caffe2 build may use a different protobuf).
  std::vector<std::string> input_names;
  for (const auto& io : Config().input()) {
    input_names.push_back(io.name());
  }
  std::vector<std::string> output_names;
  for (const auto& io : Config().output()) {
    output_names.push_back(io.name());
  }

  Caffe2Workspace* c2ws;
  Caffe2Workspace::Error err = Caffe2WorkspaceCreate(
      &c2ws, Config().name(), Config().max_batch_size(), input_names,
      output_names, gpu_device, imn_itr->second, mn_itr->second);
  if (!err.IsOk()) {
    return tensorflow::errors::Internal(err.Message());
  }

  context.workspace_.reset(c2ws);

  TF_RETURN_IF_ERROR(context.ValidateInputs(Config().input()));
  TF_RETURN_IF_ERROR(context.ValidateOutputs(Config().output()));

  return tensorflow::Status::OK();
}

tensorflow::Status
NetDefBundle::Context::ValidateInputs(
    const ::google::protobuf::RepeatedPtrField<ModelInput>& ios)
{
  for (const auto& io : ios) {
    TF_RETURN_IF_ERROR(
        ValidateModelInput(io, workspace_->PotentialInputNames()));

    if (ConvertDataType(io.data_type()) ==
        Caffe2Workspace::DataType::TYPE_INVALID) {
      return tensorflow::errors::Internal(
          "unsupported datatype ", DataType_Name(io.data_type()),
          " for input '", io.name(), "' for model '", name_, "'");
    }
  }

  return tensorflow::Status::OK();
}


tensorflow::Status
NetDefBundle::Context::ValidateOutputs(
    const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios)
{
  for (const auto& io : ios) {
    TF_RETURN_IF_ERROR(
        ValidateModelOutput(io, workspace_->PotentialOutputNames()));

    if (ConvertDataType(io.data_type()) ==
        Caffe2Workspace::DataType::TYPE_INVALID) {
      return tensorflow::errors::Internal(
          "unsupported datatype ", DataType_Name(io.data_type()),
          " for output '", io.name(), "' for model '", name_, "'");
    }
  }

  return tensorflow::Status::OK();
}

void
NetDefBundle::Run(
    uint32_t runner_idx, std::vector<Scheduler::Payload>* payloads,
    std::function<void(tensorflow::Status)> OnCompleteQueuedPayloads)
{
  // Each runner executes using the corresponding context...
  if (runner_idx >= contexts_.size()) {
    OnCompleteQueuedPayloads(tensorflow::errors::Internal(
        "unexpected runner index", runner_idx, ", max allowed ",
        contexts_.size()));
    return;
  }

  std::vector<ModelInferStats::ScopedTimer> compute_timers;
  for (auto& payload : *payloads) {
    // Stop queue timer when the payload is scheduled to run
    payload.queue_timer_.reset();

    compute_timers.emplace_back();
    payload.stats_->StartComputeTimer(&compute_timers.back());
    payload.stats_->SetGPUDevice(contexts_[runner_idx].gpu_device_);
  }

  OnCompleteQueuedPayloads(contexts_[runner_idx].Run(this, payloads));
}

tensorflow::Status
NetDefBundle::Context::SetFixedSizedInputTensor(
    const std::string& name, const std::vector<int64_t>& shape,
    const Caffe2Workspace::DataType dtype, const size_t batch1_byte_size,
    const size_t total_byte_size, std::vector<Scheduler::Payload>* payloads,
    std::vector<std::unique_ptr<char[]>>* input_buffers)
{
  // The entire input tensor must be delivered as a single
  // contiguous chunk so create a buffer large enough to hold the
  // entire dynamic batched input.
  input_buffers->emplace_back(new char[total_byte_size]);
  char* buffer = input_buffers->back().get();

  size_t buffer_copy_offset = 0;

  // Visit the payloads in order and copy the input tensors to
  // 'buffer'. Skip payloads that had errors since they are not
  // included in the dynamic batch.
  for (auto& payload : *payloads) {
    if (!payload.status_.ok()) {
      continue;
    }

    const InferRequestHeader& request_header =
        payload.request_provider_->RequestHeader();
    const size_t expected_byte_size =
        request_header.batch_size() * batch1_byte_size;

    int input_idx = 0;
    for (const auto& input : request_header.input()) {
      if (input.name() == name) {
        size_t copied_byte_size = 0;
        while (payload.compute_status_.ok()) {
          const void* content;
          size_t content_byte_size = expected_byte_size - copied_byte_size;
          payload.compute_status_ =
              payload.request_provider_->GetNextInputContent(
                  input_idx, &content, &content_byte_size, false);
          if (!payload.compute_status_.ok()) {
            break;
          }

          // No more input content available then done with copying...
          if (content == nullptr) {
            break;
          }

          if ((buffer_copy_offset + copied_byte_size + content_byte_size) >
              total_byte_size) {
            payload.compute_status_ = tensorflow::errors::InvalidArgument(
                "unexpected size ",
                buffer_copy_offset + copied_byte_size + content_byte_size,
                " for inference input '", name, "', expecting ",
                total_byte_size);
            break;
          }

          memcpy(
              static_cast<char*>(buffer) + buffer_copy_offset +
                  copied_byte_size,
              content, content_byte_size);
          copied_byte_size += content_byte_size;
        }

        if (payload.compute_status_.ok() &&
            (copied_byte_size != expected_byte_size)) {
          payload.compute_status_ = tensorflow::errors::Internal(
              "expected ", expected_byte_size,
              " bytes of data for inference input '", name, "', got ",
              copied_byte_size);
        }

        break;
      }

      input_idx++;
    }

    buffer_copy_offset += expected_byte_size;
  }

  Caffe2Workspace::Error err = workspace_->SetInputTensor(
      name, shape, dtype, static_cast<const char*>(buffer), total_byte_size);
  if (!err.IsOk()) {
    return tensorflow::errors::Internal(err.Message());
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
NetDefBundle::Context::ReadFixedSizedOutputTensor(
    const std::string& name, const std::vector<int64_t>& shape,
    const Caffe2Workspace::DataType dtype, const size_t dtype_byte_size,
    const size_t total_batch_size, std::vector<Scheduler::Payload>* payloads)
{
  std::vector<int64_t> content_shape;
  const char* content = nullptr;
  size_t byte_size = 0;
  Caffe2Workspace::Error err = workspace_->GetOutputTensor(
      name, shape, dtype, &content, &byte_size, &content_shape);
  if (!err.IsOk()) {
    return tensorflow::errors::Internal(err.Message());
  }

  const size_t total_byte_size =
      GetElementCount(content_shape) * dtype_byte_size;
  const size_t batch1_byte_size = total_byte_size / total_batch_size;

  if (byte_size != total_byte_size) {
    return tensorflow::errors::Internal(
        "unexpected size for output '", name, "', byte-size ",
        std::to_string(byte_size), " does not equal ",
        std::to_string(total_batch_size), " * ",
        std::to_string(batch1_byte_size));
  }

  size_t content_offset = 0;

  for (auto& payload : *payloads) {
    if (!payload.status_.ok()) {
      continue;
    }

    const InferRequestHeader& request_header =
        payload.request_provider_->RequestHeader();
    const size_t expected_byte_size =
        request_header.batch_size() * batch1_byte_size;

    // If 'payload' requested this output then copy it from
    // 'content'. If it did not request this output then just
    // skip it in the 'content'.
    if (payload.response_provider_->RequiresOutput(name)) {
      void* buffer;
      tensorflow::Status status = payload.response_provider_->GetOutputBuffer(
          name, &buffer, expected_byte_size, content_shape);
      if (status.ok()) {
        memcpy(buffer, content + content_offset, expected_byte_size);
      }

      if (!status.ok()) {
        payload.compute_status_ = status;
      }
    }

    content_offset += expected_byte_size;
  }

  return tensorflow::Status::OK();
}


tensorflow::Status
NetDefBundle::Context::Run(
    const NetDefBundle* base, std::vector<Scheduler::Payload>* payloads)
{
  LOG_VERBOSE(1) << "Running " << name_ << " with " << payloads->size()
                 << " request payloads";

  const InferRequestHeader* input_request_header = nullptr;

  // For each request in 'payloads' collect the total batch size for
  // this inference execution. The batch-size, number of inputs, and
  // size of each input has already been checked by each payloads
  // request provider so don't need to do that here.
  size_t total_batch_size = 0;
  for (auto& payload : *payloads) {
    if (payload.status_.ok()) {
      total_batch_size +=
          payload.request_provider_->RequestHeader().batch_size();

      // All payloads must have equally-sized input tensors so use any
      // payload as the representative for the input tensors.
      input_request_header = &(payload.request_provider_->RequestHeader());
    }
  }

  // If there are no valid payloads then no need to run the
  // inference. The payloads will have their error status set so can
  // just return.
  if (total_batch_size == 0) {
    return tensorflow::Status::OK();
  }

  // total_batch_size can be 1 for models that don't support batching
  // (i.e. max_batch_size_ == 0).
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size_)) {
    return tensorflow::errors::Internal(
        "dynamic batch size ", total_batch_size, " for '", name_,
        "', max allowed is ", max_batch_size_);
  }

  // Hold reference to each buffer of input data to that it stays
  // until the inference has completed.
  std::vector<std::unique_ptr<char[]>> input_buffers;

  // Create a tensor for each input sized correctly for the total
  // payload batch size. Concatenate input values from each payload
  // into the corresponding tensor.
  for (const auto& input : input_request_header->input()) {
    const std::string& name = input.name();

    const ModelInput* input_config;
    TF_RETURN_IF_ERROR(base->GetInput(name, &input_config));

    // Get the shape of the input.  The provider has already checked
    // that the request shape is valid so don't need to do it here.
    std::vector<int64_t> shape;

    // If model supports batching then prepend the batch dimension
    // onto the input shape.
    if (max_batch_size_ != NO_BATCHING) {
      shape.push_back(total_batch_size);
    }

    size_t batch1_element_cnt = 1;
    for (auto dim : input.dims()) {
      shape.push_back(dim);
      batch1_element_cnt *= dim;
    }

    // Checked at initialization time to make sure that STRING is not
    // being used for an input, so can just assume fixed-sized here.
    const Caffe2Workspace::DataType dtype =
        ConvertDataType(input_config->data_type());
    const size_t batch1_byte_size =
        batch1_element_cnt * GetDataTypeByteSize(input_config->data_type());
    const size_t total_byte_size = total_batch_size * batch1_byte_size;

    TF_RETURN_IF_ERROR(SetFixedSizedInputTensor(
        name, shape, dtype, batch1_byte_size, total_byte_size, payloads,
        &input_buffers));
  }

  // Run...
  Caffe2Workspace::Error err = workspace_->Run();
  if (!err.IsOk()) {
    return tensorflow::errors::Internal(err.Message());
  }

  // Make sure each output is of the expected size and copy it into
  // the payload responses.
  for (const auto& output : base->Config().output()) {
    const std::string& name = output.name();

    const ModelOutput* output_config;
    TF_RETURN_IF_ERROR(base->GetOutput(name, &output_config));

    // Get the shape of the output from the model configuration.
    std::vector<int64_t> shape;

    // If model supports batching then prepend the batch dimension
    // onto the output shape.
    if (max_batch_size_ != NO_BATCHING) {
      shape.push_back(total_batch_size);
    }

    for (auto dim : output_config->dims()) {
      shape.push_back(dim);
    }

    // Checked at initialization time to make sure that STRING is not
    // being used for an output, so can just assume fixed-sized here.
    const Caffe2Workspace::DataType dtype =
        ConvertDataType(output_config->data_type());
    TF_RETURN_IF_ERROR(ReadFixedSizedOutputTensor(
        name, shape, dtype, GetDataTypeByteSize(output_config->data_type()),
        total_batch_size, payloads));
  }

  return tensorflow::Status::OK();
}

std::ostream&
operator<<(std::ostream& out, const NetDefBundle& pb)
{
  out << "name=" << pb.Name() << std::endl;
  out << "contexts:" << std::endl;
  for (const auto& context : pb.contexts_) {
    out << "  name=" << context.name_ << ", gpu="
        << ((context.gpu_device_ == NetDefBundle::Context::NO_GPU_DEVICE)
                ? "<none>"
                : std::to_string(context.gpu_device_));
  }

  return out;
}

}}  // namespace nvidia::inferenceserver
