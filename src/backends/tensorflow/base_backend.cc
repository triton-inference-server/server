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

#include "src/backends/tensorflow/base_backend.h"

#include <set>
#include "src/backends/tensorflow/tf_utils.h"
#include "src/backends/tensorflow/tf_virtual_device.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.pb.h"
#include "src/core/model_config_utils.h"
#include "src/core/provider.h"
#include "src/core/server_status.h"

#ifdef TRTIS_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRTIS_ENABLE_GPU

namespace nvidia { namespace inferenceserver {

BaseBackend::Context::Context(
    const std::string& name, const int gpu_device, const int max_batch_size)
    : BackendContext(name, gpu_device, max_batch_size),
      trtistf_model_(nullptr, TRTISTF_ModelDelete)
{
}

BaseBackend::Context::~Context()
{
  LOG_VERBOSE(1) << "~BaseBackend::Context ";
}

Status
BaseBackend::Init(const std::string& path, const ModelConfig& config)
{
  RETURN_IF_ERROR(SetModelConfig(path, config));
  return Status::Success;
}

Status
BaseBackend::CreateExecutionContexts(
    const std::shared_ptr<GraphDefBackendFactory::Config>& backend_config,
    const std::unordered_map<std::string, std::string>& paths)
{
  if (LOG_VERBOSE_IS_ON(1)) {
    LOG_INFO << "Creating execution contexts for:";
    for (const auto p : paths) {
      LOG_INFO << "  " << p.first << ": " << p.second;
    }
  }

  uint32_t total_context_cnt = 0;

  for (const auto& group : Config().instance_group()) {
    for (int c = 0; c < group.count(); c++) {
      if (group.kind() == ModelInstanceGroup::KIND_CPU) {
        const std::string instance_name =
            group.name() + "_" + std::to_string(c) + "_cpu";
        RETURN_IF_ERROR(CreateExecutionContext(
            instance_name, Context::NO_GPU_DEVICE, backend_config, paths));
        total_context_cnt++;
      } else if (group.kind() == ModelInstanceGroup::KIND_MODEL) {
        const std::string instance_name =
            group.name() + "_" + std::to_string(c) + "_model_device";
        RETURN_IF_ERROR(CreateExecutionContext(
            instance_name, Context::MODEL_DEVICE, backend_config, paths));
      } else {
        for (int gpu_device : group.gpus()) {
          const std::string instance_name = group.name() + "_" +
                                            std::to_string(c) + "_gpu" +
                                            std::to_string(gpu_device);
          RETURN_IF_ERROR(CreateExecutionContext(
              instance_name, gpu_device, backend_config, paths));
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

  LOG_VERBOSE(1) << "backend for " << Name() << std::endl << *this;

  return Status::Success;
}

Status
BaseBackend::CreateExecutionContext(
    const std::string& instance_name, const int gpu_device,
    const std::shared_ptr<GraphDefBackendFactory::Config>& backend_config,
    const std::unordered_map<std::string, std::string>& paths)
{
  // For a GPU context, determine the model file to use for device
  // compute capability. CPU always uses the default model file.
  std::string cc_model_filename;
  int vgpu_device = gpu_device;

  if (gpu_device == Context::NO_GPU_DEVICE) {
    cc_model_filename = Config().default_model_filename();

    LOG_INFO << "Creating instance " << instance_name << " on CPU using "
             << cc_model_filename;
  } else if (gpu_device == Context::MODEL_DEVICE) {
    cc_model_filename = Config().default_model_filename();

    LOG_INFO << "Creating instance " << instance_name
             << " on devices as specified in " << cc_model_filename;
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

    const std::string cc =
        std::to_string(cuprops.major) + "." + std::to_string(cuprops.minor);
    const auto& cc_itr = Config().cc_model_filenames().find(cc);
    cc_model_filename = (cc_itr == Config().cc_model_filenames().end())
                            ? Config().default_model_filename()
                            : cc_itr->second;

    // Get virtual device tracker instance, and get next device id
    if (VirtualDeviceTracker::HasVirtualDevice()) {
      RETURN_IF_ERROR(
          VirtualDeviceTracker::GetNextVirtualDevice(gpu_device, &vgpu_device));
    }

    LOG_INFO << "Creating instance " << instance_name << " on GPU "
             << vgpu_device << " (" << cc << ") using " << cc_model_filename;
#else
    return Status(RequestStatusCode::INTERNAL, "GPU instances not supported");
#endif  // TRTIS_ENABLE_GPU
  }

  const auto& gdp_itr = paths.find(cc_model_filename);
  if (gdp_itr == paths.end()) {
    return Status(
        RequestStatusCode::INTERNAL,
        "unable to find model '" + cc_model_filename + "' for " + Name());
  }

  // Max batch size. A value of 0 in the config becomes NO_BATCHING.
  const int mbs = (Config().max_batch_size() <= 0) ? Context::NO_BATCHING
                                                   : Config().max_batch_size();

  contexts_.emplace_back(new Context(instance_name, gpu_device, mbs));
  const std::unique_ptr<Context>& context = contexts_.back();

  auto graphdef_backend_config =
      std::static_pointer_cast<GraphDefBackendFactory::Config>(backend_config);

  RETURN_IF_ERROR(context->ValidateInputs(Config().input()));
  RETURN_IF_ERROR(context->ValidateOutputs(Config().output()));

  RETURN_IF_ERROR(CreateTRTISTFModel(
      graphdef_backend_config, vgpu_device, Config().optimization().has_graph(),
      Config().optimization().graph().level(), gdp_itr->second,
      &context->trtistf_model_, &context->input_name_map_,
      &context->output_name_map_));

  return Status::Success;
}

Status
BaseBackend::Context::ValidateInputs(
    const ::google::protobuf::RepeatedPtrField<ModelInput>& ios)
{
  for (const auto& io : ios) {
    if (ConvertDataType(io.data_type()) ==
        TRTISTF_DataType::TRTISTF_TYPE_INVALID) {
      return Status(
          RequestStatusCode::INTERNAL,
          "unsupported datatype " + DataType_Name(io.data_type()) +
              " for input '" + io.name() + "' for model '" + name_ + "'");
    }
  }

  return Status::Success;
}


Status
BaseBackend::Context::ValidateOutputs(
    const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios)
{
  for (const auto& io : ios) {
    if (ConvertDataType(io.data_type()) ==
        TRTISTF_DataType::TRTISTF_TYPE_INVALID) {
      return Status(
          RequestStatusCode::INTERNAL,
          "unsupported datatype " + DataType_Name(io.data_type()) +
              " for output '" + io.name() + "' for model '" + name_ + "'");
    }
  }

  return Status::Success;
}

void
BaseBackend::Run(
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

namespace {

void
FillStringTensor(TRTISTF_Tensor* tensor, const size_t idx, const size_t cnt)
{
  for (size_t c = 0; c < cnt; ++c) {
    TRTISTF_TensorSetString(tensor, idx + c, nullptr);
  }
}

void
SetStringInputTensor(
    TRTISTF_Tensor* tensor, const std::string& input_name,
    const size_t batch1_element_cnt, std::vector<Scheduler::Payload>* payloads)
{
  size_t tensor_element_idx = 0;

  // Visit the payloads in order and copy the input values into the
  // input tensor. Skip payloads that had errors since they are not
  // included in the dynamic batch.
  for (auto& payload : *payloads) {
    const InferRequestHeader& request_header =
        payload.request_provider_->RequestHeader();
    const size_t expected_element_cnt =
        request_header.batch_size() * batch1_element_cnt;
    size_t element_idx = 0;

    const void* vcontent;
    size_t content_byte_size = expected_element_cnt * sizeof(uint32_t);
    payload.status_ = payload.request_provider_->GetNextInputContent(
        input_name, &vcontent, &content_byte_size, true);
    if (!payload.status_.IsOk()) {
      FillStringTensor(
          tensor, tensor_element_idx + element_idx,
          expected_element_cnt - element_idx);
      continue;
    }

    const char* content = reinterpret_cast<const char*>(vcontent);

    // Parse content and assign them to the 'tensor'. Each string
    // in 'content' is a 4-byte length followed by the string
    // itself with no null-terminator.
    while (content_byte_size >= sizeof(uint32_t)) {
      if (element_idx >= expected_element_cnt) {
        payload.status_ = Status(
            RequestStatusCode::INVALID_ARG,
            "unexpected number of string elements " +
                std::to_string(element_idx + 1) + " for inference input '" +
                input_name + "', expecting " +
                std::to_string(expected_element_cnt));
        FillStringTensor(
            tensor, tensor_element_idx + element_idx,
            expected_element_cnt - element_idx);
        break;
      }

      const uint32_t len = *(reinterpret_cast<const uint32_t*>(content));
      content += sizeof(uint32_t);
      content_byte_size -= sizeof(uint32_t);

      if (content_byte_size < len) {
        payload.status_ = Status(
            RequestStatusCode::INVALID_ARG,
            "incomplete string data for inference input '" + input_name +
                "', expecting string of length " + std::to_string(len) +
                " but only " + std::to_string(content_byte_size) +
                " bytes available");
        FillStringTensor(
            tensor, tensor_element_idx + element_idx,
            expected_element_cnt - element_idx);
        break;
      }

      std::string str(content, len);
      content += len;
      content_byte_size -= len;

      TRTISTF_TensorSetString(
          tensor, tensor_element_idx + element_idx, str.c_str());
      element_idx++;
    }

    if (payload.status_.IsOk() && (element_idx != expected_element_cnt)) {
      payload.status_ = Status(
          RequestStatusCode::INTERNAL,
          "expected " + std::to_string(expected_element_cnt) +
              " strings for inference input '" + input_name + "', got " +
              std::to_string(element_idx));
      FillStringTensor(
          tensor, tensor_element_idx + element_idx,
          expected_element_cnt - element_idx);
    }

    tensor_element_idx += expected_element_cnt;
  }
}

void
ReadFixedSizedOutputTensor(
    TRTISTF_Tensor* tensor, const std::string& output_name,
    const std::vector<int64_t>& shape, const size_t batch1_byte_size,
    std::vector<Scheduler::Payload>* payloads)
{
  size_t tensor_copy_offset = 0;

  for (auto& payload : *payloads) {
    const InferRequestHeader& request_header =
        payload.request_provider_->RequestHeader();
    const size_t expected_byte_size =
        request_header.batch_size() * batch1_byte_size;

    // If 'payload' requested this output then copy it from the
    // GPU. If it did not request this output then just skip it in
    // the output buffer.
    if ((payload.response_provider_ != nullptr) &&
        payload.response_provider_->RequiresOutput(output_name)) {
      void* content = nullptr;
      Status status = payload.response_provider_->AllocateOutputBuffer(
          output_name, &content, expected_byte_size, shape);
      if (!status.IsOk()) {
        payload.status_ = status;
      } else {
        if ((tensor_copy_offset + expected_byte_size) >
            TRTISTF_TensorDataByteSize(tensor)) {
          payload.status_ = Status(
              RequestStatusCode::INVALID_ARG,
              "unexpected size " +
                  std::to_string(tensor_copy_offset + expected_byte_size) +
                  " for inference output '" + output_name + "', expecting " +
                  std::to_string(TRTISTF_TensorDataByteSize(tensor)));
        } else {
          memcpy(
              content, TRTISTF_TensorData(tensor) + tensor_copy_offset,
              expected_byte_size);
        }
      }
    }

    tensor_copy_offset += expected_byte_size;
  }
}

void
ReadStringOutputTensor(
    TRTISTF_Tensor* tensor, const std::string& output_name,
    const std::vector<int64_t>& shape, const size_t batch1_element_cnt,
    std::vector<Scheduler::Payload>* payloads)
{
  size_t tensor_element_idx = 0;

  for (auto& payload : *payloads) {
    const InferRequestHeader& request_header =
        payload.request_provider_->RequestHeader();
    const size_t expected_element_cnt =
        request_header.batch_size() * batch1_element_cnt;

    // If 'payload' requested this output then copy it from the
    // GPU. If it did not request this output then just skip it in
    // the output tensor.
    if ((payload.response_provider_ != nullptr) &&
        payload.response_provider_->RequiresOutput(output_name)) {
      // Serialize the output tensor strings. Each string is
      // serialized as a 4-byte length followed by the string itself
      // with no null-terminator.
      std::string serialized;
      for (size_t e = 0; e < expected_element_cnt; ++e) {
        const char* cstr = TRTISTF_TensorString(tensor, tensor_element_idx + e);
        const uint32_t len = strlen(cstr);
        serialized.append(
            reinterpret_cast<const char*>(&len), sizeof(uint32_t));
        if (len > 0) {
          serialized.append(cstr);
        }
      }

      void* content;
      Status status = payload.response_provider_->AllocateOutputBuffer(
          output_name, &content, serialized.size(), shape);
      if (status.IsOk()) {
        memcpy(
            content, reinterpret_cast<const void*>(serialized.c_str()),
            serialized.size());
      } else {
        payload.status_ = status;
      }
    }

    tensor_element_idx += expected_element_cnt;
  }
}

}  // namespace

Status
BaseBackend::Context::SetInput(
    const std::string& name, const DataType datatype, const DimsList& dims,
    const size_t total_batch_size, std::vector<Scheduler::Payload>* payloads,
    TRTISTF_TensorList** input_tensors)
{
  // Get the shape of the input. The provider has already checked
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

  const std::string* input_tensor_name = &name;
  const auto& tn_itr = input_name_map_.find(name);
  if (tn_itr != input_name_map_.end()) {
    input_tensor_name = &tn_itr->second;
  }

  const TRTISTF_DataType dtype = ConvertDataType(datatype);
  TRTISTF_Tensor* tensor = TRTISTF_TensorNew(
      input_tensor_name->c_str(), dtype, shape.size(),
      (shape.size() == 0) ? nullptr : &shape[0]);
  if (tensor == nullptr) {
    return Status(
        RequestStatusCode::INTERNAL,
        "failed to create input tensor '" + name + "' with shape " +
            DimsListToString(shape) + " and data type " +
            DataType_Name(datatype) + " for '" + name_ + "'");
  }

  TRTISTF_TensorList* tlink = TRTISTF_TensorListNew(tensor, *input_tensors);
  *input_tensors = tlink;

  if (dtype != TRTISTF_DataType::TRTISTF_TYPE_STRING) {
    const size_t batch1_byte_size =
        batch1_element_cnt * TRTISTF_TensorDataTypeByteSize(tensor);
    if ((batch1_byte_size * total_batch_size) !=
        TRTISTF_TensorDataByteSize(tensor)) {
      return Status(
          RequestStatusCode::INTERNAL,
          "failed to create input tensor '" + name +
              "' with expected byte size " +
              std::to_string(batch1_byte_size * total_batch_size) + ", got " +
              std::to_string(TRTISTF_TensorDataByteSize(tensor)));
    }
    SetFixedSizedInputTensor(tensor, name, batch1_byte_size, payloads);
  } else {
    SetStringInputTensor(tensor, name, batch1_element_cnt, payloads);
  }

  return Status::Success;
}

void
BaseBackend::Context::SetFixedSizedInputTensor(
    TRTISTF_Tensor* tensor, const std::string& input_name,
    const size_t batch1_byte_size, std::vector<Scheduler::Payload>* payloads)
{
  char* buffer = TRTISTF_TensorData(tensor);

  // Visit the payloads in order and copy the input values into the
  // input tensor. Skip payloads that had errors since they are not
  // included in the dynamic batch.
  std::vector<size_t> expected_byte_sizes;
  for (auto& payload : *payloads) {
    const InferRequestHeader& request_header =
        payload.request_provider_->RequestHeader();
    expected_byte_sizes.push_back(
        request_header.batch_size() * batch1_byte_size);
  }

  SetInputBuffer(
      input_name, expected_byte_sizes, payloads, TRTSERVER_MEMORY_CPU, buffer);
}

Status
BaseBackend::Context::Run(
    const BaseBackend* base, std::vector<Scheduler::Payload>* payloads)
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

  // Create a tensor for each input sized correctly for the total
  // payload batch size. Concatenate input values from each payload
  // into the corresponding tensor.

  // Smart pointer is TensorList** as the pointer to input head (TensorList*)
  // will be updated in SetInput()
  TRTISTF_TensorList* input_head_ptr = nullptr;
  static auto input_deleter = [](TRTISTF_TensorList** list) {
    if (list != nullptr) {
      TRTISTF_TensorListDelete(*list);
    }
  };
  std::unique_ptr<TRTISTF_TensorList*, decltype(input_deleter)> input_tensors(
      &input_head_ptr, input_deleter);

  // Inputs from the request...
  for (const auto& input : input_request_provider->RequestHeader().input()) {
    const std::string& name = input.name();

    const ModelInput* input_config;
    RETURN_IF_ERROR(base->GetInput(name, &input_config));

    RETURN_IF_ERROR(SetInput(
        name, input_config->data_type(), input.dims(), total_batch_size,
        payloads, input_tensors.get()));
  }

  // Additional inputs added to the provider...
  const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
      input_override_map = input_request_provider->GetInputOverride();
  if (input_override_map != nullptr) {
    for (const auto& pr : *input_override_map) {
      const std::string& name = pr.first;
      const std::shared_ptr<InferRequestProvider::InputOverride>& override =
          pr.second;
      RETURN_IF_ERROR(SetInput(
          name, override->datatype_, override->dims_, total_batch_size,
          payloads, input_tensors.get()));
    }
  }

  // Collect the names of outputs requested by any request
  // payload. Skip payloads that have an error.
  std::set<std::string> required_outputs;
  for (auto& payload : *payloads) {
    const InferRequestHeader& request_header =
        payload.request_provider_->RequestHeader();
    for (const auto& output : request_header.output()) {
      required_outputs.insert(output.name());
    }
  }

  // Create the vector of required output names using the names
  // expected by the model.
  std::vector<std::string> model_output_names;
  const char* output_names_cstr[required_outputs.size()];
  {
    size_t oidx = 0;
    for (const auto& name : required_outputs) {
      model_output_names.push_back(name);
      const auto& tn_itr = output_name_map_.find(name);
      if (tn_itr == output_name_map_.end()) {
        output_names_cstr[oidx] = name.c_str();
      } else {
        output_names_cstr[oidx] = tn_itr->second.c_str();
      }
      oidx++;
    }
  }

  // Run. Session will update the 'output_tensors'.
  std::unique_ptr<TRTISTF_TensorList, decltype(&TRTISTF_TensorListDelete)>
      output_tensors(nullptr, TRTISTF_TensorListDelete);

  {
    TRTISTF_TensorList* rtl;
    RETURN_IF_TRTISTF_ERROR(TRTISTF_ModelRun(
        trtistf_model_.get(), *(input_tensors.release()),
        required_outputs.size(), output_names_cstr, &rtl));
    output_tensors.reset(rtl);
  }

  // Make sure each output is of the expected size and copy it into
  // the appropriate response providers.
  TRTISTF_TensorList* output_tensor_itr = output_tensors.get();
  for (const auto& name : model_output_names) {
    const ModelOutput* output_config;
    RETURN_IF_ERROR(base->GetOutput(name, &output_config));

    TRTISTF_Tensor* output_tensor = output_tensor_itr->tensor_;

    // Get the shape and datatype of the output from the output
    // tensor.
    TRTISTF_Shape* shape = TRTISTF_TensorShape(output_tensor);
    std::vector<int64_t> shapevec;

    bool skip_element_cnt = (max_batch_size_ != NO_BATCHING);
    size_t batch1_element_cnt = 1;
    for (size_t itr = 0; itr < shape->rank_; itr++) {
      const int64_t dim = shape->dims_[itr];
      shapevec.push_back(dim);
      if (!skip_element_cnt) {
        batch1_element_cnt *= dim;
      }
      skip_element_cnt = false;
    }

    const DimsList& output_dims = (output_config->has_reshape())
                                      ? output_config->reshape().shape()
                                      : output_config->dims();

    // verify shape of output matches shape from model config
    const int batch_offset = ((max_batch_size_ == NO_BATCHING) ? 0 : 1);

    for (int i = 0; i < output_dims.size(); i++) {
      if (output_dims[i] != -1) {
        if (output_dims[i] != shapevec[i + batch_offset]) {
          return Status(
              RequestStatusCode::INVALID_ARG,
              "unexpected shape for output '" + name +
                  "', model configuration shape is " +
                  DimsListToString(shapevec) + ", inference shape is " +
                  DimsListToString(output_dims));
        }
      }
    }

    TRTISTF_DataType dtype = ConvertDataType(output_config->data_type());
    if (dtype != TRTISTF_TensorDataType(output_tensor)) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "unexpected datatype " +
              DataType_Name(
                  ConvertDataType(TRTISTF_TensorDataType(output_tensor))) +
              " for inference output '" + name + "', expecting " +
              DataType_Name(output_config->data_type()));
    }

    if (dtype != TRTISTF_DataType::TRTISTF_TYPE_STRING) {
      const size_t batch1_byte_size =
          batch1_element_cnt * TRTISTF_TensorDataTypeByteSize(output_tensor);
      ReadFixedSizedOutputTensor(
          output_tensor, name, shapevec, batch1_byte_size, payloads);
    } else {
      ReadStringOutputTensor(
          output_tensor, name, shapevec, batch1_element_cnt, payloads);
    }

    output_tensor_itr = output_tensor_itr->next_;
  }

  return Status::Success;
}

std::ostream&
operator<<(std::ostream& out, const BaseBackend& pb)
{
  out << "name=" << pb.Name() << std::endl;
  out << "contexts:" << std::endl;
  for (const auto& context : pb.contexts_) {
    out << "  name=" << context->name_ << ", gpu="
        << ((context->gpu_device_ == BaseBackend::Context::NO_GPU_DEVICE)
                ? "<none>"
                : std::to_string(context->gpu_device_))
        << ", max_batch_size="
        << ((context->max_batch_size_ == BaseBackend::Context::NO_BATCHING)
                ? "<none>"
                : std::to_string(context->max_batch_size_))
        << std::endl;
  }

  return out;
}

}}  // namespace nvidia::inferenceserver
