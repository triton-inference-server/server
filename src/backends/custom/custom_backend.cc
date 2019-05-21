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

#include "src/backends/custom/custom_backend.h"

#include <stdint.h>
#include "src/backends/custom/loader.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/model_config_utils.h"
#include "src/core/provider.h"
#include "src/core/server_status.h"

#ifdef TRTIS_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRTIS_ENABLE_GPU

namespace nvidia { namespace inferenceserver {

CustomBackend::Context::Context(
    const std::string& name, const int gpu_device, const int max_batch_size)
    : name_(name), gpu_device_(gpu_device), max_batch_size_(max_batch_size),
      library_handle_(nullptr), library_context_handle_(nullptr),
      InitializeFn_(nullptr), FinalizeFn_(nullptr), ErrorStringFn_(nullptr),
      ExecuteFn_(nullptr)
{
}

CustomBackend::Context::~Context()
{
  LOG_VERBOSE(1) << "~CustomBackend::Context " << name_;
  if (FinalizeFn_ != nullptr) {
    int err = FinalizeFn_(library_context_handle_);
    if (err != 0) {
      LOG_ERROR << "error finalizing custom library: (" << err << ") "
                << LibraryErrorString(err);
    }
  }

  UnloadCustom(library_handle_);

  library_context_handle_ = nullptr;
  library_handle_ = nullptr;
}

Status
CustomBackend::Init(
    const std::string& path, const std::vector<std::string>& server_params,
    const ModelConfig& config)
{
  RETURN_IF_ERROR(ValidateModelConfig(config, kCustomPlatform));
  RETURN_IF_ERROR(SetModelConfig(path, config));

  server_params_ = server_params;

  return Status::Success;
}

Status
CustomBackend::CreateExecutionContexts(
    const std::unordered_map<std::string, std::string>& libraries)
{
  uint32_t total_context_cnt = 0;

  // Create the context for each instance.
  for (const auto& group : Config().instance_group()) {
    for (int c = 0; c < group.count(); c++) {
      if (group.kind() == ModelInstanceGroup::KIND_CPU) {
        const std::string instance_name =
            group.name() + "_" + std::to_string(c) + "_cpu";
        RETURN_IF_ERROR(CreateExecutionContext(
            instance_name, Context::NO_GPU_DEVICE, libraries));
        total_context_cnt++;
      } else {
        for (int gpu_device : group.gpus()) {
          const std::string instance_name = group.name() + "_" +
                                            std::to_string(c) + "_gpu" +
                                            std::to_string(gpu_device);
          RETURN_IF_ERROR(
              CreateExecutionContext(instance_name, gpu_device, libraries));
          total_context_cnt++;
        }
      }
    }
  }

  // Create a scheduler with one thread for each context available for
  // this model. Each runner is exclusively tied to the context.
  RETURN_IF_ERROR(SetConfiguredScheduler(
      total_context_cnt,
      [this](uint32_t runner_idx) -> Status { return InitBackend(runner_idx); },
      [this](
          uint32_t runner_idx, std::vector<Scheduler::Payload>* payloads,
          std::function<void(Status)> func) {
        RunBackend(runner_idx, payloads, func);
      }));

  LOG_VERBOSE(1) << "custom backend for " << Name() << std::endl << *this;
  return Status::Success;
}

Status
CustomBackend::CreateExecutionContext(
    const std::string& instance_name, const int gpu_device,
    const std::unordered_map<std::string, std::string>& libraries)
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

  const auto& mn_itr = libraries.find(cc_model_filename);
  if (mn_itr == libraries.end()) {
    return Status(
        RequestStatusCode::INTERNAL, "unable to find Custom model '" +
                                         cc_model_filename + "' for " + Name());
  }

  if (gpu_device == Context::NO_GPU_DEVICE) {
    LOG_INFO << "Creating instance " << instance_name << " on CPU using "
             << cc_model_filename;
  } else {
    LOG_INFO << "Creating instance " << instance_name << " on GPU "
             << gpu_device << " (" << cc << ") using " << cc_model_filename;
  }
  LOG_VERBOSE(1) << Config().DebugString();

  // Max batch size. A value of 0 in the config becomes NO_BATCHING.
  const int mbs = (Config().max_batch_size() <= 0) ? Context::NO_BATCHING
                                                   : Config().max_batch_size();

  contexts_.emplace_back(new Context(instance_name, gpu_device, mbs));
  const std::unique_ptr<Context>& context = contexts_.back();

  // 'mn_itr->second' is the path to the shared library file to use
  // for that context (e.g. model_name/1/libcustom.so). Load that
  // library as it provides the custom backend implementation.
  RETURN_IF_ERROR(LoadCustom(
      mn_itr->second, &(context->library_handle_), &(context->InitializeFn_),
      &(context->FinalizeFn_), &(context->ErrorStringFn_),
      &(context->ExecuteFn_)));

  return Status::Success;
}

Status
CustomBackend::InitBackend(uint32_t runner_idx)
{
  // Each runner executes using the corresponding context...
  if (runner_idx >= contexts_.size()) {
    return Status(
        RequestStatusCode::INTERNAL,
        "unexpected runner index" + std::to_string(runner_idx) +
            ", max allowed " + std::to_string(contexts_.size()));
  }

  const auto& context = contexts_[runner_idx];

  // Call the initialization function to get the custom context
  // associated with this specific instance.
  std::string serialized_config;
  Config().SerializeToString(&serialized_config);

  CustomInitializeData init_data;
  init_data.instance_name = context->name_.c_str();
  init_data.serialized_model_config = serialized_config.c_str();
  init_data.serialized_model_config_size = serialized_config.size();
  init_data.gpu_device_id = context->gpu_device_;

  std::vector<const char*> server_param_values;
  for (const auto& param : server_params_) {
    server_param_values.push_back(param.c_str());
  }

  init_data.server_parameter_cnt = server_param_values.size();
  if (server_param_values.size() > 0) {
    init_data.server_parameters = &server_param_values[0];
  } else {
    init_data.server_parameters = nullptr;
  }

  int err =
      context->InitializeFn_(&init_data, &(context->library_context_handle_));
  if (err != 0) {
    return Status(
        RequestStatusCode::INTERNAL, "initialize error for '" + Name() +
                                         "': (" + std::to_string(err) + ") " +
                                         context->LibraryErrorString(err));
  }

  return Status::Success;
}

void
CustomBackend::RunBackend(
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

  OnCompleteQueuedPayloads(contexts_[runner_idx]->Run(this, payloads));
}

Status
CustomBackend::Context::Run(
    CustomBackend* base, std::vector<Scheduler::Payload>* payloads)
{
  LOG_VERBOSE(1) << "Running " << name_ << " with " << payloads->size()
                 << " request payloads";

  // Each payload will have the same number and shape for inputs. Get
  // the shape for each input into a vector suitable to passing via
  // the custom backend interface. As a performance improvement for
  // models that don't have any variable-size input tensors we could
  // calculate the input tensor shapes once during backend
  // initialization.
  std::unordered_map<std::string, std::unique_ptr<std::vector<int64_t>>>
      input_shapes;

  if (!payloads->empty()) {
    const InferRequestHeader& request_header =
        payloads->front().request_provider_->RequestHeader();

    for (const auto& input : request_header.input()) {
      std::unique_ptr<std::vector<int64_t>> shape(new std::vector<int64_t>());
      for (auto d : input.dims()) {
        shape->push_back(d);
      }
      input_shapes.emplace(std::make_pair(input.name(), std::move(shape)));
    }
  }

  // For each request in 'payloads' collect the total batch size for
  // this inference execution. The batch-size, number of inputs, and
  // size of each input has already been checked by each payload's
  // request provider so don't need to do that here.
  uint32_t total_batch_size = 0;
  uint32_t total_inputs = 0;
  uint32_t total_requested_outputs = 0;
  for (auto& payload : *payloads) {
    if (!payload.status_.IsOk()) {
      return Status(
          RequestStatusCode::INTERNAL,
          "unexpected payload with non-OK status given to custom runner for '" +
              name_ + "'");
    }

    const InferRequestHeader& request_header =
        payload.request_provider_->RequestHeader();

    total_batch_size += request_header.batch_size();
    total_inputs += request_header.input_size();
    total_requested_outputs += request_header.output_size();
  }

  // If there are no valid payloads then no need to run the
  // inference. The payloads will have their error status set so can
  // just return.
  if (total_batch_size == 0) {
    return Status::Success;
  }

  // total_batch_size can be 1 for models that don't support batching
  // (i.e. max_batch_size_ == 0).
  if ((total_batch_size != 1) &&
      (total_batch_size > (uint32_t)max_batch_size_)) {
    return Status(
        RequestStatusCode::INTERNAL,
        "dynamic batch size " + std::to_string(total_batch_size) + " for '" +
            name_ + "', max allowed is " + std::to_string(max_batch_size_));
  }

  // We use the following to hold pointers to all the input and output
  // names of the payloads. We don't want this to resize as that will
  // invalidate the pointers so set the capacity big enough to hold
  // all the pointers for all the payloads.
  std::vector<const char*> work_input_name_ptrs;
  work_input_name_ptrs.reserve(total_inputs);
  std::vector<const char*> work_output_name_ptrs;
  work_output_name_ptrs.reserve(total_requested_outputs);

  // Similarly for input dim sizes and the dimension values.
  std::vector<size_t> work_input_dim_cnts;
  work_input_dim_cnts.reserve(total_inputs);
  std::vector<const int64_t*> work_input_dims_ptrs;
  work_input_dims_ptrs.reserve(total_inputs);

  // We use the following to hold contexts needed for the input and
  // output callbacks. We don't want this to resize as that will
  // invalidate the pointers so set the capacity big enough to hold
  // the contexts for all the payloads.
  std::vector<GetInputOutputContext> work_io_contexts;
  work_io_contexts.reserve(payloads->size());

  // Collect the payload information into a array of custom::Payload
  // structs that can be passed to the backend. Every payload must
  // have an OK status (checked above) so we don't bother to check
  // that here.
  std::vector<CustomPayload> custom_payloads;
  for (auto& payload : *payloads) {
    const InferRequestHeader& request_header =
        payload.request_provider_->RequestHeader();

    custom_payloads.emplace_back();
    CustomPayload& custom_payload = custom_payloads.back();
    custom_payload.batch_size = request_header.batch_size();

    // Inputs
    custom_payload.input_cnt = request_header.input_size();
    custom_payload.input_names = nullptr;
    custom_payload.input_shape_dim_cnts = nullptr;
    custom_payload.input_shape_dims = nullptr;
    for (const auto& input : request_header.input()) {
      auto itr = input_shapes.find(input.name());
      if (itr == input_shapes.end()) {
        return Status(
            RequestStatusCode::INTERNAL, "unable to find shape for input '" +
                                             input.name() + "' for '" + name_ +
                                             "'");
      }

      std::unique_ptr<std::vector<int64_t>>& shape = itr->second;

      work_input_name_ptrs.push_back(input.name().c_str());
      work_input_dim_cnts.push_back(shape->size());
      work_input_dims_ptrs.push_back(
          (shape->size() == 0) ? nullptr : &(shape->at(0)));
      if (custom_payload.input_names == nullptr) {
        custom_payload.input_names = &work_input_name_ptrs.back();
        custom_payload.input_shape_dim_cnts = &work_input_dim_cnts.back();
        custom_payload.input_shape_dims = &work_input_dims_ptrs.back();
      }
    }

    // Outputs
    custom_payload.output_cnt = request_header.output_size();
    custom_payload.required_output_names = nullptr;
    for (const auto& output : request_header.output()) {
      work_output_name_ptrs.push_back(output.name().c_str());
      if (custom_payload.required_output_names == nullptr) {
        custom_payload.required_output_names = &work_output_name_ptrs.back();
      }
    }

    work_io_contexts.emplace_back(this, &payload);
    custom_payload.input_context = &work_io_contexts.back();
    custom_payload.output_context = custom_payload.input_context;
    custom_payload.error_code = 0;
  }

  // Execute the custom backend which will use CustomGetOutput to get
  // the output buffers into which it will write the results for the
  // requested outputs.
  int err = ExecuteFn_(
      library_context_handle_, custom_payloads.size(), &custom_payloads[0],
      CustomGetNextInput, CustomGetOutput);
  if (err != 0) {
    return Status(
        RequestStatusCode::INTERNAL, "execute error for '" + name_ + "': (" +
                                         std::to_string(err) + ") " +
                                         LibraryErrorString(err));
  }

  // Transfer payload errors back to the Payload objects.
  for (size_t i = 0; i < custom_payloads.size(); ++i) {
    if (custom_payloads[i].error_code != 0) {
      (*payloads)[i].status_ = Status(
          RequestStatusCode::INTERNAL,
          "payload error for '" + name_ + "': (" +
              std::to_string(custom_payloads[i].error_code) + ") " +
              LibraryErrorString(custom_payloads[i].error_code));
    }
  }

  return Status::Success;
}

bool
CustomBackend::Context::GetNextInput(
    GetInputOutputContext* input_context, const char* cname,
    const void** content, uint64_t* content_byte_size)
{
  const std::string name(cname);
  Scheduler::Payload* payload = input_context->payload_;

  Status status = payload->request_provider_->GetNextInputContent(
      name, content, content_byte_size, false);
  return status.IsOk();
}

bool
CustomBackend::Context::GetOutput(
    GetInputOutputContext* output_context, const char* cname,
    size_t shape_dim_cnt, int64_t* shape_dims, uint64_t content_byte_size,
    void** content)
{
  const std::string name(cname);
  Scheduler::Payload* payload = output_context->payload_;

  *content = nullptr;

  // If there is no response provider return content == nullptr with
  // OK status as an indication that the output should not be written.
  if (payload->response_provider_ != nullptr) {
    std::vector<int64_t> shape;
    if (shape_dim_cnt > 0) {
      shape.assign(shape_dims, shape_dims + shape_dim_cnt);
    }

    Status status = payload->response_provider_->AllocateOutputBuffer(
        name, content, content_byte_size, shape);
    return status.IsOk();
  }

  return true;
}

std::string
CustomBackend::Context::LibraryErrorString(const int err)
{
  if (ErrorStringFn_ != nullptr) {
    const char* str = ErrorStringFn_(library_context_handle_, err);
    if (str != nullptr) {
      return std::string(str);
    }
  }

  return "<no error string>";
}

std::ostream&
operator<<(std::ostream& out, const CustomBackend& pb)
{
  out << "name=" << pb.Name() << std::endl;
  out << "contexts:" << std::endl;
  for (const auto& context : pb.contexts_) {
    out << "  name=" << context->name_ << ", gpu="
        << ((context->gpu_device_ == CustomBackend::Context::NO_GPU_DEVICE)
                ? "<none>"
                : std::to_string(context->gpu_device_))
        << std::endl;
  }

  return out;
}

bool
CustomGetNextInput(
    void* input_context, const char* name, const void** content,
    uint64_t* content_byte_size)
{
  CustomBackend::Context::GetInputOutputContext* icontext =
      static_cast<CustomBackend::Context::GetInputOutputContext*>(
          input_context);
  return icontext->context_->GetNextInput(
      icontext, name, content, content_byte_size);
}

bool
CustomGetOutput(
    void* output_context, const char* name, size_t shape_dim_cnt,
    int64_t* shape_dims, uint64_t content_byte_size, void** content)
{
  CustomBackend::Context::GetInputOutputContext* ocontext =
      static_cast<CustomBackend::Context::GetInputOutputContext*>(
          output_context);
  return ocontext->context_->GetOutput(
      ocontext, name, shape_dim_cnt, shape_dims, content_byte_size, content);
}

}}  // namespace nvidia::inferenceserver
