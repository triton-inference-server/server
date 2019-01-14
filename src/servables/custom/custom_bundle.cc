// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include "src/servables/custom/custom_bundle.h"

#include <stdint.h>
#include "cuda/include/cuda_runtime_api.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/server_status.h"
#include "src/core/utils.h"
#include "src/servables/custom/loader.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/lib/io/path.h"

namespace nvidia { namespace inferenceserver {

CustomBundle::Context::Context(
    const std::string& name, const int gpu_device, const int max_batch_size)
    : name_(name), gpu_device_(gpu_device), max_batch_size_(max_batch_size),
      library_handle_(nullptr), library_context_handle_(nullptr),
      InitializeFn_(nullptr), FinalizeFn_(nullptr), ErrorStringFn_(nullptr),
      ExecuteFn_(nullptr)
{
}

CustomBundle::Context::Context(Context&& o)
    : name_(std::move(o.name_)), gpu_device_(o.gpu_device_),
      max_batch_size_(o.max_batch_size_), outputs_(std::move(o.outputs_)),
      library_handle_(o.library_handle_),
      library_context_handle_(o.library_context_handle_),
      InitializeFn_(o.InitializeFn_), FinalizeFn_(o.FinalizeFn_),
      ErrorStringFn_(o.ErrorStringFn_), ExecuteFn_(o.ExecuteFn_)
{
  o.gpu_device_ = NO_GPU_DEVICE;
  o.max_batch_size_ = NO_BATCHING;
  o.library_handle_ = nullptr;
  o.library_context_handle_ = nullptr;
  o.InitializeFn_ = nullptr;
  o.FinalizeFn_ = nullptr;
  o.ErrorStringFn_ = nullptr;
  o.ExecuteFn_ = nullptr;
}

CustomBundle::Context::~Context()
{
  LOG_VERBOSE(1) << "~CustomBundle::Context ";
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

tensorflow::Status
CustomBundle::Init(
    const tensorflow::StringPiece& path, const ModelConfig& config)
{
  TF_RETURN_IF_ERROR(ValidateModelConfig(config, kCustomPlatform));
  TF_RETURN_IF_ERROR(SetModelConfig(path, config));

  return tensorflow::Status::OK();
}

tensorflow::Status
CustomBundle::CreateExecutionContexts(
    const std::unordered_map<std::string, std::string>& libraries)
{
  uint32_t total_context_cnt = 0;

  // Create the context for each instance.
  for (const auto& group : Config().instance_group()) {
    for (int c = 0; c < group.count(); c++) {
      if (group.kind() == ModelInstanceGroup::KIND_CPU) {
        const std::string instance_name =
            group.name() + "_" + std::to_string(c) + "_cpu";
        TF_RETURN_IF_ERROR(CreateExecutionContext(
            instance_name, Context::NO_GPU_DEVICE, libraries));
      } else {
        for (int gpu_device : group.gpus()) {
          const std::string instance_name = group.name() + "_" +
                                            std::to_string(c) + "_gpu" +
                                            std::to_string(gpu_device);
          TF_RETURN_IF_ERROR(
              CreateExecutionContext(instance_name, gpu_device, libraries));
        }
      }

      total_context_cnt++;
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

  LOG_VERBOSE(1) << "custom bundle for " << Name() << std::endl << *this;
  return tensorflow::Status::OK();
}

tensorflow::Status
CustomBundle::CreateExecutionContext(
    const std::string& instance_name, const int gpu_device,
    const std::unordered_map<std::string, std::string>& libraries)
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

  const auto& mn_itr = libraries.find(cc_model_filename);
  if (mn_itr == libraries.end()) {
    return tensorflow::errors::Internal(
        "unable to find Custom model '", cc_model_filename, "' for ", Name());
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

  contexts_.emplace_back(instance_name, gpu_device, mbs);
  Context& context = contexts_.back();

  // Initialize 'context' for a specific 'gpu_device'. Collect the
  // outputs and their byte-sizes.
  for (const auto& io : Config().output()) {
    context.outputs_.insert({io.name(), GetByteSize(io)});
  }

  // 'mn_itr->second' is the path to the shared library file to use
  // for that context (e.g. model_name/1/libcustom.so). Load that
  // library as it provides the custom backend implementation.
  TF_RETURN_IF_ERROR(LoadCustom(
      mn_itr->second, &context.library_handle_, &context.InitializeFn_,
      &context.FinalizeFn_, &context.ErrorStringFn_, &context.ExecuteFn_));

  // Call the initialization function to get the custom context
  // associated with this specific instance.
  std::string serialized_config;
  Config().SerializeToString(&serialized_config);
  int err = context.InitializeFn_(
      serialized_config.c_str(), serialized_config.size(), gpu_device,
      &context.library_context_handle_);
  if (err != 0) {
    return tensorflow::errors::Internal(
        "initialize error for '", Name(), "': (", err, ") ",
        context.LibraryErrorString(err));
  }

  return tensorflow::Status::OK();
}

void
CustomBundle::Run(
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
    compute_timers.emplace_back();
    payload.stats_->StartComputeTimer(&compute_timers.back());
    payload.stats_->SetGPUDevice(contexts_[runner_idx].gpu_device_);
  }

  OnCompleteQueuedPayloads(contexts_[runner_idx].Run(payloads));
}

tensorflow::Status
CustomBundle::Context::Run(std::vector<Scheduler::Payload>* payloads)
{
  LOG_VERBOSE(1) << "Running " << name_ << " with " << payloads->size()
                 << " request payloads";

  // For each request in 'payloads' collect the total batch size for
  // this inference execution. The batch-size, number of inputs, and
  // size of each input has already been checked by each payloads
  // request provider so don't need to do that here.
  uint32_t total_batch_size = 0;
  uint32_t total_requested_outputs = 0;
  for (auto& payload : *payloads) {
    const InferRequestHeader& request_header =
        payload.request_provider_->RequestHeader();

    if ((size_t)request_header.output().size() > outputs_.size()) {
      payload.status_ = tensorflow::errors::InvalidArgument(
          "expected at most ", outputs_.size(), " outputs but got ",
          request_header.output().size());
      continue;
    }

    // Validate that all requested outputs are allowed and of the
    // correct size.
    for (const auto& output : request_header.output()) {
      const std::string& name = output.name();

      const auto& ii_iter = outputs_.find(name);
      if (ii_iter == outputs_.end()) {
        payload.status_ = tensorflow::errors::InvalidArgument(
            "unexpected inference output '", name, "' for '", name_, "'");
        break;
      }

      const size_t expected_byte_size = ii_iter->second;
      if (output.byte_size() != expected_byte_size) {
        payload.status_ = tensorflow::errors::InvalidArgument(
            "unexpected size ", output.byte_size(), " for inference output '",
            name, "', expecting ", expected_byte_size);
        break;
      }
    }

    if (!payload.status_.ok()) {
      continue;
    }

    total_batch_size += request_header.batch_size();
    total_requested_outputs += request_header.output_size();
  }

  // If there are no valid payloads then no need to run the
  // inference. The payloads will have their error status set so can
  // just return.
  if (total_batch_size == 0) {
    return tensorflow::Status::OK();
  }

  // total_batch_size can be 1 for models that don't support batching
  // (i.e. max_batch_size_ == 0).
  if ((total_batch_size != 1) &&
      (total_batch_size > (uint32_t)max_batch_size_)) {
    return tensorflow::errors::Internal(
        "dynamic batch size ", total_batch_size, " for '", name_,
        "', max allowed is ", max_batch_size_);
  }

  // We use the following to hold pointers to all the output names of
  // the payloads. We don't want this to resize as that will
  // invalidate the pointers so set the capacity big enough to hold
  // all the pointers for all the payloads.
  std::vector<const char*> work_output_name_ptrs;
  work_output_name_ptrs.reserve(total_requested_outputs);

  // We use the following to hold contexts needed for the input and
  // output callbacks. We don't want this to resize as that will
  // invalidate the pointers so set the capacity big enough to hold
  // the contexts for all the payloads.
  std::vector<GetInputOutputContext> work_io_contexts;
  work_io_contexts.reserve(payloads->size());

  // Collect the payload information into a array of custom::Payload
  // structs that can be passed to the backend.
  std::vector<CustomPayload> custom_payloads;
  for (auto& payload : *payloads) {
    if (!payload.status_.ok()) {
      continue;
    }

    const InferRequestHeader& request_header =
        payload.request_provider_->RequestHeader();

    custom_payloads.emplace_back();
    CustomPayload& custom_payload = custom_payloads.back();
    custom_payload.batch_size = request_header.batch_size();

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
    return tensorflow::errors::Internal(
        "execute error for '", name_, "': (", err, ") ",
        LibraryErrorString(err));
  }

  // Transfer payload errors back to the Payload objects.
  for (size_t i = 0; i < custom_payloads.size(); ++i) {
    if (custom_payloads[i].error_code != 0) {
      (*payloads)[i].status_ = tensorflow::errors::Internal(
          "payload error for '", name_, "': (", err, ") ",
          LibraryErrorString(err));
    }
  }

  return tensorflow::Status::OK();
}

bool
CustomBundle::Context::GetNextInput(
    GetInputOutputContext* input_context, const char* cname,
    const void** content, uint64_t* content_byte_size)
{
  const std::string name(cname);
  Scheduler::Payload* payload = input_context->payload_;

  *content = nullptr;
  *content_byte_size = 0;

  // If a payload has errors then it never should have been passed to
  // the custom backend.
  if (!payload->status_.ok()) {
    LOG_ERROR << "can't get tensor input for payload with non-ok status";
    return false;
  }

  const InferRequestHeader& request_header =
      payload->request_provider_->RequestHeader();

  int input_idx = 0;
  for (const auto& input : request_header.input()) {
    if (input.name() == name) {
      tensorflow::Status status =
          payload->request_provider_->GetNextInputContent(
              input_idx, content, content_byte_size, false);
      return status.ok();
    }

    input_idx++;
  }

  // Something went very wrong since unable to find the requested
  // input.
  LOG_ERROR << "can't get tensor values for unknown input '" << name << "'";
  return false;
}

bool
CustomBundle::Context::GetOutput(
    GetInputOutputContext* output_context, const char* cname,
    uint64_t content_byte_size, void** content)
{
  const std::string name(cname);
  Scheduler::Payload* payload = output_context->payload_;

  *content = nullptr;

  // If a payload has errors then it never should have been passed to
  // the custom backend.
  if (!payload->status_.ok()) {
    LOG_ERROR << "can't get output buffer for payload with non-ok status";
    return false;
  }

  const InferRequestHeader& request_header =
      payload->request_provider_->RequestHeader();

  int output_idx = 0;
  for (const auto& output : request_header.output()) {
    if (output.name() == name) {
      tensorflow::Status status = payload->response_provider_->GetOutputBuffer(
          output_idx, content, content_byte_size);
      return status.ok();
    }

    output_idx++;
  }

  // Something went very wrong since unable to find the requested output.
  LOG_ERROR << "can't get output buffer for unknown output '" << name << "'";
  return false;
}

std::string
CustomBundle::Context::LibraryErrorString(const int err)
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
operator<<(std::ostream& out, const CustomBundle& pb)
{
  out << "name=" << pb.Name() << std::endl;
  out << "contexts:" << std::endl;
  for (const auto& context : pb.contexts_) {
    out << "  name=" << context.name_ << ", gpu="
        << ((context.gpu_device_ == CustomBundle::Context::NO_GPU_DEVICE)
                ? "<none>"
                : std::to_string(context.gpu_device_));
  }

  return out;
}

bool
CustomGetNextInput(
    void* input_context, const char* name, const void** content,
    uint64_t* content_byte_size)
{
  CustomBundle::Context::GetInputOutputContext* icontext =
      static_cast<CustomBundle::Context::GetInputOutputContext*>(input_context);
  return icontext->context_->GetNextInput(
      icontext, name, content, content_byte_size);
}

bool
CustomGetOutput(
    void* output_context, const char* name, uint64_t content_byte_size,
    void** content)
{
  CustomBundle::Context::GetInputOutputContext* ocontext =
      static_cast<CustomBundle::Context::GetInputOutputContext*>(
          output_context);
  return ocontext->context_->GetOutput(
      ocontext, name, content_byte_size, content);
}

}}  // namespace nvidia::inferenceserver
