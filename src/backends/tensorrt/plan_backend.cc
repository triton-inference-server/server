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

#include "src/backends/tensorrt/plan_backend.h"

#include <NvInfer.h>
#include <stdint.h>
#include <future>
#include "src/backends/tensorrt/loader.h"
#include "src/backends/tensorrt/plan_utils.h"
#include "src/core/constants.h"
#include "src/core/cuda_utils.h"
#include "src/core/logging.h"
#include "src/core/metrics.h"
#include "src/core/model_config_cuda.h"
#include "src/core/model_config_utils.h"
#include "src/core/nvtx.h"

namespace nvidia { namespace inferenceserver {

namespace {

Status
CreateCudaEvent(
    const std::string& event_name, unsigned int event_flags, cudaEvent_t* event)
{
  // Not adding 'cudaEventBlockingSync' to reduce gaps between the time of
  // event record and the time of signaling blocking thread.
  // The busy waiting only happens when there is inflight request.
  auto cuerr = cudaEventCreateWithFlags(event, event_flags);
  if (cuerr != cudaSuccess) {
    return Status(
        Status::Code::INTERNAL, "unable to create CUDA event for " +
                                    event_name + ": " +
                                    cudaGetErrorString(cuerr));
  }
  return Status::Success;
}

// Utilities for warmup feature
TRITONSERVER_Error*
WarmupResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  *buffer = malloc(byte_size);
  if (*buffer != nullptr) {
    *actual_memory_type = TRITONSERVER_MEMORY_CPU;
    *actual_memory_type_id = 0;
    return nullptr;
  }

  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INTERNAL,
      "failed to allocate output buffer for warmup.");
}

TRITONSERVER_Error*
WarmupResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  free(buffer);
  return nullptr;
}

ResponseAllocator warmup_allocator = ResponseAllocator(
    WarmupResponseAlloc, WarmupResponseRelease, nullptr /* start_fn */);

void
WarmupResponseComplete(
    TRITONSERVER_InferenceResponse* iresponse, const uint32_t flags,
    void* userp)
{
  if (iresponse != nullptr) {
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceResponseError(iresponse), "warmup error");
    // Just delete the response, warmup doesn't check for correctness
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceResponseDelete(iresponse),
        "deleting warmup response");
  }
}

void
WarmupRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  if ((flags & TRITONSERVER_REQUEST_RELEASE_ALL) != 0) {
    TRITONSERVER_InferenceRequestDelete(request);
    if (userp != nullptr) {
      auto warmup_promise = reinterpret_cast<std::promise<void>*>(userp);
      warmup_promise->set_value();
    }
  }
}

}  // namespace

PlanBackend::Context::Context(
    const std::string& name, const int gpu_device, const int max_batch_size,
    const bool enable_pinned_input, const bool enable_pinned_output,
    std::unique_ptr<MetricModelReporter>&& metric_reporter)
    : BackendContext(
          name, gpu_device, max_batch_size, enable_pinned_input,
          enable_pinned_output, std::move(metric_reporter)),
      engine_(nullptr), is_shared_engine_(true), total_bindings_(0),
      num_expected_bindings_(0)
{
  stream_ = nullptr;
  input_copy_stream_ = nullptr;

  next_set_ = 0;
  for (size_t idx = 0; idx < EVENT_SET_COUNT; idx++) {
    events_[idx].input_ready_ = nullptr;
    events_[idx].ready_for_input_ = nullptr;
    events_[idx].output_ready_ = nullptr;
    events_[idx].ready_for_output_ = nullptr;
  }
  support_batching_ = (max_batch_size != NO_BATCHING);
}

PlanBackend::Context::~Context()
{
  LOG_VERBOSE(1) << "~PlanBackend::Context ";

  cudaSetDevice(gpu_device_);
  for (auto buffer : buffers_) {
    if (buffer != nullptr) {
      cudaError_t err = cudaFree(buffer);
      if (err != cudaSuccess) {
        LOG_ERROR << "Failed to free cuda memory for '" << name_
                  << "': " << cudaGetErrorString(err);
      }
    }
  }

  for (auto& trt_context : trt_contexts_) {
    for (const auto& cuda_graph_execs : trt_context.second.cuda_graph_execs_) {
      for (const auto& pr : cuda_graph_execs) {
        cudaError_t err = cudaGraphExecDestroy(pr.second.cuda_graph_exec_);
        if (err != cudaSuccess) {
          LOG_ERROR << "Failed to destroy cuda graph exec: "
                    << cudaGetErrorString(err);
        }
      }
    }
    trt_context.second.cuda_graph_execs_.clear();

    for (const auto& cuda_graph : trt_context.second.cuda_graphs_) {
      cudaError_t err = cudaGraphDestroy(cuda_graph);
      if (err != cudaSuccess) {
        LOG_ERROR << "Failed to destroy cuda graph: "
                  << cudaGetErrorString(err);
      }
    }
    trt_context.second.cuda_graphs_.clear();

    if (trt_context.second.context_ != nullptr) {
      trt_context.second.context_->destroy();
      trt_context.second.context_ = nullptr;
    }
  }

  if (stream_ != nullptr) {
    cudaError_t err = cudaStreamDestroy(stream_);
    if (err != cudaSuccess) {
      LOG_ERROR << "Failed to destroy cuda stream: " << cudaGetErrorString(err);
    }
    stream_ = nullptr;
  }

  if ((engine_ != nullptr) && (!is_shared_engine_)) {
    engine_->destroy();
    engine_ = nullptr;
  }

  DestroyEventSet();

  // Notify the completion thread to exit
  completion_queue_.Put(std::move(std::unique_ptr<Payload>()));
  if (completion_thread_.joinable()) {
    completion_thread_.join();
  }
}

Status
PlanBackend::CreateExecutionContexts(
    const std::unordered_map<std::string, std::vector<char>>& models)
{
  // TensorRT engine creation is not thread-safe, so multiple creations
  // are serialized with a global lock.
  static std::mutex global_context_mu;
  std::lock_guard<std::mutex> glock(global_context_mu);

  // Only need to map device to runner when creating contexts, after that,
  // only runner idx is needed.
  std::map<int, size_t> device_to_runner_map;

  // Create a runtime/engine/context trifecta for each instance.
  //
  // TODO [DLIS-14] This can be optimized by sharing a runtime (across
  // all instances?), and sharing an engine across instances that have
  // access to the same GPU.
  for (const auto& group : Config().instance_group()) {
    // TensorRT requires that every context have a GPU.
    if ((group.kind() != inference::ModelInstanceGroup::KIND_GPU) ||
        (group.gpus().size() == 0)) {
      return Status(
          Status::Code::INVALID_ARG,
          "instance group " + group.name() + " of model " + Name() +
              " must be KIND_GPU and must specify at least one GPU id");
    }

    for (int c = 0; c < group.count(); c++) {
      for (int gpu_device : group.gpus()) {
        size_t runner_idx = 0;
        if (Config().has_sequence_batching()) {
          // For sequence batcher, there must be one runner per instance
          // instead of one runner per device
          runner_idx = available_context_queue_.size();
          available_context_queue_.emplace_back(new SyncQueue<size_t>());
          next_context_.emplace_back(-1);
        } else {
          auto it = device_to_runner_map.find(gpu_device);
          if (it == device_to_runner_map.end()) {
            it = device_to_runner_map
                     .emplace(gpu_device, available_context_queue_.size())
                     .first;
            available_context_queue_.emplace_back(new SyncQueue<size_t>());
            next_context_.emplace_back(-1);
          }
          runner_idx = it->second;
        }
        // The last entry in contexts_ is the newly created context
        auto& queue = available_context_queue_[runner_idx];
        queue->Put(contexts_.size());

        const std::string instance_name = group.name() + "_" +
                                          std::to_string(c) + "_gpu" +
                                          std::to_string(gpu_device);

        // Determine the model file to use for device compute capability
        cudaDeviceProp cuprops;
        auto cuerr = cudaGetDeviceProperties(&cuprops, gpu_device);
        if (cuerr != cudaSuccess) {
          return Status(
              Status::Code::INTERNAL,
              "unable to get CUDA device properties for " + Name() + ": " +
                  cudaGetErrorString(cuerr));
        }

        const std::string cc =
            std::to_string(cuprops.major) + "." + std::to_string(cuprops.minor);
        const auto& cc_itr = Config().cc_model_filenames().find(cc);
        const std::string& cc_model_filename =
            (cc_itr == Config().cc_model_filenames().end())
                ? Config().default_model_filename()
                : cc_itr->second;

        const auto& mn_itr = models.find(cc_model_filename);
        if (mn_itr == models.end()) {
          return Status(
              Status::Code::INTERNAL, "unable to find PLAN model '" +
                                          cc_model_filename + "' for " +
                                          Name());
        }

        // Create shared engine for the device if haven't tried so.
        auto eit = device_engines_.find(gpu_device);
        if (eit == device_engines_.end()) {
          eit = device_engines_
                    .emplace(gpu_device, std::make_pair(nullptr, nullptr))
                    .first;

          // Create a CUDA engine shared by all contexts
          cuerr = cudaSetDevice(gpu_device);
          if (cuerr != cudaSuccess) {
            return Status(
                Status::Code::INTERNAL, "unable to set device for " + Name() +
                                            ": " + cudaGetErrorString(cuerr));
          }

          RETURN_IF_ERROR(LoadPlan(
              mn_itr->second, &eit->second.first, &eit->second.second));
          // Validate whether the engine can be shared
          bool is_dynamic = false;
          for (int idx = 0; idx < eit->second.second->getNbBindings(); idx++) {
            auto dims = eit->second.second->getBindingDimensions(idx);
            // Detect whether dynamic or not
            if (ContainsWildcard(dims)) {
              is_dynamic = true;
              break;
            }
          }
          // Model with dynamic shapes can't share engine, set to engine to
          // 'nullptr' as hint, but keeping runtime as it can be used repeatedly
          if (is_dynamic) {
            if (eit->second.second != nullptr) {
              eit->second.second->destroy();
              eit->second.second = nullptr;
            }
          }
        }

        LOG_INFO << "Creating instance " << instance_name << " on GPU "
                 << gpu_device << " (" << cc << ") using " << cc_model_filename;
        RETURN_IF_ERROR(CreateExecutionContext(
            instance_name, gpu_device, mn_itr->second, group.profile(), queue));
      }
    }
  }

  // Create a scheduler with one thread for each context queue specified for
  // this model. Each runner is responsible to dispatch tasks to contexts
  // assigned to the corresponding queue. For different scheduler type, the
  // context queue will be formed differently to fit the scheduler's need.
  RETURN_IF_ERROR(SetConfiguredScheduler(
      available_context_queue_.size(),
      [this](uint32_t runner_idx) -> Status {
        // Obtain any context as the next context for the corresponding runner
        next_context_[runner_idx] = available_context_queue_[runner_idx]->Get();
        return Status::Success;
      },
      [this](
          uint32_t runner_idx,
          std::vector<std::unique_ptr<InferenceRequest>>&& requests) {
        Run(runner_idx, std::move(requests));
      }));

  LOG_VERBOSE(1) << "plan backend for " << Name() << std::endl << *this;

  return Status::Success;
}

Status
PlanBackend::Context::InitOptimizationProfiles(
    const ::google::protobuf::RepeatedPtrField<std::string>& profile_names)
{
  total_bindings_ = engine_->getNbBindings();
  const int total_profiles = engine_->getNbOptimizationProfiles();

  // TRT sets the optimization profile index to be 0 implicitly with the first
  // context creation. As currently triton supports one context per engine,
  // in order to set the specified profile_index, another context is created
  // and the previous context is destroyed.
  auto default_trt_context = engine_->createExecutionContext();
  if (default_trt_context == nullptr) {
    return Status(Status::Code::INTERNAL, "unable to create TensorRT context");
  }

  if (total_profiles == 0) {
    num_expected_bindings_ = total_bindings_;
  } else {
    num_expected_bindings_ = total_bindings_ / total_profiles;
  }

  // No optimization profile is set for this TensorRT plan
  if ((total_profiles == 0) || profile_names.empty()) {
    auto it =
        trt_contexts_
            .emplace(
                0, TensorRTContext(
                       "default", 0, num_expected_bindings_, EVENT_SET_COUNT))
            .first;
    it->second.context_ = default_trt_context;
    default_trt_context = nullptr;
  } else {
    // Create one TRT context for each specified profile
    for (const auto& profile_name : profile_names) {
      int profile_index = 0;
      RETURN_IF_ERROR(GetProfileIndex(profile_name, &profile_index));
      auto res = trt_contexts_.emplace(
          profile_index, TensorRTContext(
                             profile_name, profile_index,
                             num_expected_bindings_, EVENT_SET_COUNT));
      if (!res.second) {
        LOG_WARNING << profile_name << " maps to profile index "
                    << profile_index << " which has been mapped by "
                    << res.first->second.profile_name_
                    << ", existing optimization profile will be reused";
        continue;
      }
      if (profile_index == 0) {
        res.first->second.context_ = default_trt_context;
        default_trt_context = nullptr;
      } else {
        res.first->second.context_ = engine_->createExecutionContext();
        if (res.first->second.context_ == nullptr) {
          return Status(
              Status::Code::INTERNAL, "unable to create TensorRT context");
        }
        if (!res.first->second.context_->setOptimizationProfileAsync(
                profile_index, stream_)) {
          return Status(
              Status::Code::INVALID_ARG,
              "Can not set the specified optimization profile " + profile_name +
                  "[" + std::to_string(profile_index) + "] for " + name_ +
                  ". Expected optimization profile index range 0-" +
                  std::to_string(engine_->getNbOptimizationProfiles() - 1));
        }
      }
    }

    // profile 0 is not specified
    if (default_trt_context != nullptr) {
      default_trt_context->destroy();
    }
  }

  return Status::Success;
}

Status
PlanBackend::CreateExecutionContext(
    const std::string& instance_name, const int gpu_device,
    const std::vector<char>& model,
    const ::google::protobuf::RepeatedPtrField<std::string>& profile_names,
    const std::shared_ptr<SyncQueue<size_t>>& context_queue)
{
  // Max batch size. A value of 0 in the config becomes NO_BATCHING.
  const int mbs = (Config().max_batch_size() <= 0) ? Context::NO_BATCHING
                                                   : Config().max_batch_size();
  const bool pinned_input =
      Config().optimization().input_pinned_memory().enable();
  const bool pinned_output =
      Config().optimization().output_pinned_memory().enable();

  std::unique_ptr<MetricModelReporter> metric_reporter;
#ifdef TRITON_ENABLE_METRICS
  if (Metrics::Enabled()) {
    metric_reporter.reset(new MetricModelReporter(
        Name(), Version(), gpu_device, Config().metric_tags()));
  }
#endif  // TRITON_ENABLE_METRICS

  contexts_.emplace_back(new Context(
      instance_name, gpu_device, mbs, pinned_input, pinned_output,
      std::move(metric_reporter)));
  Context* context = static_cast<Context*>(contexts_.back().get());
  auto context_idx = contexts_.size() - 1;

  // Set the device before preparing the context.
  auto cuerr = cudaSetDevice(gpu_device);
  if (cuerr != cudaSuccess) {
    return Status(
        Status::Code::INTERNAL, "unable to set device for " + Name() + ": " +
                                    cudaGetErrorString(cuerr));
  }

  // Create CUDA streams associated with the execution context
  const int cuda_stream_priority =
      GetCudaStreamPriority(Config().optimization().priority());
  RETURN_IF_ERROR(context->CreateCudaStream(cuda_stream_priority));
  RETURN_IF_ERROR(context->CreateCudaStream(
      cuda_stream_priority, &context->input_copy_stream_));

  // Create CUDA events associated with the execution states
  RETURN_IF_ERROR(
      context->InitEventSet(Config().optimization().cuda().busy_wait_events()));

  auto eit = device_engines_.find(gpu_device);
  if (eit->second.second == nullptr) {
    context->is_shared_engine_ = false;
    RETURN_IF_ERROR(LoadPlan(model, &eit->second.first, &context->engine_));
  } else {
    context->engine_ = eit->second.second;
  }

  RETURN_IF_ERROR(context->InitOptimizationProfiles(profile_names));

  // Collect all the expected input and allowed output tensor names
  // and validate that the model configuration specifies only those.
  std::set<std::string> allowed_inputs, allowed_outputs, allowed_shape_tensors;
  for (int i = 0; i < context->num_expected_bindings_; ++i) {
    if (context->engine_->bindingIsInput(i)) {
      allowed_inputs.emplace(context->engine_->getBindingName(i));
    } else {
      allowed_outputs.emplace(context->engine_->getBindingName(i));
    }
    if (context->engine_->isExecutionBinding(i)) {
      LOG_VERBOSE(1) << "Detected " << context->engine_->getBindingName(i)
                     << " as execution binding for " + Name();
    }
    if (context->engine_->isShapeBinding(i)) {
      allowed_shape_tensors.emplace(context->engine_->getBindingName(i));
      LOG_VERBOSE(1) << "Detected " << context->engine_->getBindingName(i)
                     << " as shape binding for " + Name();
    }
  }

  for (const auto& io : Config().input()) {
    RETURN_IF_ERROR(CheckAllowedModelInput(io, allowed_inputs));
  }

  for (const auto& io : Config().output()) {
    RETURN_IF_ERROR(CheckAllowedModelOutput(io, allowed_outputs));
  }

  RETURN_IF_ERROR(
      context->ValidateInputs(Config().input(), allowed_shape_tensors));
  RETURN_IF_ERROR(
      context->ValidateOutputs(Config().output(), allowed_shape_tensors));
  // Initialize the inputs and outputs. Make sure the model matches
  // what is in the configuration. Allocate memory for the maximum
  // possible batch size: min(engine maximum, config maximum)
  context->byte_sizes_ =
      std::vector<uint64_t>(context->num_expected_bindings_, 0);
  context->buffers_ =
      std::vector<void*>(context->num_expected_bindings_, nullptr);
  context->buffer_is_ragged_ =
      std::vector<bool>(context->num_expected_bindings_, false);
  context->batch_inputs_ =
      std::vector<std::shared_ptr<Context::BatchInputData>>(
          context->num_expected_bindings_, nullptr);
  context->buffer_bindings_ =
      std::vector<void*>(context->total_bindings_, nullptr);
  context->io_shape_mapping_ =
      std::vector<std::pair<std::string, std::vector<int64_t>>>(
          context->num_expected_bindings_);

  RETURN_IF_ERROR(
      context->InitializeConfigShapeInputBindings(Config().input()));
  RETURN_IF_ERROR(
      context->InitializeConfigExecuteInputBindings(Config().input()));
  RETURN_IF_ERROR(context->InitializeSequenceControlInputBindings(Config()));
  RETURN_IF_ERROR(context->InitializeBatchInputBindings(Config()));
  for (const auto& trt_context : context->trt_contexts_) {
    if (!trt_context.second.context_->allInputDimensionsSpecified()) {
      return Status(
          Status::Code::INTERNAL,
          "failed to specify the dimensions of all input bindings");
    }
    if (!trt_context.second.context_->allInputShapesSpecified()) {
      return Status(
          Status::Code::INTERNAL,
          "failed to specify the values for all input shape tensors");
    }
  }

  // Validate the batch dimension against the implicit batch dimension if
  // available.
  if (context->engine_->hasImplicitBatchDimension() &&
      (context->max_batch_size_ > context->engine_->getMaxBatchSize())) {
    return Status(
        Status::Code::INVALID_ARG,
        "unexpected configuration maximum batch size " +
            std::to_string(Config().max_batch_size()) + " for '" + Name() +
            "', model maximum is " +
            std::to_string(context->engine_->getMaxBatchSize()));
  }

  // Batch output must be processed before other outputs
  RETURN_IF_ERROR(context->InitializeBatchOutputBindings(Config()));
  RETURN_IF_ERROR(
      context->InitializeConfigShapeOutputBindings(Config().output()));
  RETURN_IF_ERROR(
      context->InitializeConfigExecuteOutputBindings(Config().output()));

  // Make sure every index which corresponds to an execution binding is
  // initialized.
  for (int i = 0; i < context->num_expected_bindings_; ++i) {
    if (context->buffers_[i] == nullptr &&
        context->engine_->isExecutionBinding(i)) {
      return Status(
          Status::Code::INVALID_ARG,
          "expected configuration for " +
              std::string(
                  (context->engine_->bindingIsInput(i) ? "input" : "output")) +
              " '" + context->engine_->getBindingName(i) + "' for " + Name());
    }
  }

  // Passing the queue for available contexts here so that completion thread
  // knows where to inform that the context is ready for inputs.
  context->completion_thread_ = std::thread(
      &Context::ProcessResponse, context, context_idx, context_queue);

  // CUDA 10.1 starts to support CUDA graphs.
  // If enabled, build CUDA graphs with a set of graph specs.
#ifdef TRITON_ENABLE_CUDA_GRAPH
  const bool use_cuda_graphs = Config().optimization().cuda().graphs();
  if (use_cuda_graphs) {
    std::vector<GraphSpec> graph_specs;
    RETURN_IF_ERROR(InitializeGraphSpecs(&graph_specs));

    // CUDA graph will be captured for every TRT contexts as CUDA graph is
    // merely capturing GPU activities for a given execution.
    for (auto& graph_spec : graph_specs) {
      for (auto& trt_context : context->trt_contexts_) {
        if (UseTensorRTv2API(context->engine_)) {
          graph_spec.captured_ =
              context->BuildCudaGraphV2(&(trt_context.second), graph_spec);
        } else {
          graph_spec.captured_ =
              context->BuildCudaGraph(&(trt_context.second), graph_spec);
        }
      }
    }
  }
#endif
  context->allow_inexact_match_ =
      Config().optimization().cuda().allow_inexact_match();

  if (UseTensorRTv2API(context->engine_)) {
    std::string profiles_str;
    for (const auto& trt_context : context->trt_contexts_) {
      profiles_str +=
          (" " + trt_context.second.profile_name_ + "[" +
           std::to_string(trt_context.first) + "];");
    }
    LOG_INFO << "Created instance " << instance_name << " on GPU " << gpu_device
             << " with stream priority " << cuda_stream_priority
             << " and optimization profile" << profiles_str;
  } else {
    LOG_INFO << "Created instance " << instance_name << " on GPU " << gpu_device
             << " with stream priority " << cuda_stream_priority;
  }

  return Status::Success;
}

Status
PlanBackend::Context::ValidateInputs(
    const ::google::protobuf::RepeatedPtrField<inference::ModelInput>& ios,
    const std::set<std::string>& allowed_shape_tensors)
{
  for (const auto& io : ios) {
    if (!ConvertDataTypeToTrtType(io.data_type()).first) {
      return Status(
          Status::Code::INTERNAL,
          "unsupported datatype " + inference::DataType_Name(io.data_type()) +
              " for input '" + io.name() + "' for model '" + name_ + "'");
    }

    // Check the shape tensor specification
    if (allowed_shape_tensors.find(io.name()) != allowed_shape_tensors.end()) {
      if (!io.is_shape_tensor()) {
        return Status(
            Status::Code::INTERNAL,
            "input '" + io.name() + "' for model '" + name_ +
                "' is a shape tensor but the model configuration doesn't mark "
                "it as a shape tensor.");
      }
    } else {
      if (io.is_shape_tensor()) {
        return Status(
            Status::Code::INTERNAL,
            "input '" + io.name() + "' for model '" + name_ +
                "' is incorrectly marked as a shape tensor in the model "
                "configuration.");
      }
    }
  }

  return Status::Success;
}


Status
PlanBackend::Context::ValidateOutputs(
    const ::google::protobuf::RepeatedPtrField<inference::ModelOutput>& ios,
    const std::set<std::string>& allowed_shape_tensors)
{
  for (const auto& io : ios) {
    if (!ConvertDataTypeToTrtType(io.data_type()).first) {
      return Status(
          Status::Code::INTERNAL,
          "unsupported datatype " + inference::DataType_Name(io.data_type()) +
              " for output '" + io.name() + "' for model '" + name_ + "'");
    }

    // Check the shape tensor specification
    if (allowed_shape_tensors.find(io.name()) != allowed_shape_tensors.end()) {
      if (!io.is_shape_tensor()) {
        return Status(
            Status::Code::INTERNAL,
            "output '" + io.name() + "' for model '" + name_ +
                "' is a shape tensor but the model configuration doesn't mark "
                "it as a shape tensor.");
      }
    } else {
      if (io.is_shape_tensor()) {
        return Status(
            Status::Code::INTERNAL,
            "output '" + io.name() + "' for model '" + name_ +
                "' is incorrectly marked as a shape tensor in the model "
                "configuration.");
      }
    }
  }

  return Status::Success;
}

Status
PlanBackend::Context::InitializeShapeInputBinding(
    const std::string& input_name, const inference::DataType input_datatype,
    const DimsList& model_config_dims)
{
  // the maximum byte sizes across all profiles
  int64_t max_byte_size = 0;
  int io_index = engine_->getBindingIndex(input_name.c_str());
  for (auto& trt_context : trt_contexts_) {
    auto& profile_index = trt_context.first;
    auto& context = trt_context.second;
    int binding_index = num_expected_bindings_ * profile_index + io_index;
    if (io_index < 0) {
      return Status(
          Status::Code::NOT_FOUND,
          "input '" + input_name + "' not found for " + name_);
    }

    if (buffers_[io_index] != nullptr) {
      return Status(
          Status::Code::INVALID_ARG, "input '" + input_name +
                                         "' has already appeared as an " +
                                         "input or output for " + name_);
    }

    if (!engine_->bindingIsInput(binding_index)) {
      return Status(
          Status::Code::INVALID_ARG,
          "input '" + input_name +
              "' is expected to be an output in model for " + name_);
    }

    // Skip if the binding is not a shape tensor
    if (!engine_->isShapeBinding(binding_index)) {
      return Status::Success;
    }


    if (input_datatype != inference::DataType::TYPE_INT32) {
      return Status(
          Status::Code::INVALID_ARG,
          "unexpected datatype " + inference::DataType_Name(input_datatype) +
              "  in model configuration for shape input '" + input_name +
              "', expecting " +
              inference::DataType_Name(inference::DataType::TYPE_INT32) +
              " for " + name_);
    }

    inference::DataType dt =
        ConvertTrtTypeToDataType(engine_->getBindingDataType(binding_index));
    if (dt != input_datatype) {
      return Status(
          Status::Code::INVALID_ARG,
          "unexpected datatype " + inference::DataType_Name(dt) +
              " in engine for shape input '" + input_name + "', expecting " +
              inference::DataType_Name(input_datatype) + " for " + name_);
    }

    MemoryFormat fmt =
        ConvertTrtFmtToFmt(engine_->getBindingFormat(binding_index));
    if (fmt != MemoryFormat::LINEAR) {
      return Status(
          Status::Code::INVALID_ARG,
          "unexpected tensor format " + MemoryFormat_Name(fmt) +
              " for input '" + input_name +
              "'. Only LINEAR memory format is supported at present.");
    }
    // placeholder that does nothing
    padding_info_[binding_index] = std::make_pair(0, 0);

    nvinfer1::Dims engine_dims = engine_->getBindingDimensions(binding_index);
    if (ContainsWildcard(engine_dims)) {
      context.is_dynamic_per_binding_[io_index] = true;
    }

    RETURN_IF_ERROR(CompareShapeDimsSupported(
        name_, input_name, engine_dims, model_config_dims, support_batching_));

    RETURN_IF_ERROR(GetProfileDimensions(io_index, profile_index, &context));

    if (!context.context_->setBindingDimensions(
            binding_index, context.max_dims_[io_index])) {
      return Status(
          Status::Code::INTERNAL,
          "trt failed to set binding dimension to " +
              DimsDebugString(context.max_dims_[io_index]) + " for input '" +
              input_name + "' for " + name_);
    }

    context.nb_shape_values_ = (context.max_dims_[io_index].nbDims == 0)
                                   ? 1
                                   : context.max_dims_[io_index].d[0];
    context.max_shapes_[io_index] = engine_->getProfileShapeValues(
        binding_index, profile_index, nvinfer1::OptProfileSelector::kMAX);
    context.min_shapes_[io_index] = engine_->getProfileShapeValues(
        binding_index, profile_index, nvinfer1::OptProfileSelector::kMIN);
    context.opt_shapes_[io_index] = engine_->getProfileShapeValues(
        binding_index, profile_index, nvinfer1::OptProfileSelector::kOPT);

    if (!context.context_->setInputShapeBinding(
            binding_index, context.max_shapes_[io_index])) {
      return Status(
          Status::Code::INTERNAL,
          "trt failed to set the input shape binding for '" + input_name +
              "' for " + name_ + ".");
    }

    if (engine_->isExecutionBinding(binding_index)) {
      std::vector<int64_t> dim_vec;
      DimsToDimVec(
          context.context_->getBindingDimensions(binding_index),
          padding_info_[binding_index], &dim_vec);
      int64_t byte_size = GetByteSize(dt, dim_vec);
      max_byte_size = std::max(max_byte_size, byte_size);
    }
  }

  if (max_byte_size != NO_BATCHING) {
    // Allocate CUDA memory. We rely on buffer_bindings_ being non-nullptr to
    // indicate that the buffer has been correctly initalized so even
    // for zero-sized tensors always allocate something.
    void* buffer;
    cudaError_t err = cudaMalloc(&buffer, std::max((int64_t)1, max_byte_size));
    if (err != cudaSuccess) {
      return Status(
          Status::Code::INTERNAL, "unable to allocate memory for input '" +
                                      input_name + " for " + name_ + ": " +
                                      cudaGetErrorString(err));
    }

    byte_sizes_[io_index] = max_byte_size;
    buffers_[io_index] = buffer;

    // Set buffer bindings of all optimization profile since buffer is allocated
    for (auto& trt_context : trt_contexts_) {
      auto binding_index =
          num_expected_bindings_ * trt_context.first + io_index;
      buffer_bindings_[binding_index] = buffers_[io_index];
    }
  }

  return Status::Success;
}

Status
PlanBackend::Context::InitializeExecuteInputBinding(
    const std::string& input_name, const inference::DataType input_datatype,
    const DimsList& model_config_dims, const bool is_control,
    const bool is_ragged)
{
  // the maximum byte sizes across all profiles
  int64_t max_byte_size = 0;
  int io_index = engine_->getBindingIndex(input_name.c_str());
  for (auto& trt_context : trt_contexts_) {
    auto& profile_index = trt_context.first;
    auto& context = trt_context.second;
    int binding_index = num_expected_bindings_ * profile_index + io_index;
    if (io_index < 0) {
      return Status(
          Status::Code::NOT_FOUND,
          "input '" + input_name + "' not found for " + name_);
    }

    // Skip if shape binding is encountered
    if (engine_->isShapeBinding(binding_index)) {
      return Status::Success;
    }

    if (buffers_[io_index] != nullptr) {
      return Status(
          Status::Code::INVALID_ARG, "input '" + input_name +
                                         "' has already appeared as an " +
                                         "input or output for " + name_);
    }

    if (!engine_->bindingIsInput(binding_index)) {
      return Status(
          Status::Code::INVALID_ARG,
          "input '" + input_name +
              "' is expected to be an output in model for " + name_);
    }

    inference::DataType dt =
        ConvertTrtTypeToDataType(engine_->getBindingDataType(binding_index));
    if (dt != input_datatype) {
      return Status(
          Status::Code::INVALID_ARG,
          "unexpected datatype " + inference::DataType_Name(dt) +
              " for input '" + input_name + "', expecting " +
              inference::DataType_Name(input_datatype) + " for " + name_);
    }

    MemoryFormat fmt =
        ConvertTrtFmtToFmt(engine_->getBindingFormat(binding_index));
    if (fmt == MemoryFormat::INVALID) {
      return Status(
          Status::Code::INVALID_ARG, "unexpected tensor format " +
                                         MemoryFormat_Name(fmt) +
                                         " for input '" + input_name + "'.");
    }

    nvinfer1::Dims engine_dims = engine_->getBindingDimensions(binding_index);
    int vector_size = MemoryFormat_VectorSize(fmt);
    if (vector_size > 1) {
      int vector_dim = MemoryFormat_VectorDim(fmt);
      int dim_idx = engine_dims.nbDims - vector_dim;
      int64_t padding_offset =
          vector_size - (engine_dims.d[dim_idx] % vector_size);
      padding_info_[binding_index] = std::make_pair(dim_idx, padding_offset);
    } else {
      // placeholder that does nothing
      padding_info_[binding_index] = std::make_pair(0, 0);
    }

    // Detect whether dynamic or not
    if (ContainsWildcard(engine_dims)) {
      context.is_dynamic_per_binding_[io_index] = true;
    }


    if (!(is_control && context.is_dynamic_per_binding_[io_index])) {
      if (!is_ragged) {
        RETURN_IF_ERROR(CompareDimsSupported(
            name_, input_name, engine_dims, model_config_dims,
            support_batching_, (!engine_->hasImplicitBatchDimension()),
            false /* compare_exact */, padding_info_[binding_index]));
      } else {
        // For ragged input, the input will be concatenated and flatten, so
        // expecting engine dims to be [-1]
        if ((engine_dims.nbDims != 1) || (engine_dims.d[0] != -1)) {
          return Status(
              Status::Code::INVALID_ARG,
              "model '" + name_ + "', tensor '" + input_name +
                  "': for the model to support ragged input, the engine shape"
                  " should be [-1], got :" +
                  DimsDebugString(engine_dims));
        }
      }
    } else {
      Status status =
          ValidateControlDimsDynamic(engine_dims, support_batching_);
      if (!status.IsOk()) {
        return Status(
            Status::Code::INVALID_ARG,
            "unexpected shape " + DimsDebugString(engine_dims) +
                " for control input '" + input_name + "' for model " + name_ +
                ": " + status.Message());
      }
    }

    int64_t byte_size = 0;

    if (UseTensorRTv2API(engine_)) {
      RETURN_IF_ERROR(GetProfileDimensions(io_index, profile_index, &context));
    }

    if (UseTensorRTv2API(engine_)) {
      std::vector<int64_t> maximum_dims;
      if (!is_ragged) {
        Status status = ValidateDimension(
            model_config_dims, context.min_dims_[io_index],
            context.max_dims_[io_index], support_batching_,
            padding_info_[binding_index]);
        if (!status.IsOk()) {
          return Status(
              Status::Code::INTERNAL,
              "model configuration specified invalid shape for input '" +
                  input_name + "' for " + name_ +
                  ". Error details: " + status.Message());
        }
        RETURN_IF_ERROR(MaximumDims(
            context.max_dims_[io_index], model_config_dims, support_batching_,
            max_batch_size_, padding_info_[binding_index], &maximum_dims));
        byte_size = GetByteSize(dt, maximum_dims);
        // Update the maximum dimension with respect to the allocated buffer
        DimVecToDims(
            maximum_dims, padding_info_[binding_index],
            &context.max_dims_[io_index]);
      } else {
        byte_size = GetDataTypeByteSize(dt) * context.max_dims_[io_index].d[0];
      }

      if (!context.context_->setBindingDimensions(
              binding_index, context.max_dims_[io_index])) {
        return Status(
            Status::Code::INTERNAL,
            "trt failed to set binding dimension to " +
                DimsDebugString(context.max_dims_[io_index]) + " for input '" +
                input_name + "' for " + name_);
      }
    } else {
      byte_size = GetByteSize(max_batch_size_, dt, model_config_dims);
    }

    if (byte_size == -1) {
      return Status(
          Status::Code::INTERNAL, "unable to calculate size for input '" +
                                      input_name + " for " + name_);
    }
    max_byte_size = std::max(max_byte_size, byte_size);
  }

  // Allocate CUDA memory. We rely on buffer_bindings_ being non-nullptr to
  // indicate that the buffer has been correctly initalized so even
  // for zero-sized tensors always allocate something.
  void* buffer;
  cudaError_t err = cudaMalloc(&buffer, std::max((int64_t)1, max_byte_size));
  if (err != cudaSuccess) {
    return Status(
        Status::Code::INTERNAL, "unable to allocate memory for input '" +
                                    input_name + " for " + name_ + ": " +
                                    cudaGetErrorString(err));
  }

  byte_sizes_[io_index] = max_byte_size;
  buffers_[io_index] = buffer;
  buffer_is_ragged_[io_index] = is_ragged;

  // Set buffer bindings of all optimization profile since buffer is allocated
  for (auto& trt_context : trt_contexts_) {
    auto binding_index = num_expected_bindings_ * trt_context.first + io_index;
    buffer_bindings_[binding_index] = buffers_[io_index];
  }

  return Status::Success;
}

Status
PlanBackend::Context::InitializeSequenceControlInputBindings(
    const inference::ModelConfig& config)
{
  if (config.has_sequence_batching()) {
    std::vector<inference::ModelSequenceBatching::Control::Kind> boolean_kinds{
        inference::ModelSequenceBatching::Control::CONTROL_SEQUENCE_START,
        inference::ModelSequenceBatching::Control::CONTROL_SEQUENCE_END,
        inference::ModelSequenceBatching::Control::CONTROL_SEQUENCE_READY};

    for (const inference::ModelSequenceBatching::Control::Kind control_kind :
         boolean_kinds) {
      const bool required = false;

      std::string tensor_name;
      inference::DataType tensor_datatype;
      RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
          config.sequence_batching(), config.name(), control_kind, required,
          &tensor_name, &tensor_datatype, nullptr, nullptr, nullptr, nullptr));
      if (!tensor_name.empty()) {
        // Control tensors must have shape [1].
        DimsList dims;
        dims.Add(1);

        RETURN_IF_ERROR(InitializeExecuteInputBinding(
            tensor_name, tensor_datatype, dims, true));
      }
    }

    std::vector<inference::ModelSequenceBatching::Control::Kind> typdef_kinds{
        inference::ModelSequenceBatching::Control::CONTROL_SEQUENCE_CORRID};

    for (const inference::ModelSequenceBatching::Control::Kind control_kind :
         typdef_kinds) {
      const bool required = false;

      std::string tensor_name;
      inference::DataType tensor_datatype;
      RETURN_IF_ERROR(GetTypedSequenceControlProperties(
          config.sequence_batching(), config.name(), control_kind, required,
          &tensor_name, &tensor_datatype));
      if (!tensor_name.empty()) {
        // Control tensors must have shape [1].
        DimsList dims;
        dims.Add(1);

        RETURN_IF_ERROR(InitializeExecuteInputBinding(
            tensor_name, tensor_datatype, dims, true));
      }
    }
  }

  return Status::Success;
}

Status
PlanBackend::Context::InitializeBatchInputBindings(
    const inference::ModelConfig& config)
{
  for (const auto& batch_input : config.batch_input()) {
    for (const auto& tensor_name : batch_input.target_name()) {
      inference::DataType tensor_datatype = batch_input.data_type();
      DimsList dims;
      if (max_batch_size_ == NO_BATCHING) {
        // If the model doesn't support batching, the range of some batch input
        // kind is convergent to a fixed value, need to specify the fixed value
        // in such case.
        switch (batch_input.kind()) {
          case inference::BatchInput::BATCH_ELEMENT_COUNT:
          case inference::BatchInput::BATCH_ACCUMULATED_ELEMENT_COUNT:
            dims.Add(1);
            break;
          case inference::BatchInput::BATCH_ACCUMULATED_ELEMENT_COUNT_WITH_ZERO:
            dims.Add(2);
            break;
          default:
            dims.Add(-1);
            break;
        }
      } else {
        // Batch inputs are ragged inputs which will be concatenated and
        // flatten, so expecting dims to be [-1]
        dims.Add(-1);
      }

      RETURN_IF_ERROR(InitializeExecuteInputBinding(
          tensor_name, tensor_datatype, dims, false, true));

      int io_index = engine_->getBindingIndex(tensor_name.c_str());
      batch_inputs_[io_index].reset(new BatchInputData(
          batch_input,
          new AllocatedMemory(
              byte_sizes_[io_index], TRITONSERVER_MEMORY_CPU_PINNED, 0)));
    }
  }

  return Status::Success;
}

Status
PlanBackend::Context::InitializeConfigShapeInputBindings(
    const ::google::protobuf::RepeatedPtrField<inference::ModelInput>& ios)
{
  for (const auto& io : ios) {
    const DimsList& model_config_dims =
        (io.has_reshape()) ? io.reshape().shape() : io.dims();
    RETURN_IF_ERROR(InitializeShapeInputBinding(
        io.name(), io.data_type(), model_config_dims));
  }

  return Status::Success;
}

Status
PlanBackend::Context::InitializeConfigExecuteInputBindings(
    const ::google::protobuf::RepeatedPtrField<inference::ModelInput>& ios)
{
  for (const auto& io : ios) {
    const DimsList& model_config_dims =
        (io.has_reshape()) ? io.reshape().shape() : io.dims();
    RETURN_IF_ERROR(InitializeExecuteInputBinding(
        io.name(), io.data_type(), model_config_dims, false,
        io.allow_ragged_batch()));
  }

  return Status::Success;
}

Status
PlanBackend::Context::InitializeConfigShapeOutputBindings(
    const ::google::protobuf::RepeatedPtrField<inference::ModelOutput>& ios)
{
  for (const auto& io : ios) {
    // the maximum byte sizes across all profiles
    int64_t max_byte_size = 0;

    // Skip if this output is not a shape tensor
    if (!io.is_shape_tensor()) {
      continue;
    }

    int io_index = engine_->getBindingIndex(io.name().c_str());
    for (auto& trt_context : trt_contexts_) {
      auto& profile_index = trt_context.first;
      auto& context = trt_context.second;
      int binding_index = num_expected_bindings_ * profile_index + io_index;
      if (binding_index < 0) {
        return Status(
            Status::Code::NOT_FOUND,
            "output '" + io.name() + "' not found for " + name_);
      }

      if (buffers_[io_index] != nullptr) {
        return Status(
            Status::Code::INVALID_ARG, "output '" + io.name() +
                                           "' has already appeared as an " +
                                           "input or output for " + name_);
      }

      if (engine_->bindingIsInput(binding_index)) {
        return Status(
            Status::Code::INVALID_ARG,
            "output '" + io.name() +
                "' is expected to be an input in model for " + name_);
      }

      if (io.data_type() != inference::DataType::TYPE_INT32) {
        return Status(
            Status::Code::INVALID_ARG,
            "unexpected datatype " + inference::DataType_Name(io.data_type()) +
                "  in model configuration for shape output '" + io.name() +
                "', expecting " +
                inference::DataType_Name(inference::DataType::TYPE_INT32) +
                " for " + name_);
      }

      inference::DataType dt =
          ConvertTrtTypeToDataType(engine_->getBindingDataType(binding_index));
      if (dt != io.data_type()) {
        return Status(
            Status::Code::INVALID_ARG,
            "unexpected datatype " + inference::DataType_Name(dt) +
                " for inference output '" + io.name() + "', expecting " +
                inference::DataType_Name(io.data_type()) + " for " + name_);
      }

      MemoryFormat fmt =
          ConvertTrtFmtToFmt(engine_->getBindingFormat(binding_index));
      if (fmt != MemoryFormat::LINEAR) {
        return Status(
            Status::Code::INVALID_ARG,
            "unexpected tensor format " + MemoryFormat_Name(fmt) +
                " for output '" + io.name() +
                "'. Only LINEAR memory format is supported at present.");
      }
      // placeholder that does nothing
      padding_info_[binding_index] = std::make_pair(0, 0);

      const DimsList& model_config_dims =
          (io.has_reshape()) ? io.reshape().shape() : io.dims();

      nvinfer1::Dims engine_dims = engine_->getBindingDimensions(binding_index);
      if (ContainsWildcard(engine_dims)) {
        context.is_dynamic_per_binding_[io_index];
      }

      RETURN_IF_ERROR(CompareShapeDimsSupported(
          name_, io.name(), engine_dims, model_config_dims, support_batching_));


      const nvinfer1::Dims output_dim =
          context.context_->getBindingDimensions(binding_index);
      std::vector<int64_t> dim_vec;
      DimsToDimVec(output_dim, padding_info_[binding_index], &dim_vec);
      int64_t byte_size = GetByteSize(dt, dim_vec);

      max_byte_size = std::max(max_byte_size, byte_size);
    }

    if (max_byte_size != NO_BATCHING) {
      // Allocate CUDA memory. We rely on buffer_bindings_ being non-nullptr to
      // indicate that the buffer has been correctly initalized so even
      // for zero-sized tensors always allocate something.
      void* buffer;
      cudaError_t err =
          cudaMalloc(&buffer, std::max((int64_t)1, max_byte_size));
      if (err != cudaSuccess) {
        return Status(
            Status::Code::INTERNAL, "unable to allocate memory for input '" +
                                        io.name() + " for " + name_ + ": " +
                                        std::string(cudaGetErrorString(err)));
      }

      byte_sizes_[io_index] = max_byte_size;
      buffers_[io_index] = buffer;

      // Set buffer bindings of all optimization profile since buffer is
      // allocated
      for (auto& trt_context : trt_contexts_) {
        auto binding_index =
            num_expected_bindings_ * trt_context.first + io_index;
        buffer_bindings_[binding_index] = buffers_[io_index];
      }
    }
  }

  return Status::Success;
}

Status
PlanBackend::Context::InitializeConfigExecuteOutputBindings(
    const ::google::protobuf::RepeatedPtrField<inference::ModelOutput>& ios)
{
  for (const auto& io : ios) {
    // the maximum byte sizes across all profiles
    int64_t max_byte_size = 0;
    // Skip if the output is specified to be a shape tensor
    if (io.is_shape_tensor()) {
      continue;
    }
    int io_index = engine_->getBindingIndex(io.name().c_str());
    for (auto& trt_context : trt_contexts_) {
      auto& profile_index = trt_context.first;
      auto& context = trt_context.second;
      int binding_index = num_expected_bindings_ * profile_index + io_index;
      if (binding_index < 0) {
        return Status(
            Status::Code::NOT_FOUND,
            "output '" + io.name() + "' not found for " + name_);
      }

      if (buffers_[io_index] != nullptr) {
        return Status(
            Status::Code::INVALID_ARG, "output '" + io.name() +
                                           "' has already appeared as an " +
                                           "input or output for " + name_);
      }

      if (engine_->bindingIsInput(binding_index)) {
        return Status(
            Status::Code::INVALID_ARG,
            "output '" + io.name() +
                "' is expected to be an input in model for " + name_);
      }

      inference::DataType dt =
          ConvertTrtTypeToDataType(engine_->getBindingDataType(binding_index));
      if (dt != io.data_type()) {
        return Status(
            Status::Code::INVALID_ARG,
            "unexpected datatype " + inference::DataType_Name(dt) +
                " for inference output '" + io.name() + "', expecting " +
                inference::DataType_Name(io.data_type()) + " for " + name_);
      }

      MemoryFormat fmt =
          ConvertTrtFmtToFmt(engine_->getBindingFormat(binding_index));
      if (fmt == MemoryFormat::INVALID) {
        return Status(
            Status::Code::INVALID_ARG, "unexpected tensor format " +
                                           MemoryFormat_Name(fmt) +
                                           " for output '" + io.name() + "'.");
      }

      const DimsList& model_config_dims =
          (io.has_reshape()) ? io.reshape().shape() : io.dims();

      nvinfer1::Dims engine_dims = engine_->getBindingDimensions(binding_index);
      int vector_size = MemoryFormat_VectorSize(fmt);
      if (vector_size > 1) {
        int vector_dim = MemoryFormat_VectorDim(fmt);
        int dim_idx = engine_dims.nbDims - vector_dim;
        int64_t padding_offset =
            vector_size - (engine_dims.d[dim_idx] % vector_size);
        padding_info_[binding_index] = std::make_pair(dim_idx, padding_offset);
      } else {
        // placeholder that does nothing
        padding_info_[binding_index] = std::make_pair(0, 0);
      }

      // Skip 'batch_output' validation as it is not exact match to model dims
      if (!buffer_is_ragged_[io_index]) {
        RETURN_IF_ERROR(CompareDimsSupported(
            name_, io.name(), engine_dims, model_config_dims, support_batching_,
            (!engine_->hasImplicitBatchDimension()), false /* compare_exact */,
            padding_info_[binding_index]));
      }

      int64_t byte_size;
      if (UseTensorRTv2API(engine_)) {
        const nvinfer1::Dims output_dim =
            context.context_->getBindingDimensions(binding_index);
        std::vector<int64_t> dim_vec;
        DimsToDimVec(output_dim, padding_info_[binding_index], &dim_vec);
        byte_size = GetByteSize(dt, dim_vec);
      } else {
        byte_size = GetByteSize(max_batch_size_, dt, model_config_dims);
      }

      if (byte_size == -1) {
        return Status(
            Status::Code::INTERNAL, "unable to calculate size for output '" +
                                        io.name() + " for " + name_);
      }
      max_byte_size = std::max(max_byte_size, byte_size);
    }

    // Allocate CUDA memory. We rely on buffer_bindings_ being non-nullptr to
    // indicate that the buffer has been correctly initalized so even
    // for zero-sized tensors always allocate something.
    void* buffer;
    cudaError_t err = cudaMalloc(&buffer, std::max((int64_t)1, max_byte_size));
    if (err != cudaSuccess) {
      return Status(
          Status::Code::INTERNAL, "unable to allocate memory for input '" +
                                      io.name() + " for " + name_ + ": " +
                                      std::string(cudaGetErrorString(err)));
    }

    byte_sizes_[io_index] = max_byte_size;
    buffers_[io_index] = buffer;
    // Whether the output needs to be scattered based on input
    if (buffer_is_ragged_[io_index]) {
      std::vector<int64_t> output_shape;
      const DimsList& model_config_dims =
          (io.has_reshape()) ? io.reshape().shape() : io.dims();
      if (support_batching_) {
        output_shape.push_back(-1);
      }
      for (const auto& dim : model_config_dims) {
        output_shape.push_back(dim);
      }
      io_shape_mapping_[io_index].second = output_shape;
    }

    // Set buffer bindings of all optimization profile since buffer is allocated
    for (auto& trt_context : trt_contexts_) {
      auto binding_index =
          num_expected_bindings_ * trt_context.first + io_index;
      buffer_bindings_[binding_index] = buffers_[io_index];
    }
  }

  return Status::Success;
}

Status
PlanBackend::Context::InitializeBatchOutputBindings(
    const inference::ModelConfig& config)
{
  for (const auto& io : config.batch_output()) {
    for (const auto& name : io.target_name()) {
      // FIXME Currently not handling the case that batch output is shape tensor
      int io_index = engine_->getBindingIndex(name.c_str());

      if (engine_->isShapeBinding(io_index)) {
        return Status(
            Status::Code::INVALID_ARG,
            "batch output '" + name + "' can not be shape binding");
      }

      // Whether the output needs to be scattered based on input
      if (io.kind() != inference::BatchOutput::BATCH_SCATTER_WITH_INPUT_SHAPE) {
        return Status(
            Status::Code::INVALID_ARG,
            "batch output kind other than"
            "BATCH_SCATTER_WITH_INPUT_SHAPE is not supported for " +
                name_);
      }
      // Set hints to for InitializeBatchOutputBindings()
      buffer_is_ragged_[io_index] = true;
      io_shape_mapping_[io_index] =
          std::make_pair(io.source_input(0), std::vector<int64_t>());
    }
  }

  return Status::Success;
}

Status
PlanBackend::Context::GetProfileDimensions(
    const int io_index, const int profile_index, TensorRTContext* context)
{
  int binding_index = (profile_index * num_expected_bindings_) + io_index;
  context->max_dims_[io_index] = engine_->getProfileDimensions(
      binding_index, profile_index, nvinfer1::OptProfileSelector::kMAX);
  context->min_dims_[io_index] = engine_->getProfileDimensions(
      binding_index, profile_index, nvinfer1::OptProfileSelector::kMIN);
  context->opt_dims_[io_index] = engine_->getProfileDimensions(
      binding_index, profile_index, nvinfer1::OptProfileSelector::kOPT);
  return Status::Success;
}

// CUDA 10.1 starts to support CUDA graphs.
#ifdef TRITON_ENABLE_CUDA_GRAPH
Status
PlanBackend::InitializeGraphSpecs(std::vector<GraphSpec>* graph_specs)
{
  graph_specs->clear();
  if (Config().optimization().cuda().graph_spec_size() == 0) {
    // No graph spec is provided, use default specs
    // Graphs are most likely to help for small batch sizes so by
    // default build for batch sizes 1, 2, 3, 4, 6, 8, 12, 16, 'max_batch_size'.
    // If preferred batch size is specified, then the batch sizes will be
    // 1, preferred batch sizes, 'max_batch_size'.
    std::set<int> cuda_graph_batch_sizes{1};
    if (Config().has_dynamic_batching()) {
      for (const auto bs : Config().dynamic_batching().preferred_batch_size()) {
        cuda_graph_batch_sizes.emplace(bs);
      }
    } else if (
        Config().has_sequence_batching() &&
        Config().sequence_batching().has_oldest()) {
      for (const auto bs :
           Config().sequence_batching().oldest().preferred_batch_size()) {
        cuda_graph_batch_sizes.emplace(bs);
      }
    } else {
      cuda_graph_batch_sizes = {1, 2, 3, 4, 6, 8, 12, 16};
    }
    if (Config().max_batch_size() > 0) {
      cuda_graph_batch_sizes.emplace(Config().max_batch_size());
    }

    for (const auto bs : cuda_graph_batch_sizes) {
      graph_specs->emplace_back();
      graph_specs->back().batch_size_ = bs;
    }
  } else {
    for (const auto& config_spec :
         Config().optimization().cuda().graph_spec()) {
      graph_specs->emplace_back();
      auto& graph_spec = graph_specs->back();
      graph_spec.batch_size_ = config_spec.batch_size();
      for (const auto& input : config_spec.input()) {
        std::vector<int64_t> input_shape;
        for (const auto& dim : input.second.dim()) {
          input_shape.emplace_back(dim);
        }
        graph_spec.shapes_[input.first] = std::move(input_shape);
      }
    }
  }
  return Status::Success;
}

bool
PlanBackend::Context::BuildCudaGraph(
    TensorRTContext* trt_context, const GraphSpec& graph_spec)
{
  // 1 is special case as non-batching model has 'max_batch_size == 0'
  int batch_size = (graph_spec.batch_size_ == 0) ? 1 : graph_spec.batch_size_;
  std::vector<int64_t> cuda_graph_key{batch_size};
  auto cuda_graph = TensorRTContext::CudaGraph();
  for (int bindex = 0; bindex < num_expected_bindings_; ++bindex) {
    // FIXME handle shape tensor properly, for now if model uses shape tensor
    // then cuda graph is not captured
    if (engine_->isShapeBinding(bindex)) {
      LOG_WARNING << "Detected shape tensor, CUDA graph is not captured for "
                  << name_;
      return false;
    }
  }

  // Enqueue to TRT to setup resources properly BEFORE capturing CUDA graph
  if (!trt_context->context_->enqueue(
          batch_size, buffer_bindings_.data(), stream_, nullptr)) {
    LOG_WARNING << "unable to record CUDA graph for " << name_;
    return false;
  }

  bool captured = true;
  for (int set_idx = 0; set_idx < EVENT_SET_COUNT; set_idx++) {
    // The same spec has been captured
    if (trt_context->cuda_graph_execs_[set_idx].find(cuda_graph_key) !=
        trt_context->cuda_graph_execs_[set_idx].end()) {
      LOG_WARNING << "Detected duplicated CUDA graph specification for "
                  << name_ << ", skipping the duplicated specification";
      return true;
    }
    cudaGraph_t graph;
    auto cuerr = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "unable to start CUDA graph for " << name_ << ": "
                << cudaGetErrorString(cuerr);
      captured = false;
    } else {
      auto context = trt_context->context_;
      if (!context->enqueue(
              batch_size, buffer_bindings_.data(), stream_,
              &events_[set_idx].ready_for_input_)) {
        LOG_WARNING << "unable to record CUDA graph for " << name_;
        captured = false;
      }

      cuerr = cudaStreamEndCapture(stream_, &graph);
      if (cuerr != cudaSuccess) {
        LOG_ERROR << "unable to finish CUDA graph for " << name_ << ": "
                  << cudaGetErrorString(cuerr);
        captured = false;
      }

      if (captured) {
        cudaGraphExec_t graph_exec;
        cuerr = cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
        if (cuerr != cudaSuccess) {
          LOG_ERROR << "unable to instantiate CUDA graph for " << name_ << ": "
                    << cudaGetErrorString(cuerr);
          captured = false;
        } else {
          cuda_graph.cuda_graph_exec_ = graph_exec;

          trt_context->cuda_graphs_.push_back(graph);
          trt_context->cuda_graph_execs_[set_idx].insert(
              std::make_pair(cuda_graph_key, cuda_graph));
        }
      }
    }
  }

  if (captured) {
    LOG_VERBOSE(1) << "captured CUDA graph for " << name_ << ", batch size "
                   << batch_size;
  }

  return captured;
}

bool
PlanBackend::Context::BuildCudaGraphV2(
    TensorRTContext* trt_context, const GraphSpec& graph_spec)
{
  // FIXME handle shape tensor properly, for now if model uses shape tensor
  // then cuda graph is not captured
  for (int i = 0; i < num_expected_bindings_; ++i) {
    if (engine_->isShapeBinding(i)) {
      LOG_WARNING << "Detected shape tensor, CUDA graph is not captured for "
                  << name_;
      return false;
    }
  }

  std::vector<int64_t> cuda_graph_key;
  auto cuda_graph = TensorRTContext::CudaGraph();
  if (!SetCudaGraphShape(trt_context, graph_spec, &cuda_graph_key, &cuda_graph)
           .IsOk()) {
    return false;
  }

  // Enqueue to TRT to setup resources properly BEFORE capturing CUDA graph
  if (!trt_context->context_->enqueueV2(
          buffer_bindings_.data(), stream_, nullptr)) {
    LOG_WARNING << "unable to record CUDA graph for " << name_;
    return false;
  }

  bool captured = true;

  for (int set_idx = 0; set_idx < EVENT_SET_COUNT; set_idx++) {
    cudaGraph_t graph;
    auto cuerr = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "unable to start CUDA graph for " << name_ << ": "
                << cudaGetErrorString(cuerr);
      captured = false;
    } else {
      auto context = trt_context->context_;
      if (!context->enqueueV2(
              buffer_bindings_.data(), stream_,
              &events_[set_idx].ready_for_input_)) {
        LOG_WARNING << "unable to record CUDA graph for " << name_;
        captured = false;
      }

      cuerr = cudaStreamEndCapture(stream_, &graph);
      if (cuerr != cudaSuccess) {
        LOG_ERROR << "unable to finish CUDA graph for " << name_ << ": "
                  << cudaGetErrorString(cuerr);
        captured = false;
      }

      if (captured) {
        cudaGraphExec_t graph_exec;
        cuerr = cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
        if (cuerr != cudaSuccess) {
          LOG_ERROR << "unable to instantiate CUDA graph for " << name_ << ": "
                    << cudaGetErrorString(cuerr);
          captured = false;
        } else {
          cuda_graph.cuda_graph_exec_ = graph_exec;

          trt_context->cuda_graphs_.push_back(graph);
          trt_context->cuda_graph_execs_[set_idx].insert(
              std::make_pair(cuda_graph_key, cuda_graph));
        }
      }
    }
  }

  if (captured) {
    LOG_VERBOSE(1) << "captured CUDA graph for " << name_ << ", batch size "
                   << graph_spec.batch_size_;
  }

  return captured;
}
#endif

Status
PlanBackend::Context::SetCudaGraphShape(
    TensorRTContext* trt_context, const GraphSpec& graph_spec,
    std::vector<int64_t>* cuda_graph_key,
    TensorRTContext::CudaGraph* cuda_graph)
{
  // 1 is special case as non-batching model has 'max_batch_size == 0'
  int batch_size = (graph_spec.batch_size_ == 0) ? 1 : graph_spec.batch_size_;
  int binding_offset = trt_context->profile_idx_ * num_expected_bindings_;
  *cuda_graph_key = std::vector<int64_t>{batch_size};
  for (int bindex = 0; bindex < num_expected_bindings_; bindex++) {
    auto io_index = binding_offset + bindex;
    if (!engine_->bindingIsInput(io_index)) {
      continue;
    }
    // Empty shapes indicates the graph spec is added by default,
    // for default graph spec, opt dims are used.
    if (graph_spec.shapes_.empty()) {
      auto shape = trt_context->opt_dims_[bindex];
      shape.d[0] = batch_size;
      if (!trt_context->context_->setBindingDimensions(io_index, shape)) {
        return Status(
            Status::Code::INTERNAL,
            "trt failed to set binding dimension to " + DimsDebugString(shape) +
                " for binding " + std::to_string(io_index) + " for " + name_);
      }
      std::vector<int64_t> dims;
      DimsToDimVec(shape, padding_info_[io_index], &dims);
      cuda_graph->input_dims_.emplace_back(dims);
      cuda_graph_key->insert(cuda_graph_key->end(), dims.begin(), dims.end());
    } else {
      const std::string& name = engine_->getBindingName(bindex);
      auto it = graph_spec.shapes_.find(name);
      if (it != graph_spec.shapes_.end()) {
        // For ragged input, assume the shape in graph spec is proper shape
        // after ragged.
        if (buffer_is_ragged_[bindex]) {
          cuda_graph->input_dims_.emplace_back();
        } else {
          cuda_graph->input_dims_.emplace_back();
          cuda_graph->input_dims_.back().push_back(batch_size);
        }
        auto& shape = cuda_graph->input_dims_.back();
        shape.insert(shape.end(), it->second.begin(), it->second.end());
        nvinfer1::Dims trt_shape;
        DimVecToDims(shape, padding_info_[io_index], &trt_shape);
        if (!trt_context->context_->setBindingDimensions(io_index, trt_shape)) {
          return Status(
              Status::Code::INTERNAL,
              "trt failed to set binding dimension to " +
                  DimsDebugString(trt_shape) + " for binding " +
                  std::to_string(io_index) + " for " + name_);
        }
        cuda_graph_key->insert(
            cuda_graph_key->end(), shape.begin(), shape.end());
      } else {
        return Status(
            Status::Code::INVALID_ARG,
            "trt failed to set binding dimension for unknown input '" + name +
                "' for " + name_);
      }
    }
  }
  return Status::Success;
}

void
PlanBackend::Context::FindClosestCudaGraph(
    const TensorRTContext& trt_context,
    const std::vector<int64_t>& cuda_graph_key,
    const TensorRTContext::CudaGraph** cuda_graph, bool* found_exact)
{
  *cuda_graph = nullptr;
  auto itr =
      trt_context.cuda_graph_execs_[next_set_].lower_bound(cuda_graph_key);
  if (itr != trt_context.cuda_graph_execs_[next_set_].end()) {
    *found_exact = (itr->first == cuda_graph_key);
    if (*found_exact) {
      *cuda_graph = &itr->second;
      return;
    } else if (allow_inexact_match_) {
      // For vector as key, returned lower bound may not satisfy requirements
      // that all dims must be >= actual dims
      for (; itr != trt_context.cuda_graph_execs_[next_set_].end(); itr++) {
        bool found = true;
        for (size_t key_idx = 0; key_idx < cuda_graph_key.size(); key_idx++) {
          if (cuda_graph_key[key_idx] > itr->first[key_idx]) {
            found = false;
            break;
          }
        }
        if (found) {
          *cuda_graph = &itr->second;
          return;
        }
      }
    }
  }
  return;
}

PlanBackend::~PlanBackend()
{
  // Must destory all TensorRT contexts before engine
  contexts_.clear();

  for (auto& device_engine : device_engines_) {
    cudaSetDevice(device_engine.first);
    auto& runtime = device_engine.second.first;
    auto& engine = device_engine.second.second;
    if (engine != nullptr) {
      engine->destroy();
      engine = nullptr;
    }
    if (runtime != nullptr) {
      runtime->destroy();
      runtime = nullptr;
    }
  }
}

void
PlanBackend::Run(
    uint32_t runner_idx,
    std::vector<std::unique_ptr<InferenceRequest>>&& requests)
{
  // Each runner executes using the corresponding context...
  if (runner_idx >= available_context_queue_.size()) {
    InferenceRequest::RespondIfError(
        requests, Status(
                      Status::Code::INTERNAL,
                      "unexpected runner index" + std::to_string(runner_idx) +
                          ", max allowed " +
                          std::to_string(available_context_queue_.size())));
    return;
  }

  contexts_[next_context_[runner_idx]]->Run(this, std::move(requests));

  auto context =
      static_cast<Context*>(contexts_[next_context_[runner_idx]].get());

  bool run_failed = true;
  for (const auto& request : context->payload_->requests_) {
    if (request != nullptr) {
      run_failed = false;
      break;
    }
  }

  // On error, handle the response here instead of delegating to the completion
  // thread as the completion thread will wait on CUDA events unconditionally,
  // which can be ignored on error.
  if (run_failed) {
    // On inference error, place the context back to the queue immediately
    // as all works for the context should be ignored.
    available_context_queue_[runner_idx]->Put(next_context_[runner_idx]);

  } else {
    auto event_set_idx = context->next_set_;
    context->next_set_ = (event_set_idx + 1) % context->EVENT_SET_COUNT;
    // Put the details needed by the ProcessResponse thread on the queue
    context->completion_queue_.Put(std::move(context->payload_));
  }

  // Set the next context to be executed on this runner, will block
  // until there is available context for the runner
  next_context_[runner_idx] = available_context_queue_[runner_idx]->Get();
}


void
PlanBackend::WarmUp(uint32_t runner_idx, WarmupData& sample)
{
  // Different from Run(), the contexts in available_context_queue_[runner_idx]
  // also need to be executed
  //
  // Exhaust available contexts for the 'runner_idx', and also get
  // the number of available contexts
  std::vector<size_t> contexts;
  while (!available_context_queue_[runner_idx]->Empty()) {
    contexts.push_back(available_context_queue_[runner_idx]->Get());
  }
  contexts.push_back(next_context_[runner_idx]);

  std::vector<std::promise<void>> completion_promises(contexts.size());
  for (size_t idx = 0; idx < contexts.size(); idx++) {
    std::vector<std::unique_ptr<InferenceRequest>> requests;
    // Duplicate the sample request if it is not the last context.
    bool run_failed =
        !DuplicateWarmupRequests(sample.requests_, &requests).IsOk();
    requests.back()->SetReleaseCallback(
        WarmupRequestComplete, &completion_promises[idx]);
    // Capture timestamp before run to avoid incorrect accumulation from
    // sequential warmup runs
    for (auto& request : requests) {
#ifdef TRITON_ENABLE_STATS
      request->CaptureRequestStartNs();
#endif  // TRITON_ENABLE_STATS
      request->CaptureQueueStartNs();
    }

    auto context = static_cast<Context*>(contexts_[contexts[idx]].get());
    if (!run_failed) {
      context->Run(this, std::move(requests));
      // If one of the contexts can't run properly, the whole warmup should
      // abort
      run_failed = true;
      for (const auto& request : context->payload_->requests_) {
        if (request != nullptr) {
          run_failed = false;
          break;
        }
      }
    }
    if (run_failed) {
      // Clean up the rest of the contexts back to queue,
      // the contexts before will be handled by completion function
      available_context_queue_[runner_idx]->Put(contexts[idx]);
      for (auto rest_idx = idx + 1; rest_idx < contexts.size(); rest_idx++) {
        available_context_queue_[runner_idx]->Put(contexts[rest_idx]);
        completion_promises[rest_idx].set_value();
      }
      break;
    }

    // Place in completion queue
    auto event_set_idx = context->next_set_;
    context->next_set_ = (event_set_idx + 1) % context->EVENT_SET_COUNT;
    context->completion_queue_.Put(std::move(context->payload_));
  }

  // Wait for all inflight executions to be finished.
  for (auto& completion_promise : completion_promises) {
    completion_promise.get_future().get();
  }
  for (auto& request : sample.requests_) {
    InferenceRequest::Release(
        std::move(request), TRITONSERVER_REQUEST_RELEASE_ALL);
  }

  // Need to reset the next context to be executed on this runner
  // as all contexts are in the queue at this point
  next_context_[runner_idx] = available_context_queue_[runner_idx]->Get();
}

Status
PlanBackend::DuplicateWarmupRequests(
    const std::vector<std::unique_ptr<InferenceRequest>>& warmup_requests,
    std::vector<std::unique_ptr<InferenceRequest>>* requests)
{
  for (auto& request : warmup_requests) {
    // Need to construct the request via standard procedure to set up
    // unexposed request members properly
    requests->emplace_back(new InferenceRequest(this, Version()));
    auto& lrequest = requests->back();
    for (const auto& input_pair : request->OriginalInputs()) {
      InferenceRequest::Input* linput;
      RETURN_IF_ERROR(lrequest->AddOriginalInput(
          input_pair.first, input_pair.second.DType(),
          input_pair.second.OriginalShape(), &linput));
      RETURN_IF_ERROR(linput->SetData(input_pair.second.Data()));
    }
    lrequest->PrepareForInference();
    for (const auto& override_input_pair : request->OverrideInputs()) {
      RETURN_IF_ERROR(lrequest->AddOverrideInput(override_input_pair.second));
    }

    RETURN_IF_ERROR(lrequest->SetResponseCallback(
        &warmup_allocator, nullptr, WarmupResponseComplete, nullptr));
    RETURN_IF_ERROR(
        lrequest->SetReleaseCallback(WarmupRequestComplete, nullptr));
  }
  return Status::Success;
}


bool
PlanBackend::Context::SetOutputShapeTensorBuffer(
    const int32_t* content, std::unique_ptr<InferenceResponse>* response,
    InferenceResponse::Output* response_output,
    const size_t tensor_element_count, const int64_t batch_size,
    cudaStream_t stream)
{
  bool cuda_copy = false;

  const size_t expected_byte_size = tensor_element_count * sizeof(int32_t);

  // Allocate a buffer large enough to hold the serialized tensor.
  TRITONSERVER_MemoryType actual_memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t actual_memory_type_id = 0;

  char* buffer;
  Status status = response_output->AllocateDataBuffer(
      (void**)&buffer, expected_byte_size, &actual_memory_type,
      &actual_memory_type_id);
  if (!status.IsOk()) {
    LOG_STATUS_ERROR(
        InferenceResponse::SendWithStatus(
            std::move(*response), TRITONSERVER_RESPONSE_COMPLETE_FINAL, status),
        "error sending TRT response");
    return cuda_copy;
  }

  const size_t nb_shape_values = tensor_element_count / batch_size;

  // Copy the serialized tensor into the allocated buffer.
  bool cuda_used = false;
  size_t content_offset = support_batching_ ? 1 : 0;
  size_t buffer_offset = 0;
  for (int i = 0; i < batch_size; i++) {
    status = CopyBuffer(
        response_output->Name(), TRITONSERVER_MEMORY_CPU /* src_memory_type */,
        0 /* src_memory_type_id */, actual_memory_type, actual_memory_type_id,
        nb_shape_values * sizeof(int32_t), (void*)(content + content_offset),
        (void*)(buffer + buffer_offset), stream_, &cuda_used);
    cuda_copy |= cuda_used;
    buffer_offset += (nb_shape_values * sizeof(int32_t));
  }

  if (!status.IsOk()) {
    LOG_STATUS_ERROR(
        InferenceResponse::SendWithStatus(
            std::move(*response), TRITONSERVER_RESPONSE_COMPLETE_FINAL, status),
        "error sending TensorFlow response");
    return cuda_copy;
  }

  return cuda_copy;
}

void
PlanBackend::Context::Run(
    InferenceBackend* base,
    std::vector<std::unique_ptr<InferenceRequest>>&& requests)
{
  Status status;

  LOG_VERBOSE(1) << "Running " << name_ << " with " << requests.size()
                 << " requests";

  NVTX_RANGE(nvtx_, "Run " + name_);

  // Need to move the InferenceRequest objects as the lifetime must
  // be extended till ProcessResponse completes. The requsts
  payload_.reset(new Payload(base, next_set_, std::move(requests)));
  INFER_STATS_SET_TIMESTAMP(payload_->compute_start_ns_);

  cudaSetDevice(gpu_device_);

  const InferenceRequest* repr_input_request = nullptr;

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.
  for (auto& request : payload_->requests_) {
    if (request == nullptr) {
      InferenceRequest::RespondIfError(
          payload_->requests_,
          Status(
              Status::Code::INTERNAL,
              "null request given to TensorRT runner for '" + name_ + "'"));
      return;
    }

    payload_->total_batch_size_ += std::max(1U, request->BatchSize());

    // All requests must have equally-sized input tensors so use any
    // request as the representative for the input tensors.
    repr_input_request = request.get();
  }

  // If there are no valid requests then no need to run the
  // inference. This should never happen unless called with an empty
  // 'requests' for some reason.
  if (payload_->total_batch_size_ == 0) {
    return;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size_ == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if ((payload_->total_batch_size_ != 1) &&
      (payload_->total_batch_size_ > (size_t)max_batch_size_)) {
    InferenceRequest::RespondIfError(
        payload_->requests_,
        Status(
            Status::Code::INTERNAL,
            "dynamic batch size " +
                std::to_string(payload_->total_batch_size_) + " for '" + name_ +
                "', max allowed is " + std::to_string(max_batch_size_)),
        true /* release_requests */);
    return;
  }

  std::map<int32_t, std::vector<int32_t>> request_shape_values;
  // Scheduler ensures all the requests have identical shape values so use
  // values from any shape tensor
  status = GetRequestShapeValues(
      payload_->total_batch_size_, payload_->requests_.front(),
      &request_shape_values);
  if (!status.IsOk()) {
    InferenceRequest::RespondIfError(
        payload_->requests_, status, true /* release_requests */);
    return;
  }

  std::map<int, PlanBackend::Context::TensorRTContext>::iterator citr;
  status = GetMostOptimizedProfile(
      payload_->total_batch_size_, payload_->requests_, request_shape_values,
      &citr);

  if (!status.IsOk()) {
    LOG_ERROR << status.Message();
  }

  int binding_offset = citr->first * num_expected_bindings_;

  // At this point we are committed to running inference with all
  // 'requests'. Create a response for each request. During input
  // processing if there is an error with any request that error will
  // be sent immediately with the corresponding response (and the
  // response unique_ptr will then be nullptr). The request object
  // itself will not be released until after all inferencing is done
  // (below) as we may need to access the request object when
  // determine how to process outputs (for example, even if we don't
  // need the outputs for a request that has an error, we do need to
  // know the size of those outputs associated with the request so we
  // can skip them in the output tensors).
  payload_->responses_.reserve(payload_->requests_.size());

  for (auto& request : payload_->requests_) {
    std::unique_ptr<InferenceResponse> response;
    Status status = request->ResponseFactory().CreateResponse(&response);
    if (!status.IsOk()) {
      InferenceRequest::RespondIfError(request, status);
      response.reset();
    }

    payload_->responses_.emplace_back(std::move(response));
  }

  // For each input, concatenate input values from each request into
  // the corresponding binding.
  std::vector<int64_t> input_dims{(int64_t)payload_->total_batch_size_};
  BackendInputCollector collector(
      payload_->requests_, &payload_->responses_, enable_pinned_input_,
      input_copy_stream_, events_[next_set_].input_ready_);
  for (int bindex = 0; bindex < num_expected_bindings_; ++bindex) {
    int io_index = binding_offset + bindex;
    if (!engine_->bindingIsInput(io_index)) {
      continue;
    }

    const std::string& name = engine_->getBindingName(bindex);

    // Set the shape binding if needed. If unable to set the shape binding
    // then fail all requests.
    if (engine_->isShapeBinding(io_index)) {
      auto it = request_shape_values.find(io_index);
      if (it != request_shape_values.end()) {
        status = ValidateShapeValues(
            it->second, citr->second.min_shapes_[io_index],
            citr->second.max_shapes_[io_index], citr->second.nb_shape_values_,
            support_batching_);
      } else {
        status = Status(
            Status::Code::INTERNAL,
            "unable to find shape values for shape input '" + name +
                "' in request for " + name_);
        FAIL_ALL_AND_RETURN_IF_ERROR(
            payload_->requests_, payload_->responses_, metric_reporter_.get(),
            status, "missing shape values for the shape tensor");
      }
      if (status.IsOk()) {
        citr->second.context_->setInputShapeBinding(io_index, &(it->second[0]));
      } else {
        FAIL_ALL_AND_RETURN_IF_ERROR(
            payload_->requests_, payload_->responses_, metric_reporter_.get(),
            status, "invalid shape values encountered for shape inputs");
      }
    }

    // Skip the upcoming section if not an execution tensor
    if (!engine_->isExecutionBinding(io_index)) {
      continue;
    }

    if (buffer_is_ragged_[bindex]) {
      std::vector<int64_t> ragged_shape{0};
      inference::DataType datatype;
      // FIXME inefficient as looping in this way may iterate the same
      // source_input multiple times
      if (batch_inputs_[bindex] != nullptr) {
        const auto& batch_input = batch_inputs_[bindex]->first;
        auto& allocated_memory = batch_inputs_[bindex]->second;
        TRITONSERVER_MemoryType mem_type;
        int64_t mem_type_id;
        char* input_buffer =
            allocated_memory->MutableBuffer(&mem_type, &mem_type_id);
        FAIL_ALL_AND_RETURN_IF_ERROR(
            payload_->requests_, payload_->responses_, metric_reporter_.get(),
            collector.BatchInputShape(batch_input, &ragged_shape),
            "error getting the bath input shape");

        datatype = batch_input.data_type();
        const size_t total_byte_size = GetByteSize(datatype, ragged_shape);

        FAIL_ALL_AND_RETURN_IF_ERROR(
            payload_->requests_, payload_->responses_, metric_reporter_.get(),
            collector.ProcessBatchInput(
                batch_input, input_buffer, total_byte_size, mem_type,
                mem_type_id),
            "error setting the bath input value");

        FAIL_ALL_AND_RETURN_IF_ERROR(
            payload_->requests_, payload_->responses_, metric_reporter_.get(),
            SetBindingDimensions(
                name, ragged_shape, citr->second, bindex, io_index,
                &input_dims),
            "error setting the binding dimension");

        if (batch_input.kind() !=
            inference::BatchInput::BATCH_MAX_ELEMENT_COUNT_AS_SHAPE) {
          bool cuda_used = false;
          FAIL_ALL_AND_RETURN_IF_ERROR(
              payload_->requests_, payload_->responses_, metric_reporter_.get(),
              CopyBuffer(
                  name, mem_type, mem_type_id, TRITONSERVER_MEMORY_GPU,
                  gpu_device_, total_byte_size, input_buffer, buffers_[bindex],
                  input_copy_stream_, &cuda_used),
              "error copying the batch input buffer");
          if (cuda_used) {
            cudaEventRecord(
                events_[next_set_].input_ready_, input_copy_stream_);
          }
        }
      } else {
        for (size_t req_idx = 0; req_idx < payload_->requests_.size();
             req_idx++) {
          const InferenceRequest::Input* repr_input;
          FAIL_ALL_AND_RETURN_IF_ERROR(
              payload_->requests_, payload_->responses_, metric_reporter_.get(),
              payload_->requests_[req_idx]->ImmutableInput(name, &repr_input),
              "failed to obtain the input '" + name + "'");
          ragged_shape[0] += GetElementCount(repr_input->ShapeWithBatchDim());
          if (req_idx == 0) {
            datatype = repr_input->DType();
          }
        }

        const size_t total_byte_size = GetByteSize(datatype, ragged_shape);

        FAIL_ALL_AND_RETURN_IF_ERROR(
            payload_->requests_, payload_->responses_, metric_reporter_.get(),
            SetBindingDimensions(
                name, ragged_shape, citr->second, bindex, io_index,
                &input_dims),
            "error setting the binding dimension");

        collector.ProcessTensor(
            name, datatype, static_cast<char*>(buffers_[bindex]),
            total_byte_size, TRITONSERVER_MEMORY_GPU, gpu_device_);
      }
    } else {
      const InferenceRequest::Input* repr_input;
      FAIL_ALL_AND_RETURN_IF_ERROR(
          payload_->requests_, payload_->responses_, metric_reporter_.get(),
          repr_input_request->ImmutableInput(name, &repr_input),
          "failed to obtain the input '" + name + "'");
      // Get the shape of the input. The request has already checked
      // that the request shape is valid so don't need to do it here.
      const auto& batch1_shape = repr_input->Shape();

      // The shape for the entire input batch, [total_batch_size, ...]
      std::vector<int64_t> batchn_shape;
      batchn_shape.reserve(batch1_shape.size() + 1);
      if (max_batch_size_ != NO_BATCHING) {
        if (!engine_->isShapeBinding(io_index)) {
          batchn_shape.push_back(payload_->total_batch_size_);
        }
      }
      batchn_shape.insert(
          batchn_shape.end(), batch1_shape.begin(), batch1_shape.end());
      const inference::DataType datatype = repr_input->DType();

      const size_t total_byte_size = GetByteSize(datatype, batchn_shape);

      // Set the binding dimension so that output dimensions can be obtained
      if (UseTensorRTv2API(engine_) && !engine_->isShapeBinding(io_index)) {
        FAIL_ALL_AND_RETURN_IF_ERROR(
            payload_->requests_, payload_->responses_, metric_reporter_.get(),
            SetBindingDimensions(
                name, batchn_shape, citr->second, bindex, io_index,
                &input_dims),
            "error setting the binding dimension");
      }

      if ((engine_->isShapeBinding(io_index)) && (support_batching_)) {
        // Set the first 4 bytes to the shape value representing the
        // batch size.
        bool cuda_used = false;
        status = CopyBuffer(
            name, TRITONSERVER_MEMORY_CPU, 0, TRITONSERVER_MEMORY_GPU,
            gpu_device_, sizeof(int32_t), (void*)&payload_->total_batch_size_,
            static_cast<char*>(buffers_[bindex]), input_copy_stream_,
            &cuda_used);
        FAIL_ALL_AND_RETURN_IF_ERROR(
            payload_->requests_, payload_->responses_, metric_reporter_.get(),
            status, "error input data for the batch");

        // Copy rest of the shape values to the buffer.
        status = CopyBuffer(
            name, TRITONSERVER_MEMORY_CPU, 0, TRITONSERVER_MEMORY_GPU,
            gpu_device_, total_byte_size, (void*)&request_shape_values[bindex],
            (static_cast<char*>(buffers_[bindex]) + sizeof(int32_t)),
            input_copy_stream_, &cuda_used);
        FAIL_ALL_AND_RETURN_IF_ERROR(
            payload_->requests_, payload_->responses_, metric_reporter_.get(),
            status, "error input data");
      } else {
        collector.ProcessTensor(
            name, datatype, static_cast<char*>(buffers_[bindex]),
            total_byte_size, TRITONSERVER_MEMORY_GPU, gpu_device_);
      }
    }
  }
  collector.Finalize();

  const TensorRTContext::CudaGraph* cuda_graph = nullptr;
  bool found_exact = false;
  // FIXME closest_cuda_graph
  FindClosestCudaGraph(citr->second, input_dims, &cuda_graph, &found_exact);
  if ((cuda_graph != nullptr) && !found_exact && (UseTensorRTv2API(engine_))) {
    size_t input_idx = 0;
    for (int bindex = 0; bindex < num_expected_bindings_; ++bindex) {
      int io_index = binding_offset + bindex;
      if (!engine_->bindingIsInput(io_index) ||
          engine_->isShapeBinding(io_index)) {
        continue;
      }
      FAIL_ALL_AND_RETURN_IF_ERROR(
          payload_->requests_, payload_->responses_, metric_reporter_.get(),
          SetBindingDimensions(
              "CUDA graph input", cuda_graph->input_dims_[input_idx],
              citr->second, bindex, io_index, nullptr),
          "error setting the binding dimension");
      // Initialize additional entries in batch input
      if (batch_inputs_[bindex] != nullptr) {
        const auto& batch_input = batch_inputs_[bindex]->first;
        const size_t total_byte_size = GetByteSize(
            batch_input.data_type(), cuda_graph->input_dims_[input_idx]);

        auto& allocated_memory = batch_inputs_[bindex]->second;
        TRITONSERVER_MemoryType mem_type;
        int64_t mem_type_id;
        char* input_buffer =
            allocated_memory->MutableBuffer(&mem_type, &mem_type_id);

        FAIL_ALL_AND_RETURN_IF_ERROR(
            payload_->requests_, payload_->responses_, metric_reporter_.get(),
            collector.ProcessBatchInput(
                batch_input, input_buffer, total_byte_size, mem_type,
                mem_type_id),
            "error setting the bath input value");
        if (batch_input.kind() !=
            inference::BatchInput::BATCH_MAX_ELEMENT_COUNT_AS_SHAPE) {
          bool cuda_used = false;
          FAIL_ALL_AND_RETURN_IF_ERROR(
              payload_->requests_, payload_->responses_, metric_reporter_.get(),
              CopyBuffer(
                  "CUDA graph batch input", mem_type, mem_type_id,
                  TRITONSERVER_MEMORY_GPU, gpu_device_, total_byte_size,
                  input_buffer, buffers_[bindex], input_copy_stream_,
                  &cuda_used),
              "error copying the batch input buffer");
          if (cuda_used) {
            cudaEventRecord(
                events_[next_set_].input_ready_, input_copy_stream_);
          }
        }
      }
      input_idx++;
    }
  }

  // Ensure inputs are ready before execution. Output buffers will always be
  // available at this point as the execution and output copy are on the same
  // stream.
  cudaStreamWaitEvent(stream_, events_[next_set_].input_ready_, 0);

  // Async execute the inference using a CUDA graph if available for
  // the batch-size, otherwise execution normally.
  if (cuda_graph != nullptr) {
    LOG_VERBOSE(1) << "Context with profile " << citr->second.profile_name_
                   << " [" << std::to_string(citr->first)
                   << "] is launching CUDA graph for " << name_;
    cudaError_t err = cudaGraphLaunch(cuda_graph->cuda_graph_exec_, stream_);
    if (err != cudaSuccess) {
      cudaStreamSynchronize(stream_);
      FAIL_ALL_AND_RETURN_IF_ERROR(
          payload_->requests_, payload_->responses_, metric_reporter_.get(),
          Status(
              Status::Code::INTERNAL, "unable to execute graph for inference " +
                                          name_ + ": " +
                                          cudaGetErrorString(err)),
          "failed to run TRT inference");
    }
    // Event recorded during CUDA graph capture is not visible outside of the
    // graph, need to explicitly record it.
    cudaEventRecord(events_[next_set_].ready_for_input_, stream_);
  } else {
    LOG_VERBOSE(1) << "Context with profile " << citr->second.profile_name_
                   << " [" << std::to_string(citr->first)
                   << "] is being executed for " << name_;
    if (!citr->second.context_->allInputDimensionsSpecified()) {
      FAIL_ALL_AND_RETURN_IF_ERROR(
          payload_->requests_, payload_->responses_, metric_reporter_.get(),
          Status(
              Status::Code::INTERNAL,
              "failed to specify the dimensions of all input bindings"),
          "failed to run TRT inference");
    }
    if (!citr->second.context_->allInputShapesSpecified()) {
      FAIL_ALL_AND_RETURN_IF_ERROR(
          payload_->requests_, payload_->responses_, metric_reporter_.get(),
          Status(
              Status::Code::INTERNAL,
              "failed to specify the values for all input shape tensors"),
          "failed to run TRT inference");
    }
    if (UseTensorRTv2API(engine_)) {
      if (!citr->second.context_->enqueueV2(
              buffer_bindings_.data(), stream_,
              &events_[next_set_].ready_for_input_)) {
        cudaStreamSynchronize(stream_);
        FAIL_ALL_AND_RETURN_IF_ERROR(
            payload_->requests_, payload_->responses_, metric_reporter_.get(),
            Status(
                Status::Code::INTERNAL,
                "unable to enqueue for inference " + name_),
            "failed to run TRT inference");
      }
    } else {
      if (!citr->second.context_->enqueue(
              payload_->total_batch_size_, buffer_bindings_.data(), stream_,
              &events_[next_set_].ready_for_input_)) {
        cudaStreamSynchronize(stream_);
        FAIL_ALL_AND_RETURN_IF_ERROR(
            payload_->requests_, payload_->responses_, metric_reporter_.get(),
            Status(
                Status::Code::INTERNAL,
                "unable to enqueue for inference " + name_),
            "failed to run TRT inference");
      }
    }
  }

  cudaEventRecord(events_[next_set_].ready_for_output_, stream_);

  // Collect the names of requested outputs. Do not include outputs
  // for requests that have already responded with an error.
  std::set<std::string> required_outputs;
  for (size_t idx = 0; idx < payload_->requests_.size(); idx++) {
    const auto& request = payload_->requests_[idx];
    const auto& response = payload_->responses_[idx];
    if (response != nullptr) {
      for (const auto& output_name : request->ImmutableRequestedOutputs()) {
        required_outputs.insert(output_name);
      }
    }
  }

  // For each requested output verify that the output can accept the
  // actual model output and then copy that output from the GPU
  payload_->responder_.reset(new BackendResponder(
      payload_->requests_, &payload_->responses_, max_batch_size_,
      enable_pinned_output_, stream_, events_[next_set_].output_ready_));
  for (int bindex = 0; bindex < num_expected_bindings_; ++bindex) {
    int io_index = binding_offset + bindex;
    if (engine_->bindingIsInput(io_index)) {
      continue;
    }

    const std::string& name = engine_->getBindingName(bindex);

    nvinfer1::Dims dims;
    dims = citr->second.context_->getBindingDimensions(io_index);

    // Make sure each output is of the expected size and copy it into
    // the payload responses.
    bool cuda_copy = false;
    if (engine_->isShapeBinding(io_index)) {
      // Custom handling for shape tensors
      // Obtain the shape value
      if (dims.nbDims != 0) {
        int32_t* shape_value_ptr =
            (int32_t*)malloc(dims.d[0] * sizeof(int32_t));
        if (!citr->second.context_->getShapeBinding(
                io_index, shape_value_ptr)) {
          FAIL_ALL_AND_RETURN_IF_ERROR(
              payload_->requests_, payload_->responses_, metric_reporter_.get(),
              Status(
                  Status::Code::INTERNAL,
                  "failed to retrieve the output shape values from binding '" +
                      name + "'"),
              "failed to get TRT response");
        }

        // The first shape value must be equal to the total batch_size
        if (support_batching_ &&
            payload_->total_batch_size_ != (uint32_t)*shape_value_ptr) {
          FAIL_ALL_AND_RETURN_IF_ERROR(
              payload_->requests_, payload_->responses_, metric_reporter_.get(),
              Status(
                  Status::Code::INTERNAL,
                  "unexpected batch shape value " +
                      std::to_string(*shape_value_ptr) + " for '" + name +
                      "', total batch size was " +
                      std::to_string(payload_->total_batch_size_)),
              "failed to run TRT response");
        }

        std::vector<int64_t> batchn_shape;
        if (support_batching_) {
          batchn_shape.push_back(payload_->total_batch_size_);
          batchn_shape.push_back(dims.d[0] - 1);
        } else {
          batchn_shape.push_back(dims.d[0]);
        }

        for (size_t idx = 0; idx < payload_->responses_.size(); idx++) {
          auto& request = payload_->requests_[idx];
          auto& response = payload_->responses_[idx];

          if (support_batching_) {
            batchn_shape[0] = request->BatchSize();
          }

          const size_t tensor_element_cnt = GetElementCount(batchn_shape);

          inference::DataType dt = ConvertTrtTypeToDataType(
              engine_->getBindingDataType(binding_offset + bindex));

          // Only need an response tensor for requested outputs.
          if ((response != nullptr) &&
              (request->ImmutableRequestedOutputs().find(name) !=
               request->ImmutableRequestedOutputs().end())) {
            InferenceResponse::Output* response_output = nullptr;
            response->AddOutput(name, dt, batchn_shape, &response_output);
            cuda_copy |= SetOutputShapeTensorBuffer(
                shape_value_ptr, &response, response_output, tensor_element_cnt,
                batchn_shape[0], stream_);
          }
        }

        free(shape_value_ptr);
      }
    } else if (buffer_is_ragged_[bindex]) {
      // FIXME add correctness checking like below
      inference::DataType dt = ConvertTrtTypeToDataType(
          engine_->getBindingDataType(binding_offset + bindex));
      payload_->responder_->ProcessTensor(
          name, io_shape_mapping_[bindex].first, dt,
          io_shape_mapping_[bindex].second,
          static_cast<const char*>(buffers_[bindex]), TRITONSERVER_MEMORY_GPU,
          gpu_device_);
    } else {
      std::vector<int64_t> batchn_shape;

      if (engine_->hasImplicitBatchDimension() && support_batching_) {
        batchn_shape.push_back(payload_->total_batch_size_);
      }

      for (int i = 0; i < dims.nbDims; ++i) {
        batchn_shape.push_back(dims.d[i]);
      }

      inference::DataType dt = ConvertTrtTypeToDataType(
          engine_->getBindingDataType(binding_offset + bindex));

      size_t batch1_byte_size = GetByteSize(dt, batchn_shape);
      if (support_batching_) {
        batch1_byte_size /= payload_->total_batch_size_;
      }

      if (byte_sizes_[bindex] <
          (batch1_byte_size * payload_->total_batch_size_)) {
        FAIL_ALL_AND_RETURN_IF_ERROR(
            payload_->requests_, payload_->responses_, metric_reporter_.get(),
            Status(
                Status::Code::INTERNAL,
                "unexpected size for output '" + name + "', byte-size " +
                    std::to_string(byte_sizes_[bindex]) + " is less than " +
                    std::to_string(payload_->total_batch_size_) + " * " +
                    std::to_string(batch1_byte_size)),
            "failed to run TRT response");
      }

      payload_->responder_->ProcessTensor(
          name, dt, batchn_shape, static_cast<const char*>(buffers_[bindex]),
          TRITONSERVER_MEMORY_GPU, gpu_device_);
    }
  }
}

Status
PlanBackend::Context::SetBindingDimensions(
    const std::string& input_name, const std::vector<int64_t>& shape,
    const TensorRTContext& trt_context, const size_t binding_idx,
    const size_t io_idx, std::vector<int64_t>* input_dims)
{
  if (input_dims != nullptr) {
    input_dims->insert(input_dims->end(), shape.begin(), shape.end());
  }
  nvinfer1::Dims this_dim;
  // Set the binding dimension so that output dimensions can be obtained
  if (!DimVecToDims(shape, padding_info_[io_idx], &this_dim)) {
    return Status(
        Status::Code::INTERNAL, "failed to create dims object for " +
                                    DimsListToString(shape) + " for input '" +
                                    input_name + "' for " + name_ + ".");
  }
  auto status = ValidateDimension(
      this_dim, trt_context.min_dims_[binding_idx],
      trt_context.max_dims_[binding_idx], false);
  if (!status.IsOk()) {
    return Status(
        Status::Code::INTERNAL, "request specifies invalid shape for input '" +
                                    input_name + "' for " + name_ +
                                    ". Error details: " + status.Message());
  }

  if (!trt_context.is_dynamic_per_binding_[binding_idx]) {
    // No need to set dimension for the binding that does not inlcude
    // dynamic shape.
    return Status::Success;
  }

  if (!trt_context.context_->setBindingDimensions(io_idx, this_dim)) {
    return Status(
        Status::Code::INTERNAL, "trt failed to set binding dimension to " +
                                    DimsDebugString(this_dim) + " for input '" +
                                    input_name + "' for " + name_);
  }
  return Status::Success;
}

void
PlanBackend::Context::ProcessResponse(
    size_t context_idx, std::shared_ptr<SyncQueue<size_t>> context_queue)
{
  while (true) {
    NVTX_RANGE(nvtx_, "ProcessResponse " + context_idx);
    auto payload = std::move(completion_queue_.Get());
    if (payload.get() == nullptr) {
      break;
    }
    auto& event_set = events_[payload->event_set_idx_];

#ifdef TRITON_ENABLE_STATS
    cudaEventSynchronize(event_set.input_ready_);
    INFER_STATS_DECL_TIMESTAMP(compute_input_end_ns);
#endif  // TRITON_ENABLE_STATS

    // The model execution associated with the current context
    // has consumed the inputs. Put the context back into the available queue
    // so that it can begin enqueuing new memcpys into the input buffers
    cudaEventSynchronize(event_set.ready_for_input_);
    context_queue->Put(context_idx);
    NVTX_MARKER("plan_input_available");

#ifdef TRITON_ENABLE_STATS
    cudaEventSynchronize(event_set.ready_for_output_);
    INFER_STATS_DECL_TIMESTAMP(compute_output_start_ns);
#endif  // TRITON_ENABLE_STATS

    // Call Finalize() here to defer CUDA synchronization as much as possible
    payload->responder_->Finalize();
    cudaEventSynchronize(event_set.output_ready_);
    NVTX_MARKER("plan_output_ready");
    // Compute ends when the output data copy is completed
    INFER_STATS_DECL_TIMESTAMP(compute_end_ns);

#ifdef TRITON_ENABLE_STATS

    // Report stats and trace
    for (size_t i = 0; i < payload->requests_.size(); ++i) {
      auto& request = payload->requests_[i];
      request->ReportStatistics(
          metric_reporter_.get(), (payload->responses_[i] != nullptr),
          payload->compute_start_ns_, compute_input_end_ns,
          compute_output_start_ns, compute_end_ns);

#ifdef TRITON_ENABLE_TRACING
      if (request->Trace() != nullptr) {
        auto& trace = request->Trace();
        trace->Report(
            TRITONSERVER_TRACE_COMPUTE_START, payload->compute_start_ns_);
        trace->Report(
            TRITONSERVER_TRACE_COMPUTE_INPUT_END, compute_input_end_ns);
        trace->Report(
            TRITONSERVER_TRACE_COMPUTE_OUTPUT_START, compute_output_start_ns);
        trace->Report(TRITONSERVER_TRACE_COMPUTE_END, compute_end_ns);
      }
#endif  // TRITON_ENABLE_TRACING
    }

    // Also reporting batch stats
    payload->inference_backend_->MutableStatsAggregator()
        ->UpdateInferBatchStats(
            metric_reporter_.get(), payload->total_batch_size_,
            payload->compute_start_ns_, compute_input_end_ns,
            compute_output_start_ns, compute_end_ns);
#endif  // TRITON_ENABLE_STATS

    // Send all the responses that haven't already been sent because of
    // an earlier error.
    for (auto& response : payload->responses_) {
      if (response != nullptr) {
        LOG_STATUS_ERROR(
            InferenceResponse::Send(
                std::move(response), TRITONSERVER_RESPONSE_COMPLETE_FINAL),
            "failed to send TRT backend response");
      }
    }

    // Release all requests.
    for (auto& request : payload->requests_) {
      InferenceRequest::Release(
          std::move(request), TRITONSERVER_REQUEST_RELEASE_ALL);
    }
  }
}

Status
PlanBackend::Context::InitEventSet(bool busy_wait_events)
{
  unsigned int event_flags =
      (busy_wait_events ? cudaEventDefault : cudaEventBlockingSync) |
      cudaEventDisableTiming;

  for (size_t idx = 0; idx < EVENT_SET_COUNT; idx++) {
    RETURN_IF_ERROR(CreateCudaEvent(
        "Set " + std::to_string(idx) + " ready for input", event_flags,
        &events_[idx].ready_for_input_));
    RETURN_IF_ERROR(CreateCudaEvent(
        "Set " + std::to_string(idx) + " input ready", event_flags,
        &events_[idx].input_ready_));
    RETURN_IF_ERROR(CreateCudaEvent(
        "Set " + std::to_string(idx) + " ready for output", event_flags,
        &events_[idx].ready_for_output_));
    RETURN_IF_ERROR(CreateCudaEvent(
        "Set " + std::to_string(idx) + " output ready", event_flags,
        &events_[idx].output_ready_));
  }
  return Status::Success;
}

Status
PlanBackend::Context::DestroyEventSet()
{
  for (size_t idx = 0; idx < EVENT_SET_COUNT; idx++) {
    if (events_[idx].ready_for_input_ != nullptr) {
      cudaEventDestroy(events_[idx].ready_for_input_);
    }
    if (events_[idx].input_ready_ != nullptr) {
      cudaEventDestroy(events_[idx].input_ready_);
    }
    if (events_[idx].ready_for_output_ != nullptr) {
      cudaEventDestroy(events_[idx].ready_for_output_);
    }
    if (events_[idx].output_ready_ != nullptr) {
      cudaEventDestroy(events_[idx].output_ready_);
    }
  }
  return Status::Success;
}

Status
PlanBackend::Context::GetRequestShapeValues(
    size_t total_batch_size, const std::unique_ptr<InferenceRequest>& request,
    std::map<int, std::vector<int32_t>>* request_shape_values)
{
  // Visit all the inputs and extract the shape values present in the request
  Status status;
  for (const auto& pr : request->ImmutableInputs()) {
    const std::string& input_name = pr.first;
    const auto& repr_input = pr.second;
    const auto& batch1_shape = repr_input->Shape();

    int io_index = engine_->getBindingIndex(input_name.c_str());
    if (engine_->isShapeBinding(io_index)) {
      auto it =
          request_shape_values->emplace(io_index, std::vector<int32_t>()).first;
      if (max_batch_size_ != NO_BATCHING) {
        it->second.push_back((int32_t)total_batch_size);
      }

      // For now being conservative and requiring that shape tensors
      // be in a single buffer on the CPU. We can handle more cases in
      // future if necessary.
      const auto& data = repr_input->Data();
      if (data->BufferCount() != 1) {
        return Status(
            Status::Code::INVALID_ARG,
            "shape tensor for input '" + input_name +
                "' must be in single contiguous buffer on CPU");
      }

      size_t data_byte_size;
      TRITONSERVER_MemoryType data_memory_type;
      int64_t data_memory_id;
      const char* data_buffer = data->BufferAt(
          0 /* idx */, &data_byte_size, &data_memory_type, &data_memory_id);
      if ((data_buffer == nullptr) ||
          (data_memory_type == TRITONSERVER_MEMORY_GPU)) {
        return Status(
            Status::Code::INVALID_ARG,
            "shape tensor for input '" + input_name +
                "' must be in single contiguous buffer on CPU");
      }

      // Shape tensors datatype is INT32.
      const int64_t element_cnt = GetElementCount(batch1_shape);
      const size_t expected_byte_size =
          element_cnt * GetDataTypeByteSize(inference::DataType::TYPE_INT32);

      if (expected_byte_size != data_byte_size) {
        return Status(
            Status::Code::INVALID_ARG, "shape tensor for input '" + input_name +
                                           "' expected byte size is " +
                                           std::to_string(expected_byte_size) +
                                           ", got " +
                                           std::to_string(data_byte_size));
      }

      const int32_t* dims = reinterpret_cast<const int32_t*>(data_buffer);
      for (int64_t i = 0; i < element_cnt; ++i) {
        it->second.push_back(dims[i]);
      }
    }
  }

  return Status::Success;
}

Status
PlanBackend::Context::GetMostOptimizedProfile(
    size_t total_batch_size,
    const std::vector<std::unique_ptr<InferenceRequest>>& requests,
    const std::map<int, std::vector<int32_t>>& request_shape_values,
    std::map<int, PlanBackend::Context::TensorRTContext>::iterator* citr)
{
  // Returns the TensorRT context that uses profile with shortest Manhattan
  // distance in terms of input dimensions
  // [TODO] traverse it with more efficient data structure (i.e. K-D tree)
  *citr = trt_contexts_.begin();
  if (trt_contexts_.size() != 1) {
    int64_t shortest_distance = LLONG_MAX;
    for (auto cit = trt_contexts_.begin(); cit != trt_contexts_.end(); cit++) {
      int64_t current_distance = 0;
      EvaluateTensorRTContext(
          cit, total_batch_size, requests, request_shape_values,
          &current_distance);
      if (current_distance < shortest_distance) {
        *citr = cit;
        shortest_distance = current_distance;
      }
    }
    if (shortest_distance == LLONG_MAX) {
      std::string profiles_str;
      for (const auto& trt_context : trt_contexts_) {
        profiles_str +=
            (" " + trt_context.second.profile_name_ + "[" +
             std::to_string(trt_context.first) + "]");
      }
      return Status(
          Status::Code::INVALID_ARG,
          "failed to find any Optimization Profile among [" + profiles_str +
              "] to support the "
              "requested dimensions (or shape values), proceeding with first "
              "profile.");
    }
  }

  LOG_VERBOSE(1) << "Optimization profile " << (*citr)->second.profile_name_
                 << " [" << std::to_string((*citr)->first)
                 << "] is selected for " << name_;

  return Status::Success;
}


Status
PlanBackend::Context::EvaluateTensorRTContext(
    std::map<int, PlanBackend::Context::TensorRTContext>::iterator& citr,
    size_t total_batch_size,
    const std::vector<std::unique_ptr<InferenceRequest>>& requests,
    const std::map<int, std::vector<int32_t>>& request_shape_values,
    int64_t* error_distance)
{
  *error_distance = 0;
  int binding_offset = citr->second.profile_idx_ * num_expected_bindings_;
  for (const auto& pr : requests[0]->ImmutableInputs()) {
    const auto input = pr.second;
    int io_index = engine_->getBindingIndex(input->Name().c_str());
    if (buffer_is_ragged_[io_index]) {
      std::vector<int64_t> shape{0};
      for (const auto& request : requests) {
        const InferenceRequest::Input* repr_input;
        request->ImmutableInput(input->Name(), &repr_input);
        shape[0] += GetElementCount(repr_input->ShapeWithBatchDim());
      }
      auto status = ValidateDimension(
          shape, citr->second.min_dims_[io_index],
          citr->second.max_dims_[io_index], false,
          padding_info_[binding_offset + io_index]);
      if (!status.IsOk()) {
        *error_distance = LLONG_MAX;
        break;
      } else {
        const auto& opt_dims = citr->second.opt_dims_[io_index];
        *error_distance += std::abs(opt_dims.d[0] - shape[0]);
      }
    } else {
      auto status = ValidateDimension(
          input->Shape(), citr->second.min_dims_[io_index],
          citr->second.max_dims_[io_index], support_batching_,
          padding_info_[binding_offset + io_index]);
      bool valid_bs =
          (!support_batching_) || (((int64_t)total_batch_size >=
                                    citr->second.min_dims_[io_index].d[0]) &&
                                   ((int64_t)total_batch_size <=
                                    citr->second.max_dims_[io_index].d[0]));

      bool missing_shape_values = false;
      if (engine_->isShapeBinding(io_index)) {
        auto it = request_shape_values.find(io_index);
        if (it != request_shape_values.end()) {
          status = ValidateShapeValues(
              it->second, citr->second.min_shapes_[io_index],
              citr->second.max_shapes_[io_index], citr->second.nb_shape_values_,
              support_batching_);
          valid_bs =
              (!support_batching_) || (((int32_t)total_batch_size >=
                                        *citr->second.min_shapes_[io_index]) &&
                                       ((int64_t)total_batch_size <=
                                        *citr->second.max_shapes_[io_index]));
        } else {
          missing_shape_values = true;
        }
      }

      if (!status.IsOk() || !valid_bs || missing_shape_values) {
        *error_distance = LLONG_MAX;
        break;
      } else {
        const auto& opt_dims = citr->second.opt_dims_[io_index];
        *error_distance += std::abs(opt_dims.d[0] - (int64_t)total_batch_size);
        for (int idx = 1; idx < opt_dims.nbDims; idx++) {
          *error_distance +=
              std::abs(opt_dims.d[idx] - input->Shape()[idx - 1]);
        }
        if (engine_->isShapeBinding(io_index)) {
          const auto* opt_shape_values = citr->second.opt_shapes_[io_index];
          *error_distance +=
              std::abs(*opt_shape_values - (int64_t)total_batch_size);
          auto it = request_shape_values.find(io_index);
          for (size_t idx = 1; idx < citr->second.nb_shape_values_; idx++) {
            *error_distance +=
                std::abs(*(opt_shape_values + idx) - it->second[idx - 1]);
          }
        }
      }
    }
  }

  return Status::Success;
}

std::ostream&
operator<<(std::ostream& out, const PlanBackend& pb)
{
  out << "name=" << pb.Name() << std::endl;
  out << "contexts:" << std::endl;
  for (size_t idx = 0; idx < pb.contexts_.size(); idx++) {
    auto context = static_cast<PlanBackend::Context*>(pb.contexts_[idx].get());
    out << "  name=" << context->name_ << ", gpu="
        << ((context->gpu_device_ == PlanBackend::Context::NO_GPU_DEVICE)
                ? "<none>"
                : std::to_string(context->gpu_device_))
        << ", max_batch_size="
        << ((context->max_batch_size_ == PlanBackend::Context::NO_BATCHING)
                ? "<none>"
                : std::to_string(context->max_batch_size_))
        << std::endl
        << "  bindings:" << std::endl;

    for (int i = 0; i < context->num_expected_bindings_; ++i) {
      out << "    " << i
          << ": max possible byte_size=" << context->byte_sizes_[i]
          << ", buffer=" << context->buffers_[i] << " ]" << std::endl;
    }
  }

  return out;
}

}}  // namespace nvidia::inferenceserver
