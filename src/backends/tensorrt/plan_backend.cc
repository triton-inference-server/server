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
#include "src/core/infer_stats.h"
#include "src/core/logging.h"
#include "src/core/model_config_cuda.h"
#include "src/core/model_config_utils.h"
#include "src/core/nvtx.h"

namespace nvidia { namespace inferenceserver {

namespace {

Status
CreateCudaEvent(const std::string& event_name, cudaEvent_t* event)
{
  // Not adding 'cudaEventBlockingSync' to reduce gaps between the time of
  // event record and the time of signaling blocking thread.
  // The busy waiting only happens when there is inflight request.
  auto cuerr = cudaEventCreateWithFlags(event, cudaEventDisableTiming);
  if (cuerr != cudaSuccess) {
    return Status(
        Status::Code::INTERNAL, "unable to create CUDA event for " +
                                    event_name + ": " +
                                    cudaGetErrorString(cuerr));
  }
  return Status::Success;
}

}  // namespace

PlanBackend::Context::Context(
    const std::string& name, const int gpu_device, const int max_batch_size,
    const bool enable_pinned_input, const bool enable_pinned_output)
    : BackendContext(
          name, gpu_device, max_batch_size, enable_pinned_input,
          enable_pinned_output),
      engine_(nullptr), is_shared_engine_(true), is_dynamic_(false),
      total_bindings_(0), num_expected_bindings_(0)
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
    for (const auto& pr : trt_context.second.cuda_graph_execs_) {
      cudaError_t err = cudaGraphExecDestroy(pr.second);
      if (err != cudaSuccess) {
        LOG_ERROR << "Failed to destroy cuda graph exec: "
                  << cudaGetErrorString(err);
      }
    }
    trt_context.second.cuda_graph_execs_.clear();

    for (const auto& pr : trt_context.second.cuda_graphs_) {
      cudaError_t err = cudaGraphDestroy(pr.second);
      if (err != cudaSuccess) {
        LOG_ERROR << "Failed to destroy cuda graph exec: "
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
  completion_queue_.Put(std::make_tuple(nullptr, nullptr, 0, nullptr));
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
    if ((group.kind() != ModelInstanceGroup::KIND_GPU) ||
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
          uint32_t runner_idx, std::vector<Scheduler::Payload>* payloads,
          std::function<void(Status)> func) {
        Run(runner_idx, payloads, func);
      },
      [this](
          uint32_t runner_idx, const InferenceRequest::Input& input,
          const Scheduler::Payload& payload,
          std::vector<int64_t>* shape) -> Status {
        return PeekShapeTensor(runner_idx, input, payload, shape);
      }));

  LOG_VERBOSE(1) << "plan backend for " << Name() << std::endl << *this;

  return Status::Success;
}

Status
PlanBackend::PeekShapeTensor(
    uint32_t runner_idx, const InferenceRequest::Input& input,
    const Scheduler::Payload& payload, std::vector<int64_t>* shape)
{
  // Each runner performs the peek using the corresponding since it
  // may require a CUDA stream to get the tensor contents.
  if (runner_idx >= contexts_.size()) {
    return Status(
        Status::Code::INTERNAL,
        "unexpected runner index" + std::to_string(runner_idx) +
            ", max allowed " + std::to_string(contexts_.size()));
  }

  return contexts_[runner_idx]->PeekShapeTensor(input, payload, shape);
}

Status
PlanBackend::Context::InitOptimizationProfiles(
    const ::google::protobuf::RepeatedPtrField<std::string>& profile_names)
{
  total_bindings_ = engine_->getNbBindings();
  const int total_profiles = engine_->getNbOptimizationProfiles();

  // TRT sets the optimization profile index to be 0 implicitly with the first
  // context creation. As currently TRTIS supports one context per engine,
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
            .emplace(0, TensorRTContext("default", num_expected_bindings_))
            .first;
    it->second.context_ = default_trt_context;
    default_trt_context = nullptr;
  } else {
    // Create one TRT context for each specified profile
    for (const auto& profile_name : profile_names) {
      int profile_index = 0;
      RETURN_IF_ERROR(GetProfileIndex(profile_name, &profile_index));
      auto res = trt_contexts_.emplace(
          profile_index, TensorRTContext(profile_name, num_expected_bindings_));
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
        if (!res.first->second.context_->setOptimizationProfile(
                profile_index)) {
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

  contexts_.emplace_back(
      new Context(instance_name, gpu_device, mbs, pinned_input, pinned_output));
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
  RETURN_IF_ERROR(context->InitEventSet());

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
  context->buffer_bindings_ =
      std::vector<void*>(context->total_bindings_, nullptr);

  RETURN_IF_ERROR(
      context->InitializeConfigShapeInputBindings(Config().input()));
  RETURN_IF_ERROR(
      context->InitializeConfigExecuteInputBindings(Config().input()));
  RETURN_IF_ERROR(context->InitializeSequenceControlInputBindings(Config()));
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
  // If enabled, build CUDA graphs for a default set of graph
  // sizes. Graphs are most likely to help for small batch sizes so by
  // default build for batch sizes 1, 2, 3, 4, 6, 8, 12, 16, 'max_batch_size'.
  // If preferred batch size is specified, then the batch sizes will be
  // 1, preferred batch sizes, 'max_batch_size'. If any
  // build fails don't attempt for any larger batch sizes.
#ifdef TRTIS_ENABLE_CUDA_GRAPH
  const bool use_cuda_graphs = Config().optimization().cuda().graphs();
  if (use_cuda_graphs) {
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

    // CUDA graph will be captured for every TRT contexts as CUDA graph is
    // merely capturing GPU activities for a given execution.
    // But CUDA graph will only be captured for fixed shape model as it only
    // captures activities for the shapes used, so it may misbehave for other
    // shapes.
    if (!context->is_dynamic_) {
      for (auto& trt_context : context->trt_contexts_) {
        for (int bs : cuda_graph_batch_sizes) {
          // 1 is special case as non-batching model has 'max_batch_size == 0'
          if ((bs <= Config().max_batch_size()) || (bs == 1)) {
            if (!context->BuildCudaGraph(&(trt_context.second), bs)) {
              break;
            }
          }
        }
      }
    }
  }
#endif

  if (context->is_dynamic_) {
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
    const ::google::protobuf::RepeatedPtrField<ModelInput>& ios,
    const std::set<std::string>& allowed_shape_tensors)
{
  for (const auto& io : ios) {
    if (!ConvertDataTypeToTrtType(io.data_type()).first) {
      return Status(
          Status::Code::INTERNAL,
          "unsupported datatype " + DataType_Name(io.data_type()) +
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
    const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios,
    const std::set<std::string>& allowed_shape_tensors)
{
  for (const auto& io : ios) {
    if (!ConvertDataTypeToTrtType(io.data_type()).first) {
      return Status(
          Status::Code::INTERNAL,
          "unsupported datatype " + DataType_Name(io.data_type()) +
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
    const std::string& input_name, const DataType input_datatype,
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

    // The presence of shape binding indicates the dynamic model plan
    is_dynamic_ = true;

    if (input_datatype != TYPE_INT32) {
      return Status(
          Status::Code::INVALID_ARG,
          "unexpected datatype " + DataType_Name(input_datatype) +
              "  in model configuration for shape input '" + input_name +
              "', expecting " + DataType_Name(TYPE_INT32) + " for " + name_);
    }

    DataType dt =
        ConvertTrtTypeToDataType(engine_->getBindingDataType(binding_index));
    if (dt != input_datatype) {
      return Status(
          Status::Code::INVALID_ARG,
          "unexpected datatype " + DataType_Name(dt) +
              " in engine for shape input '" + input_name + "', expecting " +
              DataType_Name(input_datatype) + " for " + name_);
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

    nvinfer1::Dims engine_dims = engine_->getBindingDimensions(binding_index);

    RETURN_IF_ERROR(CompareShapeDimsSupported(
        name_, input_name, engine_dims, model_config_dims, support_batching_));

    context.max_dims_[io_index] = engine_->getProfileDimensions(
        binding_index, profile_index, nvinfer1::OptProfileSelector::kMAX);
    context.min_dims_[io_index] = engine_->getProfileDimensions(
        binding_index, profile_index, nvinfer1::OptProfileSelector::kMIN);
    context.opt_dims_[io_index] = engine_->getProfileDimensions(
        binding_index, profile_index, nvinfer1::OptProfileSelector::kOPT);

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
          context.context_->getBindingDimensions(binding_index), &dim_vec);
      int64_t byte_size = GetByteSize(dt, dim_vec);
      max_byte_size = std::max(max_byte_size, byte_size);
    }
  }

  if (max_byte_size != 0) {
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
    const std::string& input_name, const DataType input_datatype,
    const DimsList& model_config_dims, const bool is_control)
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

    DataType dt =
        ConvertTrtTypeToDataType(engine_->getBindingDataType(binding_index));
    if (dt != input_datatype) {
      return Status(
          Status::Code::INVALID_ARG,
          "unexpected datatype " + DataType_Name(dt) + " for input '" +
              input_name + "', expecting " + DataType_Name(input_datatype) +
              " for " + name_);
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

    nvinfer1::Dims engine_dims = engine_->getBindingDimensions(binding_index);
    // Detect whether dynamic or not
    if (ContainsWildcard(engine_dims)) {
      is_dynamic_ = true;
    }

    // Validate whether the binding supports maximum batch size specification in
    // the config
    if ((!engine_->hasImplicitBatchDimension()) &&
        (!ContainsWildcardAtExplicitBatchDim(engine_dims)) &&
        (max_batch_size_ > 1)) {
      return Status(
          Status::Code::INVALID_ARG,
          "unexpected configuration maximum batch size " +
              std::to_string(max_batch_size_) + " for '" + name_ +
              "', model maximum is 1 as model does not contain an implicit "
              "batch dimension nor the explicit batch-dimension of '" +
              input_name + "' is a wildcard.");
    }

    if (!(is_control && is_dynamic_)) {
      RETURN_IF_ERROR(CompareDimsSupported(
          name_, input_name, engine_dims, model_config_dims, support_batching_,
          is_dynamic_, false /* compare_exact */));
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
    std::vector<int64_t> maximum_dims;
    if (!is_dynamic_) {
      byte_size = GetByteSize(max_batch_size_, dt, model_config_dims);
    } else {
      context.max_dims_[io_index] = engine_->getProfileDimensions(
          binding_index, profile_index, nvinfer1::OptProfileSelector::kMAX);
      context.min_dims_[io_index] = engine_->getProfileDimensions(
          binding_index, profile_index, nvinfer1::OptProfileSelector::kMIN);
      context.opt_dims_[io_index] = engine_->getProfileDimensions(
          binding_index, profile_index, nvinfer1::OptProfileSelector::kOPT);

      Status status = ValidateDimension(
          model_config_dims, context.min_dims_[io_index],
          context.max_dims_[io_index], support_batching_);
      if (!status.IsOk()) {
        return Status(
            Status::Code::INTERNAL,
            "model configuration specified invalid shape for input '" +
                input_name + "' for " + name_ +
                ". Error details: " + status.Message());
      }
      RETURN_IF_ERROR(MaximumDims(
          context.max_dims_[io_index], model_config_dims, support_batching_,
          max_batch_size_, &maximum_dims));
      byte_size = GetByteSize(dt, maximum_dims);
      // Update the maximum dimension with respect to the allocated buffer
      DimVecToDims(maximum_dims, &context.max_dims_[io_index]);

      if (!context.context_->setBindingDimensions(
              binding_index, context.max_dims_[io_index])) {
        return Status(
            Status::Code::INTERNAL,
            "trt failed to set binding dimension to " +
                DimsDebugString(context.max_dims_[io_index]) + " for input '" +
                input_name + "' for " + name_);
      }
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

  // Set buffer bindings of all optimization profile since buffer is allocated
  for (auto& trt_context : trt_contexts_) {
    auto binding_index = num_expected_bindings_ * trt_context.first + io_index;
    buffer_bindings_[binding_index] = buffers_[io_index];
  }

  return Status::Success;
}

Status
PlanBackend::Context::InitializeSequenceControlInputBindings(
    const ModelConfig& config)
{
  if (config.has_sequence_batching()) {
    std::vector<ModelSequenceBatching::Control::Kind> boolean_kinds{
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_START,
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_END,
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_READY};

    for (const ModelSequenceBatching::Control::Kind control_kind :
         boolean_kinds) {
      const bool required = false;

      std::string tensor_name;
      DataType tensor_datatype;
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

    std::vector<ModelSequenceBatching::Control::Kind> typdef_kinds{
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_CORRID};

    for (const ModelSequenceBatching::Control::Kind control_kind :
         typdef_kinds) {
      const bool required = false;

      std::string tensor_name;
      DataType tensor_datatype;
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
PlanBackend::Context::InitializeConfigShapeInputBindings(
    const ::google::protobuf::RepeatedPtrField<ModelInput>& ios)
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
    const ::google::protobuf::RepeatedPtrField<ModelInput>& ios)
{
  for (const auto& io : ios) {
    const DimsList& model_config_dims =
        (io.has_reshape()) ? io.reshape().shape() : io.dims();
    RETURN_IF_ERROR(InitializeExecuteInputBinding(
        io.name(), io.data_type(), model_config_dims));
  }

  return Status::Success;
}

Status
PlanBackend::Context::InitializeConfigShapeOutputBindings(
    const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios)
{
  for (const auto& io : ios) {
    // the maximum byte sizes across all profiles
    int64_t max_byte_size = 0;

    // Skip if this output is not a shape tensor
    if (!io.is_shape_tensor()) {
      continue;
    }
    is_dynamic_ = true;

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

      if (io.data_type() != TYPE_INT32) {
        return Status(
            Status::Code::INVALID_ARG,
            "unexpected datatype " + DataType_Name(io.data_type()) +
                "  in model configuration for shape output '" + io.name() +
                "', expecting " + DataType_Name(TYPE_INT32) + " for " + name_);
      }

      DataType dt =
          ConvertTrtTypeToDataType(engine_->getBindingDataType(binding_index));
      if (dt != io.data_type()) {
        return Status(
            Status::Code::INVALID_ARG,
            "unexpected datatype " + DataType_Name(dt) +
                " for inference output '" + io.name() + "', expecting " +
                DataType_Name(io.data_type()) + " for " + name_);
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

      const DimsList& model_config_dims =
          (io.has_reshape()) ? io.reshape().shape() : io.dims();

      nvinfer1::Dims engine_dims = engine_->getBindingDimensions(binding_index);

      RETURN_IF_ERROR(CompareShapeDimsSupported(
          name_, io.name(), engine_dims, model_config_dims, support_batching_));


      const nvinfer1::Dims output_dim =
          context.context_->getBindingDimensions(binding_index);
      std::vector<int64_t> dim_vec;
      DimsToDimVec(output_dim, &dim_vec);
      int64_t byte_size = GetByteSize(dt, dim_vec);

      max_byte_size = std::max(max_byte_size, byte_size);
    }

    if (max_byte_size != 0) {
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
    const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios)
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

      DataType dt =
          ConvertTrtTypeToDataType(engine_->getBindingDataType(binding_index));
      if (dt != io.data_type()) {
        return Status(
            Status::Code::INVALID_ARG,
            "unexpected datatype " + DataType_Name(dt) +
                " for inference output '" + io.name() + "', expecting " +
                DataType_Name(io.data_type()) + " for " + name_);
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

      const DimsList& model_config_dims =
          (io.has_reshape()) ? io.reshape().shape() : io.dims();

      nvinfer1::Dims engine_dims = engine_->getBindingDimensions(binding_index);

      // Validate whether the binding supports maximum batch size specification
      // in the config
      if ((!engine_->hasImplicitBatchDimension()) &&
          (!ContainsWildcardAtExplicitBatchDim(engine_dims)) &&
          (max_batch_size_ > 1)) {
        return Status(
            Status::Code::INVALID_ARG,
            "unexpected configuration maximum batch size " +
                std::to_string(max_batch_size_) + " for '" + name_ +
                "', model maximum is 1 as model does not contain an implicit "
                "batch dimension nor the explicit batch-dimension of '" +
                io.name() + "' is a wildcard.");
      }

      RETURN_IF_ERROR(CompareDimsSupported(
          name_, io.name(), engine_dims, model_config_dims, support_batching_,
          is_dynamic_, false /* compare_exact */));

      int64_t byte_size;
      if (!is_dynamic_) {
        byte_size = GetByteSize(max_batch_size_, dt, model_config_dims);
      } else {
        const nvinfer1::Dims output_dim =
            context.context_->getBindingDimensions(binding_index);
        std::vector<int64_t> dim_vec;
        DimsToDimVec(output_dim, &dim_vec);
        byte_size = GetByteSize(dt, dim_vec);
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
PlanBackend::Context::PeekShapeTensor(
    const InferenceRequest::Input& input, const Scheduler::Payload& payload,
    std::vector<int64_t>* shape)
{
  // It is the caller's responsibility to only call on shape tensors,
  // which means the datatype must be INT32.
  int64_t element_cnt = GetElementCount(input.Shape());
  size_t expected_byte_size =
      element_cnt * GetDataTypeByteSize(DataType::TYPE_INT32);

  const char* content;
  size_t content_byte_size = expected_byte_size;

  // Get the tensor contents into contiguous CPU memory...
  std::unique_ptr<AllocatedMemory> contiguous_buffer;
  bool cuda_copy = false;
  RETURN_IF_ERROR(GetContiguousInputContent(
      input.Name(), TRITONSERVER_MEMORY_CPU, 0 /* src_memory_type_id */,
      payload, &content, &content_byte_size, &contiguous_buffer, &cuda_copy));
  if (expected_byte_size != content_byte_size) {
    return Status(
        Status::Code::INTERNAL,
        "unexpected content size of shape tensor peek. Got " +
            std::to_string(content_byte_size) + " expecting " +
            std::to_string(expected_byte_size));
  }

  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }

  shape->clear();

  const int32_t* dims = reinterpret_cast<const int32_t*>(content);
  for (int64_t i = 0; i < element_cnt; ++i) {
    shape->push_back(dims[i]);
  }

  // Peeking is expensive so use input override to record the value.
  auto overrides = payload.request_->MutableOverrideInputs();
  auto pr = overrides->find(input.Name());
  if (pr == overrides->end()) {
    std::shared_ptr<InferenceRequest::Input> override;
    RETURN_IF_ERROR(payload.request_->AddOverrideInput(
        input.Name(), DataType::TYPE_INT32, input.Shape(), content_byte_size,
        &override));

    // If a buffer was allocated to hold the shape then want to take
    // ownership of that for the override. Otherwise the override can
    // just point to the existing data for the input which is already
    // contiguous.
    if ((contiguous_buffer != nullptr) &&
        (contiguous_buffer->TotalByteSize() > 0)) {
      std::shared_ptr<AllocatedMemory> buf(contiguous_buffer.release());
      RETURN_IF_ERROR(override->SetData(buf));
    } else {
      RETURN_IF_ERROR(override->AppendData(
          content, content_byte_size, TRITONSERVER_MEMORY_CPU, 0));
    }
  }

  return Status::Success;
}

// CUDA 10.1 starts to support CUDA graphs.
#ifdef TRTIS_ENABLE_CUDA_GRAPH
bool
PlanBackend::Context::BuildCudaGraph(
    TensorRTContext* trt_context, const int batch_size)
{
  bool captured = true;

  cudaGraph_t graph;
  auto cuerr = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
  if (cuerr != cudaSuccess) {
    LOG_ERROR << "unable to start CUDA graph for " << name_ << ": "
              << cudaGetErrorString(cuerr);
    captured = false;
  } else {
    auto context = trt_context->context_;
    if (!context->enqueue(
            batch_size, buffer_bindings_.data(), stream_, nullptr)) {
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
        trt_context->cuda_graphs_.insert(std::make_pair(batch_size, graph));
        trt_context->cuda_graph_execs_.insert(
            std::make_pair(batch_size, graph_exec));
      }
    }
  }

  if (captured) {
    LOG_VERBOSE(1) << "captured CUDA graph for " << name_ << ", batch size "
                   << batch_size;
  }

  return captured;
}
#endif

PlanBackend::~PlanBackend()
{
  // Must destory all TensorRT contexts before engine
  contexts_.clear();

  for (auto& device_engine : device_engines_) {
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
    uint32_t runner_idx, std::vector<Scheduler::Payload>* payloads,
    std::function<void(Status)> OnCompleteQueuedPayloads)
{
  // Each runner executes using the corresponding context...
  if (runner_idx >= available_context_queue_.size()) {
    OnCompleteQueuedPayloads(Status(
        Status::Code::INTERNAL,
        "unexpected runner index" + std::to_string(runner_idx) +
            ", max allowed " +
            std::to_string(available_context_queue_.size())));
    return;
  }

#ifdef TRTIS_ENABLE_STATS
  // Stop queue timer and start compute timer when the payload is
  // scheduled to run
  for (auto& payload : *payloads) {
    if (payload.stats_ != nullptr) {
      payload.stats_->CaptureTimestamp(
          ModelInferStats::TimestampKind::kComputeStart);
      payload.stats_->SetGPUDevice(
          contexts_[next_context_[runner_idx]]->gpu_device_);
    }
  }
#endif  // TRTIS_ENABLE_STATS

  auto status = contexts_[next_context_[runner_idx]]->Run(this, payloads);

  // On error, handle the response here instead of delegating to the completion
  // thread as the completion thread will wait on CUDA events unconditionally,
  // which can be ignored on error.
  if (!status.IsOk()) {
#ifdef TRTIS_ENABLE_STATS
    // Stop compute timers.
    for (auto& payload : *payloads) {
      if (payload.stats_ != nullptr) {
        payload.stats_->CaptureTimestamp(
            ModelInferStats::TimestampKind::kComputeEnd);
      }
    }
#endif  // TRTIS_ENABLE_STATS

    OnCompleteQueuedPayloads(status);
    // On inference error, place the context back to the queue immediately
    // as all works for the context should be ignored.
    available_context_queue_[runner_idx]->Put(next_context_[runner_idx]);
  } else {
    auto context =
        static_cast<Context*>(contexts_[next_context_[runner_idx]].get());
    auto event_set_idx = context->next_set_;
    context->next_set_ = (event_set_idx + 1) % context->EVENT_SET_COUNT;
    auto outputs = std::make_shared<std::vector<OutputInfo>>();
    outputs->swap(context->outputs_);
    context->completion_queue_.Put(std::make_tuple(
        OnCompleteQueuedPayloads, payloads, event_set_idx, std::move(outputs)));
  }

  // Set the next context to be executed on this runner, will block
  // until there is available context for the runner
  next_context_[runner_idx] = available_context_queue_[runner_idx]->Get();
}

void
PlanBackend::WarmUp(
    uint32_t runner_idx, const WarmupData& sample,
    std::function<void(Status)> OnCompleteWarmup)
{
  // Each runner executes using the corresponding context...
  if (runner_idx >= available_context_queue_.size()) {
    OnCompleteWarmup(Status(
        Status::Code::INTERNAL,
        "unexpected runner index" + std::to_string(runner_idx) +
            ", max allowed " +
            std::to_string(available_context_queue_.size())));
    return;
  }

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

  std::vector<std::promise<Status>> completion_promises(contexts.size());
  Status status;
  for (size_t idx = 0; idx < contexts.size(); idx++) {
    // Prepare payloads. A set of payloads is required for each context
    auto payloads = std::make_shared<std::vector<Scheduler::Payload>>();

    // Add the sample request directly to the payloads. For the case of
    // batch-size 1 no other request is needed.
    payloads->emplace_back(nullptr, sample.request_, nullptr, nullptr);

    // For batch-size > 1 make copies of the request to fill out the
    // payloads
    for (size_t idx = 1; idx < sample.batch_size_; idx++) {
      auto request = std::make_shared<InferenceRequest>(*sample.request_);
      payloads->emplace_back(nullptr, request, nullptr, nullptr);
    }

    // Run context
    if (status.IsOk()) {
      status = contexts_[contexts[idx]]->Run(this, payloads.get());
    }

    // If one of the contexts can't run properly, the whole warmup should abort
    if (!status.IsOk()) {
      // Clean up the rest of the contexts back to queue,
      // the contexts before will be handled by completion function
      for (auto rest_idx = idx; idx < contexts.size(); idx++) {
        available_context_queue_[runner_idx]->Put(contexts[rest_idx]);
        completion_promises[rest_idx].set_value(status);
      }
      break;
    }

    // Place in completion queue
    auto context = static_cast<Context*>(contexts_[contexts[idx]].get());
    auto event_set_idx = context->next_set_;
    context->next_set_ = (event_set_idx + 1) % context->EVENT_SET_COUNT;
    auto outputs = std::make_shared<std::vector<OutputInfo>>();
    outputs->swap(context->outputs_);
    auto& completion_promise = completion_promises[idx];

    context->completion_queue_.Put(std::make_tuple(
        [payloads, &completion_promise](Status status) {
          completion_promise.set_value(status);
        },
        payloads.get(), event_set_idx, std::move(outputs)));
  }

  // Wait for all inflight executions to be finished.
  for (auto& completion_promise : completion_promises) {
    auto completion_status = completion_promise.get_future().get();
    if (!completion_status.IsOk()) {
      status = completion_status;
    }
  }

  // Need to reset the next context to be executed on this runner
  // as all contexts are in the queue at this point
  next_context_[runner_idx] = available_context_queue_[runner_idx]->Get();

  OnCompleteWarmup(status);
}

Status
PlanBackend::Context::Run(
    const InferenceBackend* base, std::vector<Scheduler::Payload>* payloads)
{
  Status status;

  LOG_VERBOSE(1) << "Running " << name_ << " with " << payloads->size()
                 << " request payloads";
  NVTX_RANGE(nvtx_, "Run " + name_);

  // keep indirect buffers from previous run until now as scheduler
  // thread doesn't check when 'input_ready' event is triggered.
  inputs_.clear();
  outputs_.clear();

  cudaSetDevice(gpu_device_);

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

  std::map<int32_t, std::vector<int32_t>> request_shape_values;
  // Scheduler ensures all the payloads have identical shape values so use
  // values from any shape tensor
  GetRequestShapeValues(
      total_batch_size, payloads->front(), &request_shape_values);

  // total_batch_size can be 1 for models that don't support batching
  // (i.e. max_batch_size_ == 0).
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size_)) {
    return Status(
        Status::Code::INTERNAL,
        "dynamic batch size " + std::to_string(total_batch_size) + " for '" +
            name_ + "', max allowed is " + std::to_string(max_batch_size_));
  }

  auto citr = GetMostOptimizedProfile(
      total_batch_size, *repr_input_request, request_shape_values);

  int binding_offset = citr->first * num_expected_bindings_;

  // For each input, concatenate input values from each payload into
  // the corresponding binding.
  for (int bindex = 0; bindex < num_expected_bindings_; ++bindex) {
    int io_index = binding_offset + bindex;
    if (!engine_->bindingIsInput(io_index)) {
      continue;
    }

    const std::string& name = engine_->getBindingName(bindex);

    // Set the shape binding if needed
    if (engine_->isShapeBinding(io_index)) {
      auto it = request_shape_values.find(io_index);
      if (it != request_shape_values.end()) {
        status = ValidateShapeValues(
            it->second, citr->second.min_shapes_[io_index],
            citr->second.max_shapes_[io_index], citr->second.nb_shape_values_,
            support_batching_);
      } else {
        return Status(
            Status::Code::INTERNAL,
            "unable to find shape values for shape input '" + name +
                "' in request for " + name_);
      }
      if (status.IsOk()) {
        citr->second.context_->setInputShapeBinding(io_index, &(it->second[0]));
      } else {
        return Status(
            Status::Code::INTERNAL,
            "request specifies invalid shape values for shape input '" + name +
                "' for " + name_ + ". Error details: " + status.Message());
      }
    }

    // Skip the upcoming section if not an execution tensor
    if (!engine_->isExecutionBinding(io_index)) {
      continue;
    }

    // Get the shape of the input. The request has already checked
    // that the request shape is valid so don't need to do it here.
    size_t batch1_byte_size;

    std::vector<int64_t> shape;
    if (is_dynamic_) {
      const InferenceRequest::Input* input;
      RETURN_IF_ERROR(repr_input_request->ImmutableInput(name, &input));
      shape = input->Shape();

      DataType dt =
          ConvertTrtTypeToDataType(engine_->getBindingDataType(io_index));

      batch1_byte_size = GetByteSize(dt, shape);
      if (max_batch_size_ != 0) {
        // The first element of the vector will be the batch size and should not
        // be included in the batch1_byte_size computation above.
        shape.insert(shape.begin(), total_batch_size);
      }
    } else {
      batch1_byte_size = byte_sizes_[bindex] / std::max(1, max_batch_size_);
    }

    // Set the binding dimension so that output dimensions can be obtained
    if (is_dynamic_ && (!engine_->isShapeBinding(io_index))) {
      nvinfer1::Dims this_dim;
      if (!DimVecToDims(shape, &this_dim)) {
        return Status(
            Status::Code::INTERNAL,
            "failed to create dims object for " + DimsListToString(shape) +
                " for input '" + name + "' for " + name_ + ".");
      }
      status = ValidateDimension(
          this_dim, citr->second.min_dims_[bindex],
          citr->second.max_dims_[bindex], false);
      if (!status.IsOk()) {
        return Status(
            Status::Code::INTERNAL,
            "request specifies invalid shape for input '" + name + "' for " +
                name_ + ". Error details: " + status.Message());
      }
      if (!citr->second.context_->setBindingDimensions(io_index, this_dim)) {
        return Status(
            Status::Code::INTERNAL, "trt failed to set binding dimension to " +
                                        DimsDebugString(this_dim) +
                                        " for input '" + name + "' for " +
                                        name_);
      }
    }

    if (!engine_->isShapeBinding(io_index)) {
      // Visit the payloads in order and copy the input tensors to
      // GPU. Skip payloads that had errors since they are not included
      // in the dynamic batch.
      std::vector<size_t> expected_byte_sizes;
      for (auto& payload : *payloads) {
        expected_byte_sizes.push_back(
            payload.request_->BatchSize() * batch1_byte_size);
      }

      inputs_.emplace_back();
      auto& input = inputs_.back();
      input.input_buffer_ = static_cast<char*>(buffers_[bindex]);
      input.memory_type_ = TRITONSERVER_MEMORY_GPU;
      input.memory_type_id_ = gpu_device_;
      SetInputBuffer(
          name, expected_byte_sizes, payloads, input_copy_stream_, &input);
    } else {
      // Set the shape values using the first payload for extracting the status
      // of the copy
      SetShapeInputBuffer(
          name, total_batch_size, batch1_byte_size, support_batching_,
          &(*payloads)[0], TRITONSERVER_MEMORY_GPU, gpu_device_,
          static_cast<char*>(buffers_[bindex]));
    }
  }

  // No synchronization here as we know that the below copies will be using
  // 'input_copy_stream_', thus the copies issued above will be done.
  // i.e. if used 'input_copy_stream_', then the order is preserved. Otherwise,
  // the copy is h2h and it is synchronized anyway.
  for (auto& input : inputs_) {
    for (auto& indirect_buffer : input.indirect_buffers_) {
      bool cuda_used;
      TRITONSERVER_MemoryType buffer_memory_type;
      int64_t buffer_memory_id;
      size_t buffer_byte_size;
      auto buffer =
          std::get<0>(indirect_buffer)
              ->BufferAt(
                  0, &buffer_byte_size, &buffer_memory_type, &buffer_memory_id);
      auto status = CopyBuffer(
          "indirect buffer", buffer_memory_type, buffer_memory_id,
          input.memory_type_, input.memory_type_id_, buffer_byte_size, buffer,
          input.input_buffer_ + std::get<1>(indirect_buffer),
          input_copy_stream_, &cuda_used);
      if (!status.IsOk()) {
        for (const auto& payload_idx : std::get<2>(indirect_buffer)) {
          (*payloads)[payload_idx].status_ = status;
        }
      }
    }
  }

  cudaEventRecord(events_[next_set_].input_ready_, input_copy_stream_);

  // Ensure inputs are ready before execution. Output buffers will always be
  // available at this point as the execution and output copy are on the same
  // stream.
  cudaStreamWaitEvent(stream_, events_[next_set_].input_ready_, 0);

  // Async execute the inference using a CUDA graph if available for
  // the batch-size, otherwise execution normally.
  auto itr = citr->second.cuda_graph_execs_.find(total_batch_size);
  if (itr != citr->second.cuda_graph_execs_.end()) {
    cudaError_t err = cudaGraphLaunch(itr->second, stream_);
    if (err != cudaSuccess) {
      cudaStreamSynchronize(stream_);
      return Status(
          Status::Code::INTERNAL, "unable to execute graph for inference " +
                                      name_ + ": " + cudaGetErrorString(err));
    }
    // CUDA graph doesn't know when input is consumed, need to record
    // the event at the end
    // [TODO] can we include event record when capturing the graph?
    cudaEventRecord(events_[next_set_].ready_for_input_, stream_);
  } else {
    LOG_VERBOSE(1) << "Context with profile " << citr->second.profile_name_
                   << " [" << std::to_string(citr->first)
                   << "] is being executed for " << name_;
    if (is_dynamic_) {
      if (!citr->second.context_->allInputDimensionsSpecified()) {
        return Status(
            Status::Code::INTERNAL,
            "failed to specify the dimensions of all input bindings");
      }
      if (!citr->second.context_->allInputShapesSpecified()) {
        return Status(
            Status::Code::INTERNAL,
            "failed to specify the values for all input shape tensors");
      }
    }
    if (!engine_->hasImplicitBatchDimension()) {
      if (!citr->second.context_->enqueueV2(
              buffer_bindings_.data(), stream_,
              &events_[next_set_].ready_for_input_)) {
        cudaStreamSynchronize(stream_);
        return Status(
            Status::Code::INTERNAL, "unable to enqueue for inference " + name_);
      }
    } else {
      if (!citr->second.context_->enqueue(
              total_batch_size, buffer_bindings_.data(), stream_,
              &events_[next_set_].ready_for_input_)) {
        cudaStreamSynchronize(stream_);
        return Status(
            Status::Code::INTERNAL, "unable to enqueue for inference " + name_);
      }
    }
  }

  cudaEventRecord(events_[next_set_].ready_for_output_, stream_);

  // For each requested output verify that the output can accept the
  // actual model output and then copy that output from the GPU
  bool cuda_copy = false;
  for (int bindex = 0; bindex < num_expected_bindings_; ++bindex) {
    int io_index = binding_offset + bindex;
    if (engine_->bindingIsInput(io_index)) {
      continue;
    }

    outputs_.emplace_back();
    auto& output = outputs_.back();

    const std::string& name = engine_->getBindingName(bindex);

    nvinfer1::Dims dims;
    if (is_dynamic_) {
      dims = citr->second.context_->getBindingDimensions(io_index);
    } else {
      dims = engine_->getBindingDimensions(io_index);
    }

    if (engine_->isShapeBinding(io_index)) {
      // Obtain the shape value
      if (dims.nbDims != 0) {
        int32_t* shape_value_ptr =
            (int32_t*)malloc(dims.d[0] * sizeof(int32_t));
        if (!citr->second.context_->getShapeBinding(
                io_index, shape_value_ptr)) {
          return Status(
              Status::Code::INTERNAL,
              "failed to retrieve the output shape values from binding '" +
                  name + "'");
        }

        // The first shape value must be equal to the total batch_size
        if (support_batching_ &&
            total_batch_size != (uint32_t)*shape_value_ptr) {
          return Status(
              Status::Code::INTERNAL, "unexpected batch shape value " +
                                          std::to_string(*shape_value_ptr) +
                                          " for '" + name +
                                          "', total batch size was " +
                                          std::to_string(total_batch_size));
        }

        std::vector<int64_t> content_shape;
        if (support_batching_) {
          content_shape.push_back(total_batch_size);
          content_shape.push_back(dims.d[0] - 1);
        } else {
          content_shape.push_back(dims.d[0]);
        }

        cuda_copy |= SetOutputShapeTensorBuffer(
            name, shape_value_ptr, content_shape, support_batching_,
            TRITONSERVER_MEMORY_CPU, 0, payloads);

        free(shape_value_ptr);
      }
    } else {
      output.output_buffer_ = static_cast<const char*>(buffers_[bindex]);
      output.memory_type_ = TRITONSERVER_MEMORY_GPU;
      output.memory_type_id_ = gpu_device_;

      if (!is_dynamic_ && support_batching_) {
        output.output_shape_.insert(
            output.output_shape_.begin(), total_batch_size);
      }

      for (int i = 0; i < dims.nbDims; ++i) {
        output.output_shape_.push_back(dims.d[i]);
      }

      DataType dt = ConvertTrtTypeToDataType(
          engine_->getBindingDataType(binding_offset + bindex));

      size_t batch1_byte_size = GetByteSize(dt, output.output_shape_);
      if (support_batching_) {
        batch1_byte_size /= total_batch_size;
      }

      if (byte_sizes_[bindex] < (batch1_byte_size * total_batch_size)) {
        return Status(
            Status::Code::INTERNAL,
            "unexpected size for output '" + name + "', byte-size " +
                std::to_string(byte_sizes_[bindex]) + " is less than " +
                std::to_string(total_batch_size) + " * " +
                std::to_string(batch1_byte_size));
      }

      cuda_copy |=
          SetFixedSizeOutputBuffer(name, batch1_byte_size, &output, payloads);
    }
  }

  cudaEventRecord(events_[next_set_].output_ready_, stream_);

  return Status::Success;
}

void
PlanBackend::Context::ProcessResponse(
    size_t context_idx, std::shared_ptr<SyncQueue<size_t>> context_queue)
{
  while (true) {
    NVTX_RANGE(nvtx_, "ProcessResponse " + context_idx);
    auto OnCompleteMetaData = completion_queue_.Get();
    auto& OnComplete = std::get<0>(OnCompleteMetaData);
    if (OnComplete == nullptr) {
      break;
    }
    auto& event_set = events_[std::get<2>(OnCompleteMetaData)];
    auto& payloads = std::get<1>(OnCompleteMetaData);
#ifdef TRTIS_ENABLE_STATS
    // Only need to wait for input copy for recording stats
    cudaEventSynchronize(event_set.input_ready_);
    for (auto& payload : *payloads) {
      if (payload.stats_ != nullptr) {
        payload.stats_->CaptureTimestamp(
            ModelInferStats::TimestampKind::kComputeInputEnd);
      }
    }
#endif  // TRTIS_ENABLE_STATS

    // The model execution associated with the OnCompletePair
    // has consumed the inputs. Put the context back into the available queue
    // so that it can begin enqueuing new memcpys into the input buffers
    cudaEventSynchronize(event_set.ready_for_input_);
    context_queue->Put(context_idx);
    NVTX_MARKER("plan_input_available");

#ifdef TRTIS_ENABLE_STATS
    cudaEventSynchronize(event_set.ready_for_output_);
    for (auto& payload : *payloads) {
      if (payload.stats_ != nullptr) {
        payload.stats_->CaptureTimestamp(
            ModelInferStats::TimestampKind::kComputeOutputStart);
      }
    }
#endif  // TRTIS_ENABLE_STATS

    cudaEventSynchronize(event_set.output_ready_);
    NVTX_MARKER("plan_output_ready");
    // Issue the last steps here if outputs are placed in indirect buffer
    // Note that the copies are expected to be HtoH if any.
    for (auto& output : *(std::get<3>(OnCompleteMetaData))) {
      NVTX_RANGE(nvtx_, "IndirectOutputBufferCopy");
      for (auto& indirect_buffer : output.indirect_buffers_) {
        bool cuda_used;
        TRITONSERVER_MemoryType src_memory_type;
        int64_t src_memory_type_id;
        // placeholder, copy byte size is determined by dst_byte_size
        size_t src_byte_size;
        auto src = indirect_buffer.first->BufferAt(
            0, &src_byte_size, &src_memory_type, &src_memory_type_id);
        TRITONSERVER_MemoryType dst_memory_type;
        int64_t dst_memory_type_id;
        for (auto& payload_output : indirect_buffer.second) {
          char* dst = payload_output.second->MutableBuffer(
              &dst_memory_type, &dst_memory_type_id);
          auto dst_byte_size = payload_output.second->TotalByteSize();
          (*payloads)[payload_output.first].status_ = CopyBuffer(
              "indirect buffer", src_memory_type, src_memory_type_id,
              dst_memory_type, dst_memory_type_id, dst_byte_size, src, dst,
              stream_, &cuda_used);
          src += dst_byte_size;
          if (cuda_used) {
            (*payloads)[payload_output.first].status_ = Status(
                Status::Code::INTERNAL,
                "unexpected cuda copy from indirect buffer to output buffer");
          }
        }
      }
    }

#ifdef TRTIS_ENABLE_STATS
    // Stop compute timers.
    for (auto& payload : *payloads) {
      if (payload.stats_ != nullptr) {
        payload.stats_->CaptureTimestamp(
            ModelInferStats::TimestampKind::kComputeEnd);
      }
    }
#endif  // TRTIS_ENABLE_STATS

    // Just trigger the callback, Payloads are all-set
    {
      NVTX_RANGE(nvtx_, "OnComplete callback");
      OnComplete(Status::Success);
    }
  }
}

Status
PlanBackend::Context::InitEventSet()
{
  for (size_t idx = 0; idx < EVENT_SET_COUNT; idx++) {
    RETURN_IF_ERROR(CreateCudaEvent(
        "Set " + std::to_string(idx) + " ready for input",
        &events_[idx].ready_for_input_));
    RETURN_IF_ERROR(CreateCudaEvent(
        "Set " + std::to_string(idx) + " input ready",
        &events_[idx].input_ready_));
    RETURN_IF_ERROR(CreateCudaEvent(
        "Set " + std::to_string(idx) + " ready for output",
        &events_[idx].ready_for_output_));
    RETURN_IF_ERROR(CreateCudaEvent(
        "Set " + std::to_string(idx) + " output ready",
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
    size_t total_batch_size, const Scheduler::Payload& payload,
    std::map<int, std::vector<int32_t>>* request_shape_values)
{
  // Visit all the inputs and extract the shape values present in the request
  Status status;
  for (const auto& pr : payload.request_->ImmutableInputs()) {
    const auto input = pr.second;
    int io_index = engine_->getBindingIndex(input->Name().c_str());
    if (engine_->isShapeBinding(io_index)) {
      auto it =
          request_shape_values->emplace(io_index, std::vector<int32_t>()).first;
      if (max_batch_size_ != 0) {
        it->second.push_back((int32_t)total_batch_size);
      }

      // Using Peek to read shape values from the tensor.
      std::vector<int64_t> shape;
      if (!PeekShapeTensor(*input, payload, &shape).IsOk()) {
        return Status(
            Status::Code::INTERNAL,
            "unable to peek shape values for input '" + input->Name() + "'");
      }
      for (auto value : shape) {
        it->second.push_back((int32_t)value);
      }
    }
  }
  return Status::Success;
}


std::map<int, PlanBackend::Context::TensorRTContext>::iterator
PlanBackend::Context::GetMostOptimizedProfile(
    size_t total_batch_size, const InferenceRequest& input_request,
    const std::map<int, std::vector<int32_t>>& request_shape_values)
{
  // Returns the TensorRT context that uses profile with shortest Manhattan
  // distance in terms of input dimensions
  // [TODO] traverse it with more efficient data structure (i.e. K-D tree)
  auto ret_it = trt_contexts_.begin();
  if (trt_contexts_.size() != 1) {
    int64_t shortest_distance = LLONG_MAX;
    for (auto cit = trt_contexts_.begin(); cit != trt_contexts_.end(); cit++) {
      int64_t current_distance = 0;
      for (const auto& pr : input_request.ImmutableInputs()) {
        const auto input = pr.second;
        int io_index = engine_->getBindingIndex(input->Name().c_str());
        nvinfer1::Dims engine_dims = engine_->getBindingDimensions(io_index);
        // If the input has no dynamic shape nor is a shape binding, then skip
        // it as distance will be 0
        if (!(ContainsWildcard(engine_dims) ||
              engine_->isShapeBinding(io_index))) {
          continue;
        }
        auto status = ValidateDimension(
            input->Shape(), cit->second.min_dims_[io_index],
            cit->second.max_dims_[io_index], true);
        bool valid_bs =
            (((int64_t)total_batch_size >=
              cit->second.min_dims_[io_index].d[0]) &&
             ((int64_t)total_batch_size <=
              cit->second.max_dims_[io_index].d[0]));

        bool missing_shape_values = false;
        if (valid_bs && status.IsOk() && engine_->isShapeBinding(io_index)) {
          auto it = request_shape_values.find(io_index);
          if (it != request_shape_values.end()) {
            status = ValidateShapeValues(
                it->second, cit->second.min_shapes_[io_index],
                cit->second.max_shapes_[io_index], cit->second.nb_shape_values_,
                support_batching_);
            valid_bs =
                (((int32_t)total_batch_size >=
                  *cit->second.min_shapes_[io_index]) &&
                 ((int64_t)total_batch_size <=
                  *cit->second.max_shapes_[io_index]));
          } else {
            missing_shape_values = true;
          }
        }


        if (!status.IsOk() || !valid_bs || missing_shape_values) {
          current_distance = LLONG_MAX;
          break;
        } else {
          const auto& opt_dims = cit->second.opt_dims_[io_index];
          current_distance +=
              std::abs(opt_dims.d[0] - (int64_t)total_batch_size);
          for (int idx = 1; idx < opt_dims.nbDims; idx++) {
            current_distance +=
                std::abs(opt_dims.d[idx] - input->Shape()[idx - 1]);
          }
          if (engine_->isShapeBinding(io_index)) {
            const auto* opt_shape_values = cit->second.opt_shapes_[io_index];
            current_distance +=
                std::abs(*opt_shape_values - (int64_t)total_batch_size);
            auto it = request_shape_values.find(io_index);
            for (size_t idx = 1; idx < cit->second.nb_shape_values_; idx++) {
              current_distance +=
                  std::abs(*(opt_shape_values + idx) - it->second[idx - 1]);
            }
          }
        }
      }

      if (current_distance < shortest_distance) {
        ret_it = cit;
        shortest_distance = current_distance;
      }
    }
  }

  LOG_VERBOSE(1) << "Optimization profile " << ret_it->second.profile_name_
                 << " [" << std::to_string(ret_it->first)
                 << "] is selected for " << name_;

  return ret_it;
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
