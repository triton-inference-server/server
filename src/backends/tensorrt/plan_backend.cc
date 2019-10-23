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

#include "src/backends/tensorrt/plan_backend.h"

#include <NvInfer.h>
#include <stdint.h>
#include <mutex>
#include "src/backends/tensorrt/loader.h"
#include "src/backends/tensorrt/plan_utils.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config_cuda.h"
#include "src/core/model_config_utils.h"
#include "src/core/provider.h"
#include "src/core/server_status.h"

namespace nvidia { namespace inferenceserver {

PlanBackend::Context::Context(
    const std::string& name, const int gpu_device, const int max_batch_size,
    const int profile_index)
    : BackendContext(name, gpu_device, max_batch_size), runtime_(nullptr),
      engine_(nullptr), context_(nullptr), is_dynamic_(false),
      profile_index_(profile_index), min_dims_(nullptr), max_dims_(nullptr),
      max_dynamic_batch_size_(INT_MAX), total_bindings_(0),
      num_expected_bindings_(0), byte_sizes_(nullptr), buffers_(nullptr)
{
  stream_ = nullptr;
}

PlanBackend::Context::~Context()
{
  LOG_VERBOSE(1) << "~PlanBackend::Context ";

  if (byte_sizes_ != nullptr) {
    delete[] byte_sizes_;
    byte_sizes_ = nullptr;
  }

  if (min_dims_ != nullptr) {
    delete[] min_dims_;
    min_dims_ = nullptr;
  }

  if (max_dims_ != nullptr) {
    delete[] max_dims_;
    max_dims_ = nullptr;
  }

  if (buffers_ != nullptr) {
    for (int i = 0; i < total_bindings_; ++i) {
      if (buffers_[i] != nullptr) {
        cudaError_t err = cudaFree(buffers_[i]);
        if (err != cudaSuccess) {
          LOG_ERROR << "Failed to free cuda memory for '" << name_
                    << "': " << cudaGetErrorString(err);
        }
      }
    }

    delete[] buffers_;
    buffers_ = nullptr;
  }

  for (const auto& pr : cuda_graph_execs_) {
    cudaError_t err = cudaGraphExecDestroy(pr.second);
    if (err != cudaSuccess) {
      LOG_ERROR << "Failed to destroy cuda graph exec: "
                << cudaGetErrorString(err);
    }
  }
  cuda_graph_execs_.clear();

  for (const auto& pr : cuda_graphs_) {
    cudaError_t err = cudaGraphDestroy(pr.second);
    if (err != cudaSuccess) {
      LOG_ERROR << "Failed to destroy cuda graph exec: "
                << cudaGetErrorString(err);
    }
  }
  cuda_graphs_.clear();

  if (stream_ != nullptr) {
    cudaError_t err = cudaStreamDestroy(stream_);
    if (err != cudaSuccess) {
      LOG_ERROR << "Failed to destroy cuda stream: " << cudaGetErrorString(err);
    }
    stream_ = nullptr;
  }

  if (context_ != nullptr) {
    context_->destroy();
    context_ = nullptr;
  }
  if (engine_ != nullptr) {
    engine_->destroy();
    engine_ = nullptr;
  }
  if (runtime_ != nullptr) {
    runtime_->destroy();
    runtime_ = nullptr;
  }
}

Status
PlanBackend::Init(const std::string& path, const ModelConfig& config)
{
  RETURN_IF_ERROR(ValidateModelConfig(config, kTensorRTPlanPlatform));
  RETURN_IF_ERROR(SetModelConfig(path, config));

  return Status::Success;
}

Status
PlanBackend::CreateExecutionContexts(
    const std::unordered_map<std::string, std::vector<char>>& models)
{
  // TensorRT engine creation is not thread-safe, so multiple creations
  // are serialized with a global lock.
  static std::mutex global_context_mu;
  std::lock_guard<std::mutex> glock(global_context_mu);

  uint32_t total_context_cnt = 0;

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
          RequestStatusCode::INVALID_ARG,
          "instance group " + group.name() + " of model " + Name() +
              " must be KIND_GPU and must specify at least one GPU id");
    }

    for (int c = 0; c < group.count(); c++) {
      for (int gpu_device : group.gpus()) {
        const std::string instance_name = group.name() + "_" +
                                          std::to_string(c) + "_gpu" +
                                          std::to_string(gpu_device);

        const std::string profile_name = group.profile();
        RETURN_IF_ERROR(CreateExecutionContext(
            instance_name, gpu_device, models, profile_name));
        total_context_cnt++;
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

  LOG_VERBOSE(1) << "plan backend for " << Name() << std::endl << *this;

  return Status::Success;
}

void
PlanBackend::Context::InitProfile()
{
  const int total_profiles = engine_->getNbOptimizationProfiles();
  total_bindings_ = engine_->getNbBindings();
  if (total_profiles == 0) {
    num_expected_bindings_ = total_bindings_;
  } else {
    num_expected_bindings_ = total_bindings_ / total_profiles;
  }
  binding_offset_ = profile_index_ * num_expected_bindings_;
}

Status
PlanBackend::CreateExecutionContext(
    const std::string& instance_name, const int gpu_device,
    const std::unordered_map<std::string, std::vector<char>>& models,
    const std::string profile_name)
{
  cudaError_t cuerr;

  // Determine the model file to use for device compute capability
  cudaDeviceProp cuprops;
  cuerr = cudaGetDeviceProperties(&cuprops, gpu_device);
  if (cuerr != cudaSuccess) {
    return Status(
        RequestStatusCode::INTERNAL,
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
        RequestStatusCode::INTERNAL,
        "unable to find PLAN model '" + cc_model_filename + "' for " + Name());
  }

  LOG_INFO << "Creating instance " << instance_name << " on GPU " << gpu_device
           << " (" << cc << ") using " << cc_model_filename;

  // Max batch size. A value of 0 in the config becomes NO_BATCHING.
  const int mbs = (Config().max_batch_size() <= 0) ? Context::NO_BATCHING
                                                   : Config().max_batch_size();

  const int profile_index = GetProfileIndex(profile_name);
  contexts_.emplace_back(
      new Context(instance_name, gpu_device, mbs, profile_index));
  const std::unique_ptr<Context>& context = contexts_.back();

  // Set the device before generating engine and context.
  cuerr = cudaSetDevice(gpu_device);
  if (cuerr != cudaSuccess) {
    return Status(
        RequestStatusCode::INTERNAL, "unable to set device for " + Name() +
                                         ": " + cudaGetErrorString(cuerr));
  }

  RETURN_IF_ERROR(
      LoadPlan(mn_itr->second, &context->runtime_, &context->engine_));

  // Now the TRT execution context
  context->context_ = context->engine_->createExecutionContext();
  if (context->context_ == nullptr) {
    return Status(
        RequestStatusCode::INTERNAL, "unable to create TensorRT context");
  }

  // TRT sets the optimization profile index to be 0 implicitly with the first
  // context creation. As currently TRTIS supports one context per engine,
  // in order to set the specified profile_index, another context is created
  // and the previous context is destroyed.
  if (profile_index != 0) {
    auto first_context = context->context_;
    context->context_ = context->engine_->createExecutionContext();
    if (context->context_ == nullptr) {
      return Status(
          RequestStatusCode::INTERNAL, "unable to create TensorRT context");
    }
    if (!context->context_->setOptimizationProfile(profile_index)) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "Can not set the specified optimization profile " +
              std::to_string(profile_index) + " for " + Name() +
              ". Expected optimization profile index range 0-" +
              std::to_string(
                  context->engine_->getNbOptimizationProfiles() - 1));
    }
    first_context->destroy();
  }

  context->InitProfile();

  // Collect all the expected input and allowed output tensor names
  // and validate that the model configuration specifies only those.
  std::set<std::string> allowed_inputs, allowed_outputs;
  for (int i = 0; i < context->num_expected_bindings_; ++i) {
    if (context->engine_->bindingIsInput(i)) {
      allowed_inputs.emplace(context->engine_->getBindingName(i));
    } else {
      allowed_outputs.emplace(context->engine_->getBindingName(i));
    }
  }

  for (const auto& io : Config().input()) {
    RETURN_IF_ERROR(CheckAllowedModelInput(io, allowed_inputs));
  }

  for (const auto& io : Config().output()) {
    RETURN_IF_ERROR(CheckAllowedModelOutput(io, allowed_outputs));
  }

  RETURN_IF_ERROR(context->ValidateInputs(Config().input()));
  RETURN_IF_ERROR(context->ValidateOutputs(Config().output()));
  // Initialize the inputs and outputs. Make sure the model matches
  // what is in the configuration. Allocate memory for the maximum
  // possible batch size: min(engine maximum, config maximum)
  context->byte_sizes_ = new uint64_t[context->num_expected_bindings_];
  context->min_dims_ = new nvinfer1::Dims[context->num_expected_bindings_];
  context->max_dims_ = new nvinfer1::Dims[context->num_expected_bindings_];
  context->buffers_ = new void*[context->total_bindings_]();  // init to nullptr

  const bool support_batching = (mbs != Context::NO_BATCHING);

  RETURN_IF_ERROR(context->InitializeConfigInputBindings(
      Config().input(), support_batching));
  RETURN_IF_ERROR(context->InitializeSequenceControlInputBindings(
      Config(), support_batching));
  if (!context->context_->allInputDimensionsSpecified()) {
    return Status(
        RequestStatusCode::INTERNAL,
        "failed to specify the dimensions of all input bindings");
  }

  // As we have visited all the input bindings at this point, if any of the
  // shapes had dynamic dimension then is_dynamic_ flag would be set
  if (!context->is_dynamic_ &&
      (context->max_batch_size_ > context->engine_->getMaxBatchSize())) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "unexpected configuration maximum batch size " +
            std::to_string(Config().max_batch_size()) + " for '" + Name() +
            "', model maximum is " +
            std::to_string(context->engine_->getMaxBatchSize()));
  } else if (
      context->is_dynamic_ &&
      (context->max_batch_size_ > context->max_dynamic_batch_size_)) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "unexpected configuration maximum batch size " +
            std::to_string(Config().max_batch_size()) + " for '" + Name() +
            "' profile [" + profile_name + "], model maximum is " +
            std::to_string(context->max_dynamic_batch_size_));
  }
  RETURN_IF_ERROR(context->InitializeConfigOutputBindings(
      Config().output(), support_batching));

  // Make sure every index is initialized.
  for (int i = 0; i < context->num_expected_bindings_; ++i) {
    if (context->buffers_[context->binding_offset_ + i] == nullptr) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "expected configuration for " +
              std::string(
                  (context->engine_->bindingIsInput(i) ? "input" : "output")) +
              " '" + context->engine_->getBindingName(i) + "' for " + Name());
    }
  }

  // Create CUDA stream associated with the execution context
  const int cuda_stream_priority =
      GetCudaStreamPriority(Config().optimization().priority());
  RETURN_IF_ERROR(context->CreateCudaStream(cuda_stream_priority));

  // CUDA 10.1 starts to support CUDA graphs.
  // If enabled, build CUDA graphs for a default set of graph
  // sizes. Graphs are most likely to help for small batch sizes so by
  // default build for batch sizes 1, 2, 3, 4, 6, 8, 12, 16. If any
  // build fails don't attempt for any larger batch sizes.
#ifdef TRTIS_ENABLE_CUDA_GRAPH
  const bool use_cuda_graphs = Config().optimization().cuda().graphs();
  if (use_cuda_graphs) {
    if (context->BuildCudaGraph(1)) {
      for (int bs : std::vector<int>{2, 3, 4, 6, 8, 12, 16}) {
        if (bs <= Config().max_batch_size()) {
          if (!context->BuildCudaGraph(bs)) {
            break;
          }
        }
      }
    }
  }
#endif

  if (context->is_dynamic_) {
    LOG_INFO << "Created instance " << instance_name << " on GPU " << gpu_device
             << " (" << cc << ") with stream priority " << cuda_stream_priority
             << " and optimization profile " << profile_name;
  } else {
    LOG_INFO << "Created instance " << instance_name << " on GPU " << gpu_device
             << " (" << cc << ") with stream priority " << cuda_stream_priority;
  }

  return Status::Success;
}

Status
PlanBackend::Context::ValidateInputs(
    const ::google::protobuf::RepeatedPtrField<ModelInput>& ios)
{
  for (const auto& io : ios) {
    if (!ConvertDataTypeToTrtType(io.data_type()).first) {
      return Status(
          RequestStatusCode::INTERNAL,
          "unsupported datatype " + DataType_Name(io.data_type()) +
              " for input '" + io.name() + "' for model '" + name_ + "'");
    }
  }

  return Status::Success;
}


Status
PlanBackend::Context::ValidateOutputs(
    const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios)
{
  for (const auto& io : ios) {
    if (!ConvertDataTypeToTrtType(io.data_type()).first) {
      return Status(
          RequestStatusCode::INTERNAL,
          "unsupported datatype " + DataType_Name(io.data_type()) +
              " for output '" + io.name() + "' for model '" + name_ + "'");
    }
  }

  return Status::Success;
}

Status
PlanBackend::Context::InitializeInputBinding(
    const std::string& input_name, const DataType input_datatype,
    const DimsList& model_config_dims, const bool support_batching,
    const bool is_control)
{
  int index = binding_offset_ + engine_->getBindingIndex(input_name.c_str());
  if (index < 0) {
    return Status(
        RequestStatusCode::NOT_FOUND,
        "input '" + input_name + "' not found for " + name_);
  }

  if (buffers_[index] != nullptr) {
    return Status(
        RequestStatusCode::INVALID_ARG, "input '" + input_name +
                                            "' has already appeared as an " +
                                            "input or output for " + name_);
  }

  if (!engine_->bindingIsInput(index)) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "input '" + input_name + "' is expected to be an output in model for " +
            name_);
  }

  DataType dt = ConvertTrtTypeToDataType(engine_->getBindingDataType(index));
  if (dt != input_datatype) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "unexpected datatype " + DataType_Name(dt) + " for input '" +
            input_name + "', expecting " + DataType_Name(input_datatype) +
            " for " + name_);
  }

  MemoryFormat fmt = ConvertTrtFmtToFmt(engine_->getBindingFormat(index));
  if (fmt != MemoryFormat::LINEAR) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "unexpected tensor format " + MemoryFormat_Name(fmt) + " for input '" +
            input_name +
            "'. Only LINEAR memory format is supported at present.");
  }

  nvinfer1::Dims engine_dims = engine_->getBindingDimensions(index);
  // Detect whether dynamic or not
  if (ContainsWildcard(engine_dims)) {
    is_dynamic_ = true;
  }

  if (!(is_control && is_dynamic_)) {
    RETURN_IF_ERROR(CompareDimsSupported(
        name_, input_name, engine_dims, model_config_dims, support_batching,
        is_dynamic_));
  } else {
    Status status = ValidateControlDimsDynamic(engine_dims, support_batching);
    if (!status.IsOk()) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "unexpected dimensions " + DimsDebugString(engine_dims) +
              " for control input '" + input_name + "' for " + name_ +
              ". Error details: " + status.Message());
    }
  }

  int64_t byte_size;
  std::vector<int64_t> maximum_dims;
  if (!is_dynamic_) {
    byte_size = GetByteSize(max_batch_size_, dt, model_config_dims);
  } else {
    nvinfer1::Dims max_profile_dims = engine_->getProfileDimensions(
        index, profile_index_, nvinfer1::OptProfileSelector::kMAX);
    min_dims_[index - binding_offset_] = engine_->getProfileDimensions(
        index, profile_index_, nvinfer1::OptProfileSelector::kMIN);
    Status status = ValidateDimension(
        model_config_dims, min_dims_[index - binding_offset_], max_profile_dims,
        support_batching);
    if (!status.IsOk()) {
      return Status(
          RequestStatusCode::INTERNAL,
          "model config specifies invalid shape for input '" + input_name +
              "' for " + name_ + ". Error details: " + status.Message());
    }
    RETURN_IF_ERROR(MaximumDims(
        max_profile_dims, model_config_dims, &maximum_dims, support_batching));
    if (support_batching) {
      if (max_dynamic_batch_size_ > maximum_dims[0]) {
        max_dynamic_batch_size_ = maximum_dims[0];
      }
    } else {
      max_dynamic_batch_size_ = 1;
    }
    byte_size = GetByteSize(max_batch_size_, dt, maximum_dims);
    // Update the maximum dimension with respect to the allocated buffer
    DimVecToDims(maximum_dims, &max_dims_[index - binding_offset_]);
  }
  if (byte_size == -1) {
    return Status(
        RequestStatusCode::INTERNAL,
        "unable to calculate size for input '" + input_name + " for " + name_);
  }

  // Allocate CUDA memory. We rely on buffers_ being non-nullptr to
  // indicate that the buffer has been correctly initalized so even
  // for zero-sized tensors always allocate something.
  void* buffer;
  cudaError_t err = cudaMalloc(&buffer, std::max((int64_t)1, byte_size));
  if (err != cudaSuccess) {
    return Status(
        RequestStatusCode::INTERNAL, "unable to allocate memory for input '" +
                                         input_name + " for " + name_ + ": " +
                                         cudaGetErrorString(err));
  }

  byte_sizes_[index - binding_offset_] = byte_size;
  buffers_[index] = buffer;

  if (is_dynamic_) {
    nvinfer1::Dims input_dim;
    if (!DimVecToDims(maximum_dims, &input_dim)) {
      return Status(
          RequestStatusCode::INTERNAL,
          "failed to create dims object for " + DimsListToString(maximum_dims) +
              " for input '" + input_name + "' for " + name_ + ".");
    }
    if (!context_->setBindingDimensions(index, input_dim)) {
      return Status(
          RequestStatusCode::INTERNAL,
          "trt failed to set binding dimension to " +
              DimsDebugString(input_dim) + " for input '" + input_name +
              "' for " + name_);
    }
  }

  return Status::Success;
}

Status
PlanBackend::Context::InitializeSequenceControlInputBindings(
    const ModelConfig& config, const bool support_batching)
{
  if (config.has_sequence_batching()) {
    std::vector<ModelSequenceBatching::Control::Kind> kinds{
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_START,
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_READY};

    for (const ModelSequenceBatching::Control::Kind control_kind : kinds) {
      std::string tensor_name;
      DataType tensor_datatype;
      RETURN_IF_ERROR(GetSequenceControlProperties(
          config.sequence_batching(), config.name(), control_kind,
          true /* required */, &tensor_name, &tensor_datatype, nullptr, nullptr,
          nullptr, nullptr));

      // Control tensors must have shape [1].
      DimsList dims;
      dims.Add(1);

      RETURN_IF_ERROR(InitializeInputBinding(
          tensor_name, tensor_datatype, dims, support_batching, true));
    }
  }

  return Status::Success;
}

Status
PlanBackend::Context::InitializeConfigInputBindings(
    const ::google::protobuf::RepeatedPtrField<ModelInput>& ios,
    const bool support_batching)
{
  for (const auto& io : ios) {
    const DimsList& model_config_dims =
        (io.has_reshape()) ? io.reshape().shape() : io.dims();
    RETURN_IF_ERROR(InitializeInputBinding(
        io.name(), io.data_type(), model_config_dims, support_batching));
  }

  return Status::Success;
}

Status
PlanBackend::Context::InitializeConfigOutputBindings(
    const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios,
    const bool support_batching)
{
  for (const auto& io : ios) {
    int index = binding_offset_ + engine_->getBindingIndex(io.name().c_str());
    if (index < 0) {
      return Status(
          RequestStatusCode::NOT_FOUND,
          "output '" + io.name() + "' not found for " + name_);
    }

    if (buffers_[index] != nullptr) {
      return Status(
          RequestStatusCode::INVALID_ARG, "output '" + io.name() +
                                              "' has already appeared as an " +
                                              "input or output for " + name_);
    }

    if (engine_->bindingIsInput(index)) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "output '" + io.name() +
              "' is expected to be an input in model for " + name_);
    }

    DataType dt = ConvertTrtTypeToDataType(engine_->getBindingDataType(index));
    if (dt != io.data_type()) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "unexpected datatype " + DataType_Name(dt) +
              " for inference output '" + io.name() + "', expecting " +
              DataType_Name(io.data_type()) + " for " + name_);
    }

    MemoryFormat fmt = ConvertTrtFmtToFmt(engine_->getBindingFormat(index));
    if (fmt != MemoryFormat::LINEAR) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "unexpected tensor format " + MemoryFormat_Name(fmt) +
              " for output '" + io.name() +
              "'. Only LINEAR memory format is supported at present.");
    }

    const DimsList& model_config_dims =
        (io.has_reshape()) ? io.reshape().shape() : io.dims();

    nvinfer1::Dims engine_dims = engine_->getBindingDimensions(index);
    RETURN_IF_ERROR(CompareDimsSupported(
        name_, io.name(), engine_dims, model_config_dims, support_batching,
        is_dynamic_));

    int64_t byte_size;
    if (!is_dynamic_) {
      byte_size = GetByteSize(max_batch_size_, dt, model_config_dims);
      if (byte_size == -1) {
        return Status(
            RequestStatusCode::INTERNAL,
            "unable to calculate size for output '" + io.name() + " for " +
                name_);
      }
    } else {
      const nvinfer1::Dims output_dim = context_->getBindingDimensions(index);
      std::vector<int64_t> dim_vec;
      DimsToDimVec(output_dim, &dim_vec);
      byte_size = GetByteSize(max_batch_size_, dt, dim_vec);
    }

    // Allocate CUDA memory. We rely on buffers_ being non-nullptr to
    // indicate that the buffer has been correctly initalized so even
    // for zero-sized tensors always allocate something.
    void* buffer;
    cudaError_t err = cudaMalloc(&buffer, std::max((int64_t)1, byte_size));
    if (err != cudaSuccess) {
      return Status(
          RequestStatusCode::INTERNAL,
          "unable to allocate memory for input '" + io.name() + " for " +
              name_ + ": " + std::string(cudaGetErrorString(err)));
    }

    byte_sizes_[index - binding_offset_] = byte_size;
    buffers_[index] = buffer;
  }

  return Status::Success;
}

void
PlanBackend::Run(
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

  // Stop queue timer and start compute timer when the payload is
  // scheduled to run
  for (auto& payload : *payloads) {
    if (payload.stats_ != nullptr) {
      payload.stats_->CaptureTimestamp(
          ModelInferStats::TimestampKind::kComputeStart);
      payload.stats_->SetGPUDevice(contexts_[runner_idx]->gpu_device_);
    }
  }

  Status status = contexts_[runner_idx]->Run(payloads);

  // Stop compute timers.
  for (auto& payload : *payloads) {
    if (payload.stats_ != nullptr) {
      payload.stats_->CaptureTimestamp(
          ModelInferStats::TimestampKind::kComputeEnd);
    }
  }

  OnCompleteQueuedPayloads(status);
}

// CUDA 10.1 starts to support CUDA graphs.
#ifdef TRTIS_ENABLE_CUDA_GRAPH
bool
PlanBackend::Context::BuildCudaGraph(const int batch_size)
{
  bool captured = true;
  cudaError_t cuerr;

  cudaGraph_t graph;
  cuerr = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
  if (cuerr != cudaSuccess) {
    LOG_ERROR << "unable to start CUDA graph for " << name_ << ": "
              << cudaGetErrorString(cuerr);
    captured = false;
  } else {
    if (!context_->enqueue(batch_size, buffers_, stream_, nullptr)) {
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
        cuda_graphs_.insert(std::make_pair(batch_size, graph));
        cuda_graph_execs_.insert(std::make_pair(batch_size, graph_exec));
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

Status
PlanBackend::Context::Run(std::vector<Scheduler::Payload>* payloads)
{
  LOG_VERBOSE(1) << "Running " << name_ << " with " << payloads->size()
                 << " request payloads";

  cudaSetDevice(gpu_device_);

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

  // For each input, concatenate input values from each payload into
  // the corresponding binding.
  for (int bindex = 0; bindex < num_expected_bindings_; ++bindex) {
    if (!engine_->bindingIsInput(binding_offset_ + bindex)) {
      continue;
    }

    const std::string& name = engine_->getBindingName(bindex);

    // Get the shape of the input. The provider has already checked
    // that the request shape is valid so don't need to do it here.
    size_t batch1_byte_size;

    std::vector<int64_t> shape;
    if (is_dynamic_) {
      for (const auto& input :
           input_request_provider->RequestHeader().input()) {
        const std::string& this_name = input.name();
        if (this_name.compare(name) == 0) {
          for (auto dim : input.dims()) {
            shape.push_back(dim);
          }
          break;
        }
      }

      DataType dt = ConvertTrtTypeToDataType(
          engine_->getBindingDataType(binding_offset_ + bindex));

      batch1_byte_size = GetByteSize(dt, shape);
      if (max_batch_size_ != 0) {
        // The first element of the vector will be the batch size and should not
        // be included in the batch1_byte_size computation above.
        shape.insert(shape.begin(), total_batch_size);
      }
    } else {
      batch1_byte_size = byte_sizes_[bindex] / std::max(1, max_batch_size_);
    }

    // Visit the payloads in order and copy the input tensors to
    // GPU. Skip payloads that had errors since they are not included
    // in the dynamic batch.
    std::vector<size_t> expected_byte_sizes;
    for (auto& payload : *payloads) {
      const InferRequestHeader& request_header =
          payload.request_provider_->RequestHeader();
      expected_byte_sizes.push_back(
          request_header.batch_size() * batch1_byte_size);
    }

    SetInputBuffer(
        name, expected_byte_sizes, payloads, TRTSERVER_MEMORY_GPU, gpu_device_,
        static_cast<char*>(buffers_[binding_offset_ + bindex]));

    // Set the binding dimension so that output dimensions can be obtained
    if (is_dynamic_) {
      nvinfer1::Dims this_dim;
      if (!DimVecToDims(shape, &this_dim)) {
        return Status(
            RequestStatusCode::INTERNAL,
            "failed to create dims object for " + DimsListToString(shape) +
                " for input '" + name + "' for " + name_ + ".");
      }
      Status status = ValidateDimension(
          this_dim, min_dims_[bindex], max_dims_[bindex], false);
      if (!status.IsOk()) {
        return Status(
            RequestStatusCode::INTERNAL,
            "request specifies invalid shape for input '" + name + "' for " +
                name_ + ". Error details: " + status.Message());
      }
      if (!context_->setBindingDimensions(binding_offset_ + bindex, this_dim)) {
        return Status(
            RequestStatusCode::INTERNAL,
            "trt failed to set binding dimension to " +
                DimsDebugString(this_dim) + " for input '" + name + "' for " +
                name_);
      }
    }
  }

  for (auto& payload : *payloads) {
    if (payload.stats_ != nullptr) {
      payload.stats_->CaptureTimestamp(
          ModelInferStats::TimestampKind::kComputeInputEnd);
    }
  }

  // Async execute the inference using a CUDA graph if available for
  // the batch-size, otherwise execution normally.
  auto itr = cuda_graph_execs_.find(total_batch_size);
  if (itr != cuda_graph_execs_.end()) {
    cudaError_t err = cudaGraphLaunch(itr->second, stream_);
    if (err != cudaSuccess) {
      cudaStreamSynchronize(stream_);
      return Status(
          RequestStatusCode::INTERNAL,
          "unable to execute graph for inference " + name_ + ": " +
              cudaGetErrorString(err));
    }
  } else {
    if (is_dynamic_) {
      if (!context_->allInputDimensionsSpecified()) {
        return Status(
            RequestStatusCode::INTERNAL,
            "failed to specify the dimensions of all input bindings");
      }
      if (!context_->enqueueV2(buffers_, stream_, nullptr)) {
        cudaStreamSynchronize(stream_);
        return Status(
            RequestStatusCode::INTERNAL,
            "unable to enqueue for inference " + name_);
      }
    } else {
      if (!context_->enqueue(total_batch_size, buffers_, stream_, nullptr)) {
        cudaStreamSynchronize(stream_);
        return Status(
            RequestStatusCode::INTERNAL,
            "unable to enqueue for inference " + name_);
      }
    }
  }

  for (auto& payload : *payloads) {
    if (payload.stats_ != nullptr) {
      payload.stats_->CaptureTimestamp(
          ModelInferStats::TimestampKind::kComputeOutputStart);
    }
  }

  // For each requested output verify that the output can accept the
  // actual model output and then copy that output from the GPU
  bool cuda_copy = false;
  for (int bindex = 0; bindex < num_expected_bindings_; ++bindex) {
    if (engine_->bindingIsInput(binding_offset_ + bindex)) {
      continue;
    }

    const std::string& name = engine_->getBindingName(bindex);

    nvinfer1::Dims dims;
    if (is_dynamic_) {
      dims = context_->getBindingDimensions(binding_offset_ + bindex);
    } else {
      dims = engine_->getBindingDimensions(binding_offset_ + bindex);
    }

    std::vector<int64_t> shape;

    if (!is_dynamic_ && max_batch_size_ != NO_BATCHING) {
      shape.insert(shape.begin(), total_batch_size);
    }

    for (int i = 0; i < dims.nbDims; ++i) {
      shape.push_back(dims.d[i]);
    }

    DataType dt = ConvertTrtTypeToDataType(
        engine_->getBindingDataType(binding_offset_ + bindex));

    size_t batch1_byte_size = GetByteSize(dt, shape);
    if (max_batch_size_ != NO_BATCHING) {
      batch1_byte_size /= total_batch_size;
    }

    if (byte_sizes_[bindex] < (batch1_byte_size * total_batch_size)) {
      return Status(
          RequestStatusCode::INTERNAL,
          "unexpected size for output '" + name + "', byte-size " +
              std::to_string(byte_sizes_[bindex]) + " is less than " +
              std::to_string(total_batch_size) + " * " +
              std::to_string(batch1_byte_size));
    }

    cuda_copy |= SetFixedSizeOutputBuffer(
        name, batch1_byte_size,
        static_cast<char*>(buffers_[binding_offset_ + bindex]), shape,
        TRTSERVER_MEMORY_GPU /* src_memory_type */, gpu_device_, payloads);
  }

  // Wait for the copy-out to complete
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
  return Status::Success;
}

std::ostream&
operator<<(std::ostream& out, const PlanBackend& pb)
{
  out << "name=" << pb.Name() << std::endl;
  out << "contexts:" << std::endl;
  for (const auto& context : pb.contexts_) {
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
      out << "    " << i << ": byte_size=" << context->byte_sizes_[i]
          << ", buffer=" << context->buffers_[context->binding_offset_ + i]
          << " ]" << std::endl;
    }
  }

  return out;
}

}}  // namespace nvidia::inferenceserver
