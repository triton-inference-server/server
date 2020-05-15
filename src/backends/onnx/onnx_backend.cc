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

#include "src/backends/onnx/onnx_backend.h"

#include <stdint.h>
#include <mutex>
#include "src/backends/onnx/loader.h"
#include "src/backends/onnx/onnx_utils.h"
#include "src/core/constants.h"
#include "src/core/cuda_utils.h"
#include "src/core/logging.h"
#include "src/core/metrics.h"
#include "src/core/model_config_cuda.h"
#include "src/core/model_config_utils.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_provider_factory.h>
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

#ifdef TRITON_ENABLE_ONNXRUNTIME_TENSORRT
#include <tensorrt_provider_factory.h>
#endif  // TRITON_ENABLE_ONNXRUNTIME_TENSORRT

#ifdef TRITON_ENABLE_ONNXRUNTIME_OPENVINO
#include <openvino_provider_factory.h>
#endif  // TRITON_ENABLE_ONNXRUNTIME_OPENVINO

namespace nvidia { namespace inferenceserver {

OnnxBackend::Context::Context(
    const std::string& name, const int gpu_device, const int max_batch_size,
    const bool enable_pinned_input, const bool enable_pinned_output,
    std::unique_ptr<MetricModelReporter>&& metric_reporter)
    : BackendContext(
          name, gpu_device, max_batch_size, enable_pinned_input,
          enable_pinned_output, std::move(metric_reporter)),
      session_(nullptr), allocator_(nullptr)
{
}

OnnxBackend::Context::~Context()
{
  LOG_VERBOSE(1) << "~OnnxBackend::Context ";

  ReleaseOrtRunResources();
  if (session_ != nullptr) {
    OnnxLoader::UnloadSession(session_);
  }
  // 'allocator_' is default allocator which is managed by ONNX Runtime
}

Status
OnnxBackend::CreateExecutionContexts(
    const std::unordered_map<std::string, std::pair<bool, std::string>>& models)
{
  // Create a "prototype" session option, which will be cloned and set
  // context-specific option on context creation.
  OrtSessionOptions* session_options;
  RETURN_IF_ORT_ERROR(ort_api->CreateSessionOptions(&session_options));

  OrtResourceWrapper<OrtSessionOptions*> options_wrapper(
      session_options, ort_api->ReleaseSessionOptions);
  RETURN_IF_ORT_ERROR(ort_api->SetIntraOpNumThreads(session_options, 1));

  // set graph optimization level
  GraphOptimizationLevel optimization_level =
      GraphOptimizationLevel::ORT_ENABLE_ALL;
  if (Config().optimization().has_graph()) {
    int graph_level = Config().optimization().graph().level();
    if (graph_level == -1) {
      optimization_level = GraphOptimizationLevel::ORT_ENABLE_BASIC;
    } else if (graph_level == 1) {
      optimization_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
    }
  }
  RETURN_IF_ORT_ERROR(ort_api->SetSessionGraphOptimizationLevel(
      session_options, optimization_level));

  RETURN_IF_ERROR(CreateExecutionContextsHelper(session_options, models));

  LOG_VERBOSE(1) << "onnx backend for " << Name() << std::endl << *this;

  return Status::Success;
}

Status
OnnxBackend::CreateExecutionContextsHelper(
    OrtSessionOptions* session_options,
    const std::unordered_map<std::string, std::pair<bool, std::string>>& models)
{
  uint32_t total_context_cnt = 0;

  // Create a session for each instance.
  for (const auto& group : Config().instance_group()) {
    for (int c = 0; c < group.count(); c++) {
      if (group.kind() == ModelInstanceGroup::KIND_CPU) {
        const std::string instance_name =
            group.name() + "_" + std::to_string(c) + "_cpu";
        RETURN_IF_ERROR(CreateExecutionContext(
            instance_name, Context::NO_GPU_DEVICE, session_options, models));
        total_context_cnt++;
      } else {
        for (int gpu_device : group.gpus()) {
          const std::string instance_name = group.name() + "_" +
                                            std::to_string(c) + "_gpu" +
                                            std::to_string(gpu_device);
          RETURN_IF_ERROR(CreateExecutionContext(
              instance_name, gpu_device, session_options, models));
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
          uint32_t runner_idx,
          std::vector<std::unique_ptr<InferenceRequest>>&& requests) {
        Run(runner_idx, std::move(requests));
      }));

  return Status::Success;
}

Status
OnnxBackend::CreateExecutionContext(
    const std::string& instance_name, const int gpu_device,
    OrtSessionOptions* base_session_options,
    const std::unordered_map<std::string, std::pair<bool, std::string>>& models)
{
  // For a GPU context, determine the model file to use for device
  // compute capability. CPU always uses the default model file.
  std::string cc;
  std::string cc_model_filename;
  if (gpu_device == Context::NO_GPU_DEVICE) {
    cc_model_filename = Config().default_model_filename();
  } else {
#ifdef TRITON_ENABLE_GPU
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
#endif  // TRITON_ENABLE_GPU
  }

  const auto& op_itr = models.find(cc_model_filename);
  if (op_itr == models.end()) {
    return Status(
        Status::Code::INTERNAL,
        "unable to find model '" + cc_model_filename + "' for " + Name());
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

  RETURN_IF_ERROR(context->CreateCudaStream());

  // Set Onnx session option with proper device
  OrtSessionOptions* session_options;
  RETURN_IF_ORT_ERROR(
      ort_api->CloneSessionOptions(base_session_options, &session_options));

  OrtResourceWrapper<OrtSessionOptions*> options_wrapper(
      session_options, ort_api->ReleaseSessionOptions);

  // Set execution execution_accelerators (execution providers in ONNX Runtime)
  if (gpu_device != Context::NO_GPU_DEVICE) {
#ifdef TRITON_ENABLE_GPU
    if (Config().optimization().has_execution_accelerators()) {
      // Don't need to ensure uniqueness of the providers,
      // ONNX Runtime will check it.
      for (const auto& execution_accelerator :
           Config()
               .optimization()
               .execution_accelerators()
               .gpu_execution_accelerator()) {
#ifdef TRITON_ENABLE_ONNXRUNTIME_TENSORRT
        if (execution_accelerator.name() == kTensorRTExecutionAccelerator) {
          RETURN_IF_ORT_ERROR(OrtSessionOptionsAppendExecutionProvider_Tensorrt(
              session_options, gpu_device));
          LOG_VERBOSE(1) << "TensorRT Execution Accelerator is set for "
                         << instance_name << " on device " << gpu_device;
        } else
#endif  // TRITON_ENABLE_ONNXRUNTIME_TENSORRT
        {
          return Status(
              Status::Code::INVALID_ARG, "unknown Execution Accelerator '" +
                                             execution_accelerator.name() +
                                             "' is requested");
        }
      }
    }
    RETURN_IF_ORT_ERROR(OrtSessionOptionsAppendExecutionProvider_CUDA(
        session_options, gpu_device));
    LOG_VERBOSE(1) << "CUDA Execution Accelerator is set for " << instance_name
                   << " on device " << gpu_device;
#else
    return Status(Status::Code::INTERNAL, "GPU instances not supported");
#endif  // TRITON_ENABLE_GPU
  }

  bool need_lock = false;
  if (Config().optimization().has_execution_accelerators()) {
    for (const auto& execution_accelerator : Config()
                                                 .optimization()
                                                 .execution_accelerators()
                                                 .cpu_execution_accelerator()) {
      if (execution_accelerator.name() == kOpenVINOExecutionAccelerator) {
#ifdef TRITON_ENABLE_ONNXRUNTIME_OPENVINO
        need_lock = true;
        RETURN_IF_ORT_ERROR(OrtSessionOptionsAppendExecutionProvider_OpenVINO(
            session_options, "CPU"));
        LOG_VERBOSE(1) << "OpenVINO Execution Accelerator is set for "
                       << instance_name << " on device CPU";
#else
        return Status(
            Status::Code::INVALID_ARG,
            "OpenVINO Execution Accelerator is not enabled");
#endif  // TRITON_ENABLE_ONNXRUNTIME_OPENVINO
      } else {
        return Status(
            Status::Code::INVALID_ARG, "unknown Execution Accelerator '" +
                                           execution_accelerator.name() +
                                           "' is requested");
      }
    }
  }

  // ONNX session creation with OpenVINO is not thread-safe,
  // so multiple creations are serialized with a global lock.
  static std::mutex global_context_mu;
  std::unique_lock<std::mutex> glock(global_context_mu, std::defer_lock);
  if (need_lock) {
    glock.lock();
  }

  RETURN_IF_ERROR(OnnxLoader::LoadSession(
      op_itr->second, session_options, &context->session_));
  RETURN_IF_ORT_ERROR(
      ort_api->GetAllocatorWithDefaultOptions(&context->allocator_));

  size_t expected_input_cnt = (size_t)Config().input().size();

  // If this is a sequence model then make sure that the required
  // inputs are present in the model and have the correct shape and
  // datatype.
  if (Config().has_sequence_batching()) {
    bool have_start, have_end, have_ready, have_corrid;
    RETURN_IF_ERROR(context->ValidateBooleanSequenceControl(
        Config().name(), Config().sequence_batching(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_START,
        false /* required */, &have_start));
    RETURN_IF_ERROR(context->ValidateBooleanSequenceControl(
        Config().name(), Config().sequence_batching(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_END,
        false /* required */, &have_end));
    RETURN_IF_ERROR(context->ValidateBooleanSequenceControl(
        Config().name(), Config().sequence_batching(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_READY,
        false /* required */, &have_ready));
    RETURN_IF_ERROR(context->ValidateTypedSequenceControl(
        Config().name(), Config().sequence_batching(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_CORRID,
        false /* required */, &have_corrid));
    if (have_start) {
      expected_input_cnt += 1;
    }
    if (have_end) {
      expected_input_cnt += 1;
    }
    if (have_ready) {
      expected_input_cnt += 1;
    }
    if (have_corrid) {
      expected_input_cnt += 1;
    }
  }

  RETURN_IF_ERROR(context->ValidateInputs(
      Config().name(), Config().input(), expected_input_cnt));
  RETURN_IF_ERROR(context->ValidateOutputs(Config().name(), Config().output()));

  return Status::Success;
}

Status
OnnxBackend::Context::ValidateBooleanSequenceControl(
    const std::string& model_name, const ModelSequenceBatching& batcher,
    const ModelSequenceBatching::Control::Kind control_kind, bool required,
    bool* have_control)
{
  std::string tensor_name;
  DataType tensor_datatype;
  RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
      batcher, model_name, control_kind, required, &tensor_name,
      &tensor_datatype, nullptr, nullptr, nullptr, nullptr));
  *have_control = !tensor_name.empty();
  if (*have_control) {
    OnnxTensorInfoMap input_tensor_infos;
    RETURN_IF_ERROR(InputInfos(session_, allocator_, input_tensor_infos));
    const auto& iit = input_tensor_infos.find(tensor_name);
    if (iit == input_tensor_infos.end()) {
      return Status(
          Status::Code::INTERNAL,
          "configuration specified sequence control '" + tensor_name +
              "', but model does not provide that input");
    }

    // Control tensors must have shape [1].
    const int nonbatch_start_idx = (max_batch_size_ > 0) ? 1 : 0;
    std::vector<int64_t> debatched_dims;
    for (size_t i = nonbatch_start_idx; i < iit->second.dims_.size(); i++) {
      debatched_dims.push_back(iit->second.dims_[i]);
    }

    if ((debatched_dims.size() != 1) || (debatched_dims[0] != 1)) {
      return Status(
          Status::Code::INVALID_ARG,
          "unable to load model '" + model_name + "', sequence control '" +
              tensor_name + "' in model has dims " +
              DimsListToString(debatched_dims) + " but dims [1] is expected");
    }

    if (ConvertToOnnxDataType(tensor_datatype) != iit->second.type_) {
      return Status(
          Status::Code::INVALID_ARG,
          "unable to load model '" + model_name + "', sequence control '" +
              tensor_name + "', the model expects data-type " +
              OnnxDataTypeName(iit->second.type_) +
              " but the model configuration specifies data-type " +
              DataType_Name(tensor_datatype));
    }
  }

  return Status::Success;
}

Status
OnnxBackend::Context::ValidateTypedSequenceControl(
    const std::string& model_name, const ModelSequenceBatching& batcher,
    const ModelSequenceBatching::Control::Kind control_kind, bool required,
    bool* have_control)
{
  std::string tensor_name;
  DataType tensor_datatype;
  RETURN_IF_ERROR(GetTypedSequenceControlProperties(
      batcher, model_name, control_kind, required, &tensor_name,
      &tensor_datatype));
  *have_control = !tensor_name.empty();
  if (*have_control) {
    OnnxTensorInfoMap input_tensor_infos;
    RETURN_IF_ERROR(InputInfos(session_, allocator_, input_tensor_infos));
    const auto& iit = input_tensor_infos.find(tensor_name);
    if (iit == input_tensor_infos.end()) {
      return Status(
          Status::Code::INTERNAL,
          "configuration specified sequence control '" + tensor_name +
              "', but model does not provide that input");
    }

    // Control tensors must have shape [1].
    const int nonbatch_start_idx = (max_batch_size_ > 0) ? 1 : 0;
    std::vector<int64_t> debatched_dims;
    for (size_t i = nonbatch_start_idx; i < iit->second.dims_.size(); i++) {
      debatched_dims.push_back(iit->second.dims_[i]);
    }

    if ((debatched_dims.size() != 1) || (debatched_dims[0] != 1)) {
      return Status(
          Status::Code::INVALID_ARG,
          "unable to load model '" + model_name + "', sequence control '" +
              tensor_name + "' in model has dims " +
              DimsListToString(debatched_dims) + " but dims [1] is expected");
    }

    if (ConvertToOnnxDataType(tensor_datatype) != iit->second.type_) {
      return Status(
          Status::Code::INVALID_ARG,
          "unable to load model '" + model_name + "', sequence control '" +
              tensor_name + "', the model expects data-type " +
              OnnxDataTypeName(iit->second.type_) +
              " but the model configuration specifies data-type " +
              DataType_Name(tensor_datatype));
    }
  }

  return Status::Success;
}

Status
OnnxBackend::Context::ValidateInputs(
    const std::string& model_name,
    const ::google::protobuf::RepeatedPtrField<ModelInput>& ios,
    const size_t expected_input_cnt)
{
  std::set<std::string> input_tensor_names;
  RETURN_IF_ERROR(InputNames(session_, input_tensor_names));

  OnnxTensorInfoMap input_tensor_infos;
  RETURN_IF_ERROR(InputInfos(session_, allocator_, input_tensor_infos));

  if (input_tensor_infos.size() != expected_input_cnt) {
    return Status(
        Status::Code::INVALID_ARG,
        "unable to load model '" + model_name + "', configuration expects " +
            std::to_string(expected_input_cnt) + " inputs, model provides " +
            std::to_string(input_tensor_infos.size()));
  }

  for (const auto& io : ios) {
    auto iit = input_tensor_infos.find(io.name());
    if (iit == input_tensor_infos.end()) {
      RETURN_IF_ERROR(CheckAllowedModelInput(io, input_tensor_names));
    }

    auto onnx_data_type = ConvertToOnnxDataType(io.data_type());
    if (onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      return Status(
          Status::Code::INTERNAL,
          "unsupported datatype " + DataType_Name(io.data_type()) +
              " for input '" + io.name() + "' for model '" + model_name + "'");
    } else if (onnx_data_type != iit->second.type_) {
      return Status(
          Status::Code::INVALID_ARG,
          "unable to load model '" + model_name + ", unexpected datatype " +
              DataType_Name(ConvertFromOnnxDataType(iit->second.type_)) +
              " for input '" + io.name() + "', expecting " +
              DataType_Name(io.data_type()));
    }

    // If a reshape is provided for the input then use that when
    // validating that the model matches what is expected.
    const DimsList& dims =
        (io.has_reshape()) ? io.reshape().shape() : io.dims();
    RETURN_IF_ERROR(CompareDimsSupported(
        model_name, io.name(), iit->second.dims_, dims, max_batch_size_,
        false /* compare_exact */));
  }

  return Status::Success;
}

Status
OnnxBackend::Context::ValidateOutputs(
    const std::string& model_name,
    const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios)
{
  std::set<std::string> output_tensor_names;
  RETURN_IF_ERROR(OutputNames(session_, output_tensor_names));

  OnnxTensorInfoMap output_tensor_infos;
  RETURN_IF_ERROR(OutputInfos(session_, allocator_, output_tensor_infos));

  for (const auto& io : ios) {
    auto iit = output_tensor_infos.find(io.name());
    if (iit == output_tensor_infos.end()) {
      RETURN_IF_ERROR(CheckAllowedModelOutput(io, output_tensor_names));
    }

    auto onnx_data_type = ConvertToOnnxDataType(io.data_type());
    if (onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      return Status(
          Status::Code::INTERNAL,
          "unsupported datatype " + DataType_Name(io.data_type()) +
              " for output '" + io.name() + "' for model '" + model_name + "'");
    } else if (onnx_data_type != iit->second.type_) {
      return Status(
          Status::Code::INVALID_ARG,
          "unable to load model '" + model_name + ", unexpected datatype " +
              DataType_Name(ConvertFromOnnxDataType(iit->second.type_)) +
              " for output '" + io.name() + "', expecting " +
              DataType_Name(io.data_type()));
    }

    // If a reshape is provided for the input then use that when
    // validating that the model matches what is expected.
    const DimsList& dims =
        (io.has_reshape()) ? io.reshape().shape() : io.dims();
    RETURN_IF_ERROR(CompareDimsSupported(
        model_name, io.name(), iit->second.dims_, dims, max_batch_size_,
        true /* compare_exact */));
  }

  return Status::Success;
}

void
OnnxBackend::Context::Run(
    InferenceBackend* base,
    std::vector<std::unique_ptr<InferenceRequest>>&& requests)
{
  LOG_VERBOSE(1) << "Running " << name_ << " with " << requests.size()
                 << " request requests";

  INFER_STATS_DECL_TIMESTAMP(compute_start_ns);

  // For each request in 'requests' collect the total batch size for
  // this inference execution. The batch-size, number of inputs, and
  // size of each input has already been checked by each requests
  // request provider so don't need to do that here.
  size_t total_batch_size = 0;
  for (auto& request : requests) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (request == nullptr) {
      InferenceRequest::RespondIfError(
          requests,
          Status(
              Status::Code::INTERNAL,
              "null request given to TensorFlow runner for '" + name_ + "'"),
          true /* release_requests */);
      return;
    }

    total_batch_size += std::max(1U, request->BatchSize());
  }

  // If there are no valid payloads then no need to run the
  // inference. The payloads will have their error status set so can
  // just return.
  if (total_batch_size == 0) {
    return;
  }

  // total_batch_size can be 1 for models that don't support batching
  // (i.e. max_batch_size_ == 0).
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size_)) {
    InferenceRequest::RespondIfError(
        requests,
        Status(
            Status::Code::INTERNAL,
            "dynamic batch size " + std::to_string(total_batch_size) +
                " for '" + name_ + "', max allowed is " +
                std::to_string(max_batch_size_)),
        true /* release_requests */);
    return;
  }

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
  std::vector<std::unique_ptr<InferenceResponse>> responses;
  responses.reserve(requests.size());

  for (auto& request : requests) {
    std::unique_ptr<InferenceResponse> response;
    Status status = request->ResponseFactory().CreateResponse(&response);
    if (!status.IsOk()) {
      InferenceRequest::RespondIfError(request, status);
      response.reset();
    }

    responses.emplace_back(std::move(response));
  }

  // use Scoped wrapper to clean up Ort tensors when Run() returns
  static auto io_tensor_deleter = [](Context* ctx) {
    if (ctx != nullptr) {
      ctx->ReleaseOrtRunResources();
    }
  };
  OrtResourceWrapper<Context*> io_tensor_wrapper(this, io_tensor_deleter);

  // Hold reference to each buffer of input data so that it stays
  // until the inference has completed.
  std::vector<std::unique_ptr<AllocatedMemory>> input_buffers;
  std::vector<const char*> input_names;
  bool cuda_copy = false;
  FAIL_ALL_AND_RETURN_IF_ERROR(
      requests, responses, metric_reporter_.get(),
      SetInputTensors(
          total_batch_size, requests, &responses, &input_buffers, &input_names,
          &cuda_copy),
      "error sending ONNX response");

  // Request to retrieve all output specified in model config
  // and reserve placeholder for output tensors
  std::vector<const char*> output_names;
  for (const auto& output : base->Config().output()) {
    output_names.emplace_back(output.name().c_str());
    output_tensors_.emplace_back(nullptr);
  }

  // Wait for any in-flight input tensor copies to complete.
#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif

  INFER_STATS_DECL_TIMESTAMP(compute_input_end_ns);

  // Run...
  FAIL_ALL_AND_RETURN_IF_ERROR(
      requests, responses, metric_reporter_.get(),
      OrtRun(input_names, output_names), "error sending ONNX response");

  INFER_STATS_DECL_TIMESTAMP(compute_output_start_ns);

  FAIL_ALL_AND_RETURN_IF_ERROR(
      requests, responses, metric_reporter_.get(),
      ReadOutputTensors(total_batch_size, output_names, requests, &responses),
      "error sending ONNX response");

#ifdef TRITON_ENABLE_STATS
  INFER_STATS_DECL_TIMESTAMP(compute_end_ns);

  // Report stats and trace
  for (size_t i = 0; i < requests.size(); ++i) {
    auto& request = requests[i];
    request->ReportStatistics(
        metric_reporter_.get(), (responses[i] != nullptr), compute_start_ns,
        compute_input_end_ns, compute_output_start_ns, compute_end_ns);

#ifdef TRITON_ENABLE_TRACING
    if (request->Trace() != nullptr) {
      auto& trace = request->Trace();
      trace->Report(TRITONSERVER_TRACE_COMPUTE_START, compute_start_ns);
      trace->Report(TRITONSERVER_TRACE_COMPUTE_INPUT_END, compute_input_end_ns);
      trace->Report(
          TRITONSERVER_TRACE_COMPUTE_OUTPUT_START, compute_output_start_ns);
      trace->Report(TRITONSERVER_TRACE_COMPUTE_END, compute_end_ns);
    }
#endif  // TRITON_ENABLE_TRACING
  }

  // Also reporting batch stats
  base->MutableStatsAggregator()->UpdateInferBatchStats(
      metric_reporter_.get(), total_batch_size, compute_start_ns,
      compute_input_end_ns, compute_output_start_ns, compute_end_ns);
#endif  // TRITON_ENABLE_STATS

  // Send all the responses that haven't already been sent because of
  // an earlier error.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_STATUS_ERROR(
          InferenceResponse::Send(std::move(response)),
          "failed to send TensorFlow backend response");
    }
  }

  // Release all requests.
  for (auto& request : requests) {
    InferenceRequest::Release(std::move(request));
  }
}

Status
OnnxBackend::Context::SetInputTensors(
    size_t total_batch_size,
    const std::vector<std::unique_ptr<InferenceRequest>>& requests,
    std::vector<std::unique_ptr<InferenceResponse>>* responses,
    std::vector<std::unique_ptr<AllocatedMemory>>* input_buffers,
    std::vector<const char*>* input_names, bool* cuda_copy)
{
  BackendInputCollector collector(
      requests, responses, enable_pinned_input_, stream_);
  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  for (const auto& pr : requests[0]->ImmutableInputs()) {
    const auto& name = pr.first;
    const auto& repr_input = pr.second;
    const auto& batch1_shape = repr_input->Shape();

    input_names->emplace_back(name.c_str());
    input_tensors_.emplace_back(nullptr);

    // The shape for the entire input patch, [total_batch_size, ...]
    std::vector<int64_t> batchn_shape;
    batchn_shape.reserve(batch1_shape.size() + 1);
    if (max_batch_size_ != NO_BATCHING) {
      batchn_shape.push_back(total_batch_size);
    }
    batchn_shape.insert(
        batchn_shape.end(), batch1_shape.begin(), batch1_shape.end());

    const DataType datatype = repr_input->DType();

    // [TODO] currently ONNX Runtime only recognize input data on CPU
    // https://github.com/microsoft/onnxruntime/issues/1621
    if (datatype != TYPE_STRING) {
      input_buffers->emplace_back(new AllocatedMemory(
          GetByteSize(datatype, batchn_shape), TRITONSERVER_MEMORY_CPU_PINNED,
          0));
      TRITONSERVER_MemoryType mem_type;
      auto input_buffer = input_buffers->back()->MutableBuffer(&mem_type);
      auto total_byte_size = input_buffers->back()->TotalByteSize();

      // Create ORT Tensor
      const OrtMemoryInfo* allocator_info;
      RETURN_IF_ORT_ERROR(
          ort_api->AllocatorGetInfo(allocator_, &allocator_info));
      RETURN_IF_ORT_ERROR(ort_api->CreateTensorWithDataAsOrtValue(
          allocator_info, (void*)input_buffer, total_byte_size,
          batchn_shape.data(), batchn_shape.size(),
          ConvertToOnnxDataType(datatype), &input_tensors_.back()));

      collector.ProcessTensor(
          name, datatype, batch1_shape, input_buffer, total_byte_size, mem_type,
          0);
    } else {
      // For String input, we need to obtain tensor info differently
      size_t batch1_element_cnt = GetElementCount(batch1_shape);
      size_t total_byte_size = 0;
      std::vector<size_t> expected_byte_sizes;
      std::vector<size_t> expected_element_cnts;
      expected_byte_sizes.reserve(requests.size());
      expected_element_cnts.reserve(requests.size());
      for (size_t ridx = 0; ridx < requests.size(); ++ridx) {
        expected_element_cnts.push_back(
            std::max(1U, requests[ridx]->BatchSize()) * batch1_element_cnt);

        const InferenceRequest::Input* in;
        auto status = requests[ridx]->ImmutableInput(name, &in);
        // Skip input in this request if failed to retrieve it
        if (!status.IsOk()) {
          if ((*responses)[ridx] != nullptr) {
            InferenceResponse::SendWithStatus(
                std::move((*responses)[ridx]), status);
          }
          expected_byte_sizes.push_back(0);
        } else {
          expected_byte_sizes.push_back(in->Data()->TotalByteSize());
        }
        total_byte_size += expected_byte_sizes.back();
      }
      // For string input, the copy to contiguous buffer is needed because ORT
      // expects elements to be C strings thus we need to modify input buffer.
      // Reserve one more byte at the end of input_buffer to ensure last
      // element of String data can become valid C string.
      input_buffers->emplace_back(new AllocatedMemory(
          total_byte_size + 1, TRITONSERVER_MEMORY_CPU_PINNED, 0));
      TRITONSERVER_MemoryType mem_type;
      auto input_buffer = input_buffers->back()->MutableBuffer(&mem_type);
      size_t buffer_offset = 0;
      bool string_cuda_copy = false;
      for (size_t ridx = 0; ridx < requests.size(); ++ridx) {
        const InferenceRequest::Input* in;
        auto status = requests[ridx]->ImmutableInput(name, &in);
        if (status.IsOk() && ((*responses)[ridx] != nullptr)) {
          const void* src_buffer;
          size_t src_byte_size;
          TRITONSERVER_MemoryType src_memory_type;
          int64_t src_memory_type_id;
          size_t input_offset = 0;
          for (size_t idx = 0; idx < in->DataBufferCount(); ++idx) {
            status = in->DataBuffer(
                idx, &src_buffer, &src_byte_size, &src_memory_type,
                &src_memory_type_id);
            if (status.IsOk()) {
              if (input_offset + src_byte_size > expected_byte_sizes[ridx]) {
                status = Status(
                    Status::Code::INVALID_ARG,
                    "buffer size for input '" + name +
                        "' exceeds batch byte size " +
                        std::to_string(expected_byte_sizes[ridx]));
              } else {
                bool cuda_used = false;
                status = CopyBuffer(
                    name, src_memory_type, src_memory_type_id, mem_type, 0,
                    src_byte_size, src_buffer,
                    input_buffer + buffer_offset + input_offset, stream_,
                    &cuda_used);
                *cuda_copy |= cuda_used;
              }
            }
            if (status.IsOk()) {
              input_offset += src_byte_size;
            } else {
              break;
            }
          }
        }
        if (!status.IsOk() && ((*responses)[ridx] != nullptr)) {
          InferenceResponse::SendWithStatus(
              std::move((*responses)[ridx]), status);
        }
        buffer_offset += expected_byte_sizes[ridx];
      }

#ifdef TRITON_ENABLE_GPU
      // Synchronize to ensure the buffer is ready to be modified
      if (string_cuda_copy) {
        cudaStreamSynchronize(stream_);
      }
#endif  // TRITON_ENABLE_GPU

      std::vector<const char*> string_data;
      // Modify input buffer and set string expected by ORT
      SetStringInputBuffer(
          name, expected_byte_sizes, expected_element_cnts, responses,
          input_buffer, &string_data);
      input_buffer[total_byte_size] = 0;

      RETURN_IF_ORT_ERROR(ort_api->CreateTensorAsOrtValue(
          allocator_, batchn_shape.data(), batchn_shape.size(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, &input_tensors_.back()));
      RETURN_IF_ORT_ERROR(ort_api->FillStringTensor(
          input_tensors_.back(), string_data.data(), string_data.size()));
    }
  }
  // Finalize...
  *cuda_copy |= collector.Finalize();
  return Status::Success;
}

Status
OnnxBackend::Context::OrtRun(
    const std::vector<const char*>& input_names,
    const std::vector<const char*>& output_names)
{
  RETURN_IF_ORT_ERROR(ort_api->Run(
      session_, NULL /* run options */, input_names.data(),
      (const OrtValue* const*)input_tensors_.data(), input_tensors_.size(),
      output_names.data(), output_names.size(), output_tensors_.data()));
  return Status::Success;
}

void
OnnxBackend::Context::SetStringInputBuffer(
    const std::string& name, const std::vector<size_t>& expected_byte_sizes,
    const std::vector<size_t>& expected_element_cnts,
    std::vector<std::unique_ptr<InferenceResponse>>* responses,
    char* input_buffer, std::vector<const char*>* string_data)
{
  // offset for each response
  size_t buffer_copy_offset = 0;
  for (size_t idx = 0; idx < expected_byte_sizes.size(); idx++) {
    const size_t expected_byte_size = expected_byte_sizes[idx];
    const size_t expected_element_cnt = expected_element_cnts[idx];

    size_t element_cnt = 0;
    if ((*responses)[idx] != nullptr) {
      size_t remaining_bytes = expected_byte_size;
      char* data_content = input_buffer + buffer_copy_offset;
      // Continue if the remaining bytes may still contain size info
      while (remaining_bytes >= sizeof(uint32_t)) {
        if (element_cnt >= expected_element_cnt) {
          InferenceResponse::SendWithStatus(
              std::move((*responses)[idx]),
              Status(
                  Status::Code::INVALID_ARG,
                  "unexpected number of string elements " +
                      std::to_string(element_cnt + 1) +
                      " for inference input '" + name + "', expecting " +
                      std::to_string(expected_element_cnt)));
          break;
        }

        const uint32_t len = *(reinterpret_cast<const uint32_t*>(data_content));
        remaining_bytes -= sizeof(uint32_t);
        // Make first byte of size info 0, so that if there is string data
        // in front of it, the data becomes valid C string.
        *data_content = 0;
        data_content = data_content + sizeof(uint32_t);
        if (len > remaining_bytes) {
          InferenceResponse::SendWithStatus(
              std::move((*responses)[idx]),
              Status(
                  Status::Code::INVALID_ARG,
                  "incomplete string data for inference input '" + name +
                      "', expecting string of length " + std::to_string(len) +
                      " but only " + std::to_string(remaining_bytes) +
                      " bytes available"));
          break;
        } else {
          string_data->push_back(data_content);
          element_cnt++;
          data_content = data_content + len;
          remaining_bytes -= len;
        }
      }
    }

    FillStringData(string_data, expected_element_cnt - element_cnt);

    buffer_copy_offset += expected_byte_size;
  }
}

void
OnnxBackend::Context::FillStringData(
    std::vector<const char*>* string_data, size_t cnt)
{
  static const char* empty = "";
  for (size_t c = 0; c < cnt; c++) {
    string_data->push_back(empty);
  }
}

Status
OnnxBackend::Context::ReadOutputTensors(
    size_t total_batch_size, const std::vector<const char*>& output_names,
    const std::vector<std::unique_ptr<InferenceRequest>>& requests,
    std::vector<std::unique_ptr<InferenceResponse>>* responses)
{
  BackendResponder responder(
      requests, responses, max_batch_size_, enable_pinned_output_, stream_);

  // Use to hold string output contents
  bool cuda_copy = false;
  std::vector<std::vector<char>> string_buffers;
  for (size_t idx = 0; idx < output_names.size(); idx++) {
    std::string name = std::string(output_names[idx]);

    OrtValue* output_tensor = output_tensors_[idx];
    if (output_tensor == nullptr) {
      return Status(
          Status::Code::INTERNAL,
          "output tensor '" + name + "' does not found");
    }

    // Get output type and shape
    OrtTypeInfo* typeinfo;
    RETURN_IF_ORT_ERROR(ort_api->GetTypeInfo(output_tensor, &typeinfo));
    OrtResourceWrapper<OrtTypeInfo*> typeinfo_wrapper(
        typeinfo, ort_api->ReleaseTypeInfo);

    const OrtTensorTypeAndShapeInfo* type_and_shape;
    RETURN_IF_ORT_ERROR(
        ort_api->CastTypeInfoToTensorInfo(typeinfo, &type_and_shape));

    size_t num_dims;
    RETURN_IF_ORT_ERROR(ort_api->GetDimensionsCount(type_and_shape, &num_dims));

    std::vector<int64_t> batchn_shape(num_dims);
    RETURN_IF_ORT_ERROR(ort_api->GetDimensions(
        type_and_shape, batchn_shape.data(), batchn_shape.size()));
    const size_t element_count = GetElementCount(batchn_shape);

    ONNXTensorElementDataType type;
    RETURN_IF_ORT_ERROR(ort_api->GetTensorElementType(type_and_shape, &type));

    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
      const size_t batch1_element_cnt = element_count / total_batch_size;
      size_t total_length = 0;
      RETURN_IF_ORT_ERROR(
          ort_api->GetStringTensorDataLength(output_tensor, &total_length));

      string_buffers.emplace_back(std::vector<char>(total_length));
      auto content = string_buffers.back().data();
      size_t offsets[element_count + 1];
      RETURN_IF_ORT_ERROR(ort_api->GetStringTensorContent(
          output_tensor, content, total_length, offsets, element_count));
      // Mark "passed end byte offset"
      offsets[element_count] = total_length;

      cuda_copy |= SetStringOutputBuffer(
          name, batch1_element_cnt, content, offsets, &batchn_shape, requests,
          responses);
    } else {
      // Fixed size data type...
      char* output_buffer = nullptr;
      RETURN_IF_ORT_ERROR(
          ort_api->GetTensorMutableData(output_tensor, (void**)&output_buffer));

      // [TODO] currently ONNX output data are always on CPU
      // https://github.com/microsoft/onnxruntime/issues/1621
      responder.ProcessTensor(
          name, ConvertFromOnnxDataType(type), batchn_shape, output_buffer,
          TRITONSERVER_MEMORY_CPU, 0);
    }
  }

  // Finalize and wait for any pending buffer copies.
  cuda_copy |= responder.Finalize();

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif  // TRITON_ENABLE_GPU
  return Status::Success;
}

bool
OnnxBackend::Context::SetStringOutputBuffer(
    const std::string& name, const size_t batch1_element_cnt,
    const char* content, const size_t* offsets,
    std::vector<int64_t>* batchn_shape,
    const std::vector<std::unique_ptr<InferenceRequest>>& requests,
    std::vector<std::unique_ptr<InferenceResponse>>* responses)
{
  size_t element_idx = 0;
  bool cuda_copy = false;
  for (size_t ridx = 0; ridx < requests.size(); ++ridx) {
    const auto& request = requests[ridx];
    auto& response = (*responses)[ridx];
    const size_t expected_element_cnt =
        std::max(1U, request->BatchSize()) * batch1_element_cnt;

    // If 'request' requested this output then copy it from
    // 'content'. If it did not request this output then just
    // skip it in the 'content'.
    if ((response != nullptr) &&
        (request->ImmutableRequestedOutputs().find(name) !=
         request->ImmutableRequestedOutputs().end())) {
      if (max_batch_size_ != NO_BATCHING) {
        (*batchn_shape)[0] = request->BatchSize();
      }
      InferenceResponse::Output* response_output = nullptr;
      response->AddOutput(
          name, DataType::TYPE_STRING, *batchn_shape, request->BatchSize(),
          &response_output);
      // Calculate expected byte size in advance using string offsets
      const size_t data_byte_size =
          offsets[element_idx + expected_element_cnt] - offsets[element_idx];
      const size_t expected_byte_size =
          data_byte_size + sizeof(uint32_t) * expected_element_cnt;

      void* buffer;
      TRITONSERVER_MemoryType actual_memory_type =
          TRITONSERVER_MEMORY_CPU_PINNED;
      int64_t actual_memory_type_id = 0;
      Status status = response_output->AllocateDataBuffer(
          &buffer, expected_byte_size, &actual_memory_type,
          &actual_memory_type_id);
      if (status.IsOk()) {
        bool cuda_used = false;
        size_t copied_byte_size = 0;
        for (size_t e = 0; e < expected_element_cnt; ++e) {
          const uint32_t len =
              offsets[element_idx + e + 1] - offsets[element_idx + e];
          // Prepend size of the string
          status = CopyBuffer(
              name, TRITONSERVER_MEMORY_CPU /* src_memory_type */,
              0 /* src_memory_type_id */, actual_memory_type,
              actual_memory_type_id, sizeof(uint32_t),
              static_cast<const void*>(&len),
              static_cast<char*>(buffer) + copied_byte_size, stream_,
              &cuda_used);
          if (!status.IsOk()) {
            break;
          }

          cuda_copy |= cuda_used;
          copied_byte_size += sizeof(uint32_t);

          // Copy raw string content
          status = CopyBuffer(
              name, TRITONSERVER_MEMORY_CPU /* src_memory_type */,
              0 /* src_memory_type_id */, actual_memory_type,
              actual_memory_type_id, len, content + offsets[element_idx + e],
              static_cast<char*>(buffer) + copied_byte_size, stream_,
              &cuda_used);
          if (!status.IsOk()) {
            break;
          }

          cuda_copy |= cuda_used;
          copied_byte_size += len;
        }
      }
      if (!status.IsOk()) {
        InferenceResponse::SendWithStatus(std::move(response), status);
      }
    }

    element_idx += expected_element_cnt;
  }

  return cuda_copy;
}

void
OnnxBackend::Context::ReleaseOrtRunResources()
{
  // Release input tensor if set
  for (auto& tensor : input_tensors_) {
    if (tensor != nullptr) {
      ort_api->ReleaseValue(tensor);
    }
  }
  input_tensors_.clear();

  // Release output tensor if set
  for (auto& tensor : output_tensors_) {
    if (tensor != nullptr) {
      ort_api->ReleaseValue(tensor);
    }
  }
  output_tensors_.clear();
}

std::ostream&
operator<<(std::ostream& out, const OnnxBackend& pb)
{
  out << "name=" << pb.Name() << std::endl;
  out << "contexts:" << std::endl;
  for (const auto& context : pb.contexts_) {
    out << "  name=" << context->name_ << ", gpu="
        << ((context->gpu_device_ == OnnxBackend::Context::NO_GPU_DEVICE)
                ? "<none>"
                : std::to_string(context->gpu_device_))
        << ", max_batch_size="
        << ((context->max_batch_size_ == OnnxBackend::Context::NO_BATCHING)
                ? "<none>"
                : std::to_string(context->max_batch_size_))
        << std::endl;
  }

  return out;
}

}}  // namespace nvidia::inferenceserver
