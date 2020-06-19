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
#include "src/core/metrics.h"
#include "src/core/model_config.h"
#include "src/core/model_config_utils.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

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
    const bool enable_pinned_input, const bool enable_pinned_output,
    std::unique_ptr<MetricModelReporter>&& metric_reporter)
    : BackendContext(
          name, gpu_device, max_batch_size, enable_pinned_input,
          enable_pinned_output, std::move(metric_reporter))
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
          uint32_t runner_idx,
          std::vector<std::unique_ptr<InferenceRequest>>&& requests) {
        Run(runner_idx, std::move(requests));
      }));

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
NetDefBackend::Context::ReadOutputTensors(
    const InferenceBackend* base, size_t total_batch_size,
    const std::vector<std::unique_ptr<InferenceRequest>>& requests,
    std::vector<std::unique_ptr<InferenceResponse>>* responses)
{
  BackendResponder responder(
      requests, responses, max_batch_size_, enable_pinned_output_, stream_);
  // Make sure each output is of the expected size and copy it into
  // the payload responses.
  bool cuda_copy = false;
  for (const auto& output : base->Config().output()) {
    const std::string& name = output.name();

    const ModelOutput* output_config;
    RETURN_IF_ERROR(base->GetOutput(name, &output_config));

    // Checked at initialization time to make sure that STRING is not
    // being used for an output, so can just assume fixed-sized here.
    const Caffe2Workspace::DataType dtype =
        ConvertDataType(output_config->data_type());

    const char* output_buffer = nullptr;
    size_t byte_size = 0;
    std::vector<int64_t> batchn_shape;
    Caffe2Workspace::Error err = workspace_->GetOutputTensor(
        name, dtype, &output_buffer, &byte_size, &batchn_shape);
    if (!err.IsOk()) {
      return Status(Status::Code::INTERNAL, err.Message());
    }

    responder.ProcessTensor(
        name, output_config->data_type(), batchn_shape, output_buffer,
        TRITONSERVER_MEMORY_CPU, 0);
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

Status
NetDefBackend::Context::SetInputTensors(
    size_t total_batch_size,
    const std::vector<std::unique_ptr<InferenceRequest>>& requests,
    std::vector<std::unique_ptr<InferenceResponse>>* responses,
    BackendInputCollector* collector,
    std::vector<std::unique_ptr<AllocatedMemory>>* input_buffers,
    bool* cuda_copy)
{
  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  for (const auto& pr : requests[0]->ImmutableInputs()) {
    const auto& name = pr.first;
    const auto& repr_input = pr.second;
    const auto& batch1_shape = repr_input->Shape();

    // The shape for the entire input patch, [total_batch_size, ...]
    std::vector<int64_t> batchn_shape;
    batchn_shape.reserve(batch1_shape.size() + 1);
    if (max_batch_size_ != NO_BATCHING) {
      batchn_shape.push_back(total_batch_size);
    }
    batchn_shape.insert(
        batchn_shape.end(), batch1_shape.begin(), batch1_shape.end());
    const DataType datatype = repr_input->DType();

    // Checked at initialization time to make sure that STRING is not
    // being used for an input, so can just assume fixed-sized here.
    const Caffe2Workspace::DataType dtype = ConvertDataType(datatype);

    // The entire input tensor must be delivered as a single
    // contiguous chunk so create a buffer large enough to hold the
    // entire dynamic batched input.
    input_buffers->emplace_back(new AllocatedMemory(
        GetByteSize(datatype, batchn_shape), TRITONSERVER_MEMORY_CPU_PINNED,
        0));
    TRITONSERVER_MemoryType mem_type;
    auto input_buffer = input_buffers->back()->MutableBuffer(&mem_type);
    auto total_byte_size = input_buffers->back()->TotalByteSize();

    collector->ProcessTensor(
        name, datatype, batch1_shape, input_buffer, total_byte_size, mem_type,
        0);

    Caffe2Workspace::Error err = workspace_->SetInputTensor(
        name, batchn_shape, dtype, static_cast<const char*>(input_buffer),
        total_byte_size);
    if (!err.IsOk()) {
      return Status(Status::Code::INTERNAL, err.Message());
    }
  }
  // Finalize...
  *cuda_copy |= collector->Finalize();
  return Status::Success;
}

void
NetDefBackend::Context::Run(
    InferenceBackend* base,
    std::vector<std::unique_ptr<InferenceRequest>>&& requests)
{
  LOG_VERBOSE(1) << "Running " << name_ << " with " << requests.size()
                 << " request payloads";

  INFER_STATS_DECL_TIMESTAMP(compute_start_ns);

  // For each request in 'payloads' collect the total batch size for
  // this inference execution. The batch-size, number of inputs, and
  // size of each input has already been checked by each request
  // normalizer so don't need to do that here.
  size_t total_batch_size = 0;
  for (auto& request : requests) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (request == nullptr) {
      InferenceRequest::RespondIfError(
          requests,
          Status(
              Status::Code::INTERNAL,
              "null request given to Caffe2 runner for '" + name_ + "'"),
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

  // Hold reference to each buffer of input data to that it stays
  // until the inference has completed.
  std::vector<std::unique_ptr<AllocatedMemory>> input_buffers;

  // Create a tensor for each input sized correctly for the total
  // payload batch size. Concatenate input values from each payload
  // into the corresponding tensor.

  // Inputs from the request...
  bool cuda_copy = false;
  BackendInputCollector collector(
      requests, &responses, enable_pinned_input_, stream_);
  FAIL_ALL_AND_RETURN_IF_ERROR(
      requests, responses, metric_reporter_.get(),
      SetInputTensors(
          total_batch_size, requests, &responses, &collector, &input_buffers,
          &cuda_copy),
      "error sending Caffe2 response");

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif  // TRITON_ENABLE_GPU

  INFER_STATS_DECL_TIMESTAMP(compute_input_end_ns);

  // Run...
  Caffe2Workspace::Error err = workspace_->Run();
  if (!err.IsOk()) {
    auto status = Status(Status::Code::INTERNAL, err.Message());
    FAIL_ALL_AND_RETURN_IF_ERROR(
        requests, responses, metric_reporter_.get(), status,
        "error sending Caffe2 response");
  }

  INFER_STATS_DECL_TIMESTAMP(compute_output_start_ns);

  FAIL_ALL_AND_RETURN_IF_ERROR(
      requests, responses, metric_reporter_.get(),
      ReadOutputTensors(base, total_batch_size, requests, &responses),
      "error sending Caffe2 response");

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
          InferenceResponse::Send(
              std::move(response), TRITONSERVER_RESPONSE_COMPLETE_FINAL),
          "failed to send TensorFlow backend response");
    }
  }

  // Release all requests.
  for (auto& request : requests) {
    InferenceRequest::Release(
        std::move(request), TRITONSERVER_REQUEST_RELEASE_ALL);
  }
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
