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

#include "src/backends/pytorch/libtorch_backend.h"

#include <stdint.h>
#include <exception>
#include <memory>
#include "src/core/constants.h"
#include "src/core/cuda_utils.h"
#include "src/core/logging.h"
#include "src/core/metrics.h"
#include "src/core/model_config_cuda.h"
#include "src/core/model_config_utils.h"

#ifdef TRITON_ENABLE_GPU
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>
#include "src/core/cuda_utils.h"
#endif  // TRITON_ENABLE_GPU

namespace nvidia { namespace inferenceserver {

LibTorchBackend::Context::Context(
    const std::string& name, const int gpu_device, const int max_batch_size,
    const bool enable_pinned_input, const bool enable_pinned_output,
    std::unique_ptr<MetricModelReporter>&& metric_reporter)
    : BackendContext(
          name, gpu_device, max_batch_size, enable_pinned_input,
          enable_pinned_output, std::move(metric_reporter)),
      device_(torch::Device(torch::kCPU))
{
}

LibTorchBackend::Context::~Context()
{
  torch_model_.reset();
#ifdef TRITON_ENABLE_GPU
  c10::cuda::CUDACachingAllocator::emptyCache();
#endif  // TRITON_ENABLE_GPU
  LOG_VERBOSE(1) << "~LibTorchBackend::Context ";
}

std::pair<bool, torch::ScalarType>
ConvertDataTypeToTorchType(const DataType& dtype)
{
  torch::ScalarType type = torch::kInt;
  switch (dtype) {
    case TYPE_BOOL:
      type = torch::kBool;
      break;
    case TYPE_UINT8:
      type = torch::kByte;
      break;
    case TYPE_INT8:
      type = torch::kChar;
      break;
    case TYPE_INT16:
      type = torch::kShort;
      break;
    case TYPE_INT32:
      type = torch::kInt;
      break;
    case TYPE_INT64:
      type = torch::kLong;
      break;
    case TYPE_FP16:
      type = torch::kHalf;
      break;
    case TYPE_FP32:
      type = torch::kFloat;
      break;
    case TYPE_FP64:
      type = torch::kDouble;
      break;
    case TYPE_UINT16:
    case TYPE_UINT32:
    case TYPE_UINT64:
    case TYPE_STRING:
    default:
      return std::make_pair(false, type);
  }

  return std::make_pair(true, type);
}

DataType
ConvertTorchTypeToDataType(const torch::ScalarType& ttype)
{
  switch (ttype) {
    case torch::kBool:
      return TYPE_BOOL;
    case torch::kByte:
      return TYPE_UINT8;
    case torch::kChar:
      return TYPE_INT8;
    case torch::kShort:
      return TYPE_INT16;
    case torch::kInt:
      return TYPE_INT32;
    case torch::kLong:
      return TYPE_INT64;
    case torch::kHalf:
      return TYPE_FP16;
    case torch::kFloat:
      return TYPE_FP32;
    case torch::kDouble:
      return TYPE_FP64;
    default:
      return TYPE_FP32;
  }
}

Status
LibTorchBackend::CreateExecutionContexts(
    const std::unordered_map<std::string, std::string>& models)
{
  uint32_t total_context_cnt = 0;

  // Create a context for each instance.
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

  return Status::Success;
}

Status
LibTorchBackend::CreateExecutionContext(
    const std::string& instance_name, const int gpu_device,
    const std::unordered_map<std::string, std::string>& models)
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

  const auto& lp_itr = models.find(cc_model_filename);
  if (lp_itr == models.end()) {
    return Status(
        Status::Code::INTERNAL, "unable to find LibTorch model '" +
                                    cc_model_filename + "' for " + Name());
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

  if (gpu_device == Context::NO_GPU_DEVICE) {
    context->device_ = torch::Device(torch::kCPU);
  } else {
    context->device_ = torch::Device(torch::kCUDA, gpu_device);
  }

  try {
    // lp_itr->second is the torch model serialized to string
    std::istringstream model_stream(lp_itr->second);
    context->torch_model_ = std::make_shared<torch::jit::script::Module>(
        torch::jit::load(model_stream, context->device_));
  }
  catch (const std::exception& ex) {
    return Status(
        Status::Code::INTERNAL, "load failed for libtorch model -> '" +
                                    Config().name() + "': " + ex.what());
  }

  RETURN_IF_ERROR(context->ValidateInputs(Config().input()));
  RETURN_IF_ERROR(context->ValidateOutputs(Config().output()));
  if (Config().has_sequence_batching()) {
    RETURN_IF_ERROR(
        context->ValidateControlInputs(Config().sequence_batching()));
  }
  return Status::Success;
}

Status
LibTorchBackend::Context::ValidateInputs(
    const ::google::protobuf::RepeatedPtrField<ModelInput>& ios)
{
  std::string deliminator = "__";
  int ip_index = 0;

  for (const auto& io : ios) {
    const auto pr = ConvertDataTypeToTorchType(io.data_type());
    if (!pr.first) {
      return Status(
          Status::Code::INTERNAL,
          "unsupported datatype " + DataType_Name(io.data_type()) +
              " for input '" + io.name() + "' for model '" + name_ + "'");
    } else {
      const std::string& name = io.name();
      try {
        int start_pos = name.find(deliminator);
        if (start_pos == -1) {
          throw std::invalid_argument(
              "Input '" + name +
              "' does not follow naming convention i.e. <name>__<index>.");
        }
        input_index_map_[name] = ip_index;
        ip_index = std::atoi(name.substr(start_pos + 2).c_str());
      }
      catch (std::exception& ex) {
        return Status(
            Status::Code::INTERNAL,
            "Input '" + name +
                "' does not follow naming convention i.e. <name>__<index>.");
      }
    }
  }

  return Status::Success;
}

Status
LibTorchBackend::Context::ValidateControlInputs(
    const ModelSequenceBatching& batching)
{
  std::string deliminator = "__";
  int ip_index = 0;

  for (const auto& io : batching.control_input()) {
    const std::string& name = io.name();
    try {
      int start_pos = name.find(deliminator);
      if (start_pos == -1) {
        throw std::invalid_argument(
            "Input '" + name +
            "' does not follow naming convention i.e. <name>__<index>.");
      }
      ip_index = std::atoi(name.substr(start_pos + 2).c_str());
      input_index_map_[name] = ip_index;
    }
    catch (std::exception& ex) {
      return Status(
          Status::Code::INTERNAL,
          "Input '" + name +
              "' does not follow naming convention i.e. <name>__<index>.");
    }
  }

  return Status::Success;
}

Status
LibTorchBackend::Context::ValidateOutputs(
    const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios)
{
  std::string deliminator = "__";
  int op_index;

  for (const auto& io : ios) {
    const auto pr = ConvertDataTypeToTorchType(io.data_type());
    if (!pr.first) {
      return Status(
          Status::Code::INTERNAL,
          "unsupported datatype " + DataType_Name(io.data_type()) +
              " for output '" + io.name() + "' for model '" + name_ + "'");
    } else {
      const std::string& name = io.name();
      try {
        int start_pos = name.find(deliminator);
        if (start_pos == -1) {
          throw std::invalid_argument(
              "Output '" + name +
              "' does not follow naming convention i.e. <name>__<index>.");
        }
        op_index = std::atoi(name.substr(start_pos + 2).c_str());
        output_index_map_[name] = op_index;
      }
      catch (std::exception& ex) {
        return Status(
            Status::Code::INTERNAL,
            "Output '" + name +
                "' does not follow naming convention i.e. <name>__<index>.");
      }
    }
  }

  return Status::Success;
}

Status
LibTorchBackend::Context::SetInputTensors(
    size_t total_batch_size,
    const std::vector<std::unique_ptr<InferenceRequest>>& requests,
    std::vector<std::unique_ptr<InferenceResponse>>* responses,
    std::vector<std::unique_ptr<AllocatedMemory>>* input_buffers,
    std::vector<torch::jit::IValue>* inputs, bool* cuda_copy)
{
  BackendInputCollector collector(
      requests, responses, enable_pinned_input_, stream_);

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  for (const auto& pr : requests[0]->ImmutableInputs()) {
    const std::string& input_name = pr.first;
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

    int ip_index = input_index_map_[input_name];

    const auto torch_dtype = ConvertDataTypeToTorchType(datatype);
    if (!torch_dtype.first) {
      return Status(
          Status::Code::INTERNAL, "Failed to convert DataType '" +
                                      DataType_Name(datatype) +
                                      "' to Torch datatype");
    }

    // Checked at initialization time to make sure that STRING is not
    // being used for an input, so can just assume fixed-sized here.
    const size_t total_byte_size = GetByteSize(datatype, batchn_shape);

    // The entire input tensor must be delivered as a single
    // contiguous chunk so create a buffer large enough to hold the
    // entire dynamic batched input.
    input_buffers->emplace_back(new AllocatedMemory(
        total_byte_size,
        (gpu_device_ == NO_GPU_DEVICE) ? TRITONSERVER_MEMORY_CPU_PINNED
                                       : TRITONSERVER_MEMORY_GPU,
        (gpu_device_ == NO_GPU_DEVICE) ? 0 : gpu_device_));

    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    auto input_buffer =
        input_buffers->back()->MutableBuffer(&memory_type, &memory_type_id);

    collector.ProcessTensor(
        input_name, datatype, batch1_shape, input_buffer, total_byte_size,
        memory_type, memory_type_id);

    torch::TensorOptions options{torch_dtype.second};
    auto updated_options = (memory_type == TRITONSERVER_MEMORY_GPU)
                               ? options.device(torch::kCUDA, memory_type_id)
                               : options.device(torch::kCPU);

    torch::Tensor input_tensor =
        torch::from_blob(input_buffer, batchn_shape, updated_options);

    if (input_tensor.nbytes() != total_byte_size) {
      return Status(
          Status::Code::INTERNAL,
          "unexpected size " + std::to_string(total_byte_size) +
              " for inference input '" + input_name + "', expecting " +
              std::to_string(input_tensor.nbytes()));
    }
    (*inputs)[ip_index] = input_tensor;
  }

  // Finalize...
  *cuda_copy |= collector.Finalize();
  return Status::Success;
}

Status
LibTorchBackend::Context::ReadOutputTensors(
    const InferenceBackend* base, size_t total_batch_size,
    const std::vector<std::unique_ptr<InferenceRequest>>& requests,
    std::vector<std::unique_ptr<InferenceResponse>>* responses,
    std::vector<torch::Tensor>* outputs,
    std::unordered_map<std::string, int>* output_index_map)
{
  BackendResponder responder(
      requests, responses, max_batch_size_, enable_pinned_output_, stream_);
  // Make sure each output is of the expected size and copy it into
  // the payload responses.
  bool cuda_copy = false;
  for (const auto& output : base->Config().output()) {
    const std::string& name = output.name();
    int op_index = (*output_index_map)[name];

    const ModelOutput* output_config;
    RETURN_IF_ERROR(base->GetOutput(name, &output_config));

    // Checked at initialization time to make sure that STRING is not
    // being used for an output, so can just assume fixed-sized here.
    const DataType dtype = output_config->data_type();

    const char* output_buffer = nullptr;
    size_t byte_size = 0;
    std::vector<int64_t> batchn_shape;
    RETURN_IF_ERROR(GetOutputTensor(
        outputs, op_index, name, dtype, &output_buffer, &byte_size,
        &batchn_shape));

    responder.ProcessTensor(
        name, dtype, batchn_shape, output_buffer,
        (device_ == torch::kCPU) ? TRITONSERVER_MEMORY_CPU
                                 : TRITONSERVER_MEMORY_GPU,
        (device_ == torch::kCPU) ? 0 : gpu_device_);
  }

  // Finalize and wait for any pending buffer copies.
  cuda_copy |= responder.Finalize();

#ifdef TRTIS_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif  // TRTIS_ENABLE_GPU
  return Status::Success;
}

Status
LibTorchBackend::Context::GetOutputTensor(
    std::vector<torch::Tensor>* outputs_, const int& op_index,
    const std::string& name, const DataType dtype, const char** content,
    size_t* byte_size, std::vector<int64_t>* content_shape)
{
  try {
    torch::Tensor output_flat = (*outputs_)[op_index].contiguous().flatten();

    // verify output datatype matches datatype from model config
    DataType rec_dtype = ConvertTorchTypeToDataType(output_flat.scalar_type());
    if (dtype != rec_dtype) {
      return Status(
          Status::Code::INVALID_ARG,
          "unexpected datatype " + DataType_Name(rec_dtype) +
              " for inference output '" + name + "', expecting " +
              DataType_Name(dtype));
    }

    *byte_size = output_flat.nbytes();
    *content = static_cast<const char*>(output_flat.data_ptr());

    //  Set content shape
    auto shape = (*outputs_)[op_index].sizes();
    for (auto itr = shape.begin(); itr != shape.end(); itr++) {
      content_shape->push_back(*itr);
    }
  }
  catch (std::exception& ex) {
    return Status(Status::Code::INTERNAL, "failed to get LibTorch output");
  }

  return Status::Success;
}

void
LibTorchBackend::Context::Run(
    InferenceBackend* base,
    std::vector<std::unique_ptr<InferenceRequest>>&& requests)
{
  LOG_VERBOSE(1) << "Running " << name_ << " with " << requests.size()
                 << " requests";

  INFER_STATS_DECL_TIMESTAMP(compute_start_ns);

  const InferenceRequest* repr_input_request = nullptr;

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.
  size_t total_batch_size = 0;
  for (auto& request : requests) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (request == nullptr) {
      InferenceRequest::RespondIfError(
          requests,
          Status(
              Status::Code::INTERNAL,
              "null request given to LibTorch runner for '" + name_ + "'"),
          true /* release_requests */);
      return;
    }

    total_batch_size += std::max(1U, request->BatchSize());

    // All requests must have equally-sized input tensors so use any
    // request as the representative for the input tensors.
    repr_input_request = request.get();
  }

  // If there are no valid requests then no need to run the
  // inference. This should never happen unless called with an empty
  // 'requests' for some reason.
  if (total_batch_size == 0) {
    return;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size_ == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
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

  size_t input_count = repr_input_request->ImmutableInputs().size();

  // Hold reference to each buffer of input data to that it stays
  // until the inference has completed.
  std::vector<std::unique_ptr<AllocatedMemory>> input_buffers;

  // Store input and output tensors
  std::vector<torch::jit::IValue> inputs_(input_count);
  std::vector<torch::Tensor> outputs_;

  bool cuda_copy = false;

  // Collect the request inputs into contiguous input tensors.
  FAIL_ALL_AND_RETURN_IF_ERROR(
      requests, responses, metric_reporter_.get(),
      SetInputTensors(
          total_batch_size, requests, &responses, &input_buffers, &inputs_,
          &cuda_copy),
      "error sending LibTorch response");

  // Collect the names of requested outputs. Do not include outputs
  // for requests that have already responded with an error.
  std::set<std::string> required_outputs;
  for (size_t idx = 0; idx < requests.size(); idx++) {
    const auto& request = requests[idx];
    const auto& response = responses[idx];
    if (response != nullptr) {
      for (const auto& output_name : request->ImmutableRequestedOutputs()) {
        required_outputs.insert(output_name);
      }
    }
  }

  // Wait for any in-flight input tensor copies to complete.
#ifdef TRTIS_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif

  INFER_STATS_DECL_TIMESTAMP(compute_input_end_ns);

  // Run...
  FAIL_ALL_AND_RETURN_IF_ERROR(
      requests, responses, metric_reporter_.get(), Execute(&inputs_, &outputs_),
      "error running LibTorch model");

  INFER_STATS_DECL_TIMESTAMP(compute_output_start_ns);

  // Verify output indices are valid with number of outputs after execution
  int max_index = outputs_.size() - 1;
  for (const auto& name : required_outputs) {
    int op_index = output_index_map_[name];
    if ((op_index < 0) || (op_index > max_index)) {
      Status status = Status(
          Status::Code::INVALID_ARG,
          "The output " + name +
              " in the model configuration refers to an output index which "
              "doesn't exist. This model has " +
              std::to_string(max_index + 1) + " outputs");

      FAIL_ALL_AND_RETURN_IF_ERROR(
          requests, responses, metric_reporter_.get(), status,
          "error creating LibTorch output tensor");
    }
  }

  // Create the response tensors and copy the appropriate tensor data
  // into each.
  FAIL_ALL_AND_RETURN_IF_ERROR(
      requests, responses, metric_reporter_.get(),
      ReadOutputTensors(
          base, total_batch_size, requests, &responses, &outputs_,
          &output_index_map_),
      "error sending LibTorch response");

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif  // TRTIS_ENABLE_GPU

#ifdef TRTIS_ENABLE_STATS
  INFER_STATS_DECL_TIMESTAMP(compute_end_ns);

  // Report stats and trace
  for (size_t i = 0; i < requests.size(); ++i) {
    auto& request = requests[i];
    request->ReportStatistics(
        metric_reporter_.get(), (responses[i] != nullptr), compute_start_ns,
        compute_input_end_ns, compute_output_start_ns, compute_end_ns);

#ifdef TRTIS_ENABLE_TRACING
    if (request->Trace() != nullptr) {
      auto& trace = request->Trace();
      trace->Report(TRITONSERVER_TRACE_COMPUTE_START, compute_start_ns);
      trace->Report(TRITONSERVER_TRACE_COMPUTE_INPUT_END, compute_input_end_ns);
      trace->Report(
          TRITONSERVER_TRACE_COMPUTE_OUTPUT_START, compute_output_start_ns);
      trace->Report(TRITONSERVER_TRACE_COMPUTE_END, compute_end_ns);
    }
#endif  // TRTIS_ENABLE_TRACING
  }

  // Also reporting batch stats
  base->MutableStatsAggregator()->UpdateInferBatchStats(
      metric_reporter_.get(), total_batch_size, compute_start_ns,
      compute_input_end_ns, compute_output_start_ns, compute_end_ns);
#endif  // TRTIS_ENABLE_STATS

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
LibTorchBackend::Context::Execute(
    std::vector<torch::jit::IValue>* inputs_,
    std::vector<torch::Tensor>* outputs_)
{
  torch::jit::IValue model_outputs_;

  try {
    model_outputs_ = torch_model_->forward(*inputs_);
    auto model_outputs_tuple = model_outputs_.toTuple();
    for (auto& m_op : model_outputs_tuple->elements()) {
      outputs_->push_back(m_op.toTensor());
    }
  }
  catch (std::exception& ex) {
    try {
      auto model_output_tensor = model_outputs_.toTensor();
      outputs_->push_back(model_output_tensor);
    }
    catch (std::exception& exx) {
      LOG_VERBOSE(1) << ex.what();
      return Status(Status::Code::INTERNAL, "failed to run model '" + name_);
    }
  }

  return Status::Success;
}

std::ostream&
operator<<(std::ostream& out, const LibTorchBackend& pb)
{
  out << "name=" << pb.Name() << std::endl;
  out << "contexts:" << std::endl;
  for (const auto& context : pb.contexts_) {
    out << "  name=" << context->name_ << ", gpu="
        << ((context->gpu_device_ == LibTorchBackend::Context::NO_GPU_DEVICE)
                ? "<none>"
                : std::to_string(context->gpu_device_));
  }

  return out;
}

}}  // namespace nvidia::inferenceserver
