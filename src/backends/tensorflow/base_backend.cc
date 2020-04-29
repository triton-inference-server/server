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

#include "src/backends/tensorflow/base_backend.h"

#include <set>
#include "src/backends/tensorflow/tf_utils.h"
#include "src/backends/tensorflow/tf_virtual_device.h"
#include "src/core/constants.h"
#include "src/core/cuda_utils.h"
#include "src/core/logging.h"
#include "src/core/model_config.pb.h"
#include "src/core/model_config_utils.h"
#include "src/core/server_status.h"

#ifdef TRTIS_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRTIS_ENABLE_GPU

namespace nvidia { namespace inferenceserver {

BaseBackend::Context::Context(
    const std::string& name, const int gpu_device, const int max_batch_size,
    const bool enable_pinned_input, const bool enable_pinned_output)
    : BackendContext(
          name, gpu_device, max_batch_size, enable_pinned_input,
          enable_pinned_output),
      trtistf_model_(nullptr, TRTISTF_ModelDelete),
      input_device_id_(MODEL_DEVICE)
{
}

BaseBackend::Context::~Context()
{
  LOG_VERBOSE(1) << "~BaseBackend::Context ";
}

Status
BaseBackend::Init(
    const std::string& path, const ModelConfig& model_config,
    const GraphDefBackendFactory::Config* backend_config,
    const std::string& platform)
{
  RETURN_IF_ERROR(InferenceBackend::Init(path, model_config, platform));
  backend_config_ = backend_config;
  return Status::Success;
}

Status
BaseBackend::CreateExecutionContexts(
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
            instance_name, Context::NO_GPU_DEVICE, paths));
        total_context_cnt++;
      } else if (group.kind() == ModelInstanceGroup::KIND_MODEL) {
        const std::string instance_name =
            group.name() + "_" + std::to_string(c) + "_model_device";
        RETURN_IF_ERROR(CreateExecutionContext(
            instance_name, Context::MODEL_DEVICE, paths));
        total_context_cnt++;
      } else {
        for (int gpu_device : group.gpus()) {
          const std::string instance_name = group.name() + "_" +
                                            std::to_string(c) + "_gpu" +
                                            std::to_string(gpu_device);
          RETURN_IF_ERROR(
              CreateExecutionContext(instance_name, gpu_device, paths));
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
      },
      [this](
          uint32_t runner_idx, const InferenceRequest::Input& input,
          const std::unique_ptr<InferenceRequest>& request,
          std::vector<int64_t>* shape) -> Status { return Status::Success; }));

  LOG_VERBOSE(1) << "backend for " << Name() << std::endl << *this;

  return Status::Success;
}

Status
BaseBackend::CreateExecutionContext(
    const std::string& instance_name, const int gpu_device,
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
          Status::Code::INTERNAL, "unable to get CUDA device properties for " +
                                      Name() + ": " +
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
    return Status(Status::Code::INTERNAL, "GPU instances not supported");
#endif  // TRTIS_ENABLE_GPU
  }

  const auto& gdp_itr = paths.find(cc_model_filename);
  if (gdp_itr == paths.end()) {
    return Status(
        Status::Code::INTERNAL,
        "unable to find model '" + cc_model_filename + "' for " + Name());
  }

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

  RETURN_IF_ERROR(context->CreateCudaStream());

  RETURN_IF_ERROR(context->ValidateInputs(Config().input()));
  RETURN_IF_ERROR(context->ValidateOutputs(Config().output()));

  TRTISTF_TFTRTConfig* tftrt_config_ptr = nullptr;
  TRTISTF_TFTRTConfig tftrt_config;
  if (Config().optimization().has_execution_accelerators()) {
    // Set default values. is_dynamic_op is always true for online
    // TF-TRT.
    tftrt_config.minimum_segment_size_ = 3;
    tftrt_config.max_workspace_size_bytes_ = 1 << 30;
    tftrt_config.max_cached_engines_ = 100;
    tftrt_config.max_batch_size_ = std::max(Config().max_batch_size(), 1);
    tftrt_config.precision_mode_ = TRTISTF_MODE_FP32;
    tftrt_config.is_dynamic_op_ = true;

    if (!Config()
             .optimization()
             .execution_accelerators()
             .cpu_execution_accelerator()
             .empty()) {
      return Status(
          Status::Code::INVALID_ARG,
          "CPU Execution Accelerator is not supported in TensorFlow backend");
    }

    if (gpu_device == Context::NO_GPU_DEVICE) {
      return Status(
          Status::Code::INVALID_ARG,
          "GPU Execution Accelerator can only be set on non-CPU backend "
          "context");
    }
    for (const auto& execution_accelerator : Config()
                                                 .optimization()
                                                 .execution_accelerators()
                                                 .gpu_execution_accelerator()) {
      if (execution_accelerator.name() == kTensorRTExecutionAccelerator) {
        // Validate and set parameters
        for (const auto& parameter : execution_accelerator.parameters()) {
          if (parameter.first == "precision_mode") {
            if (parameter.second == "FP32") {
              tftrt_config.precision_mode_ = TRTISTF_MODE_FP32;
            } else if (parameter.second == "FP16") {
              tftrt_config.precision_mode_ = TRTISTF_MODE_FP16;
            } else {
              return Status(
                  Status::Code::INVALID_ARG, "unsupported precision mode '" +
                                                 parameter.second +
                                                 "' is requested");
            }
          } else if (parameter.first == "minimum_segment_size") {
            RETURN_IF_ERROR(ParseLongLongParameter(
                parameter.first, parameter.second,
                &tftrt_config.minimum_segment_size_));
          } else if (parameter.first == "max_workspace_size_bytes") {
            RETURN_IF_ERROR(ParseLongLongParameter(
                parameter.first, parameter.second,
                &tftrt_config.max_workspace_size_bytes_));
          } else if (parameter.first == "max_cached_engines") {
            RETURN_IF_ERROR(ParseLongLongParameter(
                parameter.first, parameter.second,
                &tftrt_config.max_cached_engines_));
          } else {
            return Status(
                Status::Code::INVALID_ARG,
                "unknown parameter '" + parameter.first +
                    "' is provided for TensorRT Execution Accelerator");
          }
        }
        LOG_VERBOSE(1) << "TensorRT Execution Accelerator is set for "
                       << instance_name;
      } else if (execution_accelerator.name() == kGPUIOExecutionAccelerator) {
        // GPU I/O can be set, set hint
        if ((gpu_device != Context::NO_GPU_DEVICE) &&
            (gpu_device != Context::MODEL_DEVICE)) {
          // In TensorFlow, TF device (vGPU) is used for device utilities
          context->input_device_id_ = vgpu_device;
        }
      } else {
        return Status(
            Status::Code::INVALID_ARG, "unknown Execution Accelerator '" +
                                           execution_accelerator.name() +
                                           "' is requested");
      }
    }
    tftrt_config_ptr = &tftrt_config;
  }

  RETURN_IF_ERROR(CreateTRTISTFModel(
      backend_config_, vgpu_device, Config().optimization().has_graph(),
      Config().optimization().graph().level(), gdp_itr->first, gdp_itr->second,
      &context->trtistf_model_, &context->input_name_map_,
      &context->output_name_map_, tftrt_config_ptr));


  if (context->input_device_id_ != Context::MODEL_DEVICE) {
    const size_t num_inputs = Config().input_size();
    const size_t num_outputs = Config().output_size();
    std::vector<const char*> input_names, output_names;
    std::vector<TRTISTF_DataType> input_types, output_types;
    for (const auto& io : Config().input()) {
      input_names.push_back(io.name().c_str());
      input_types.push_back(ConvertDataType(io.data_type()));
    }
    for (const auto& io : Config().output()) {
      output_names.push_back(io.name().c_str());
      output_types.push_back(ConvertDataType(io.data_type()));
    }
    TRTISTF_ModelMakeCallable(
        context->trtistf_model_.get(), input_names.data(), input_types.data(),
        num_inputs, output_names.data(), output_types.data(), num_outputs);
  }

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
          Status::Code::INTERNAL,
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
          Status::Code::INTERNAL,
          "unsupported datatype " + DataType_Name(io.data_type()) +
              " for output '" + io.name() + "' for model '" + name_ + "'");
    }
  }

  return Status::Success;
}

namespace {

void
FillStringTensor(TRTISTF_Tensor* tensor, const size_t idx, const size_t cnt)
{
  for (size_t c = 0; c < cnt; ++c) {
    TRTISTF_TensorSetString(tensor, idx + c, nullptr, 0);
  }
}

bool
SetStringOutputBuffer(
    TRTISTF_Tensor* tensor, std::unique_ptr<InferenceResponse>* response,
    InferenceResponse::Output* response_output,
    const size_t tensor_element_count, const size_t tensor_offset,
    cudaStream_t stream)
{
  bool cuda_copy = false;

  // Serialize the output tensor strings. Each string is serialized as
  // a 4-byte length followed by the string itself with no
  // null-terminator.
  std::string serialized;
  for (size_t e = 0; e < tensor_element_count; ++e) {
    size_t len;
    const char* cstr = TRTISTF_TensorString(tensor, tensor_offset + e, &len);
    serialized.append(reinterpret_cast<const char*>(&len), sizeof(uint32_t));
    if (len > 0) {
      serialized.append(cstr, len);
    }
  }

  // Allocate a buffer large enough to hold the serialized tensor.
  TRITONSERVER_MemoryType actual_memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t actual_memory_type_id = 0;

  void* buffer;
  Status status = response_output->AllocateDataBuffer(
      &buffer, serialized.size(), &actual_memory_type, &actual_memory_type_id);
  if (!status.IsOk()) {
    LOG_STATUS_ERROR(
        InferenceResponse::SendWithStatus(std::move(*response), status),
        "error sending TensorFlow response");
    return cuda_copy;
  }

  // Copy the serialized tensor into the allocated buffer.
  bool cuda_used = false;
  status = CopyBuffer(
      response_output->Name(), TRITONSERVER_MEMORY_CPU /* src_memory_type */,
      0 /* src_memory_type_id */, actual_memory_type, actual_memory_type_id,
      serialized.size(), reinterpret_cast<const void*>(serialized.c_str()),
      buffer, stream, &cuda_used);
  cuda_copy |= cuda_used;

  if (!status.IsOk()) {
    LOG_STATUS_ERROR(
        InferenceResponse::SendWithStatus(std::move(*response), status),
        "error sending TensorFlow response");
    return cuda_copy;
  }

  return cuda_copy;
}

}  // namespace

// FIXME instead of returning status, errors should be reported in the
// corresponding request/response objects
Status
BaseBackend::Context::SetInput(
    const std::string& name, const DataType datatype,
    const std::vector<int64_t>& dims, const size_t total_batch_size,
    std::vector<std::unique_ptr<InferenceRequest>>* requests,
    std::vector<InputInfo>* inputs, TRTISTF_TensorList** input_tensors,
    bool* cuda_copy)
{
  // Get the shape of the input. The request normalizer has already
  // checked that the request shape is valid so don't need to do it
  // here.
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

  // Only try to create a tensor on specific device if 'input_device_id_' is set
  const TRTISTF_DataType dtype = ConvertDataType(datatype);
  TRTISTF_Tensor* tensor = TRTISTF_TensorNew(
      input_tensor_name->c_str(), dtype, shape.size(),
      (shape.size() == 0) ? nullptr : &shape[0], input_device_id_);
  if (tensor == nullptr) {
    return Status(
        Status::Code::INTERNAL,
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
          Status::Code::INTERNAL,
          "failed to create input tensor '" + name +
              "' with expected byte size " +
              std::to_string(batch1_byte_size * total_batch_size) + ", got " +
              std::to_string(TRTISTF_TensorDataByteSize(tensor)));
    }
    inputs->emplace_back();
    SetFixedSizedInputTensor(
        tensor, name, batch1_byte_size, requests, &inputs->back(), cuda_copy);
  } else {
    SetStringInputTensor(tensor, name, batch1_element_cnt, requests);
  }

  return Status::Success;
}

void
BaseBackend::Context::SetFixedSizedInputTensor(
    TRTISTF_Tensor* tensor, const std::string& input_name,
    const size_t batch1_byte_size,
    std::vector<std::unique_ptr<InferenceRequest>>* requests, InputInfo* input,
    bool* cuda_copy)
{
  input->input_buffer_ = TRTISTF_TensorData(tensor);

  // Visit the requests in order and copy the input values into the
  // input tensor. Skip requests that had errors since they are not
  // included in the dynamic batch.
  std::vector<size_t> expected_byte_sizes;
  for (auto& irequest : *requests) {
    expected_byte_sizes.push_back(irequest->BatchSize() * batch1_byte_size);
  }

  input->memory_type_ = (TRTISTF_TensorIsGPUTensor(tensor))
                            ? TRITONSERVER_MEMORY_GPU
                            : TRITONSERVER_MEMORY_CPU;
  input->memory_type_id_ =
      (TRTISTF_TensorIsGPUTensor(tensor)) ? gpu_device_ : 0;
  LOG_VERBOSE(1) << "input '" << input_name
                 << "' is GPU tensor: " << TRTISTF_TensorIsGPUTensor(tensor);
  *cuda_copy |=
      SetInputBuffer(input_name, expected_byte_sizes, requests, input);
}

void
BaseBackend::Context::SetStringInputTensor(
    TRTISTF_Tensor* tensor, const std::string& input_name,
    const size_t batch1_element_cnt,
    std::vector<std::unique_ptr<InferenceRequest>>* requests)
{
  size_t tensor_element_idx = 0;

  // Visit the requests in order and copy the input values into the
  // input tensor. Skip requests that had errors since they are not
  // included in the dynamic batch.
  for (auto& irequest : *requests) {
    const size_t expected_element_cnt =
        irequest->BatchSize() * batch1_element_cnt;
    size_t element_idx = 0;

    // For string data type, we always need to copy the data to CPU so that
    // we can read string length and construct the string properly.
    auto buffer_memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
    int64_t buffer_memory_type_id = 0;
    const char* content;
    size_t content_byte_size = expected_element_cnt * sizeof(uint32_t);
    // If contiguous buffer is created, it needs to live until tensor is filled
    std::unique_ptr<AllocatedMemory> contiguous_buffer;
    bool cuda_copy = false;
    Status status = GetContiguousInputContent(
        input_name, buffer_memory_type, buffer_memory_type_id, irequest,
        &content, &content_byte_size, &contiguous_buffer, &cuda_copy);

    if (!status.IsOk()) {
      InferenceRequest::RespondIfError(irequest, status);
      FillStringTensor(
          tensor, tensor_element_idx + element_idx,
          expected_element_cnt - element_idx);
      continue;
    }

    // [TODO] defer synchronize as far as possible, need rework on setting
    // String input. i.e. get all contiguous data first, then sync and set.
#ifdef TRTIS_ENABLE_GPU
    if (cuda_copy) {
      cudaStreamSynchronize(stream_);
    }
#endif  // TRTIS_ENABLE_GPU

    // Parse content and assign them to the 'tensor'. Each string
    // in 'content' is a 4-byte length followed by the string
    // itself with no null-terminator.
    while (content_byte_size >= sizeof(uint32_t)) {
      if (element_idx >= expected_element_cnt) {
        InferenceRequest::RespondIfError(
            irequest,
            Status(
                Status::Code::INVALID_ARG,
                "unexpected number of string elements " +
                    std::to_string(element_idx + 1) + " for inference input '" +
                    input_name + "', expecting " +
                    std::to_string(expected_element_cnt)));
        FillStringTensor(
            tensor, tensor_element_idx + element_idx,
            expected_element_cnt - element_idx);
        break;
      }

      const uint32_t len = *(reinterpret_cast<const uint32_t*>(content));
      content += sizeof(uint32_t);
      content_byte_size -= sizeof(uint32_t);

      if (content_byte_size < len) {
        InferenceRequest::RespondIfError(
            irequest,
            Status(
                Status::Code::INVALID_ARG,
                "incomplete string data for inference input '" + input_name +
                    "', expecting string of length " + std::to_string(len) +
                    " but only " + std::to_string(content_byte_size) +
                    " bytes available"));
        FillStringTensor(
            tensor, tensor_element_idx + element_idx,
            expected_element_cnt - element_idx);
        break;
      }

      TRTISTF_TensorSetString(
          tensor, tensor_element_idx + element_idx, content, len);
      content += len;
      content_byte_size -= len;
      element_idx++;
    }

    if ((irequest != nullptr) && (element_idx != expected_element_cnt)) {
      InferenceRequest::RespondIfError(
          irequest, Status(
                        Status::Code::INTERNAL,
                        "expected " + std::to_string(expected_element_cnt) +
                            " strings for inference input '" + input_name +
                            "', got " + std::to_string(element_idx)));
      FillStringTensor(
          tensor, tensor_element_idx + element_idx,
          expected_element_cnt - element_idx);
    }

    tensor_element_idx += expected_element_cnt;
  }
}

void
BaseBackend::Context::Run(
    const InferenceBackend* base,
    std::vector<std::unique_ptr<InferenceRequest>>&& requests)
{
  LOG_VERBOSE(1) << "Running " << name_ << " with " << requests.size()
                 << " requests";

#ifdef TRTIS_ENABLE_STATS
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  auto compute_start_ns = TIMESPEC_TO_NANOS(ts);
#endif  // TRTIS_ENABLE_STATS

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
              "null request given to TensorFlow runner for '" + name_ + "'"),
          true /* release_requests */);
      return;
    }

    total_batch_size += request->BatchSize();

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

  // Create a tensor for each input sized correctly for the total
  // batch size. Concatenate input values from each request into the
  // corresponding tensor.

  // Unique pointer is TensorList** as the pointer to input head
  // (TensorList*) will be updated in SetInput()
  TRTISTF_TensorList* input_head_ptr = nullptr;
  static auto input_deleter = [](TRTISTF_TensorList** list) {
    if (list != nullptr) {
      TRTISTF_TensorListDelete(*list);
    }
  };
  std::unique_ptr<TRTISTF_TensorList*, decltype(input_deleter)> input_tensors(
      &input_head_ptr, input_deleter);

  // Inputs from the request...
  std::vector<InputInfo> inputs;
  bool cuda_copy = false;
  for (const auto& pr : repr_input_request->ImmutableInputs()) {
    const InferenceRequest::Input* input = pr.second;
    const std::string& name = input->Name();

    // FIXME
    SetInput(
        name, input->DType(), input->Shape(), total_batch_size, &requests,
        &inputs, input_tensors.get(), &cuda_copy);
  }

  // Collect the names of requested outputs. Do not include outputs
  // for requests that have already responded with an error.
  std::set<std::string> required_outputs;
  for (size_t idx = 0; idx < requests.size(); idx++) {
    const auto& request = requests[idx];
    const auto& response = responses[idx];
    if (response != nullptr) {
      for (const auto& pr : request->ImmutableRequestedOutputs()) {
        required_outputs.insert(pr.first);
      }
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

#ifdef TRTIS_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }

  cuda_copy = false;
  for (auto& input : inputs) {
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
          input.input_buffer_ + std::get<1>(indirect_buffer), stream_,
          &cuda_used);
      if (!status.IsOk()) {
        for (const auto& request_idx : std::get<2>(indirect_buffer)) {
          if (responses[request_idx] != nullptr) {
            LOG_STATUS_ERROR(
                InferenceResponse::SendWithStatus(
                    std::move(responses[request_idx]), status),
                "error sending TensorFlow response");
          }
        }
      } else {
        cuda_copy |= cuda_used;
      }
    }
  }

  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif  // TRTIS_ENABLE_GPU

#ifdef TRTIS_ENABLE_STATS
  clock_gettime(CLOCK_MONOTONIC, &ts);
  auto compute_input_end_ns = TIMESPEC_TO_NANOS(ts);
#endif  // TRTIS_ENABLE_STATS

  // Run. Session will update the 'output_tensors'.
  std::unique_ptr<TRTISTF_TensorList, decltype(&TRTISTF_TensorListDelete)>
      output_tensors(nullptr, TRTISTF_TensorListDelete);

  {
    TRTISTF_TensorList* rtl;

    TRTISTF_Error* err = TRTISTF_ModelRun(
        trtistf_model_.get(), *(input_tensors.release()),
        required_outputs.size(), output_names_cstr, &rtl);
    if (err != nullptr) {
      // Something went wrong with the entire batch inference. For
      // every response that has not already been sent with an
      // error... send it now...
      for (auto& response : responses) {
        if (response != nullptr) {
          LOG_STATUS_ERROR(
              InferenceResponse::SendWithStatus(
                  std::move(response),
                  Status(Status::Code::INTERNAL, err->msg_)),
              "error sending TensorFlow response");
        }
      }
      TRTISTF_ErrorDelete(err);
    }

    output_tensors.reset(rtl);
  }

#ifdef TRTIS_ENABLE_STATS
  clock_gettime(CLOCK_MONOTONIC, &ts);
  auto compute_output_start_ns = TIMESPEC_TO_NANOS(ts);
#endif  // TRTIS_ENABLE_STATS

  // Create the response tensors and copy the appropriate tensor data
  // into each. For tensors with string data type we must handle
  // ourselves since we must use TF-specific string tensor APIs.
  cuda_copy = false;

  {
    BackendResponder responder(
        requests, &responses, enable_pinned_output_, stream_);

    TRTISTF_TensorList* output_tensor_itr = output_tensors.get();
    for (const auto& name : model_output_names) {
      TRTISTF_Tensor* output_tensor = output_tensor_itr->tensor_;

      TRTISTF_DataType tf_datatype = TRTISTF_TensorDataType(output_tensor);
      TRTISTF_Shape* tf_shape = TRTISTF_TensorShape(output_tensor);

      const DataType datatype = ConvertDataType(tf_datatype);
      std::vector<int64_t> shape;
      shape.reserve(tf_shape->rank_);
      for (size_t itr = 0; itr < tf_shape->rank_; itr++) {
        const int64_t dim = tf_shape->dims_[itr];
        shape.push_back(dim);
      }

      // Custom handling for string/bytes tensor...
      if (datatype == DataType::TYPE_STRING) {
        size_t tensor_offset = 0;
        const size_t tensor_element_count = GetElementCount(shape);

        for (size_t idx = 0; idx < responses.size(); idx++) {
          auto& request = requests[idx];
          auto& response = responses[idx];

          // Only need an response tensor for requested outputs.
          if ((response != nullptr) &&
              (request->ImmutableRequestedOutputs().find(name) !=
               request->ImmutableRequestedOutputs().end())) {
            InferenceResponse::Output* response_output = nullptr;
            response->AddOutput(name, datatype, shape, &response_output);
            cuda_copy |= SetStringOutputBuffer(
                output_tensor, &response, response_output, tensor_element_count,
                tensor_offset, stream_);
          }

          tensor_offset += tensor_element_count;
        }
      }
      // Use the responder for non-STRING datatype...
      else {  // datatype != DataType::TYPE_STRING
        responder.ProcessTensor(
            name, datatype, shape, TRTISTF_TensorData(output_tensor),
            (TRTISTF_TensorIsGPUTensor(output_tensor))
                ? TRITONSERVER_MEMORY_GPU
                : TRITONSERVER_MEMORY_CPU,
            (TRTISTF_TensorIsGPUTensor(output_tensor)) ? gpu_device_ : 0);
      }

      output_tensor_itr = output_tensor_itr->next_;
    }

    // Finalize and wait for any pending buffer copies.
    cuda_copy |= responder.Finalize();
  }

#ifdef TRTIS_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif  // TRTIS_ENABLE_GPU

#ifdef TRTIS_ENABLE_STATS
  clock_gettime(CLOCK_MONOTONIC, &ts);
  auto compute_end_ns = TIMESPEC_TO_NANOS(ts);

  // Report stats
  auto compute_input_duration_ns = compute_input_end_ns - compute_start_ns;
  auto compute_infer_duration_ns =
      compute_output_start_ns - compute_input_end_ns;
  auto compute_output_duration_ns = compute_end_ns - compute_output_start_ns;
  for (size_t i = 0; i < requests.size(); ++i) {
    requests[i]->Report(
        (responses[i] != nullptr), gpu_device_, compute_start_ns,
        compute_input_end_ns, compute_output_start_ns, compute_end_ns,
        compute_input_duration_ns, compute_infer_duration_ns,
        compute_output_duration_ns);
  }
  // Also reporting batch stats
  base->StatsCollector()->UpdateInferBatchStats(
      gpu_device_, total_batch_size, compute_input_duration_ns,
      compute_infer_duration_ns, compute_output_duration_ns);
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
