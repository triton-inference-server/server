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
      }));

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

// This function will return a tensor's contents as a contiguous
// chunk. In some cases this will require copying the data. If that
// happens, 'contiguous_buffer' will be set to hold the contiguous
// chunk and 'cuda_copy' will be set to indicate whether CUDA copy is
// conducted.  The data copy can be avoided if the input is already in
// a contiguous chunk and the input is located in memory type and id
// specified.
Status
GetContiguousInputContent(
    const InferenceRequest::Input* rinput, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id, const char** content, size_t* content_byte_size,
    std::unique_ptr<AllocatedMemory>* contiguous_buffer, cudaStream_t stream,
    bool* cuda_copy)
{
  *cuda_copy = false;
  contiguous_buffer->reset();

  // Check input buffers to see if data copy is necessary
  MemoryReference input_buffers;
  size_t chunk_count = 0;
  bool type_mismatch = false;
  for (size_t idx = 0; idx < rinput->DataBufferCount(); ++idx) {
    TRITONSERVER_MemoryType src_memory_type;
    int64_t src_memory_type_id;
    size_t src_byte_size;
    const void* src_ptr;

    RETURN_IF_ERROR(rinput->DataBuffer(
        idx, &src_ptr, &src_byte_size, &src_memory_type, &src_memory_type_id));

    if (src_ptr != nullptr) {
      input_buffers.AddBuffer(
          (const char*)src_ptr, src_byte_size, src_memory_type,
          src_memory_type_id);
      chunk_count++;
      type_mismatch |=
          ((src_memory_type != memory_type) ||
           (src_memory_type_id != memory_type_id));
    }
  }

  if (chunk_count == 0) {
    *content = nullptr;
    *content_byte_size = 0;
  } else if ((chunk_count == 1) && !type_mismatch) {
    *content = input_buffers.BufferAt(
        0, content_byte_size, &memory_type, &memory_type_id);
  } else {
    contiguous_buffer->reset(new AllocatedMemory(
        input_buffers.TotalByteSize(), memory_type, memory_type_id));
    auto dst_ptr =
        (*contiguous_buffer)->MutableBuffer(&memory_type, &memory_type_id);
    if (dst_ptr == nullptr) {
      return Status(
          Status::Code::INTERNAL, "failed to allocate contiguous buffer");
    }

    size_t offset = 0;
    for (size_t i = 0; i < chunk_count; i++) {
      bool cuda_used;
      TRITONSERVER_MemoryType src_memory_type;
      int64_t src_memory_type_id;
      auto src_ptr = input_buffers.BufferAt(
          i, content_byte_size, &src_memory_type, &src_memory_type_id);
      RETURN_IF_ERROR(CopyBuffer(
          rinput->Name(), src_memory_type, src_memory_type_id, memory_type,
          memory_type_id, *content_byte_size, src_ptr, dst_ptr + offset, stream,
          &cuda_used));
      *cuda_copy |= cuda_used;
      offset += *content_byte_size;
    }

    *content = dst_ptr;
    *content_byte_size = (*contiguous_buffer)->TotalByteSize();
  }

  return Status::Success;
}

void
FillStringTensor(TRTISTF_Tensor* tensor, const size_t idx, const size_t cnt)
{
  for (size_t c = 0; c < cnt; ++c) {
    TRTISTF_TensorSetString(tensor, idx + c, nullptr, 0);
  }
}

bool
SetStringInputTensor(
    TRTISTF_Tensor* tensor, const InferenceRequest::Input* request_input,
    const size_t request_element_cnt, const size_t tensor_offset,
    std::unique_ptr<InferenceResponse>* response, cudaStream_t stream)
{
  bool cuda_copy = false;
  size_t element_idx = 0;

  // For string data type, we always need to have the data on CPU so
  // that we can read string length and construct the string
  // properly. So if the request's input tensor is not in CPU need to
  // copy it there.
  auto buffer_memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
  int64_t buffer_memory_type_id = 0;
  const char* content;
  size_t content_byte_size;

  std::unique_ptr<AllocatedMemory> contiguous_buffer;
  Status status = GetContiguousInputContent(
      request_input, buffer_memory_type, buffer_memory_type_id, &content,
      &content_byte_size, &contiguous_buffer, stream, &cuda_copy);
  if (!status.IsOk()) {
    InferenceResponse::SendWithStatus(std::move(*response), status);
    FillStringTensor(
        tensor, tensor_offset + element_idx, request_element_cnt - element_idx);
    return cuda_copy;
  }

#ifdef TRTIS_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream);
    cuda_copy = false;
  }
#endif  // TRTIS_ENABLE_GPU

  // Parse content and assign to 'tensor'. Each string in 'content'
  // is a 4-byte length followed by the string itself with no
  // null-terminator.
  while (content_byte_size >= sizeof(uint32_t)) {
    if (element_idx >= request_element_cnt) {
      InferenceResponse::SendWithStatus(
          std::move(*response),
          Status(
              Status::Code::INVALID_ARG,
              "unexpected number of string elements " +
                  std::to_string(element_idx + 1) + " for inference input '" +
                  request_input->Name() + "', expecting " +
                  std::to_string(request_element_cnt)));
      FillStringTensor(
          tensor, tensor_offset + element_idx,
          request_element_cnt - element_idx);
      return cuda_copy;
    }

    const uint32_t len = *(reinterpret_cast<const uint32_t*>(content));
    content += sizeof(uint32_t);
    content_byte_size -= sizeof(uint32_t);

    if (content_byte_size < len) {
      InferenceResponse::SendWithStatus(
          std::move(*response),
          Status(
              Status::Code::INVALID_ARG,
              "incomplete string data for inference input '" +
                  request_input->Name() + "', expecting string of length " +
                  std::to_string(len) + " but only " +
                  std::to_string(content_byte_size) + " bytes available"));
      FillStringTensor(
          tensor, tensor_offset + element_idx,
          request_element_cnt - element_idx);
      return cuda_copy;
    }

    TRTISTF_TensorSetString(tensor, tensor_offset + element_idx, content, len);
    content += len;
    content_byte_size -= len;
    element_idx++;
  }

  if ((*response != nullptr) && (element_idx != request_element_cnt)) {
    InferenceResponse::SendWithStatus(
        std::move(*response),
        Status(
            Status::Code::INTERNAL,
            "expected " + std::to_string(request_element_cnt) +
                " strings for inference input '" + request_input->Name() +
                "', got " + std::to_string(element_idx)));
    FillStringTensor(
        tensor, tensor_offset + element_idx, request_element_cnt - element_idx);
  }

  return cuda_copy;
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

void
BaseBackend::Context::Run(
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
              "null request given to TensorFlow runner for '" + name_ + "'"),
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

  // Collect the request inputs into contiguous input tensors. For
  // tensors with string data type we must handle ourselves since we
  // must use TF-specific string tensor APIs.
  bool cuda_copy = false;

  {
    BackendInputCollector collector(
        requests, &responses, enable_pinned_input_, stream_);

    for (const auto& pr : repr_input_request->ImmutableInputs()) {
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

      // The name of the input in the model can be different...
      const std::string* input_tensor_name = &input_name;
      const auto& tn_itr = input_name_map_.find(*input_tensor_name);
      if (tn_itr != input_name_map_.end()) {
        input_tensor_name = &tn_itr->second;
      }

      // Create a TF tensor to hold the entire input batch. Only try
      // to create a tensor on a specific device if 'input_device_id_'
      // is set. If unable to create the tensor then fail all
      // requests.
      TRTISTF_Tensor* tensor = TRTISTF_TensorNew(
          input_tensor_name->c_str(), ConvertDataType(datatype),
          batchn_shape.size(),
          (batchn_shape.size() == 0) ? nullptr : &batchn_shape[0],
          input_device_id_);
      if (tensor == nullptr) {
        Status status = Status(
            Status::Code::INTERNAL,
            "failed to create input tensor '" + input_name + "' with shape " +
                DimsListToString(batchn_shape) + " and data type " +
                DataType_Name(datatype) + " for '" + name_ + "'");

        FAIL_ALL_AND_RETURN_IF_ERROR(
            requests, responses, status,
            "error creating TensorFlow input tensor");
      }

      // Add the new TF tensor to the list of TF inputs.
      TRTISTF_TensorList* tlink = TRTISTF_TensorListNew(tensor, *input_tensors);
      *input_tensors = tlink;

      // Custom handling for string/bytes tensor...
      if (datatype == DataType::TYPE_STRING) {
        size_t tensor_offset = 0;
        const size_t batch1_element_cnt = GetElementCount(batch1_shape);

        for (size_t idx = 0; idx < requests.size(); idx++) {
          auto& request = requests[idx];
          auto& response = responses[idx];

          const size_t request_element_cnt =
              std::max(1U, request->BatchSize()) * batch1_element_cnt;

          const InferenceRequest::Input* request_input;
          Status status = request->ImmutableInput(input_name, &request_input);
          if (!status.IsOk() && (response != nullptr)) {
            InferenceResponse::SendWithStatus(std::move(response), status);
          }

          cuda_copy |= SetStringInputTensor(
              tensor, request_input, request_element_cnt, tensor_offset,
              &response, stream_);

          tensor_offset += request_element_cnt;
        }
      }
      // Use the collector for non-STRING datatype...
      else {  // datatype != DataType::TYPE_STRING
        collector.ProcessTensor(
            input_name, datatype, batch1_shape, TRTISTF_TensorData(tensor),
            TRTISTF_TensorDataByteSize(tensor),
            (TRTISTF_TensorIsGPUTensor(tensor)) ? TRITONSERVER_MEMORY_GPU
                                                : TRITONSERVER_MEMORY_CPU,
            (TRTISTF_TensorIsGPUTensor(tensor)) ? gpu_device_ : 0);
      }
    }

    // Finalize...
    cuda_copy |= collector.Finalize();
  }

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

  // Wait for any in-flight input tensor copies to complete.
#ifdef TRTIS_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif

  INFER_STATS_DECL_TIMESTAMP(compute_input_end_ns);

  // Run. Session will update the 'output_tensors'.
  std::unique_ptr<TRTISTF_TensorList, decltype(&TRTISTF_TensorListDelete)>
      output_tensors(nullptr, TRTISTF_TensorListDelete);

  {
    TRTISTF_TensorList* rtl = nullptr;

    TRTISTF_Error* err = TRTISTF_ModelRun(
        trtistf_model_.get(), *(input_tensors.release()),
        required_outputs.size(), output_names_cstr, &rtl);
    if (err != nullptr) {
      auto status = Status(Status::Code::INTERNAL, err->msg_);
      TRTISTF_ErrorDelete(err);
      // Something went wrong with the entire batch inference. For
      // every response that has not already been sent with an
      // error... send it now...
      FAIL_ALL_AND_RETURN_IF_ERROR(
          requests, responses, status, "error sending TensorFlow response");
    }

    output_tensors.reset(rtl);
  }

  INFER_STATS_DECL_TIMESTAMP(compute_output_start_ns);

  // Create the response tensors and copy the appropriate tensor data
  // into each. For tensors with string data type we must handle
  // ourselves since we must use TF-specific string tensor APIs.
  cuda_copy = false;

  {
    BackendResponder responder(
        requests, &responses, max_batch_size_, enable_pinned_output_, stream_);

    TRTISTF_TensorList* output_tensor_itr = output_tensors.get();
    for (const auto& name : model_output_names) {
      TRTISTF_Tensor* output_tensor = output_tensor_itr->tensor_;

      TRTISTF_DataType tf_datatype = TRTISTF_TensorDataType(output_tensor);
      TRTISTF_Shape* tf_shape = TRTISTF_TensorShape(output_tensor);

      const DataType datatype = ConvertDataType(tf_datatype);

      // batchn_shape holds the shape of the entire tensor batch, but
      // is overwritten below and used as the shape for each response
      // output.
      std::vector<int64_t> batchn_shape;
      batchn_shape.reserve(tf_shape->rank_);
      for (size_t itr = 0; itr < tf_shape->rank_; itr++) {
        const int64_t dim = tf_shape->dims_[itr];
        batchn_shape.push_back(dim);
      }

      // Custom handling for string/bytes tensor...
      if (datatype == DataType::TYPE_STRING) {
        size_t tensor_offset = 0;

        for (size_t idx = 0; idx < responses.size(); idx++) {
          auto& request = requests[idx];
          auto& response = responses[idx];

          if (max_batch_size_ != NO_BATCHING) {
            batchn_shape[0] = request->BatchSize();
          }

          const size_t tensor_element_cnt = GetElementCount(batchn_shape);

          // Only need an response tensor for requested outputs.
          if ((response != nullptr) &&
              (request->ImmutableRequestedOutputs().find(name) !=
               request->ImmutableRequestedOutputs().end())) {
            InferenceResponse::Output* response_output = nullptr;
            response->AddOutput(
                name, datatype, batchn_shape, request->BatchSize(),
                &response_output);
            cuda_copy |= SetStringOutputBuffer(
                output_tensor, &response, response_output, tensor_element_cnt,
                tensor_offset, stream_);
          }

          tensor_offset += tensor_element_cnt;
        }
      }
      // Use the responder for non-STRING datatype...
      else {  // datatype != DataType::TYPE_STRING
        responder.ProcessTensor(
            name, datatype, batchn_shape, TRTISTF_TensorData(output_tensor),
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
  INFER_STATS_DECL_TIMESTAMP(compute_end_ns);

  // Report stats and trace
  for (size_t i = 0; i < requests.size(); ++i) {
    auto& request = requests[i];
    request->ReportStatistics(
        (responses[i] != nullptr), compute_start_ns, compute_input_end_ns,
        compute_output_start_ns, compute_end_ns);

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
      total_batch_size, compute_start_ns, compute_input_end_ns,
      compute_output_start_ns, compute_end_ns);
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
