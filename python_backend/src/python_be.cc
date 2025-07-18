// Copyright 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "python_be.h"

#include <filesystem>

#include "correlation_id.h"
#include "gpu_buffers.h"
#include "infer_payload.h"
#include "model_loader.h"
#include "pb_log.h"

namespace triton { namespace backend { namespace python {

namespace bi = boost::interprocess;

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      stub_to_parent_thread_(false)
{
}

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::CheckIncomingRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    size_t& total_batch_size)
{
  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  int max_batch_size = model_state->MaxBatchSize();

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.
  total_batch_size = 0;
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          std::string(
              "null request given to Python backend for '" + Name() + "'")
              .c_str());
    }
  }

  for (size_t i = 0; i < request_count; i++) {
    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
      }
      if (err != nullptr) {
        return err;
      }
    } else {
      ++total_batch_size;
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return nullptr;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string(
            "batch size " + std::to_string(total_batch_size) + " for '" +
            Name() + "', max allowed is " + std::to_string(max_batch_size))
            .c_str());
  }

  return nullptr;  // success
}

bool
ModelInstanceState::ExistsInClosedRequests(intptr_t closed_request)
{
  std::lock_guard<std::mutex> guard{closed_requests_mutex_};
  return std::find(
             closed_requests_.begin(), closed_requests_.end(),
             closed_request) != closed_requests_.end();
}

void
ModelInstanceState::SetErrorForResponseSendMessage(
    ResponseSendMessage* response_send_message,
    std::shared_ptr<TRITONSERVER_Error*> error,
    std::unique_ptr<PbString>& error_message)
{
  if (error && *error != nullptr) {
    response_send_message->has_error = true;
    LOG_IF_EXCEPTION(
        error_message = PbString::Create(
            Stub()->ShmPool(), TRITONSERVER_ErrorMessage(*error)));
    response_send_message->error = error_message->ShmHandle();
    response_send_message->is_error_set = true;
  }
}

bool
ModelInstanceState::IsStubProcessAlive()
{
  boost::posix_time::ptime timeout =
      boost::get_system_time() + boost::posix_time::seconds(1);
  bi::scoped_lock<bi::interprocess_mutex> lock(*Stub()->HealthMutex(), timeout);

  // Check if lock has been acquired.
  if (lock) {
    return Stub()->IpcControl()->stub_health;
  } else {
    // If It failed to obtain the lock, it means that the stub has been
    // stuck or exited while holding the health mutex lock.
    return false;
  }
}

TRITONSERVER_Error*
ModelInstanceState::SaveRequestsToSharedMemory(
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<std::unique_ptr<InferRequest>>& pb_infer_requests,
    AllocatedSharedMemory<char>& request_batch,
    std::shared_ptr<std::vector<TRITONBACKEND_Response*>>& responses)
{
  // Clear any existing items in the requests vector
  pb_infer_requests.clear();

  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  RETURN_IF_EXCEPTION(
      request_batch = Stub()->ShmPool()->Construct<char>(
          sizeof(RequestBatch) +
          request_count * sizeof(bi::managed_external_buffer::handle_t)));

  RequestBatch* request_batch_shm_ptr =
      reinterpret_cast<RequestBatch*>(request_batch.data_.get());
  request_batch_shm_ptr->batch_size = request_count;

  bi::managed_external_buffer::handle_t* requests_shm =
      reinterpret_cast<bi::managed_external_buffer::handle_t*>(
          request_batch.data_.get() + sizeof(RequestBatch));

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    uint32_t requested_input_count = 0;
    RETURN_IF_ERROR(
        TRITONBACKEND_RequestInputCount(request, &requested_input_count));

    uint32_t requested_output_count = 0;
    RETURN_IF_ERROR(
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));

    std::vector<std::shared_ptr<PbTensor>> pb_input_tensors;
    for (size_t iidx = 0; iidx < requested_input_count; ++iidx) {
      std::shared_ptr<PbTensor> pb_input_tensor;

      RETURN_IF_ERROR(
          GetInputTensor(iidx, pb_input_tensor, request, responses));
      pb_input_tensors.emplace_back(std::move(pb_input_tensor));
    }

    std::set<std::string> requested_output_names;
    // Append the list of requested outputs to the inference_request
    for (size_t iidx = 0; iidx < requested_output_count; ++iidx) {
      const char* requested_output_name;
      RETURN_IF_ERROR(TRITONBACKEND_RequestOutputName(
          request, iidx, &requested_output_name));
      requested_output_names.emplace(requested_output_name);
    }

    triton::common::TritonJson::Value parameters_json(
        triton::common::TritonJson::ValueType::OBJECT);
    uint32_t parameter_count;
    RETURN_IF_ERROR(
        TRITONBACKEND_RequestParameterCount(request, &parameter_count));
    for (size_t i = 0; i < parameter_count; i++) {
      const char* name;
      TRITONSERVER_ParameterType type;
      const void* vvalue;
      RETURN_IF_ERROR(
          TRITONBACKEND_RequestParameter(request, i, &name, &type, &vvalue));
      if (type == TRITONSERVER_PARAMETER_INT) {
        RETURN_IF_ERROR(parameters_json.AddInt(
            name, *(reinterpret_cast<const int64_t*>(vvalue))));
      } else if (type == TRITONSERVER_PARAMETER_BOOL) {
        RETURN_IF_ERROR(parameters_json.AddBool(
            name, *(reinterpret_cast<const bool*>(vvalue))));
      } else if (type == TRITONSERVER_PARAMETER_STRING) {
        std::string string = reinterpret_cast<const char*>(vvalue);
        RETURN_IF_ERROR(parameters_json.AddString(name, string));
      } else if (type == TRITONSERVER_PARAMETER_DOUBLE) {
        RETURN_IF_ERROR(parameters_json.AddDouble(
            name, *(reinterpret_cast<const double*>(vvalue))));
      } else {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("Unsupported parameter type for parameter '") + name +
             "'.")
                .c_str());
      }
    }

    triton::common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(parameters_json.Write(&buffer));
    const auto& parameters_string = buffer.Contents();

    // request id
    const char* id;
    RETURN_IF_ERROR(TRITONBACKEND_RequestId(request, &id));

    uint64_t correlation_id_uint = 0;
    CorrelationId correlation_id;

    auto error =
        TRITONBACKEND_RequestCorrelationId(request, &correlation_id_uint);
    if (error != nullptr) {
      TRITONSERVER_ErrorDelete(error);
      const char* correlation_id_string = "";
      RETURN_IF_ERROR(TRITONBACKEND_RequestCorrelationIdString(
          request, &correlation_id_string));
      correlation_id = CorrelationId(std::string(correlation_id_string));
    } else {
      correlation_id = CorrelationId(correlation_id_uint);
    }

    uint32_t flags;
    RETURN_IF_ERROR(TRITONBACKEND_RequestFlags(request, &flags));

    // Do not return if error in this case, because Triton core
    // will return an error if tracing is disabled (see PYBE PR#295).
    // For the same reason, we do not log the error message, otherwise
    // when Triton is compiled without tracing, it'll constantly log
    // this error.
    TRITONSERVER_InferenceTrace* triton_trace;
    auto err = TRITONBACKEND_RequestTrace(request, &triton_trace);
    if (err != nullptr) {
      triton_trace = nullptr;
      TRITONSERVER_ErrorDelete(err);
    }
    const char* val = nullptr;
    if (triton_trace != nullptr) {
      LOG_IF_ERROR(
          TRITONSERVER_InferenceTraceContext(triton_trace, &val),
          "failed to retrieve trace context");
    }
    std::string context = (val != nullptr) ? std::string(val) : "";

    InferenceTrace trace =
        InferenceTrace(reinterpret_cast<void*>(triton_trace), context);

    uint64_t request_timeout;
    RETURN_IF_ERROR(TRITONBACKEND_InferenceRequestTimeoutMicroseconds(
        request, &request_timeout));

    std::unique_ptr<InferRequest> infer_request;
    TRITONBACKEND_ResponseFactory* factory_ptr = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseFactoryNew(&factory_ptr, request));

    infer_request = std::make_unique<InferRequest>(
        id, correlation_id, pb_input_tensors, requested_output_names,
        model_state->Name(), model_state->Version(), parameters_string, flags,
        request_timeout, reinterpret_cast<intptr_t>(factory_ptr),
        reinterpret_cast<intptr_t>(request),
        PreferredMemory(PreferredMemory::kDefault, 0), trace);
    RETURN_IF_EXCEPTION(infer_request->SaveToSharedMemory(Stub()->ShmPool()));
    requests_shm[r] = infer_request->ShmHandle();
    pb_infer_requests.emplace_back(std::move(infer_request));
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::LaunchStubProcess()
{
  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  Stub() = std::make_unique<StubLauncher>(
      "MODEL_INSTANCE_STUB", Name(), DeviceId(),
      TRITONSERVER_InstanceGroupKindString(Kind()));
  RETURN_IF_ERROR(Stub()->Initialize(model_state));
  RETURN_IF_ERROR(Stub()->Setup());
  StartMonitor();
  RETURN_IF_ERROR(Stub()->Launch());

  thread_pool_ = std::make_unique<boost::asio::thread_pool>(
      model_state->StateForBackend()->thread_pool_size);

  request_executor_ = std::make_unique<RequestExecutor>(
      Stub()->ShmPool(), model_state->TritonServer());

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::GetInputTensor(
    const uint32_t input_idx, std::shared_ptr<PbTensor>& input_tensor,
    TRITONBACKEND_Request* request,
    std::shared_ptr<std::vector<TRITONBACKEND_Response*>>& responses)
{
  NVTX_RANGE(nvtx_, "GetInputTensor " + Name());
  const char* input_name;
  // Load iidx'th input name
  RETURN_IF_ERROR(
      TRITONBACKEND_RequestInputName(request, input_idx, &input_name));

  // Load iidx'th input
  TRITONBACKEND_Input* in;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, input_name, &in));

  // Load input properties
  TRITONSERVER_DataType input_dtype;
  const int64_t* input_shape;
  uint32_t input_dims_count;
  uint64_t input_byte_size;
  uint32_t input_buffer_count;

  RETURN_IF_ERROR(TRITONBACKEND_InputPropertiesForHostPolicy(
      in, HostPolicyName().c_str(), &input_name, &input_dtype, &input_shape,
      &input_dims_count, &input_byte_size, &input_buffer_count));

  // Only use input collector when a response array is provided.
  std::unique_ptr<BackendInputCollector> collector;
  if (responses) {
    collector = std::make_unique<BackendInputCollector>(
        &request, 1, responses.get(), Model()->TritonMemoryManager(),
        false /* pinned_enable */, CudaStream(), nullptr, nullptr, 0,
        HostPolicyName().c_str());
  }

  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  bool cpu_only_tensors = model_state->ForceCPUOnlyInputTensors();

  if (input_dtype == TRITONSERVER_TYPE_BYTES) {
    cpu_only_tensors = true;
  }

#ifdef TRITON_ENABLE_GPU
  CUDAHandler& cuda_handler = CUDAHandler::getInstance();
  // If CUDA driver API is not available, the input tensors will be moved to
  // CPU.
  if (!cuda_handler.IsAvailable() && !cpu_only_tensors) {
    if (!cuda_handler.GetErrorString().empty()) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN, (std::string(
                                      "Forcing CPU only input tensors: " +
                                      cuda_handler.GetErrorString()))
                                     .c_str());
    }
    cuda_handler.ClearErrorString();
    cpu_only_tensors = true;
  }
#endif

  TRITONSERVER_MemoryType src_memory_type;
  int64_t src_memory_type_id;
  uint64_t src_byte_size;
  const void* src_ptr;
  RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
      in, 0 /* input buffer index */, &src_ptr, &src_byte_size,
      &src_memory_type, &src_memory_type_id));

// If TRITON_ENABLE_GPU is false, we need to copy the tensors
// to the CPU.
#ifndef TRITON_ENABLE_GPU
  cpu_only_tensors = true;
#endif  // TRITON_ENABLE_GPU

  if (cpu_only_tensors || src_memory_type != TRITONSERVER_MEMORY_GPU) {
    input_tensor = std::make_shared<PbTensor>(
        std::string(input_name),
        std::vector<int64_t>(input_shape, input_shape + input_dims_count),
        input_dtype, TRITONSERVER_MEMORY_CPU /* memory_type */,
        0 /* memory_type_id */, nullptr /* buffer ptr*/, input_byte_size,
        nullptr /* DLManagedTensor */);
    RETURN_IF_EXCEPTION(input_tensor->SaveToSharedMemory(
        Stub()->ShmPool(), false /* copy_gpu */));
    char* input_buffer = reinterpret_cast<char*>(input_tensor->DataPtr());

    if (collector) {
      collector->ProcessTensor(
          input_name, input_buffer, input_byte_size,
          TRITONSERVER_MEMORY_CPU /* memory_type */, 0 /* memory_type_id */);
    } else {
      size_t byte_size = input_byte_size;
      RETURN_IF_ERROR(backend::ReadInputTensor(
          request, input_name, input_buffer, &byte_size));
    }

    if (input_dtype == TRITONSERVER_TYPE_BYTES) {
      const char* content = reinterpret_cast<char*>(input_tensor->DataPtr());
      size_t content_byte_size = input_tensor->ByteSize();
      int64_t request_element_cnt = 0;
      RETURN_IF_ERROR(
          GetElementCount(input_tensor->Dims(), &request_element_cnt));
      RETURN_IF_ERROR(ValidateStringBuffer(
          content, content_byte_size, request_element_cnt, input_name,
          nullptr /* str_list */));
    }
  } else {
#ifdef TRITON_ENABLE_GPU
    // Attempt to use the cuda shared memory pool for GPU tensor.
    ShareCUDAMemoryPool(src_memory_type_id);

    // Retrieving GPU input tensors
    const void* buffer = nullptr;
    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> alloc_perference;
    alloc_perference = {{TRITONSERVER_MEMORY_GPU, src_memory_type_id}};

    // collector is used in the non-decoupled mode.
    if (collector) {
      // The ProcessTensor function will try to allocate the buffer in the CUDA
      // pool first.
      RETURN_IF_ERROR(collector->ProcessTensor(
          input_name, nullptr, 0, alloc_perference,
          reinterpret_cast<const char**>(&buffer), &input_byte_size,
          &src_memory_type, &src_memory_type_id));
      // If the tensor is using the cuda shared memory, we need to extract the
      // handle that was used to create the device pointer. This is because of a
      // limitation in the legacy CUDA IPC API that doesn't allow getting the
      // handle of an exported pointer. If the cuda handle exists, it indicates
      // that the cuda shared memory was used and the input is in a single
      // buffer.
      // [FIXME] For the case where the input is in cuda shared memory and uses
      // multiple input buffers this needs to be changed.
      TRITONSERVER_BufferAttributes* buffer_attributes;

      // This value is not used.
      const void* buffer_p;
      RETURN_IF_ERROR(TRITONBACKEND_InputBufferAttributes(
          in, 0, &buffer_p, &buffer_attributes));

      input_tensor = std::make_shared<PbTensor>(
          std::string(input_name),
          std::vector<int64_t>(input_shape, input_shape + input_dims_count),
          input_dtype, src_memory_type, src_memory_type_id,
          const_cast<void*>(buffer), input_byte_size,
          nullptr /* DLManagedTensor */);

      cudaIpcMemHandle_t* cuda_ipc_handle;
      RETURN_IF_ERROR(TRITONSERVER_BufferAttributesCudaIpcHandle(
          buffer_attributes, reinterpret_cast<void**>(&cuda_ipc_handle)));
      if (cuda_ipc_handle != nullptr) {
        RETURN_IF_EXCEPTION(input_tensor->SaveToSharedMemory(
            Stub()->ShmPool(), false /* copy_gpu */));
        RETURN_IF_EXCEPTION(
            input_tensor->Memory()->SetCudaIpcHandle(cuda_ipc_handle));
      } else {
        RETURN_IF_EXCEPTION(input_tensor->SaveToSharedMemory(
            Stub()->ShmPool(), true /* copy_gpu */));
      }
    } else {
      // Try to use the cuda shared memory pool first.
      void* dev_ptr;
      BackendMemory* backend_memory;
      std::unique_ptr<BackendMemory> lbackend_memory;
      RETURN_IF_ERROR(BackendMemory::Create(
          reinterpret_cast<TRITONBACKEND_MemoryManager*>(
              Stub()
                  ->ShmPool()
                  ->GetCUDAMemoryPoolManager()
                  ->TritonMemoryManager()),
          {BackendMemory::AllocationType::GPU_POOL,
           BackendMemory::AllocationType::GPU},
          src_memory_type_id, input_byte_size, &backend_memory));

      dev_ptr = backend_memory->MemoryPtr();
      lbackend_memory.reset(backend_memory);

      size_t byte_size = input_byte_size;

      bool cuda_used = false;
      RETURN_IF_ERROR(backend::ReadInputTensor(
          request, input_name, reinterpret_cast<char*>(dev_ptr), &byte_size,
          TRITONSERVER_MEMORY_GPU, src_memory_type_id, CudaStream(),
          &cuda_used));

      if (cuda_used) {
#ifdef TRITON_ENABLE_GPU
        cudaStreamSynchronize(stream_);
#endif
      }

      input_tensor = std::make_shared<PbTensor>(
          std::string(input_name),
          std::vector<int64_t>(input_shape, input_shape + input_dims_count),
          input_dtype, src_memory_type, src_memory_type_id,
          const_cast<void*>(dev_ptr), input_byte_size,
          nullptr /* DLManagedTensor */);

      input_tensor->SetMemory(std::move(
          PbMemory::Create(Stub()->ShmPool(), std::move(lbackend_memory))));

      RETURN_IF_EXCEPTION(input_tensor->SaveToSharedMemory(
          Stub()->ShmPool(), true /* copy_gpu */));
    }
#else
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Python backend does not support GPU tensors.");
#endif  // TRITON_ENABLE_GPU
  }

  return nullptr;
}

void
ModelInstanceState::ExecuteBLSRequest(
    std::shared_ptr<IPCMessage> ipc_message, const bool is_decoupled)
{
  bool is_response_batch_set = false;
  std::unique_ptr<InferResponse> infer_response;
  ResponseBatch* response_batch = nullptr;
  std::unique_ptr<PbString> pb_error_message;
  std::unique_ptr<IPCMessage> bls_response;
  AllocatedSharedMemory<char> response_batch_shm;
  bi::managed_external_buffer::handle_t* response_handle = nullptr;

  try {
    bls_response =
        IPCMessage::Create(Stub()->ShmPool(), false /* inline_response */);

    AllocatedSharedMemory<char> request_batch =
        Stub()->ShmPool()->Load<char>(ipc_message->Args());
    RequestBatch* request_batch_shm_ptr =
        reinterpret_cast<RequestBatch*>(request_batch.data_.get());

    bls_response->Command() = PYTHONSTUB_InferExecResponse;
    ipc_message->ResponseHandle() = bls_response->ShmHandle();

    // The response batch of the handle will contain a ResponseBatch
    PrepareResponseBatch(
        &response_batch, response_batch_shm, &bls_response, &response_handle);

    is_response_batch_set = true;
    bool has_gpu_tensor = false;
    GPUBuffersHelper gpu_buffer_helper;

    PythonBackendException pb_exception(std::string{});
    if (request_batch_shm_ptr->batch_size == 1) {
      std::shared_ptr<InferRequest> infer_request;
      bi::managed_external_buffer::handle_t* request_handle =
          reinterpret_cast<bi::managed_external_buffer::handle_t*>(
              request_batch.data_.get() + sizeof(RequestBatch));
      infer_request = InferRequest::LoadFromSharedMemory(
          Stub()->ShmPool(), *request_handle, false /* open_cuda_handle */,
          nullptr /* is_model_decoupled */);

      // If the BLS inputs are in GPU an additional round trip between the
      // stub process and the main process is required. The reason is that we
      // need to first allocate the GPU memory from the memory pool and then
      // ask the stub process to fill in those allocated buffers.
      try {
        for (auto& input_tensor : infer_request->Inputs()) {
          if (!input_tensor->IsCPU()) {
#ifdef TRITON_ENABLE_GPU
            // Attempt to use the cuda shared memory pool for GPU tensor.
            ShareCUDAMemoryPool(input_tensor->MemoryTypeId());
            BackendMemory* backend_memory;
            std::unique_ptr<BackendMemory> lbackend_memory;
            has_gpu_tensor = true;
            TRITONSERVER_Error* error = BackendMemory::Create(
                Model()->TritonMemoryManager(),
                {BackendMemory::AllocationType::GPU_POOL,
                 BackendMemory::AllocationType::GPU},
                input_tensor->MemoryTypeId(), input_tensor->ByteSize(),
                &backend_memory);
            if (error != nullptr) {
              LOG_MESSAGE(
                  TRITONSERVER_LOG_ERROR, TRITONSERVER_ErrorMessage(error));
              break;
            }
            lbackend_memory.reset(backend_memory);
            input_tensor->SetMemory(std::move(PbMemory::Create(
                Stub()->ShmPool(), std::move(lbackend_memory))));
            gpu_buffer_helper.AddBuffer(input_tensor->Memory()->ShmHandle());
#endif  // TRITON_ENABLE_GPU
          }
        }
      }
      catch (const PythonBackendException& exception) {
        gpu_buffer_helper.SetError(Stub()->ShmPool(), exception.what());
        pb_exception = exception;
      }

      // Wait for the extra round trip to complete. The stub process will fill
      // in the data for the GPU tensors. If there is an error, the extra round
      // trip must be still completed, otherwise the stub process will always be
      // waiting for a message from the parent process.
      if (has_gpu_tensor) {
        gpu_buffer_helper.Complete(Stub()->ShmPool());
        request_batch_shm_ptr->gpu_buffers_handle =
            gpu_buffer_helper.ShmHandle();

        bi::scoped_lock<bi::interprocess_mutex> lock{
            *(ipc_message->ResponseMutex())};
        ipc_message->ResponseCondition()->notify_all();
        ipc_message->ResponseCondition()->wait(lock);
      }

      if (pb_exception.what() == std::string{""}) {
        auto callback = std::bind(
            &ModelInstanceState::SendBLSDecoupledResponse, this,
            std::placeholders::_1);
        std::shared_ptr<InferPayload> infer_payload =
            std::make_shared<InferPayload>(is_decoupled, callback);

        auto response_future =
            request_executor_->Infer(infer_request, infer_payload);
        infer_response = response_future.get();

        if (is_decoupled && (infer_response->Id() != nullptr)) {
          // Need to manage the lifetime of InferPayload object for bls
          // decoupled responses.
          std::lock_guard<std::mutex> lock(infer_payload_mu_);
          infer_payload_[reinterpret_cast<intptr_t>(infer_payload.get())] =
              infer_payload;
        }

        PrepareResponseHandle(&infer_response, response_handle);
      } else {
        throw pb_exception;
      }
    }
  }
  catch (const PythonBackendException& pb_exception) {
    if (is_response_batch_set) {
      response_batch->has_error = true;
      LOG_IF_EXCEPTION(
          pb_error_message =
              PbString::Create(Stub()->ShmPool(), pb_exception.what()));

      if (pb_error_message != nullptr) {
        response_batch->is_error_set = true;
        response_batch->error = pb_error_message->ShmHandle();
      }
    } else {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, pb_exception.what());
    }
  }

  // At this point, the stub has notified the parent process that it has
  // finished loading the inference response from shared memory.
  {
    bi::scoped_lock<bi::interprocess_mutex> lock{
        *(ipc_message->ResponseMutex())};
    ipc_message->ResponseCondition()->notify_all();
    ipc_message->ResponseCondition()->wait(lock);
  }
}

void
ModelInstanceState::StubToParentMQMonitor()
{
  while (stub_to_parent_thread_) {
    bi::managed_external_buffer::handle_t handle =
        Stub()->StubToParentMessageQueue()->Pop();
    if (handle == DUMMY_MESSAGE) {
      break;
    }
    std::unique_ptr<IPCMessage> message =
        IPCMessage::LoadFromSharedMemory(Stub()->ShmPool(), handle);

    switch (message->Command()) {
      case PYTHONSTUB_LogRequest: {
        ProcessLogRequest(message);
        break;
      }
      case PYTHONSTUB_BLSDecoupledInferPayloadCleanup:
      case PYTHONSTUB_DecoupledResponseFactoryCleanup: {
        ProcessCleanupRequest(message);
        break;
      }
      case PYTHONSTUB_IsRequestCancelled: {
        ProcessIsRequestCancelled(message);
        break;
      }
      case PYTHONSTUB_MetricFamilyRequestNew:
      case PYTHONSTUB_MetricFamilyRequestDelete: {
        ProcessMetricFamilyRequest(message);
        break;
      }
      case PYTHONSTUB_MetricRequestNew:
      case PYTHONSTUB_MetricRequestDelete:
      case PYTHONSTUB_MetricRequestValue:
      case PYTHONSTUB_MetricRequestIncrement:
      case PYTHONSTUB_MetricRequestSet:
      case PYTHONSTUB_MetricRequestObserve: {
        ProcessMetricRequest(message);
        break;
      }
      case PYTHONSTUB_ModelReadinessRequest:
      case PYTHONSTUB_LoadModelRequest:
      case PYTHONSTUB_UnloadModelRequest: {
        ProcessModelControlRequest(message);
        break;
      }
      case PYTHONSTUB_ResponseSend: {
        std::shared_ptr<IPCMessage> response_send_message = std::move(message);
        std::packaged_task<void()> task([this, response_send_message] {
          ResponseSendDecoupled(response_send_message);
        });
        boost::asio::post(*thread_pool_, std::move(task));
        break;
      }
      case PYTHONSTUB_InferExecRequest:
      case PYTHONSTUB_InferStreamExecRequest: {
        std::shared_ptr<IPCMessage> bls_execute = std::move(message);
        std::packaged_task<void()> task([this, bls_execute] {
          ExecuteBLSRequest(
              bls_execute,
              (bls_execute->Command() == PYTHONSTUB_InferStreamExecRequest));
        });
        boost::asio::post(*thread_pool_, std::move(task));
        break;
      }
      case PYTHONSTUB_CancelBLSInferRequest: {
        ProcessCancelBLSRequest(message);
        break;
      }
      default: {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR, "Unexpected message type received.");
        break;
      }
    }
  }
}

void
ModelInstanceState::ProcessLogRequest(
    const std::unique_ptr<IPCMessage>& message)
{
  AllocatedSharedMemory<LogSendMessage> log_message_response =
      Stub()->ShmPool()->Load<LogSendMessage>(message->Args());
  std::unique_ptr<PbLog> pb_log_message =
      PbLogShm::LoadFromSharedMemory(Stub()->ShmPool(), message->Args());

  const std::string& filename = pb_log_message->Filename();
  uint32_t line = pb_log_message->Line();
  const std::string& log_message = pb_log_message->Message();
  LogLevel level = pb_log_message->Level();

  switch (level) {
    case LogLevel::kInfo: {
      TRITONSERVER_LogMessage(
          TRITONSERVER_LOG_INFO, (filename.c_str()), line,
          (log_message.c_str()));
      break;
    }
    case LogLevel::kWarning: {
      TRITONSERVER_LogMessage(
          TRITONSERVER_LOG_WARN, (filename.c_str()), line,
          (log_message.c_str()));
      break;
    }
    case LogLevel::kError: {
      TRITONSERVER_LogMessage(
          TRITONSERVER_LOG_ERROR, (filename.c_str()), line,
          (log_message.c_str()));
      break;
    }
    case LogLevel::kVerbose: {
      TRITONSERVER_LogMessage(
          TRITONSERVER_LOG_VERBOSE, (filename.c_str()), line,
          (log_message.c_str()));
      break;
    }
  }
  // Send confirmation back to pb_stub.cc that the message
  // was received.
  LogSendMessage* send_message_payload =
      reinterpret_cast<LogSendMessage*>(log_message_response.data_.get());
  {
    bi::scoped_lock<bi::interprocess_mutex> guard{send_message_payload->mu};
    send_message_payload->waiting_on_stub = true;
    send_message_payload->cv.notify_all();
    while (send_message_payload->waiting_on_stub) {
      send_message_payload->cv.wait(guard);
    }
  }
}

void
ModelInstanceState::ProcessCleanupRequest(
    const std::unique_ptr<IPCMessage>& message)
{
  AllocatedSharedMemory<char> cleanup_request_message =
      Stub()->ShmPool()->Load<char>(message->Args());
  CleanupMessage* cleanup_message_ptr =
      reinterpret_cast<CleanupMessage*>(cleanup_request_message.data_.get());
  intptr_t id = reinterpret_cast<intptr_t>(cleanup_message_ptr->id);
  if (message->Command() == PYTHONSTUB_BLSDecoupledInferPayloadCleanup) {
    // Remove the InferPayload object from the map.
    std::lock_guard<std::mutex> lock(infer_payload_mu_);
    infer_payload_.erase(id);
  } else if (message->Command() == PYTHONSTUB_DecoupledResponseFactoryCleanup) {
    // Delete response factory
    std::unique_ptr<
        TRITONBACKEND_ResponseFactory, backend::ResponseFactoryDeleter>
        response_factory(reinterpret_cast<TRITONBACKEND_ResponseFactory*>(id));
  }

  {
    bi::scoped_lock<bi::interprocess_mutex> lock{*(message->ResponseMutex())};
    cleanup_message_ptr->waiting_on_stub = true;
    message->ResponseCondition()->notify_all();
  }
}

void
ModelInstanceState::ProcessCancelBLSRequest(
    const std::unique_ptr<IPCMessage>& message)
{
  AllocatedSharedMemory<CancelBLSRequestMessage> message_shm =
      Stub()->ShmPool()->Load<CancelBLSRequestMessage>(message->Args());
  CancelBLSRequestMessage* message_payload =
      reinterpret_cast<CancelBLSRequestMessage*>(message_shm.data_.get());

  {
    bi::scoped_lock<bi::interprocess_mutex> lk{message_payload->mu};

    intptr_t id = reinterpret_cast<intptr_t>(message_payload->infer_payload_id);
    try {
      {
        std::lock_guard<std::mutex> lock(infer_payload_mu_);
        if (infer_payload_.find(id) != infer_payload_.end()) {
          infer_payload_[id]->SafeCancelRequest();
        }
      }
      message_payload->is_cancelled = true;
    }
    catch (const PythonBackendException& pb_exception) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, pb_exception.what());
    }

    message_payload->waiting_on_stub = true;
    message_payload->cv.notify_all();
    while (message_payload->waiting_on_stub) {
      message_payload->cv.wait(lk);
    }
  }
}

void
ModelInstanceState::ProcessIsRequestCancelled(
    const std::unique_ptr<IPCMessage>& message)
{
  AllocatedSharedMemory<IsCancelledMessage> message_shm =
      Stub()->ShmPool()->Load<IsCancelledMessage>(message->Args());
  IsCancelledMessage* message_payload =
      reinterpret_cast<IsCancelledMessage*>(message_shm.data_.get());

  {
    bi::scoped_lock<bi::interprocess_mutex> lk{message_payload->mu};

    if (message_payload->response_factory_address != 0) {
      TRITONBACKEND_ResponseFactory* response_factory =
          reinterpret_cast<TRITONBACKEND_ResponseFactory*>(
              message_payload->response_factory_address);
      TRITONBACKEND_ResponseFactoryIsCancelled(
          response_factory, &message_payload->is_cancelled);
    } else if (message_payload->request_address != 0) {
      TRITONBACKEND_Request* request = reinterpret_cast<TRITONBACKEND_Request*>(
          message_payload->request_address);
      TRITONBACKEND_RequestIsCancelled(request, &message_payload->is_cancelled);
    } else {
      throw PythonBackendException("Cannot determine request cancellation");
    }

    message_payload->waiting_on_stub = true;
    message_payload->cv.notify_all();
    while (message_payload->waiting_on_stub) {
      message_payload->cv.wait(lk);
    }
  }
}

template <typename T, typename MessageType>
void
ModelInstanceState::ProcessMessage(
    const std::unique_ptr<IPCMessage>& ipc_message,
    std::function<void(std::unique_ptr<T>&, MessageType*)> request_handler)
{
  AllocatedSharedMemory<MessageType> message =
      Stub()->ShmPool()->Load<MessageType>(ipc_message->Args());
  MessageType* message_ptr =
      reinterpret_cast<MessageType*>(message.data_.get());
  std::unique_ptr<PbString> pb_error_message;
  PythonBackendException pb_exception(std::string{});
  std::unique_ptr<T> object =
      T::LoadFromSharedMemory(Stub()->ShmPool(), message_ptr->message);

  ScopedDefer _([message_ptr] {
    {
      bi::scoped_lock<bi::interprocess_mutex> guard{message_ptr->mu};
      message_ptr->waiting_on_stub = true;
      message_ptr->cv.notify_all();
      while (message_ptr->waiting_on_stub) {
        message_ptr->cv.wait(guard);
      }
    }
  });

  try {
    request_handler(object, message_ptr);
  }
  catch (const PythonBackendException& exception) {
    pb_exception = exception;
  }

  if (pb_exception.what() != std::string{}) {
    message_ptr->has_error = true;
    LOG_IF_EXCEPTION(
        pb_error_message =
            PbString::Create(Stub()->ShmPool(), pb_exception.what()));
    message_ptr->error = pb_error_message->ShmHandle();
    message_ptr->is_error_set = true;
  }
}

void
ModelInstanceState::ProcessMetricFamilyRequest(
    const std::unique_ptr<IPCMessage>& message)
{
  auto command = message->Command();
  ProcessMessage<MetricFamily, CustomMetricsMessage>(
      message, [this, command](
                   std::unique_ptr<MetricFamily>& metric_family,
                   CustomMetricsMessage* metrics_message_ptr) {
        switch (command) {
          case PYTHONSTUB_MetricFamilyRequestNew: {
            metrics_message_ptr->address =
                metric_family->InitializeTritonMetricFamily();
            break;
          }
          case PYTHONSTUB_MetricFamilyRequestDelete: {
            metric_family->ClearTritonMetricFamily();
            break;
          }
          default: {
            throw PythonBackendException("Unknown metric family request kind");
          }
        }
      });
}

void
ModelInstanceState::ProcessMetricRequest(
    const std::unique_ptr<IPCMessage>& message)
{
  auto command = message->Command();
  ProcessMessage<Metric, CustomMetricsMessage>(
      message, [this, command](
                   std::unique_ptr<Metric>& metric,
                   CustomMetricsMessage* metrics_message_ptr) {
        try {
          switch (command) {
            case PYTHONSTUB_MetricRequestNew: {
              metrics_message_ptr->address = metric->InitializeTritonMetric();
              break;
            }
            case PYTHONSTUB_MetricRequestIncrement:
            case PYTHONSTUB_MetricRequestSet:
            case PYTHONSTUB_MetricRequestObserve:
            case PYTHONSTUB_MetricRequestValue: {
              metric->HandleMetricOperation(metrics_message_ptr, command);
              break;
            }
            case PYTHONSTUB_MetricRequestDelete: {
              metric->ClearTritonMetric();
              break;
            }
            default: {
              throw PythonBackendException("Unknown metric request kind");
            }
          }
        }
        catch (const PythonBackendException& exception) {
          throw exception;
        }
      });
}

void
ModelInstanceState::ProcessModelControlRequest(
    const std::unique_ptr<IPCMessage>& message)
{
  auto command = message->Command();
  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  ProcessMessage<ModelLoader, ModelLoaderMessage>(
      message, [this, command, model_state](
                   std::unique_ptr<ModelLoader>& model_loader,
                   ModelLoaderMessage* model_loader_msg_ptr) {
        switch (command) {
          case PYTHONSTUB_LoadModelRequest: {
            model_loader->LoadModel(model_state->TritonServer());
            break;
          }
          case PYTHONSTUB_UnloadModelRequest: {
            model_loader->UnloadModel(model_state->TritonServer());
            break;
          }
          case PYTHONSTUB_ModelReadinessRequest: {
            model_loader_msg_ptr->is_model_ready =
                model_loader->IsModelReady(model_state->TritonServer());
            break;
          }
          default: {
            throw PythonBackendException("Unknown model loader request kind");
          }
        }
      });
}

TRITONSERVER_Error*
ModelInstanceState::SendMessageToStub(
    bi::managed_external_buffer::handle_t message)
{
  bool success = false;
  while (!success) {
    uint64_t timeout_miliseconds = 1000;
    {
      boost::posix_time::ptime timeout =
          boost::get_system_time() +
          boost::posix_time::milliseconds(timeout_miliseconds);

      bi::scoped_lock<bi::interprocess_mutex> lock(
          *(Stub()->HealthMutex()), timeout);

      // Check if lock has been acquired.
      if (lock) {
        Stub()->IpcControl()->stub_health = false;
      } else {
        // If it failed to obtain the lock, it means that the stub has been
        // stuck or exited while holding the health mutex lock.
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, "Failed to obtain the health mutex.");
      }
    }

    Stub()->StubMessageQueue()->Push(
        message, timeout_miliseconds /* duration ms */, success);

    if (!success && !IsStubProcessAlive()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "Stub process is not healthy.");
    }
  }

  return nullptr;  // success
}

void
ModelInstanceState::SendMessageAndReceiveResponse(
    bi::managed_external_buffer::handle_t message,
    bi::managed_external_buffer::handle_t& response,
    std::shared_ptr<std::vector<TRITONBACKEND_Response*>>& responses,
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  auto error = SendMessageToStub(message);
  if (error != nullptr) {
    RespondErrorToAllRequests(
        TRITONSERVER_ErrorMessage(error), responses, requests, request_count);

    return;
  }

  bi::managed_external_buffer::handle_t response_message;
  error = Stub()->ReceiveMessageFromStub(response_message);
  if (error != nullptr) {
    RespondErrorToAllRequests(
        TRITONSERVER_ErrorMessage(error), responses, requests, request_count);

    return;
  }

  response = response_message;
}

void
ModelInstanceState::RespondErrorToAllRequests(
    const char* message,
    std::shared_ptr<std::vector<TRITONBACKEND_Response*>>& responses,
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  for (uint32_t r = 0; r < request_count; ++r) {
    if ((*responses)[r] == nullptr)
      continue;

    std::string err_message =
        std::string(
            "Failed to process the request(s) for model instance '" + Name() +
            "', message: ") +
        message;

    TRITONSERVER_Error* err =
        TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, err_message.c_str());
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
            (*responses)[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
        "failed sending response");

    (*responses)[r] = nullptr;
    TRITONSERVER_ErrorDelete(err);
  }
}


void
ModelInstanceState::StartMonitor()
{
  stub_to_parent_thread_ = true;
  stub_to_parent_queue_monitor_ =
      std::thread(&ModelInstanceState::StubToParentMQMonitor, this);
}

void
ModelInstanceState::TerminateMonitor()
{
  if (stub_to_parent_thread_) {
    stub_to_parent_thread_ = false;
    // Push a dummy message to signal the thread to terminate.
    Stub()->StubToParentMessageQueue()->Push(DUMMY_MESSAGE);
    stub_to_parent_queue_monitor_.join();
  }
}

void
ModelInstanceState::ResponseSendDecoupled(
    std::shared_ptr<IPCMessage> response_send_message)
{
  AllocatedSharedMemory<ResponseSendMessage> send_message =
      Stub()->ShmPool()->Load<ResponseSendMessage>(
          response_send_message->Args());

  ResponseSendMessage* send_message_payload =
      reinterpret_cast<ResponseSendMessage*>(send_message.data_.get());
  std::unique_ptr<PbString> error_message;
  ScopedDefer response_factory_deleter([send_message_payload] {
    if (send_message_payload->flags == TRITONSERVER_RESPONSE_COMPLETE_FINAL) {
      TRITONBACKEND_ResponseFactory* response_factory =
          reinterpret_cast<TRITONBACKEND_ResponseFactory*>(
              send_message_payload->response_factory_address);
      std::unique_ptr<
          TRITONBACKEND_ResponseFactory, backend::ResponseFactoryDeleter>
          lresponse_factory(reinterpret_cast<TRITONBACKEND_ResponseFactory*>(
              response_factory));
    }
  });
  ScopedDefer _([send_message_payload] {
    {
      bi::scoped_lock<bi::interprocess_mutex> guard{send_message_payload->mu};
      send_message_payload->is_stub_turn = true;
      send_message_payload->cv.notify_all();

      while (send_message_payload->is_stub_turn) {
        send_message_payload->cv.wait(guard);
      }
    }
  });

  TRITONBACKEND_ResponseFactory* response_factory =
      reinterpret_cast<TRITONBACKEND_ResponseFactory*>(
          send_message_payload->response_factory_address);
  if (send_message_payload->flags == TRITONSERVER_RESPONSE_COMPLETE_FINAL) {
    {
      std::lock_guard<std::mutex> guard{closed_requests_mutex_};
      closed_requests_.push_back(send_message_payload->request_address);
    }
  }

  if (send_message_payload->response != 0) {
    std::unique_ptr<InferResponse> infer_response =
        InferResponse::LoadFromSharedMemory(
            Stub()->ShmPool(), send_message_payload->response,
            false /* open cuda ipc handle */);

    bool requires_deferred_callback = false;
    TRITONBACKEND_Response* response;
    SetErrorForResponseSendMessage(
        send_message_payload,
        WrapTritonErrorInSharedPtr(
            TRITONBACKEND_ResponseNewFromFactory(&response, response_factory)),
        error_message);

    std::vector<std::pair<std::unique_ptr<PbMemory>, void*>> gpu_output_buffers;
    GPUBuffersHelper gpu_buffer_helper;

#ifdef TRITON_ENABLE_GPU
    for (auto& output_tensor : infer_response->OutputTensors()) {
      if (!output_tensor->IsCPU()) {
        // Attempt to use the cuda shared memory pool for GPU tensor.
        ShareCUDAMemoryPool(output_tensor->MemoryTypeId());
      }
    }
#endif  // TRITON_ENABLE_GPU

    infer_response->Send(
        response, CudaStream(), requires_deferred_callback,
        send_message_payload->flags, Stub()->ShmPool(), gpu_buffer_helper,
        gpu_output_buffers);

    if (requires_deferred_callback) {
      gpu_buffer_helper.Complete(Stub()->ShmPool());
      send_message_payload->gpu_buffers_handle = gpu_buffer_helper.ShmHandle();

      // Additional round trip so that the stub can fill the GPU output buffers.
      {
        bi::scoped_lock<bi::interprocess_mutex> guard{send_message_payload->mu};
        send_message_payload->is_stub_turn = true;
        send_message_payload->cv.notify_all();

        while (send_message_payload->is_stub_turn) {
          send_message_payload->cv.wait(guard);
        }
      }

      bool cuda_copy = false;
      for (auto& output_buffer_pair : gpu_output_buffers) {
        auto& pb_memory = output_buffer_pair.first;
        void* pointer = output_buffer_pair.second;
        bool cuda_used;

        try {
          if (pb_memory->MemoryType() == TRITONSERVER_MEMORY_CPU) {
            THROW_IF_TRITON_ERROR(CopyBuffer(
                "Failed to copy the CPU output tensor to buffer.",
                TRITONSERVER_MEMORY_CPU, 0, TRITONSERVER_MEMORY_CPU, 0,
                pb_memory->ByteSize(), pb_memory->DataPtr(), pointer,
                CudaStream(), &cuda_used));
            cuda_copy |= cuda_used;
          } else if (
              (pb_memory->MemoryType() == TRITONSERVER_MEMORY_GPU) &&
              pb_memory->UseCUDASharedPool() &&
              (pb_memory->DataPtr() != pointer)) {
            // If the data pointer from pb_memory is not the same as the
            // pointer, it means that the Triton-provided buffer is not used
            // during tensor transfer. Instead, an intermediate buffer that uses
            // CUDA shared memory pool is used. In this case, we need to copy
            // the data from the intermediate buffer back to the Triton-provided
            // buffer.
            THROW_IF_TRITON_ERROR(CopyBuffer(
                "Failed to copy the GPU output tensor to buffer.",
                TRITONSERVER_MEMORY_GPU, pb_memory->MemoryTypeId(),
                TRITONSERVER_MEMORY_GPU, pb_memory->MemoryTypeId(),
                pb_memory->ByteSize(), pb_memory->DataPtr(), pointer,
                CudaStream(), &cuda_used));
            cuda_copy |= cuda_used;
          }
#ifdef TRITON_ENABLE_GPU
          if (cuda_copy) {
            cudaStreamSynchronize(stream_);
          }
#endif  // TRITON_ENABLE_GPU
        }
        catch (const PythonBackendException& pb_exception) {
          TRITONSERVER_Error* error = TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              (std::string(
                   "Failed to copy output tensor to Triton-provided buffer: ") +
               pb_exception.what())
                  .c_str());
          SetErrorForResponseSendMessage(
              send_message_payload, WrapTritonErrorInSharedPtr(error),
              error_message);
        }
      }
    }
  } else {
    TRITONSERVER_Error* error = TRITONBACKEND_ResponseFactorySendFlags(
        response_factory, send_message_payload->flags);
    SetErrorForResponseSendMessage(
        send_message_payload, WrapTritonErrorInSharedPtr(error), error_message);
  }
}

TRITONSERVER_Error*
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<std::unique_ptr<InferRequest>>& pb_infer_requests,
    PbMetricReporter& reporter)
{
  NVTX_RANGE(nvtx_, "ProcessRequests " + Name());
  closed_requests_ = {};
  ModelState* model_state = reinterpret_cast<ModelState*>(Model());

  size_t total_batch_size = 0;
  RETURN_IF_ERROR(
      CheckIncomingRequests(requests, request_count, total_batch_size));

  // No request to process
  if (total_batch_size == 0) {
    return nullptr;  // success
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " + Name() +
       ", executing " + std::to_string(request_count) + " requests")
          .c_str());

  AllocatedSharedMemory<char> request_batch;
  std::shared_ptr<std::vector<TRITONBACKEND_Response*>> responses;

  RETURN_IF_ERROR(SaveRequestsToSharedMemory(
      requests, request_count, pb_infer_requests, request_batch, responses));

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);
  reporter.SetComputeStartNs(compute_start_ns);

  std::unique_ptr<IPCMessage> ipc_message;
  RETURN_IF_EXCEPTION(
      ipc_message =
          IPCMessage::Create(Stub()->ShmPool(), false /*inline_response*/));
  ipc_message->Command() = PYTHONSTUB_CommandType::PYTHONSTUB_ExecuteRequest;
  ipc_message->Args() = request_batch.handle_;

  ScopedDefer execute_finalize([this] {
    // Push a dummy message to signal the thread to terminate.
    Stub()->StubMessageQueue()->Push(DUMMY_MESSAGE);
  });

  std::unique_ptr<IPCMessage> response;
  {
    Stub()->StubMessageQueue()->Push(ipc_message->ShmHandle());
    bi::managed_external_buffer::handle_t response_message;
    RETURN_IF_ERROR(Stub()->ReceiveMessageFromStub(response_message));
    response =
        IPCMessage::LoadFromSharedMemory(Stub()->ShmPool(), response_message);
  }
  char* ipc_message_shm =
      reinterpret_cast<char*>(response->GetAllocatedSharedMemory().data_.get());
  ResponseBatch* response_batch_shm_ptr =
      reinterpret_cast<ResponseBatch*>(ipc_message_shm + sizeof(IPCMessageShm));

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);
  reporter.SetComputeEndNs(compute_end_ns);
  reporter.SetBatchStatistics(total_batch_size);

  if (response_batch_shm_ptr->has_error) {
    // Clean up the response factory if an error occurred. The
    // `is_response_factory_deleted` flag indicates whether the response factory
    // has been deleted for some corner cases.
    if (!response_batch_shm_ptr->is_response_factory_deleted) {
      for (uint32_t r = 0; r < request_count; r++) {
        TRITONBACKEND_ResponseFactory* response_factory =
            reinterpret_cast<TRITONBACKEND_ResponseFactory*>(
                pb_infer_requests[r]->GetResponseFactoryAddress());
        std::unique_ptr<
            TRITONBACKEND_ResponseFactory, backend::ResponseFactoryDeleter>
            lresponse_factory(reinterpret_cast<TRITONBACKEND_ResponseFactory*>(
                response_factory));
      }
    }
    if (response_batch_shm_ptr->is_error_set) {
      auto error = PbString::LoadFromSharedMemory(
          Stub()->ShmPool(), response_batch_shm_ptr->error);
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, error->String().c_str());
    }

    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "Failed to process the requests.");
  }

  if (response_batch_shm_ptr->batch_size > 0) {
    bi::managed_external_buffer::handle_t* response_shm_handle =
        reinterpret_cast<bi::managed_external_buffer::handle_t*>(
            ipc_message_shm + sizeof(ResponseBatch) + sizeof(IPCMessageShm));

    std::shared_ptr<std::vector<TRITONBACKEND_Response*>> responses(
        new std::vector<TRITONBACKEND_Response*>());
    responses->reserve(request_count);
    for (size_t i = 0; i < request_count; i++) {
      // It is possible to have multiple responses batched together in a single
      // response batch shm, where some of the responses are None due to the
      // usage of response sender, so only create a TRITONBACKEND_Response
      // object for the valid responses.
      if (response_shm_handle[i] == 0) {
        responses->emplace_back(nullptr);
      } else {
        TRITONBACKEND_Response* response;
        auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
        if (err == nullptr) {
          responses->emplace_back(response);
        } else {
          responses->emplace_back(nullptr);
          LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
          TRITONSERVER_ErrorDelete(err);
        }
      }
    }

    std::vector<bool> requires_deferred_callback;

    bool has_gpu_output = false;
    std::vector<std::unique_ptr<InferResponse>> shm_responses;
    std::vector<std::vector<std::pair<std::unique_ptr<PbMemory>, void*>>>
        gpu_output_buffers(request_count);
    GPUBuffersHelper gpu_buffer_helper;

    for (uint32_t r = 0; r < request_count; ++r) {
      NVTX_RANGE(nvtx_, "LoadingResponse " + Name());
      requires_deferred_callback.push_back(false);
      if (response_shm_handle[r] == 0) {
        continue;
      }
      TRITONBACKEND_Response* response = (*responses)[r];
      TRITONBACKEND_Request* request = requests[r];
      uint32_t requested_output_count = 0;

      shm_responses.emplace_back(nullptr);
      std::unique_ptr<InferResponse>& infer_response = shm_responses.back();
      try {
        if (pb_infer_requests[r]->ReleaseFlags() ==
            TRITONSERVER_REQUEST_RELEASE_RESCHEDULE) {
          // For rescheduled requests, we do not need to send a response.
          LOG_IF_ERROR(
              TRITONBACKEND_ResponseDelete((*responses)[r]),
              "failed to delete response");
          (*responses)[r] = nullptr;
          continue;
        }
        {
          TRITONBACKEND_ResponseFactory* response_factory =
              reinterpret_cast<TRITONBACKEND_ResponseFactory*>(
                  pb_infer_requests[r]->GetResponseFactoryAddress());
          std::unique_ptr<
              TRITONBACKEND_ResponseFactory, backend::ResponseFactoryDeleter>
              lresponse_factory(
                  reinterpret_cast<TRITONBACKEND_ResponseFactory*>(
                      response_factory));
        }
        infer_response = InferResponse::LoadFromSharedMemory(
            Stub()->ShmPool(), response_shm_handle[r],
            false /* open_cuda_handle */);
        if (infer_response->HasError()) {
          TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(
              infer_response->Error()->Code(),
              infer_response->Error()->Message().c_str());

          LOG_IF_ERROR(
              TRITONBACKEND_ResponseSend(
                  (*responses)[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
              "failed sending response");
          TRITONSERVER_ErrorDelete(err);
          (*responses)[r] = nullptr;

          // Reset the release flags for the request.
          pb_infer_requests[r]->SetReleaseFlags(
              TRITONSERVER_REQUEST_RELEASE_ALL);

          // If has_error is true, we do not look at the response tensors.
          continue;
        }
      }
      catch (const PythonBackendException& pb_exception) {
        TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, pb_exception.what());
        LOG_IF_ERROR(
            TRITONBACKEND_ResponseSend(
                (*responses)[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
            "failed sending response");
        TRITONSERVER_ErrorDelete(err);
        (*responses)[r] = nullptr;

        // Reset the release flags for the request.
        pb_infer_requests[r]->SetReleaseFlags(TRITONSERVER_REQUEST_RELEASE_ALL);

        continue;
      }

      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestOutputCount(request, &requested_output_count));
      std::set<std::string> requested_output_names;
      for (size_t j = 0; j < requested_output_count; ++j) {
        const char* output_name;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_RequestOutputName(request, j, &output_name));
        requested_output_names.insert(output_name);
      }

      bool require_deferred_callback = false;

#ifdef TRITON_ENABLE_GPU
      for (auto& output_tensor : infer_response->OutputTensors()) {
        if (output_tensor->MemoryType() == TRITONSERVER_MEMORY_GPU) {
          // Attempt to use the cuda shared memory pool for GPU tensor.
          ShareCUDAMemoryPool(output_tensor->MemoryTypeId());
        }
      }
#endif  // TRITON_ENABLE_GPU

      gpu_output_buffers[r] =
          std::vector<std::pair<std::unique_ptr<PbMemory>, void*>>{};
      infer_response->Send(
          response, CudaStream(), require_deferred_callback,
          TRITONSERVER_RESPONSE_COMPLETE_FINAL, Stub()->ShmPool(),
          gpu_buffer_helper, gpu_output_buffers[r], requested_output_names);

      requires_deferred_callback[r] = require_deferred_callback;

      if (requires_deferred_callback[r]) {
        has_gpu_output = true;
      }
    }

    execute_finalize.Complete();

    // If the output tensor is in GPU, there will be a second round trip
    // required for filling the GPU buffers provided by the main process.
    if (has_gpu_output) {
      ipc_message->Command() =
          PYTHONSTUB_CommandType::PYTHONSTUB_LoadGPUBuffers;
      gpu_buffer_helper.Complete(Stub()->ShmPool());
      ipc_message->Args() = gpu_buffer_helper.ShmHandle();
      bi::managed_external_buffer::handle_t response_message;
      SendMessageAndReceiveResponse(
          ipc_message->ShmHandle(), response_message, responses, requests, 0);

      bool cuda_copy = false;

      uint32_t response_index = 0;
      for (auto& gpu_output_buffer : gpu_output_buffers) {
        for (auto& buffer_memory_pair : gpu_output_buffer) {
          auto& pb_memory = buffer_memory_pair.first;
          void* pointer = buffer_memory_pair.second;
          bool cuda_used = false;

          if (pb_memory->MemoryType() == TRITONSERVER_MEMORY_CPU) {
            GUARDED_RESPOND_IF_ERROR(
                responses, response_index,
                CopyBuffer(
                    "Failed to copy the output tensor to buffer.",
                    TRITONSERVER_MEMORY_CPU, 0, TRITONSERVER_MEMORY_CPU, 0,
                    pb_memory->ByteSize(), pb_memory->DataPtr(), pointer,
                    CudaStream(), &cuda_used));
            cuda_copy |= cuda_used;
          } else if (
              (pb_memory->MemoryType() == TRITONSERVER_MEMORY_GPU) &&
              pb_memory->UseCUDASharedPool() &&
              (pb_memory->DataPtr() != pointer)) {
            // If the data pointer from pb_memory is not the same as the
            // pointer, it means that the Triton-provided buffer is not used
            // during tensor transfer. Instead, an intermediate buffer that uses
            // CUDA shared memory pool is used. In this case, we need to copy
            // the data from the intermediate buffer back to the Triton-provided
            // buffer.
            GUARDED_RESPOND_IF_ERROR(
                responses, response_index,
                CopyBuffer(
                    "Failed to copy the output tensor to buffer.",
                    TRITONSERVER_MEMORY_GPU, pb_memory->MemoryTypeId(),
                    TRITONSERVER_MEMORY_GPU, pb_memory->MemoryTypeId(),
                    pb_memory->ByteSize(), pb_memory->DataPtr(), pointer,
                    CudaStream(), &cuda_used));
            cuda_copy |= cuda_used;
          }
        }
        response_index++;
#ifdef TRITON_ENABLE_GPU
        if (cuda_copy) {
          cudaStreamSynchronize(stream_);
        }
#endif  // TRITON_ENABLE_GPU
      }
    }

    for (uint32_t r = 0; r < request_count; ++r) {
      if (requires_deferred_callback[r]) {
        shm_responses[r]->DeferredSendCallback();
      }
    }
  }

  return nullptr;  // success
}

void
ModelInstanceState::PrepareResponseBatch(
    ResponseBatch** response_batch,
    AllocatedSharedMemory<char>& response_batch_shm,
    std::unique_ptr<IPCMessage>* ipc_message,
    bi::managed_external_buffer::handle_t** response_handle)
{
  response_batch_shm = Stub()->ShmPool()->Construct<char>(
      sizeof(ResponseBatch) + sizeof(bi::managed_external_buffer::handle_t));
  *response_batch =
      reinterpret_cast<ResponseBatch*>(response_batch_shm.data_.get());
  (*ipc_message)->Args() = response_batch_shm.handle_;

  *response_handle = reinterpret_cast<bi::managed_external_buffer::handle_t*>(
      response_batch_shm.data_.get() + sizeof(ResponseBatch));

  (*response_batch)->batch_size = 1;
  (*response_batch)->has_error = false;
  (*response_batch)->is_error_set = false;
  (*response_batch)->cleanup = false;
  (*response_batch)->response_size = 1;
}

void
ModelInstanceState::PrepareResponseHandle(
    std::unique_ptr<InferResponse>* infer_response,
    bi::managed_external_buffer::handle_t* response_handle)
{
#ifdef TRITON_ENABLE_GPU
  for (auto& output_tensor : (*infer_response)->OutputTensors()) {
    if (!output_tensor->IsCPU()) {
      // Attempt to use the cuda shared memory pool for GPU tensor.
      ShareCUDAMemoryPool(output_tensor->MemoryTypeId());
      // It's possible that the CUDA memory pool offset isn't set correctly,
      // even if the BLS output is using CUDA memory. This can occur when the
      // CUDA memory pool hasn't been shared with the stub process at the time
      // the BLS output is allocated during the ResponseAlloc callback. In such
      // cases, we need to adjust the CUDA pool offset accordingly.
      if (!output_tensor->Memory()->UseCUDASharedPool()) {
        output_tensor->Memory()->UpdateCUDAOffset(
            Stub()->ShmPool()->GetCUDAMemoryPoolManager());
      }
    }
  }
#endif  // TRITON_ENABLE_GPU

  (*infer_response)->SaveToSharedMemory(Stub()->ShmPool());

  for (auto& output_tensor : (*infer_response)->OutputTensors()) {
    if (!output_tensor->IsCPU()) {
#ifdef TRITON_ENABLE_GPU
      std::unique_ptr<MemoryRecord> memory_record;
      // Need to transfer the ownership of the BackendMemory to the
      // MemoryManager so that the lifetime of the BackendMemory is managed.
      memory_record = std::make_unique<BackendMemoryRecord>(
          output_tensor->Memory()->GetBackendMemory());
      uint64_t memory_release_id =
          Stub()->GetMemoryManager()->AddRecord(std::move(memory_record));
      output_tensor->Memory()->SetMemoryReleaseId(memory_release_id);
#endif
    }
  }

  *response_handle = (*infer_response)->ShmHandle();
}

void
ModelInstanceState::SendBLSDecoupledResponse(
    std::unique_ptr<InferResponse> infer_response)
{
  bool is_response_batch_set = false;
  ResponseBatch* response_batch = nullptr;
  std::unique_ptr<PbString> pb_error_message;
  std::unique_ptr<IPCMessage> ipc_message;
  AllocatedSharedMemory<char> response_batch_shm;
  bi::managed_external_buffer::handle_t* response_handle = nullptr;

  try {
    ipc_message =
        IPCMessage::Create(Stub()->ShmPool(), true /* inline_response */);
    ipc_message->Args() = response_batch_shm.handle_;
    ipc_message->Command() = PYTHONSTUB_InferStreamExecResponse;
    PrepareResponseBatch(
        &response_batch, response_batch_shm, &ipc_message, &response_handle);
    is_response_batch_set = true;
    response_batch->waiting_on_stub = false;
    PrepareResponseHandle(&infer_response, response_handle);
  }
  catch (const PythonBackendException& pb_exception) {
    if (is_response_batch_set) {
      response_batch->has_error = true;
      LOG_IF_EXCEPTION(
          pb_error_message =
              PbString::Create(Stub()->ShmPool(), pb_exception.what()));

      if (pb_error_message != nullptr) {
        response_batch->is_error_set = true;
        response_batch->error = pb_error_message->ShmHandle();
      }
    } else {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, pb_exception.what());
    }
  }

  {
    bi::scoped_lock<bi::interprocess_mutex> lock{
        *(ipc_message->ResponseMutex())};
    Stub()->ParentToStubMessageQueue()->Push(ipc_message->ShmHandle());
    while (!response_batch->waiting_on_stub) {
      ipc_message->ResponseCondition()->wait(lock);
    }
  }
}

void
ModelInstanceState::ShareCUDAMemoryPool(const int32_t device_id)
{
#ifdef TRITON_ENABLE_GPU
  try {
    Stub()->ShareCUDAMemoryPool(Model()->TritonMemoryManager(), device_id);
  }
  catch (const PythonBackendException& ex) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        (std::string("Failed to share CUDA memory pool with stub process: ") +
         ex.what() + ". Will use CUDA IPC.")
            .c_str());
  }
#endif  // TRITON_ENABLE_GPU
}

ModelInstanceState::~ModelInstanceState()
{
  Stub()->UpdateHealth();
  if (Stub()->IsHealthy()) {
    // Wait for all the pending tasks to finish.
    thread_pool_->wait();
  }
  // Terminate stub first to allow any last messages to be received by the back
  // end before deallocating the queue memory
  Stub()->TerminateStub();
  TerminateMonitor();
  Stub()->ClearQueues();
  Stub().reset();
}

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  // Auto-complete the configuration if requested...
  bool auto_complete_config = false;
  RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(
      triton_model, &auto_complete_config));
  if (auto_complete_config) {
    RETURN_IF_ERROR((*state)->LaunchAutoCompleteStubProcess());
    (*state)->ModelConfig() = std::move((*state)->Stub()->AutoCompleteConfig());
    RETURN_IF_ERROR((*state)->SetModelConfig());

    (*state)->Stub()->UpdateHealth();
    (*state)->Stub()->TerminateStub();
    (*state)->Stub()->ClearQueues();
    (*state)->Stub().reset();
  }

  RETURN_IF_ERROR((*state)->ValidateModelConfig());

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model, true /* allow_optional */)
{
  TRITONBACKEND_Backend* backend;
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelBackend(triton_model, &backend));

  const char* path = nullptr;
  TRITONBACKEND_ArtifactType artifact_type;
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelRepository(triton_model, &artifact_type, &path));
  python_execution_env_ = "";
  force_cpu_only_input_tensors_ = true;
  decoupled_ = false;

  void* bstate;
  THROW_IF_BACKEND_MODEL_ERROR(TRITONBACKEND_BackendState(backend, &bstate));
  backend_state_ = reinterpret_cast<BackendState*>(bstate);

  runtime_modeldir_ = backend_state_->runtime_modeldir;
  triton::common::TritonJson::Value params;
  common::TritonJson::Value model_config;
  if (model_config_.Find("parameters", &params)) {
    // Skip the EXECUTION_ENV_PATH variable if it doesn't exist.
    TRITONSERVER_Error* error =
        GetParameterValue(params, "EXECUTION_ENV_PATH", &python_execution_env_);
    if (error == nullptr) {
      std::string relative_path_keyword = "$$TRITON_MODEL_DIRECTORY";
      size_t relative_path_loc =
          python_execution_env_.find(relative_path_keyword);
      if (relative_path_loc != std::string::npos) {
        python_execution_env_.replace(
            relative_path_loc, relative_path_loc + relative_path_keyword.size(),
            path);
      }
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Using Python execution env ") + python_execution_env_)
              .c_str());
    } else {
      // Delete the error
      TRITONSERVER_ErrorDelete(error);
    }

    triton::common::TritonJson::Value model_transaction_policy;
    if (model_config_.Find(
            "model_transaction_policy", &model_transaction_policy)) {
      triton::common::TritonJson::Value decoupled;
      if (model_transaction_policy.Find("decoupled", &decoupled)) {
        auto error = decoupled.AsBool(&decoupled_);
        if (error != nullptr) {
          throw BackendModelException(error);
        }
      }
    }

    // Skip the FORCE_CPU_ONLY_INPUT_TENSORS variable if it doesn't exits.
    std::string force_cpu_only_input_tensor;
    error = nullptr;
    error = GetParameterValue(
        params, "FORCE_CPU_ONLY_INPUT_TENSORS", &force_cpu_only_input_tensor);
    if (error == nullptr) {
      if (force_cpu_only_input_tensor == "yes") {
        force_cpu_only_input_tensors_ = true;
        LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (std::string("Forcing CPU only input tensors.")).c_str());
      } else if (force_cpu_only_input_tensor == "no") {
        force_cpu_only_input_tensors_ = false;
        LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (std::string("Input tensors can be both in CPU and GPU. "
                         "FORCE_CPU_ONLY_INPUT_TENSORS is off."))
                .c_str());
      } else {
        throw BackendModelException(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            (std::string("Incorrect value for FORCE_CPU_ONLY_INPUT_TENSORS: ") +
             force_cpu_only_input_tensor + "'")
                .c_str()));
      }
    } else {
      // Delete the error
      TRITONSERVER_ErrorDelete(error);
    }
  }

  if (artifact_type != TRITONBACKEND_ARTIFACT_FILESYSTEM) {
    throw BackendModelException(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("unsupported artifact type for model '") + Name() + "'")
            .c_str()));
  }
}

TRITONSERVER_Error*
ModelState::LaunchAutoCompleteStubProcess()
{
  Stub() = std::make_unique<StubLauncher>("AUTOCOMPLETE_STUB");
  RETURN_IF_ERROR(Stub()->Initialize(this));
  try {
    RETURN_IF_ERROR(Stub()->Setup());
    RETURN_IF_ERROR(Stub()->Launch());
  }
  catch (const BackendModelException& ex) {
    Stub()->UpdateHealth();
    Stub()->TerminateStub();
    Stub()->ClearQueues();
    Stub().reset();
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // We have the json DOM for the model configuration...
  triton::common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  return nullptr;
}

TRITONSERVER_Error*
ModelState::SetModelConfig()
{
  BackendModel::SetModelConfig();
  // `Update model_transaction_policy` if setting was set
  // with `set_model_transaction_policy`
  triton::common::TritonJson::Value model_transaction_policy;
  bool is_decoupled = false;
  if (ModelConfig().Find(
          "model_transaction_policy", &model_transaction_policy)) {
    triton::common::TritonJson::Value decoupled;
    if (model_transaction_policy.Find("decoupled", &decoupled)) {
      auto error = decoupled.AsBool(&is_decoupled);
      if (error != nullptr) {
        throw BackendModelException(error);
      }
      SetDecoupled(is_decoupled);
    }
  }

  return nullptr;
}


extern "C" {

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  // Check backend version to ensure compatibility
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "Triton backend API version does not support this backend");
  }

  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("backend configuration:\n") + buffer).c_str());

  triton::common::TritonJson::Value backend_config;
  if (byte_size != 0) {
    RETURN_IF_ERROR(backend_config.Parse(buffer, byte_size));
  }

  std::unique_ptr<BackendState> backend_state(new BackendState());
  triton::common::TritonJson::Value cmdline;
  backend_state->shm_default_byte_size = 1 * 1024 * 1024;  // 1 MB
  backend_state->shm_growth_byte_size = 1 * 1024 * 1024;   // 1 MB
  backend_state->stub_timeout_seconds = 30;
  backend_state->shm_message_queue_size = 1000;
  backend_state->thread_pool_size = 32;
  // Initialize shared memory region prefix to include backend's name
  // to avoid collision between python backend and python-based backends.
  backend_state->shared_memory_region_prefix =
      "triton_" + name + "_backend_shm_region_";
  std::string default_backend_dir_string;

  if (backend_config.Find("cmdline", &cmdline)) {
    triton::common::TritonJson::Value shm_growth_size;
    std::string shm_growth_byte_size;
    if (cmdline.Find("shm-growth-byte-size", &shm_growth_size)) {
      RETURN_IF_ERROR(shm_growth_size.AsString(&shm_growth_byte_size));
      try {
        backend_state->shm_growth_byte_size = std::stol(shm_growth_byte_size);
        if (backend_state->shm_growth_byte_size <= 0) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("shm-growth-byte-size") +
               " can't be smaller than or equal to zero.")
                  .c_str());
        }
      }
      catch (const std::invalid_argument& ia) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, ia.what());
      }
    }

    triton::common::TritonJson::Value shm_default_size;
    std::string shm_default_byte_size;
    if (cmdline.Find("shm-default-byte-size", &shm_default_size)) {
      RETURN_IF_ERROR(shm_default_size.AsString(&shm_default_byte_size));
      try {
        backend_state->shm_default_byte_size = std::stol(shm_default_byte_size);
        // Shared memory default byte size can't be less than 1 MB.
        if (backend_state->shm_default_byte_size < 1 * 1024 * 1024) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("shm-default-byte-size") +
               " can't be smaller than 4 MiBs")
                  .c_str());
        }
      }
      catch (const std::invalid_argument& ia) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, ia.what());
      }
    }

    triton::common::TritonJson::Value thread_pool_size;
    std::string thread_pool_count;
    if (cmdline.Find("thread-pool-size", &thread_pool_size)) {
      RETURN_IF_ERROR(thread_pool_size.AsString(&thread_pool_count));
      try {
        backend_state->thread_pool_size = std::stol(thread_pool_count);
        // Shared memory default byte size can't be less than 4 MBs.
        if (backend_state->thread_pool_size < 1) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("thread-pool-size") + " can't be less than 1.")
                  .c_str());
        }
      }
      catch (const std::invalid_argument& ia) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, ia.what());
      }
    }

    triton::common::TritonJson::Value shm_region_prefix;
    std::string shm_region_prefix_str;
    if (cmdline.Find("shm-region-prefix-name", &shm_region_prefix)) {
      RETURN_IF_ERROR(shm_region_prefix.AsString(&shm_region_prefix_str));
      // Shared memory default byte size can't be less than 4 MBs.
      if (shm_region_prefix_str.size() == 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("shm-region-prefix-name") +
             " must at least contain one character.")
                .c_str());
      }
      backend_state->shared_memory_region_prefix = shm_region_prefix_str;
    }

    triton::common::TritonJson::Value shm_message_queue_size;
    std::string shm_message_queue_size_str;
    if (cmdline.Find("shm_message_queue_size", &shm_message_queue_size)) {
      RETURN_IF_ERROR(
          shm_message_queue_size.AsString(&shm_message_queue_size_str));
      try {
        backend_state->shm_message_queue_size =
            std::stol(shm_message_queue_size_str);
      }
      catch (const std::invalid_argument& ia) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, ia.what());
      }
    }

    triton::common::TritonJson::Value stub_timeout_seconds;
    std::string stub_timeout_string_seconds;
    if (cmdline.Find("stub-timeout-seconds", &stub_timeout_seconds)) {
      RETURN_IF_ERROR(
          stub_timeout_seconds.AsString(&stub_timeout_string_seconds));
      try {
        backend_state->stub_timeout_seconds =
            std::stol(stub_timeout_string_seconds);
        if (backend_state->stub_timeout_seconds <= 0) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("stub-timeout-seconds") +
               " can't be smaller than or equal to zero.")
                  .c_str());
        }
      }
      catch (const std::invalid_argument& ia) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, ia.what());
      }
    }

    triton::common::TritonJson::Value default_backend_dir;
    if (cmdline.Find("backend-directory", &default_backend_dir)) {
      RETURN_IF_ERROR(
          default_backend_dir.AsString(&default_backend_dir_string));
    }
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("Shared memory configuration is shm-default-byte-size=") +
       std::to_string(backend_state->shm_default_byte_size) +
       ",shm-growth-byte-size=" +
       std::to_string(backend_state->shm_growth_byte_size) +
       ",stub-timeout-seconds=" +
       std::to_string(backend_state->stub_timeout_seconds))
          .c_str());

  // Use BackendArtifacts to determine the location of Python files
  const char* clocation;
  TRITONBACKEND_ArtifactType artifact_type;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendArtifacts(backend, &artifact_type, &clocation));

  const char os_slash = std::filesystem::path::preferred_separator;
  std::string location(clocation);
#ifdef _WIN32
  const std::string stub_executable_name = "triton_python_backend_stub.exe";
  SanitizePath(location);
  SanitizePath(default_backend_dir_string);
#else
  const std::string stub_executable_name = "triton_python_backend_stub";
#endif
  // Check if `triton_python_backend_stub` and `triton_python_backend_utils.py`
  // are located under `location`.
  std::string default_python_backend_dir =
      default_backend_dir_string + os_slash + "python";
  std::string backend_stub_path = location + os_slash + stub_executable_name;
  std::string backend_utils =
      location + os_slash + "triton_python_backend_utils.py";
  // Both, stub and utils should be in the same location
  if (FileExists(backend_stub_path) && FileExists(backend_utils)) {
    backend_state->python_lib = location;
    // If `location` is default location of a python backend,
    // then we are using default python backend.
    if (default_python_backend_dir == location) {
      backend_state->runtime_modeldir = "";
    } else {
      // If `location` is not default location of a python backend,
      // then we are using a python backend based backend and model.py stored
      // in the received location.
      backend_state->runtime_modeldir = location;
    }
  } else {
    // If stub and utils are not found in received `location`,
    // then we are using a python backend based backend and stub and utils are
    // stored in the default python backend location.
    if (!default_backend_dir_string.empty()) {
      std::string backend_stub_path = default_backend_dir_string + os_slash +
                                      "python" + os_slash +
                                      stub_executable_name;
      if (!FileExists(backend_stub_path)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_NOT_FOUND,
            (stub_executable_name + " is not found. Searched paths: " +
             default_backend_dir_string + os_slash + "python and " + location)
                .c_str());
      }
    }
    backend_state->runtime_modeldir = location;
    backend_state->python_lib =
        default_backend_dir_string + os_slash + "python";
  }
// FIXME [DLIS-5969]: Enable for Windows when custom execution environments
// are supported.
#ifndef _WIN32
  backend_state->env_manager = std::make_unique<EnvironmentManager>();
#endif

  RETURN_IF_ERROR(TRITONBACKEND_BackendSetState(
      backend, reinterpret_cast<void*>(backend_state.get())));

  backend_state.release();
  return nullptr;
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "TRITONBACKEND_Finalize: Start");
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  auto backend_state = reinterpret_cast<BackendState*>(vstate);
  delete backend_state;
  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "TRITONBACKEND_Finalize: End");
  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_VERBOSE, __FILE__, __LINE__,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  TRITONBACKEND_Backend* backend;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));

  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
          .c_str());

  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  RETURN_IF_ERROR(instance_state->LaunchStubProcess());
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: instance "
                   "initialization successful ") +
       name + " (device " + std::to_string(device_id) + ")")
          .c_str());

  return nullptr;
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));

  TRITONSERVER_Error* error = nullptr;

  // If restart is equal to true, it indicates that the stub process is
  // unhealthy and needs a restart.
  // TODO: Implement restart on decoupled

  std::vector<std::unique_ptr<InferRequest>> infer_requests;
  {
    uint64_t exec_start_ns = 0;
    SET_TIMESTAMP(exec_start_ns);

    PbMetricReporter reporter(
        instance_state->TritonModelInstance(), requests, request_count,
        nullptr);
    reporter.SetExecStartNs(exec_start_ns);

    error = instance_state->ProcessRequests(
        requests, request_count, infer_requests, reporter);

    uint64_t exec_end_ns = 0;
    SET_TIMESTAMP(exec_end_ns);
    reporter.SetExecEndNs(exec_end_ns);

    if (error != nullptr) {
      reporter.SetSuccessStatus(false);
      for (uint32_t r = 0; r < request_count; ++r) {
        TRITONBACKEND_Request* request = requests[r];
        if (!instance_state->ExistsInClosedRequests(
                reinterpret_cast<intptr_t>(request))) {
          TRITONBACKEND_Response* response = nullptr;
          LOG_IF_ERROR(
              TRITONBACKEND_ResponseNew(&response, request),
              "Failed to create a new response.");

          if (response != nullptr) {
            LOG_IF_ERROR(
                TRITONBACKEND_ResponseSend(
                    response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, error),
                "Failed to send the error response.");
          }
        }
      }

      for (auto& infer_request : infer_requests) {
        // Reset the release flags for all the requests.
        infer_request->SetReleaseFlags(TRITONSERVER_REQUEST_RELEASE_ALL);
      }
    }
  }

  // The InferRequest object might not be created if an error occurs. Explicitly
  // update the release flags here based on the number of InferRequest objects.
  std::vector<uint32_t> request_release_flags(
      request_count, TRITONSERVER_REQUEST_RELEASE_ALL);
  for (size_t i = 0; i < infer_requests.size(); ++i) {
    request_release_flags[i] = infer_requests[i]->ReleaseFlags();
  }

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    try {
      THROW_IF_TRITON_ERROR(
          TRITONBACKEND_RequestRelease(request, request_release_flags[r]));
    }
    catch (const PythonBackendException& pb_exception) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("Failed to release request: ") + pb_exception.what())
              .c_str());
      if (request_release_flags[r] == TRITONSERVER_REQUEST_RELEASE_RESCHEDULE) {
        // If error occurs during request rescheduling, release the request with
        // `TRITONSERVER_REQUEST_RELEASE_ALL` flag.
        LOG_IF_ERROR(
            TRITONBACKEND_RequestRelease(
                request, TRITONSERVER_REQUEST_RELEASE_ALL),
            "Failed to release request.");
      }
    }
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelInstanceExecute: model instance name ") +
       instance_state->Name() + " released " + std::to_string(request_count) +
       " requests")
          .c_str());

  return nullptr;
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_GetBackendAttribute(
    TRITONBACKEND_Backend* backend,
    TRITONBACKEND_BackendAttribute* backend_attributes)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      "TRITONBACKEND_GetBackendAttribute: setting attributes");
  // Specify different preferred instance kind based on backend compatibility,
  // so Triton core won't blindly auto-complete kind that may not be supported.
  // Other instance groups setting are set to "no value" so that Triton core
  // will auto-complete them with default policy.
#ifdef TRITON_ENABLE_GPU
  RETURN_IF_ERROR(TRITONBACKEND_BackendAttributeAddPreferredInstanceGroup(
      backend_attributes, TRITONSERVER_INSTANCEGROUPKIND_GPU, 0, nullptr, 0));
#else
  RETURN_IF_ERROR(TRITONBACKEND_BackendAttributeAddPreferredInstanceGroup(
      backend_attributes, TRITONSERVER_INSTANCEGROUPKIND_CPU, 0, nullptr, 0));
#endif

  // This backend can safely handle parallel calls to
  // TRITONBACKEND_ModelInstanceInitialize (thread-safe).
  RETURN_IF_ERROR(TRITONBACKEND_BackendAttributeSetParallelModelInstanceLoading(
      backend_attributes, true));

  return nullptr;
}

}  // extern "C"
}}}  // namespace triton::backend::python
