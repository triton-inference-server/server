// Copyright 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "request_executor.h"

#include <future>

#include "correlation_id.h"
#include "pb_utils.h"
#include "scoped_defer.h"
#include "triton/backend/backend_common.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace python {

TRITONSERVER_Error*
CreateTritonErrorFromException(const PythonBackendException& pb_exception)
{
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INTERNAL, pb_exception.what());
}

TRITONSERVER_Error*
MemoryTypeToTritonMemoryType(
    TRITONSERVER_MemoryType* triton_memory_type,
    const PreferredMemory::MemoryType& memory_type)
{
  switch (memory_type) {
    case PreferredMemory::MemoryType::kCPU:
      *triton_memory_type = TRITONSERVER_MEMORY_CPU;
      break;
    case PreferredMemory::MemoryType::kGPU:
      *triton_memory_type = TRITONSERVER_MEMORY_GPU;
      break;

    default:
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "Unknown memory type");
  }

  return nullptr;
}

void
InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  if (request != nullptr) {
    RequestCompletionUserp* completion_userp =
        reinterpret_cast<RequestCompletionUserp*>(userp);
    completion_userp->infer_payload->SetRequestAddress(0L);

    LOG_IF_ERROR(
        TRITONSERVER_InferenceRequestDelete(request),
        "Failed to delete inference request.");

    delete completion_userp;
  }
}

void
InferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  auto linfer_payload = reinterpret_cast<InferPayload*>(userp);
  std::shared_ptr<InferPayload> infer_payload = linfer_payload->GetPtr();
  std::unique_ptr<InferResponse> infer_response;
  std::vector<std::shared_ptr<PbTensor>> output_tensors;
  std::shared_ptr<PbError> pb_error;
  std::string parameters_string;
  TRITONSERVER_Error_Code error_code = TRITONSERVER_ERROR_INTERNAL;

  if (response != nullptr) {
    try {
      TRITONSERVER_Error* server_error =
          TRITONSERVER_InferenceResponseError(response);
      if (server_error != nullptr) {
        error_code = TRITONSERVER_ErrorCode(server_error);
      }
      THROW_IF_TRITON_ERROR(server_error);

      uint32_t output_count;
      THROW_IF_TRITON_ERROR(
          TRITONSERVER_InferenceResponseOutputCount(response, &output_count));

      for (uint32_t idx = 0; idx < output_count; ++idx) {
        const char* cname;
        TRITONSERVER_DataType datatype;
        const int64_t* shape;
        uint64_t dim_count;
        const void* base;
        size_t byte_size;
        TRITONSERVER_MemoryType memory_type;
        int64_t memory_type_id;
        void* userp;

        THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceResponseOutput(
            response, idx, &cname, &datatype, &shape, &dim_count, &base,
            &byte_size, &memory_type, &memory_type_id, &userp));
        std::string sname = cname;
        std::vector<int64_t> dims_vector{shape, shape + dim_count};

        if (memory_type != TRITONSERVER_MEMORY_GPU) {
          if (byte_size != 0) {
            std::shared_ptr<PbTensor> pb_tensor = std::make_shared<PbTensor>(
                sname, dims_vector, datatype, memory_type, memory_type_id,
                const_cast<void*>(base), byte_size,
                nullptr /* DLManagedTensor */);

            // Load the data so that it is deallocated automatically.
            std::unique_ptr<PbMemory> pb_memory(
                reinterpret_cast<PbMemory*>(userp));
            pb_tensor->SetMemory(std::move(pb_memory));
            output_tensors.push_back(pb_tensor);
          } else {
            output_tensors.push_back(std::make_shared<PbTensor>(
                sname, dims_vector, datatype, memory_type, memory_type_id,
                const_cast<void*>(base), byte_size,
                nullptr /* DLManagedTensor */));
          }
        } else {
          std::shared_ptr<PbTensor> pb_tensor = std::make_shared<PbTensor>(
              sname, dims_vector, datatype, memory_type, memory_type_id,
              const_cast<void*>(base), byte_size,
              nullptr /* DLManagedTensor */);

          std::unique_ptr<PbMemory> pb_memory(
              reinterpret_cast<PbMemory*>(userp));
          pb_tensor->SetMemory(std::move(pb_memory));
          output_tensors.push_back(pb_tensor);
        }
      }

      triton::common::TritonJson::Value parameters_json(
          triton::common::TritonJson::ValueType::OBJECT);
      uint32_t parameter_count;
      THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceResponseParameterCount(
          response, &parameter_count));

      for (size_t i = 0; i < parameter_count; i++) {
        const char* name;
        TRITONSERVER_ParameterType type;
        const void* vvalue;
        THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceResponseParameter(
            response, i, &name, &type, &vvalue));
        if (type == TRITONSERVER_PARAMETER_INT) {
          THROW_IF_TRITON_ERROR(parameters_json.AddInt(
              name, *(reinterpret_cast<const int64_t*>(vvalue))));
        } else if (type == TRITONSERVER_PARAMETER_BOOL) {
          THROW_IF_TRITON_ERROR(parameters_json.AddBool(
              name, *(reinterpret_cast<const bool*>(vvalue))));
        } else if (type == TRITONSERVER_PARAMETER_STRING) {
          std::string string = reinterpret_cast<const char*>(vvalue);
          THROW_IF_TRITON_ERROR(parameters_json.AddString(name, string));
        } else {
          throw PythonBackendException(
              (std::string("Unsupported parameter type for parameter '") +
               name + "'."));
        }
      }

      triton::common::TritonJson::WriteBuffer buffer;
      THROW_IF_TRITON_ERROR(parameters_json.Write(&buffer));
      parameters_string = buffer.Contents();
    }
    catch (const PythonBackendException& pb_exception) {
      if (response != nullptr) {
        LOG_IF_ERROR(
            TRITONSERVER_InferenceResponseDelete(response),
            "Failed to delete inference response.");

        response = nullptr;
      }
      pb_error = std::make_shared<PbError>(pb_exception.what(), error_code);
      output_tensors.clear();
    }

    if (!infer_payload->IsDecoupled()) {
      infer_response = std::make_unique<InferResponse>(
          output_tensors, pb_error, parameters_string,
          true /* is_last_response */);
    } else {
      if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) == 0) {
        // Not the last response.
        infer_response = std::make_unique<InferResponse>(
            output_tensors, pb_error, parameters_string,
            false /* is_last_response */, userp /* id */);
      } else {
        // The last response.
        infer_response = std::make_unique<InferResponse>(
            output_tensors, pb_error, parameters_string,
            true /* is_last_response */, userp /* id */);
      }
    }

    LOG_IF_ERROR(
        TRITONSERVER_InferenceResponseDelete(response),
        "Failed to release BLS inference response.");
  } else if (
      (infer_payload)->IsDecoupled() &&
      (flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) != 0) {
    // An empty response may be the last response for decoupled models.
    infer_response = std::make_unique<InferResponse>(
        output_tensors, pb_error, "" /* parameters */,
        true /* is_last_response */, userp /* id */);
  } else {
    pb_error = std::make_shared<PbError>("Unexpected empty response.");
    infer_response = std::make_unique<InferResponse>(
        output_tensors, pb_error, "" /* parameters */,
        true /* is_last_response */, userp /* id */);
  }

  infer_payload->SetValue(std::move(infer_response));
}

TRITONSERVER_Error*
ResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  auto p = reinterpret_cast<ResponseAllocatorUserp*>(userp);
  std::unique_ptr<SharedMemoryManager> shm_pool(
      reinterpret_cast<SharedMemoryManager*>(p->shm_pool));

  ScopedDefer _([&shm_pool] { shm_pool.release(); });

  if (p->preferred_memory.PreferredMemoryType() ==
      PreferredMemory::MemoryType::kDefault) {
    *actual_memory_type = preferred_memory_type;
    *actual_memory_type_id = preferred_memory_type_id;
  } else {
    TRITONSERVER_MemoryType user_preferred_memory_type;
    RETURN_IF_ERROR(MemoryTypeToTritonMemoryType(
        &user_preferred_memory_type,
        p->preferred_memory.PreferredMemoryType()));
    *actual_memory_type = user_preferred_memory_type;
    *actual_memory_type_id = p->preferred_memory.PreferredDeviceId();
  }

  // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
  // need to do any other book-keeping.
  if (byte_size == 0) {
    *buffer = nullptr;
    *buffer_userp = nullptr;
  } else {
    switch (*actual_memory_type) {
      case TRITONSERVER_MEMORY_CPU:
#ifndef TRITON_ENABLE_GPU
      case TRITONSERVER_MEMORY_GPU:
#endif
      case TRITONSERVER_MEMORY_CPU_PINNED: {
        *actual_memory_type = TRITONSERVER_MEMORY_CPU;
        *actual_memory_type_id = 0;
        try {
          std::unique_ptr<PbMemory> pb_memory = PbMemory::Create(
              shm_pool, *actual_memory_type, *actual_memory_type_id, byte_size,
              nullptr /* data */, false /* copy_gpu */);
          *buffer = pb_memory->DataPtr();
          *buffer_userp = reinterpret_cast<void*>(pb_memory.get());
          pb_memory.release();
        }
        catch (const PythonBackendException& pb_exception) {
          TRITONSERVER_Error* err =
              CreateTritonErrorFromException(pb_exception);
          return err;
        }

      } break;
#ifdef TRITON_ENABLE_GPU
      case TRITONSERVER_MEMORY_GPU: {
        BackendMemory* backend_memory;
        std::unique_ptr<BackendMemory> lbackend_memory;
        try {
          THROW_IF_TRITON_ERROR(BackendMemory::Create(
              reinterpret_cast<TRITONBACKEND_MemoryManager*>(
                  shm_pool->GetCUDAMemoryPoolManager()->TritonMemoryManager()),
              {BackendMemory::AllocationType::GPU_POOL,
               BackendMemory::AllocationType::GPU},
              *actual_memory_type_id, byte_size, &backend_memory));
          lbackend_memory.reset(backend_memory);

          std::unique_ptr<PbMemory> pb_memory = PbMemory::Create(
              shm_pool, std::move(lbackend_memory), true /* copy_gpu */);
          *buffer = pb_memory->DataPtr();
          *buffer_userp = reinterpret_cast<void*>(pb_memory.get());
          pb_memory.release();
        }
        catch (const PythonBackendException& pb_exception) {
          TRITONSERVER_Error* err =
              CreateTritonErrorFromException(pb_exception);
          return err;
        }
        break;
      }
#endif
    }
  }

  return nullptr;  // Success
}

void
InferRequestCancel(intptr_t request_address)
{
  if (request_address == 0L) {
    return;
  }

  TRITONSERVER_InferenceRequest* irequest =
      reinterpret_cast<TRITONSERVER_InferenceRequest*>(request_address);
  THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestCancel(irequest));
}

TRITONSERVER_Error*
OutputBufferQuery(
    TRITONSERVER_ResponseAllocator* allocator, void* userp,
    const char* tensor_name, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
{
  // Always attempt to return the memory in the requested memory_type and
  // memory_type_id.
  return nullptr;  // Success
}

TRITONSERVER_Error*
ResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  return nullptr;  // Success
}

RequestExecutor::RequestExecutor(
    std::unique_ptr<SharedMemoryManager>& shm_pool, TRITONSERVER_Server* server)
    : server_(server), shm_pool_(shm_pool)
{
  TRITONSERVER_ResponseAllocator* allocator;
  THROW_IF_TRITON_ERROR(TRITONSERVER_ResponseAllocatorNew(
      &allocator, ResponseAlloc, ResponseRelease, nullptr /* start_fn */));
  THROW_IF_TRITON_ERROR(TRITONSERVER_ResponseAllocatorSetQueryFunction(
      allocator, OutputBufferQuery));
  response_allocator_ = allocator;
}

std::future<std::unique_ptr<InferResponse>>
RequestExecutor::Infer(
    std::shared_ptr<InferRequest>& infer_request,
    std::shared_ptr<InferPayload>& infer_payload)
{
  std::future<std::unique_ptr<InferResponse>> response_future;
  std::unique_ptr<InferResponse> infer_response;
  bool is_ready = false;
  const char* model_name = infer_request->ModelName().c_str();
  TRITONSERVER_InferenceRequest* irequest = nullptr;
  RequestCompletionUserp* completion_userp = nullptr;

  try {
    int64_t model_version = infer_request->ModelVersion();
    THROW_IF_TRITON_ERROR(TRITONSERVER_ServerModelIsReady(
        server_, model_name, model_version, &is_ready));

    if (!is_ready) {
      throw PythonBackendException(
          (std::string("Failed for execute the inference request. Model '") +
           model_name + "' is not ready.")
              .c_str());
    }

    uint32_t txn_flags;
    THROW_IF_TRITON_ERROR(TRITONSERVER_ServerModelTransactionProperties(
        server_, model_name, model_version, &txn_flags, nullptr /* voidp */));
    infer_request->SetIsDecoupled(
        (txn_flags & TRITONSERVER_TXN_DECOUPLED) != 0);

    if (!infer_payload->IsDecoupled() && infer_request->IsDecoupled()) {
      // Decoupled API is only supported by using stream API
      throw PythonBackendException(
          std::string("Model ") + model_name +
          " is using the decoupled. The current BLS request call doesn't "
          "support models using the decoupled transaction policy. Please use "
          "'decoupled=True' argument to the 'exec' or 'async_exec' calls for "
          "decoupled models.'");
    }

    // Inference
    THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestNew(
        &irequest, server_, model_name, model_version));

    THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestSetId(
        irequest, infer_request->RequestId().c_str()));

    if (infer_request->GetCorrelationId().Type() ==
        CorrelationIdDataType::UINT64) {
      THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestSetCorrelationId(
          irequest, infer_request->GetCorrelationId().UnsignedIntValue()));
    } else {
      THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestSetCorrelationIdString(
          irequest, infer_request->GetCorrelationId().StringValue().c_str()));
    }

    THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestSetFlags(
        irequest, infer_request->Flags()));

    THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(
        irequest, infer_request->Timeout()));

    completion_userp = new RequestCompletionUserp(infer_payload);
    THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestSetReleaseCallback(
        irequest, InferRequestComplete,
        reinterpret_cast<void*>(completion_userp)));

    TRITONSERVER_InferenceTrace* trace = nullptr;
    if (infer_request->GetTrace().TritonTrace() != nullptr) {
      THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceTraceSpawnChildTrace(
          reinterpret_cast<TRITONSERVER_InferenceTrace*>(
              infer_request->GetTrace().TritonTrace()),
          &trace));
    }

    const std::string& param_str = infer_request->Parameters();
    triton::common::TritonJson::Value param;
    THROW_IF_TRITON_ERROR(param.Parse(param_str.c_str(), param_str.length()));
    std::vector<std::string> param_keys;
    THROW_IF_TRITON_ERROR(param.Members(&param_keys));
    for (const auto& key : param_keys) {
      triton::common::TritonJson::Value value;
      if (!param.Find(key.c_str(), &value)) {
        throw PythonBackendException("Unexpected missing key on parameters");
      }
      if (value.IsString()) {
        std::string string_value;
        THROW_IF_TRITON_ERROR(value.AsString(&string_value));
        THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestSetStringParameter(
            irequest, key.c_str(), string_value.c_str()));
      } else if (value.IsInt()) {
        int64_t int_value = 0;
        THROW_IF_TRITON_ERROR(value.AsInt(&int_value));
        THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestSetIntParameter(
            irequest, key.c_str(), int_value));
      } else if (value.IsBool()) {
        bool bool_value = false;
        THROW_IF_TRITON_ERROR(value.AsBool(&bool_value));
        THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestSetBoolParameter(
            irequest, key.c_str(), bool_value));
      } else {
        throw PythonBackendException("Unsupported value type on parameters");
      }
    }

    for (auto& infer_input : infer_request->Inputs()) {
      THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestAddInput(
          irequest, infer_input->Name().c_str(),
          static_cast<TRITONSERVER_DataType>(infer_input->TritonDtype()),
          infer_input->Dims().data(), infer_input->Dims().size()));

      THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestAppendInputData(
          irequest, infer_input->Name().c_str(), infer_input->DataPtr(),
          infer_input->ByteSize(), infer_input->MemoryType(),
          infer_input->MemoryTypeId()));
    }

    for (auto& requested_output_name : infer_request->RequestedOutputNames()) {
      THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestAddRequestedOutput(
          irequest, requested_output_name.c_str()));
    }

    {
      infer_payload->SetFuture(response_future);

      ResponseAllocatorUserp response_allocator_userp(
          shm_pool_.get(), infer_request->GetPreferredMemory());
      infer_payload->SetResponseAllocUserp(response_allocator_userp);

      THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestSetResponseCallback(
          irequest, response_allocator_,
          reinterpret_cast<void*>(infer_payload->ResponseAllocUserp().get()),
          InferResponseComplete, reinterpret_cast<void*>(infer_payload.get())));

      // Store the inference request address submitted to the Triton server for
      // retrieval
      infer_payload->SetRequestAddress(reinterpret_cast<intptr_t>(irequest));
      infer_payload->SetRequestCancellationFunc(InferRequestCancel);

      THROW_IF_TRITON_ERROR(
          TRITONSERVER_ServerInferAsync(server_, irequest, trace));
    }
  }
  catch (const PythonBackendException& pb_exception) {
    infer_payload->SetRequestAddress(0L);
    if (completion_userp != nullptr) {
      delete completion_userp;
    }

    LOG_IF_ERROR(
        TRITONSERVER_InferenceRequestDelete(irequest),
        "Failed to delete inference request.");

    throw PythonBackendException(
        std::string("Model ") + model_name +
        " - Error when running inference: " + pb_exception.what());
  }

  return response_future;
}

RequestExecutor::~RequestExecutor()
{
  if (response_allocator_ != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_ResponseAllocatorDelete(response_allocator_),
        "Failed to delete allocator.");
  }
}
}}};  // namespace triton::backend::python
