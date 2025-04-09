// Copyright 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "infer_handler.h"

#ifndef NDEBUG
uint64_t
NextUniqueId()
{
  static std::atomic<uint64_t> id(0);
  return ++id;
}
#endif  // NDEBUG

namespace triton { namespace server { namespace grpc {

TRITONSERVER_Error*
OutputBufferAttributesHelper(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    const TensorShmMap& shm_map,
    TRITONSERVER_BufferAttributes* buffer_attributes)
{
  // We only need to set the cuda ipc handle here. The rest of the buffer
  // attributes have been properly populated by triton core.
  if (tensor_name != nullptr) {
    const auto& pr = shm_map.find(tensor_name);

    if (pr != shm_map.end()) {
      if (pr->second.memory_type_ == TRITONSERVER_MEMORY_GPU) {
        RETURN_IF_ERR(TRITONSERVER_BufferAttributesSetCudaIpcHandle(
            buffer_attributes, pr->second.cuda_ipc_handle_));
      }
    }
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
OutputBufferQueryHelper(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t* byte_size, const TensorShmMap& shm_map,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
{
  // Check if shared memory is used if named tensor is provided
  if (tensor_name != nullptr) {
    const auto& pr = shm_map.find(tensor_name);
    if (pr != shm_map.end()) {
      // The output is in shared memory so check that shared memory
      // size is at least large enough for the output, if byte size is provided
      if ((byte_size != nullptr) && (*byte_size > pr->second.byte_size_)) {
        // Don't return error yet and just set to the default properties for
        // GRPC buffer, error will be raised when allocation happens
        *memory_type = TRITONSERVER_MEMORY_CPU;
        *memory_type_id = 0;
      } else {
        *memory_type = pr->second.memory_type_;
        *memory_type_id = pr->second.memory_type_id_;
      }
      return nullptr;  // Success
    }
  }

  // Not using shared memory so a buffer created directly in
  // the response protobuf will be used, and the type will be CPU.
  *memory_type = TRITONSERVER_MEMORY_CPU;
  *memory_type_id = 0;
  return nullptr;  // Success
}

// Make sure to keep InferResponseAlloc and OutputBufferQuery logic in sync
TRITONSERVER_Error*
InferResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  AllocPayload<inference::ModelInferResponse>* payload =
      reinterpret_cast<AllocPayload<inference::ModelInferResponse>*>(userp);

  // ModelInfer RPC expects exactly one response per request. Hence,
  // will be creating and using just one response object.
  inference::ModelInferResponse* response =
      payload->response_queue_->GetNonDecoupledResponse();
  return ResponseAllocatorHelper(
      allocator, tensor_name, byte_size, preferred_memory_type,
      preferred_memory_type_id, response, payload->shm_map_, buffer,
      buffer_userp, actual_memory_type, actual_memory_type_id);
}

// Make sure to keep InferResponseAllocCallback and OutputBufferQuery logic in
// sync
TRITONSERVER_Error*
InferResponseAllocCallback(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  AllocPayloadCallback<inference::ModelInferResponse>* payload =
      reinterpret_cast<AllocPayloadCallback<inference::ModelInferResponse>*>(
          userp);

  // ModelInfer RPC expects exactly one response per request. Hence,
  // Get pointer directly from the modified payload instead of the queue.
  inference::ModelInferResponse* response = payload->response_ptr_;
  return ResponseAllocatorHelper(
      allocator, tensor_name, byte_size, preferred_memory_type,
      preferred_memory_type_id, response, payload->shm_map_, buffer,
      buffer_userp, actual_memory_type, actual_memory_type_id);
}

// Make sure to keep InferResponseAlloc and OutputBufferQuery logic in sync
TRITONSERVER_Error*
OutputBufferQuery(
    TRITONSERVER_ResponseAllocator* allocator, void* userp,
    const char* tensor_name, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
{
  AllocPayloadCallback<inference::ModelInferResponse>* payload =
      reinterpret_cast<AllocPayloadCallback<inference::ModelInferResponse>*>(
          userp);

  return OutputBufferQueryHelper(
      allocator, tensor_name, byte_size, payload->shm_map_, memory_type,
      memory_type_id);
}

// Make sure to keep InferResponseAlloc, OutputBufferQuery, and
// OutputBufferAttributes logic in sync
TRITONSERVER_Error*
OutputBufferAttributes(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    TRITONSERVER_BufferAttributes* buffer_attributes, void* userp,
    void* buffer_userp)
{
  AllocPayloadCallback<inference::ModelInferResponse>* payload =
      reinterpret_cast<AllocPayloadCallback<inference::ModelInferResponse>*>(
          userp);

  return OutputBufferAttributesHelper(
      allocator, tensor_name, payload->shm_map_, buffer_attributes);
  return nullptr;  // Success
}

TRITONSERVER_Error*
InferResponseFree(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  LOG_VERBOSE(1) << "GRPC free: "
                 << "size " << byte_size << ", addr " << buffer;

  // Don't do anything when releasing a buffer since InferResponseAlloc
  // wrote directly into the response protobuf.
  return nullptr;  // Success
}

TRITONSERVER_Error*
InferGRPCToInputHelper(
    const std::string& input_name, const std::string& model_name,
    const TRITONSERVER_DataType tensor_dt, const TRITONSERVER_DataType input_dt,
    const size_t binary_data_byte_size)
{
  if (binary_data_byte_size != 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected explicit tensor data for input tensor '" + input_name +
            "' for model '" + model_name +
            "', binary data was already supplied.")
            .c_str());
  }

  if (tensor_dt != input_dt) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected explicit tensor data for input tensor '" + input_name +
            "' for model '" + model_name + "' of type '" +
            TRITONSERVER_DataTypeString(tensor_dt) + "', expected datatype '" +
            TRITONSERVER_DataTypeString(input_dt) + "'")
            .c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
InferResponseStart(TRITONSERVER_ResponseAllocator* allocator, void* userp)
{
  // AllocPayload<inference::ModelInferResponse>* payload =
  //     reinterpret_cast<AllocPayload<inference::ModelInferResponse>*>(userp);

  // ModelInfer RPC expects exactly one response per request. Hence, always call
  // GetNonDecoupledResponse() to create one response object on response start.
  // payload->response_queue_->GetNonDecoupledResponse();

  return nullptr;  // success
}

TRITONSERVER_Error*
SetInferenceRequestMetadata(
    TRITONSERVER_InferenceRequest* inference_request,
    const inference::ModelInferRequest& request, StateParameters& state_params)
{
  RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetId(
      inference_request, request.id().c_str()));

  uint32_t flags = 0;
  for (auto param : request.parameters()) {
    if (param.first.compare("sequence_id") == 0) {
      const auto& infer_param = param.second;
      if (infer_param.parameter_choice_case() ==
          inference::InferParameter::ParameterChoiceCase::kInt64Param) {
        RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetCorrelationId(
            inference_request, infer_param.int64_param()));
      } else if (
          infer_param.parameter_choice_case() ==
          inference::InferParameter::ParameterChoiceCase::kStringParam) {
        RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetCorrelationIdString(
            inference_request, infer_param.string_param().c_str()));
      } else {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "invalid value type for 'sequence_id' parameter, expected "
            "int64_param or string_param.");
      }
    } else if (param.first.compare("sequence_start") == 0) {
      const auto& infer_param = param.second;
      if (infer_param.parameter_choice_case() !=
          inference::InferParameter::ParameterChoiceCase::kBoolParam) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "invalid value type for 'sequence_start' parameter, expected "
            "bool_param.");
      }
      if (infer_param.bool_param()) {
        flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_START;
      }
    } else if (param.first.compare("sequence_end") == 0) {
      const auto& infer_param = param.second;
      if (infer_param.parameter_choice_case() !=
          inference::InferParameter::ParameterChoiceCase::kBoolParam) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "invalid value type for 'sequence_end' parameter, expected "
            "bool_param.");
      }
      if (infer_param.bool_param()) {
        flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_END;
      }
    } else if (param.first.compare("priority") == 0) {
      const auto& infer_param = param.second;
      if (infer_param.parameter_choice_case() ==
          inference::InferParameter::ParameterChoiceCase::kInt64Param) {
        if (infer_param.int64_param() < 0) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              "invalid value for 'priority', expected value >= 0.");
        }
        RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetPriorityUInt64(
            inference_request, infer_param.int64_param()));
      } else if (
          infer_param.parameter_choice_case() ==
          inference::InferParameter::ParameterChoiceCase::kUint64Param) {
        RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetPriorityUInt64(
            inference_request, infer_param.uint64_param()));
      } else {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "invalid value type for 'priority' parameter, expected "
            "int64_param or uint64_param.");
      }
    } else if (param.first.compare("timeout") == 0) {
      const auto& infer_param = param.second;
      if (infer_param.parameter_choice_case() !=
          inference::InferParameter::ParameterChoiceCase::kInt64Param) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "invalid value type for 'timeout' parameter, expected "
            "int64_param.");
      }
      RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(
          inference_request, infer_param.int64_param()));
    } else if (param.first.rfind("triton_", 0) == 0) {
      if (!Contains(TRITON_RESERVED_REQUEST_PARAMS, param.first)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string(
                 "parameter keys starting with 'triton_' are reserved for "
                 "Triton "
                 "usage. Only the following keys starting with 'triton_' are "
                 "allowed: ") +
             Join(TRITON_RESERVED_REQUEST_PARAMS, " "))
                .c_str());
      }
      RETURN_IF_ERR(SetStateParameterFromTritonParameter(state_params, param));
    } else {
      const auto& infer_param = param.second;
      if (infer_param.parameter_choice_case() ==
          inference::InferParameter::ParameterChoiceCase::kInt64Param) {
        RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetIntParameter(
            inference_request, param.first.c_str(), infer_param.int64_param()));
      } else if (
          infer_param.parameter_choice_case() ==
          inference::InferParameter::ParameterChoiceCase::kBoolParam) {
        RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetBoolParameter(
            inference_request, param.first.c_str(), infer_param.bool_param()));
      } else if (
          infer_param.parameter_choice_case() ==
          inference::InferParameter::ParameterChoiceCase::kStringParam) {
        RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetStringParameter(
            inference_request, param.first.c_str(),
            infer_param.string_param().c_str()));
      } else if (
          infer_param.parameter_choice_case() ==
          inference::InferParameter::ParameterChoiceCase::kDoubleParam) {
        RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetDoubleParameter(
            inference_request, param.first.c_str(),
            infer_param.double_param()));
      } else {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string(
                "invalid value type for '" + param.first +
                "' parameter, expected "
                "int64_param, bool_param, or string_param.")
                .c_str());
      }
    }
  }

  RETURN_IF_ERR(
      TRITONSERVER_InferenceRequestSetFlags(inference_request, flags));

  for (const auto& input : request.inputs()) {
    RETURN_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
        inference_request, input.name().c_str(),
        TRITONSERVER_StringToDataType(input.datatype().c_str()),
        input.shape().data(), input.shape_size()));
  }

  for (const auto& output : request.outputs()) {
    RETURN_IF_ERR(TRITONSERVER_InferenceRequestAddRequestedOutput(
        inference_request, output.name().c_str()));
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
SetStateParameterFromTritonParameter(
    StateParameters& state_params,
    const std::pair<std::string, inference::InferParameter>& param)
{
  const auto& key = param.first;
  const auto& value = param.second;
  if (key == "triton_enable_empty_final_response") {
    if (value.parameter_choice_case() !=
        inference::InferParameter::ParameterChoiceCase::kBoolParam) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("invalid value type for '") + key +
           std::string("' parameter, expected bool_param."))
              .c_str());
    }
    state_params.enable_empty_final_response_ = value.bool_param();
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
InferGRPCToInput(
    const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
    const std::shared_ptr<SharedMemoryManager>& shm_manager,
    const inference::ModelInferRequest& request,
    std::list<std::string>* serialized_data,
    TRITONSERVER_InferenceRequest* inference_request,
    std::vector<std::shared_ptr<const SharedMemoryManager::SharedMemoryInfo>>*
        shm_regions_info)
{
  // Verify that the batch-byte-size of each input matches the size of
  // the provided tensor data (provided raw or from shared memory)
  int index = 0;
  for (const auto& io : request.inputs()) {
    const void* base;
    size_t byte_size = 0;
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;

    std::string region_name;
    int64_t offset;
    bool has_shared_memory;
    RETURN_IF_ERR(
        ParseSharedMemoryParams<inference::ModelInferRequest::InferInputTensor>(
            io, &has_shared_memory, &region_name, &offset, &byte_size));

    TRITONSERVER_BufferAttributes* buffer_attributes;
    RETURN_IF_ERR(TRITONSERVER_BufferAttributesNew(&buffer_attributes));
    auto buffer_attributes_del =
        [](TRITONSERVER_BufferAttributes* buffer_attributes) {
          TRITONSERVER_BufferAttributesDelete(buffer_attributes);
        };
    std::unique_ptr<
        TRITONSERVER_BufferAttributes, decltype(buffer_attributes_del)>
        buffer_attrsl(buffer_attributes, buffer_attributes_del);
    char* cuda_ipc_handle = nullptr;

    if (has_shared_memory) {
      if (io.has_contents()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string(
                "unexpected 'content' provided when using shared memory "
                "for "
                "input tensor '" +
                io.name() + "' for model '" + request.model_name() + "'")
                .c_str());
      }
      void* tmp;
      std::shared_ptr<const SharedMemoryManager::SharedMemoryInfo> shm_info =
          nullptr;
      RETURN_IF_ERR(shm_manager->GetMemoryInfo(
          region_name, offset, byte_size, &tmp, &memory_type, &memory_type_id,
          &shm_info));
      base = tmp;
      shm_regions_info->emplace_back(shm_info);

      if (memory_type == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
        RETURN_IF_ERR(shm_manager->GetCUDAHandle(
            region_name,
            reinterpret_cast<cudaIpcMemHandle_t**>(&cuda_ipc_handle)));
#endif
      }
    } else {
      if (io.has_contents() && (!request.raw_input_contents().empty())) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string(
                "contents field must not be specified when using "
                "raw_input_contents for '" +
                io.name() + "' for model '" + request.model_name() + "'")
                .c_str());
      } else if (io.has_contents()) {
        // Check the presence of explicit tensors
        TRITONSERVER_DataType dtype =
            TRITONSERVER_StringToDataType(io.datatype().c_str());
        const size_t elem_byte_size = TRITONSERVER_DataTypeByteSize(dtype);
        if (io.contents().bool_contents_size() != 0) {
          RETURN_IF_ERR(InferGRPCToInputHelper(
              io.name(), request.model_name(), TRITONSERVER_TYPE_BOOL, dtype,
              byte_size));
          base = (const void*)io.contents().bool_contents().data();
          byte_size = io.contents().bool_contents_size() * elem_byte_size;
        }

        if (io.contents().int_contents_size() != 0) {
          if (dtype == TRITONSERVER_TYPE_INT8) {
            RETURN_IF_ERR(InferGRPCToInputHelper(
                io.name(), request.model_name(), TRITONSERVER_TYPE_INT8, dtype,
                byte_size));
            serialized_data->emplace_back();
            auto& serialized = serialized_data->back();
            serialized.reserve(
                io.contents().int_contents_size() * elem_byte_size);
            for (const auto& element : io.contents().int_contents()) {
              // Assuming the system is little-endian, picking the
              // least significant byte of 32-bit integer as a
              // int8 element
              serialized.append(
                  reinterpret_cast<const char*>(&element), elem_byte_size);
            }
            base = serialized.c_str();
            byte_size = serialized.size();
          } else if (dtype == TRITONSERVER_TYPE_INT16) {
            RETURN_IF_ERR(InferGRPCToInputHelper(
                io.name(), request.model_name(), TRITONSERVER_TYPE_INT16, dtype,
                byte_size));
            serialized_data->emplace_back();
            auto& serialized = serialized_data->back();
            serialized.reserve(
                io.contents().int_contents_size() * elem_byte_size);
            for (const auto& element : io.contents().int_contents()) {
              // Assuming the system is little-endian, picking the
              // least 2 significant bytes of 32-bit integer as a
              // int16 element
              serialized.append(
                  reinterpret_cast<const char*>(&element), elem_byte_size);
            }
            base = serialized.c_str();
            byte_size = serialized.size();
          } else {
            RETURN_IF_ERR(InferGRPCToInputHelper(
                io.name(), request.model_name(), TRITONSERVER_TYPE_INT32, dtype,
                byte_size));
            base = (const void*)io.contents().int_contents().data();
            byte_size = io.contents().int_contents_size() * elem_byte_size;
          }
        }

        if (io.contents().int64_contents_size() != 0) {
          RETURN_IF_ERR(InferGRPCToInputHelper(
              io.name(), request.model_name(), TRITONSERVER_TYPE_INT64, dtype,
              byte_size));
          base = (const void*)io.contents().int64_contents().data();
          byte_size = io.contents().int64_contents_size() * elem_byte_size;
        }

        if (io.contents().uint_contents_size() != 0) {
          if (dtype == TRITONSERVER_TYPE_UINT8) {
            RETURN_IF_ERR(InferGRPCToInputHelper(
                io.name(), request.model_name(), TRITONSERVER_TYPE_UINT8, dtype,
                byte_size));
            serialized_data->emplace_back();
            auto& serialized = serialized_data->back();
            serialized.reserve(
                io.contents().uint_contents_size() * elem_byte_size);
            for (const auto& element : io.contents().uint_contents()) {
              // Assuming the system is little-endian, picking the
              // least significant byte of 32-bit unsigned integer as a
              // uint8 element
              serialized.append(
                  reinterpret_cast<const char*>(&element), elem_byte_size);
            }
            base = serialized.c_str();
            byte_size = serialized.size();
          } else if (dtype == TRITONSERVER_TYPE_UINT16) {
            RETURN_IF_ERR(InferGRPCToInputHelper(
                io.name(), request.model_name(), TRITONSERVER_TYPE_UINT16,
                dtype, byte_size));
            serialized_data->emplace_back();
            auto& serialized = serialized_data->back();
            serialized.reserve(
                io.contents().uint_contents_size() * elem_byte_size);
            for (const auto& element : io.contents().uint_contents()) {
              // Assuming the system is little-endian, picking the
              // least 2 significant bytes of 32-bit integer as a
              // uint16 element
              serialized.append(
                  reinterpret_cast<const char*>(&element), elem_byte_size);
            }
            base = serialized.c_str();
            byte_size = serialized.size();
          } else {
            RETURN_IF_ERR(InferGRPCToInputHelper(
                io.name(), request.model_name(), TRITONSERVER_TYPE_UINT32,
                dtype, byte_size));
            base = (const void*)io.contents().uint_contents().data();
            byte_size = io.contents().uint_contents_size() * elem_byte_size;
          }
        }

        if (io.contents().uint64_contents_size() != 0) {
          RETURN_IF_ERR(InferGRPCToInputHelper(
              io.name(), request.model_name(), TRITONSERVER_TYPE_UINT64, dtype,
              byte_size));
          base = (const void*)io.contents().uint64_contents().data();
          byte_size = io.contents().uint64_contents_size() * elem_byte_size;
        }

        if (io.contents().fp32_contents_size() != 0) {
          RETURN_IF_ERR(InferGRPCToInputHelper(
              io.name(), request.model_name(), TRITONSERVER_TYPE_FP32, dtype,
              byte_size));
          base = (const void*)io.contents().fp32_contents().data();
          byte_size = io.contents().fp32_contents_size() * elem_byte_size;
        }

        if (io.contents().fp64_contents_size() != 0) {
          RETURN_IF_ERR(InferGRPCToInputHelper(
              io.name(), request.model_name(), TRITONSERVER_TYPE_FP64, dtype,
              byte_size));
          base = (const void*)io.contents().fp64_contents().data();
          byte_size = io.contents().fp64_contents_size() * elem_byte_size;
        }

        if (io.contents().bytes_contents_size() != 0) {
          RETURN_IF_ERR(InferGRPCToInputHelper(
              io.name(), request.model_name(), TRITONSERVER_TYPE_BYTES, dtype,
              byte_size));

          serialized_data->emplace_back();
          auto& serialized = serialized_data->back();

          // Serialize the output tensor strings. Each string is
          // serialized as a 4-byte length followed by the string itself
          // with no null-terminator.
          for (const auto& element : io.contents().bytes_contents()) {
            uint32_t len{(uint32_t)element.size()};
            serialized.append(
                reinterpret_cast<const char*>(&len), sizeof(uint32_t));
            if (element.size() > 0) {
              serialized.append(element.c_str(), len);
            }
          }
          base = serialized.c_str();
          byte_size = serialized.size();
        }
      } else if (request.raw_input_contents().size() > index) {
        // Try to read the raw contents if available
        const std::string& raw = request.raw_input_contents()[index++];
        base = raw.c_str();
        byte_size = raw.size();
      } else {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string(
                "unable to find data for input tensor '" + io.name() +
                "' for model '" + request.model_name() + "' in request.")
                .c_str());
      }
    }

    if (cuda_ipc_handle != nullptr) {
      RETURN_IF_ERR(TRITONSERVER_BufferAttributesSetCudaIpcHandle(
          buffer_attributes, reinterpret_cast<void*>(cuda_ipc_handle)));
    }

    RETURN_IF_ERR(TRITONSERVER_BufferAttributesSetMemoryType(
        buffer_attributes, memory_type));
    RETURN_IF_ERR(TRITONSERVER_BufferAttributesSetMemoryTypeId(
        buffer_attributes, memory_type_id));
    RETURN_IF_ERR(
        TRITONSERVER_BufferAttributesSetByteSize(buffer_attributes, byte_size));
    RETURN_IF_ERR(
        TRITONSERVER_InferenceRequestAppendInputDataWithBufferAttributes(
            inference_request, io.name().c_str(), base, buffer_attributes));
  }

  return nullptr;  // success
}

void
InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  LOG_VERBOSE(1) << "ModelInferHandler::InferRequestComplete!";

  RequestReleasePayload* request_release_payload =
      static_cast<RequestReleasePayload*>(userp);

  if ((flags & TRITONSERVER_REQUEST_RELEASE_ALL) != 0) {
    delete request_release_payload;
  }
}

ModelInferCallbackHandler::ModelInferCallbackHandler(
    const std::string& name,
    const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
    TraceManager* trace_manager,
    const std::shared_ptr<SharedMemoryManager>& shm_manager,
    grpc_compression_level compression_level,
    RestrictedFeatures& restricted_keys,
    const std::string& forward_header_pattern)
    : name_(name), tritonserver_(tritonserver), trace_manager_(trace_manager),
      shm_manager_(shm_manager), compression_level_(compression_level),
      restricted_kv_(restricted_keys.Get(RestrictedCategory::INFERENCE)),
      header_forward_pattern_(forward_header_pattern),
      header_forward_regex_(forward_header_pattern)
{
  FAIL_IF_ERR(
      TRITONSERVER_ResponseAllocatorNew(
          &allocator_, InferResponseAllocCallback, InferResponseFree,
          InferResponseStart),
      "creating inference response allocator");
  FAIL_IF_ERR(
      TRITONSERVER_ResponseAllocatorSetQueryFunction(
          allocator_, OutputBufferQuery),
      "setting allocator's query function");
  FAIL_IF_ERR(
      TRITONSERVER_ResponseAllocatorSetBufferAttributesFunction(
          allocator_, OutputBufferAttributes),
      "setting allocator's output buffer attributes function");
}

ModelInferCallbackHandler::~ModelInferCallbackHandler()
{
  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_ResponseAllocatorDelete(allocator_),
      "deleting response allocator");
}

/**
 * @brief Handles gRPC ModelInfer requests using the callback API pattern
 *
 * Request flow path:
 * 1. Client creates and sends ModelInferRequest via gRPC
 * 2. gRPC framework deserializes the protobuf message
 * 3. gRPC calls this handler based on service registration
 * 4. This function creates a callback state and reactor to manage async
 * lifecycle
 * 5. The Execute method initiates processing with proper ownership transfer
 *
 * Memory management:
 * - CallbackState manages lifecycle of request/response objects
 * - Ownership transfers to completion callbacks for async cleanup
 * - Response memory allocation handled through allocator_
 * - Shared memory regions tracked and released after completion
 *
 * @param context The gRPC server context for this request
 * @param request The deserialized ModelInferRequest from client
 * @param response Output parameter for the ModelInferResponse to client
 * @return ::grpc::ServerUnaryReactor* Reactor that signals request completion
 */
::grpc::ServerUnaryReactor*
ModelInferCallbackHandler::HandleModelInfer(
    ::grpc::CallbackServerContext* context,
    const inference::ModelInferRequest* request,
    inference::ModelInferResponse* response)
{
  auto* reactor = context->DefaultReactor();

  // Check preconditions
  if (!ExecutePrecondition(context)) {
    reactor->Finish(::grpc::Status(
        ::grpc::StatusCode::UNAVAILABLE, "This protocol is restricted"));
    return reactor;
  }

  // Create callback state
  auto callback_state = std::make_unique<CallbackState>(
      response, reactor, context, tritonserver_);

  // Execute the request
  Execute(context, request, response, reactor, callback_state);

  return reactor;
}

void
ModelInferCallbackHandler::InferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  LOG_VERBOSE(1) << "[InferResponseComplete START] Received userp "
                    "(CallbackState*) address: "
                 << userp;
  std::unique_ptr<CallbackState> callback_state(
      static_cast<CallbackState*>(userp));
  LOG_VERBOSE(1) << "[InferResponseComplete] CallbackState unique_ptr now owns "
                    "state at address: "
                 << callback_state.get();
  if (response != nullptr) {
    // Use the pre-allocated response directly from the callback state
    ::grpc::Status status = ::grpc::Status::OK;

    // Get the response from the payload's response queue as a fallback
    LOG_VERBOSE(1)
        << "[InferResponseComplete] Attempting to retrieve response pointer "
           "directly from callback_state->response_ which points to: "
        << callback_state->response_;
    inference::ModelInferResponse* grpc_response = callback_state->response_;

    // If not available in callback state, try to get from response queue
    if (grpc_response == nullptr) {
      LOG_VERBOSE(1)
          << "[InferResponseComplete] >>> Fallback Triggered! grpc_response "
             "from state was NULL, attempting fallback from queue.";
      grpc_response = callback_state->alloc_payload_.response_ptr_;
    }

    if (grpc_response != nullptr) {
      // Process the response
      LOG_VERBOSE(1)
          << "InferResponseComplete: Checking response object at address: "
          << grpc_response;
      TRITONSERVER_Error* err = InferResponseCompleteCommonCallback(
          callback_state->tritonserver_.get(), response, *grpc_response,
          callback_state->alloc_payload_);

      if (err != nullptr) {
        GrpcStatusUtil::Create(&status, err);
        TRITONSERVER_ErrorDelete(err);
      }
    } else {
      status = ::grpc::Status(
          ::grpc::StatusCode::INTERNAL,
          "response object not found in callback");
    }

    // For callback API, we complete the RPC by finishing the reactor
    // Only finish the reactor when we get the final response or on error
    if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) || !status.ok()) {
      callback_state->reactor_->Finish(status);
    }
  } else {
    // Handle null response case
    callback_state->reactor_->Finish(
        ::grpc::Status(::grpc::StatusCode::INTERNAL, "null response"));
  }

#ifdef TRITON_ENABLE_TRACING
  if (callback_state->trace_ != nullptr) {
    callback_state->trace_timestamps_.emplace_back(std::make_pair(
        "INFER_RESPONSE_COMPLETE", TraceManager::CaptureTimestamp()));
  }
#endif  // TRITON_ENABLE_TRACING

  // Always delete the TRITONSERVER_InferenceResponse
  if (response != nullptr) {
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceResponseDelete(response),
        "deleting inference response");
  }
}

bool
ModelInferCallbackHandler::ExecutePrecondition(
    ::grpc::CallbackServerContext* context)
{
  if (!restricted_kv_.first.empty()) {
    const auto& metadata = context->client_metadata();
    const auto it = metadata.find(restricted_kv_.first);
    return (it != metadata.end()) && (it->second == restricted_kv_.second);
  }
  return true;
}

// Implement the new private helper function
TRITONSERVER_Error*
ModelInferCallbackHandler::ForwardHeadersAsParametersCallback(
    TRITONSERVER_InferenceRequest* irequest,
    const ::grpc::CallbackServerContext* context)
{
  TRITONSERVER_Error* err = nullptr;
  // Use the members stored in *this* specific handler instance
  if (!header_forward_pattern_.empty()) {
    const auto& metadata =
        context->client_metadata();  // Use the passed context
    for (const auto& pair : metadata) {
      // Need to convert grpc::string_ref to std::string for RE2/Triton API
      std::string key_str(pair.first.data(), pair.first.length());
      std::string value_str(pair.second.data(), pair.second.length());

      // Use the regex member stored in *this* handler instance
      if (RE2::PartialMatch(key_str, header_forward_regex_)) {
        err = TRITONSERVER_InferenceRequestSetStringParameter(
            irequest, key_str.c_str(), value_str.c_str());
        if (err != nullptr) {
          break;  // Exit loop on error
        }
      }
    }
  }
  return err;
}

void
ModelInferCallbackHandler::Execute(
    ::grpc::CallbackServerContext* context,
    const inference::ModelInferRequest* request,
    inference::ModelInferResponse* response,
    ::grpc::ServerUnaryReactor* reactor,
    std::unique_ptr<CallbackState>& callback_state)
{
  TRITONSERVER_Error* err = nullptr;
  TRITONSERVER_InferenceRequest* irequest = nullptr;
  LOG_VERBOSE(1) << "[Execute START] Incoming response object address: "
                 << response;
  // --- Step 1: Receive & Validate ---
  int64_t requested_model_version;
  err = GetModelVersionFromString(
      request->model_version(), &requested_model_version);

  // Check if model has decoupled transaction policy (not supported by this RPC)
  if (err == nullptr) {
    uint32_t txn_flags;
    // Query model properties
    err = TRITONSERVER_ServerModelTransactionProperties(
        tritonserver_.get(), request->model_name().c_str(),
        requested_model_version, &txn_flags, nullptr /* voidp */);
    if ((err == nullptr) && (txn_flags & TRITONSERVER_TXN_DECOUPLED) != 0) {
      // Set error if decoupled
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "ModelInfer RPC doesn't support models with decoupled "
          "transaction policy");
    }
  }

  // --- Step 2: Prepare Triton Request Object ---
  if (err == nullptr) {
    // Create the core Triton request object
    err = TRITONSERVER_InferenceRequestNew(
        &irequest, tritonserver_.get(), request->model_name().c_str(),
        requested_model_version);
  }

  // Populate request metadata (ID, sequence flags, priority, params, etc.)
  if (err == nullptr) {
    StateParameters state_params;  // Temporary params for this call scope
    err = SetInferenceRequestMetadata(irequest, *request, state_params);
  }

  // Forward relevant gRPC headers as Triton parameters
  if (err == nullptr) {
    err = ForwardHeadersAsParametersCallback(irequest, context);
  }

  // --- Step 3: Process Input Tensors ---
  if (err == nullptr) {
    // Parse inputs from request, handle shared memory (if any),
    // serialize string data, and add data pointers/attributes to irequest.
    // Serialized data stored in callback_state->serialized_data_
    // SHM info stored in callback_state->shm_regions_info_
    err = InferGRPCToInput(
        tritonserver_, shm_manager_, *request,
        &callback_state->serialized_data_, irequest,
        &callback_state->shm_regions_info_);
  }

  // --- Step 4: Prepare for Response Handling (Callback Specific) ---
  std::shared_ptr<ResponseQueue<inference::ModelInferResponse>> response_queue =
      nullptr;
  if (err == nullptr) {
    // Use the externally provided response object directly.
    // Store the external response pointer in the state for later access.
    callback_state->response_ = response;
    LOG_VERBOSE(1) << "[Execute] Stored response object address in "
                      "callback_state->response_: "
                   << callback_state->response_;
    // Clear the externally provided response object directly.
    response->Clear();  // Ensure it's empty before Triton writes to it
  }

  // Prepare the allocator payload: info needed by allocation callback later.
  // Moves serialized input data into the payload. References the
  // response_queue.
  if (err == nullptr) {
    err = InferAllocatorPayloadCallback<inference::ModelInferResponse>(
        tritonserver_, shm_manager_, *request,
        std::move(callback_state->serialized_data_), callback_state->response_,
        &callback_state->alloc_payload_, &callback_state->shm_regions_info_);
  }

  // --- Step 5: Setup Automatic Cleanup Payloads & Register Callbacks ---
  // Create payload for request release callback (manages irequest lifetime)
  auto request_release_payload = std::make_unique<RequestReleasePayload>(
      std::shared_ptr<TRITONSERVER_InferenceRequest>(
          irequest, [](TRITONSERVER_InferenceRequest* r) {
            // Custom deleter: Ensures delete is called via shared_ptr lifecycle
            if (r != nullptr) {
              LOG_TRITONSERVER_ERROR(
                  TRITONSERVER_InferenceRequestDelete(r),
                  "deleting inference request via shared_ptr custom deleter");
            }
          }));

  // Register the release callback (cleans up request_release_payload &
  // irequest)
  if (err == nullptr) {
    err = TRITONSERVER_InferenceRequestSetReleaseCallback(
        irequest, InferRequestComplete, request_release_payload.get());
  }

  // Register the response callback (processes result, finishes RPC, cleans up
  // callback_state)
  if (err == nullptr) {
    // Note: Passing callback_state.get() transfers potential ownership to the
    // callback mechanism upon success (see step 7).
    err = TRITONSERVER_InferenceRequestSetResponseCallback(
        irequest, allocator_, &callback_state->alloc_payload_,
        InferResponseComplete, callback_state.get());
  }

  // --- Optional: Setup Tracing ---
  TRITONSERVER_InferenceTrace* triton_trace = nullptr;
#ifdef TRITON_ENABLE_TRACING
  if (err == nullptr && trace_manager_ != nullptr) {
    // Setup and start tracing if configured
    GrpcServerCarrier carrier(context);
    auto start_options =
        trace_manager_->GetTraceStartOptions(carrier, request->model_name());
    callback_state->trace_ =
        std::move(trace_manager_->SampleTrace(start_options));
    if (callback_state->trace_ != nullptr) {
      triton_trace = callback_state->trace_->trace_;
    }
  }
#endif  // TRITON_ENABLE_TRACING

  // Get request ID for logging, handle potential null irequest if error
  // occurred early
  const char* request_id_cstr = "";
  std::string request_id = "<unknown>";
  if (irequest != nullptr) {
    auto id_err = TRITONSERVER_InferenceRequestId(irequest, &request_id_cstr);
    if (id_err == nullptr && request_id_cstr != nullptr &&
        strlen(request_id_cstr) > 0) {
      request_id = request_id_cstr;
    }
    TRITONSERVER_ErrorDelete(id_err);  // Delete error from ID retrieval if any
  }


  // --- Step 6: Start Asynchronous Inference ---
  if (err == nullptr) {
    err = TRITONSERVER_ServerInferAsync(
        tritonserver_.get(), irequest, triton_trace);
  }

  // --- Step 7/8: Handle Outcome (Success or Error) ---
  if (err == nullptr) {
    // --- Success Path ---
    // Inference successfully submitted to Triton core.
    // Release ownership of payloads to the callback mechanism.
    // Callbacks (InferResponseComplete, InferRequestComplete) are now
    // responsible for cleanup.
    LOG_VERBOSE(1) << "[Execute SUCCESS] Releasing ownership of callback_state "
                      "at address: "
                   << callback_state.get();
    callback_state.release();
    request_release_payload.release();
    // Execute function finishes here; gRPC call waits for reactor->Finish() in
    // callback.
    LOG_VERBOSE(1) << "[request id: " << request_id << "] "
                   << "Async inference submitted successfully.";

  } else {
    // --- Error Path ---
    // An error occurred during setup before submitting to Triton.
    LOG_VERBOSE(1) << "[request id: " << request_id << "] "
                   << "Setup failed before submitting inference: "
                   << TRITONSERVER_ErrorMessage(err);

    // Create gRPC status from Triton error
    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);

    // Perform explicit cleanup as callbacks won't run
    TRITONSERVER_ErrorDelete(err);  // Delete the primary Triton error
    if (irequest != nullptr) {
      // Explicitly delete the request object as the release callback won't run
      // Note: The shared_ptr in request_release_payload will handle this
      // gracefully
      //       when the unique_ptr goes out of scope below, due to the custom
      //       deleter. However, explicit deletion here is safe and clear.
      LOG_TRITONSERVER_ERROR(
          TRITONSERVER_InferenceRequestDelete(irequest),
          "explicitly deleting inference request due to setup error");
      irequest =
          nullptr;  // Avoid potential double delete if shared_ptr logic changes
    }
    // Note: callback_state and request_release_payload unique_ptrs will
    //       automatically clean up their managed objects when they go out of
    //       scope now, as .release() was not called.

    // Immediately finish the gRPC call with the error status
    reactor->Finish(status);
    // Execute function finishes here.
  }
}
//===========================================================================
//  The following section contains the handling mechanism for ModelInfer RPC.
//  This implementation is tuned towards performance and reducing latency.
//===========================================================================

void
ModelInferHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>(cq_);
  context->SetCompressionLevel(compression_level_);
  State* state = StateNew(tritonserver_.get(), context);

#ifdef TRITON_ENABLE_TRACING
  // Can't create trace as we don't know the model to be requested,
  // track timestamps in 'state'
  state->trace_timestamps_.emplace_back(
      std::make_pair("GRPC_WAITREAD_START", TraceManager::CaptureTimestamp()));
#endif  // TRITON_ENABLE_TRACING

  service_->RequestModelInfer(
      state->context_->ctx_.get(), &state->request_,
      state->context_->responder_.get(), cq_, cq_, state);

  LOG_VERBOSE(1) << "New request handler for " << Name() << ", "
                 << state->unique_id_;
}

bool
ModelInferHandler::Process(
    InferHandler::State* state, bool rpc_ok, bool is_notification)
{
  // There are multiple handlers registered in the gRPC service.
  // Hence, there we can have a case where a handler thread is
  // making progress in the state machine for a request and the
  // other thread is issuing cancellation on the same request.
  // Need to protect the state transitions for these cases.
  std::lock_guard<std::recursive_mutex> lock(state->step_mtx_);

  if (state->delay_process_ms_ != 0) {
    // Will delay the Process execution by the specified time.
    // This can be used to test the flow when cancellation request
    // issued for the request, which is still at START step.
    LOG_INFO << "Delaying the Process execution by " << state->delay_process_ms_
             << " ms...";
    std::this_thread::sleep_for(
        std::chrono::milliseconds(state->delay_process_ms_));
  }

  if (is_notification) {
    state->context_->SetReceivedNotification(true);
  }

  // Handle notification for cancellation which can be raised
  // asynchronously if detected on the network.
  if (state->IsGrpcContextCancelled()) {
    if (is_notification) {
      // Received the cancellation notification
      LOG_VERBOSE(1) << "Cancellation notification received for " << Name()
                     << ", rpc_ok=" << rpc_ok << ", context "
                     << state->context_->unique_id_ << " step "
                     << state->context_->step_ << ", state "
                     << state->unique_id_ << " step " << state->step_;
    }

    bool skip_handle_cancellation = false;
    if (rpc_ok && (state->step_ == Steps::START) &&
        (state->context_->step_ != Steps::CANCELLED)) {
#ifdef TRITON_ENABLE_TRACING
      // Can't create trace as we don't know the model to be requested,
      // track timestamps in 'state'
      state->trace_timestamps_.emplace_back(std::make_pair(
          "GRPC_WAITREAD_END", TraceManager::CaptureTimestamp()));
#endif  // TRITON_ENABLE_TRACING
      // Need to create a new request object here explicitly for step START,
      // because we will never leave this if body. Refer to PR 7325.
      // This is a special case for ModelInferHandler, since we have 2 threads,
      // and each of them can process cancellation. ModelStreamInfer has only 1
      // thread, and cancellation at step START was not reproducible in a
      // single thread scenario.
      StartNewRequest();
    } else if (
        state->step_ == Steps::COMPLETE || state->step_ == Steps::FINISH) {
      // If the request is completed, simply ignore the cancellation.
      skip_handle_cancellation = true;
    }

    if (!skip_handle_cancellation) {
      bool resume = state->context_->HandleCancellation(state, rpc_ok, Name());
      return resume;
    }
  }


  LOG_VERBOSE(1) << "Process for " << Name() << ", rpc_ok=" << rpc_ok << ", "
                 << state->unique_id_ << " step " << state->step_;

  // We need an explicit finish indicator. Can't use 'state->step_'
  // because we launch an async thread that could update 'state's
  // step_ to be FINISH before this thread exits this function.
  bool finished = false;

  // If RPC failed on a new request then the server is shutting down
  // and so we should do nothing (including not registering for a new
  // request). If RPC failed on a non-START step then there is nothing
  // we can do since we one execute one step.
  const bool shutdown = (!rpc_ok && (state->step_ == Steps::START));
  if (shutdown) {
    state->step_ = Steps::FINISH;
    finished = true;
  }

  if (state->step_ == Steps::START) {
#ifdef TRITON_ENABLE_TRACING
    // Can't create trace as we don't know the model to be requested,
    // track timestamps in 'state'
    state->trace_timestamps_.emplace_back(
        std::make_pair("GRPC_WAITREAD_END", TraceManager::CaptureTimestamp()));
#endif  // TRITON_ENABLE_TRACING

    // Start a new request to replace this one...
    if (!shutdown) {
      StartNewRequest();
    }

    if (ExecutePrecondition(state)) {
      Execute(state);
    } else {
      ::grpc::Status status = ::grpc::Status(
          ::grpc::StatusCode::UNAVAILABLE,
          std::string("This protocol is restricted, expecting header '") +
              restricted_kv_.first + "'");


#ifdef TRITON_ENABLE_TRACING
      state->trace_timestamps_.emplace_back(
          std::make_pair("GRPC_SEND_START", TraceManager::CaptureTimestamp()));
#endif  // TRITON_ENABLE_TRACING

      state->step_ = Steps::COMPLETE;
      state->context_->responder_->Finish(
          inference::ModelInferResponse(), status, state);
    }

  } else if (state->step_ == Steps::COMPLETE) {
#ifdef TRITON_ENABLE_TRACING
    state->trace_timestamps_.emplace_back(
        std::make_pair("GRPC_SEND_END", TraceManager::CaptureTimestamp()));
#endif  // TRITON_ENABLE_TRACING

    state->step_ = Steps::FINISH;
  } else if (state->step_ == Steps::FINISH) {
    finished = true;
  }

  return !finished;
}

TRITONSERVER_Error*
ResponseAllocatorHelper(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, inference::ModelInferResponse* response,
    const TensorShmMap& shm_map, void** buffer, void** buffer_userp,
    TRITONSERVER_MemoryType* actual_memory_type, int64_t* actual_memory_type_id)
{
  *buffer = nullptr;
  *buffer_userp = nullptr;
  *actual_memory_type = preferred_memory_type;
  *actual_memory_type_id = preferred_memory_type_id;

  LOG_VERBOSE(1) << "AllocatorHelper: Modifying response object at address: "
                 << response;
  // We add an output contents even if the 'byte_size' == 0 because we
  // expect to have a contents for every output.
  inference::ModelInferResponse::InferOutputTensor* output_tensor =
      response->add_outputs();
  output_tensor->set_name(tensor_name);
  std::string* raw_output = response->add_raw_output_contents();
  LOG_VERBOSE(1) << "AllocatorHelper: After add_outputs for " << tensor_name
                 << ", response->outputs_size() = " << response->outputs_size();
  if (byte_size > 0) {
    const auto& pr = shm_map.find(tensor_name);
    if (pr != shm_map.end()) {
      // The output is in shared memory so check that shared memory
      // size is at least large enough for the output.
      if (byte_size > pr->second.byte_size_) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "shared memory size specified with the request for output '" +
                std::string(tensor_name) + "' (" +
                std::to_string(pr->second.byte_size_) +
                " bytes) should be at least " + std::to_string(byte_size) +
                " bytes to hold the results")
                .c_str());
      }

      *buffer = const_cast<void*>(pr->second.base_);
      *actual_memory_type = pr->second.memory_type_;
      *actual_memory_type_id = pr->second.memory_type_id_;

      LOG_VERBOSE(1) << "GRPC: using shared-memory for '" << tensor_name
                     << "', size: " << byte_size << ", addr: " << *buffer;
      return nullptr;  // Success
    }

    // Not using shared memory so allocate a buffer. The buffer we
    // create is directly in the response protobuf so we can't
    // allocate any type other than CPU.
    //
    // FIXME we could use pinned CPU memory here.
    if (*actual_memory_type != TRITONSERVER_MEMORY_CPU) {
      LOG_VERBOSE(1) << "GRPC: unable to provide '" << tensor_name << "' in "
                     << TRITONSERVER_MemoryTypeString(*actual_memory_type)
                     << ", will use "
                     << TRITONSERVER_MemoryTypeString(TRITONSERVER_MEMORY_CPU);
      *actual_memory_type = TRITONSERVER_MEMORY_CPU;
      *actual_memory_type_id = 0;
    }

    raw_output->resize(byte_size);
    *buffer = static_cast<void*>(&((*raw_output)[0]));

    LOG_VERBOSE(1) << "GRPC: using buffer for '" << tensor_name
                   << "', size: " << byte_size << ", addr: " << *buffer;
  }

  return nullptr;  // Success
}

void
ModelInferHandler::Execute(InferHandler::State* state)
{
  TRITONSERVER_Error* err = nullptr;
  const inference::ModelInferRequest& request = state->request_;
  auto response_queue = state->response_queue_;
  int64_t requested_model_version;
  if (err == nullptr) {
    err = GetModelVersionFromString(
        request.model_version(), &requested_model_version);
  }

  if (err == nullptr) {
    uint32_t txn_flags;
    err = TRITONSERVER_ServerModelTransactionProperties(
        tritonserver_.get(), request.model_name().c_str(),
        requested_model_version, &txn_flags, nullptr /* voidp */);
    if ((err == nullptr) && (txn_flags & TRITONSERVER_TXN_DECOUPLED) != 0) {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "ModelInfer RPC doesn't support models with decoupled "
          "transaction policy");
    }
  }

  // Create the inference request which contains all the
  // input information needed for an inference.
  TRITONSERVER_InferenceRequest* irequest = nullptr;
  if (err == nullptr) {
    err = TRITONSERVER_InferenceRequestNew(
        &irequest, tritonserver_.get(), request.model_name().c_str(),
        requested_model_version);
  }

  if (err == nullptr) {
    state->inference_request_ = {
        irequest, [](TRITONSERVER_InferenceRequest* request) {
          LOG_TRITONSERVER_ERROR(
              TRITONSERVER_InferenceRequestDelete(request),
              "deleting gRPC inference request");
        }};
    err = SetInferenceRequestMetadata(irequest, request, state->parameters_);
  }

  if (err == nullptr) {
    err = ForwardHeadersAsParameters(irequest, state);
  }

  // Will be used to hold the serialized data in case explicit string
  // tensors are present in the request.
  std::list<std::string> serialized_data;

  // Maintain shared pointers(read-only reference) to the shared memory block's
  // information for the shared memory regions used by the request. These
  // pointers will automatically increase the usage count, preventing
  // unregistration of the shared memory. This vector must be cleared in the
  // `InferResponseComplete` callback (after inference) to decrease the count
  // and permit unregistration. The vector will be included in
  // `response_release_payload` for the callback.
  std::vector<std::shared_ptr<const SharedMemoryManager::SharedMemoryInfo>>
      shm_regions_info;

  if (err == nullptr) {
    err = InferGRPCToInput(
        tritonserver_, shm_manager_, request, &serialized_data, irequest,
        &shm_regions_info);
  }
  if (err == nullptr) {
    err = InferAllocatorPayload<inference::ModelInferResponse>(
        tritonserver_, shm_manager_, request, std::move(serialized_data),
        response_queue, &state->alloc_payload_, &shm_regions_info);
  }

  auto request_release_payload =
      std::make_unique<RequestReleasePayload>(state->inference_request_);
  auto response_release_payload = std::make_unique<ResponseReleasePayload>(
      state, std::move(shm_regions_info), shm_manager_);

  if (err == nullptr) {
    err = TRITONSERVER_InferenceRequestSetReleaseCallback(
        irequest, InferRequestComplete,
        request_release_payload.get() /* request_release_userp */);
  }
  if (err == nullptr) {
    err = TRITONSERVER_InferenceRequestSetResponseCallback(
        irequest, allocator_,
        &state->alloc_payload_ /* response_allocator_userp */,
        InferResponseComplete,
        response_release_payload.get() /* response_userp */);
  }
  // Get request ID for logging in case of error.
  const char* request_id = "";
  if (irequest != nullptr) {
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceRequestId(irequest, &request_id),
        "unable to retrieve request ID string");
  }

  if (!strncmp(request_id, "", 1)) {
    request_id = "<id_unknown>";
  }
  if (err == nullptr) {
    TRITONSERVER_InferenceTrace* triton_trace = nullptr;
#ifdef TRITON_ENABLE_TRACING
    if (trace_manager_) {
      GrpcServerCarrier carrier(state->context_->ctx_.get());
      auto start_options =
          trace_manager_->GetTraceStartOptions(carrier, request.model_name());
      state->trace_ = std::move(trace_manager_->SampleTrace(start_options));
      if (state->trace_ != nullptr) {
        triton_trace = state->trace_->trace_;
      }
    }
#endif  // TRITON_ENABLE_TRACING

    state->step_ = ISSUED;
    err = TRITONSERVER_ServerInferAsync(
        tritonserver_.get(), irequest, triton_trace);
  }

  // If not error then state->step_ == ISSUED and inference request
  // has initiated... completion callback will transition to
  // COMPLETE or CANCELLED. Recording the state and the irequest
  // to handle gRPC stream cancellation.
  if (err == nullptr) {
    state->context_->InsertInflightState(state);
    // The payload will be cleaned in release callback.
    request_release_payload.release();
    response_release_payload.release();
  } else {
    // If error go immediately to COMPLETE.
    LOG_VERBOSE(1) << "[request id: " << request_id << "] "
                   << "Infer failed: " << TRITONSERVER_ErrorMessage(err);

    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);

    inference::ModelInferResponse error_response;

#ifdef TRITON_ENABLE_TRACING
    if (trace_manager_) {
      state->trace_timestamps_.emplace_back(
          std::make_pair("GRPC_SEND_START", TraceManager::CaptureTimestamp()));
    }
#endif  // TRITON_ENABLE_TRACING

    state->step_ = Steps::COMPLETE;
    state->context_->responder_->Finish(error_response, status, state);
  }
}

void
ModelInferHandler::InferResponseComplete(
    TRITONSERVER_InferenceResponse* iresponse, const uint32_t flags,
    void* userp)
{
  ResponseReleasePayload* response_release_payload(
      static_cast<ResponseReleasePayload*>(userp));
  auto state = response_release_payload->state_;

  // There are multiple handlers registered in the gRPC service
  // Hence, we would need to properly synchronize this thread
  // and the handler thread handling async cancellation
  // notification.
  std::lock_guard<std::recursive_mutex> lock(state->step_mtx_);

  if (state->delay_response_complete_exec_ms_ != 0) {
    // Will delay the Process execution of state at step ISSUED by the
    // specified time. This can be used to test the flow when cancellation
    // request issued for the request before InferResponseComplete.
    LOG_INFO << "Delaying InferResponseComplete execution by "
             << state->delay_response_complete_exec_ms_ << " ms...";
    std::this_thread::sleep_for(
        std::chrono::milliseconds(state->delay_response_complete_exec_ms_));
  }

  // Increment the callback index if received valid 'iresponse'
  if (iresponse != nullptr) {
    state->cb_count_++;
  }

  LOG_VERBOSE(1) << "ModelInferHandler::InferResponseComplete, "
                 << state->unique_id_ << " step " << state->step_;

  // Allow sending 1 response and final flag separately, only mark
  // non-inflight when seeing final flag
  if (flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) {
    state->context_->EraseInflightState(state);
  }

  // If gRPC Stream is cancelled then no need of forming and returning
  // a response.
  if (state->IsGrpcContextCancelled()) {
    // Clean-up the received response object.
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceResponseDelete(iresponse),
        "deleting GRPC inference response");

    state->context_->EraseInflightState(state);
    state->step_ = Steps::CANCELLED;

    LOG_VERBOSE(1) << "ModelInferHandler::InferResponseComplete, "
                   << state->unique_id_
                   << ", skipping response generation as grpc transaction was "
                      "cancelled... ";

    if (state->delay_enqueue_ms_ != 0) {
      // Will delay PutTaskBackToQueue by the specified time.
      // This can be used to test the flow when cancellation request
      // issued for the request during InferResponseComplete
      // callback right before Process in the notification thread.
      LOG_INFO << "Delaying PutTaskBackToQueue by " << state->delay_enqueue_ms_
               << " ms...";
      std::this_thread::sleep_for(
          std::chrono::milliseconds(state->delay_enqueue_ms_));
    }

    // Send state back to the queue so that state can be released
    // in the next cycle.
    state->context_->PutTaskBackToQueue(state);

    delete response_release_payload;
    return;
  }

  TRITONSERVER_Error* err = nullptr;
  // This callback is expected to be called exactly once for each request.
  // Will use the single response object in the response list to hold the
  // information.
  inference::ModelInferResponse* response =
      state->response_queue_->GetResponseAt(0);
  bool response_created = false;
  if (response == nullptr) {
    LOG_ERROR << "expected allocator to have created a response object";
    err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "No response object found in the callback");
    response_created = true;
    response = new inference::ModelInferResponse();
  }

  if (state->cb_count_ != 1) {
    err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, std::string(
                                         "expected a single response, got " +
                                         std::to_string(state->cb_count_))
                                         .c_str());
  } else if (iresponse != nullptr) {
    err = InferResponseCompleteCommon<inference::ModelInferResponse>(
        state->tritonserver_, iresponse, *response, state->alloc_payload_);
#ifdef TRITON_ENABLE_TRACING
    state->trace_timestamps_.emplace_back(std::make_pair(
        "INFER_RESPONSE_COMPLETE", TraceManager::CaptureTimestamp()));
#endif  // TRITON_ENABLE_TRACING
  }

  if (err != nullptr) {
    response->Clear();
  }

  GrpcStatusUtil::Create(&state->status_, err);
  TRITONSERVER_ErrorDelete(err);

  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceResponseDelete(iresponse),
      "deleting GRPC inference response");

  // Defer sending the response until FINAL flag is seen or
  // there is error
  if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) == 0) {
    return;
  }


#ifdef TRITON_ENABLE_TRACING
  state->trace_timestamps_.emplace_back(
      std::make_pair("GRPC_SEND_START", TraceManager::CaptureTimestamp()));
#endif  // TRITON_ENABLE_TRACING

  if (state->delay_response_completion_ms_ != 0) {
    // Will delay the Process execution of state at step COMPLETE by the
    // specified time. This can be used to test the flow when cancellation
    // request issued for the request, which is at InferResponseComplete.
    LOG_INFO << "Delaying InferResponseComplete by "
             << state->delay_response_completion_ms_ << " ms...";
    std::this_thread::sleep_for(
        std::chrono::milliseconds(state->delay_response_completion_ms_));
  }

  state->step_ = Steps::COMPLETE;
  state->context_->responder_->Finish(*response, state->status_, state);
  if (response_created) {
    delete response;
  }

  delete response_release_payload;
}

}}}  // namespace triton::server::grpc
