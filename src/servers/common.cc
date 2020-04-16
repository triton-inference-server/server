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

#include "src/servers/common.h"

namespace nvidia { namespace inferenceserver {

void
RequestStatusUtil::Create(
    RequestStatus* status, TRTSERVER_Error* err, uint64_t request_id,
    const std::string& server_id)
{
  status->set_code(
      (err == nullptr)
          ? RequestStatusCode::SUCCESS
          : RequestStatusUtil::CodeToStatus(TRTSERVER_ErrorCode(err)));

  if (err != nullptr) {
    status->set_msg(TRTSERVER_ErrorMessage(err));
  }

  status->set_server_id(server_id);
  status->set_request_id(request_id);
}

void
RequestStatusUtil::Create(
    RequestStatus* status, uint64_t request_id, const std::string& server_id,
    RequestStatusCode code, const std::string& msg)
{
  status->Clear();
  status->set_code(code);
  status->set_msg(msg);
  status->set_server_id(server_id);
  status->set_request_id(request_id);
}

void
RequestStatusUtil::Create(
    RequestStatus* status, uint64_t request_id, const std::string& server_id,
    RequestStatusCode code)
{
  status->Clear();
  status->set_code(code);
  status->set_server_id(server_id);
  status->set_request_id(request_id);
}

RequestStatusCode
RequestStatusUtil::CodeToStatus(TRTSERVER_Error_Code code)
{
  switch (code) {
    case TRTSERVER_ERROR_UNKNOWN:
      return RequestStatusCode::UNKNOWN;
    case TRTSERVER_ERROR_INTERNAL:
      return RequestStatusCode::INTERNAL;
    case TRTSERVER_ERROR_NOT_FOUND:
      return RequestStatusCode::NOT_FOUND;
    case TRTSERVER_ERROR_INVALID_ARG:
      return RequestStatusCode::INVALID_ARG;
    case TRTSERVER_ERROR_UNAVAILABLE:
      return RequestStatusCode::UNAVAILABLE;
    case TRTSERVER_ERROR_UNSUPPORTED:
      return RequestStatusCode::UNSUPPORTED;
    case TRTSERVER_ERROR_ALREADY_EXISTS:
      return RequestStatusCode::ALREADY_EXISTS;

    default:
      break;
  }

  return RequestStatusCode::UNKNOWN;
}

uint64_t
RequestStatusUtil::NextUniqueRequestId()
{
  static std::atomic<uint64_t> id(0);
  return ++id;
}

TRTSERVER_Error*
SetTRTSERVER_InferenceRequestOptions(
    TRTSERVER_InferenceRequestOptions* request_options,
    const InferRequestHeader& request_header)
{
  RETURN_IF_ERR(TRTSERVER_InferenceRequestOptionsSetId(
      request_options, request_header.id()));
  RETURN_IF_ERR(TRTSERVER_InferenceRequestOptionsSetFlags(
      request_options, request_header.flags()));
  RETURN_IF_ERR(TRTSERVER_InferenceRequestOptionsSetCorrelationId(
      request_options, request_header.correlation_id()));
  RETURN_IF_ERR(TRTSERVER_InferenceRequestOptionsSetBatchSize(
      request_options, request_header.batch_size()));
  RETURN_IF_ERR(TRTSERVER_InferenceRequestOptionsSetPriority(
      request_options, request_header.priority()));
  RETURN_IF_ERR(TRTSERVER_InferenceRequestOptionsSetTimeoutMicroseconds(
      request_options, request_header.timeout_microseconds()));

  for (const auto& input : request_header.input()) {
    RETURN_IF_ERR(TRTSERVER_InferenceRequestOptionsAddInput(
        request_options, input.name().c_str(), input.dims().data(),
        input.dims_size(), input.batch_byte_size()));
  }

  for (const auto& output : request_header.output()) {
    if (output.has_cls()) {
      RETURN_IF_ERR(TRTSERVER_InferenceRequestOptionsAddClassificationOutput(
          request_options, output.name().c_str(), output.cls().count()));
    } else {
      RETURN_IF_ERR(TRTSERVER_InferenceRequestOptionsAddOutput(
          request_options, output.name().c_str()));
    }
  }
  return nullptr;  // Success
}

std::string
MemoryTypeString(TRTSERVER_Memory_Type memory_type)
{
  switch (memory_type) {
    case TRTSERVER_MEMORY_CPU:
      return "CPU memory";
    case TRTSERVER_MEMORY_CPU_PINNED:
      return "Pinned CPU memory";
    case TRTSERVER_MEMORY_GPU:
      return "GPU memory";
    default:
      return "unknown memory type";
  }
}

size_t
GetDataTypeByteSize(const std::string& protocol_dtype)
{
  if ((protocol_dtype.compare("BOOL") == 0) ||
      (protocol_dtype.compare("INT8") == 0) ||
      (protocol_dtype.compare("UINT8") == 0)) {
    return 1;
  } else if (
      (protocol_dtype.compare("INT16") == 0) ||
      (protocol_dtype.compare("UINT16") == 0) ||
      (protocol_dtype.compare("FP16") == 0)) {
    return 2;
  } else if (
      (protocol_dtype.compare("INT32") == 0) ||
      (protocol_dtype.compare("UINT32") == 0) ||
      (protocol_dtype.compare("FP32") == 0)) {
    return 4;
  } else if (
      (protocol_dtype.compare("INT64") == 0) ||
      (protocol_dtype.compare("UINT64") == 0) ||
      (protocol_dtype.compare("FP64") == 0)) {
    return 8;
  } else {
    // If the data type is unknown or bytes (variable) then return 0
    return 0;
  }
}

TRITONSERVER_Error*
GetModelVersionFromString(
    const std::string& version_string, int64_t* version_int)
{
  if (version_string.empty()) {
    *version_int = -1;
  } else {
    try {
      *version_int = std::stol(version_string);
    }
    catch (std::exception& e) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "failed to get model version from specified version string '" +
              version_string + "' (details: " + e.what() +
              "), version should be an integral value > 0")
              .c_str());
    }
    if (*version_int < 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "invalid model version specified '" +
              std::to_string(*version_int) +
              "' , version should be an integral value > 0")
              .c_str());
    }
  }

  return nullptr;  // Success
}

//
// TRITON
//

namespace {

TRTSERVER_Error_Code
TritonErrorCodeToTrt(TRITONSERVER_Error_Code code)
{
  switch (code) {
    case TRITONSERVER_ERROR_INTERNAL:
      return TRTSERVER_ERROR_INTERNAL;
    case TRITONSERVER_ERROR_NOT_FOUND:
      return TRTSERVER_ERROR_NOT_FOUND;
    case TRITONSERVER_ERROR_INVALID_ARG:
      return TRTSERVER_ERROR_INVALID_ARG;
    case TRITONSERVER_ERROR_UNAVAILABLE:
      return TRTSERVER_ERROR_UNAVAILABLE;
    case TRITONSERVER_ERROR_UNSUPPORTED:
      return TRTSERVER_ERROR_UNSUPPORTED;
    case TRITONSERVER_ERROR_ALREADY_EXISTS:
      return TRTSERVER_ERROR_ALREADY_EXISTS;
    case TRITONSERVER_ERROR_UNKNOWN:
    default:
      return TRTSERVER_ERROR_UNKNOWN;
  }
}

}  // namespace


TRTSERVER_Error*
TritonErrorToTrt(TRITONSERVER_Error* err)
{
  if (err != nullptr) {
    auto triton_err = TRTSERVER_ErrorNew(
        TritonErrorCodeToTrt(TRITONSERVER_ErrorCode(err)),
        TRITONSERVER_ErrorMessage(err));
    TRITONSERVER_ErrorDelete(err);
    return triton_err;
  }
  return nullptr;
}

TRTSERVER_Memory_Type
TritonMemTypeToTrt(TRITONSERVER_MemoryType mem_type)
{
  switch (mem_type) {
    case TRITONSERVER_MEMORY_CPU:
      return TRTSERVER_MEMORY_CPU;
      break;
    case TRITONSERVER_MEMORY_CPU_PINNED:
      return TRTSERVER_MEMORY_CPU_PINNED;
    default:
      return TRTSERVER_MEMORY_GPU;
      break;
  }
}

TRITONSERVER_MemoryType
TrtMemTypeToTriton(TRTSERVER_Memory_Type mem_type)
{
  switch (mem_type) {
    case TRTSERVER_MEMORY_CPU:
      return TRITONSERVER_MEMORY_CPU;
      break;
    case TRTSERVER_MEMORY_CPU_PINNED:
      return TRITONSERVER_MEMORY_CPU_PINNED;
    default:
      return TRITONSERVER_MEMORY_GPU;
      break;
  }
}

// FIXMEV2 remove this and use TRITONSERVER_MemoryTypeString
std::string
MemoryTypeString(TRITONSERVER_MemoryType memory_type)
{
  switch (memory_type) {
    case TRITONSERVER_MEMORY_CPU:
      return "CPU memory";
    case TRITONSERVER_MEMORY_CPU_PINNED:
      return "Pinned CPU memory";
    case TRITONSERVER_MEMORY_GPU:
      return "GPU memory";
    default:
      return "unknown memory type";
  }
}

}}  // namespace nvidia::inferenceserver
