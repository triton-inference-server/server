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

#ifdef TRTIS_ENABLE_GRPC_V2

TRTSERVER_Error*
SetInferenceRequestOptions(
    TRTSERVER_InferenceRequestOptions* request_options,
    const ModelInferRequest& request)
{
  RETURN_IF_ERR(TRTSERVER_InferenceRequestOptionsSetIdStr(
      request_options, request.id().c_str()));
  // FIXMEV2 parameters
  // RETURN_IF_ERR(TRTSERVER_InferenceRequestOptionsSetFlags(
  //    request_options, request_header.flags()));
  RETURN_IF_ERR(TRTSERVER_InferenceRequestOptionsSetCorrelationId(
      request_options, request.sequence_id()));

  // FIXMEV2 raw contents size?? Do we need it?
  for (const auto& input : request.inputs()) {
    RETURN_IF_ERR(TRTSERVER_InferenceRequestOptionsAddInput(
        request_options, input.name().c_str(), input.shape().data(),
        input.shape_size(), input.contents().raw_contents().size()));
  }

  for (const auto& output : request.outputs()) {
    // FIXMEV2 parameters
    if (output.parameters().find("classification") !=
        output.parameters().end()) {
      const auto& infer_param = output.parameters().at("classification");
      if (infer_param.parameter_choice_case() !=
          InferParameter::ParameterChoiceCase::kInt64Param) {
        return TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INVALID_ARG,
            "invalid value type for 'classification' parameter, expected "
            "int64_param.");
      }
      RETURN_IF_ERR(TRTSERVER_InferenceRequestOptionsAddClassificationOutput(
          request_options, output.name().c_str(), infer_param.int64_param()));
    } else {
      RETURN_IF_ERR(TRTSERVER_InferenceRequestOptionsAddOutput(
          request_options, output.name().c_str()));
    }
  }
  return nullptr;  // Success
}

#endif  // TRTIS_ENABLE_GRPC_V2

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

const char*
GetDataTypeProtocolString(const DataType dtype)
{
  switch (dtype) {
    case TYPE_BOOL:
      return "BOOL";
    case TYPE_UINT8:
      return "UINT8";
    case TYPE_UINT16:
      return "UINT16";
    case TYPE_UINT32:
      return "UINT32";
    case TYPE_UINT64:
      return "UINT64";
    case TYPE_INT8:
      return "INT8";
    case TYPE_INT16:
      return "INT16";
    case TYPE_INT32:
      return "INT32";
    case TYPE_INT64:
      return "INT64";
    case TYPE_FP16:
      return "FP16";
    case TYPE_FP32:
      return "FP32";
    case TYPE_FP64:
      return "FP64";
    case TYPE_STRING:
      return "BYTES";
    default:
      break;
  }

  return "";
}

}}  // namespace nvidia::inferenceserver
