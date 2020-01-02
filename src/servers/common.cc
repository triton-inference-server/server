// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
    InferRequestHeader request_header_protobuf)
{
  RETURN_IF_ERR(TRTSERVER_InferenceRequestOptionsSetId(
      request_options, request_header_protobuf.id()));
  RETURN_IF_ERR(TRTSERVER_InferenceRequestOptionsSetFlags(
      request_options, request_header_protobuf.flags()));
  RETURN_IF_ERR(TRTSERVER_InferenceRequestOptionsSetCorrelationId(
      request_options, request_header_protobuf.correlation_id()));
  RETURN_IF_ERR(TRTSERVER_InferenceRequestOptionsSetBatchSize(
      request_options, request_header_protobuf.batch_size()));

  for (const auto& input : request_header_protobuf.input()) {
    RETURN_IF_ERR(TRTSERVER_InferenceRequestOptionsAddInput(
        request_options, input.name().c_str(), input.dims().data(),
        input.dims_size(), input.batch_byte_size()));
  }

  for (const auto& output : request_header_protobuf.output()) {
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

}}  // namespace nvidia::inferenceserver
