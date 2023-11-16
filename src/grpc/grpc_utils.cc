// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "grpc_utils.h"

namespace triton { namespace server { namespace grpc {

std::ostream&
operator<<(std::ostream& out, const Steps& step)
{
  switch (step) {
    case START:
      out << "START";
      break;
    case COMPLETE:
      out << "COMPLETE";
      break;
    case FINISH:
      out << "FINISH";
      break;
    case ISSUED:
      out << "ISSUED";
      break;
    case READ:
      out << "READ";
      break;
    case WRITEREADY:
      out << "WRITEREADY";
      break;
    case WRITTEN:
      out << "WRITTEN";
      break;
    case WAITING_NOTIFICATION:
      out << "WAITING_NOTIFICATION";
      break;
    case CANCELLATION_ISSUED:
      out << "CANCELLATION_ISSUED";
      break;
    case CANCELLED:
      out << "CANCELLED";
      break;
    case PARTIAL_COMPLETION:
      out << "PARTIAL_COMPLETION";
      break;
  }

  return out;
}

void
GrpcStatusUtil::Create(::grpc::Status* status, TRITONSERVER_Error* err)
{
  if (err == nullptr) {
    *status = ::grpc::Status::OK;
  } else {
    *status = ::grpc::Status(
        GrpcStatusUtil::CodeToStatus(TRITONSERVER_ErrorCode(err)),
        TRITONSERVER_ErrorMessage(err));
  }
}

::grpc::StatusCode
GrpcStatusUtil::CodeToStatus(TRITONSERVER_Error_Code code)
{
  // GRPC status codes:
  // https://github.com/grpc/grpc/blob/master/include/grpc/impl/codegen/status.h
  switch (code) {
    case TRITONSERVER_ERROR_UNKNOWN:
      return ::grpc::StatusCode::UNKNOWN;
    case TRITONSERVER_ERROR_INTERNAL:
      return ::grpc::StatusCode::INTERNAL;
    case TRITONSERVER_ERROR_NOT_FOUND:
      return ::grpc::StatusCode::NOT_FOUND;
    case TRITONSERVER_ERROR_INVALID_ARG:
      return ::grpc::StatusCode::INVALID_ARGUMENT;
    case TRITONSERVER_ERROR_UNAVAILABLE:
      return ::grpc::StatusCode::UNAVAILABLE;
    case TRITONSERVER_ERROR_UNSUPPORTED:
      return ::grpc::StatusCode::UNIMPLEMENTED;
    case TRITONSERVER_ERROR_ALREADY_EXISTS:
      return ::grpc::StatusCode::ALREADY_EXISTS;
    case TRITONSERVER_ERROR_CANCELLED:
      return ::grpc::StatusCode::CANCELLED;
  }

  return ::grpc::StatusCode::UNKNOWN;
}

TRITONSERVER_Error*
ParseClassificationParams(
    const inference::ModelInferRequest::InferRequestedOutputTensor& output,
    bool* has_classification, uint32_t* classification_count)
{
  *has_classification = false;

  const auto& class_it = output.parameters().find("classification");
  if (class_it != output.parameters().end()) {
    *has_classification = true;

    const auto& param = class_it->second;
    if (param.parameter_choice_case() !=
        inference::InferParameter::ParameterChoiceCase::kInt64Param) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "invalid value type for 'classification' parameter, expected "
          "int64_param");
    }

    const int64_t cnt = param.int64_param();
    if (cnt <= 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "invalid value for 'classification' parameter, expected >= 0");
    }

    *classification_count = cnt;
  }

  return nullptr;  // success
}

void
ReadFile(const std::string& filename, std::string& data)
{
  data.clear();
  if (!filename.empty()) {
    std::ifstream file(filename.c_str(), std::ios::in);
    if (file.is_open()) {
      std::stringstream ss;
      ss << file.rdbuf();
      file.close();
      data = ss.str();
    }
  }
}

}}}  // namespace triton::server::grpc
