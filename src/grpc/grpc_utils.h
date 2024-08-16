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
#pragma once

#include <list>
#include <memory>
#include <unordered_map>

#include "../classification.h"
#include "../common.h"
#include "../shared_memory_manager.h"
#include "grpc_service.grpc.pb.h"
#include "triton/common/logging.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace server { namespace grpc {

// The step of processing that the state is in. Every state must
// recognize START, COMPLETE and FINISH and the others are optional.
typedef enum {
  // This marks the starting stage of the RPC
  START,
  // This marks that RPC is complete.
  COMPLETE,
  // This marks the stage where all the notifications from the gRPC
  // completion queue is received and state can be safely released.
  FINISH,
  // This stage means that RPC has been issued to Triton for inference
  // and is waiting for the server callbacks or cancellation to be
  // invoked.
  ISSUED,
  // This stage means the request has been read from the network and
  // can be sent to Triton for execution.
  READ,
  // This stage means that the response is ready to be written back to
  // the network.
  WRITEREADY,
  // This stage means that response has been written completely to the
  // network.
  WRITTEN,
  // This marks the special stage for the state object to differentiate
  // the tag delivered from AsyncNotifyWhenDone() method.
  WAITING_NOTIFICATION,
  // This stage means that the cancellation for the RPC has been issued
  // to the server.
  CANCELLATION_ISSUED,
  // This stage marks that the state has been successfully cancelled.
  CANCELLED,
  // This is intermediary stage where the state has been been partially
  // completed by grpc responder Finish call or AsyncNotifyWhenDone()
  // notification. The other next call will move the stage to fully
  // complete.
  PARTIAL_COMPLETION
} Steps;

typedef enum {
  // No error from CORE seen yet
  NONE,
  // Error from CORE encountered, waiting to be picked up by completion queue to
  // initiate cancellation
  ERROR_ENCOUNTERED,
  // Error from CORE encountered, stream closed
  // This state is added to avoid double cancellation
  ERROR_HANDLING_COMPLETE
} TritonGRPCErrorSteps;

class gRPCErrorTracker {
 public:
  // True if set by user via header
  // Can be accessed without a lock, as set only once in startstream
  std::atomic<bool> triton_grpc_error_;

  // Indicates the state of triton_grpc_error, only relevant if special
  // triton_grpc_error feature set to true by client
  TritonGRPCErrorSteps grpc_stream_error_state_;

  // Constructor
  gRPCErrorTracker()
      : triton_grpc_error_(false),
        grpc_stream_error_state_(TritonGRPCErrorSteps::NONE)
  {
  }
  // Changes the state of grpc_stream_error_state_ to ERROR_HANDLING_COMPLETE,
  // indicating we have closed the stream and initiated the cancel flow
  void MarkGRPCErrorHandlingComplete();

  // Returns true ONLY when GRPC_ERROR from CORE is waiting to be processed.
  bool CheckAndUpdateGRPCError();

  // Marks error after it has been responded to
  void MarkGRPCErrorEncountered();

  // Checks if error already responded to in triton_grpc_error mode
  bool GRPCErrorEncountered();
};
// Debugging helper
std::ostream& operator<<(std::ostream& out, const Steps& step);

//
// GrpcStatusUtil
//
class GrpcStatusUtil {
 public:
  static void Create(::grpc::Status* status, TRITONSERVER_Error* err);
  static ::grpc::StatusCode CodeToStatus(TRITONSERVER_Error_Code code);
};

template <typename TensorType>
TRITONSERVER_Error*
ParseSharedMemoryParams(
    const TensorType& tensor, bool* has_shared_memory, std::string* region_name,
    int64_t* offset, size_t* byte_size)
{
  *has_shared_memory = false;
  *offset = 0 /* default value */;
  const auto& region_it = tensor.parameters().find("shared_memory_region");
  if (region_it != tensor.parameters().end()) {
    *has_shared_memory = true;
    const auto& infer_param = region_it->second;
    if (infer_param.parameter_choice_case() !=
        inference::InferParameter::ParameterChoiceCase::kStringParam) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "invalid value type for 'shared_memory_region' parameter for "
              "tensor '" +
              tensor.name() + "', expected string_param.")
              .c_str());
    }
    *region_name = infer_param.string_param();
  }

  const auto& offset_it = tensor.parameters().find("shared_memory_offset");
  if (offset_it != tensor.parameters().end()) {
    if (!*has_shared_memory) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "'shared_memory_offset' can not be specified without "
              "'shared_memory_region' parameter for tensor '" +
              tensor.name() + "'")
              .c_str());
    }
    const auto& infer_param = offset_it->second;
    if (infer_param.parameter_choice_case() !=
        inference::InferParameter::ParameterChoiceCase::kInt64Param) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "invalid value type for 'shared_memory_offset' parameter for "
              "tensor '" +
              tensor.name() + "', expected int64_param.")
              .c_str());
    }
    *offset = infer_param.int64_param();
  }

  const auto& bs_it = tensor.parameters().find("shared_memory_byte_size");
  if (bs_it != tensor.parameters().end()) {
    if (!*has_shared_memory) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "'shared_memory_byte_size' can not be specified without "
              "'shared_memory_region' parameter for tensor '" +
              tensor.name() + "'")
              .c_str());
    }
    const auto& infer_param = bs_it->second;
    if (infer_param.parameter_choice_case() !=
        inference::InferParameter::ParameterChoiceCase::kInt64Param) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "invalid value type for 'shared_memory_byte_size' parameter "
              "for "
              "tensor '" +
              tensor.name() + "', expected int64_param.")
              .c_str());
    }
    *byte_size = infer_param.int64_param();
  } else {
    if (*has_shared_memory) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "'shared_memory_byte_size' must be specified along with "
              "'shared_memory_region' parameter for tensor '" +
              tensor.name() + "'")
              .c_str());
    }
  }

  return nullptr;
}

TRITONSERVER_Error* ParseClassificationParams(
    const inference::ModelInferRequest::InferRequestedOutputTensor& output,
    bool* has_classification, uint32_t* classification_count);


void ReadFile(const std::string& filename, std::string& data);
}}}  // namespace triton::server::grpc
