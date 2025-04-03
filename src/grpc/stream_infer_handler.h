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
#pragma once

#include "infer_handler.h"

namespace triton { namespace server { namespace grpc {

// Make sure to keep InferResponseAlloc and OutputBufferQuery logic in sync
TRITONSERVER_Error* StreamInferResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id);

//
// Additional Stream Infer utilities
//
TRITONSERVER_Error* StreamInferResponseStart(
    TRITONSERVER_ResponseAllocator* allocator, void* userp);

// Make sure to keep InferResponseAlloc and OutputBufferQuery logic in sync
TRITONSERVER_Error* StreamOutputBufferQuery(
    TRITONSERVER_ResponseAllocator* allocator, void* userp,
    const char* tensor_name, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id);

// Make sure to keep InferResponseAlloc, OutputBufferQuery, and
// OutputBufferAttributes logic in sync
TRITONSERVER_Error* StreamOutputBufferAttributes(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    TRITONSERVER_BufferAttributes* buffer_attributes, void* userp,
    void* buffer_userp);

class ModelStreamInferHandler
    : public InferHandler<
          inference::GRPCInferenceService::AsyncService,
          ::grpc::ServerAsyncReaderWriter<
              inference::ModelStreamInferResponse,
              inference::ModelInferRequest>,
          inference::ModelInferRequest, inference::ModelStreamInferResponse> {
 public:
  ModelStreamInferHandler(
      const std::string& name,
      const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
      TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      inference::GRPCInferenceService::AsyncService* service,
      ::grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count,
      size_t max_response_queue_size, grpc_compression_level compression_level,
      std::pair<std::string, std::string> restricted_kv,
      const std::string& header_forward_pattern)
      : InferHandler(
            name, tritonserver, service, cq, max_state_bucket_count,
            max_response_queue_size, restricted_kv, header_forward_pattern),
        trace_manager_(trace_manager), shm_manager_(shm_manager),
        compression_level_(compression_level)
  {
    // Create the allocator that will be used to allocate buffers for
    // the result tensors.
    FAIL_IF_ERR(
        TRITONSERVER_ResponseAllocatorNew(
            &allocator_, StreamInferResponseAlloc, InferResponseFree,
            StreamInferResponseStart),
        "creating response allocator");
    FAIL_IF_ERR(
        TRITONSERVER_ResponseAllocatorSetQueryFunction(
            allocator_, StreamOutputBufferQuery),
        "setting allocator's query function");
    FAIL_IF_ERR(
        TRITONSERVER_ResponseAllocatorSetBufferAttributesFunction(
            allocator_, StreamOutputBufferAttributes),
        "setting allocator's output buffer attribute query function");
  }

  ~ModelStreamInferHandler()
  {
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_ResponseAllocatorDelete(allocator_),
        "deleting response allocator");
  }

 protected:
  void StartNewRequest() override;
  bool Process(
      State* state, bool rpc_ok, bool is_notification = false) override;

 private:
  static void StreamInferResponseComplete(
      TRITONSERVER_InferenceResponse* response, const uint32_t flags,
      void* userp);
  static void StateWriteResponse(InferHandler::State* state);
  bool Finish(State* state);

  TraceManager* trace_manager_;
  std::shared_ptr<SharedMemoryManager> shm_manager_;
  TRITONSERVER_ResponseAllocator* allocator_;

  grpc_compression_level compression_level_;
};

}}}  // namespace triton::server::grpc
