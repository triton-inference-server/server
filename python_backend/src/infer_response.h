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

#pragma once

#include <future>

#include "gpu_buffers.h"
#include "pb_error.h"
#include "pb_tensor.h"
#include "pb_utils.h"
#include "scoped_defer.h"

namespace triton { namespace backend { namespace python {

struct ResponseShm {
  uint32_t outputs_size;
  bi::managed_external_buffer::handle_t parameters;
  bi::managed_external_buffer::handle_t error;
  bool has_error;
  // Indicates whether this error has a message or not.
  bool is_error_set;
  void* id;
  bool is_last_response;
};

#define SET_ERROR_AND_RETURN(E, X)           \
  do {                                       \
    TRITONSERVER_Error* raasnie_err__ = (X); \
    if (raasnie_err__ != nullptr) {          \
      *E = raasnie_err__;                    \
      return;                                \
    }                                        \
  } while (false)

#define SET_ERROR_AND_RETURN_IF_EXCEPTION(E, X)                \
  do {                                                         \
    try {                                                      \
      (X);                                                     \
    }                                                          \
    catch (const PythonBackendException& pb_exception) {       \
      TRITONSERVER_Error* rarie_err__ = TRITONSERVER_ErrorNew( \
          TRITONSERVER_ERROR_INTERNAL, pb_exception.what());   \
      *E = rarie_err__;                                        \
      return;                                                  \
    }                                                          \
  } while (false)

class InferResponse {
 public:
  InferResponse(
      const std::vector<std::shared_ptr<PbTensor>>& output_tensors,
      std::shared_ptr<PbError> error = nullptr, std::string parameters = "",
      const bool is_last_response = true, void* id = nullptr);
  std::vector<std::shared_ptr<PbTensor>>& OutputTensors();
  const std::string& Parameters() const;  // JSON serializable unless empty
  void SaveToSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool, bool copy_gpu = true);
  static std::unique_ptr<InferResponse> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t response_handle,
      bool open_cuda_handle);
  bool HasError();
  std::shared_ptr<PbError>& Error();
  bi::managed_external_buffer::handle_t ShmHandle();
  void PruneOutputTensors(const std::set<std::string>& requested_output_names);
  std::unique_ptr<std::future<std::unique_ptr<InferResponse>>>
  GetNextResponse();
  void SetNextResponseHandle(
      bi::managed_external_buffer::handle_t next_response_handle);
  bi::managed_external_buffer::handle_t NextResponseHandle();
  void* Id();
  bool IsLastResponse();

#ifndef TRITON_PB_STUB
  /// Send an inference response. If the response has a GPU tensor, sending the
  /// response needs to be done in two step. The boolean
  /// 'requires_deferred_callback' indicates whether DeferredSendCallback method
  /// should be called or not.
  void Send(
      TRITONBACKEND_Response* response, void* cuda_stream,
      bool& requires_deferred_callback, const uint32_t flags,
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      GPUBuffersHelper& gpu_buffer_helper,
      std::vector<std::pair<std::unique_ptr<PbMemory>, void*>>& output_buffers,
      const std::set<std::string>& requested_output_names = {});

  void DeferredSendCallback();
#endif

  // Disallow copying the inference response object.
  DISALLOW_COPY_AND_ASSIGN(InferResponse);

 private:
  InferResponse(
      AllocatedSharedMemory<char>& response_shm,
      std::vector<std::shared_ptr<PbTensor>>& output_tensors,
      std::shared_ptr<PbError>& pb_error, const bool is_last_response, void* id,
      std::shared_ptr<PbString>& parameters_shm, std::string& parameters);
  std::vector<std::shared_ptr<PbTensor>> output_tensors_;

  std::shared_ptr<PbError> error_;
  bi::managed_external_buffer::handle_t shm_handle_;
  AllocatedSharedMemory<char> response_shm_;
  std::vector<std::pair<std::unique_ptr<PbMemory>, void*>> gpu_output_buffers_;
  std::unique_ptr<ScopedDefer> deferred_send_callback_;
  bool is_last_response_;
  // Representing the request id that the response was created from.
  void* id_;

  std::shared_ptr<PbString> parameters_shm_;
  std::string parameters_;
};

}}}  // namespace triton::backend::python
