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

#include "src/backends/backend/examples/backend_input_collector.h"

#include "src/backends/backend/examples/backend_utils.h"

namespace nvidia { namespace inferenceserver { namespace backend {

//
// BackendInputCollector
//
BackendInputCollector::~BackendInputCollector()
{
#ifdef TRITON_ENABLE_GPU
  for (auto& pinned_memory : pinned_memories_) {
    cudaFreeHost(pinned_memory);
  }
#endif  // TRITON_ENABLE_GPU
}

void
BackendInputCollector::ProcessTensor(
    const char* input_name, char* buffer, const size_t buffer_byte_size,
    const TRITONSERVER_MemoryType memory_type, const int64_t memory_type_id)
{
  // A value of CPU_PINNED indicates that pinned memory buffer is not
  // needed for this tensor. Any other value indicates that a pinned
  // memory buffer is needed when the target memory type matches
  // 'use_pinned_memory_type'.
  TRITONSERVER_MemoryType use_pinned_memory_type =
      TRITONSERVER_MEMORY_CPU_PINNED;
  if (pinned_enabled_ && (memory_type != TRITONSERVER_MEMORY_CPU_PINNED)) {
    use_pinned_memory_type = (memory_type == TRITONSERVER_MEMORY_CPU)
                                 ? TRITONSERVER_MEMORY_GPU
                                 : TRITONSERVER_MEMORY_CPU;
  }

  size_t buffer_offset = 0;

  for (size_t idx = 0; idx < request_count_; idx++) {
    auto& request = requests_[idx];
    auto& response = (*responses_)[idx];

    // If there are pending copies from tensor buffer that is not
    // contiguous with 'response's part of that buffer, then need to
    // go ahead and perform the pending copies so that can start a new
    // contiguous region if necessary.
    if ((pending_pinned_byte_size_ > 0) &&
        (buffer_offset !=
         (pending_pinned_byte_size_ + pending_pinned_offset_))) {
      need_sync_ |= FlushPendingPinned(
          buffer, buffer_byte_size, memory_type, memory_type_id);
    }

    TRITONBACKEND_Input* input;
    RESPOND_AND_SET_NULL_IF_ERROR(
        &response, TRITONBACKEND_RequestInput(request, input_name, &input));
    uint64_t byte_size;
    RESPOND_AND_SET_NULL_IF_ERROR(
        &response,
        TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, nullptr, nullptr, &byte_size, nullptr));
    if (response != nullptr) {
      need_sync_ |= SetFixedSizeInputTensor(
          input, buffer_offset, buffer, buffer_byte_size, memory_type,
          memory_type_id, use_pinned_memory_type, &response);
    }

    buffer_offset += byte_size;
  }

  // Done with the tensor, flush any pending pinned copies.
  need_sync_ |=
      FlushPendingPinned(buffer, buffer_byte_size, memory_type, memory_type_id);
#ifdef TRITON_ENABLE_GPU
  if (need_sync_ && (event_ != nullptr)) {
    cudaEventRecord(event_, stream_);
  }
#endif  // TRITON_ENABLE_GPU
}

bool
BackendInputCollector::Finalize()
{
#ifdef TRITON_ENABLE_GPU
  if ((!deferred_pinned_.empty()) && need_sync_) {
    if (event_ != nullptr) {
      cudaEventSynchronize(event_);
    } else {
      cudaStreamSynchronize(stream_);
    }
    need_sync_ = false;
  }
#endif  // TRITON_ENABLE_GPU

  // After the above sync all the GPU->pinned copies are complete. Any
  // deferred copies of pinned->CPU can now be done.
  for (auto& def : deferred_pinned_) {
    bool cuda_used = false;
    auto err = CopyBuffer(
        "pinned buffer", TRITONSERVER_MEMORY_CPU_PINNED, 0,
        def.tensor_memory_type_, def.tensor_memory_id_, def.pinned_memory_size_,
        def.pinned_memory_, def.tensor_buffer_ + def.tensor_buffer_offset_,
        stream_, &cuda_used);
    need_sync_ |= cuda_used;

    // If something goes wrong with the copy all the pending
    // responses fail...
    if (err != nullptr) {
      for (auto& pr : def.requests_) {
        auto& response = pr.first;
        if (*response != nullptr) {
          LOG_IF_ERROR(
              TRITONBACKEND_ResponseSend(
                  *response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
              "failed to send error response");
          *response = nullptr;
        }
      }
      TRITONSERVER_ErrorDelete(err);
    }
  }

#ifdef TRITON_ENABLE_GPU
  // Record the new event location if deferred copies occur
  if ((!deferred_pinned_.empty()) && need_sync_ && (event_ != nullptr)) {
    cudaEventRecord(event_, stream_);
  }
#endif  // TRITON_ENABLE_GPU
  deferred_pinned_.clear();

  return need_sync_;
}

bool
BackendInputCollector::SetFixedSizeInputTensor(
    TRITONBACKEND_Input* request_input, const size_t tensor_buffer_offset,
    char* tensor_buffer, const size_t tensor_buffer_byte_size,
    const TRITONSERVER_MemoryType tensor_memory_type,
    const int64_t tensor_memory_type_id,
    const TRITONSERVER_MemoryType use_pinned_memory_type,
    TRITONBACKEND_Response** response)
{
  bool cuda_copy = false;

  const char* name;
  uint32_t buffer_count;
  RESPOND_AND_SET_NULL_IF_ERROR(
      response, TRITONBACKEND_InputProperties(
                    request_input, &name, nullptr, nullptr, nullptr, nullptr,
                    &buffer_count));
  if (*response == nullptr) {
    return cuda_copy;
  }

  // First iterate through the buffers to ensure the byte size is proper
  size_t total_byte_size = 0;
  for (size_t idx = 0; idx < buffer_count; ++idx) {
    const void* src_buffer;
    size_t src_byte_size;
    TRITONSERVER_MemoryType src_memory_type;
    int64_t src_memory_type_id;

    RESPOND_AND_SET_NULL_IF_ERROR(
        response, TRITONBACKEND_InputBuffer(
                      request_input, idx, &src_buffer, &src_byte_size,
                      &src_memory_type, &src_memory_type_id));
    total_byte_size += src_byte_size;
  }

  if ((tensor_buffer_offset + total_byte_size) > tensor_buffer_byte_size) {
    RESPOND_AND_SET_NULL_IF_ERROR(
        response,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string(
                "unexpected total byte size " +
                std::to_string(tensor_buffer_offset + total_byte_size) +
                " for input '" + name + "', expecting " +
                std::to_string(tensor_buffer_byte_size))
                .c_str()));
    return cuda_copy;
  } else if (response == nullptr) {
    return cuda_copy;
  }

  // Request input tensor data may be in multiple non-contiguous
  // buffers.
  size_t input_offset = 0;
  for (size_t idx = 0; idx < buffer_count; ++idx) {
    const void* src_buffer;
    size_t src_byte_size;
    TRITONSERVER_MemoryType src_memory_type;
    int64_t src_memory_type_id;

    RESPOND_AND_SET_NULL_IF_ERROR(
        response, TRITONBACKEND_InputBuffer(
                      request_input, idx, &src_buffer, &src_byte_size,
                      &src_memory_type, &src_memory_type_id));
    if (*response == nullptr) {
      return cuda_copy;
    }

    // If the request buffer matches the memory type that should use an
    // intermediate pinned memory buffer for the transfer, then just
    // record the input as pending and increase the size required for
    // the intermediate pinned buffer. We only do this check for the
    // first buffer of an input and apply the same policy for all
    // buffers. So if an inputs data is split over different memory
    // types this may not be ideal but that should be a very rare
    // situation.
    if ((idx == 0) &&
        (use_pinned_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) &&
        (src_memory_type == use_pinned_memory_type)) {
      if (pending_pinned_byte_size_ == 0) {
        pending_pinned_offset_ = tensor_buffer_offset;
      }

      pending_pinned_byte_size_ += total_byte_size;
      pending_pinned_inputs_.push_back(std::make_pair(response, request_input));
      return cuda_copy;
    }

    // Direct copy without intermediate pinned memory.
    bool cuda_used = false;
    RESPOND_AND_SET_NULL_IF_ERROR(
        response,
        CopyBuffer(
            name, src_memory_type, src_memory_type_id, tensor_memory_type,
            tensor_memory_type_id, src_byte_size, src_buffer,
            tensor_buffer + tensor_buffer_offset + input_offset, stream_,
            &cuda_used));
    cuda_copy |= cuda_used;
    if (*response == nullptr) {
      return cuda_copy;
    }

    input_offset += src_byte_size;
  }

  return cuda_copy;
}

bool
BackendInputCollector::FlushPendingPinned(
    char* tensor_buffer, const size_t tensor_buffer_byte_size,
    const TRITONSERVER_MemoryType tensor_memory_type,
    const int64_t tensor_memory_type_id)
{
  bool cuda_copy = false;

  // Will be copying from CPU->pinned->GPU or GPU->pinned->CPU

  // Always need a pinned buffer...
  auto pinned_memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
  int64_t pinned_memory_id = 0;

  char* pinned_memory = nullptr;
#ifdef TRITON_ENABLE_GPU
  auto cuerr = cudaHostAlloc(
      (void**)&pinned_memory, pending_pinned_byte_size_, cudaHostAllocPortable);
  if (cuerr != cudaSuccess) {
    pinned_memory_type = TRITONSERVER_MEMORY_CPU;
  }
#else
  pinned_memory_type = TRITONSERVER_MEMORY_CPU;
#endif  // TRITON_ENABLE_GPU

  // If the pinned buffer isn't actually pinned memory then just
  // perform a direct copy. In this case 'pinned_memory' is just
  // deallocated and not used.
  if (pinned_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
    pinned_memory = nullptr;

    size_t offset = 0;
    for (auto& pr : pending_pinned_inputs_) {
      auto& response = pr.first;
      auto& request_input = pr.second;

      uint64_t byte_size;
      RESPOND_AND_SET_NULL_IF_ERROR(
          response, TRITONBACKEND_InputProperties(
                        request_input, nullptr, nullptr, nullptr, nullptr,
                        &byte_size, nullptr));

      cuda_copy |= SetFixedSizeInputTensor(
          request_input, pending_pinned_offset_ + offset, tensor_buffer,
          tensor_buffer_byte_size, tensor_memory_type, tensor_memory_type_id,
          TRITONSERVER_MEMORY_CPU_PINNED, response);
      offset += byte_size;
    }
  }
  // We have a pinned buffer so copy the pending input buffer(s) into
  // the pinned memory.
  else {  // pinned_memory_type == TRITONSERVER_MEMORY_CPU_PINNED
    bool cuda_used = false;
    size_t offset = 0;
    for (auto& pr : pending_pinned_inputs_) {
      auto& response = pr.first;
      auto& request_input = pr.second;

      uint64_t byte_size;
      RESPOND_AND_SET_NULL_IF_ERROR(
          response, TRITONBACKEND_InputProperties(
                        request_input, nullptr, nullptr, nullptr, nullptr,
                        &byte_size, nullptr));

      cuda_used |= SetFixedSizeInputTensor(
          request_input, offset, pinned_memory, pending_pinned_byte_size_,
          pinned_memory_type, pinned_memory_id, TRITONSERVER_MEMORY_CPU_PINNED,
          response);
      offset += byte_size;
    }

    cuda_copy |= cuda_used;

    // If the copy was not async (i.e. if request input was in CPU so
    // a CPU->CPU-PINNED copy was performed above), then the pinned
    // buffer now holds the tensor contents and we can immediately
    // issue the copies from the pinned buffer to the tensor.
    //
    // Otherwise the GPU->CPU-PINNED async copies are in flight and we
    // simply remember the pinned buffer and the corresponding
    // request inputs so that we can do the pinned->CPU copies in
    // finalize after we have waited for all async copies to complete.
    if (!cuda_used) {
      auto err = CopyBuffer(
          "pinned buffer", pinned_memory_type, pinned_memory_id,
          tensor_memory_type, tensor_memory_type_id, pending_pinned_byte_size_,
          pinned_memory, tensor_buffer + pending_pinned_offset_, stream_,
          &cuda_used);
      cuda_copy |= cuda_used;

      // If something goes wrong with the copy all the pending
      // responses fail...
      if (err != nullptr) {
        for (auto& pr : pending_pinned_inputs_) {
          auto& response = pr.first;
          if (*response != nullptr) {
            LOG_IF_ERROR(
                TRITONBACKEND_ResponseSend(
                    *response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
                "failed to send error response");
            *response = nullptr;
          }
        }
        TRITONSERVER_ErrorDelete(err);
      }
    } else {  // cuda_used
      deferred_pinned_.emplace_back(
          pinned_memory, pending_pinned_byte_size_, tensor_buffer,
          pending_pinned_offset_, tensor_memory_type, tensor_memory_type_id,
          std::move(pending_pinned_inputs_));
    }
  }

  // Pending pinned copies are handled...
  pending_pinned_byte_size_ = 0;
  pending_pinned_offset_ = 0;
  pending_pinned_inputs_.clear();

  // Need to hold on to the allocated pinned buffer as there are still
  // copies in flight. Will delete it in finalize.
  if (pinned_memory != nullptr) {
    pinned_memories_.push_back(pinned_memory);
  }

  return cuda_copy;
}

}}}  // namespace nvidia::inferenceserver::backend
