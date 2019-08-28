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

#include "src/core/backend_context.h"

#include "src/core/logging.h"
#include "src/core/provider.h"

namespace nvidia { namespace inferenceserver {

BackendContext::BackendContext(
    const std::string& name, const int gpu_device, const int max_batch_size)
    : name_(name), gpu_device_(gpu_device), max_batch_size_(max_batch_size)
{
#ifdef TRTIS_ENABLE_GPU
  stream_ = nullptr;
#endif  // TRTIS_ENABLE_GPU
}

BackendContext::~BackendContext()
{
#ifdef TRTIS_ENABLE_GPU
  if (stream_ != nullptr) {
    cudaError_t err = cudaStreamDestroy(stream_);
    if (err != cudaSuccess) {
      LOG_ERROR << "Failed to destroy cuda stream: " << cudaGetErrorString(err);
    }
    stream_ = nullptr;
  }
#endif  // TRTIS_ENABLE_GPU
}

Status
BackendContext::CreateCudaStream(const int cuda_stream_priority)
{
#ifdef TRTIS_ENABLE_GPU
  int device_cnt;
  auto cuerr = cudaGetDeviceCount(&device_cnt);
  // Do nothing if there is no CUDA device since all data transfer will be done
  // within CPU memory
  if ((cuerr != cudaErrorNoDevice) && (cuerr != cudaErrorInsufficientDriver)) {
    if (cuerr == cudaSuccess) {
      cuerr = cudaStreamCreateWithPriority(
          &stream_, cudaStreamDefault, cuda_stream_priority);
    }
    if (cuerr != cudaSuccess) {
      return Status(
          RequestStatusCode::INTERNAL, "unable to create stream for " + name_ +
                                           ": " + cudaGetErrorString(cuerr));
    }
  }
#endif  // TRTIS_ENABLE_GPU
  return Status::Success;
}

bool
BackendContext::SetInputBuffer(
    const std::string& name, const std::vector<size_t>& expected_byte_sizes,
    std::vector<Scheduler::Payload>* payloads,
    TRTSERVER_Memory_Type dst_memory_type, char* input_buffer)
{
  bool cuda_copy = false;
  // Visit the payloads in order and copy the input tensors to
  // 'buffer'.
  size_t buffer_copy_offset = 0;
  for (size_t idx = 0; idx < expected_byte_sizes.size(); idx++) {
    auto& payload = (*payloads)[idx];
    const size_t expected_byte_size = expected_byte_sizes[idx];

    size_t copied_byte_size = 0;
    while (payload.status_.IsOk()) {
      auto src_memory_type = dst_memory_type;
      const void* content;
      size_t content_byte_size = expected_byte_size - copied_byte_size;
      payload.status_ = payload.request_provider_->GetNextInputContent(
          name, &content, &content_byte_size, &src_memory_type, false);
      if (!payload.status_.IsOk()) {
        break;
      }

      // No more input content available then done with copying...
      if (content == nullptr) {
        break;
      }

      if ((copied_byte_size + content_byte_size) > expected_byte_size) {
        payload.status_ = Status(
            RequestStatusCode::INVALID_ARG,
            "unexpected size " +
                std::to_string(copied_byte_size + content_byte_size) +
                " for inference input '" + name + "', expecting " +
                std::to_string(expected_byte_size));
        break;
      }

      if (content_byte_size > 0) {
        bool cuda_used = false;
        payload.status_ = CopyBuffer(
            name, src_memory_type, dst_memory_type, content_byte_size, content,
            input_buffer + buffer_copy_offset + copied_byte_size, &cuda_used);
        cuda_copy |= cuda_used;
      }
      copied_byte_size += content_byte_size;
    }

    if (payload.status_.IsOk() && (copied_byte_size != expected_byte_size)) {
      payload.status_ = Status(
          RequestStatusCode::INTERNAL,
          "expected " + std::to_string(expected_byte_size) +
              " bytes of data for inference input '" + name + "', got " +
              std::to_string(copied_byte_size));
    }

    buffer_copy_offset += expected_byte_size;
  }

  return cuda_copy;
}

bool
BackendContext::SetFixedSizeOutputBuffer(
    const std::string& name, const size_t batch1_byte_size, const char* content,
    const std::vector<int64_t>& content_shape,
    TRTSERVER_Memory_Type src_memory_type,
    std::vector<Scheduler::Payload>* payloads)
{
  bool cuda_copy = false;
  size_t content_offset = 0;
  for (auto& payload : *payloads) {
    const InferRequestHeader& request_header =
        payload.request_provider_->RequestHeader();
    const size_t expected_byte_size =
        request_header.batch_size() * batch1_byte_size;

    // If 'payload' should have valid output (status ok) and
    // if 'payload' requested this output then copy it from
    // 'content'. If it did not request this output then just
    // skip it in the 'content'.
    if (payload.status_.IsOk() && (payload.response_provider_ != nullptr) &&
        payload.response_provider_->RequiresOutput(name)) {
      auto dst_memory_type = src_memory_type;
      void* buffer = nullptr;

      // try to get buffer with the same memory type as the output tensor
      Status status = payload.response_provider_->AllocateOutputBuffer(
          name, &buffer, expected_byte_size, content_shape, src_memory_type);

      if (status.IsOk() && (expected_byte_size != 0)) {
        if ((buffer == nullptr) && (src_memory_type != TRTSERVER_MEMORY_CPU)) {
          // Use default (CPU memory type) if preferred type can't be fulfilled
          status = payload.response_provider_->AllocateOutputBuffer(
              name, &buffer, expected_byte_size, content_shape);
          dst_memory_type = TRTSERVER_MEMORY_CPU;
        }

        if (status.IsOk()) {
          if (buffer == nullptr) {
            status = Status(
                RequestStatusCode::INTERNAL,
                "all attempts to allocate buffer for output '" + name +
                    "' failed");
          } else {
            bool cuda_used = false;
            status = CopyBuffer(
                name, src_memory_type, dst_memory_type, expected_byte_size,
                content + content_offset, buffer, &cuda_used);
            cuda_copy |= cuda_used;
          }
        }
      }

      payload.status_ = status;
    }

    content_offset += expected_byte_size;
  }

  return cuda_copy;
}

Status
BackendContext::CopyBuffer(
    const std::string& name, const TRTSERVER_Memory_Type src_memory_type,
    const TRTSERVER_Memory_Type dst_memory_type, const size_t byte_size,
    const void* src, void* dst, bool* cuda_used)
{
  *cuda_used = false;

  if ((src_memory_type == TRTSERVER_MEMORY_CPU) &&
      (dst_memory_type == TRTSERVER_MEMORY_CPU)) {
    memcpy(dst, src, byte_size);
  } else {
#ifdef TRTIS_ENABLE_GPU
    // [TODO] use cudaMemcpyDefault if UVM is supported for the device
    auto copy_kind = cudaMemcpyDeviceToDevice;
    if (src_memory_type == TRTSERVER_MEMORY_CPU) {
      copy_kind = cudaMemcpyHostToDevice;
    } else if (dst_memory_type == TRTSERVER_MEMORY_CPU) {
      copy_kind = cudaMemcpyDeviceToHost;
    }
    cudaError_t err = cudaMemcpyAsync(dst, src, byte_size, copy_kind, stream_);
    if (err != cudaSuccess) {
      return Status(
          RequestStatusCode::INTERNAL,
          "failed to use CUDA copy for tensor '" + name +
              "': " + std::string(cudaGetErrorString(err)));
    } else {
      *cuda_used = true;
    }
#else
    return Status(
        RequestStatusCode::INTERNAL, "try to use CUDA copy for tensor '" +
                                         name + "' while GPU is not supported");
#endif  // TRTIS_ENABLE_GPU
  }
  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
