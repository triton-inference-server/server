// Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "src/core/cuda_utils.h"
#include "src/core/logging.h"
#include "src/core/metric_model_reporter.h"
#include "src/core/nvtx.h"
#include "triton/common/sync_queue.h"

#ifdef TRITON_ENABLE_GPU
#include "src/core/kernel.h"
#endif  // TRITON_ENABLE_GPU

namespace nvidia { namespace inferenceserver {

namespace {

TRITONSERVER_MemoryType
GetUsePinnedMemoryType(TRITONSERVER_MemoryType ref_buffer_type)
{
  // The following matrix is used for both input and output.
  // src   \ dest | non-pinned    | pinned     | device
  // non-pinned   | memcpy        | memcpy     | buffer needed
  // pinned       | memcpy        | memcpy     | cudaMemcpy
  // device       | buffer needed | cudaMemcpy | cudaMemcpy
  if (ref_buffer_type == TRITONSERVER_MEMORY_CPU_PINNED) {
    return TRITONSERVER_MEMORY_CPU_PINNED;
  }

  return (ref_buffer_type == TRITONSERVER_MEMORY_CPU) ? TRITONSERVER_MEMORY_GPU
                                                      : TRITONSERVER_MEMORY_CPU;
}

}  // namespace

//
// BackendContext
//
BackendContext::BackendContext(
    const std::string& name, const int gpu_device, const int max_batch_size,
    const bool enable_pinned_input, const bool enable_pinned_output,
    const size_t gather_kernel_buffer_threshold,
    std::shared_ptr<MetricModelReporter>&& metric_reporter)
    : name_(name), gpu_device_(gpu_device), max_batch_size_(max_batch_size),
      enable_pinned_input_(enable_pinned_input),
      enable_pinned_output_(enable_pinned_output),
      gather_kernel_buffer_threshold_(gather_kernel_buffer_threshold),
      metric_reporter_(std::move(metric_reporter))
{
#ifdef TRITON_ENABLE_GPU
  stream_ = nullptr;
#endif  // TRITON_ENABLE_GPU
}

BackendContext::~BackendContext()
{
#ifdef TRITON_ENABLE_GPU
  if (stream_ != nullptr) {
    cudaError_t err = cudaStreamDestroy(stream_);
    if (err != cudaSuccess) {
      LOG_ERROR << "Failed to destroy cuda stream: " << cudaGetErrorString(err);
    }
    stream_ = nullptr;
  }
#endif  // TRITON_ENABLE_GPU
}

Status
BackendContext::CreateCudaStream(
    const int cuda_stream_priority, cudaStream_t* stream)
{
#ifdef TRITON_ENABLE_GPU
  if (gpu_device_ != NO_GPU_DEVICE) {
    // Make sure that correct device is set before creating stream and
    // then restore the device to what was set by the caller.
    int current_device;
    auto cuerr = cudaGetDevice(&current_device);
    bool overridden = false;
    if (cuerr == cudaSuccess) {
      overridden = (current_device != gpu_device_);
      if (overridden) {
        cuerr = cudaSetDevice(gpu_device_);
      }
    }

    if (cuerr == cudaSuccess) {
      cudaStream_t* s = (stream == nullptr) ? &stream_ : stream;
      cuerr = cudaStreamCreateWithPriority(
          s, cudaStreamDefault, cuda_stream_priority);
    }

    if (overridden) {
      cudaSetDevice(current_device);
    }

    if (cuerr != cudaSuccess) {
      return Status(
          Status::Code::INTERNAL, "unable to create stream for " + name_ +
                                      ": " + cudaGetErrorString(cuerr));
    }
  }
#endif  // TRITON_ENABLE_GPU

  return Status::Success;
}

Status
BackendContext::CompareOutputDims(
    const std::string& tensor_name, const std::vector<int64_t>& model_shape,
    const DimsList& dims, const bool supports_batching)
{
  if (supports_batching) {
    DimsList full_dims;
    full_dims.Add(WILDCARD_DIM);
    for (int i = 0; i < dims.size(); ++i) {
      full_dims.Add(dims[i]);
    }

    bool succ = (model_shape.size() == (size_t)full_dims.size());
    if (succ) {
      for (int i = 0; i < full_dims.size(); ++i) {
        const int64_t model_dim = model_shape[i];
        if (full_dims[i] != WILDCARD_DIM) {
          succ &= (model_dim == full_dims[i]);
        }
      }
    }

    if (!succ) {
      return Status(
          Status::Code::INVALID_ARG,
          "tensor '" + tensor_name + "': the model expects " +
              std::to_string(model_shape.size()) + " dimensions (shape " +
              DimsListToString(model_shape) +
              ") but the model configuration specifies " +
              std::to_string(full_dims.size()) +
              " dimensions (an initial batch dimension because max_batch_size "
              "> 0 followed by the explicit tensor shape, making complete "
              "shape " +
              DimsListToString(full_dims) + ")");
    }
  } else {
    // ! supports_batching
    bool succ = (model_shape.size() == (size_t)dims.size());
    if (succ) {
      for (int i = 0; i < dims.size(); ++i) {
        const int64_t model_dim = model_shape[i];
        if (dims[i] != WILDCARD_DIM) {
          succ &= (model_dim == dims[i]);
        }
      }
    }

    if (!succ) {
      return Status(
          Status::Code::INVALID_ARG,
          "tensor '" + tensor_name + "': the model expects " +
              std::to_string(model_shape.size()) + " dimensions (shape " +
              DimsListToString(model_shape) +
              ") but the model configuration specifies " +
              std::to_string(dims.size()) + " dimensions (shape " +
              DimsListToString(dims) + ")");
    }
  }

  return Status::Success;
}

//
// BackendResponder
//
void
BackendResponder::ProcessTensor(
    const std::string& name, const inference::DataType datatype,
    std::vector<int64_t>& batchn_shape, const char* buffer,
    const TRITONSERVER_MemoryType memory_type, const int64_t memory_type_id)
{
  // A value of CPU_PINNED indicates that pinned memory buffer is not
  // needed for this tensor. Any other value indicates that a pinned
  // memory buffer is needed when the target memory type matches
  // 'use_pinned_memory_type'.
  TRITONSERVER_MemoryType use_pinned_memory_type =
      TRITONSERVER_MEMORY_CPU_PINNED;
  if (pinned_enabled_) {
    use_pinned_memory_type = GetUsePinnedMemoryType(memory_type);
  }

  size_t tensor_offset = 0;

  for (size_t idx = 0; idx < responses_->size(); idx++) {
    auto& request = requests_[idx];
    auto& response = (*responses_)[idx];

    // If then pending copies are from tensor buffer that is not
    // contiguous with 'response's part of that buffer, then need to
    // go ahead and perform the pending copies so that can start a
    // new contiguous region if necessary.
    if ((pending_pinned_byte_size_ > 0) &&
        (tensor_offset !=
         (pending_pinned_byte_size_ + pending_pinned_offset_))) {
      need_sync_ |= FlushPendingPinned(buffer, memory_type, memory_type_id);
    }

    // Override shape to be correct for this response.
    if (max_batch_size_ != BackendContext::NO_BATCHING) {
      batchn_shape[0] = request->BatchSize();
    }

    const size_t tensor_byte_size = GetByteSize(datatype, batchn_shape);

    InferenceResponse::Output* response_output = nullptr;
    if ((response != nullptr) &&
        (request->ImmutableRequestedOutputs().find(name) !=
         request->ImmutableRequestedOutputs().end())) {
      response->AddOutput(name, datatype, batchn_shape, &response_output);
      need_sync_ |= SetFixedSizeOutputBuffer(
          &response, response_output, tensor_byte_size, tensor_offset, buffer,
          memory_type, memory_type_id, use_pinned_memory_type);
    }

    tensor_offset += tensor_byte_size;
  }

  // Done with the tensor, flush any pending pinned copies.
  need_sync_ |= FlushPendingPinned(buffer, memory_type, memory_type_id);
#ifdef TRITON_ENABLE_GPU
  if (need_sync_ && (event_ != nullptr)) {
    cudaEventRecord(event_, stream_);
  }
#endif  // TRITON_ENABLE_GPU
}

void
BackendResponder::ProcessTensor(
    const std::string& name, const std::string& input_name,
    const inference::DataType datatype,
    const std::vector<int64_t>& batchn_shape, const char* buffer,
    const TRITONSERVER_MemoryType memory_type, const int64_t memory_type_id)
{
  // A value of CPU_PINNED indicates that pinned memory buffer is not
  // needed for this tensor. Any other value indicates that a pinned
  // memory buffer is needed when the target memory type matches
  // 'use_pinned_memory_type'.
  TRITONSERVER_MemoryType use_pinned_memory_type =
      TRITONSERVER_MEMORY_CPU_PINNED;
  if (pinned_enabled_) {
    use_pinned_memory_type = GetUsePinnedMemoryType(memory_type);
  }

  size_t tensor_offset = 0;

  for (size_t idx = 0; idx < responses_->size(); idx++) {
    auto& request = requests_[idx];
    auto& response = (*responses_)[idx];

    // If then pending copies are from tensor buffer that is not
    // contiguous with 'response's part of that buffer, then need to
    // go ahead and perform the pending copies so that can start a
    // new contiguous region if necessary.
    if ((pending_pinned_byte_size_ > 0) &&
        (tensor_offset !=
         (pending_pinned_byte_size_ + pending_pinned_offset_))) {
      need_sync_ |= FlushPendingPinned(buffer, memory_type, memory_type_id);
    }

    // Override shape to be correct for this response, with a naive assumption
    // that the dynamic dimension in output is mapped to the same dimension
    // in the input
    const InferenceRequest::Input* input = nullptr;
    request->ImmutableInput(input_name, &input);
    const auto& input_batchn_shape = input->ShapeWithBatchDim();
    auto output_batchn_shape = batchn_shape;
    for (size_t dim_idx = 0; dim_idx < output_batchn_shape.size(); dim_idx++) {
      if (output_batchn_shape[dim_idx] == -1) {
        output_batchn_shape[dim_idx] = input_batchn_shape[dim_idx];
      }
    }

    const size_t tensor_byte_size = GetByteSize(datatype, output_batchn_shape);

    InferenceResponse::Output* response_output = nullptr;
    if ((response != nullptr) &&
        (request->ImmutableRequestedOutputs().find(name) !=
         request->ImmutableRequestedOutputs().end())) {
      response->AddOutput(
          name, datatype, output_batchn_shape, &response_output);
      need_sync_ |= SetFixedSizeOutputBuffer(
          &response, response_output, tensor_byte_size, tensor_offset, buffer,
          memory_type, memory_type_id, use_pinned_memory_type);
    }

    tensor_offset += tensor_byte_size;
  }

  // Done with the tensor, flush any pending pinned copies.
  need_sync_ |= FlushPendingPinned(buffer, memory_type, memory_type_id);
#ifdef TRITON_ENABLE_GPU
  if (need_sync_ && (event_ != nullptr)) {
    cudaEventRecord(event_, stream_);
  }
#endif  // TRITON_ENABLE_GPU
}

bool
BackendResponder::Finalize()
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
    auto pinned_memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
    int64_t pinned_memory_id = 0;
    char* pinned_buffer = def.pinned_memory_->MutableBuffer(
        &pinned_memory_type, &pinned_memory_id);

    size_t offset = 0;
    for (auto& pr : def.responses_) {
      std::unique_ptr<InferenceResponse>* response = pr.first;
      InferenceResponse::Output* response_output = pr.second;

      const void* response_buffer;
      size_t response_byte_size;
      TRITONSERVER_MemoryType response_memory_type;
      int64_t response_memory_type_id;
      void* userp;

      Status status = response_output->DataBuffer(
          &response_buffer, &response_byte_size, &response_memory_type,
          &response_memory_type_id, &userp);
      if (!status.IsOk()) {
        LOG_STATUS_ERROR(
            InferenceResponse::SendWithStatus(
                std::move(*response), TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                status),
            "error sending failure response");
      } else {
        bool cuda_used = false;
        status = CopyBuffer(
            response_output->Name(), pinned_memory_type, pinned_memory_id,
            response_memory_type, response_memory_type_id, response_byte_size,
            pinned_buffer + offset, const_cast<void*>(response_buffer), stream_,
            &cuda_used);
        need_sync_ |= cuda_used;

        if (!status.IsOk()) {
          LOG_STATUS_ERROR(
              InferenceResponse::SendWithStatus(
                  std::move(*response), TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                  status),
              "error sending failure response");
        }
      }

      offset += response_byte_size;
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
BackendResponder::SetFixedSizeOutputBuffer(
    std::unique_ptr<InferenceResponse>* response,
    InferenceResponse::Output* response_output, const size_t tensor_byte_size,
    const size_t tensor_offset, const char* tensor_buffer,
    const TRITONSERVER_MemoryType tensor_memory_type,
    const int64_t tensor_memory_type_id,
    const TRITONSERVER_MemoryType use_pinned_memory_type)
{
  void* buffer = nullptr;
  bool cuda_copy = false;

  TRITONSERVER_MemoryType actual_memory_type = tensor_memory_type;
  int64_t actual_memory_type_id = tensor_memory_type_id;

  Status status = response_output->AllocateDataBuffer(
      &buffer, tensor_byte_size, &actual_memory_type, &actual_memory_type_id);
  if (!status.IsOk()) {
    LOG_STATUS_ERROR(
        InferenceResponse::SendWithStatus(
            std::move(*response), TRITONSERVER_RESPONSE_COMPLETE_FINAL, status),
        "error sending failure response");
    return cuda_copy;
  }

  // If the response buffer matches the memory type that should use an
  // intermediate pinned memory buffer for the transfer, then just
  // record the response as pending and increase the size required for
  // the intermediate pinned buffer.
  if ((use_pinned_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) &&
      (actual_memory_type == use_pinned_memory_type)) {
    if (pending_pinned_byte_size_ == 0) {
      pending_pinned_offset_ = tensor_offset;
    }

    pending_pinned_byte_size_ += tensor_byte_size;
    pending_pinned_outputs_.push_back(
        std::make_pair(response, response_output));
  } else {
    // Direct copy without intermediate pinned memory.
    bool cuda_used = false;
    status = CopyBuffer(
        response_output->Name(), tensor_memory_type, tensor_memory_type_id,
        actual_memory_type, actual_memory_type_id, tensor_byte_size,
        tensor_buffer + tensor_offset, buffer, stream_, &cuda_used);
    cuda_copy |= cuda_used;

    if (!status.IsOk()) {
      LOG_STATUS_ERROR(
          InferenceResponse::SendWithStatus(
              std::move(*response), TRITONSERVER_RESPONSE_COMPLETE_FINAL,
              status),
          "error sending failure response");
      return cuda_copy;
    }
  }

  return cuda_copy;
}

bool
BackendResponder::FlushPendingPinned(
    const char* tensor_buffer, const TRITONSERVER_MemoryType tensor_memory_type,
    const int64_t tensor_memory_type_id)
{
  bool cuda_copy = false;

  // Will be copying from CPU->pinned->GPU or GPU->pinned->CPU

  // Always need a pinned buffer...
  auto pinned_memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
  int64_t pinned_memory_id = 0;

  std::unique_ptr<AllocatedMemory> pinned_memory(new AllocatedMemory(
      pending_pinned_byte_size_, pinned_memory_type, pinned_memory_id));
  char* pinned_buffer =
      pinned_memory->MutableBuffer(&pinned_memory_type, &pinned_memory_id);

  // If the pinned buffer isn't actually pinned memory then just
  // perform a direct copy. In this case 'pinned_memory' is just
  // deallocated and not used.
  if (pinned_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
    pinned_memory.reset();

    size_t offset = 0;
    for (auto& pr : pending_pinned_outputs_) {
      std::unique_ptr<InferenceResponse>* response = pr.first;
      InferenceResponse::Output* response_output = pr.second;

      const void* response_buffer;
      size_t response_byte_size;
      TRITONSERVER_MemoryType response_memory_type;
      int64_t response_memory_type_id;
      void* userp;

      Status status = response_output->DataBuffer(
          &response_buffer, &response_byte_size, &response_memory_type,
          &response_memory_type_id, &userp);
      if (!status.IsOk()) {
        LOG_STATUS_ERROR(
            InferenceResponse::SendWithStatus(
                std::move(*response), TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                status),
            "error sending failure response");
      } else {
        bool cuda_used = false;
        status = CopyBuffer(
            response_output->Name(), tensor_memory_type, tensor_memory_type_id,
            response_memory_type, response_memory_type_id, response_byte_size,
            tensor_buffer + pending_pinned_offset_ + offset,
            const_cast<void*>(response_buffer), stream_, &cuda_used);
        cuda_copy |= cuda_used;

        if (!status.IsOk()) {
          LOG_STATUS_ERROR(
              InferenceResponse::SendWithStatus(
                  std::move(*response), TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                  status),
              "error sending failure response");
        }
      }

      offset += response_byte_size;
    }
  }
  // We have a pinned buffer so do a single copy of a block of tensor
  // data to the pinned buffer.
  else {  // pinned_memory_type == TRITONSERVER_MEMORY_CPU_PINNED
    bool cuda_used = false;
    Status status = CopyBuffer(
        "pinned buffer", tensor_memory_type, tensor_memory_type_id,
        pinned_memory_type, pinned_memory_id, pending_pinned_byte_size_,
        tensor_buffer + pending_pinned_offset_, pinned_buffer, stream_,
        &cuda_used);
    cuda_copy |= cuda_used;

    // If something goes wrong with the copy all the pending
    // responses fail...
    if (!status.IsOk()) {
      for (auto& pr : pending_pinned_outputs_) {
        LOG_STATUS_ERROR(
            InferenceResponse::SendWithStatus(
                std::move(*(pr.first)), TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                status),
            "error sending failure response");
      }
    }

    // If the copy was not async (i.e. if tensor was in CPU so a
    // CPU->CPU-PINNED copy was performed above), then the pinned
    // buffer now holds the tensor contents and we can immediately
    // issue the copies from the pinned buffer to the
    // responses.
    //
    // Otherwise the GPU->CPU-PINNED async copies are in flight and we
    // simply remember the pinned buffer and the corresponding
    // response outputs so that we can do the pinned->CPU copies in
    // finalize after we have waited for all async copies to complete.
    if (!cuda_used) {
      size_t offset = 0;
      for (auto& pr : pending_pinned_outputs_) {
        std::unique_ptr<InferenceResponse>* response = pr.first;
        InferenceResponse::Output* response_output = pr.second;

        const void* response_buffer;
        size_t response_byte_size;
        TRITONSERVER_MemoryType response_memory_type;
        int64_t response_memory_type_id;
        void* userp;

        Status status = response_output->DataBuffer(
            &response_buffer, &response_byte_size, &response_memory_type,
            &response_memory_type_id, &userp);
        if (!status.IsOk()) {
          LOG_STATUS_ERROR(
              InferenceResponse::SendWithStatus(
                  std::move(*response), TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                  status),
              "error sending failure response");
        } else {
          bool cuda_used = false;
          status = CopyBuffer(
              response_output->Name(), pinned_memory_type, pinned_memory_id,
              response_memory_type, response_memory_type_id, response_byte_size,
              pinned_buffer + offset, const_cast<void*>(response_buffer),
              stream_, &cuda_used);
          cuda_copy |= cuda_used;

          if (!status.IsOk()) {
            LOG_STATUS_ERROR(
                InferenceResponse::SendWithStatus(
                    std::move(*response), TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                    status),
                "error sending failure response");
          }
        }

        offset += response_byte_size;
      }
    } else {
      deferred_pinned_.emplace_back(
          std::move(pinned_memory), std::move(pending_pinned_outputs_));
    }
  }

  // Pending pinned copies are handled...
  pending_pinned_byte_size_ = 0;
  pending_pinned_offset_ = 0;
  pending_pinned_outputs_.clear();

  // Need to hold on to the allocated pinned buffer as there are still
  // copies in flight. Will delete it in finalize.
  if (pinned_memory != nullptr) {
    pinned_memories_.push_back(std::move(pinned_memory));
  }

  return cuda_copy;
}


//
// BackendInputCollector
//
void
BackendInputCollector::ProcessTensor(
    const std::string& name, const inference::DataType datatype, char* buffer,
    const size_t buffer_byte_size, const TRITONSERVER_MemoryType memory_type,
    const int64_t memory_type_id)
{
  // A value of CPU_PINNED indicates that pinned memory buffer is not
  // needed for this tensor. Any other value indicates that a pinned
  // memory buffer is needed when the target memory type matches
  // 'use_pinned_memory_type'.
  TRITONSERVER_MemoryType use_pinned_memory_type =
      TRITONSERVER_MEMORY_CPU_PINNED;
  if (pinned_enabled_) {
    use_pinned_memory_type = GetUsePinnedMemoryType(memory_type);
  }
  const bool use_kernel = (kernel_buffer_threshold_ != 0);

  size_t buffer_offset = 0;

  for (size_t idx = 0; idx < requests_.size(); idx++) {
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
    if ((pending_copy_kernel_buffer_byte_size_ > 0) &&
        (buffer_offset != (pending_copy_kernel_buffer_byte_size_ +
                           pending_copy_kernel_buffer_offset_))) {
      need_sync_ |= FlushPendingCopyKernel(
          buffer, buffer_byte_size, memory_type, memory_type_id);
    }

    const InferenceRequest::Input* request_input;
    Status status = request->ImmutableInput(name, &request_input);
    if (!status.IsOk() && (response != nullptr)) {
      InferenceResponse::SendWithStatus(
          std::move(response), TRITONSERVER_RESPONSE_COMPLETE_FINAL, status);
    } else {
      need_sync_ |= SetFixedSizeInputTensor(
          request_input, buffer_offset, buffer, buffer_byte_size, memory_type,
          memory_type_id, use_pinned_memory_type, use_kernel, true, &response);
    }

    buffer_offset += request_input->Data()->TotalByteSize();
  }

  // Done with the tensor, flush any pending pinned copies.
  need_sync_ |=
      FlushPendingPinned(buffer, buffer_byte_size, memory_type, memory_type_id);
  need_sync_ |= FlushPendingCopyKernel(
      buffer, buffer_byte_size, memory_type, memory_type_id);
#ifdef TRITON_ENABLE_GPU
  if (need_sync_ && (event_ != nullptr)) {
    cudaEventRecord(event_, stream_);
  }
#endif  // TRITON_ENABLE_GPU
}

Status
BackendInputCollector::BatchInputShape(
    const inference::BatchInput& batch_input, std::vector<int64_t>* shape)
{
  *shape = std::vector<int64_t>{0};
  switch (batch_input.kind()) {
    case inference::BatchInput::BATCH_ELEMENT_COUNT:
    case inference::BatchInput::BATCH_ACCUMULATED_ELEMENT_COUNT: {
      (*shape)[0] = requests_.size();
      break;
    }
    case inference::BatchInput::BATCH_ACCUMULATED_ELEMENT_COUNT_WITH_ZERO: {
      (*shape)[0] = requests_.size() + 1;
      break;
    }
    case inference::BatchInput::BATCH_MAX_ELEMENT_COUNT_AS_SHAPE: {
      const auto& source_input = batch_input.source_input(0);
      for (size_t req_idx = 0; req_idx < requests_.size(); req_idx++) {
        const InferenceRequest::Input* repr_input;
        RETURN_IF_ERROR(
            requests_[req_idx]->ImmutableInput(source_input, &repr_input));
        (*shape)[0] = std::max(
            (*shape)[0], GetElementCount(repr_input->ShapeWithBatchDim()));
      }
      break;
    }
    default:
      break;
  }
  return Status::Success;
}

Status
BackendInputCollector::ProcessBatchInput(
    const inference::BatchInput& batch_input, char* buffer,
    const size_t buffer_byte_size, const TRITONSERVER_MemoryType memory_type,
    const int64_t memory_type_id)
{
#ifdef TRITON_ENABLE_GPU
  if (buffer_ready_event_ != nullptr) {
    cudaEventSynchronize(buffer_ready_event_);
    buffer_ready_event_ = nullptr;
  }
#endif  // TRITON_ENABLE_GPU
  char* input_buffer = buffer;
  std::unique_ptr<AllocatedMemory> internal_buffer;
  // Need a CPU buffer for modifying the value
  if (memory_type == TRITONSERVER_MEMORY_GPU) {
    internal_buffer.reset(new AllocatedMemory(
        buffer_byte_size, TRITONSERVER_MEMORY_CPU_PINNED, 0));
    input_buffer = internal_buffer->MutableBuffer();
  }
  const auto& data_type = batch_input.data_type();
  switch (batch_input.kind()) {
    case inference::BatchInput::BATCH_ELEMENT_COUNT: {
      const auto& source_input = batch_input.source_input(0);
      if (data_type == inference::TYPE_FP32) {
        SetElementCount<float>(source_input, input_buffer, buffer_byte_size);
      } else {
        SetElementCount<int32_t>(source_input, input_buffer, buffer_byte_size);
      }
      break;
    }
    case inference::BatchInput::BATCH_ACCUMULATED_ELEMENT_COUNT: {
      const auto& source_input = batch_input.source_input(0);
      if (data_type == inference::TYPE_FP32) {
        SetAccumulatedElementCount<float>(
            source_input, input_buffer, buffer_byte_size);
      } else {
        SetAccumulatedElementCount<int32_t>(
            source_input, input_buffer, buffer_byte_size);
      }
      break;
    }
    case inference::BatchInput::BATCH_ACCUMULATED_ELEMENT_COUNT_WITH_ZERO: {
      const auto& source_input = batch_input.source_input(0);
      if (data_type == inference::TYPE_FP32) {
        *reinterpret_cast<float*>(input_buffer) = 0;
        SetAccumulatedElementCount<float>(
            source_input, input_buffer + sizeof(float),
            buffer_byte_size - sizeof(float));
      } else {
        *reinterpret_cast<int32_t*>(input_buffer) = 0;
        SetAccumulatedElementCount<int32_t>(
            source_input, input_buffer + sizeof(int32_t),
            buffer_byte_size - sizeof(int32_t));
      }
      break;
    }
    case inference::BatchInput::BATCH_MAX_ELEMENT_COUNT_AS_SHAPE:
    default:
      return Status::Success;
  }
  if (memory_type == TRITONSERVER_MEMORY_GPU) {
    TRITONSERVER_MemoryType src_mem_type;
    int64_t src_mem_id;
    input_buffer = internal_buffer->MutableBuffer(&src_mem_type, &src_mem_id);
    bool cuda_used;
    RETURN_IF_ERROR(CopyBuffer(
        "batch input buffer", src_mem_type, src_mem_id, memory_type,
        memory_type_id, buffer_byte_size, input_buffer, buffer, stream_,
        &cuda_used));
    need_sync_ |= cuda_used;
  }
  return Status::Success;
}

template <typename T>
Status
BackendInputCollector::SetElementCount(
    const std::string& source_input, char* buffer,
    const size_t buffer_byte_size)
{
  size_t buffer_offset = 0;
  for (size_t req_idx = 0; req_idx < requests_.size(); req_idx++) {
    const InferenceRequest::Input* repr_input;
    RETURN_IF_ERROR(
        requests_[req_idx]->ImmutableInput(source_input, &repr_input));
    if (buffer_offset + sizeof(T) > buffer_byte_size) {
      return Status(
          Status::Code::INVALID_ARG,
          "unexpected total byte size for batch input");
    }
    *(reinterpret_cast<T*>(buffer) + req_idx) =
        GetElementCount(repr_input->ShapeWithBatchDim());
    buffer_offset += sizeof(T);
  }
  for (; buffer_offset + sizeof(T) <= buffer_byte_size;
       buffer_offset += sizeof(T)) {
    *reinterpret_cast<T*>(buffer + buffer_offset) = 0;
  }
  return Status::Success;
}

template <typename T>
Status
BackendInputCollector::SetAccumulatedElementCount(
    const std::string& source_input, char* buffer,
    const size_t buffer_byte_size)
{
  size_t accumulated_element_count = 0;
  size_t buffer_offset = 0;
  for (size_t req_idx = 0; req_idx < requests_.size(); req_idx++) {
    const InferenceRequest::Input* repr_input;
    RETURN_IF_ERROR(
        requests_[req_idx]->ImmutableInput(source_input, &repr_input));
    if (buffer_offset + sizeof(T) > buffer_byte_size) {
      return Status(
          Status::Code::INVALID_ARG,
          "unexpected total byte size for batch input");
    }
    accumulated_element_count +=
        GetElementCount(repr_input->ShapeWithBatchDim());
    *(reinterpret_cast<T*>(buffer) + req_idx) = accumulated_element_count;
    buffer_offset += sizeof(T);
  }
  for (; buffer_offset + sizeof(T) <= buffer_byte_size;
       buffer_offset += sizeof(T)) {
    *reinterpret_cast<T*>(buffer + buffer_offset) = accumulated_element_count;
  }
  return Status::Success;
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
#ifdef TRITON_ENABLE_GPU
  if (buffer_ready_event_ != nullptr) {
    cudaEventSynchronize(buffer_ready_event_);
    buffer_ready_event_ = nullptr;
  }
#endif  // TRITON_ENABLE_GPU
  for (auto& def : deferred_pinned_) {
    if (!def.finalized_) {
      need_sync_ |= def.Finalize(stream_);
    }
  }
  for (size_t i = 0; i < async_task_count_; i++) {
    need_sync_ |= completion_queue_.Get();
  }

#ifdef TRITON_ENABLE_GPU
  // Record the new event location if deferred copies occur
  if ((!deferred_pinned_.empty()) && need_sync_ && (event_ != nullptr)) {
    cudaEventRecord(event_, stream_);
  }
#endif  // TRITON_ENABLE_GPU

  return need_sync_;
}

bool
BackendInputCollector::DeferredPinned::Finalize(cudaStream_t stream)
{
  auto pinned_memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
  int64_t pinned_memory_id = 0;
  char* pinned_buffer =
      pinned_memory_->MutableBuffer(&pinned_memory_type, &pinned_memory_id);

  bool cuda_used = false;
  Status status = CopyBuffer(
      "pinned buffer", pinned_memory_type, pinned_memory_id,
      tensor_memory_type_, tensor_memory_id_, pinned_memory_->TotalByteSize(),
      pinned_buffer, tensor_buffer_ + tensor_buffer_offset_, stream,
      &cuda_used);

  // If something goes wrong with the copy all the pending
  // responses fail...
  if (!status.IsOk()) {
    for (auto& pr : requests_) {
      std::unique_ptr<InferenceResponse>* response = pr.first;
      if (*response != nullptr) {
        LOG_STATUS_ERROR(
            InferenceResponse::SendWithStatus(
                std::move(*response), TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                status),
            "error setting failure input tensor");
      }
    }
  }
  return cuda_used;
}

bool
BackendInputCollector::SetFixedSizeInputTensor(
    const InferenceRequest::Input* request_input,
    const size_t tensor_buffer_offset, char* tensor_buffer,
    const size_t tensor_buffer_byte_size,
    const TRITONSERVER_MemoryType tensor_memory_type,
    const int64_t tensor_memory_type_id,
    const TRITONSERVER_MemoryType use_pinned_memory_type, const bool use_kernel,
    const bool wait_buffer, std::unique_ptr<InferenceResponse>* response)
{
  bool cuda_copy = false;

  if ((tensor_buffer_offset + request_input->Data()->TotalByteSize()) >
      tensor_buffer_byte_size) {
    InferenceResponse::SendWithStatus(
        std::move(*response), TRITONSERVER_RESPONSE_COMPLETE_FINAL,
        Status(
            Status::Code::INVALID_ARG,
            "unexpected total byte size " +
                std::to_string(
                    tensor_buffer_offset +
                    request_input->Data()->TotalByteSize()) +
                " for input '" + request_input->Name() + "', expecting " +
                std::to_string(tensor_buffer_byte_size)));
    return cuda_copy;
  }

  // Request input tensor data may be in multiple non-contiguous
  // buffers.
  size_t input_offset = 0;
  for (size_t idx = 0; idx < request_input->DataBufferCount(); ++idx) {
    const void* src_buffer;
    size_t src_byte_size;
    TRITONSERVER_MemoryType src_memory_type;
    int64_t src_memory_type_id;

    Status status = request_input->DataBuffer(
        idx, &src_buffer, &src_byte_size, &src_memory_type,
        &src_memory_type_id);
    if (!status.IsOk()) {
      InferenceResponse::SendWithStatus(
          std::move(*response), TRITONSERVER_RESPONSE_COMPLETE_FINAL, status);
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

      pending_pinned_byte_size_ += request_input->Data()->TotalByteSize();
      pending_pinned_inputs_.push_back(std::make_pair(response, request_input));
      return cuda_copy;
    }
    // [FIXME] support other direction if prove to be faster, all kernel
    // handling code in this class asssumes the destination buffer is on device
    // If the request buffer and the destination buffer are accessible by all
    // GPUs (i.e. pinned, device), initiate the copy via copy CUDA kernel.
    // We only do this check for the
    // first buffer of an input and apply the same policy for all
    // buffers. So if an inputs data is split over different memory
    // types this may not be ideal but that should be a very rare
    // situation.
    // Currently checked direction:
    // pinned -> device
    // same device -> device
    // different device -> device
    if (use_kernel && (idx == 0) &&
        (src_memory_type != TRITONSERVER_MEMORY_CPU) &&
        (tensor_memory_type == TRITONSERVER_MEMORY_GPU)) {
      // [FIXME] Currently not allowing copy between devices as it requires
      // peer-to-peer access to be enabled. Peer-to-peer is enabled by default,
      // but server can still runs even if it fails to enable peer-to-peer.
      // Should provide a utility to check whether a device pair allows direct
      // access and use gather kernel accordingly
      if ((src_memory_type != TRITONSERVER_MEMORY_GPU) ||
          (src_memory_type_id == tensor_memory_type_id)) {
        if (pending_copy_kernel_buffer_byte_size_ == 0) {
          pending_copy_kernel_buffer_offset_ = tensor_buffer_offset;
        }

        pending_copy_kernel_buffer_byte_size_ +=
            request_input->Data()->TotalByteSize();
        pending_copy_kernel_input_buffer_counts_ +=
            request_input->DataBufferCount();
        pending_copy_kernel_inputs_.push_back(
            std::make_pair(response, request_input));
        return cuda_copy;
      }
    }

#ifdef TRITON_ENABLE_GPU
    if (wait_buffer && (buffer_ready_event_ != nullptr)) {
      cudaEventSynchronize(buffer_ready_event_);
      buffer_ready_event_ = nullptr;
    }
#endif  // TRITON_ENABLE_GPU

    // Direct copy without intermediate pinned memory.
    bool cuda_used = false;
    status = CopyBuffer(
        request_input->Name(), src_memory_type, src_memory_type_id,
        tensor_memory_type, tensor_memory_type_id, src_byte_size, src_buffer,
        tensor_buffer + tensor_buffer_offset + input_offset, stream_,
        &cuda_used);
    cuda_copy |= cuda_used;

    if (!status.IsOk()) {
      LOG_STATUS_ERROR(
          InferenceResponse::SendWithStatus(
              std::move(*response), TRITONSERVER_RESPONSE_COMPLETE_FINAL,
              status),
          "error setting failure input tensor");
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

  std::unique_ptr<AllocatedMemory> pinned_memory(new AllocatedMemory(
      pending_pinned_byte_size_, pinned_memory_type, pinned_memory_id));
  char* pinned_buffer =
      pinned_memory->MutableBuffer(&pinned_memory_type, &pinned_memory_id);

  // If the pinned buffer isn't actually pinned memory then just
  // perform a direct copy. In this case 'pinned_memory' is just
  // deallocated and not used.
  if (pinned_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
    pinned_memory.reset();

    size_t offset = 0;
    for (auto& pr : pending_pinned_inputs_) {
      std::unique_ptr<InferenceResponse>* response = pr.first;
      const InferenceRequest::Input* request_input = pr.second;

      cuda_copy |= SetFixedSizeInputTensor(
          request_input, pending_pinned_offset_ + offset, tensor_buffer,
          tensor_buffer_byte_size, tensor_memory_type, tensor_memory_type_id,
          TRITONSERVER_MEMORY_CPU_PINNED, false, true, response);
      offset += request_input->Data()->TotalByteSize();
    }
  }
  // We have a pinned buffer so copy the pending input buffer(s) into
  // the pinned memory.
  else {  // pinned_memory_type == TRITONSERVER_MEMORY_CPU_PINNED
    bool cuda_used = false;
    size_t offset = 0;
    if (!use_async_cpu_copy_) {
      for (auto& pr : pending_pinned_inputs_) {
        std::unique_ptr<InferenceResponse>* response = pr.first;
        const InferenceRequest::Input* request_input = pr.second;

        cuda_used |= SetFixedSizeInputTensor(
            request_input, offset, pinned_buffer, pending_pinned_byte_size_,
            pinned_memory_type, pinned_memory_id,
            TRITONSERVER_MEMORY_CPU_PINNED, false, false, response);
        offset += request_input->Data()->TotalByteSize();
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
#ifdef TRITON_ENABLE_GPU
        if (buffer_ready_event_ != nullptr) {
          cudaEventSynchronize(buffer_ready_event_);
          buffer_ready_event_ = nullptr;
        }
#endif  // TRITON_ENABLE_GPU
        Status status = CopyBuffer(
            "pinned input buffer H2D", pinned_memory_type, pinned_memory_id,
            tensor_memory_type, tensor_memory_type_id,
            pending_pinned_byte_size_, pinned_buffer,
            tensor_buffer + pending_pinned_offset_, stream_, &cuda_used);
        cuda_copy |= cuda_used;

        // If something goes wrong with the copy all the pending
        // responses fail...
        if (!status.IsOk()) {
          for (auto& pr : pending_pinned_inputs_) {
            std::unique_ptr<InferenceResponse>* response = pr.first;
            if (*response != nullptr) {
              LOG_STATUS_ERROR(
                  InferenceResponse::SendWithStatus(
                      std::move(*response),
                      TRITONSERVER_RESPONSE_COMPLETE_FINAL, status),
                  "error setting failure input tensor");
            }
          }
        }
      } else {  // cuda_used
        deferred_pinned_.emplace_back(
            std::move(pinned_memory), tensor_buffer, pending_pinned_offset_,
            tensor_memory_type, tensor_memory_type_id,
            std::move(pending_pinned_inputs_));
      }
    } else {
      async_task_count_++;
      deferred_pinned_.emplace_back(
          std::move(pinned_memory), tensor_buffer, pending_pinned_offset_,
          tensor_memory_type, tensor_memory_type_id,
          std::move(pending_pinned_inputs_));
      auto& deferred_pinned = deferred_pinned_.back();
      // Mark finalized to avoid duplicated call to DeferredPinned::Finalized()
      // in BackendInputCollector::Finalize()
      deferred_pinned_.back().finalized_ = true;
      auto incomplete_count = new std::atomic<size_t>(std::min(
          deferred_pinned_.back().requests_.size(),
          triton::common::AsyncWorkQueue::WorkerCount()));
      auto pending_pinned_byte_size = pending_pinned_byte_size_;
      size_t stride = (deferred_pinned_.back().requests_.size() +
                       triton::common::AsyncWorkQueue::WorkerCount() - 1) /
                      triton::common::AsyncWorkQueue::WorkerCount();
      auto pending_it = deferred_pinned_.back().requests_.begin();
      while (pending_it != deferred_pinned_.back().requests_.end()) {
        auto end_it = pending_it;
        auto next_offset = offset;
        for (size_t idx = 0; idx < stride; idx++) {
          next_offset += (*end_it).second->Data()->TotalByteSize();
          end_it++;
          if (end_it == deferred_pinned_.back().requests_.end()) {
            break;
          }
        }

        Status status =
            COMMON_ERROR_TO_STATUS(triton::common::AsyncWorkQueue::AddTask(
                [this, offset, pinned_buffer, pinned_memory_type,
                 pending_pinned_byte_size, pinned_memory_id, pending_it, end_it,
                 incomplete_count, &deferred_pinned]() mutable {
                  for (; pending_it != end_it; pending_it++) {
                    std::unique_ptr<InferenceResponse>* response =
                        (*pending_it).first;
                    const InferenceRequest::Input* request_input =
                        (*pending_it).second;
                    SetFixedSizeInputTensor(
                        request_input, offset, pinned_buffer,
                        pending_pinned_byte_size, pinned_memory_type,
                        pinned_memory_id, TRITONSERVER_MEMORY_CPU_PINNED, false,
                        false, response);
                    offset += request_input->Data()->TotalByteSize();
                  }
                  // The last segmented task will start the next phase of
                  // the internal pinned buffer copy
                  if (incomplete_count->fetch_sub(1) == 1) {
#ifdef TRITON_ENABLE_GPU
                    if (buffer_ready_event_ != nullptr) {
                      cudaEventSynchronize(buffer_ready_event_);
                      buffer_ready_event_ = nullptr;
                    }
#endif  // TRITON_ENABLE_GPU
                    completion_queue_.Put(deferred_pinned.Finalize(stream_));
                    delete incomplete_count;
                  }
                }));
        if (!status.IsOk()) {
          for (; pending_it != end_it; pending_it++) {
            std::unique_ptr<InferenceResponse>* response = (*pending_it).first;
            if (*response != nullptr) {
              LOG_STATUS_ERROR(
                  InferenceResponse::SendWithStatus(
                      std::move(*response),
                      TRITONSERVER_RESPONSE_COMPLETE_FINAL, status),
                  "error setting failure input tensor");
            }
          }
        }
        offset = next_offset;
        pending_it = end_it;
      }
    }
  }

  // Pending pinned copies are handled...
  pending_pinned_byte_size_ = 0;
  pending_pinned_offset_ = 0;
  pending_pinned_inputs_.clear();

  // Need to hold on to the allocated pinned buffer as there are still
  // copies in flight. Will delete it in finalize.
  if (pinned_memory != nullptr) {
    in_use_memories_.push_back(std::move(pinned_memory));
  }

  return cuda_copy;
}

bool
BackendInputCollector::FlushPendingCopyKernel(
    char* tensor_buffer, const size_t tensor_buffer_byte_size,
    const TRITONSERVER_MemoryType tensor_memory_type,
    const int64_t tensor_memory_type_id)
{
  if (pending_copy_kernel_inputs_.size() == 0) {
    return false;
  }

  bool cuda_copy = false;
  Status status = Status(Status::Code::INTERNAL, "");
  // Only try to launch kernel if buffer count is large enough for
  // good GPU utilization
  if (pending_copy_kernel_input_buffer_counts_ >= kernel_buffer_threshold_) {
    status = LaunchCopyKernel(
        tensor_buffer, tensor_buffer_byte_size, tensor_memory_type,
        tensor_memory_type_id);
    cuda_copy = status.IsOk();
    LOG_VERBOSE(1) << "gather kernel launched with status: "
                   << status.AsString();
  }
  // If kernel can't be launched then just perform a direct copy.
  if (!status.IsOk()) {
    size_t offset = 0;
    for (auto& pr : pending_copy_kernel_inputs_) {
      std::unique_ptr<InferenceResponse>* response = pr.first;
      const InferenceRequest::Input* request_input = pr.second;

      cuda_copy |= SetFixedSizeInputTensor(
          request_input, pending_copy_kernel_buffer_offset_ + offset,
          tensor_buffer, tensor_buffer_byte_size, tensor_memory_type,
          tensor_memory_type_id, TRITONSERVER_MEMORY_CPU_PINNED, false, true,
          response);
      offset += request_input->Data()->TotalByteSize();
    }
  }

  // Pending kernel copies are handled...
  pending_copy_kernel_buffer_byte_size_ = 0;
  pending_copy_kernel_buffer_offset_ = 0;
  pending_copy_kernel_input_buffer_counts_ = 0;
  pending_copy_kernel_inputs_.clear();

  return cuda_copy;
}

Status
BackendInputCollector::LaunchCopyKernel(
    char* tensor_buffer, const size_t tensor_buffer_byte_size,
    const TRITONSERVER_MemoryType tensor_memory_type,
    const int64_t tensor_memory_type_id)
{
#ifdef TRITON_ENABLE_GPU
  input_ptr_buffer_host_.emplace_back(new std::vector<int8_t*>());
  byte_size_buffer_host_.emplace_back(new std::vector<size_t>());
  byte_size_offset_buffer_host_.emplace_back(new std::vector<size_t>());

  auto& input_ptr_buffer_host = *input_ptr_buffer_host_.back();
  auto& byte_size_buffer_host = *byte_size_buffer_host_.back();
  auto& byte_size_offset_buffer_host = *byte_size_offset_buffer_host_.back();

  input_ptr_buffer_host.reserve(pending_copy_kernel_input_buffer_counts_);
  byte_size_buffer_host.reserve(pending_copy_kernel_input_buffer_counts_);
  byte_size_offset_buffer_host.reserve(
      pending_copy_kernel_input_buffer_counts_);

  // placeholder for output parameters
  auto kernel_buffer_memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t kernel_buffer_memory_id = 0;
  size_t buffer_byte_size = 0;

  size_t byte_size_offset = 0;
  for (const auto& response_input : pending_copy_kernel_inputs_) {
    const auto& input = response_input.second;

    for (size_t buffer_idx = 0; buffer_idx < input->DataBufferCount();
         ++buffer_idx) {
      input_ptr_buffer_host.emplace_back();
      RETURN_IF_ERROR(input->DataBuffer(
          buffer_idx, (const void**)(&input_ptr_buffer_host.back()),
          &buffer_byte_size, &kernel_buffer_memory_type,
          &kernel_buffer_memory_id));

      byte_size_offset_buffer_host.emplace_back(byte_size_offset);
      byte_size_buffer_host.emplace_back(buffer_byte_size);
      byte_size_offset += buffer_byte_size;
    }
  }

  in_use_memories_.emplace_back(new AllocatedMemory(
      pending_copy_kernel_input_buffer_counts_ * sizeof(int8_t*),
      tensor_memory_type, tensor_memory_type_id));
  char* input_ptr_buffer = in_use_memories_.back()->MutableBuffer(
      &kernel_buffer_memory_type, &kernel_buffer_memory_id);
  if ((kernel_buffer_memory_type != tensor_memory_type) ||
      (kernel_buffer_memory_id != tensor_memory_type_id)) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to obtain memory buffer for copy kernel input");
  }
  in_use_memories_.emplace_back(new AllocatedMemory(
      pending_copy_kernel_input_buffer_counts_ * sizeof(size_t),
      tensor_memory_type, tensor_memory_type_id));
  char* byte_size_buffer = in_use_memories_.back()->MutableBuffer(
      &kernel_buffer_memory_type, &kernel_buffer_memory_id);
  if ((kernel_buffer_memory_type != tensor_memory_type) ||
      (kernel_buffer_memory_id != tensor_memory_type_id)) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to obtain memory buffer for copy kernel input");
  }
  in_use_memories_.emplace_back(new AllocatedMemory(
      pending_copy_kernel_input_buffer_counts_ * sizeof(size_t),
      tensor_memory_type, tensor_memory_type_id));
  char* byte_size_offset_buffer = in_use_memories_.back()->MutableBuffer(
      &kernel_buffer_memory_type, &kernel_buffer_memory_id);
  if ((kernel_buffer_memory_type != tensor_memory_type) ||
      (kernel_buffer_memory_id != tensor_memory_type_id)) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to obtain memory buffer for copy kernel input");
  }

  cudaMemcpyAsync(
      input_ptr_buffer, input_ptr_buffer_host.data(),
      pending_copy_kernel_input_buffer_counts_ * sizeof(int8_t*),
      cudaMemcpyDefault, stream_);
  cudaMemcpyAsync(
      byte_size_buffer, byte_size_buffer_host.data(),
      pending_copy_kernel_input_buffer_counts_ * sizeof(size_t),
      cudaMemcpyDefault, stream_);
  cudaMemcpyAsync(
      byte_size_offset_buffer, byte_size_offset_buffer_host.data(),
      pending_copy_kernel_input_buffer_counts_ * sizeof(size_t),
      cudaMemcpyDefault, stream_);
  if (buffer_ready_event_ != nullptr) {
    cudaEventSynchronize(buffer_ready_event_);
    buffer_ready_event_ = nullptr;
  }
  RETURN_IF_CUDA_ERR(
      RunGatherKernel(
          (const int8_t**)input_ptr_buffer, (const size_t*)byte_size_buffer,
          (const size_t*)byte_size_offset_buffer,
          (int8_t*)tensor_buffer + pending_copy_kernel_buffer_offset_,
          pending_copy_kernel_input_buffer_counts_, stream_),
      std::string("Failed to launch gather kernel"));
  return Status::Success;
#else
  return Status(
      Status::Code::UNSUPPORTED,
      "Copy kernel can not be launched with TRITON_ENABLE_GPU=OFF");
#endif  // TRITON_ENABLE_GPU
}

}}  // namespace nvidia::inferenceserver
