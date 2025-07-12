// Copyright 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "response_sender.h"

#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>

#include "pb_stub.h"
#include "pb_stub_utils.h"
#include "scoped_defer.h"

namespace triton { namespace backend { namespace python {

void
CheckResponseSenderArguments(
    const std::shared_ptr<InferResponse>& response, const uint32_t flags)
{
  // Check the correctness of the provided flags.
  if (flags != TRITONSERVER_RESPONSE_COMPLETE_FINAL && flags != 0) {
    throw PythonBackendException(
        "Unable to send response. Unsupported flag provided.");
  }

  if (flags == 0 && response == nullptr) {
    throw PythonBackendException(
        "Inference Response object must be provided when the response flags is "
        "set to zero.");
  }
}

ResponseSender::ResponseSender(
    intptr_t request_address, intptr_t response_factory_address,
    bool const* is_decoupled,
    const std::set<std::string>& requested_output_names,
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    const std::shared_ptr<PbCancel>& pb_cancel)
    : request_address_(request_address),
      response_factory_address_(response_factory_address),
      is_decoupled_(is_decoupled),
      requested_output_names_(requested_output_names), shm_pool_(shm_pool),
      pb_cancel_(pb_cancel), closed_(false), number_of_response_sent_(0),
      response_factory_deleted_(false)
{
}

ResponseSender::~ResponseSender()
{
  DeleteResponseFactory();
}

void
ResponseSender::UpdateStateAndCounters(
    InferResponse* response, const uint32_t flags)
{
  if (is_decoupled_ == nullptr) {
    // TODO: Can a model access the response sender on a BLS infer request?
    throw PythonBackendException(
        "Unable to send response. Response sender has no reference to the "
        "decoupled state of the model.");
  }
  bool is_decoupled = *is_decoupled_;

  std::lock_guard<std::mutex> lk(mu_);

  if (!is_decoupled) {
    if (response != nullptr && number_of_response_sent_ > 0) {
      throw PythonBackendException(
          "Unable to send response. Non-decoupled model cannot send more than "
          "one response.");
    }
    if (response == nullptr && flags == TRITONSERVER_RESPONSE_COMPLETE_FINAL &&
        number_of_response_sent_ == 0) {
      throw PythonBackendException(
          "Unable to send response. Non-decoupled model cannot send complete "
          "final before sending a response.");
    }
  }

  if (closed_) {
    throw PythonBackendException(
        "Unable to send response. Response sender has been closed.");
  }

  if (flags == TRITONSERVER_RESPONSE_COMPLETE_FINAL) {
    response_factory_deleted_.exchange(true);
    closed_ = true;
  }
  number_of_response_sent_++;
}

void
ResponseSender::Send(
    std::shared_ptr<InferResponse> infer_response, const uint32_t flags)
{
  // Release the GIL. This avoids a potential deadlock situation in the parent
  // process, where every thread in the thread pool is indirectly waiting for a
  // function in the stub process that acquires the GIL. Meanwhile, the current
  // thread, which holds the GIL, is also waiting for the parent side to have
  // the next available thread to pick up the job during resource contention.
  py::gil_scoped_release release;

  CheckResponseSenderArguments(infer_response, flags);
  UpdateStateAndCounters(infer_response.get(), flags);
  if (infer_response) {
    infer_response->PruneOutputTensors(requested_output_names_);
  }

  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();

  AllocatedSharedMemory<ResponseSendMessage> response_send_message =
      shm_pool_->Construct<ResponseSendMessage>(
          1 /* count */, true /* aligned */);

  if (infer_response) {
    infer_response->SaveToSharedMemory(shm_pool_, false /* copy_gpu */);
  }

  ResponseSendMessage* send_message_payload = response_send_message.data_.get();
  new (&(send_message_payload->mu)) bi::interprocess_mutex;
  new (&(send_message_payload->cv)) bi::interprocess_condition;

  send_message_payload->is_stub_turn = false;
  send_message_payload->request_address = request_address_;
  send_message_payload->response_factory_address = response_factory_address_;

  if (infer_response) {
    send_message_payload->response = infer_response->ShmHandle();
  } else {
    send_message_payload->response = 0;
  }

  send_message_payload->has_error = false;
  send_message_payload->is_error_set = false;
  send_message_payload->flags = flags;

  std::unique_ptr<IPCMessage> ipc_message =
      IPCMessage::Create(shm_pool_, false /* inline_response */);

  ipc_message->Command() = PYTHONSTUB_ResponseSend;
  ipc_message->Args() = response_send_message.handle_;

  ScopedDefer _([send_message_payload] {
    {
      bi::scoped_lock<bi::interprocess_mutex> guard{send_message_payload->mu};
      send_message_payload->is_stub_turn = false;
      send_message_payload->cv.notify_all();
    }
  });

  {
    bi::scoped_lock<bi::interprocess_mutex> guard{send_message_payload->mu};
    // The server will destruct the response factory if the final flag is set.
    if (flags == TRITONSERVER_RESPONSE_COMPLETE_FINAL) {
      response_factory_deleted_.exchange(true);
    }
    stub->SendIPCUtilsMessage(ipc_message);
    while (!send_message_payload->is_stub_turn) {
      send_message_payload->cv.wait(guard);
    }
  }

  bool has_gpu_output = false;
  std::vector<std::shared_ptr<PbTensor>> gpu_tensors;
  if (infer_response) {
    for (auto& tensor : infer_response->OutputTensors()) {
      if (!tensor->IsCPU()) {
        has_gpu_output = true;
        gpu_tensors.push_back(tensor);
      }
    }
  }

  if (has_gpu_output) {
    ScopedDefer _([send_message_payload] {
      bi::scoped_lock<bi::interprocess_mutex> guard{send_message_payload->mu};
      send_message_payload->is_stub_turn = false;
      send_message_payload->cv.notify_one();
      while (!send_message_payload->is_stub_turn) {
        // Wait for the stub process to send the response and populate error
        // message if any.
        send_message_payload->cv.wait(guard);
      }
    });

    AllocatedSharedMemory<GPUBuffersShm> gpu_buffers_handle =
        shm_pool_->Load<GPUBuffersShm>(
            send_message_payload->gpu_buffers_handle);
    if (!gpu_buffers_handle.data_->success) {
      std::unique_ptr<PbString> error = PbString::LoadFromSharedMemory(
          shm_pool_, gpu_buffers_handle.data_->error);
      throw PythonBackendException(
          "Failed to load GPU buffers: " + error->String());
    }

    AllocatedSharedMemory<bi::managed_external_buffer::handle_t>
        gpu_buffers_handle_shm =
            shm_pool_->Load<bi::managed_external_buffer::handle_t>(
                gpu_buffers_handle.data_->buffers);
    uint64_t gpu_buffer_count = gpu_buffers_handle.data_->buffer_count;
    if (gpu_tensors.size() != gpu_buffer_count) {
      throw PythonBackendException(
          std::string(
              "GPU buffers size does not match the provided buffers: ") +
          std::to_string(gpu_tensors.size()) +
          " != " + std::to_string(gpu_buffer_count));
    }

    std::vector<std::unique_ptr<PbMemory>> dst_buffers;

    for (size_t i = 0; i < gpu_tensors.size(); i++) {
      std::unique_ptr<PbMemory> dst_buffer = PbMemory::LoadFromSharedMemory(
          shm_pool_, gpu_buffers_handle_shm.data_.get()[i],
          true /* open_cuda_handle */);
      dst_buffers.emplace_back(std::move(dst_buffer));
      std::shared_ptr<PbTensor>& src_buffer = gpu_tensors[i];
      PbMemory::CopyBuffer(dst_buffers[i], src_buffer->Memory());
    }
  }

  if (send_message_payload->has_error) {
    if (send_message_payload->is_error_set) {
      std::unique_ptr<PbString> error = PbString::LoadFromSharedMemory(
          shm_pool_, send_message_payload->error);
      throw PythonBackendException(error->String());
    } else {
      throw PythonBackendException(
          "An error occurred while sending a response.");
    }
  }
}

bool
ResponseSender::IsCancelled()
{
  return pb_cancel_->IsCancelled();
}

bool
ResponseSender::IsClosed()
{
  std::lock_guard<std::mutex> lk(mu_);
  return closed_;
}

void
ResponseSender::Close()
{
  std::lock_guard<std::mutex> lk(mu_);
  closed_ = true;
  response_factory_deleted_.exchange(true);
}

void
ResponseSender::DeleteResponseFactory()
{
  bool already_deleted = response_factory_deleted_.exchange(true);
  if (!already_deleted) {
    std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
    stub->EnqueueCleanupId(
        reinterpret_cast<void*>(response_factory_address_),
        PYTHONSTUB_DecoupledResponseFactoryCleanup);
  }
}

}}}  // namespace triton::backend::python
