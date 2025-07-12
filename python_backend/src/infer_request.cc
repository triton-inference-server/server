// Copyright 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "infer_request.h"

#include <boost/interprocess/sync/scoped_lock.hpp>

#include "gpu_buffers.h"
#include "pb_utils.h"
#include "scoped_defer.h"
#ifdef TRITON_PB_STUB
#include "pb_stub.h"
#endif

namespace triton { namespace backend { namespace python {

InferRequest::InferRequest(
    const std::string& request_id, const CorrelationId& correlation_id,
    const std::vector<std::shared_ptr<PbTensor>>& inputs,
    const std::set<std::string>& requested_output_names,
    const std::string& model_name, const int64_t model_version,
    const std::string& parameters, const uint32_t flags, const uint64_t timeout,
    const intptr_t response_factory_address, const intptr_t request_address,
    const PreferredMemory& preferred_memory, const InferenceTrace& trace)
    : request_id_(request_id), correlation_id_(correlation_id), inputs_(inputs),
      requested_output_names_(requested_output_names), model_name_(model_name),
      model_version_(model_version), parameters_(parameters), flags_(flags),
      timeout_(timeout), response_factory_address_(response_factory_address),
      request_address_(request_address), preferred_memory_(preferred_memory),
      trace_(trace), request_release_flags_(TRITONSERVER_REQUEST_RELEASE_ALL)
{
  for (auto& input : inputs) {
    if (!input) {
      throw PythonBackendException(
          "Input tensor for request with id '" + request_id +
          "' and model name '" + model_name + "' should not be empty.");
    }
  }

  for (auto& requested_output_name : requested_output_names) {
    if (requested_output_name == "") {
      throw PythonBackendException(
          "Requested output name for request with id '" + request_id +
          "' and model name '" + model_name + "' should not be empty.");
    }
  }

#ifdef TRITON_PB_STUB
  pb_cancel_ =
      std::make_shared<PbCancel>(response_factory_address_, request_address_);
  response_sender_ = std::make_shared<ResponseSender>(
      request_address_, response_factory_address_, nullptr /* is_decoupled */,
      RequestedOutputNames(), Stub::GetOrCreateInstance()->SharedMemory(),
      pb_cancel_);
#endif
}

const std::vector<std::shared_ptr<PbTensor>>&
InferRequest::Inputs()
{
  return inputs_;
}

const std::string&
InferRequest::Parameters()
{
  return parameters_;
}

const std::string&
InferRequest::RequestId()
{
  return request_id_;
}

CorrelationId&
InferRequest::GetCorrelationId()
{
  return correlation_id_;
}

const std::set<std::string>&
InferRequest::RequestedOutputNames()
{
  return requested_output_names_;
}

const std::string&
InferRequest::ModelName()
{
  return model_name_;
}

int64_t
InferRequest::ModelVersion()
{
  return model_version_;
}

uint32_t
InferRequest::Flags()
{
  return flags_;
}

intptr_t
InferRequest::RequestAddress()
{
  return request_address_;
}

void
InferRequest::SetFlags(uint32_t flags)
{
  flags_ = flags;
}

bi::managed_external_buffer::handle_t
InferRequest::ShmHandle()
{
  return shm_handle_;
}

uint64_t
InferRequest::Timeout()
{
  return timeout_;
}

void
InferRequest::SetIsDecoupled(const bool is_decoupled)
{
  is_decoupled_ = is_decoupled;
}

bool
InferRequest::IsDecoupled()
{
  return is_decoupled_;
}

PreferredMemory&
InferRequest::GetPreferredMemory()
{
  return preferred_memory_;
}

InferenceTrace&
InferRequest::GetTrace()
{
  return trace_;
}

uint32_t
InferRequest::ReleaseFlags()
{
  request_release_flags_ = infer_request_shm_ptr_->request_release_flags;
  return request_release_flags_;
}

void
InferRequest::SetReleaseFlags(const uint32_t& flags)
{
  request_release_flags_ = flags;
  infer_request_shm_ptr_->request_release_flags = request_release_flags_;
}

void
InferRequest::SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool)
{
  AllocatedSharedMemory<char> infer_request_shm = shm_pool->Construct<char>(
      sizeof(InferRequestShm) +
      (RequestedOutputNames().size() *
       sizeof(bi::managed_external_buffer::handle_t)) +
      (Inputs().size() * sizeof(bi::managed_external_buffer::handle_t)));

  infer_request_shm_ptr_ =
      reinterpret_cast<InferRequestShm*>(infer_request_shm.data_.get());
  infer_request_shm_ptr_->input_count = Inputs().size();
  infer_request_shm_ptr_->model_version = model_version_;
  infer_request_shm_ptr_->requested_output_count =
      RequestedOutputNames().size();
  infer_request_shm_ptr_->flags = Flags();
  infer_request_shm_ptr_->address = request_address_;
  infer_request_shm_ptr_->response_factory_address = response_factory_address_;
  infer_request_shm_ptr_->is_decoupled = is_decoupled_;
  infer_request_shm_ptr_->timeout = timeout_;
  infer_request_shm_ptr_->preferred_memory = preferred_memory_;
  infer_request_shm_ptr_->request_release_flags = request_release_flags_;

  output_names_handle_shm_ptr_ =
      reinterpret_cast<bi::managed_external_buffer::handle_t*>(
          reinterpret_cast<char*>(infer_request_shm_ptr_) +
          sizeof(InferRequestShm));

  // [FIXME] This could also be a part of the single allocated memory for this
  // object.
  size_t i = 0;
  std::vector<std::unique_ptr<PbString>> requested_output_names_shm;
  for (auto& requested_output_name : requested_output_names_) {
    std::unique_ptr<PbString> requested_output_name_shm =
        PbString::Create(shm_pool, requested_output_name);
    output_names_handle_shm_ptr_[i] = requested_output_name_shm->ShmHandle();
    requested_output_names_shm.emplace_back(
        std::move(requested_output_name_shm));
    i++;
  }

  input_tensors_handle_ptr_ =
      reinterpret_cast<bi::managed_external_buffer::handle_t*>(
          reinterpret_cast<char*>(output_names_handle_shm_ptr_) +
          sizeof(bi::managed_external_buffer::handle_t) *
              RequestedOutputNames().size());
  i = 0;
  for (auto& input : Inputs()) {
    input_tensors_handle_ptr_[i] = input->ShmHandle();
    i++;
  }

  correlation_id_.SaveToSharedMemory(shm_pool);
  infer_request_shm_ptr_->correlation_id_shm_handle =
      correlation_id_.ShmHandle();

  std::unique_ptr<PbString> model_name_shm =
      PbString::Create(shm_pool, ModelName());
  infer_request_shm_ptr_->model_name_shm_handle = model_name_shm->ShmHandle();

  std::unique_ptr<PbString> request_id_shm =
      PbString::Create(shm_pool, RequestId());
  infer_request_shm_ptr_->request_id_shm_handle = request_id_shm->ShmHandle();

  std::unique_ptr<PbString> parameters_shm =
      PbString::Create(shm_pool, Parameters());
  infer_request_shm_ptr_->parameters_shm_handle = parameters_shm->ShmHandle();

  trace_.SaveToSharedMemory(shm_pool);
  infer_request_shm_ptr_->trace_shm_handle = trace_.ShmHandle();

  // Save the references to shared memory.
  infer_request_shm_ = std::move(infer_request_shm);
  request_id_shm_ = std::move(request_id_shm);
  model_name_shm_ = std::move(model_name_shm);
  parameters_shm_ = std::move(parameters_shm);
  shm_handle_ = infer_request_shm_.handle_;
  requested_output_names_shm_ = std::move(requested_output_names_shm);
}

std::unique_ptr<InferRequest>
InferRequest::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t request_handle, bool open_cuda_handle,
    bool const* is_model_decoupled)
{
  AllocatedSharedMemory<char> infer_request_shm =
      shm_pool->Load<char>(request_handle);
  InferRequestShm* infer_request_shm_ptr =
      reinterpret_cast<InferRequestShm*>(infer_request_shm.data_.get());

  std::vector<std::unique_ptr<PbString>> requested_output_names_shm;
  uint32_t requested_output_count =
      infer_request_shm_ptr->requested_output_count;

  bi::managed_external_buffer::handle_t* output_names_handle_shm_ptr =
      reinterpret_cast<bi::managed_external_buffer::handle_t*>(
          (reinterpret_cast<char*>(infer_request_shm_ptr) +
           sizeof(InferRequestShm)));

  for (size_t output_idx = 0; output_idx < requested_output_count;
       ++output_idx) {
    std::unique_ptr<PbString> pb_string = PbString::LoadFromSharedMemory(
        shm_pool, output_names_handle_shm_ptr[output_idx]);
    requested_output_names_shm.emplace_back(std::move(pb_string));
  }

  bi::managed_external_buffer::handle_t* input_names_handle_shm_ptr =
      reinterpret_cast<bi::managed_external_buffer::handle_t*>(
          (reinterpret_cast<char*>(infer_request_shm_ptr) +
           sizeof(InferRequestShm) +
           (infer_request_shm_ptr->requested_output_count *
            sizeof(bi::managed_external_buffer::handle_t))));

  std::vector<std::shared_ptr<PbTensor>> input_tensors;
  for (size_t input_idx = 0; input_idx < infer_request_shm_ptr->input_count;
       ++input_idx) {
    std::shared_ptr<PbTensor> input_tensor = PbTensor::LoadFromSharedMemory(
        shm_pool, input_names_handle_shm_ptr[input_idx], open_cuda_handle);
    input_tensors.emplace_back(std::move(input_tensor));
  }

  std::unique_ptr<CorrelationId> correlation_id_shm =
      CorrelationId::LoadFromSharedMemory(
          shm_pool, infer_request_shm_ptr->correlation_id_shm_handle);

  std::unique_ptr<InferenceTrace> infer_trace_shm =
      InferenceTrace::LoadFromSharedMemory(
          shm_pool, infer_request_shm_ptr->trace_shm_handle);

  std::unique_ptr<PbString> model_name_shm = PbString::LoadFromSharedMemory(
      shm_pool, infer_request_shm_ptr->model_name_shm_handle);
  std::unique_ptr<PbString> request_id_shm = PbString::LoadFromSharedMemory(
      shm_pool, infer_request_shm_ptr->request_id_shm_handle);
  std::unique_ptr<PbString> parameters_shm = PbString::LoadFromSharedMemory(
      shm_pool, infer_request_shm_ptr->parameters_shm_handle);

  return std::unique_ptr<InferRequest>(new InferRequest(
      infer_request_shm, request_id_shm, correlation_id_shm,
      requested_output_names_shm, model_name_shm, input_tensors, parameters_shm,
      infer_trace_shm, is_model_decoupled));
}

InferRequest::InferRequest(
    AllocatedSharedMemory<char>& infer_request_shm,
    std::unique_ptr<PbString>& request_id_shm,
    std::unique_ptr<CorrelationId>& correlation_id_shm,
    std::vector<std::unique_ptr<PbString>>& requested_output_names_shm,
    std::unique_ptr<PbString>& model_name_shm,
    std::vector<std::shared_ptr<PbTensor>>& input_tensors,
    std::unique_ptr<PbString>& parameters_shm,
    std::unique_ptr<InferenceTrace>& infer_trace_shm,
    bool const* is_model_decoupled)
    : infer_request_shm_(std::move(infer_request_shm)),
      request_id_shm_(std::move(request_id_shm)),
      requested_output_names_shm_(std::move(requested_output_names_shm)),
      model_name_shm_(std::move(model_name_shm)),
      parameters_shm_(std::move(parameters_shm))
{
  infer_request_shm_ptr_ =
      reinterpret_cast<InferRequestShm*>(infer_request_shm_.data_.get());
  output_names_handle_shm_ptr_ =
      reinterpret_cast<bi::managed_external_buffer::handle_t*>(
          reinterpret_cast<char*>(infer_request_shm_ptr_) +
          sizeof(InferRequestShm));
  input_tensors_handle_ptr_ =
      reinterpret_cast<bi::managed_external_buffer::handle_t*>(
          reinterpret_cast<char*>(infer_request_shm_ptr_) +
          sizeof(InferRequestShm) +
          sizeof(bi::managed_external_buffer::handle_t) *
              infer_request_shm_ptr_->requested_output_count);
  inputs_ = std::move(input_tensors);

  std::set<std::string> requested_output_names;
  for (size_t output_idx = 0;
       output_idx < infer_request_shm_ptr_->requested_output_count;
       ++output_idx) {
    auto& pb_string = requested_output_names_shm_[output_idx];
    requested_output_names.emplace(pb_string->String());
  }

  correlation_id_ = CorrelationId(correlation_id_shm);
  request_id_ = request_id_shm_->String();
  parameters_ = parameters_shm_->String();
  requested_output_names_ = std::move(requested_output_names);
  model_name_ = model_name_shm_->String();
  flags_ = infer_request_shm_ptr_->flags;
  model_version_ = infer_request_shm_ptr_->model_version;
  request_address_ = infer_request_shm_ptr_->address;
  response_factory_address_ = infer_request_shm_ptr_->response_factory_address;
  is_decoupled_ = infer_request_shm_ptr_->is_decoupled;
  timeout_ = infer_request_shm_ptr_->timeout;
  preferred_memory_ = infer_request_shm_ptr_->preferred_memory;
  trace_ = InferenceTrace(infer_trace_shm);
  request_release_flags_ = infer_request_shm_ptr_->request_release_flags;

#ifdef TRITON_PB_STUB
  pb_cancel_ =
      std::make_shared<PbCancel>(response_factory_address_, request_address_);
  response_sender_ = std::make_shared<ResponseSender>(
      request_address_, response_factory_address_, is_model_decoupled,
      RequestedOutputNames(), Stub::GetOrCreateInstance()->SharedMemory(),
      pb_cancel_);
#endif
}

#ifdef TRITON_PB_STUB
bool
InferRequest::IsCancelled()
{
  return pb_cancel_->IsCancelled();
}

std::shared_ptr<ResponseSender>
InferRequest::GetResponseSender()
{
  return response_sender_;
}

std::shared_ptr<InferResponse>
InferRequest::Exec(const bool is_decoupled)
{
  // Release the GIL. This avoids a potential deadlock situation in the parent
  // process, where every thread in the thread pool is indirectly waiting for a
  // function in the stub process that acquires the GIL. Meanwhile, the current
  // thread, which holds the GIL, is also waiting for the parent side to have
  // the next available thread to pick up the job during resource contention.
  py::gil_scoped_release release;

  // BLS should not be used in "initialize" or "finalize" function.
  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
  if (!stub->IsInitialized() || stub->IsFinalizing()) {
    throw PythonBackendException(
        "BLS is only supported during the 'execute' function.");
  }

  ResponseBatch* response_batch = nullptr;
  bool responses_is_set = false;
  std::unique_ptr<SharedMemoryManager>& shm_pool = stub->SharedMemory();
  bi::managed_external_buffer::handle_t* response_handle = nullptr;

  PythonBackendException pb_exception(std::string{});
  std::unique_ptr<IPCMessage> ipc_message;

  AllocatedSharedMemory<char> request_batch;
  ScopedDefer data_load_complete([&ipc_message] {
    bi::scoped_lock<bi::interprocess_mutex> lock{
        *(ipc_message->ResponseMutex())};
    ipc_message->ResponseCondition()->notify_all();
  });

  try {
    ipc_message = IPCMessage::Create(shm_pool, true /* inline_response */);
    bool has_exception = false;
    PythonBackendException pb_exception(std::string{});

    if (is_decoupled) {
      ipc_message->Command() =
          PYTHONSTUB_CommandType::PYTHONSTUB_InferStreamExecRequest;
    } else {
      ipc_message->Command() =
          PYTHONSTUB_CommandType::PYTHONSTUB_InferExecRequest;
    }

    request_batch = shm_pool->Construct<char>(
        sizeof(RequestBatch) + sizeof(bi::managed_external_buffer::handle_t));

    RequestBatch* request_batch_shm_ptr =
        reinterpret_cast<RequestBatch*>(request_batch.data_.get());
    request_batch_shm_ptr->batch_size = 1;
    ipc_message->Args() = request_batch.handle_;

    bi::managed_external_buffer::handle_t* requests_shm =
        reinterpret_cast<bi::managed_external_buffer::handle_t*>(
            request_batch.data_.get() + sizeof(RequestBatch));
    request_batch_shm_ptr->batch_size = 1;

    bool has_gpu_tensor = false;
    size_t i = 0;
    for (auto& input_tensor : inputs_) {
      input_tensor->SaveToSharedMemory(shm_pool, false /* copy_gpu */);
      if (!input_tensor->IsCPU()) {
        has_gpu_tensor = true;
      }
      ++i;
    }

    SaveToSharedMemory(shm_pool);

    // Save the shared memory offset of the request.
    *requests_shm = ShmHandle();

    // Send the BLS request to the parent process and wait for the response.
    {
      bi::scoped_lock<bi::interprocess_mutex> lock{
          *(ipc_message->ResponseMutex())};
      stub->SendIPCUtilsMessage(ipc_message);
      ipc_message->ResponseCondition()->wait(lock);
    }

    // Additional round trip required for asking the stub process
    // to fill in the GPU tensor buffers
    if (has_gpu_tensor) {
      AllocatedSharedMemory<GPUBuffersShm> gpu_buffers_shm =
          shm_pool->Load<GPUBuffersShm>(
              request_batch_shm_ptr->gpu_buffers_handle);
      AllocatedSharedMemory<bi::managed_external_buffer::handle_t>
          gpu_buffers_handle =
              shm_pool->Load<bi::managed_external_buffer::handle_t>(
                  gpu_buffers_shm.data_->buffers);
      try {
        if (!gpu_buffers_shm.data_->success) {
          std::unique_ptr<PbString> error = PbString::LoadFromSharedMemory(
              shm_pool, gpu_buffers_shm.data_->error);
          throw PythonBackendException(error->String());
        }
#ifdef TRITON_ENABLE_GPU
        size_t i = 0;
        for (auto& input_tensor : this->Inputs()) {
          if (!input_tensor->IsCPU()) {
            std::unique_ptr<PbMemory> dst_buffer =
                PbMemory::LoadFromSharedMemory(
                    shm_pool, (gpu_buffers_handle.data_.get())[i],
                    true /* open cuda handle */);
            PbMemory::CopyBuffer(dst_buffer, input_tensor->Memory());
            ++i;
          }
        }
#endif  // TRITON_ENABLE_GPU
      }
      catch (const PythonBackendException& exception) {
        // We need to catch the exception here. Otherwise, we will not notify
        // the main process and it will wait for the response forever.
        pb_exception = exception;
        has_exception = true;
      }

      {
        bi::scoped_lock<bi::interprocess_mutex> lock{
            *(ipc_message->ResponseMutex())};
        ipc_message->ResponseCondition()->notify_all();
        ipc_message->ResponseCondition()->wait(lock);
      }
    }

    // The exception will be thrown after the message was sent to the main
    // process.
    if (has_exception) {
      throw pb_exception;
    }

    // Get the response for the current message.
    std::unique_ptr<IPCMessage> bls_response = IPCMessage::LoadFromSharedMemory(
        shm_pool, ipc_message->ResponseHandle());

    AllocatedSharedMemory<char> response_batch_shm =
        shm_pool->Load<char>(bls_response->Args());
    response_batch =
        reinterpret_cast<ResponseBatch*>(response_batch_shm.data_.get());
    response_handle = reinterpret_cast<bi::managed_external_buffer::handle_t*>(
        response_batch_shm.data_.get() + sizeof(ResponseBatch));

    responses_is_set = true;
    if (response_batch->has_error) {
      if (response_batch->is_error_set) {
        std::unique_ptr<PbString> pb_string =
            PbString::LoadFromSharedMemory(shm_pool, response_batch->error);
        auto error_response = std::make_unique<InferResponse>(
            std::vector<std::shared_ptr<PbTensor>>{},
            std::make_shared<PbError>(pb_string->String()));

        return error_response;
      } else {
        auto error_response = std::make_unique<InferResponse>(
            std::vector<std::shared_ptr<PbTensor>>{},
            std::make_shared<PbError>(
                "An error occurred while performing BLS request."));

        return error_response;
      }
    }
  }
  catch (const PythonBackendException& pb_exception) {
    auto error_response = std::make_unique<InferResponse>(
        std::vector<std::shared_ptr<PbTensor>>{},
        std::make_shared<PbError>(pb_exception.what()));

    return error_response;
  }

  if (responses_is_set) {
    auto& memory_manager_message_queue = stub->MemoryManagerQueue();
    std::unique_ptr<InferResponse> return_response =
        InferResponse::LoadFromSharedMemory(
            shm_pool, *response_handle, true /* open cuda handle */);

    for (auto& output_tensor : return_response->OutputTensors()) {
      if (!output_tensor->IsCPU()) {
        uint64_t memory_release_id = output_tensor->Memory()->MemoryReleaseId();
        output_tensor->Memory()->SetMemoryReleaseCallback(
            [&memory_manager_message_queue, memory_release_id, &shm_pool]() {
              memory_manager_message_queue->Push(memory_release_id);
            });
      }
    }

    return return_response;
  } else {
    auto error_response = std::make_unique<InferResponse>(
        std::vector<std::shared_ptr<PbTensor>>{},
        std::make_shared<PbError>(
            "An error occurred while performing BLS request."));

    return error_response;
  }
}

#endif

}}}  // namespace triton::backend::python
