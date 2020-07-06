// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "src/core/infer_response.h"

#include "src/core/backend.h"
#include "src/core/logging.h"
#include "src/core/server.h"

namespace nvidia { namespace inferenceserver {

//
// InferenceResponseFactory
//
Status
InferenceResponseFactory::CreateResponse(
    std::unique_ptr<InferenceResponse>* response) const
{
  response->reset(new InferenceResponse(
      backend_, id_, allocator_, alloc_userp_, response_fn_, response_userp_,
      response_delegator_));

  return Status::Success;
}

Status
InferenceResponseFactory::SendFlags(const uint32_t flags) const
{
  void* userp = response_userp_;
  response_fn_(nullptr /* response */, flags, userp);
  return Status::Success;
}

//
// InferenceResponse
//
InferenceResponse::InferenceResponse(
    const std::shared_ptr<InferenceBackend>& backend, const std::string& id,
    const ResponseAllocator* allocator, void* alloc_userp,
    TRITONSERVER_InferenceResponseCompleteFn_t response_fn,
    void* response_userp,
    const std::function<void(std::unique_ptr<InferenceResponse>&&)>& delegator)
    : backend_(backend), id_(id), allocator_(allocator),
      alloc_userp_(alloc_userp), response_fn_(response_fn),
      response_userp_(response_userp), response_delegator_(delegator)
{
  // If the allocator has a start_fn then invoke it.
  TRITONSERVER_ResponseAllocatorStartFn_t start_fn = allocator_->StartFn();
  if (start_fn != nullptr) {
    LOG_TRITONSERVER_ERROR(
        start_fn(
            reinterpret_cast<TRITONSERVER_ResponseAllocator*>(
                const_cast<ResponseAllocator*>(allocator_)),
            alloc_userp_),
        "response allocation start failed");
  }
}

const std::string&
InferenceResponse::ModelName() const
{
  static const std::string unknown("<unknown>");
  return (backend_ == nullptr) ? unknown : backend_->Name();
}

int64_t
InferenceResponse::ActualModelVersion() const
{
  return (backend_ == nullptr) ? -1 : backend_->Version();
}

Status
InferenceResponse::AddOutput(
    const std::string& name, const DataType datatype,
    const std::vector<int64_t>& shape, InferenceResponse::Output** output)
{
  outputs_.emplace_back(name, datatype, shape, allocator_, alloc_userp_);

  LOG_VERBOSE(1) << "add response output: " << outputs_.back();

  if (backend_ != nullptr) {
    const ModelOutput* output_config;
    RETURN_IF_ERROR(backend_->GetOutput(name, &output_config));
    if (output_config->has_reshape()) {
      const bool has_batch_dim = (backend_->Config().max_batch_size() > 0);
      outputs_.back().Reshape(has_batch_dim, output_config);
    }
  }

  if (output != nullptr) {
    *output = std::addressof(outputs_.back());
  }

  return Status::Success;
}

Status
InferenceResponse::AddOutput(
    const std::string& name, const DataType datatype,
    std::vector<int64_t>&& shape, InferenceResponse::Output** output)
{
  outputs_.emplace_back(
      name, datatype, std::move(shape), allocator_, alloc_userp_);

  LOG_VERBOSE(1) << "add response output: " << outputs_.back();

  if (backend_ != nullptr) {
    const ModelOutput* output_config;
    RETURN_IF_ERROR(backend_->GetOutput(name, &output_config));
    if (output_config->has_reshape()) {
      const bool has_batch_dim = (backend_->Config().max_batch_size() > 0);
      outputs_.back().Reshape(has_batch_dim, output_config);
    }
  }

  if (output != nullptr) {
    *output = std::addressof(outputs_.back());
  }

  return Status::Success;
}

Status
InferenceResponse::ClassificationLabel(
    const InferenceResponse::Output& output, const uint32_t class_index,
    const char** label) const
{
  const auto& label_provider = backend_->GetLabelProvider();
  const std::string& l = label_provider->GetLabel(output.Name(), class_index);
  if (l.empty()) {
    *label = nullptr;
  } else {
    *label = l.c_str();
  }

  return Status::Success;
}

Status
InferenceResponse::Send(
    std::unique_ptr<InferenceResponse>&& response, const uint32_t flags)
{
  if (response->response_delegator_ != nullptr) {
    auto ldelegator = std::move(response->response_delegator_);
    ldelegator(std::move(response));
    return Status::Success;
  }
  void* userp = response->response_userp_;
  response->response_fn_(
      reinterpret_cast<TRITONSERVER_InferenceResponse*>(response.release()),
      flags, userp);
  return Status::Success;
}

Status
InferenceResponse::SendWithStatus(
    std::unique_ptr<InferenceResponse>&& response, const uint32_t flags,
    const Status& status)
{
  response->status_ = status;
  return InferenceResponse::Send(std::move(response), flags);
}

//
// InferenceResponse::Output
//
InferenceResponse::Output::~Output()
{
  Status status = ReleaseDataBuffer();
  if (!status.IsOk()) {
    LOG_ERROR << "failed to release buffer for output '" << name_
              << "': " << status.AsString();
  }
}

void
InferenceResponse::Output::Reshape(
    const bool has_batch_dim, const ModelOutput* output_config)
{
  std::deque<int64_t> variable_size_values;

  const int64_t batch_dim =
      (has_batch_dim && (shape_.size() > 0)) ? shape_[0] : -1;
  const size_t batch_dim_offset = (has_batch_dim) ? 1 : 0;

  const auto& from_shape = output_config->reshape().shape();
  const auto& to_shape = output_config->dims();
  for (int64_t idx = 0; idx < from_shape.size(); idx++) {
    if (from_shape[idx] == -1) {
      variable_size_values.push_back(shape_[idx + batch_dim_offset]);
    }
  }

  shape_.clear();
  if (batch_dim >= 0) {
    shape_.push_back(batch_dim);
  }

  for (const auto& dim : to_shape) {
    if (dim == -1) {
      shape_.push_back(variable_size_values.front());
      variable_size_values.pop_front();
    } else {
      shape_.push_back(dim);
    }
  }
}

Status
InferenceResponse::Output::DataBuffer(
    const void** buffer, size_t* buffer_byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id,
    void** userp) const
{
  *buffer = allocated_buffer_;
  *buffer_byte_size = allocated_buffer_byte_size_;
  *memory_type = allocated_memory_type_;
  *memory_type_id = allocated_memory_type_id_;
  *userp = allocated_userp_;
  return Status::Success;
}

Status
InferenceResponse::Output::AllocateDataBuffer(
    void** buffer, size_t buffer_byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
{
  if (allocated_buffer_ != nullptr) {
    return Status(
        Status::Code::ALREADY_EXISTS,
        "allocated buffer for output '" + name_ + "' already exists");
  }

  TRITONSERVER_MemoryType actual_memory_type = *memory_type;
  int64_t actual_memory_type_id = *memory_type_id;
  void* alloc_buffer_userp = nullptr;

  RETURN_IF_TRITONSERVER_ERROR(allocator_->AllocFn()(
      reinterpret_cast<TRITONSERVER_ResponseAllocator*>(
          const_cast<ResponseAllocator*>(allocator_)),
      name_.c_str(), buffer_byte_size, *memory_type, *memory_type_id,
      alloc_userp_, buffer, &alloc_buffer_userp, &actual_memory_type,
      &actual_memory_type_id));

  allocated_buffer_ = *buffer;
  allocated_buffer_byte_size_ = buffer_byte_size;
  allocated_memory_type_ = actual_memory_type;
  allocated_memory_type_id_ = actual_memory_type_id;
  allocated_userp_ = alloc_buffer_userp;

  *memory_type = actual_memory_type;
  *memory_type_id = actual_memory_type_id;

  return Status::Success;
}

Status
InferenceResponse::Output::ReleaseDataBuffer()
{
  TRITONSERVER_Error* err = nullptr;

  if (allocated_buffer_ != nullptr) {
    err = allocator_->ReleaseFn()(
        reinterpret_cast<TRITONSERVER_ResponseAllocator*>(
            const_cast<ResponseAllocator*>(allocator_)),
        allocated_buffer_, allocated_userp_, allocated_buffer_byte_size_,
        allocated_memory_type_, allocated_memory_type_id_);
  }

  allocated_buffer_ = nullptr;
  allocated_buffer_byte_size_ = 0;
  allocated_memory_type_ = TRITONSERVER_MEMORY_CPU;
  allocated_memory_type_id_ = 0;
  allocated_userp_ = nullptr;

  RETURN_IF_TRITONSERVER_ERROR(err);

  return Status::Success;
}

std::ostream&
operator<<(std::ostream& out, const InferenceResponse& response)
{
  out << "[0x" << std::addressof(response) << "] "
      << "response id: " << response.Id() << ", model: " << response.ModelName()
      << ", actual version: " << response.ActualModelVersion() << std::endl;

  out << "status:" << response.ResponseStatus().AsString() << std::endl;

  out << "outputs:" << std::endl;
  for (const auto& output : response.Outputs()) {
    out << "[0x" << std::addressof(output) << "] " << output << std::endl;
  }

  return out;
}

std::ostream&
operator<<(std::ostream& out, const InferenceResponse::Output& output)
{
  out << "output: " << output.Name()
      << ", type: " << DataTypeToProtocolString(output.DType())
      << ", shape: " << DimsListToString(output.Shape());
  return out;
}

}}  // namespace nvidia::inferenceserver
