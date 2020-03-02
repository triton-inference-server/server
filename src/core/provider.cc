// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "src/core/provider.h"

#include <deque>
#include <numeric>
#include "src/core/backend.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/model_config_utils.h"
#include "src/core/pinned_memory_manager.h"

#ifdef TRTIS_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRTIS_ENABLE_GPU

namespace nvidia { namespace inferenceserver {

//
// InferRequestProvider
//
Status
InferRequestProvider::Create(
    const std::shared_ptr<InferenceRequest>& irequest,
    std::shared_ptr<InferRequestProvider>* provider)
{
  provider->reset(new InferRequestProvider(irequest));

  for (const auto& pr : irequest->Inputs()) {
    if ((pr.second.BatchByteSize() != 0) &&
        pr.second.BatchByteSize() != pr.second.Data()->TotalByteSize()) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "unexpected size " +
              std::to_string(pr.second.Data()->TotalByteSize()) +
              " for input '" + pr.first + "', expecting " +
              std::to_string(pr.second.BatchByteSize()) + " for model '" +
              irequest->ModelName() + "'");
    }
  }

  return Status::Success;
}

const InferRequestProvider::InputOverrideMapVec&
InferRequestProvider::GetInputOverrides() const
{
  return overrides_maps_;
}

bool
InferRequestProvider::HasInputOverride(const std::string& name)
{
  for (const auto& override_map : overrides_maps_) {
    if (override_map->find(name) != override_map->end()) {
      return true;
    }
  }

  return false;
}

bool
InferRequestProvider::GetInputOverrideShape(
    const std::string& name, std::vector<int64_t>* shape)
{
  shape->clear();
  for (const auto& override_map : overrides_maps_) {
    auto it = override_map->find(name);
    if (it != override_map->end()) {
      *shape = it->second.dims_;
      return true;
    }
  }

  return false;
}

Status
InferRequestProvider::AddInputOverrides(
    const std::shared_ptr<InputOverrideMap>& overrides)
{
  if ((overrides != nullptr) && !overrides->empty()) {
    overrides_maps_.emplace_back(overrides);
  }

  return Status::Success;
}

void
InferRequestProvider::SetInputOverrideConsumed(
    const std::string& name, const bool consumed)
{
  if (consumed) {
    overrides_consumed_.insert(name);
  } else {
    overrides_consumed_.erase(name);
  }
}

bool
InferRequestProvider::GetInputOverrideContent(
    const std::string& name, const void** content, size_t* content_byte_size)
{
  for (const auto& override_map : overrides_maps_) {
    const auto& pr = override_map->find(name);
    if (pr != override_map->end()) {
      if ((*content_byte_size == 0) ||
          (overrides_consumed_.find(name) != overrides_consumed_.end())) {
        *content = nullptr;
        *content_byte_size = 0;
      } else {
        const InputOverride& override = pr->second;
        *content = reinterpret_cast<const void*>(&(override.content_[0]));
        *content_byte_size = override.content_.size();
        overrides_consumed_.insert(name);
      }

      return true;
    }
  }

  return false;
}

Status
InferRequestProvider::GetNextInputContent(
    const std::string& name, const void** content, size_t* content_byte_size,
    TRTSERVER_Memory_Type* memory_type, int64_t* memory_type_id)
{
  if (*content_byte_size == 0) {
    *content = nullptr;
    return Status::Success;
  }

  if (GetInputOverrideContent(name, content, content_byte_size)) {
    *memory_type = TRTSERVER_MEMORY_CPU;
    *memory_type_id = 0;
  } else {
    auto* inputs = irequest_->MutableInputs();
    auto it = inputs->find(name);
    if (it == inputs->end()) {
      return Status(
          RequestStatusCode::INTERNAL, "unexpected input '" + name + "'");
    }

    RETURN_IF_ERROR(it->second.NextContent(
        content, content_byte_size, memory_type, memory_type_id));
  }

  return Status::Success;
}

Status
InferRequestProvider::GetMemory(
    const std::string& name, std::shared_ptr<Memory>* input_buffer)
{
  const auto& inputs = irequest_->Inputs();
  const auto it = inputs.find(name);
  if (it == inputs.end()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "input '" + name + "' is not found in the provider");
  }

  *input_buffer = it->second.Data();

  return Status::Success;
}

Status
InferRequestProvider::GetMemoryWithOverride(
    const std::string& name, const Memory** input_buffer)
{
  const auto& inputs = irequest_->Inputs();
  const auto it = inputs.find(name);
  if (it != inputs.end()) {
    *input_buffer = it->second.Data().get();
  } else {
    // check input overrides
    auto found = false;
    for (const auto& override_map : overrides_maps_) {
      const auto& pr = override_map->find(name);
      if (pr != override_map->end()) {
        const InputOverride& override = pr->second;
        *input_buffer = &override.content_ref_;

        found = true;
        break;
      }
    }

    if (!found) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "input '" + name + "' is not found in the provider");
    }
  }

  return Status::Success;
}

//
// NULLInferRequestProvider
//
std::vector<uint8_t> NULLInferRequestProvider::buf_;
std::mutex NULLInferRequestProvider::mu_;

Status
NULLInferRequestProvider::GetNextInputContent(
    const std::string& name, const void** content, size_t* content_byte_size,
    TRTSERVER_Memory_Type* memory_type, int64_t* memory_type_id)
{
  *memory_type = TRTSERVER_MEMORY_CPU;
  *memory_type_id = 0;
  if (*content_byte_size == 0) {
    *content = nullptr;
    return Status::Success;
  }

  if (!GetInputOverrideContent(name, content, content_byte_size)) {
    auto it = inputs_remaining_bytes_.find(name);
    if ((it != inputs_remaining_bytes_.end()) && (it->second == 0)) {
      *content = nullptr;
      *content_byte_size = 0;
    } else {
      // If it is first time requesting the input, the byte size hint will be
      // used as the expected input byte size.
      if (it == inputs_remaining_bytes_.end()) {
        it = inputs_remaining_bytes_.emplace(name, *content_byte_size).first;
      }

      std::lock_guard<std::mutex> lock(mu_);

      // Must return content with all zero data. This is required by
      // string-datatype tensors where it is interpreted as all empty
      // strings. Clamp the maximum size that we allow the buffer to
      // grow to avoid massive allocation.
      if (buf_.size() < *content_byte_size) {
        constexpr size_t max_size = 16 * 1024 * 1024;
        buf_.resize(std::min(max_size, *content_byte_size), 0);
      }

      *content = &(buf_[0]);

      // byte size to be returned is the min of actual buffer size,
      // expected remaining size (content_byte_size), and actual remaining size
      *content_byte_size =
          std::min(std::min(buf_.size(), *content_byte_size), it->second);
      it->second -= *content_byte_size;
    }
  }

  return Status::Success;
}

namespace {

template <typename T>
void
AddClassResults(
    InferResponseHeader::Output* poutput, char* poutput_buffer,
    const size_t batch1_element_count, const size_t batch_size,
    const size_t cls_count,
    const std::shared_ptr<LabelProvider>& label_provider,
    const InferResponseProvider::SecondaryLabelProviderMap& lookup_map)
{
  T* probs = reinterpret_cast<T*>(poutput_buffer);
  const size_t entry_cnt = batch1_element_count;
  const size_t class_cnt = std::min(cls_count, entry_cnt);
  std::vector<size_t> idx(entry_cnt);

  for (size_t i = 0; i < batch_size; ++i) {
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&probs](size_t i1, size_t i2) {
      return probs[i1] > probs[i2];
    });

    auto bcls = poutput->add_batch_classes();
    for (size_t k = 0; k < class_cnt; ++k) {
      auto cls = bcls->add_cls();
      cls->set_idx(idx[k]);
      const auto& label = label_provider->GetLabel(poutput->name(), idx[k]);
      cls->set_label(label);

      if (label == "" && !lookup_map.empty()) {
        auto it = lookup_map.find(poutput->name());
        if (it != lookup_map.end()) {
          cls->set_label(it->second.second->GetLabel(it->second.first, idx[k]));
        }
      }

      cls->set_value(static_cast<float>(probs[idx[k]]));
    }

    probs += entry_cnt;
  }
}

}  // namespace

//
// InferResponseProvider
//
InferResponseProvider::InferResponseProvider(
    const std::shared_ptr<InferenceRequest>& irequest,
    const std::shared_ptr<LabelProvider>& label_provider,
    TRTSERVER_ResponseAllocator* allocator,
    TRTSERVER_ResponseAllocatorAllocFn_t alloc_fn, void* alloc_userp,
    TRTSERVER_ResponseAllocatorReleaseFn_t release_fn)
    : irequest_(irequest), label_provider_(label_provider),
      allocator_(allocator), alloc_fn_(alloc_fn), alloc_userp_(alloc_userp),
      release_fn_(release_fn)
{
  // Create a map from output name to the InferenceRequest::Output
  // object for that output.
  for (const auto& pr : irequest_->RequestedOutputs()) {
    const auto& output = pr.second;
    output_map_.emplace(std::make_pair(output.Name(), output));
  }
}

bool
InferResponseProvider::RequiresOutput(const std::string& name)
{
  return output_map_.find(name) != output_map_.end();
}

Status
InferResponseProvider::OutputBufferContents(
    const std::string& name, const void** content, size_t* content_byte_size,
    TRTSERVER_Memory_Type* memory_type, int64_t* memory_type_id) const
{
  for (const auto& output : outputs_) {
    if ((name == output.name_) && (output.cls_count_ == 0)) {
      *content = output.ptr_;
      *content_byte_size = output.byte_size_;
      *memory_type = output.memory_type_;
      *memory_type_id = output.memory_type_id_;
      return Status::Success;
    }
  }

  return Status(
      RequestStatusCode::UNAVAILABLE,
      "request for unallocated output '" + name + "'");
}

bool
InferResponseProvider::GetSecondaryLabelProvider(
    const std::string& name, SecondaryLabelProvider* provider)
{
  auto it = secondary_label_provider_map_.find(name);
  if (it != secondary_label_provider_map_.end()) {
    *provider = it->second;
    return true;
  }
  return false;
}

void
InferResponseProvider::SetSecondaryLabelProvider(
    const std::string& name, const SecondaryLabelProvider& provider)
{
  secondary_label_provider_map_[name] = provider;
}

Status
InferResponseProvider::FinalizeResponse(const InferenceBackend& is)
{
  InferResponseHeader* response_header = MutableResponseHeader();
  response_header->Clear();

  response_header->set_model_name(is.Name());
  response_header->set_model_version(is.Version());

  const size_t batch_size = irequest_->BatchSize();
  response_header->set_batch_size(batch_size);

  int output_idx = 0;
  for (const auto& output : outputs_) {
    const ModelOutput* output_config;
    RETURN_IF_ERROR(is.GetOutput(output.name_, &output_config));

    // Verify that the actual output shape matches what is expected by
    // the model configuration. If there is an output reshape, we've
    // already verified that reshape and dims have same element count
    // so don't need to do that here.
    bool skip_batch = (is.Config().max_batch_size() != 0);
    DimsList batch1_backend_shape;
    size_t batch1_element_count = 1;
    for (auto d : output.shape_) {
      if (!skip_batch) {
        batch1_backend_shape.Add(d);
        batch1_element_count *= (size_t)d;
      }
      skip_batch = false;
    }

    const DimsList& expected_shape = (output_config->has_reshape())
                                         ? output_config->reshape().shape()
                                         : output_config->dims();
    if (!CompareDimsWithWildcard(expected_shape, batch1_backend_shape)) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "output '" + output.name_ + "' for model '" + is.Name() +
              "' has shape " + DimsListToString(batch1_backend_shape) +
              " but model configuration specifies shape " +
              DimsListToString(expected_shape));
    }

    auto poutput = response_header->add_output();
    poutput->set_name(output.name_);
    poutput->set_data_type(output_config->data_type());

    if (output.cls_count_ == 0) {
      // Raw result...
      poutput->mutable_raw()->Clear();
      poutput->mutable_raw()->set_batch_byte_size(output.byte_size_);

      // FIXMEV2 include batch dimension in V2 shape.
      if ((irequest_->ProtocolVersion() == 2) &&
          (is.Config().max_batch_size() != 0)) {
        poutput->mutable_raw()->add_dims(batch_size);
      }

      // If there is a reshape them we need to record corresponding value for
      // variable-size dimensions so that we can set the output shape correctly.
      // If there is not a reshape then use output shape as that will have
      // actual sized in place of any wildcard dimensions.
      if (output_config->has_reshape()) {
        std::deque<int64_t> variable_size_values;
        for (int64_t idx = 0; idx < output_config->reshape().shape_size();
             idx++) {
          if (output_config->reshape().shape(idx) == -1) {
            variable_size_values.push_back(batch1_backend_shape[idx]);
          }
        }

        for (const auto& dim : output_config->dims()) {
          if (dim == -1) {
            poutput->mutable_raw()->add_dims(variable_size_values.front());
            variable_size_values.pop_front();
          } else {
            poutput->mutable_raw()->add_dims(dim);
          }
        }
      } else {
        poutput->mutable_raw()->mutable_dims()->MergeFrom(batch1_backend_shape);
      }
    } else {
      // Class result...
      switch (output_config->data_type()) {
        case DataType::TYPE_UINT8:
          AddClassResults<uint8_t>(
              poutput, output.buffer_.get(), batch1_element_count, batch_size,
              output.cls_count_, label_provider_,
              secondary_label_provider_map_);
          break;
        case DataType::TYPE_UINT16:
          AddClassResults<uint16_t>(
              poutput, output.buffer_.get(), batch1_element_count, batch_size,
              output.cls_count_, label_provider_,
              secondary_label_provider_map_);
          break;
        case DataType::TYPE_UINT32:
          AddClassResults<uint32_t>(
              poutput, output.buffer_.get(), batch1_element_count, batch_size,
              output.cls_count_, label_provider_,
              secondary_label_provider_map_);
          break;
        case DataType::TYPE_UINT64:
          AddClassResults<uint64_t>(
              poutput, output.buffer_.get(), batch1_element_count, batch_size,
              output.cls_count_, label_provider_,
              secondary_label_provider_map_);
          break;

        case DataType::TYPE_INT8:
          AddClassResults<int8_t>(
              poutput, output.buffer_.get(), batch1_element_count, batch_size,
              output.cls_count_, label_provider_,
              secondary_label_provider_map_);
          break;
        case DataType::TYPE_INT16:
          AddClassResults<int16_t>(
              poutput, output.buffer_.get(), batch1_element_count, batch_size,
              output.cls_count_, label_provider_,
              secondary_label_provider_map_);
          break;
        case DataType::TYPE_INT32:
          AddClassResults<int32_t>(
              poutput, output.buffer_.get(), batch1_element_count, batch_size,
              output.cls_count_, label_provider_,
              secondary_label_provider_map_);
          break;
        case DataType::TYPE_INT64:
          AddClassResults<int64_t>(
              poutput, output.buffer_.get(), batch1_element_count, batch_size,
              output.cls_count_, label_provider_,
              secondary_label_provider_map_);
          break;

        case DataType::TYPE_FP32:
          AddClassResults<float>(
              poutput, output.buffer_.get(), batch1_element_count, batch_size,
              output.cls_count_, label_provider_,
              secondary_label_provider_map_);
          break;
        case DataType::TYPE_FP64:
          AddClassResults<double>(
              poutput, output.buffer_.get(), batch1_element_count, batch_size,
              output.cls_count_, label_provider_,
              secondary_label_provider_map_);
          break;

        default:
          return Status(
              RequestStatusCode::INVALID_ARG,
              "class result not available for output '" + output.name_ +
                  "' due to unsupported type '" +
                  DataType_Name(output_config->data_type()) + "'");
      }
    }

    output_idx++;
  }

  return Status::Success;
}

Status
InferResponseProvider::Create(
    const std::shared_ptr<InferenceRequest>& irequest,
    const std::shared_ptr<LabelProvider>& label_provider,
    TRTSERVER_ResponseAllocator* allocator,
    TRTSERVER_ResponseAllocatorAllocFn_t alloc_fn, void* alloc_userp,
    TRTSERVER_ResponseAllocatorReleaseFn_t release_fn,
    std::shared_ptr<InferResponseProvider>* infer_provider)
{
  InferResponseProvider* provider = new InferResponseProvider(
      irequest, label_provider, allocator, alloc_fn, alloc_userp, release_fn);
  infer_provider->reset(provider);

  return Status::Success;
}

InferResponseProvider::~InferResponseProvider()
{
  for (const auto& output : outputs_) {
    if (output.release_buffer_ != nullptr) {
#ifdef TRTIS_ENABLE_GPU
      int current_device;
      auto cuerr = cudaGetDevice(&current_device);
      // Ignore error caused by CPU-only system.
      if ((cuerr != cudaSuccess) && (cuerr != cudaErrorNoDevice) &&
          (cuerr != cudaErrorInsufficientDriver)) {
        LOG_ERROR << "unable to get current CUDA device: "
                  << cudaGetErrorString(cuerr);
      }
#endif  // TRTIS_ENABLE_GPU
      TRTSERVER_Error* err = release_fn_(
          allocator_, output.release_buffer_, output.release_userp_,
          output.byte_size_, output.memory_type_, output.memory_type_id_);
#ifdef TRTIS_ENABLE_GPU
      cuerr = cudaSetDevice(current_device);
      if ((cuerr != cudaSuccess) && (cuerr != cudaErrorNoDevice) &&
          (cuerr != cudaErrorInsufficientDriver)) {
        LOG_ERROR << "unable to recover current CUDA device: "
                  << cudaGetErrorString(cuerr);
      }
#endif  // TRTIS_ENABLE_GPU
      if (err != nullptr) {
        LOG_ERROR << "failed to release result tensor '" << output.name_
                  << "': " << TRTSERVER_ErrorMessage(err);
        TRTSERVER_ErrorDelete(err);
      }
    }
  }
}

const InferResponseHeader&
InferResponseProvider::ResponseHeader() const
{
  return response_header_;
}

InferResponseHeader*
InferResponseProvider::MutableResponseHeader()
{
  return &response_header_;
}

Status
InferResponseProvider::AllocateOutputBuffer(
    const std::string& name, void** content, size_t content_byte_size,
    const std::vector<int64_t>& content_shape,
    const TRTSERVER_Memory_Type preferred_memory_type,
    const int64_t preferred_memory_type_id,
    TRTSERVER_Memory_Type* actual_memory_type, int64_t* actual_memory_type_id)
{
  *content = nullptr;

  const auto& pr = output_map_.find(name);
  if (pr == output_map_.end()) {
    return Status(
        RequestStatusCode::INTERNAL, "unexpected output '" + name + "'");
  }

  outputs_.emplace_back();
  Output* loutput = &(outputs_.back());
  loutput->name_ = name;
  loutput->shape_ = content_shape;
  loutput->cls_count_ = 0;
  loutput->ptr_ = nullptr;
  loutput->byte_size_ = content_byte_size;

  // For class result, the provider will be responsible for allocating
  // the requested memory. The user-provided allocator should only be invoked
  // once with byte size 0 when the provider allocation is succeed.
  // For class result, the actual memory type must be CPU. If preferred memory
  // type is GPU then set actual_memory_type to CPU and proceed. Otherwise,
  // return success and nullptr to align with the behavior of
  // 'TRTSERVER_ResponseAllocatorAllocFn_t'
  const bool is_class = (pr->second.ClassificationCount() > 0);
  if (is_class) {
    // For class result no additional buffer is needed.
    if (content_byte_size == 0) {
      Status(
          RequestStatusCode::INVALID_ARG,
          "Classification result is requested for output '" + name + "'" +
              " while its output buffer size is 0");
    }

    loutput->cls_count_ = pr->second.ClassificationCount();
    char* buffer = new char[content_byte_size];
    *content = static_cast<void*>(buffer);
    loutput->ptr_ = static_cast<void*>(buffer);
    loutput->buffer_.reset(buffer);
  }

  // If a buffer has been allocated for cls result, then no
  // additional buffer is needed from alloc_fn, but still need to call the
  // alloc_fn_ with byte-size == 0 since that is what the API requires.
  const size_t alloc_byte_size = (*content != nullptr) ? 0 : content_byte_size;

  void* buffer = nullptr;
  void* buffer_userp = nullptr;
  TRTSERVER_Memory_Type raw_actual_memory_type;
  int64_t raw_actual_memory_type_id;
#ifdef TRTIS_ENABLE_GPU
  int current_device;
  auto cuerr = cudaGetDevice(&current_device);
  // Ignore error caused by CPU-only system.
  if ((cuerr != cudaSuccess) && (cuerr != cudaErrorNoDevice) &&
      (cuerr != cudaErrorInsufficientDriver)) {
    return Status(
        RequestStatusCode::INTERNAL,
        "unable to get current CUDA device: " +
            std::string(cudaGetErrorString(cuerr)));
  }
#endif  // TRTIS_ENABLE_GPU
  TRTSERVER_Error* err = alloc_fn_(
      allocator_, name.c_str(), alloc_byte_size, preferred_memory_type,
      preferred_memory_type_id, alloc_userp_, &buffer, &buffer_userp,
      &raw_actual_memory_type, &raw_actual_memory_type_id);
  if (!is_class) {
    *content = buffer;
    loutput->ptr_ = buffer;
    loutput->memory_type_ = raw_actual_memory_type;
    loutput->memory_type_id_ = raw_actual_memory_type_id;
  } else {
    // If class result, then force the memory type to be CPU
    loutput->memory_type_ = TRTSERVER_MEMORY_CPU;
    loutput->memory_type_id_ = 0;
  }

  Status status;
#ifdef TRTIS_ENABLE_GPU
  cuerr = cudaSetDevice(current_device);
  if ((cuerr != cudaSuccess) && (cuerr != cudaErrorNoDevice) &&
      (cuerr != cudaErrorInsufficientDriver)) {
    status = Status(
        RequestStatusCode::INTERNAL,
        "unable to recover current CUDA device: " +
            std::string(cudaGetErrorString(cuerr)));
  }
#endif  // TRTIS_ENABLE_GPU
  if (err != nullptr) {
    status = Status(
        TrtServerCodeToRequestStatus(TRTSERVER_ErrorCode(err)),
        TRTSERVER_ErrorMessage(err));
    TRTSERVER_ErrorDelete(err);
  }
  if (!status.IsOk()) {
    return status;
  }

  loutput->release_buffer_ = buffer;
  loutput->release_userp_ = buffer_userp;
  *actual_memory_type = loutput->memory_type_;
  *actual_memory_type_id = loutput->memory_type_id_;

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
