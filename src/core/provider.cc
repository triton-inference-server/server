// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <google/protobuf/text_format.h>
#include "src/core/backend.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/model_config_utils.h"

namespace nvidia { namespace inferenceserver {

SystemMemoryReference::SystemMemoryReference() : SystemMemory() {}

const char*
SystemMemoryReference::BufferAt(size_t idx, size_t* byte_size) const
{
  if (idx >= buffer_.size()) {
    *byte_size = 0;
    return nullptr;
  }
  *byte_size = buffer_[idx].second;
  return buffer_[idx].first;
}

size_t
SystemMemoryReference::AddBuffer(const char* buffer, size_t byte_size)
{
  buffer_.emplace_back(std::make_pair(buffer, byte_size));
  return buffer_.size() - 1;
}

AllocatedSystemMemory::AllocatedSystemMemory(size_t byte_size) : SystemMemory()
{
  total_byte_size_ = byte_size;
  char* buffer = new char[byte_size];
  buffer_.reset(buffer);
}

const char*
AllocatedSystemMemory::BufferAt(size_t idx, size_t* byte_size) const
{
  if (idx != 0) {
    *byte_size = 0;
    return nullptr;
  }
  *byte_size = total_byte_size_;
  return buffer_.get();
}

char*
AllocatedSystemMemory::MutableBuffer()
{
  return buffer_.get();
}

//
// Create function for request received from GRPC
//
Status
InferRequestProvider::Create(
    const InferenceBackend& is, const InferRequest& request,
    std::shared_ptr<InferRequestProvider>* provider)
{
  const int64_t version =
      (request.model_version() >= 0) ? request.model_version() : -1;
  provider->reset(new InferRequestProvider(request.model_name(), version));

  (*provider)->request_header_ = request.meta_data();
  RETURN_IF_ERROR((*provider)->NormalizeRequestHeader(is));

  const InferRequestHeader& request_header = (*provider)->request_header_;

  // Make sure that the request is providing the same number of raw
  // input tensor data.
  if (request_header.input_size() != request.raw_input_size()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "expected tensor data for " +
            std::to_string(request_header.input_size()) + " inputs but got " +
            std::to_string(request.raw_input_size()) +
            " sets of data for model '" + (*provider)->model_name_ + "'");
  }

  // Verify that the batch-byte-size of each input matches the size of
  // the provided raw tensor data.
  size_t idx = 0;
  for (const auto& io : request_header.input()) {
    auto memory_ref = std::make_shared<SystemMemoryReference>();
    (*provider)->input_buffer_.emplace(std::make_pair(
        io.name(),
        std::make_pair(std::static_pointer_cast<SystemMemory>(memory_ref), 0)));

    if (io.batch_byte_size() != request.raw_input(idx).size()) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "unexpected size " + std::to_string(request.raw_input(idx).size()) +
              " for input '" + io.name() + "', expecting " +
              std::to_string(io.batch_byte_size()) + " for model '" +
              (*provider)->model_name_ + "'");
    }

    const std::string& raw = request.raw_input(idx++);
    memory_ref->AddBuffer(raw.c_str(), raw.size());
  }

  return Status::Success;
}

//
// Create function for request received from HTTP
//
Status
InferRequestProvider::Create(
    const InferenceBackend& is, const std::string& model_name,
    const int64_t model_version, const std::string& request_header_str,
    evbuffer* input_buffer, std::shared_ptr<InferRequestProvider>* provider)
{
  provider->reset(new InferRequestProvider(model_name, model_version));

  if (!google::protobuf::TextFormat::ParseFromString(
          request_header_str, &((*provider)->request_header_))) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "unable to parse request for model '" + model_name + "'");
  }

  RETURN_IF_ERROR((*provider)->NormalizeRequestHeader(is));

  const InferRequestHeader& request_header = (*provider)->request_header_;
  // Now need to create 'ref'. Each input has one entry in
  // SystemMemory which gives a list of all the blocks of data for that
  // input. These blocks are not necessarily contiguous so we keep
  // track of each separately to avoid needing to copy everything into
  // one buffer.
  //
  // Get the addr and size of each chunk of input data from the
  // evbuffer.
  int n = evbuffer_peek(input_buffer, -1, NULL, NULL, 0);
  if (n > 0) {
    struct evbuffer_iovec* v = static_cast<struct evbuffer_iovec*>(
        alloca(sizeof(struct evbuffer_iovec) * n));
    if (evbuffer_peek(input_buffer, -1, NULL, v, n) != n) {
      return Status(
          RequestStatusCode::INTERNAL,
          "unexpected error getting input buffers ");
    }

    int v_idx = 0;

    // Get the byte-size for each input and from that get the blocks
    // holding the data for that input
    for (const auto& io : request_header.input()) {
      auto memory_ref = std::make_shared<SystemMemoryReference>();
      (*provider)->input_buffer_.emplace(std::make_pair(
          io.name(),
          std::make_pair(
              std::static_pointer_cast<SystemMemory>(memory_ref), 0)));

      uint64_t byte_size = io.batch_byte_size();
      while ((byte_size > 0) && (v_idx < n)) {
        char* base = static_cast<char*>(v[v_idx].iov_base);
        size_t base_size;
        if (v[v_idx].iov_len > byte_size) {
          base_size = byte_size;
          v[v_idx].iov_base = static_cast<void*>(base + byte_size);
          v[v_idx].iov_len -= byte_size;
          byte_size = 0;
        } else {
          base_size = v[v_idx].iov_len;
          byte_size -= v[v_idx].iov_len;
          v_idx++;
        }
        memory_ref->AddBuffer(base, base_size);
      }

      if (byte_size != 0) {
        return Status(
            RequestStatusCode::INVALID_ARG,
            "unexpected size for input '" + io.name() +
                "', missing expecting " + std::to_string(byte_size) +
                " bytes for model '" + model_name + "'");
      }
    }

    if (v_idx != n) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "unexpected additional input data for model '" + model_name + "'");
    }
  }

  return Status::Success;
}

//
// Create function for request received from within the process
//
Status
InferRequestProvider::Create(
    const InferenceBackend& is, const std::string& model_name,
    const int64_t model_version, const InferRequestHeader& request_header,
    const std::unordered_map<std::string, std::shared_ptr<SystemMemory>>&
        input_buffer,
    std::shared_ptr<InferRequestProvider>* provider)
{
  provider->reset(new InferRequestProvider(model_name, model_version));

  (*provider)->request_header_ = request_header;

  RETURN_IF_ERROR((*provider)->NormalizeRequestHeader(is));

  for (const auto& io : (*provider)->request_header_.input()) {
    auto it = input_buffer.find(io.name());
    if (it == input_buffer.end()) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "input '" + io.name() + "' is specified in request header but" +
              " not found in memory block mapping for model '" +
              (*provider)->model_name_ + "'");
    }
    if (io.batch_byte_size() != it->second->TotalByteSize()) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "unexpected size " + std::to_string(it->second->TotalByteSize()) +
              " for input '" + io.name() + "', expecting " +
              std::to_string(io.batch_byte_size()) + " for model '" +
              (*provider)->model_name_ + "'");
    }
    (*provider)->input_buffer_[io.name()] = std::make_pair(it->second, 0);
  }

  return Status::Success;
}

Status
InferRequestProvider::NormalizeRequestHeader(const InferenceBackend& is)
{
  const ModelConfig& model_config = is.Config();

  // Make sure the request has a batch-size > 0. Even for models that
  // don't support batching the requested batch size must be 1.
  if (request_header_.batch_size() < 1) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "inference request batch-size must be >= 1 for '" + model_name_ + "'");
  }

  // Make sure request batch-size doesn't exceed what is supported by
  // the model. For models that don't support batching the request
  // batch-size will still be 1.
  if ((request_header_.batch_size() != 1) &&
      ((int)request_header_.batch_size() > model_config.max_batch_size())) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "inference request batch-size must be <= " +
            std::to_string(model_config.max_batch_size()) + " for '" +
            model_name_ + "'");
  }

  // Make sure that the request is providing the same number of inputs
  // as is expected by the model.
  if (request_header_.input_size() != model_config.input_size()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "expected " + std::to_string(model_config.input_size()) +
            " inputs but got " + std::to_string(request_header_.input_size()) +
            " inputs for model '" + model_name_ + "'");
  }

  // Update each input to have shape and batch-byte-size.
  uint64_t bs = 0;
  for (InferRequestHeader::Input& io : *request_header_.mutable_input()) {
    const ModelInput* input_config;
    RETURN_IF_ERROR(is.GetInput(io.name(), &input_config));

    // If the inference request specifies a shape for an input, make
    // sure it matches what the model expects and then calculate the
    // expected input size from that shape.
    if (io.dims_size() > 0) {
      if (!CompareDimsWithWildcard(io.dims(), input_config->dims())) {
        return Status(
            RequestStatusCode::INVALID_ARG, "unexpected shape for input '" +
                                                io.name() + "' for model '" +
                                                model_name_ + "'");
      }

      bs = GetByteSize(input_config->data_type(), io.dims());
    } else {
      // Inference request doesn't specify shape, make sure input
      // shape is fully specified in the model and calculate expected
      // size from the model configuration.
      for (auto dim : input_config->dims()) {
        if (dim < 0) {
          return Status(
              RequestStatusCode::INVALID_ARG,
              "model supports variable-size for input '" + io.name() +
                  "', request must specify input shape for model '" +
                  model_name_ + "'");
        }

        io.add_dims(dim);
      }

      bs = GetByteSize(*input_config);
    }

    // If the input's datatype is not fixed-sized (like TYPE_STRING)
    // then need to use the full-batch size specified by the
    // input. For fixed-size datatype if batch-byte-size is given
    // check to make sure that the calculated batch size matches.
    if (IsFixedSizeDataType(input_config->data_type())) {
      bs *= request_header_.batch_size();
      if ((io.batch_byte_size() != 0) && (io.batch_byte_size() != bs)) {
        return Status(
            RequestStatusCode::INVALID_ARG,
            "specific batch-byte-size for input '" + io.name() +
                "' does not match expected byte-size calculated from shape and "
                "datatype for model '" +
                model_name_ + "'");
      }
    } else {
      if (io.batch_byte_size() == 0) {
        return Status(
            RequestStatusCode::INVALID_ARG,
            "batch-byte-size must be specified for input '" + io.name() +
                "' with non-fixed-size datatype for model '" + model_name_ +
                "'");
      }

      bs = io.batch_byte_size();
    }

    io.set_batch_byte_size(bs);
  }

  return Status::Success;
}

const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
InferRequestProvider::GetInputOverride() const
{
  return overrides_;
}

Status
InferRequestProvider::SetInputOverride(
    const std::shared_ptr<InputOverrideMap>& override)
{
  overrides_ = override;
  return Status::Success;
}

bool
InferRequestProvider::GetInputOverrideContent(
    const std::string& name, const void** content, size_t* content_byte_size)
{
  if (overrides_ != nullptr) {
    const auto& pr = overrides_->find(name);
    if (pr != overrides_->end()) {
      if ((*content_byte_size == 0) ||
          (overrides_consumed_.find(name) != overrides_consumed_.end())) {
        *content = nullptr;
        *content_byte_size = 0;
      } else {
        std::shared_ptr<InputOverride>& override = pr->second;
        *content = reinterpret_cast<void*>(&(override->content_[0]));
        *content_byte_size = override->content_.size();
        overrides_consumed_.insert(name);
      }

      return true;
    }
  }

  return false;
}

//
// NULLInferRequestProvider
//
std::vector<uint8_t> NULLInferRequestProvider::buf_;
std::mutex NULLInferRequestProvider::mu_;

Status
NULLInferRequestProvider::GetNextInputContent(
    const std::string& name, const void** content, size_t* content_byte_size,
    bool force_contiguous)
{
  if (*content_byte_size == 0) {
    *content = nullptr;
    return Status::Success;
  }

  if (!GetInputOverrideContent(name, content, content_byte_size)) {
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
  }

  return Status::Success;
}

Status
InferRequestProvider::GetNextInputContent(
    const std::string& name, const void** content, size_t* content_byte_size,
    bool force_contiguous)
{
  if (*content_byte_size == 0) {
    *content = nullptr;
    return Status::Success;
  }

  if (!GetInputOverrideContent(name, content, content_byte_size)) {
    const auto& pr = input_buffer_.find(name);
    if (pr == input_buffer_.end()) {
      return Status(
          RequestStatusCode::INTERNAL, "unexpected input '" + name + "'");
    }

    auto& input_content = pr->second;

    bool isLastChunk =
        (input_content.first->BufferAt(
             input_content.second + 1, content_byte_size) == nullptr);
    if (!force_contiguous || isLastChunk) {
      *content = input_content.first->BufferAt(
          input_content.second, content_byte_size);
      if (*content_byte_size != 0) {
        input_content.second++;
      }
    } else {
      size_t total_size = 0;
      size_t start_idx = input_content.second;
      do {
        *content = input_content.first->BufferAt(
            input_content.second++, content_byte_size);
        total_size += *content_byte_size;
      } while (*content != nullptr);

      contiguous_buffers_.emplace_back();
      std::vector<char>& buf = contiguous_buffers_.back();
      buf.reserve(total_size);

      for (size_t i = start_idx; i < input_content.second; i++) {
        const auto& block = input_content.first->BufferAt(i, content_byte_size);
        buf.insert(buf.end(), block, block + *content_byte_size);
      }

      if (buf.size() != total_size) {
        return Status(RequestStatusCode::INTERNAL, "contiguous input failed");
      }

      *content = &(buf[0]);
      *content_byte_size = total_size;
    }
  }

  return Status::Success;
}

Status
InferRequestProvider::GetSystemMemory(
    const std::string& name, std::shared_ptr<SystemMemory>* input_buffer)
{
  auto it = input_buffer_.find(name);
  if (it == input_buffer_.end()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "input '" + name + "' is not found in the provider");
  }
  *input_buffer = it->second.first;
  return Status::Success;
}

Status
GRPCInferResponseProvider::Create(
    const InferRequestHeader& request_header, InferResponse* response,
    std::shared_ptr<GRPCInferResponseProvider>* infer_provider)
{
  GRPCInferResponseProvider* provider =
      new GRPCInferResponseProvider(request_header, response);
  infer_provider->reset(provider);

  return Status::Success;
}

const InferResponseHeader&
GRPCInferResponseProvider::ResponseHeader() const
{
  return response_->meta_data();
}

InferResponseHeader*
GRPCInferResponseProvider::MutableResponseHeader()
{
  return response_->mutable_meta_data();
}

Status
GRPCInferResponseProvider::GetOutputBuffer(
    const std::string& name, void** content, size_t content_byte_size,
    const std::vector<int64_t>& content_shape)
{
  Output* output;
  RETURN_IF_ERROR(CheckAndSetIfBufferedOutput(
      name, content, content_byte_size, content_shape, &output));

  // Must always add a raw output into the list so that the number and
  // order of raw output entries equals the output meta-data. But
  // leave empty if not returning raw result for the output.
  std::string* raw_output = response_->add_raw_output();
  if (output->buffer_ == nullptr) {
    raw_output->resize(content_byte_size);
    *content = static_cast<void*>(&((*raw_output)[0]));
  }

  return Status::Success;
}

HTTPInferResponseProvider::HTTPInferResponseProvider(
    evbuffer* output_buffer, const InferRequestHeader& request_header)
    : InferResponseProvider(request_header), output_buffer_(output_buffer)
{
}

Status
HTTPInferResponseProvider::Create(
    evbuffer* output_buffer, const InferenceBackend& is,
    const InferRequestHeader& request_header,
    std::shared_ptr<HTTPInferResponseProvider>* infer_provider)
{
  HTTPInferResponseProvider* provider =
      new HTTPInferResponseProvider(output_buffer, request_header);
  infer_provider->reset(provider);

  return Status::Success;
}

const InferResponseHeader&
HTTPInferResponseProvider::ResponseHeader() const
{
  return response_header_;
}

InferResponseHeader*
HTTPInferResponseProvider::MutableResponseHeader()
{
  return &response_header_;
}

Status
HTTPInferResponseProvider::GetOutputBuffer(
    const std::string& name, void** content, size_t content_byte_size,
    const std::vector<int64_t>& content_shape)
{
  *content = nullptr;

  Output* output;
  RETURN_IF_ERROR(CheckAndSetIfBufferedOutput(
      name, content, content_byte_size, content_shape, &output));

  if ((output->buffer_ == nullptr) && (content_byte_size > 0)) {
    // Reserve requested space in evbuffer...
    struct evbuffer_iovec output_iovec;
    if (evbuffer_reserve_space(
            output_buffer_, content_byte_size, &output_iovec, 1) != 1) {
      return Status(
          RequestStatusCode::INTERNAL, "failed to reserve " +
                                           std::to_string(content_byte_size) +
                                           " bytes in output tensor buffer");
    }

    if (output_iovec.iov_len < content_byte_size) {
      return Status(
          RequestStatusCode::INTERNAL,
          "reserved " + std::to_string(output_iovec.iov_len) +
              " bytes in output tensor buffer, need " +
              std::to_string(content_byte_size));
    }

    output_iovec.iov_len = content_byte_size;
    *content = output_iovec.iov_base;

    // Immediately commit the buffer space. Some backends will write
    // async to the just allocated buffer space so we are relying on
    // evbuffer not to relocate this space. Because we request a
    // contiguous chunk every time (above by allowing only a single
    // entry in output_iovec), this seems to be a valid assumption.
    if (evbuffer_commit_space(output_buffer_, &output_iovec, 1) != 0) {
      *content = nullptr;
      return Status(
          RequestStatusCode::INTERNAL,
          "failed to commit output tensors to output buffer");
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
    const InferRequestHeader::Output& output,
    const LabelProvider& label_provider)
{
  T* probs = reinterpret_cast<T*>(poutput_buffer);
  const size_t entry_cnt = batch1_element_count;
  const size_t class_cnt = std::min((size_t)output.cls().count(), entry_cnt);
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
      cls->set_label(label_provider.GetLabel(output.name(), idx[k]));

      cls->set_value(static_cast<float>(probs[idx[k]]));
    }

    probs += entry_cnt;
  }
}

}  // namespace

InferResponseProvider::InferResponseProvider(
    const InferRequestHeader& request_header)
    : request_header_(request_header)
{
  // Create a map from output name to the InferRequestHeader::Output
  // object for that output.
  for (const InferRequestHeader::Output& output : request_header.output()) {
    output_map_.emplace(std::make_pair(output.name(), &output));
  }
}

bool
InferResponseProvider::RequiresOutput(const std::string& name)
{
  return output_map_.find(name) != output_map_.end();
}

Status
InferResponseProvider::CheckAndSetIfBufferedOutput(
    const std::string& name, void** content, size_t content_byte_size,
    const std::vector<int64_t>& content_shape, Output** output)
{
  const auto& pr = output_map_.find(name);
  if (pr == output_map_.end()) {
    return Status(
        RequestStatusCode::INTERNAL, "unexpected output '" + name + "'");
  }

  outputs_.emplace_back();
  Output* loutput = &(outputs_.back());
  loutput->name_ = name;
  loutput->shape_ = content_shape;
  loutput->byte_size_ = content_byte_size;

  if (pr->second->has_cls()) {
    char* buffer = new char[content_byte_size];
    *content = static_cast<void*>(buffer);
    loutput->buffer_.reset(buffer);
  }

  *output = loutput;

  return Status::Success;
}

Status
InferResponseProvider::FinalizeResponse(const InferenceBackend& is)
{
  InferResponseHeader* response_header = MutableResponseHeader();
  response_header->Clear();

  const LabelProvider& label_provider = is.GetLabelProvider();

  response_header->set_model_name(is.Name());
  response_header->set_model_version(is.Version());

  const size_t batch_size = request_header_.batch_size();
  response_header->set_batch_size(batch_size);

  int output_idx = 0;
  for (const auto& output : outputs_) {
    auto poutput = response_header->add_output();
    poutput->set_name(output.name_);

    if (output.buffer_ == nullptr) {
      // Raw result...
      poutput->mutable_raw()->Clear();
      poutput->mutable_raw()->set_batch_byte_size(output.byte_size_);

      // If the model produces batched output, don't include the batch
      // dimension.
      bool skip = (is.Config().max_batch_size() != 0);
      for (auto d : output.shape_) {
        if (!skip) {
          poutput->mutable_raw()->add_dims(d);
        }
        skip = false;
      }
    } else {
      // Class result...
      const ModelOutput* output_config;
      RETURN_IF_ERROR(is.GetOutput(output.name_, &output_config));

      const auto& pr = output_map_.find(output.name_);
      if (pr == output_map_.end()) {
        return Status(
            RequestStatusCode::INTERNAL,
            "can't find request meta-data for output '" + output.name_ + "'");
      }
      const InferRequestHeader::Output* request_output = pr->second;

      // Determine the number of elements in a batch-1 output.
      size_t batch1_element_count = 1;
      bool skip = (is.Config().max_batch_size() != 0);
      for (auto d : output.shape_) {
        if (!skip) {
          batch1_element_count *= (size_t)d;
        }
        skip = false;
      }

      switch (output_config->data_type()) {
        case DataType::TYPE_UINT8:
          AddClassResults<uint8_t>(
              poutput, output.buffer_.get(), batch1_element_count, batch_size,
              *request_output, label_provider);
          break;
        case DataType::TYPE_UINT16:
          AddClassResults<uint16_t>(
              poutput, output.buffer_.get(), batch1_element_count, batch_size,
              *request_output, label_provider);
          break;
        case DataType::TYPE_UINT32:
          AddClassResults<uint32_t>(
              poutput, output.buffer_.get(), batch1_element_count, batch_size,
              *request_output, label_provider);
          break;
        case DataType::TYPE_UINT64:
          AddClassResults<uint64_t>(
              poutput, output.buffer_.get(), batch1_element_count, batch_size,
              *request_output, label_provider);
          break;

        case DataType::TYPE_INT8:
          AddClassResults<int8_t>(
              poutput, output.buffer_.get(), batch1_element_count, batch_size,
              *request_output, label_provider);
          break;
        case DataType::TYPE_INT16:
          AddClassResults<int16_t>(
              poutput, output.buffer_.get(), batch1_element_count, batch_size,
              *request_output, label_provider);
          break;
        case DataType::TYPE_INT32:
          AddClassResults<int32_t>(
              poutput, output.buffer_.get(), batch1_element_count, batch_size,
              *request_output, label_provider);
          break;
        case DataType::TYPE_INT64:
          AddClassResults<int64_t>(
              poutput, output.buffer_.get(), batch1_element_count, batch_size,
              *request_output, label_provider);
          break;

        case DataType::TYPE_FP32:
          AddClassResults<float>(
              poutput, output.buffer_.get(), batch1_element_count, batch_size,
              *request_output, label_provider);
          break;
        case DataType::TYPE_FP64:
          AddClassResults<double>(
              poutput, output.buffer_.get(), batch1_element_count, batch_size,
              *request_output, label_provider);
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
InternalInferResponseProvider::Create(
    const InferenceBackend& is, const InferRequestHeader& request_header,
    std::shared_ptr<InternalInferResponseProvider>* infer_provider)
{
  auto provider = new InternalInferResponseProvider(request_header);
  infer_provider->reset(provider);
  return Status::Success;
}

const InferResponseHeader&
InternalInferResponseProvider::ResponseHeader() const
{
  return response_header_;
}

InferResponseHeader*
InternalInferResponseProvider::MutableResponseHeader()
{
  return &response_header_;
}

Status
InternalInferResponseProvider::GetOutputBuffer(
    const std::string& name, void** content, size_t content_byte_size,
    const std::vector<int64_t>& content_shape)
{
  *content = nullptr;

  Output* output;
  RETURN_IF_ERROR(CheckAndSetIfBufferedOutput(
      name, content, content_byte_size, content_shape, &output));

  // Always write output tensor to an output buffer no matter
  // if output has cls field defined
  auto it = output_buffer_.find(name);
  if (it == output_buffer_.end()) {
    it = output_buffer_
             .emplace(std::make_pair(
                 name,
                 std::make_shared<AllocatedSystemMemory>(content_byte_size)))
             .first;
  }

  if (content_byte_size != it->second->TotalByteSize()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "unexpected size " + std::to_string(it->second->TotalByteSize()) +
            " for output '" + name + "', expecting " +
            std::to_string(content_byte_size));
  }

  *content = it->second->MutableBuffer();

  return Status::Success;
}

Status
InternalInferResponseProvider::GetSystemMemory(
    const std::string& name, std::shared_ptr<SystemMemory>* output_buffer)
{
  auto it = output_buffer_.find(name);
  if (it == output_buffer_.end()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "output '" + name + "' is not found in response provider");
  }
  *output_buffer = std::static_pointer_cast<SystemMemory>(it->second);
  return Status::Success;
}

InternalInferResponseProvider::InternalInferResponseProvider(
    const InferRequestHeader& request_header)
    : InferResponseProvider(request_header)
{
}

}}  // namespace nvidia::inferenceserver
