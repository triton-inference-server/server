// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include "src/core/infer.h"

#include <chrono>
#include "src/core/constants.h"
#include "src/core/dynamic_batch_scheduler.h"
#include "src/core/logging.h"
#include "src/core/sequence_batch_scheduler.h"
#include "src/core/utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace nvidia { namespace inferenceserver {

tensorflow::Status
InferRequestProvider::GetInputByteSize(
    const InferRequestHeader::Input& input, const ModelInput& input_config,
    uint64_t* byte_size)
{
  uint64_t bs = 0;

  // If the inference request specifies a shape for an input, make
  // sure it matches what the model expects and then calculate the
  // expected input size from that shape.
  if (input.dims_size() > 0) {
    if (!CompareDimsWithWildcard(input.dims(), input_config.dims())) {
      return tensorflow::errors::InvalidArgument(
          "expected equal shape for input '", input.name(), "' for model '",
          ModelName(), "'");
    }

    bs = GetByteSize(input_config.data_type(), input.dims());
  } else {
    // Inference request doesn't specify shape, make sure input
    // shape is fully specified in the model and calculate expected
    // size from the model configuration.
    for (auto dim : input_config.dims()) {
      if (dim < 0) {
        return tensorflow::errors::InvalidArgument(
            "expected shape for input '", input.name(), "' for model '",
            ModelName(), "'");
      }
    }

    bs = GetByteSize(input_config);
  }

  // If the input's datatype is not fixed-sized (like TYPE_STRING)
  // then need to use the full-batch size specified by the input.
  if (IsFixedSizeDataType(input_config.data_type())) {
    bs *= RequestHeader().batch_size();
  } else {
    bs = input.batch_byte_size();
  }

  *byte_size = bs;
  return tensorflow::Status::OK();
}

GRPCInferRequestProvider::GRPCInferRequestProvider(
    const InferRequest& request, const int version)
    : InferRequestProvider(request.model_name(), version), request_(request)
{
  content_delivered_.resize(request_.raw_input_size(), false);
}

tensorflow::Status
GRPCInferRequestProvider::Create(
    const InferenceServable& is, const InferRequest& request,
    std::shared_ptr<GRPCInferRequestProvider>* infer_provider)
{
  const int version = (request.version() >= 0) ? request.version() : -1;
  infer_provider->reset(new GRPCInferRequestProvider(request, version));

  const ModelConfig& model_config = is.Config();
  const InferRequestHeader& request_header = request.meta_data();
  const std::string& model_name = request.model_name();

  // Make sure the request has a batch-size > 0. Even for models that
  // don't support batching the requested batch size must be 1.
  if (request_header.batch_size() < 1) {
    return tensorflow::errors::InvalidArgument(
        "inference request batch-size must be >= 1 for '", model_name, "'");
  }

  // Make sure request batch-size doesn't exceed what is supported by
  // the model. For models that don't support batching the request
  // batch-size will still be 1.
  if ((request_header.batch_size() != 1) &&
      ((int)request_header.batch_size() > model_config.max_batch_size())) {
    return tensorflow::errors::InvalidArgument(
        "inference request batch-size must be <= ",
        std::to_string(model_config.max_batch_size()), " for '", model_name,
        "'");
  }

  // Make sure that the request is providing the same number of inputs
  // as is expected by the model.
  if (request_header.input_size() != request.raw_input_size()) {
    return tensorflow::errors::InvalidArgument(
        "expected tensor data for ", request_header.input_size(),
        " inputs but got ", request.raw_input_size(),
        " sets of data for model '", model_name, "'");
  }
  if (request_header.input_size() != model_config.input_size()) {
    return tensorflow::errors::InvalidArgument(
        "expected ", model_config.input_size(), " inputs but got ",
        request_header.input_size(), " inputs for model '", model_name, "'");
  }

  // Get the byte-size expected for each input and verify that the
  // request is providing exactly that size of input.
  size_t idx = 0;
  for (const auto& io : request_header.input()) {
    const ModelInput* input_config;
    TF_RETURN_IF_ERROR(is.GetInput(io.name(), &input_config));

    uint64_t byte_size = 0;
    TF_RETURN_IF_ERROR(
        (*infer_provider)->GetInputByteSize(io, *input_config, &byte_size));

    if (byte_size != request.raw_input(idx).size()) {
      return tensorflow::errors::InvalidArgument(
          "unexpected size ", request.raw_input(idx).size(), " for input '",
          io.name(), "', expecting ", byte_size, " for model '", model_name,
          "'");
    }

    idx++;
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
GRPCInferRequestProvider::GetNextInputContent(
    int idx, const void** content, size_t* content_byte_size,
    bool force_contiguous)
{
  if ((idx < 0) || (idx >= request_.raw_input_size())) {
    return tensorflow::errors::Internal("unexpected input index ", idx);
  }

  if (content_delivered_[idx]) {
    *content = nullptr;
    *content_byte_size = 0;
  } else {
    const std::string& raw = request_.raw_input(idx);
    *content = raw.c_str();
    *content_byte_size = raw.size();
    content_delivered_[idx] = true;
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
HTTPInferRequestProvider::Create(
    evbuffer* input_buffer, const InferenceServable& is,
    const std::string& model_name, const int model_version,
    const std::string& request_header_str,
    std::shared_ptr<HTTPInferRequestProvider>* infer_provider)
{
  auto provider = new HTTPInferRequestProvider(model_name, model_version);
  infer_provider->reset(provider);

  if (!tensorflow::protobuf::TextFormat::ParseFromString(
          request_header_str, &(provider->request_header_))) {
    return tensorflow::errors::InvalidArgument(
        "unable to parse request for model '", model_name, "'");
  }

  const InferRequestHeader& request_header = provider->request_header_;
  const ModelConfig& model_config = is.Config();

  // Make sure the request has a batch-size > 0. Even for models that
  // don't support batching the requested batch size must be 1.
  if (request_header.batch_size() < 1) {
    return tensorflow::errors::InvalidArgument(
        "inference request batch-size must be >= 1 for '", model_name, "'");
  }

  // Make sure request batch-size doesn't exceed what is supported by
  // the model. For models that don't support batching the request
  // batch-size will still be 1.
  if ((request_header.batch_size() != 1) &&
      ((int)request_header.batch_size() > model_config.max_batch_size())) {
    return tensorflow::errors::InvalidArgument(
        "inference request batch-size must be <= ",
        std::to_string(model_config.max_batch_size()), " for '", model_name,
        "'");
  }

  // Make sure that the request is providing the same number of inputs
  // as is expected by the model.
  if (request_header.input_size() != model_config.input_size()) {
    return tensorflow::errors::InvalidArgument(
        "expected ", model_config.input_size(), " inputs but got ",
        request_header.input_size(), " inputs for model '", model_name, "'");
  }

  // Now need to create 'contents_'. Each input has one entry in
  // 'contents_' which gives a list of all the blocks of data for that
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
      return tensorflow::errors::Internal(
          "unexpected error getting input buffers ");
    }

    int v_idx = 0;

    // Get the byte-size for each input and from that get the blocks
    // holding the that must data for that input
    for (const auto& io : request_header.input()) {
      provider->contents_idx_.push_back(0);
      provider->contents_.emplace_back();
      auto& blocks = provider->contents_.back();

      const ModelInput* input_config;
      TF_RETURN_IF_ERROR(is.GetInput(io.name(), &input_config));

      uint64_t byte_size = 0;
      TF_RETURN_IF_ERROR(
          (*infer_provider)->GetInputByteSize(io, *input_config, &byte_size));

      while ((byte_size > 0) && (v_idx < n)) {
        blocks.emplace_back();
        Block& block = blocks.back();

        char* base = static_cast<char*>(v[v_idx].iov_base);
        block.first = base;
        if (v[v_idx].iov_len > byte_size) {
          block.second = byte_size;
          v[v_idx].iov_base = static_cast<void*>(base + byte_size);
          v[v_idx].iov_len -= byte_size;
          byte_size = 0;
        } else {
          block.second = v[v_idx].iov_len;
          byte_size -= v[v_idx].iov_len;
          v_idx++;
        }
      }

      if (byte_size != 0) {
        return tensorflow::errors::InvalidArgument(
            "unexpected size for input '", io.name(), "', missing expecting ",
            byte_size, " bytes for model '", model_name, "'");
      }
    }

    if (v_idx != n) {
      return tensorflow::errors::InvalidArgument(
          "unexpected additional input data for model '", model_name, "'");
    }
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
HTTPInferRequestProvider::GetNextInputContent(
    int idx, const void** content, size_t* content_byte_size,
    bool force_contiguous)
{
  if ((idx < 0) || ((size_t)idx >= contents_.size())) {
    return tensorflow::errors::Internal("unexpected input index ", idx);
  }

  const size_t block_cnt = contents_[idx].size();
  const size_t block_idx = contents_idx_[idx];

  if (block_idx >= block_cnt) {
    *content = nullptr;
    *content_byte_size = 0;
  }
  // Return next block of data...
  else if (!force_contiguous || ((block_idx + 1) >= block_cnt)) {
    const auto& block = contents_[idx][block_idx];
    *content = block.first;
    *content_byte_size = block.second;
    contents_idx_[idx]++;
  }
  // If remaining data needs to be returned in one contiguous region
  // and there is more than one block remaining, then need to copy the
  // content into a single contiguous buffer.
  else {
    size_t total_size = 0;
    for (size_t i = block_idx; i < block_cnt; i++) {
      const auto& block = contents_[idx][i];
      total_size += block.second;
    }

    contiguous_buffers_.emplace_back();
    std::vector<char>& buf = contiguous_buffers_.back();
    buf.reserve(total_size);

    for (size_t i = block_idx; i < block_cnt; i++) {
      const auto& block = contents_[idx][i];
      buf.insert(buf.end(), block.first, block.first + block.second);
    }

    if (buf.size() != total_size) {
      return tensorflow::errors::Internal("contiguous input failed");
    }

    *content = &(buf[0]);
    *content_byte_size = total_size;
  }

  return tensorflow::Status::OK();
}


tensorflow::Status
GRPCInferResponseProvider::Create(
    const InferRequestHeader& request_header, InferResponse* response,
    std::shared_ptr<GRPCInferResponseProvider>* infer_provider)
{
  GRPCInferResponseProvider* provider =
      new GRPCInferResponseProvider(request_header, response);
  infer_provider->reset(provider);

  // Make space in the response for the output data. For outputs
  // returning raw tensor data we allocate space directly in the
  // response protobuf. For outputs returning classification we create
  // a buffer to hold the output that we can then post-process for
  // classifications.
  for (const auto& requested_output : request_header.output()) {
    std::string* output = provider->response_->add_raw_output();
    const size_t output_byte_size =
        request_header.batch_size() * requested_output.byte_size();

    if (requested_output.has_cls()) {
      provider->CreateOutputBuffer(output_byte_size);
    } else {
      // If byte-size for the output is zero, then the size is unknown
      // and we require that SetOutputBuffer be used to create and set
      // the buffer once the output size is known.
      if (output_byte_size == 0) {
        provider->AddOutputBuffer(nullptr, 0);
      } else {
        output->resize(output_byte_size);
        provider->AddOutputBuffer(
            static_cast<void*>(&((*output)[0])), output->size());
      }
    }
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
GRPCInferResponseProvider::SetOutputBuffer(
    int idx, const void* content, size_t content_byte_size)
{
  if ((idx < 0) || (idx >= response_->raw_output_size())) {
    return tensorflow::errors::Internal("unexpected output index ", idx);
  }

  std::string* output = response_->mutable_raw_output(idx);
  if (!output->empty()) {
    return tensorflow::errors::Internal(
        "buffer for output with non-fixed-size datatype already set");
  }

  *output =
      std::string(reinterpret_cast<const char*>(content), content_byte_size);

  return tensorflow::Status::OK();
}

tensorflow::Status
GRPCInferResponseProvider::FinalizeResponse(const InferenceServable& is)
{
  return FinalizeResponseHeader(is);
}

HTTPInferResponseProvider::HTTPInferResponseProvider(
    evbuffer* output_buffer, const InferRequestHeader& request_header)
    : InferResponseProvider(request_header), output_buffer_(output_buffer),
      non_fixed_size_output_buffers_(request_header.output().size())

{
  // Get the total size needed for raw output tensors that has a
  // fixed-sized datatype...
  total_raw_fixed_byte_size_ = 0;
  for (const auto& requested_output : request_header.output()) {
    if (!requested_output.has_cls()) {
      total_raw_fixed_byte_size_ +=
          request_header.batch_size() * requested_output.byte_size();
    }
  }
}

tensorflow::Status
HTTPInferResponseProvider::Create(
    evbuffer* output_buffer, const InferRequestHeader& request_header,
    std::shared_ptr<HTTPInferResponseProvider>* infer_provider)
{
  HTTPInferResponseProvider* provider =
      new HTTPInferResponseProvider(output_buffer, request_header);
  infer_provider->reset(provider);

  char* raw_output_base = nullptr;
  if (provider->total_raw_fixed_byte_size_ > 0) {
    // Reserve contiguous space in the output to hold all the raw output
    // tensor data that must be returned in the response.
    if (evbuffer_reserve_space(
            output_buffer, provider->total_raw_fixed_byte_size_,
            &provider->output_iovec_, 1) != 1) {
      return tensorflow::errors::Internal(
          "failed to reserve ", provider->total_raw_fixed_byte_size_,
          " bytes in output tensor buffer");
    }

    if (provider->output_iovec_.iov_len <
        provider->total_raw_fixed_byte_size_) {
      return tensorflow::errors::Internal(
          "reserved ", provider->output_iovec_.iov_len,
          " bytes in output tensor buffer, need ",
          provider->total_raw_fixed_byte_size_);
    }

    provider->output_iovec_.iov_len = provider->total_raw_fixed_byte_size_;
    raw_output_base = static_cast<char*>(provider->output_iovec_.iov_base);
  }

  // For outputs returning raw tensor data we allocate space directly
  // from the space reserved in 'output_buffer'. For outputs returning
  // classification we create a buffer to hold the output that we can
  // then post-process for classifications.
  bool seen_non_fixed_size = false;
  size_t raw_output_offset = 0;
  for (const auto& requested_output : request_header.output()) {
    const size_t output_byte_size =
        request_header.batch_size() * requested_output.byte_size();

    if (requested_output.has_cls()) {
      // Class output currently not supported for non-fixed-size types.
      if (output_byte_size == 0) {
        seen_non_fixed_size = true;
        return tensorflow::errors::InvalidArgument(
            "CLASS output not supported for unknown size output '",
            requested_output.name(), "'");
      }

      provider->CreateOutputBuffer(output_byte_size);
    } else {
      // If byte-size for the output is zero, then the size is unknown
      // and we require that SetOutputBuffer be used to create and set
      // the buffer once the output size is known.
      if (output_byte_size == 0) {
        provider->AddOutputBuffer(nullptr, 0);
        seen_non_fixed_size = true;
      } else {
        if (seen_non_fixed_size) {
          return tensorflow::errors::InvalidArgument(
              "HTTP API requires that output '", requested_output.name(),
              "' appear before any non-fixed-size outputs in the request "
              "header");
        }

        provider->AddOutputBuffer(
            static_cast<void*>(raw_output_base + raw_output_offset),
            output_byte_size);
        raw_output_offset += output_byte_size;
      }
    }
  }

  if (raw_output_offset != provider->total_raw_fixed_byte_size_) {
    return tensorflow::errors::Internal(
        "failed to partition ", provider->total_raw_fixed_byte_size_,
        " bytes across output tensor buffer");
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
HTTPInferResponseProvider::SetOutputBuffer(
    int idx, const void* content, size_t content_byte_size)
{
  if ((idx < 0) || (idx >= (int)non_fixed_size_output_buffers_.size())) {
    return tensorflow::errors::Internal("unexpected output index ", idx);
  }

  std::vector<uint8_t>& buf = non_fixed_size_output_buffers_[idx];
  buf.clear();
  std::copy(
      reinterpret_cast<const uint8_t*>(content),
      reinterpret_cast<const uint8_t*>(content) + content_byte_size,
      std::back_inserter(buf));

  return tensorflow::Status::OK();
}

tensorflow::Status
HTTPInferResponseProvider::FinalizeResponse(const InferenceServable& is)
{
  // Finalize the RAW outputs that have fixed-size datatype...
  if (total_raw_fixed_byte_size_ > 0) {
    if (evbuffer_commit_space(output_buffer_, &output_iovec_, 1) != 0) {
      return tensorflow::errors::Internal(
          "failed to commit output tensors to output buffer");
    }
  }

  // Add the RAW outputs that have non-fixed-size datatype...
  for (const auto& buf : non_fixed_size_output_buffers_) {
    if (!buf.empty()) {
      if (evbuffer_add(output_buffer_, &buf[0], buf.size()) != 0) {
        return tensorflow::errors::Internal(
            "failed to write ", buf.size(), " bytes to response buffer");
      }
    }
  }

  return FinalizeResponseHeader(is);
}

namespace {

template <typename T>
void
AddClassResults(
    InferResponseHeader::Output* poutput, void* poutput_buffer,
    const size_t batch_size, const InferRequestHeader::Output& output,
    const LabelProvider& label_provider)
{
  T* probs = reinterpret_cast<T*>(poutput_buffer);
  const size_t entry_cnt = (output.byte_size() / sizeof(T));
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

void
InferResponseProvider::CreateOutputBuffer(size_t byte_size)
{
  char* buffer = new char[byte_size];
  created_buffers_.emplace_back(buffer);
  buffers_.emplace_back(buffer, byte_size);
}

void
InferResponseProvider::AddOutputBuffer(void* buffer, size_t byte_size)
{
  buffers_.emplace_back(buffer, byte_size);
}

tensorflow::Status
InferResponseProvider::GetOutputBuffer(
    int idx, void** buffer, size_t buffer_byte_size)
{
  if ((idx < 0) || (idx >= (int)buffers_.size())) {
    return tensorflow::errors::Internal("unexpected output index ", idx);
  }

  if ((buffers_[idx].first == nullptr) ||
      (buffers_[idx].second != buffer_byte_size)) {
    return tensorflow::errors::Internal(
        "unexpected output size ", buffers_[idx].second);
  }

  *buffer = buffers_[idx].first;
  return tensorflow::Status::OK();
}

tensorflow::Status
InferResponseProvider::FinalizeResponseHeader(const InferenceServable& is)
{
  InferResponseHeader* response_header = MutableResponseHeader();
  response_header->Clear();

  const LabelProvider& label_provider = is.GetLabelProvider();

  response_header->set_model_name(is.Name());
  response_header->set_model_version(is.Version());

  const size_t batch_size = request_header_.batch_size();
  response_header->set_batch_size(batch_size);

  int output_idx = 0;
  for (const auto& output : request_header_.output()) {
    auto poutput = response_header->add_output();
    poutput->set_name(output.name());

    if (!output.has_cls()) {
      poutput->mutable_raw()->set_byte_size(output.byte_size());
    } else {
      void* output_buffer;
      TF_RETURN_IF_ERROR(GetOutputBuffer(
          output_idx, &output_buffer, batch_size * output.byte_size()));

      const ModelOutput* output_config;
      TF_RETURN_IF_ERROR(is.GetOutput(output.name(), &output_config));

      switch (output_config->data_type()) {
        case DataType::TYPE_UINT8:
          AddClassResults<uint8_t>(
              poutput, output_buffer, batch_size, output, label_provider);
          break;
        case DataType::TYPE_UINT16:
          AddClassResults<uint16_t>(
              poutput, output_buffer, batch_size, output, label_provider);
          break;
        case DataType::TYPE_UINT32:
          AddClassResults<uint32_t>(
              poutput, output_buffer, batch_size, output, label_provider);
          break;
        case DataType::TYPE_UINT64:
          AddClassResults<uint64_t>(
              poutput, output_buffer, batch_size, output, label_provider);
          break;

        case DataType::TYPE_INT8:
          AddClassResults<int8_t>(
              poutput, output_buffer, batch_size, output, label_provider);
          break;
        case DataType::TYPE_INT16:
          AddClassResults<int16_t>(
              poutput, output_buffer, batch_size, output, label_provider);
          break;
        case DataType::TYPE_INT32:
          AddClassResults<int32_t>(
              poutput, output_buffer, batch_size, output, label_provider);
          break;
        case DataType::TYPE_INT64:
          AddClassResults<int64_t>(
              poutput, output_buffer, batch_size, output, label_provider);
          break;

        case DataType::TYPE_FP32:
          AddClassResults<float>(
              poutput, output_buffer, batch_size, output, label_provider);
          break;
        case DataType::TYPE_FP64:
          AddClassResults<double>(
              poutput, output_buffer, batch_size, output, label_provider);
          break;

        default:
          return tensorflow::errors::InvalidArgument(
              "class result not available for output '", output.name(),
              "' due to unsupported type '",
              DataType_Name(output_config->data_type()), "'");
      }
    }

    output_idx++;
  }

  return tensorflow::Status::OK();
}

void
InferenceServable::GetMetricLabels(
    std::map<std::string, std::string>* labels, const int gpu_device) const
{
  labels->insert(std::map<std::string, std::string>::value_type(
      std::string(kMetricsLabelModelName), Name()));
  labels->insert(std::map<std::string, std::string>::value_type(
      std::string(kMetricsLabelModelVersion), std::to_string(Version())));
  for (const auto& tag : Tags()) {
    labels->insert(std::map<std::string, std::string>::value_type(
        "_" + tag.first, tag.second));
  }

  // 'gpu_device' can be -1 to indicate that the GPU is not known. In
  // that case use a metric that doesn't have the gpu_uuid label.
  if (gpu_device >= 0) {
    std::string uuid;
    if (Metrics::UUIDForCudaDevice(gpu_device, &uuid)) {
      labels->insert(std::map<std::string, std::string>::value_type(
          std::string(kMetricsLabelGpuUuid), uuid));
    }
  }
}

prometheus::Counter&
InferenceServable::GetCounterMetric(
    std::map<int, prometheus::Counter*>& metrics,
    prometheus::Family<prometheus::Counter>& family, const int gpu_device) const
{
  const auto itr = metrics.find(gpu_device);
  if (itr != metrics.end()) {
    return *(itr->second);
  }

  std::map<std::string, std::string> labels;
  GetMetricLabels(&labels, gpu_device);

  prometheus::Counter& counter = family.Add(labels);
  metrics.insert(
      std::map<int, prometheus::Counter*>::value_type(gpu_device, &counter));
  return counter;
}

prometheus::Counter&
InferenceServable::MetricInferenceSuccess(int gpu_device) const
{
  return GetCounterMetric(
      metric_inf_success_, Metrics::FamilyInferenceSuccess(), gpu_device);
}

prometheus::Counter&
InferenceServable::MetricInferenceFailure(int gpu_device) const
{
  return GetCounterMetric(
      metric_inf_failure_, Metrics::FamilyInferenceFailure(), gpu_device);
}

prometheus::Counter&
InferenceServable::MetricInferenceCount(int gpu_device) const
{
  return GetCounterMetric(
      metric_inf_count_, Metrics::FamilyInferenceCount(), gpu_device);
}

prometheus::Counter&
InferenceServable::MetricInferenceExecutionCount(int gpu_device) const
{
  return GetCounterMetric(
      metric_inf_exec_count_, Metrics::FamilyInferenceExecutionCount(),
      gpu_device);
}

prometheus::Counter&
InferenceServable::MetricInferenceRequestDuration(int gpu_device) const
{
  return GetCounterMetric(
      metric_inf_request_duration_us_,
      Metrics::FamilyInferenceRequestDuration(), gpu_device);
}

prometheus::Counter&
InferenceServable::MetricInferenceComputeDuration(int gpu_device) const
{
  return GetCounterMetric(
      metric_inf_compute_duration_us_,
      Metrics::FamilyInferenceComputeDuration(), gpu_device);
}

prometheus::Counter&
InferenceServable::MetricInferenceQueueDuration(int gpu_device) const
{
  return GetCounterMetric(
      metric_inf_queue_duration_us_, Metrics::FamilyInferenceQueueDuration(),
      gpu_device);
}

prometheus::Histogram&
InferenceServable::MetricInferenceLoadRatio(int gpu_device) const
{
  const auto itr = metric_inf_load_ratio_.find(gpu_device);
  if (itr != metric_inf_load_ratio_.end()) {
    return *(itr->second);
  }

  std::map<std::string, std::string> labels;
  GetMetricLabels(&labels, gpu_device);

  prometheus::Histogram& hist = Metrics::FamilyInferenceLoadRatio().Add(
      labels, std::vector<double>{1.05, 1.10, 1.25, 1.5, 2.0, 10.0, 50.0});
  metric_inf_load_ratio_.insert(
      std::map<int, prometheus::Histogram*>::value_type(gpu_device, &hist));
  return hist;
}

tensorflow::Status
InferenceServable::GetInput(
    const std::string& name, const ModelInput** input) const
{
  const auto itr = input_map_.find(name);
  if (itr == input_map_.end()) {
    return tensorflow::errors::InvalidArgument(
        "unexpected inference input '", name, "' for model '", Name(), "'");
  }

  *input = &itr->second;
  return tensorflow::Status::OK();
}

tensorflow::Status
InferenceServable::GetOutput(
    const std::string& name, const ModelOutput** output) const
{
  const auto itr = output_map_.find(name);
  if (itr == output_map_.end()) {
    return tensorflow::errors::InvalidArgument(
        "unexpected inference output '", name, "' for model '", Name(), "'");
  }

  *output = &itr->second;
  return tensorflow::Status::OK();
}

tensorflow::Status
InferenceServable::SetModelConfig(
    const tensorflow::StringPiece& path, const ModelConfig& config)
{
  config_ = config;
  TF_RETURN_IF_ERROR(GetModelVersionFromPath(path, &version_));
  for (const auto& tag : config_.tags()) {
    tags_.insert(
        std::map<std::string, std::string>::value_type(tag.first, tag.second));
  }

  // Initialize the input map
  for (const auto& io : config.input()) {
    input_map_.insert(std::make_pair(io.name(), io));
  }

  // Initialize the output map and label provider for each output
  const auto model_dir = tensorflow::io::Dirname(path);
  for (const auto& io : config.output()) {
    output_map_.insert(std::make_pair(io.name(), io));

    if (!io.label_filename().empty()) {
      const auto label_path =
          tensorflow::io::JoinPath(model_dir, io.label_filename());
      TF_RETURN_IF_ERROR(label_provider_.AddLabels(io.name(), label_path));
    }
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
InferenceServable::SetScheduler(std::unique_ptr<Scheduler> scheduler)
{
  if (scheduler_ != nullptr) {
    return tensorflow::errors::Internal(
        "Attempt to change scheduler not allowed");
  }

  scheduler_ = std::move(scheduler);
  return tensorflow::Status::OK();
}

tensorflow::Status
InferenceServable::SetConfiguredScheduler(
    const uint32_t runner_cnt, Scheduler::StandardRunFunc OnRun)
{
  std::unique_ptr<Scheduler> scheduler;

  // If 'sequence_batching' is configured use the SequenceBatchScheduler,
  // otherwise use the default DynamicBatchScheduler.
  if (config_.has_sequence_batching()) {
    scheduler.reset(new SequenceBatchScheduler(config_, runner_cnt, OnRun));
  } else {
    scheduler.reset(new DynamicBatchScheduler(config_, runner_cnt, OnRun));
  }

  return SetScheduler(std::move(scheduler));
}

void
InferenceServable::AsyncRun(
    std::shared_ptr<ModelInferStats> stats,
    std::shared_ptr<InferRequestProvider> request_provider,
    std::shared_ptr<InferResponseProvider> response_provider,
    std::function<void(tensorflow::Status)> OnCompleteHandleInfer)
{
  auto run_timer = std::make_shared<ModelInferStats::ScopedTimer>();
  stats->StartRunTimer(run_timer.get());

  scheduler_->Enqueue(
      stats, request_provider, response_provider,
      [OnCompleteHandleInfer, run_timer](tensorflow::Status status) mutable {
        run_timer.reset();
        OnCompleteHandleInfer(status);
      });
}

// Since callers are expecting synchronous behavior, this function
// must wait until the request is processed and the response is
// returned. This function can be simplified significantly once we
// have [DLIS-124].
void
InferenceServable::Run(
    std::shared_ptr<ModelInferStats> stats,
    std::shared_ptr<InferRequestProvider> request_provider,
    std::shared_ptr<InferResponseProvider> response_provider,
    std::function<void(tensorflow::Status)> OnCompleteHandleInfer)
{
  // Since this call is synchronous we can just use a scoped timer to
  // measure the entire run time.
  ModelInferStats::ScopedTimer run_timer;
  stats->StartRunTimer(&run_timer);

  std::mutex lmu;
  std::condition_variable lcv;
  tensorflow::Status run_status;
  bool run_completed = false;

  // Add request to queue...
  {
    scheduler_->Enqueue(
        stats, request_provider, response_provider,
        [&lmu, &lcv, &run_status, &run_completed](tensorflow::Status status) {
          // signal complete and propagate status
          {
            std::lock_guard<std::mutex> lk(lmu);
            run_status = status;
            run_completed = true;
          }
          lcv.notify_one();
        });
  }

  // [DLIS-124] must wait for request to indicate complete...
  {
    std::chrono::seconds wait_timeout(1);
    std::unique_lock<std::mutex> lk(lmu);
    while (!run_completed) {
      lcv.wait_for(lk, wait_timeout);
    }
  }

  OnCompleteHandleInfer(run_status);
}

}}  // namespace nvidia::inferenceserver
