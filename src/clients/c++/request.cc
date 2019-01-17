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

#include "src/clients/c++/request.h"

#include <curl/curl.h>
#include <google/protobuf/text_format.h>
#include <deque>
#include <iostream>
#include <memory>
#include "src/core/constants.h"

namespace nvidia { namespace inferenceserver { namespace client {

//==============================================================================

// Global initialization for libcurl. Libcurl requires global
// initialization before any other threads are created and before any
// curl methods are used. The curl_global static object is used to
// perform this initialization.
class CurlGlobal {
 public:
  CurlGlobal();
  ~CurlGlobal();

  const Error& Status() const { return err_; }

 private:
  Error err_;
};

CurlGlobal::CurlGlobal() : err_(RequestStatusCode::SUCCESS)
{
  if (curl_global_init(CURL_GLOBAL_ALL) != 0) {
    err_ = Error(RequestStatusCode::INTERNAL, "global initialization failed");
  }
}

CurlGlobal::~CurlGlobal()
{
  curl_global_cleanup();
}

static CurlGlobal curl_global;

//==============================================================================

template <>
Error
InferContext::Result::GetRawAtCursor(size_t batch_idx, std::string* out)
{
  Error err;

  const uint8_t* len_ptr;
  err = GetRawAtCursor(batch_idx, &len_ptr, sizeof(uint32_t));
  if (!err.IsOk()) {
    return err;
  }

  const uint32_t len = *(reinterpret_cast<const uint32_t*>(len_ptr));

  const uint8_t* str_ptr;
  err = GetRawAtCursor(batch_idx, &str_ptr, len);
  if (!err.IsOk()) {
    return err;
  }

  out->clear();
  std::copy(str_ptr, str_ptr + len, std::back_inserter(*out));

  return Error::Success;
}

//==============================================================================

// Use map to keep track of GRPC channels. <key, value> : <url, Channel*>
// If context is created on url that has established Channel, then reuse it.
std::map<std::string, std::shared_ptr<grpc::Channel>> grpc_channel_map_;
std::shared_ptr<grpc::Channel>
GetChannel(const std::string& url)
{
  const auto& channel_itr = grpc_channel_map_.find(url);
  if (channel_itr != grpc_channel_map_.end()) {
    return channel_itr->second;
  } else {
    grpc::ChannelArguments arguments;
    arguments.SetMaxSendMessageSize(MAX_GRPC_MESSAGE_SIZE);
    arguments.SetMaxReceiveMessageSize(MAX_GRPC_MESSAGE_SIZE);
    std::shared_ptr<grpc::Channel> channel = grpc::CreateCustomChannel(
        url, grpc::InsecureChannelCredentials(), arguments);
    grpc_channel_map_.insert(std::make_pair(url, channel));
    return channel;
  }
}

//==============================================================================

const Error Error::Success(RequestStatusCode::SUCCESS);

Error::Error(RequestStatusCode code, const std::string& msg)
    : code_(code), msg_(msg), request_id_(0)
{
}

Error::Error(RequestStatusCode code) : code_(code), request_id_(0) {}

Error::Error(const RequestStatus& status) : Error(status.code(), status.msg())
{
  server_id_ = status.server_id();
  request_id_ = status.request_id();
}

std::ostream&
operator<<(std::ostream& out, const Error& err)
{
  out << "[" << err.server_id_ << " " << err.request_id_ << "] "
      << RequestStatusCode_Name(err.code_);
  if (!err.msg_.empty()) {
    out << " - " << err.msg_;
  }
  return out;
}

//==============================================================================

ServerHealthContext::ServerHealthContext(bool verbose) : verbose_(verbose) {}

//==============================================================================

ServerStatusContext::ServerStatusContext(bool verbose) : verbose_(verbose) {}

//==============================================================================

class OptionsImpl : public InferContext::Options {
 public:
  OptionsImpl();
  ~OptionsImpl() = default;

  size_t BatchSize() const override { return batch_size_; }
  void SetBatchSize(size_t batch_size) override { batch_size_ = batch_size; }

  Error AddRawResult(
      const std::shared_ptr<InferContext::Output>& output) override;
  Error AddClassResult(
      const std::shared_ptr<InferContext::Output>& output, uint64_t k) override;

  // Options for an output
  struct OutputOptions {
    OutputOptions(InferContext::Result::ResultFormat f, uint64_t n = 0)
        : result_format(f), u64(n)
    {
    }
    InferContext::Result::ResultFormat result_format;
    uint64_t u64;
  };

  using OutputOptionsPair =
      std::pair<std::shared_ptr<InferContext::Output>, OutputOptions>;

  const std::deque<OutputOptionsPair>& Outputs() const { return outputs_; }

 private:
  size_t batch_size_;
  std::deque<OutputOptionsPair> outputs_;
};

OptionsImpl::OptionsImpl() : batch_size_(0) {}

Error
OptionsImpl::AddRawResult(const std::shared_ptr<InferContext::Output>& output)
{
  // HTTP protocol requires that outputs with non-fixed-size datatype
  // (STRING) be requested last in the request header...
  if (!IsFixedSizeDataType(output->DType())) {
    outputs_.emplace_back(std::make_pair(
        output, OutputOptions(InferContext::Result::ResultFormat::RAW)));
  } else {
    outputs_.emplace_front(std::make_pair(
        output, OutputOptions(InferContext::Result::ResultFormat::RAW)));
  }

  return Error::Success;
}

Error
OptionsImpl::AddClassResult(
    const std::shared_ptr<InferContext::Output>& output, uint64_t k)
{
  // HTTP protocol requires that outputs with non-fixed-size datatype
  // (STRING) be requested last in the request header...
  if (!IsFixedSizeDataType(output->DType())) {
    outputs_.emplace_back(std::make_pair(
        output, OutputOptions(InferContext::Result::ResultFormat::CLASS, k)));
  } else {
    outputs_.emplace_front(std::make_pair(
        output, OutputOptions(InferContext::Result::ResultFormat::CLASS, k)));
  }

  return Error::Success;
}

Error
InferContext::Options::Create(std::unique_ptr<InferContext::Options>* options)
{
  options->reset(new OptionsImpl());
  return Error::Success;
}

//==============================================================================

class InputImpl : public InferContext::Input {
 public:
  InputImpl(const ModelInput& mio);
  InputImpl(const InputImpl& obj);
  ~InputImpl() = default;

  const std::string& Name() const override { return mio_.name(); }
  int64_t ByteSize() const override { return byte_size_; }
  size_t TotalByteSize() const override { return total_byte_size_; }
  DataType DType() const override { return mio_.data_type(); }
  ModelInput::Format Format() const override { return mio_.format(); }
  const DimsList& Dims() const override { return mio_.dims(); }

  void SetBatchSize(size_t batch_size) { batch_size_ = batch_size; }

  const std::vector<int64_t>& Shape() const override { return shape_; }
  Error SetShape(const std::vector<int64_t>& dims) override;

  Error Reset() override;
  Error SetRaw(const std::vector<uint8_t>& input) override;
  Error SetRaw(const uint8_t* input, size_t input_byte_size) override;
  Error SetFromString(const std::vector<std::string>& input) override;

  // Copy into 'buf' up to 'size' bytes of this input's data. Return
  // the actual amount copied in 'input_bytes' and if the end of input
  // is reached in 'end_of_input'
  Error GetNext(
      uint8_t* buf, size_t size, size_t* input_bytes, bool* end_of_input);

  // Copy the pointer of the raw buffer at 'batch_idx' into 'buf'
  Error GetRaw(size_t batch_idx, const uint8_t** buf, size_t* byte_size) const;

  // Prepare to send this input as part of a request.
  Error PrepareForRequest();

 private:
  const ModelInput mio_;

  int64_t byte_size_;
  size_t total_byte_size_;

  bool needs_shape_;
  std::vector<int64_t> shape_;

  size_t batch_size_;
  size_t bufs_idx_, buf_pos_;
  std::vector<const uint8_t*> bufs_;
  std::vector<size_t> buf_byte_sizes_;

  // Used only for STRING type tensors set with SetFromString(). Hold
  // the "raw" serialization of the string values for each batch index
  // that are then referenced by 'bufs_'. A std::list is used to avoid
  // reallocs that could invalidate the pointer references into the
  // std::string objects.
  std::list<std::string> str_bufs_;
};

InputImpl::InputImpl(const ModelInput& mio)
    : mio_(mio), total_byte_size_(0), needs_shape_(false), batch_size_(0),
      bufs_idx_(0), buf_pos_(0)
{
  if (GetElementCount(mio) == -1) {
    byte_size_ = -1;
    needs_shape_ = true;
  } else {
    byte_size_ = GetByteSize(mio);
    if (byte_size_ == 0) {
      byte_size_ = -1;
    }
  }
}

InputImpl::InputImpl(const InputImpl& obj)
    : mio_(obj.mio_), byte_size_(obj.byte_size_),
      total_byte_size_(obj.total_byte_size_), needs_shape_(obj.needs_shape_),
      shape_(obj.shape_), batch_size_(obj.batch_size_), bufs_idx_(0),
      buf_pos_(0), bufs_(obj.bufs_), buf_byte_sizes_(obj.buf_byte_sizes_),
      str_bufs_(obj.str_bufs_)
{
}

Error
InputImpl::SetShape(const std::vector<int64_t>& dims)
{
  // Make sure the shape does not contain any invalid dimensions
  for (const auto dim : dims) {
    if (dim < 1) {
      return Error(
          RequestStatusCode::INVALID_ARG,
          "attempt to set invalid shape dimension " + std::to_string(dim) +
              ", shape dimensions must be >= 1 for input '" + Name());
    }
  }

  needs_shape_ = false;
  shape_ = dims;

  byte_size_ = GetByteSize(DType(), dims);
  if (byte_size_ == 0) {
    byte_size_ = -1;
  }

  return Error::Success;
}

Error
InputImpl::SetRaw(const uint8_t* input, size_t input_byte_size)
{
  if (needs_shape_) {
    bufs_.clear();
    buf_byte_sizes_.clear();
    str_bufs_.clear();
    return Error(
        RequestStatusCode::INVALID_ARG,
        "must set shape for variable-size input '" + Name() +
            "' before setting input data");
  }

  if (IsFixedSizeDataType(DType()) && (input_byte_size != (size_t)byte_size_)) {
    bufs_.clear();
    buf_byte_sizes_.clear();
    str_bufs_.clear();
    return Error(
        RequestStatusCode::INVALID_ARG,
        "invalid size " + std::to_string(input_byte_size) +
            " bytes for input '" + Name() + "', expects " +
            std::to_string(byte_size_) + " bytes");
  }

  if (bufs_.size() >= batch_size_) {
    bufs_.clear();
    buf_byte_sizes_.clear();
    str_bufs_.clear();
    return Error(
        RequestStatusCode::INVALID_ARG,
        "expecting " + std::to_string(batch_size_) +
            " invocations of SetRaw for input '" + Name() +
            "', one per batch entry");
  }

  total_byte_size_ += input_byte_size;

  bufs_.push_back(input);
  buf_byte_sizes_.push_back(input_byte_size);

  return Error::Success;
}

Error
InputImpl::SetRaw(const std::vector<uint8_t>& input)
{
  return SetRaw(&input[0], input.size());
}

Error
InputImpl::SetFromString(const std::vector<std::string>& input)
{
  if (DType() != DataType::TYPE_STRING) {
    bufs_.clear();
    buf_byte_sizes_.clear();
    str_bufs_.clear();
    return Error(
        RequestStatusCode::INVALID_ARG,
        "non-string tensor '" + Name() + "' cannot be set from string data");
  }

  if (needs_shape_) {
    return Error(
        RequestStatusCode::INVALID_ARG,
        "must set shape for variable-size input '" + Name() +
            "' before setting input data");
  }

  const int64_t element_count =
      (!shape_.empty()) ? GetElementCount(shape_) : GetElementCount(mio_);

  if (input.size() != (size_t)element_count) {
    bufs_.clear();
    buf_byte_sizes_.clear();
    str_bufs_.clear();
    return Error(
        RequestStatusCode::INVALID_ARG,
        "expecting " + std::to_string(element_count) + " strings for input '" +
            Name() + "', got " + std::to_string(input.size()));
  }

  // Serialize the strings into a "raw" buffer. The first 4-bytes are
  // the length of the string length. Next are the actual string
  // characters. There is *not* a null-terminator on the string.
  str_bufs_.emplace_back();
  std::string& sbuf = str_bufs_.back();
  for (const auto& str : input) {
    uint32_t len = str.size();
    sbuf.append(reinterpret_cast<const char*>(&len), sizeof(uint32_t));
    sbuf.append(str);
  }

  return SetRaw(reinterpret_cast<const uint8_t*>(&sbuf[0]), sbuf.size());
}

Error
InputImpl::GetNext(
    uint8_t* buf, size_t size, size_t* input_bytes, bool* end_of_input)
{
  size_t total_size = 0;

  while ((bufs_idx_ < bufs_.size()) && (size > 0)) {
    const size_t buf_byte_size = buf_byte_sizes_[bufs_idx_];
    const size_t csz = std::min(buf_byte_size - buf_pos_, size);
    if (csz > 0) {
      const uint8_t* input_ptr = bufs_[bufs_idx_] + buf_pos_;
      std::copy(input_ptr, input_ptr + csz, buf);
      buf_pos_ += csz;
      buf += csz;
      size -= csz;
      total_size += csz;
    }

    if (buf_pos_ == buf_byte_size) {
      bufs_idx_++;
      buf_pos_ = 0;
    }
  }

  *input_bytes = total_size;
  *end_of_input = (bufs_idx_ >= bufs_.size());
  return Error::Success;
}

Error
InputImpl::GetRaw(
    size_t batch_idx, const uint8_t** buf, size_t* byte_size) const
{
  if (batch_idx >= batch_size_) {
    return Error(
        RequestStatusCode::INVALID_ARG,
        "unexpected batch entry " + std::to_string(batch_idx) +
            " requested for input '" + Name() + "', batch size is " +
            std::to_string(batch_size_));
  }

  *buf = bufs_[batch_idx];
  *byte_size = buf_byte_sizes_[batch_idx];

  return Error::Success;
}

Error
InputImpl::Reset()
{
  bufs_.clear();
  buf_byte_sizes_.clear();
  str_bufs_.clear();
  bufs_idx_ = 0;
  buf_pos_ = 0;
  total_byte_size_ = 0;

  return Error::Success;
}

Error
InputImpl::PrepareForRequest()
{
  if (bufs_.size() != batch_size_) {
    return Error(
        RequestStatusCode::INVALID_ARG,
        "expecting " + std::to_string(batch_size_) +
            " invocations of SetRaw for input '" + Name() + "', have " +
            std::to_string(bufs_.size()));
  }

  // Reset position so request sends entire input.
  bufs_idx_ = 0;
  buf_pos_ = 0;
  return Error::Success;
}

//==============================================================================

class OutputImpl : public InferContext::Output {
 public:
  OutputImpl(const ModelOutput& mio);
  ~OutputImpl() = default;

  const std::string& Name() const override { return mio_.name(); }
  size_t ByteSize() const override { return byte_size_; }
  DataType DType() const override { return mio_.data_type(); }
  const DimsList& Dims() const override { return mio_.dims(); }

  InferContext::Result::ResultFormat ResultFormat() const
  {
    return result_format_;
  }
  void SetResultFormat(InferContext::Result::ResultFormat result_format)
  {
    result_format_ = result_format;
  }

 private:
  const ModelOutput mio_;
  const size_t byte_size_;
  InferContext::Result::ResultFormat result_format_;
};

OutputImpl::OutputImpl(const ModelOutput& mio)
    : mio_(mio), byte_size_(GetByteSize(mio)),
      result_format_(InferContext::Result::ResultFormat::RAW)
{
}

//==============================================================================

class ResultImpl : public InferContext::Result {
 public:
  ResultImpl(
      const std::shared_ptr<InferContext::Output>& output, uint64_t batch_size);
  ~ResultImpl() = default;

  const std::string& ModelName() const override { return model_name_; }
  int64_t ModelVersion() const override { return model_version_; }

  const std::shared_ptr<InferContext::Output> GetOutput() const override
  {
    return output_;
  }

  Error GetRaw(
      size_t batch_idx, const std::vector<uint8_t>** buf) const override;
  Error GetRawAtCursor(
      size_t batch_idx, const uint8_t** buf, size_t adv_byte_size) override;
  Error GetClassCount(size_t batch_idx, size_t* cnt) const override;
  Error GetClassAtCursor(size_t batch_idx, ClassResult* result) override;
  Error ResetCursors() override;
  Error ResetCursor(size_t batch_idx) override;

  // Get the result format for this result.
  InferContext::Result::ResultFormat ResultFormat() const
  {
    return result_format_;
  }

  // Set information about the model that produced this result.
  void SetModel(const std::string& name, const int64_t version)
  {
    model_name_ = name;
    model_version_ = version;
  }

  // Set results for a CLASS format result.
  void SetClassResult(const InferResponseHeader::Output& result)
  {
    class_result_ = result;
  }

  // For RAW format result, copy into the output up to 'size' bytes of
  // output data from 'buf'. Return the actual amount copied in
  // 'result_bytes'.
  Error SetNextRawResult(const uint8_t* buf, size_t size, size_t* result_bytes);

 private:
  Error SetBatchRawResult(
      const size_t batch1_byte_size, const uint8_t* buf, size_t size,
      size_t* result_bytes);

  const std::shared_ptr<InferContext::Output> output_;
  const size_t batch1_element_count_;
  const InferContext::Result::ResultFormat result_format_;
  const size_t batch_size_;

  std::vector<std::vector<uint8_t>> bufs_;
  size_t bufs_idx_;
  std::vector<size_t> bufs_pos_;
  std::vector<size_t> bufs_byte_size_;
  std::vector<uint8_t> pending_;

  std::string model_name_;
  int64_t model_version_;

  InferResponseHeader::Output class_result_;
  std::vector<size_t> class_pos_;
};

ResultImpl::ResultImpl(
    const std::shared_ptr<InferContext::Output>& output, uint64_t batch_size)
    : output_(output), batch1_element_count_(GetElementCount(output->Dims())),
      result_format_(
          reinterpret_cast<OutputImpl*>(output.get())->ResultFormat()),
      batch_size_(batch_size), bufs_(batch_size), bufs_idx_(0),
      bufs_pos_(batch_size), bufs_byte_size_(batch_size), class_pos_(batch_size)
{
}

Error
ResultImpl::GetRaw(size_t batch_idx, const std::vector<uint8_t>** buf) const
{
  if (result_format_ != InferContext::Result::ResultFormat::RAW) {
    return Error(
        RequestStatusCode::UNSUPPORTED,
        "raw result not available for non-RAW output '" + output_->Name() +
            "'");
  }

  if (batch_idx >= batch_size_) {
    return Error(
        RequestStatusCode::INVALID_ARG,
        "unexpected batch entry " + std::to_string(batch_idx) +
            " requested for output '" + output_->Name() + "', batch size is " +
            std::to_string(batch_size_));
  }

  *buf = &bufs_[batch_idx];
  return Error::Success;
}

Error
ResultImpl::GetRawAtCursor(
    size_t batch_idx, const uint8_t** buf, size_t adv_byte_size)
{
  if (result_format_ != InferContext::Result::ResultFormat::RAW) {
    return Error(
        RequestStatusCode::UNSUPPORTED,
        "raw result not available for non-RAW output '" + output_->Name() +
            "'");
  }

  if (batch_idx >= batch_size_) {
    return Error(
        RequestStatusCode::INVALID_ARG,
        "unexpected batch entry " + std::to_string(batch_idx) +
            "requested for output '" + output_->Name() + "', batch size is " +
            std::to_string(batch_size_));
  }

  if ((bufs_pos_[batch_idx] + adv_byte_size) > bufs_byte_size_[batch_idx]) {
    return Error(
        RequestStatusCode::UNSUPPORTED,
        "attempt to read beyond end of result for output '" + output_->Name() +
            "'");
  }

  *buf = &bufs_[batch_idx][bufs_pos_[batch_idx]];
  bufs_pos_[batch_idx] += adv_byte_size;
  return Error::Success;
}

Error
ResultImpl::GetClassCount(size_t batch_idx, size_t* cnt) const
{
  if (result_format_ != InferContext::Result::ResultFormat::CLASS) {
    return Error(
        RequestStatusCode::UNSUPPORTED,
        "class result not available for non-CLASS output '" + output_->Name() +
            "'");
  }

  // Number of classifications should equal expected batch size but
  // check both to be careful and to protext class_pos_ accesses.
  if ((batch_idx >= (size_t)class_result_.batch_classes().size()) ||
      (batch_idx >= batch_size_)) {
    return Error(
        RequestStatusCode::INVALID_ARG,
        "unexpected batch entry " + std::to_string(batch_idx) +
            "requested for output '" + output_->Name() + "', batch size is " +
            std::to_string(batch_size_));
  }

  const InferResponseHeader::Output::Classes& classes =
      class_result_.batch_classes(batch_idx);

  *cnt = classes.cls().size();
  return Error::Success;
}

Error
ResultImpl::GetClassAtCursor(
    size_t batch_idx, InferContext::Result::ClassResult* result)
{
  if (result_format_ != InferContext::Result::ResultFormat::CLASS) {
    return Error(
        RequestStatusCode::UNSUPPORTED,
        "class result not available for non-CLASS output '" + output_->Name() +
            "'");
  }

  // Number of classifications should equal expected batch size but
  // check both to be careful and to protext class_pos_ accesses.
  if ((batch_idx >= (size_t)class_result_.batch_classes().size()) ||
      (batch_idx >= batch_size_)) {
    return Error(
        RequestStatusCode::INVALID_ARG,
        "unexpected batch entry " + std::to_string(batch_idx) +
            "requested for output '" + output_->Name() + "', batch size is " +
            std::to_string(batch_size_));
  }

  const InferResponseHeader::Output::Classes& classes =
      class_result_.batch_classes(batch_idx);

  if (class_pos_[batch_idx] >= (size_t)classes.cls().size()) {
    return Error(
        RequestStatusCode::UNSUPPORTED,
        "attempt to read beyond end of result for output output '" +
            output_->Name() + "'");
  }

  const InferResponseHeader::Output::Class& cls =
      classes.cls(class_pos_[batch_idx]);

  result->idx = cls.idx();
  result->value = cls.value();
  result->label = cls.label();

  class_pos_[batch_idx]++;
  return Error::Success;
}

Error
ResultImpl::ResetCursors()
{
  std::fill(bufs_pos_.begin(), bufs_pos_.end(), 0);
  std::fill(class_pos_.begin(), class_pos_.end(), 0);
  return Error::Success;
}

Error
ResultImpl::ResetCursor(size_t batch_idx)
{
  if (batch_idx >= batch_size_) {
    return Error(
        RequestStatusCode::INVALID_ARG,
        "unexpected batch entry " + std::to_string(batch_idx) +
            "requested for output '" + output_->Name() + "', batch size is " +
            std::to_string(batch_size_));
  }

  bufs_pos_[batch_idx] = 0;
  class_pos_[batch_idx] = 0;
  return Error::Success;
}

Error
ResultImpl::SetBatchRawResult(
    const size_t batch1_byte_size, const uint8_t* buf, size_t size,
    size_t* result_bytes)
{
  size_t total_size = 0;

  while ((bufs_idx_ < bufs_.size()) && (size > 0)) {
    const size_t csz = std::min(batch1_byte_size - bufs_pos_[bufs_idx_], size);
    if (csz > 0) {
      std::copy(buf, buf + csz, std::back_inserter(bufs_[bufs_idx_]));
      bufs_pos_[bufs_idx_] += csz;
      bufs_byte_size_[bufs_idx_] += csz;
      buf += csz;
      size -= csz;
      total_size += csz;
    }

    if (bufs_pos_[bufs_idx_] == batch1_byte_size) {
      bufs_idx_++;
    }
  }

  *result_bytes = total_size;
  return Error::Success;
}

Error
ResultImpl::SetNextRawResult(
    const uint8_t* buf, size_t size, size_t* result_bytes)
{
  // If output has non-zero byte-size then it is an output with a
  // fixed-sized datatype and can directly assign the results to the
  // appropriate per-batch buffers.
  if (output_->ByteSize() > 0) {
    return SetBatchRawResult(output_->ByteSize(), buf, size, result_bytes);
  }

  // Output is a non-fixed-sized datatype. For now we assume that it
  // is TYPE_STRING and so we need to parse buf to get the size for
  // each batch (since 'batch1_element_count_' entries which go into a
  // batch).
  if (bufs_idx_ == bufs_.size()) {
    *result_bytes = 0;
    return Error::Success;
  }

  // If there is partial batch data then append the new data to it and
  // parse it all together.
  const size_t orig_size = size;
  if (!pending_.empty()) {
    std::copy(buf, buf + size, std::back_inserter(pending_));
    buf = &pending_[0];
    size = pending_.size();
  }

  std::vector<size_t> batch_byte_sizes;
  size_t buf_offset = 0;
  for (size_t b = 0; b < batch_size_; ++b) {
    size_t batch_byte_size = 0;
    for (size_t e = 0; e < batch1_element_count_; ++e) {
      const uint32_t len =
          *(reinterpret_cast<const uint32_t*>(buf + buf_offset));
      batch_byte_size += sizeof(len) + len;
      buf_offset += sizeof(len) + len;
      if (buf_offset > size) {
        break;
      }
    }

    if (buf_offset > size) {
      break;
    }

    batch_byte_sizes.push_back(batch_byte_size);
  }

  // Don't assign batches until we have entire batch of tensor data.
  if (batch_byte_sizes.size() < batch_size_) {
    if (pending_.empty()) {
      std::copy(buf, buf + orig_size, std::back_inserter(pending_));
    }
    *result_bytes = orig_size;
  } else {
    for (size_t sz : batch_byte_sizes) {
      size_t batch1_result_bytes = 0;
      Error err = SetBatchRawResult(sz, buf, sz, &batch1_result_bytes);
      if (!err.IsOk()) {
        return err;
      }
      if (batch1_result_bytes != sz) {
        return Error(
            RequestStatusCode::INTERNAL,
            "output '" + output_->Name() + "' expecting batch size " +
                std::to_string(sz) + " for non-fixed-sized result, got " +
                std::to_string(batch1_result_bytes));
      }

      buf += sz;
    }

    if (bufs_idx_ != bufs_.size()) {
      return Error(
          RequestStatusCode::INTERNAL,
          "output '" + output_->Name() +
              "' failed to set result for entire batch");
    }

    pending_.clear();
    *result_bytes = orig_size - (size - buf_offset);
  }

  return Error::Success;
}

//==============================================================================

InferContext::RequestTimers::RequestTimers()
{
  Reset();
}

Error
InferContext::RequestTimers::Reset()
{
  request_start_.tv_sec = 0;
  request_end_.tv_sec = 0;
  send_start_.tv_sec = 0;
  send_end_.tv_sec = 0;
  receive_start_.tv_sec = 0;
  receive_end_.tv_sec = 0;
  request_start_.tv_nsec = 0;
  request_end_.tv_nsec = 0;
  send_start_.tv_nsec = 0;
  send_end_.tv_nsec = 0;
  receive_start_.tv_nsec = 0;
  receive_end_.tv_nsec = 0;
  return Error::Success;
}

Error
InferContext::RequestTimers::Record(Kind kind)
{
  switch (kind) {
    case Kind::REQUEST_START:
      clock_gettime(CLOCK_MONOTONIC, &request_start_);
      break;
    case Kind::REQUEST_END:
      clock_gettime(CLOCK_MONOTONIC, &request_end_);
      break;
    case Kind::SEND_START:
      clock_gettime(CLOCK_MONOTONIC, &send_start_);
      break;
    case Kind::SEND_END:
      clock_gettime(CLOCK_MONOTONIC, &send_end_);
      break;
    case Kind::RECEIVE_START:
      clock_gettime(CLOCK_MONOTONIC, &receive_start_);
      break;
    case Kind::RECEIVE_END:
      clock_gettime(CLOCK_MONOTONIC, &receive_end_);
      break;
  }
  return Error::Success;
}

//==============================================================================

class RequestImpl : public InferContext::Request {
 public:
  virtual ~RequestImpl() = default;

  uint64_t Id() const { return id_; };

  // Initialize 'requested_results_' according to 'batch_size' and
  // 'requested_outs' as the placeholder for the results
  Error InitializeRequestedResults(
      const std::vector<std::shared_ptr<InferContext::Output>>& requested_outs,
      const size_t batch_size);

  // Return the results of the request. 'ready_' should always be checked
  // before calling GetResults() to ensure the request has been completed.
  virtual Error GetResults(
      std::map<std::string, std::unique_ptr<InferContext::Result>>*
          results) = 0;

 protected:
  RequestImpl(const uint64_t id);

  // Helper function called after inference to set non-RAW results in
  // 'requested_results_'.
  Error PostRunProcessing(
      std::vector<std::unique_ptr<InferContext::Result>>& results,
      const InferResponseHeader& infer_response);

  friend class InferContext;

  // Identifier seen by user
  uint64_t id_;

  // Internal identifier for asynchronous call
  uintptr_t run_index_;

  // Indicating if the request has been completed.
  bool ready_;

  // The timer for infer request.
  InferContext::RequestTimers timer_;

  // Results being collected for the requested outputs from inference
  // server response. Ordered in a vector as the HTTP API requires
  // ordering to associate results correctly.
  std::vector<std::unique_ptr<InferContext::Result>> requested_results_;

  // Current positions within output vectors when processing response.
  size_t result_pos_idx_;
};

RequestImpl::RequestImpl(const uint64_t id)
    : id_(id), ready_(false), result_pos_idx_(0)
{
}

Error
RequestImpl::InitializeRequestedResults(
    const std::vector<std::shared_ptr<InferContext::Output>>& requested_outs,
    const size_t batch_size)
{
  // Initialize the results vector to collect the requested results.
  requested_results_.clear();
  for (const auto& io : requested_outs) {
    std::unique_ptr<ResultImpl> rp(new ResultImpl(io, batch_size));
    requested_results_.emplace_back(std::move(rp));
  }
  return Error::Success;
}

Error
RequestImpl::PostRunProcessing(
    std::vector<std::unique_ptr<InferContext::Result>>& results,
    const InferResponseHeader& infer_response)
{
  // At this point, the RAW requested results have their result values
  // set. Now need to initialize non-RAW results.
  for (auto& rr : results) {
    ResultImpl* r = reinterpret_cast<ResultImpl*>(rr.get());
    r->SetModel(infer_response.model_name(), infer_response.model_version());
    switch (r->ResultFormat()) {
      case InferContext::Result::ResultFormat::RAW:
        r->ResetCursors();
        break;

      case InferContext::Result::ResultFormat::CLASS: {
        for (const auto& ir : infer_response.output()) {
          if (ir.name() == r->GetOutput()->Name()) {
            r->SetClassResult(ir);
            break;
          }
        }
        break;
      }
    }
  }
  return Error::Success;
}

//==============================================================================

InferContext::InferContext(
    const std::string& model_name, int64_t model_version,
    CorrelationID correlation_id, bool verbose)
    : model_name_(model_name), model_version_(model_version),
      correlation_id_(correlation_id), verbose_(verbose), batch_size_(0),
      async_request_id_(0), worker_(), exiting_(true)
{
}

Error
InferContext::GetInput(
    const std::string& name, std::shared_ptr<Input>* input) const
{
  for (const auto& io : inputs_) {
    if (io->Name() == name) {
      *input = io;
      return Error::Success;
    }
  }

  return Error(
      RequestStatusCode::INVALID_ARG,
      "unknown input '" + name + "' for '" + model_name_ + "'");
}

Error
InferContext::GetOutput(
    const std::string& name, std::shared_ptr<Output>* output) const
{
  for (const auto& io : outputs_) {
    if (io->Name() == name) {
      *output = io;
      return Error::Success;
    }
  }

  return Error(
      RequestStatusCode::INVALID_ARG,
      "unknown output '" + name + "' for '" + model_name_ + "'");
}

Error
InferContext::SetRunOptions(const InferContext::Options& boptions)
{
  const OptionsImpl& options = reinterpret_cast<const OptionsImpl&>(boptions);

  // If the model doesn't support batching (i.e. max_batch_size_ == 0)
  // then still allow batch size of 1 to be specified.
  uint64_t effective_max_batch_size = std::max((uint64_t)1, max_batch_size_);
  if (options.BatchSize() > effective_max_batch_size) {
    return Error(
        RequestStatusCode::INVALID_ARG,
        "run batch-size " + std::to_string(options.BatchSize()) +
            " exceeds maximum batch size " +
            std::to_string(effective_max_batch_size) + " allowed for model '" +
            model_name_ + "'");
  }

  // If batch-size 0 was requested (no batching) treat it like
  // batch-size 1.
  batch_size_ = std::max((uint64_t)1, options.BatchSize());

  // Create the InferRequestHeader protobuf. This protobuf will be
  // used for all subsequent requests.
  infer_request_.Clear();
  infer_request_.set_batch_size(batch_size_);
  infer_request_.set_correlation_id(correlation_id_);

  for (const auto& io : inputs_) {
    reinterpret_cast<InputImpl*>(io.get())->SetBatchSize(batch_size_);
  }

  requested_outputs_.clear();

  for (const auto& p : options.Outputs()) {
    const std::shared_ptr<Output>& output = p.first;
    const OptionsImpl::OutputOptions& ooptions = p.second;

    reinterpret_cast<OutputImpl*>(output.get())
        ->SetResultFormat(ooptions.result_format);
    requested_outputs_.emplace_back(output);

    auto routput = infer_request_.add_output();
    routput->set_name(output->Name());
    routput->set_byte_size(output->ByteSize());
    if (ooptions.result_format == Result::ResultFormat::CLASS) {
      routput->mutable_cls()->set_count(ooptions.u64);
    }
  }

  return Error::Success;
}

Error
InferContext::GetStat(Stat* stat)
{
  stat->completed_request_count = context_stat_.completed_request_count;
  stat->cumulative_total_request_time_ns =
      context_stat_.cumulative_total_request_time_ns;
  stat->cumulative_send_time_ns = context_stat_.cumulative_send_time_ns;
  stat->cumulative_receive_time_ns = context_stat_.cumulative_receive_time_ns;
  return Error::Success;
}

Error
InferContext::UpdateStat(const RequestTimers& timer)
{
  uint64_t request_start_ns = timer.request_start_.tv_sec * NANOS_PER_SECOND +
                              timer.request_start_.tv_nsec;
  uint64_t request_end_ns =
      timer.request_end_.tv_sec * NANOS_PER_SECOND + timer.request_end_.tv_nsec;
  uint64_t send_start_ns =
      timer.send_start_.tv_sec * NANOS_PER_SECOND + timer.send_start_.tv_nsec;
  uint64_t send_end_ns =
      timer.send_end_.tv_sec * NANOS_PER_SECOND + timer.send_end_.tv_nsec;
  uint64_t receive_start_ns = timer.receive_start_.tv_sec * NANOS_PER_SECOND +
                              timer.receive_start_.tv_nsec;
  uint64_t receive_end_ns =
      timer.receive_end_.tv_sec * NANOS_PER_SECOND + timer.receive_end_.tv_nsec;
  if ((request_start_ns >= request_end_ns) || (send_start_ns > send_end_ns) ||
      (receive_start_ns > receive_end_ns)) {
    return Error(RequestStatusCode::INVALID_ARG, "Timer not set correctly.");
  }

  uint64_t request_time_ns = request_end_ns - request_start_ns;
  uint64_t send_time_ns = send_end_ns - send_start_ns;
  uint64_t receive_time_ns = receive_end_ns - receive_start_ns;

  context_stat_.completed_request_count++;
  context_stat_.cumulative_total_request_time_ns += request_time_ns;
  context_stat_.cumulative_send_time_ns += send_time_ns;
  context_stat_.cumulative_receive_time_ns += receive_time_ns;
  return Error::Success;
}

Error
InferContext::GetReadyAsyncRequest(std::shared_ptr<Request>* request, bool wait)
{
  if (ongoing_async_requests_.size() == 0) {
    return Error(
        RequestStatusCode::UNAVAILABLE,
        "No asynchronous requests have been sent");
  }

  Error err;
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [&err, request, this, wait] {
    for (auto& ongoing_async_request : this->ongoing_async_requests_) {
      if (std::static_pointer_cast<RequestImpl>(ongoing_async_request.second)
              ->ready_) {
        *request = ongoing_async_request.second;
        err = Error::Success;
        return true;
      }
    }

    if (!wait) {
      err = Error(RequestStatusCode::UNAVAILABLE, "No completed request.");
      return true;
    } else {
      return false;
    }
  });

  lock.unlock();
  return err;
}

Error
InferContext::IsRequestReady(
    const std::shared_ptr<Request>& async_request, bool wait)
{
  if (ongoing_async_requests_.size() == 0) {
    return Error(
        RequestStatusCode::INVALID_ARG,
        "No asynchronous requests have been sent");
  }

  std::shared_ptr<RequestImpl> request =
      std::static_pointer_cast<RequestImpl>(async_request);

  auto itr = ongoing_async_requests_.find(request->run_index_);
  if (itr == ongoing_async_requests_.end()) {
    return Error(
        RequestStatusCode::INVALID_ARG,
        "No matched asynchronous request found.");
  }

  Error err = Error::Success;
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [&err, &request, wait] {
    if (!request->ready_) {
      if (wait) {
        return false;
      } else {
        err = Error(RequestStatusCode::UNAVAILABLE, "Request is not ready.");
      }
    }
    return true;
  });

  if (!err.IsOk()) {
    lock.unlock();
    return err;
  } else {
    ongoing_async_requests_.erase(itr->first);
  }
  lock.unlock();
  return Error::Success;
}

//==============================================================================

ProfileContext::ProfileContext(bool verbose) : verbose_(verbose) {}

Error
ProfileContext::StartProfile()
{
  return SendCommand("start");
}

Error
ProfileContext::StopProfile()
{
  return SendCommand("stop");
}

//==============================================================================

Error
ServerHealthHttpContext::Create(
    std::unique_ptr<ServerHealthContext>* ctx, const std::string& server_url,
    bool verbose)
{
  ctx->reset(static_cast<ServerHealthContext*>(
      new ServerHealthHttpContext(server_url, verbose)));
  return Error::Success;
}

ServerHealthHttpContext::ServerHealthHttpContext(
    const std::string& server_url, bool verbose)
    : ServerHealthContext(verbose), url_(server_url + "/" + kHealthRESTEndpoint)
{
}

Error
ServerHealthHttpContext::GetHealth(const std::string& url, bool* health)
{
  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  CURL* curl = curl_easy_init();
  if (!curl) {
    return Error(
        RequestStatusCode::INTERNAL, "failed to initialize HTTP client");
  }

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    curl_easy_cleanup(curl);
    return Error(
        RequestStatusCode::INTERNAL,
        "HTTP client failed: " + std::string(curl_easy_strerror(res)));
  }

  // Must use 64-bit integer with curl_easy_getinfo
  int64_t http_code;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_easy_cleanup(curl);

  *health = (http_code == 200) ? true : false;

  return Error::Success;
}

Error
ServerHealthHttpContext::GetReady(bool* ready)
{
  return GetHealth(url_ + "/ready", ready);
}

Error
ServerHealthHttpContext::GetLive(bool* live)
{
  return GetHealth(url_ + "/live", live);
}

//==============================================================================

Error
ServerStatusHttpContext::Create(
    std::unique_ptr<ServerStatusContext>* ctx, const std::string& server_url,
    bool verbose)
{
  ctx->reset(static_cast<ServerStatusContext*>(
      new ServerStatusHttpContext(server_url, verbose)));
  return Error::Success;
}

Error
ServerStatusHttpContext::Create(
    std::unique_ptr<ServerStatusContext>* ctx, const std::string& server_url,
    const std::string& model_name, bool verbose)
{
  ctx->reset(static_cast<ServerStatusContext*>(
      new ServerStatusHttpContext(server_url, model_name, verbose)));
  return Error::Success;
}

ServerStatusHttpContext::ServerStatusHttpContext(
    const std::string& server_url, bool verbose)
    : ServerStatusContext(verbose), url_(server_url + "/" + kStatusRESTEndpoint)
{
}

ServerStatusHttpContext::ServerStatusHttpContext(
    const std::string& server_url, const std::string& model_name, bool verbose)
    : ServerStatusContext(verbose),
      url_(server_url + "/" + kStatusRESTEndpoint + "/" + model_name)
{
}

Error
ServerStatusHttpContext::GetServerStatus(ServerStatus* server_status)
{
  server_status->Clear();
  request_status_.Clear();
  response_.clear();

  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  CURL* curl = curl_easy_init();
  if (!curl) {
    return Error(
        RequestStatusCode::INTERNAL, "failed to initialize HTTP client");
  }

  // Want binary representation of the status.
  std::string full_url = url_ + "?format=binary";
  curl_easy_setopt(curl, CURLOPT_URL, full_url.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  // response headers handled by ResponseHeaderHandler()
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, ResponseHeaderHandler);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, this);

  // response data handled by ResponseHandler()
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, ResponseHandler);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, this);

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    curl_easy_cleanup(curl);
    return Error(
        RequestStatusCode::INTERNAL,
        "HTTP client failed: " + std::string(curl_easy_strerror(res)));
  }

  // Must use 64-bit integer with curl_easy_getinfo
  int64_t http_code;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_easy_cleanup(curl);

  // Should have a request status, if not then create an error status.
  if (request_status_.code() == RequestStatusCode::INVALID) {
    request_status_.Clear();
    request_status_.set_code(RequestStatusCode::INTERNAL);
    request_status_.set_msg("status request did not return status");
  }

  // If request has failing HTTP status or the request's explicit
  // status is not SUCCESS, then signal an error.
  if ((http_code != 200) ||
      (request_status_.code() != RequestStatusCode::SUCCESS)) {
    return Error(request_status_);
  }

  // Parse the response as a ModelConfigList...
  if (!server_status->ParseFromString(response_)) {
    return Error(RequestStatusCode::INTERNAL, "failed to parse server status");
  }

  if (verbose_) {
    std::cout << server_status->DebugString() << std::endl;
  }

  return Error(request_status_);
}

size_t
ServerStatusHttpContext::ResponseHeaderHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  ServerStatusHttpContext* ctx =
      reinterpret_cast<ServerStatusHttpContext*>(userp);

  char* buf = reinterpret_cast<char*>(contents);
  size_t byte_size = size * nmemb;

  size_t idx = strlen(kStatusHTTPHeader);
  if ((idx < byte_size) && !strncasecmp(buf, kStatusHTTPHeader, idx)) {
    while ((idx < byte_size) && (buf[idx] != ':')) {
      ++idx;
    }

    if (idx < byte_size) {
      std::string hdr(buf + idx + 1, byte_size - idx - 1);

      if (!google::protobuf::TextFormat::ParseFromString(
              hdr, &ctx->request_status_)) {
        ctx->request_status_.Clear();
      }
    }
  }

  return byte_size;
}

size_t
ServerStatusHttpContext::ResponseHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  ServerStatusHttpContext* ctx =
      reinterpret_cast<ServerStatusHttpContext*>(userp);
  uint8_t* buf = reinterpret_cast<uint8_t*>(contents);
  size_t result_bytes = size * nmemb;
  std::copy(buf, buf + result_bytes, std::back_inserter(ctx->response_));
  return result_bytes;
}

//==============================================================================

class HttpRequestImpl : public RequestImpl {
 public:
  HttpRequestImpl(
      const uint64_t id,
      const std::vector<std::shared_ptr<InferContext::Input>> inputs);

  ~HttpRequestImpl();

  // Initialize the request for HTTP transfer on top of
  // RequestImpl.InitializeRequestedResults()
  Error InitializeRequest(
      const std::vector<std::shared_ptr<InferContext::Output>>&
          requested_outputs,
      const size_t batch_size);

  // Copy into 'buf' up to 'size' bytes of input data. Return the
  // actual amount copied in 'input_bytes'.
  Error GetNextInput(uint8_t* buf, size_t size, size_t* input_bytes);

  // Copy into the context 'size' bytes of result data from
  // 'buf'. Return the actual amount copied in 'result_bytes'.
  Error SetNextRawResult(const uint8_t* buf, size_t size, size_t* result_bytes);

  // @see RequestImpl.GetResults()
  Error GetResults(std::map<std::string, std::unique_ptr<InferContext::Result>>*
                       results) override;

 private:
  friend class InferHttpContext;

  // Pointer to easy handle that is processing the request
  CURL* easy_handle_;

  // Pointer to the list of the HTTP request header, keep it such that it will
  // be valid during the transfer and can be freed once transfer is completed.
  struct curl_slist* header_list_;

  // Status code for the HTTP request.
  CURLcode http_status_;

  // RequestStatus received in server response.
  RequestStatus request_status_;

  // Buffer that accumulates the serialized InferResponseHeader at the
  // end of the body.
  std::string infer_response_buffer_;

  // The inputs for the request. For asynchronous request, it should
  // be a deep copy of the inputs set by the user in case the user modifies
  // them for another request during the HTTP transfer.
  std::vector<std::shared_ptr<InferContext::Input>> inputs_;

  // Current positions within input vectors when sending request.
  size_t input_pos_idx_;
};

HttpRequestImpl::HttpRequestImpl(
    const uint64_t id,
    const std::vector<std::shared_ptr<InferContext::Input>> inputs)
    : RequestImpl(id), easy_handle_(curl_easy_init()), header_list_(nullptr),
      inputs_(inputs), input_pos_idx_(0)
{
  if (easy_handle_ != nullptr) {
    run_index_ = reinterpret_cast<uintptr_t>(easy_handle_);
  }
}

HttpRequestImpl::~HttpRequestImpl()
{
  if (easy_handle_ != nullptr) {
    curl_easy_cleanup(easy_handle_);
  }
}

Error
HttpRequestImpl::InitializeRequest(
    const std::vector<std::shared_ptr<InferContext::Output>>& requested_outputs,
    const size_t batch_size)
{
  infer_response_buffer_.clear();

  // Reset all the position indicators so that we send all inputs
  // correctly.
  request_status_.Clear();

  for (auto& io : inputs_) {
    reinterpret_cast<InputImpl*>(io.get())->PrepareForRequest();
  }

  input_pos_idx_ = 0;
  result_pos_idx_ = 0;

  return RequestImpl::InitializeRequestedResults(requested_outputs, batch_size);
}


Error
HttpRequestImpl::GetNextInput(uint8_t* buf, size_t size, size_t* input_bytes)
{
  *input_bytes = 0;

  while ((size > 0) && (input_pos_idx_ < inputs_.size())) {
    InputImpl* io = reinterpret_cast<InputImpl*>(inputs_[input_pos_idx_].get());
    size_t ib = 0;
    bool eoi = false;
    Error err = io->GetNext(buf, size, &ib, &eoi);
    if (!err.IsOk()) {
      return err;
    }

    // If input was completely read then move to the next.
    if (eoi) {
      input_pos_idx_++;
    }
    if (ib != 0) {
      *input_bytes += ib;
      size -= ib;
      buf += ib;
    }
  }

  // Sent all input bytes
  if (input_pos_idx_ >= inputs_.size()) {
    timer_.Record(InferContext::RequestTimers::Kind::SEND_END);
  }

  return Error::Success;
}

Error
HttpRequestImpl::SetNextRawResult(
    const uint8_t* buf, size_t size, size_t* result_bytes)
{
  *result_bytes = 0;

  while ((size > 0) && (result_pos_idx_ < requested_results_.size())) {
    ResultImpl* io = reinterpret_cast<ResultImpl*>(
        requested_results_[result_pos_idx_].get());
    size_t ob = 0;

    // Only try to read raw result for RAW
    if (io->ResultFormat() == InferContext::Result::ResultFormat::RAW) {
      Error err = io->SetNextRawResult(buf, size, &ob);
      if (!err.IsOk()) {
        return err;
      }
    }

    // If output couldn't accept any more bytes then move to the next.
    if (ob == 0) {
      result_pos_idx_++;
    } else {
      *result_bytes += ob;
      size -= ob;
      buf += ob;
    }
  }

  // If there is any bytes left then they belong to the response
  // header, since all the RAW results have been filled.
  if (size > 0) {
    infer_response_buffer_.append(reinterpret_cast<const char*>(buf), size);
    *result_bytes += size;
  }

  return Error::Success;
}

Error
HttpRequestImpl::GetResults(
    std::map<std::string, std::unique_ptr<InferContext::Result>>* results)
{
  InferResponseHeader infer_response;

  if (http_status_ != CURLE_OK) {
    curl_slist_free_all(header_list_);
    requested_results_.clear();
    return Error(
        RequestStatusCode::INTERNAL,
        "HTTP client failed: " + std::string(curl_easy_strerror(http_status_)));
  }

  // Must use 64-bit integer with curl_easy_getinfo
  int64_t http_code;
  curl_easy_getinfo(easy_handle_, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(header_list_);

  // Should have a request status, if not then create an error status.
  if (request_status_.code() == RequestStatusCode::INVALID) {
    request_status_.Clear();
    request_status_.set_code(RequestStatusCode::INTERNAL);
    request_status_.set_msg("infer request did not return status");
  }

  // If request has failing HTTP status or the request's explicit
  // status is not SUCCESS, then signal an error.
  if ((http_code != 200) ||
      (request_status_.code() != RequestStatusCode::SUCCESS)) {
    requested_results_.clear();
    return Error(request_status_);
  }

  // The infer response header should be available...
  if (infer_response_buffer_.empty()) {
    requested_results_.clear();
    return Error(
        RequestStatusCode::INTERNAL,
        "infer request did not return result header");
  }

  infer_response.ParseFromString(infer_response_buffer_);

  PostRunProcessing(requested_results_, infer_response);

  results->clear();
  for (auto& result : requested_results_) {
    results->insert(
        std::make_pair(result->GetOutput()->Name(), std::move(result)));
  }

  return Error(request_status_);
}

//==============================================================================

Error
InferHttpContext::Create(
    std::unique_ptr<InferContext>* ctx, const std::string& server_url,
    const std::string& model_name, int64_t model_version, bool verbose)
{
  return Create(
      ctx, 0 /* correlation_id */, server_url, model_name, model_version,
      verbose);
}

Error
InferHttpContext::Create(
    std::unique_ptr<InferContext>* ctx, CorrelationID correlation_id,
    const std::string& server_url, const std::string& model_name,
    int64_t model_version, bool verbose)
{
  InferHttpContext* ctx_ptr = new InferHttpContext(
      server_url, model_name, model_version, correlation_id, verbose);

  // Get status of the model and create the inputs and outputs.
  std::unique_ptr<ServerStatusContext> sctx;
  Error err =
      ServerStatusHttpContext::Create(&sctx, server_url, model_name, verbose);
  if (err.IsOk()) {
    ServerStatus server_status;
    err = sctx->GetServerStatus(&server_status);
    if (err.IsOk()) {
      const auto& itr = server_status.model_status().find(model_name);
      if (itr == server_status.model_status().end()) {
        err = Error(
            RequestStatusCode::INTERNAL,
            "unable to find status information for \"" + model_name + "\"");
      } else {
        const ModelConfig& model_info = itr->second.config();

        ctx_ptr->max_batch_size_ =
            static_cast<uint64_t>(std::max(0, model_info.max_batch_size()));

        // Create inputs and outputs
        for (const auto& io : model_info.input()) {
          ctx_ptr->inputs_.emplace_back(std::make_shared<InputImpl>(io));
        }
        for (const auto& io : model_info.output()) {
          ctx_ptr->outputs_.emplace_back(std::make_shared<OutputImpl>(io));
        }
      }
    }
  }

  // Create request context for synchronous request.
  ctx_ptr->sync_request_.reset(
      static_cast<Request*>(new HttpRequestImpl(0, ctx_ptr->inputs_)));

  if (err.IsOk()) {
    ctx->reset(static_cast<InferContext*>(ctx_ptr));
  } else {
    ctx->reset();
  }

  return err;
}

InferHttpContext::InferHttpContext(
    const std::string& server_url, const std::string& model_name,
    int64_t model_version, CorrelationID correlation_id, bool verbose)
    : InferContext(model_name, model_version, correlation_id, verbose),
      multi_handle_(curl_multi_init())
{
  // Process url for HTTP request
  // URL doesn't contain the version portion if using the latest version.
  url_ = server_url + "/" + kInferRESTEndpoint + "/" + model_name;
  if (model_version_ >= 0) {
    url_ += "/" + std::to_string(model_version_);
  }
}

InferHttpContext::~InferHttpContext()
{
  exiting_ = true;
  // thread not joinable if AsyncRun() is not called
  // (it is default constructed thread before the first AsyncRun() call)
  if (worker_.joinable()) {
    cv_.notify_all();
    worker_.join();
  }

  if (multi_handle_ != nullptr) {
    for (auto& request : ongoing_async_requests_) {
      CURL* easy_handle =
          std::static_pointer_cast<HttpRequestImpl>(request.second)
              ->easy_handle_;
      // Just remove, easy_cleanup will be done in ~HttpRequestImpl()
      curl_multi_remove_handle(multi_handle_, easy_handle);
    }
    curl_multi_cleanup(multi_handle_);
  }
}

Error
InferHttpContext::Run(std::map<std::string, std::unique_ptr<Result>>* results)
{
  std::shared_ptr<HttpRequestImpl> sync_request =
      std::static_pointer_cast<HttpRequestImpl>(sync_request_);

  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  Error err = PreRunProcessing(sync_request_);

  if (!err.IsOk()) {
    return err;
  }

  // Take run time
  sync_request->timer_.Reset();
  sync_request->timer_.Record(RequestTimers::Kind::REQUEST_START);
  sync_request->timer_.Record(RequestTimers::Kind::SEND_START);
  sync_request->http_status_ = curl_easy_perform(sync_request->easy_handle_);
  sync_request->timer_.Record(RequestTimers::Kind::RECEIVE_END);
  sync_request->timer_.Record(RequestTimers::Kind::REQUEST_END);

  err = UpdateStat(sync_request->timer_);
  if (!err.IsOk()) {
    std::cerr << "Failed to update context stat: " << err << std::endl;
  }
  return sync_request->GetResults(results);
}

Error
InferHttpContext::AsyncRun(std::shared_ptr<Request>* async_request)
{
  if (!multi_handle_) {
    return Error(
        RequestStatusCode::INTERNAL,
        "failed to start HTTP asynchronous client");
  } else if (exiting_) {
    // abusing variable here, exiting_ is true either when destructor is called
    // or the worker thread is not actually created.
    exiting_ = false;
    worker_ = std::thread(&InferHttpContext::AsyncTransfer, this);
  }

  // Make a copy of the current inputs
  std::vector<std::shared_ptr<Input>> inputs;
  for (const auto& io : inputs_) {
    InputImpl* input = reinterpret_cast<InputImpl*>(io.get());
    inputs.emplace_back(std::make_shared<InputImpl>(*input));
  }

  HttpRequestImpl* current_context =
      new HttpRequestImpl(async_request_id_++, inputs);
  async_request->reset(static_cast<Request*>(current_context));

  if (!current_context->easy_handle_) {
    return Error(
        RequestStatusCode::INTERNAL, "failed to initialize HTTP client");
  }

  Error err = PreRunProcessing(*async_request);

  {
    std::lock_guard<std::mutex> lock(mutex_);

    auto insert_result = ongoing_async_requests_.emplace(std::make_pair(
        reinterpret_cast<uintptr_t>(current_context->easy_handle_),
        *async_request));

    if (!insert_result.second) {
      return Error(
          RequestStatusCode::INTERNAL,
          "Failed to insert new asynchronous request context.");
    }

    curl_multi_add_handle(multi_handle_, current_context->easy_handle_);
    current_context->timer_.Reset();
    current_context->timer_.Record(RequestTimers::Kind::REQUEST_START);
    current_context->timer_.Record(RequestTimers::Kind::SEND_START);
  }

  cv_.notify_all();
  return Error(RequestStatusCode::SUCCESS);
}

Error
InferHttpContext::GetAsyncRunResults(
    std::map<std::string, std::unique_ptr<Result>>* results,
    const std::shared_ptr<Request>& async_request, bool wait)
{
  Error err = IsRequestReady(async_request, wait);
  if (!err.IsOk()) {
    return err;
  }
  std::shared_ptr<HttpRequestImpl> http_request =
      std::static_pointer_cast<HttpRequestImpl>(async_request);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    curl_multi_remove_handle(multi_handle_, http_request->easy_handle_);
  }

  err = UpdateStat(http_request->timer_);
  if (!err.IsOk()) {
    std::cerr << "Failed to update context stat: " << err << std::endl;
  }
  return http_request->GetResults(results);
}

size_t
InferHttpContext::RequestProvider(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  HttpRequestImpl* request = reinterpret_cast<HttpRequestImpl*>(userp);

  size_t input_bytes = 0;
  Error err = request->GetNextInput(
      reinterpret_cast<uint8_t*>(contents), size * nmemb, &input_bytes);
  if (!err.IsOk()) {
    std::cerr << "RequestProvider: " << err << std::endl;
    return CURL_READFUNC_ABORT;
  }

  return input_bytes;
}

size_t
InferHttpContext::ResponseHeaderHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  HttpRequestImpl* request = reinterpret_cast<HttpRequestImpl*>(userp);
  char* buf = reinterpret_cast<char*>(contents);
  size_t byte_size = size * nmemb;

  size_t idx = strlen(kStatusHTTPHeader);
  if ((idx < byte_size) && !strncasecmp(buf, kStatusHTTPHeader, idx)) {
    while ((idx < byte_size) && (buf[idx] != ':')) {
      ++idx;
    }

    if (idx < byte_size) {
      std::string hdr(buf + idx + 1, byte_size - idx - 1);
      if (!google::protobuf::TextFormat::ParseFromString(
              hdr, &request->request_status_)) {
        request->request_status_.Clear();
      }
    }
  }

  return byte_size;
}

size_t
InferHttpContext::ResponseHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  HttpRequestImpl* request = reinterpret_cast<HttpRequestImpl*>(userp);
  size_t result_bytes = 0;

  if (request->timer_.receive_start_.tv_sec == 0) {
    request->timer_.Record(RequestTimers::Kind::RECEIVE_START);
  }

  Error err = request->SetNextRawResult(
      reinterpret_cast<uint8_t*>(contents), size * nmemb, &result_bytes);
  if (!err.IsOk()) {
    std::cerr << "ResponseHandler: " << err << std::endl;
    return 0;
  }

  return result_bytes;
}

Error
InferHttpContext::PreRunProcessing(std::shared_ptr<Request>& request)
{
  std::shared_ptr<HttpRequestImpl> http_request =
      std::static_pointer_cast<HttpRequestImpl>(request);

  http_request->InitializeRequest(requested_outputs_, batch_size_);

  CURL* curl = http_request->easy_handle_;
  if (!curl) {
    return Error(
        RequestStatusCode::INTERNAL, "failed to initialize HTTP client");
  }

  std::string full_url = url_ + "?format=binary";
  curl_easy_setopt(curl, CURLOPT_URL, full_url.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  // request data provided by RequestProvider()
  curl_easy_setopt(curl, CURLOPT_READFUNCTION, RequestProvider);
  curl_easy_setopt(curl, CURLOPT_READDATA, http_request.get());

  // response headers handled by ResponseHeaderHandler()
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, ResponseHeaderHandler);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, http_request.get());

  // response data handled by ResponseHandler()
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, ResponseHandler);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, http_request.get());

  // Create the input metadata for the request now that all input
  // sizes are known. For non-fixed-sized datatypes the
  // per-batch-instance byte-size can be different for different input
  // instances in the batch... so set the batch-byte-size to the total
  // size of the batch (see api.proto).
  uint64_t total_input_byte_size = 0;
  infer_request_.mutable_input()->Clear();
  for (const auto& io : inputs_) {
    total_input_byte_size += io->TotalByteSize();

    auto rinput = infer_request_.add_input();
    rinput->set_name(io->Name());

    for (const auto s : io->Shape()) {
      rinput->add_dims(s);
    }
    if (!IsFixedSizeDataType(io->DType())) {
      rinput->set_batch_byte_size(io->TotalByteSize());
    }
  }

  // set the expected POST size. If you want to POST large amounts of
  // data, consider CURLOPT_POSTFIELDSIZE_LARGE
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, total_input_byte_size);

  // Headers to specify input and output tensors
  infer_request_str_.clear();
  infer_request_str_ = std::string(kInferRequestHTTPHeader) + ":" +
                       infer_request_.ShortDebugString();
  struct curl_slist* list = nullptr;
  list = curl_slist_append(list, "Expect:");
  list = curl_slist_append(list, "Content-Type: application/octet-stream");
  list = curl_slist_append(list, infer_request_str_.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);

  // The list should be freed after the request
  http_request->header_list_ = list;

  return Error::Success;
}

void
InferHttpContext::AsyncTransfer()
{
  int place_holder = 0;
  CURLMsg* msg = nullptr;
  do {
    bool has_completed = false;
    // sleep if no work is available
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] {
      if (this->exiting_) {
        return true;
      }
      // wake up if at least one request is not ready
      for (auto& ongoing_async_request : this->ongoing_async_requests_) {
        if (std::static_pointer_cast<HttpRequestImpl>(
                ongoing_async_request.second)
                ->ready_ == false) {
          return true;
        }
      }
      return false;
    });
    curl_multi_perform(multi_handle_, &place_holder);
    while ((msg = curl_multi_info_read(multi_handle_, &place_holder))) {
      // update request status
      uintptr_t identifier = reinterpret_cast<uintptr_t>(msg->easy_handle);
      auto itr = ongoing_async_requests_.find(identifier);
      // This shouldn't happen
      if (itr == ongoing_async_requests_.end()) {
        fprintf(
            stderr,
            "Unexpected error: received completed request that"
            " is not in the list of asynchronous requests.\n");
        curl_multi_remove_handle(multi_handle_, msg->easy_handle);
        curl_easy_cleanup(msg->easy_handle);
        continue;
      }
      std::shared_ptr<HttpRequestImpl> http_request =
          std::static_pointer_cast<HttpRequestImpl>(itr->second);

      if (msg->msg != CURLMSG_DONE) {
        // Something wrong happened.
        fprintf(stderr, "Unexpected error: received CURLMsg=%d\n", msg->msg);
      } else {
        http_request->timer_.Record(RequestTimers::Kind::RECEIVE_END);
        http_request->timer_.Record(RequestTimers::Kind::REQUEST_END);
      }
      http_request->http_status_ = msg->data.result;
      http_request->ready_ = true;
      has_completed = true;
    }
    lock.unlock();
    // if it has completed tasks, send signal in case the main thread is waiting
    if (has_completed) {
      cv_.notify_all();
    }
  } while (!exiting_);
}

//==============================================================================

Error
ProfileHttpContext::Create(
    std::unique_ptr<ProfileContext>* ctx, const std::string& server_url,
    bool verbose)
{
  ctx->reset(static_cast<ProfileContext*>(
      new ProfileHttpContext(server_url, verbose)));
  return Error::Success;
}

ProfileHttpContext::ProfileHttpContext(
    const std::string& server_url, bool verbose)
    : ProfileContext(verbose), url_(server_url + "/" + kProfileRESTEndpoint)
{
}

Error
ProfileHttpContext::SendCommand(const std::string& cmd_str)
{
  request_status_.Clear();

  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  CURL* curl = curl_easy_init();
  if (!curl) {
    return Error(
        RequestStatusCode::INTERNAL, "failed to initialize HTTP client");
  }

  // Want binary representation of the status.
  std::string full_url = url_ + "?cmd=" + cmd_str;
  curl_easy_setopt(curl, CURLOPT_URL, full_url.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  // response headers handled by ResponseHeaderHandler()
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, ResponseHeaderHandler);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, this);

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    curl_easy_cleanup(curl);
    return Error(
        RequestStatusCode::INTERNAL,
        "HTTP client failed: " + std::string(curl_easy_strerror(res)));
  }

  // Must use 64-bit integer with curl_easy_getinfo
  int64_t http_code;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_easy_cleanup(curl);

  // Should have a request status, if not then create an error status.
  if (request_status_.code() == RequestStatusCode::INVALID) {
    request_status_.Clear();
    request_status_.set_code(RequestStatusCode::INTERNAL);
    request_status_.set_msg("profile request did not return status");
  }

  return Error(request_status_);
}

size_t
ProfileHttpContext::ResponseHeaderHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  ProfileHttpContext* ctx = reinterpret_cast<ProfileHttpContext*>(userp);

  char* buf = reinterpret_cast<char*>(contents);
  size_t byte_size = size * nmemb;

  size_t idx = strlen(kStatusHTTPHeader);
  if ((idx < byte_size) && !strncasecmp(buf, kStatusHTTPHeader, idx)) {
    while ((idx < byte_size) && (buf[idx] != ':')) {
      ++idx;
    }

    if (idx < byte_size) {
      std::string hdr(buf + idx + 1, byte_size - idx - 1);

      if (!google::protobuf::TextFormat::ParseFromString(
              hdr, &ctx->request_status_)) {
        ctx->request_status_.Clear();
      }
    }
  }

  return byte_size;
}

//==============================================================================

Error
ServerHealthGrpcContext::Create(
    std::unique_ptr<ServerHealthContext>* ctx, const std::string& server_url,
    bool verbose)
{
  ctx->reset(static_cast<ServerHealthContext*>(
      new ServerHealthGrpcContext(server_url, verbose)));
  return Error::Success;
}

ServerHealthGrpcContext::ServerHealthGrpcContext(
    const std::string& server_url, bool verbose)
    : ServerHealthContext(verbose),
      stub_(GRPCService::NewStub(GetChannel(server_url)))
{
}

Error
ServerHealthGrpcContext::GetHealth(const std::string& mode, bool* health)
{
  Error err;

  HealthRequest request;
  HealthResponse response;
  grpc::ClientContext context;

  request.set_mode(mode);
  grpc::Status grpc_status = stub_->Health(&context, request, &response);
  if (grpc_status.ok()) {
    *health = response.health();
    err = Error(response.request_status());
  } else {
    // Something wrong with the GRPC connection
    err = Error(
        RequestStatusCode::INTERNAL,
        "GRPC client failed: " + std::to_string(grpc_status.error_code()) +
            ": " + grpc_status.error_message());
  }

  if (verbose_ && err.IsOk()) {
    std::cout << mode << ": " << *health << std::endl;
  }

  return err;
}

Error
ServerHealthGrpcContext::GetReady(bool* ready)
{
  return GetHealth("ready", ready);
}

Error
ServerHealthGrpcContext::GetLive(bool* live)
{
  return GetHealth("live", live);
}

//==============================================================================

Error
ServerStatusGrpcContext::Create(
    std::unique_ptr<ServerStatusContext>* ctx, const std::string& server_url,
    bool verbose)
{
  ctx->reset(static_cast<ServerStatusContext*>(
      new ServerStatusGrpcContext(server_url, verbose)));
  return Error::Success;
}

Error
ServerStatusGrpcContext::Create(
    std::unique_ptr<ServerStatusContext>* ctx, const std::string& server_url,
    const std::string& model_name, bool verbose)
{
  ctx->reset(static_cast<ServerStatusContext*>(
      new ServerStatusGrpcContext(server_url, model_name, verbose)));
  return Error::Success;
}

ServerStatusGrpcContext::ServerStatusGrpcContext(
    const std::string& server_url, bool verbose)
    : ServerStatusContext(verbose), model_name_(""),
      stub_(GRPCService::NewStub(GetChannel(server_url)))
{
}

ServerStatusGrpcContext::ServerStatusGrpcContext(
    const std::string& server_url, const std::string& model_name, bool verbose)
    : ServerStatusContext(verbose), model_name_(model_name),
      stub_(GRPCService::NewStub(GetChannel(server_url)))
{
}

Error
ServerStatusGrpcContext::GetServerStatus(ServerStatus* server_status)
{
  server_status->Clear();

  Error grpc_status;

  StatusRequest request;
  StatusResponse response;
  grpc::ClientContext context;

  request.set_model_name(model_name_);
  grpc::Status status = stub_->Status(&context, request, &response);
  if (status.ok()) {
    server_status->Swap(response.mutable_server_status());
    grpc_status = Error(response.request_status());
  } else {
    // Something wrong with the GRPC conncection
    grpc_status = Error(
        RequestStatusCode::INTERNAL,
        "GRPC client failed: " + std::to_string(status.error_code()) + ": " +
            status.error_message());
  }

  // Log server status if request is SUCCESS and verbose is true.
  if (grpc_status.Code() == RequestStatusCode::SUCCESS && verbose_) {
    std::cout << server_status->DebugString() << std::endl;
  }
  return grpc_status;
}

//==============================================================================

class GrpcRequestImpl : public RequestImpl {
 public:
  GrpcRequestImpl(const uint64_t id, const uintptr_t run_index);

  // @see RequestImpl.GetResults()
  Error GetResults(std::map<std::string, std::unique_ptr<InferContext::Result>>*
                       results) override;

 private:
  // Unmarshall and process 'grpc_response_' into 'requested_results'
  Error SetRawResult();

  friend class InferGrpcContext;

  // Variables for GRPC call
  grpc::ClientContext grpc_context_;
  grpc::Status grpc_status_;
  InferResponse grpc_response_;
};

GrpcRequestImpl::GrpcRequestImpl(const uint64_t id, const uintptr_t run_index)
    : RequestImpl(id)
{
  run_index_ = run_index;
}

Error
GrpcRequestImpl::SetRawResult()
{
  result_pos_idx_ = 0;
  for (std::string output : grpc_response_.raw_output()) {
    const uint8_t* buf = reinterpret_cast<uint8_t*>(&output[0]);
    size_t size = output.size();
    size_t result_bytes = 0;

    // Not using loop as in HTTP Infer because the output size should match
    if ((size > 0) && (result_pos_idx_ < requested_results_.size())) {
      ResultImpl* io = reinterpret_cast<ResultImpl*>(
          requested_results_[result_pos_idx_].get());

      // Only try to read raw result for RAW
      if (io->ResultFormat() == InferContext::Result::ResultFormat::RAW) {
        Error err = io->SetNextRawResult(buf, size, &result_bytes);
        if (!err.IsOk()) {
          return err;
        }
      }
    }

    if (result_bytes != size) {
      return Error(
          RequestStatusCode::INVALID,
          "Written bytes doesn't match received bytes.");
    }

    result_pos_idx_++;
  }

  return Error::Success;
}

Error
GrpcRequestImpl::GetResults(
    std::map<std::string, std::unique_ptr<InferContext::Result>>* results)
{
  results->clear();
  InferResponseHeader infer_response;

  Error err(RequestStatusCode::SUCCESS);
  if (grpc_status_.ok()) {
    infer_response.Swap(grpc_response_.mutable_meta_data());
    err = Error(grpc_response_.request_status());
    if (err.IsOk()) {
      Error set_err = SetRawResult();
      if (!set_err.IsOk()) {
        return set_err;
      }
    }
  } else {
    // Something wrong with the GRPC conncection
    err = Error(
        RequestStatusCode::INTERNAL,
        "GRPC client failed: " + std::to_string(grpc_status_.error_code()) +
            ": " + grpc_status_.error_message());
  }

  // Only continue to process result if GRPC status is SUCCESS
  if (err.Code() == RequestStatusCode::SUCCESS) {
    PostRunProcessing(requested_results_, infer_response);

    results->clear();
    for (auto& result : requested_results_) {
      results->insert(
          std::make_pair(result->GetOutput()->Name(), std::move(result)));
    }
  }

  return err;
}

//==============================================================================

Error
InferGrpcContext::Create(
    std::unique_ptr<InferContext>* ctx, const std::string& server_url,
    const std::string& model_name, int64_t model_version, bool verbose)
{
  return Create(
      ctx, 0 /* correlation_id */, server_url, model_name, model_version,
      verbose);
}

Error
InferGrpcContext::Create(
    std::unique_ptr<InferContext>* ctx, CorrelationID correlation_id,
    const std::string& server_url, const std::string& model_name,
    int64_t model_version, bool verbose)
{
  InferGrpcContext* ctx_ptr = new InferGrpcContext(
      server_url, model_name, model_version, correlation_id, verbose);

  // Create request context for synchronous request.
  ctx_ptr->sync_request_.reset(
      static_cast<Request*>(new GrpcRequestImpl(0, 0)));

  // Get status of the model and create the inputs and outputs.
  std::unique_ptr<ServerStatusContext> sctx;
  Error err =
      ServerStatusGrpcContext::Create(&sctx, server_url, model_name, verbose);
  if (err.IsOk()) {
    ServerStatus server_status;
    err = sctx->GetServerStatus(&server_status);
    if (err.IsOk()) {
      const auto& itr = server_status.model_status().find(model_name);
      if (itr == server_status.model_status().end()) {
        err = Error(
            RequestStatusCode::INTERNAL,
            "unable to find status information for \"" + model_name + "\"");
      } else {
        const ModelConfig& model_info = itr->second.config();

        ctx_ptr->max_batch_size_ =
            static_cast<uint64_t>(std::max(0, model_info.max_batch_size()));

        // Create inputs and outputs
        for (const auto& io : model_info.input()) {
          ctx_ptr->inputs_.emplace_back(std::make_shared<InputImpl>(io));
        }
        for (const auto& io : model_info.output()) {
          ctx_ptr->outputs_.emplace_back(std::make_shared<OutputImpl>(io));
        }
      }
    }
  }

  if (err.IsOk()) {
    ctx->reset(static_cast<InferContext*>(ctx_ptr));
  } else {
    ctx->reset();
  }

  return err;
}

InferGrpcContext::InferGrpcContext(
    const std::string& server_url, const std::string& model_name,
    int64_t model_version, CorrelationID correlation_id, bool verbose)
    : InferContext(model_name, model_version, correlation_id, verbose),
      stub_(GRPCService::NewStub(GetChannel(server_url)))
{
}

InferGrpcContext::~InferGrpcContext()
{
  exiting_ = true;
  // thread not joinable if AsyncRun() is not called
  // (it is default constructed thread before the first AsyncRun() call)
  if (worker_.joinable()) {
    cv_.notify_all();
    worker_.join();
  }

  // Close complete queue and drain its content
  async_request_completion_queue_.Shutdown();
  bool has_next = true;
  void* tag;
  bool ok;
  do {
    has_next = async_request_completion_queue_.Next(&tag, &ok);
  } while (has_next);
}

Error
InferGrpcContext::Run(std::map<std::string, std::unique_ptr<Result>>* results)
{
  grpc::ClientContext context;

  std::shared_ptr<GrpcRequestImpl> sync_request =
      std::static_pointer_cast<GrpcRequestImpl>(sync_request_);

  sync_request->timer_.Reset();
  // Use send timer to measure time for marshalling infer request
  sync_request->timer_.Record(RequestTimers::Kind::SEND_START);
  PreRunProcessing(sync_request_);
  sync_request->timer_.Record(RequestTimers::Kind::SEND_END);

  sync_request->timer_.Record(RequestTimers::Kind::REQUEST_START);
  sync_request->grpc_status_ =
      stub_->Infer(&context, request_, &sync_request->grpc_response_);
  sync_request->timer_.Record(RequestTimers::Kind::REQUEST_END);

  sync_request->timer_.Record(RequestTimers::Kind::RECEIVE_START);
  Error request_status = sync_request->GetResults(results);
  sync_request->timer_.Record(RequestTimers::Kind::RECEIVE_END);

  Error err = UpdateStat(sync_request->timer_);
  if (!err.IsOk()) {
    std::cerr << "Failed to update context stat: " << err << std::endl;
  }

  return request_status;
}

Error
InferGrpcContext::AsyncRun(std::shared_ptr<Request>* async_request)
{
  if (exiting_) {
    exiting_ = false;
    worker_ = std::thread(&InferGrpcContext::AsyncTransfer, this);
  }
  uintptr_t run_index;
  if (reusable_slot_.empty()) {
    run_index = ongoing_async_requests_.size();
  } else {
    run_index = reusable_slot_.back();
    reusable_slot_.pop_back();
  }

  GrpcRequestImpl* current_context =
      new GrpcRequestImpl(async_request_id_++, run_index);
  async_request->reset(static_cast<Request*>(current_context));

  auto insert_result = ongoing_async_requests_.emplace(
      std::make_pair(run_index, *async_request));

  if (!insert_result.second) {
    return Error(
        RequestStatusCode::INTERNAL,
        "Failed to insert new asynchronous request context.");
  }

  current_context->timer_.Reset();
  current_context->timer_.Record(RequestTimers::Kind::SEND_START);
  PreRunProcessing(*async_request);
  current_context->timer_.Record(RequestTimers::Kind::SEND_END);

  current_context->timer_.Record(RequestTimers::Kind::REQUEST_START);
  std::unique_ptr<grpc::ClientAsyncResponseReader<InferResponse>> rpc(
      stub_->PrepareAsyncInfer(
          &current_context->grpc_context_, request_,
          &async_request_completion_queue_));

  rpc->StartCall();

  rpc->Finish(
      &current_context->grpc_response_, &current_context->grpc_status_,
      (void*)run_index);

  cv_.notify_all();
  return Error(RequestStatusCode::SUCCESS);
}

Error
InferGrpcContext::GetAsyncRunResults(
    std::map<std::string, std::unique_ptr<Result>>* results,
    const std::shared_ptr<Request>& async_request, bool wait)
{
  Error err = IsRequestReady(async_request, wait);
  if (!err.IsOk()) {
    return err;
  }

  std::shared_ptr<GrpcRequestImpl> grpc_request =
      std::static_pointer_cast<GrpcRequestImpl>(async_request);

  reusable_slot_.push_back(grpc_request->run_index_);
  grpc_request->timer_.Record(RequestTimers::Kind::RECEIVE_START);
  Error request_status = grpc_request->GetResults(results);
  grpc_request->timer_.Record(RequestTimers::Kind::RECEIVE_END);
  err = UpdateStat(grpc_request->timer_);
  if (!err.IsOk()) {
    std::cerr << "Failed to update context stat: " << err << std::endl;
  }
  return request_status;
}

Error
InferGrpcContext::PreRunProcessing(std::shared_ptr<Request>& request)
{
  std::shared_ptr<GrpcRequestImpl> grpc_request =
      std::static_pointer_cast<GrpcRequestImpl>(request);
  grpc_request->InitializeRequestedResults(requested_outputs_, batch_size_);

  // Create the input metadata for the request now that all input
  // sizes are known. For non-fixed-sized datatypes the
  // per-batch-instance byte-size can be different for different input
  // instances in the batch... so set the batch-byte-size to the total
  // size of the batch (see api.proto).
  infer_request_.mutable_input()->Clear();
  for (auto& io : inputs_) {
    reinterpret_cast<InputImpl*>(io.get())->PrepareForRequest();

    auto rinput = infer_request_.add_input();
    rinput->set_name(io->Name());

    for (const auto s : io->Shape()) {
      rinput->add_dims(s);
    }
    if (!IsFixedSizeDataType(io->DType())) {
      rinput->set_batch_byte_size(io->TotalByteSize());
    }
  }

  request_.Clear();
  request_.set_model_name(model_name_);
  request_.set_model_version(model_version_);
  request_.mutable_meta_data()->MergeFrom(infer_request_);

  size_t input_pos_idx = 0;
  while (input_pos_idx < inputs_.size()) {
    InputImpl* io = reinterpret_cast<InputImpl*>(inputs_[input_pos_idx].get());
    std::string* new_input = request_.add_raw_input();

    // Append all batches of one input together
    for (size_t batch_idx = 0; batch_idx < batch_size_; batch_idx++) {
      const uint8_t* data_ptr;
      size_t data_byte_size;
      io->GetRaw(batch_idx, &data_ptr, &data_byte_size);
      new_input->append(
          reinterpret_cast<const char*>(data_ptr), data_byte_size);
    }
    input_pos_idx++;
  }

  return Error::Success;
}

void
InferGrpcContext::AsyncTransfer()
{
  do {
    // sleep if no work is available
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] {
      if (this->exiting_) {
        return true;
      }
      // wake up if at least one request is not ready
      for (auto& ongoing_async_request : this->ongoing_async_requests_) {
        if (std::static_pointer_cast<GrpcRequestImpl>(
                ongoing_async_request.second)
                ->ready_ == false) {
          return true;
        }
      }
      return false;
    });
    lock.unlock();
    // GRPC async APIs are thread-safe https://github.com/grpc/grpc/issues/4486
    if (!exiting_) {
      size_t got;
      bool ok = true;
      bool status = async_request_completion_queue_.Next((void**)(&got), &ok);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ok) {
          fprintf(stderr, "Unexpected not ok on client side.");
        }
        if (!status) {
          fprintf(stderr, "Completion queue is closed.");
        }
        auto itr = ongoing_async_requests_.find(got);
        if (itr == ongoing_async_requests_.end()) {
          fprintf(
              stderr,
              "Unexpected error: received completed request that"
              " is not in the list of asynchronous requests.\n");
          continue;
        }

        std::shared_ptr<GrpcRequestImpl> grpc_request =
            std::static_pointer_cast<GrpcRequestImpl>(itr->second);
        grpc_request->timer_.Record(RequestTimers::Kind::REQUEST_END);
        grpc_request->ready_ = true;
      }
      // send signal in case the main thread is waiting
      cv_.notify_all();
    }
  } while (!exiting_);
}

//==============================================================================

Error
ProfileGrpcContext::Create(
    std::unique_ptr<ProfileContext>* ctx, const std::string& server_url,
    bool verbose)
{
  ctx->reset(static_cast<ProfileContext*>(
      new ProfileGrpcContext(server_url, verbose)));
  return Error::Success;
}

ProfileGrpcContext::ProfileGrpcContext(
    const std::string& server_url, bool verbose)
    : ProfileContext(verbose),
      stub_(GRPCService::NewStub(GetChannel(server_url)))
{
}

Error
ProfileGrpcContext::SendCommand(const std::string& cmd_str)
{
  ProfileRequest request;
  ProfileResponse response;
  grpc::ClientContext context;

  request.set_cmd(cmd_str);
  grpc::Status status = stub_->Profile(&context, request, &response);
  if (status.ok()) {
    return Error(response.request_status());
  } else {
    // Something wrong with the GRPC conncection
    return Error(
        RequestStatusCode::INTERNAL,
        "GRPC client failed: " + std::to_string(status.error_code()) + ": " +
            status.error_message());
  }
}

}}}  // namespace nvidia::inferenceserver::client
