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
#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <limits>
#include <list>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "src/clients/c++/library/request.h"
#include "src/core/constants.h"
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"

namespace nvidia { namespace inferenceserver { namespace client {

// Records timestamps for different stages of request handling.
class RequestTimers {
 public:
  /// Timestamp kinds.
  enum class Kind {
    /// The start of request handling.
    REQUEST_START,

    /// The end of request handling.
    REQUEST_END,

    /// The start of sending request bytes to the server (i.e. first
    /// byte).
    SEND_START,

    /// The end of sending request bytes to the server (i.e. last
    /// byte).
    SEND_END,

    /// The start of receiving response bytes from the server
    /// (i.e. first byte).
    RECV_START,

    /// The end of receiving response bytes from the server (i.e. last
    /// byte).
    RECV_END,

    COUNT__
  };

  /// Construct a timer with zero-ed timestamps.
  RequestTimers() : timestamps_((size_t)Kind::COUNT__) { Reset(); }

  /// Reset all timestamp values to zero. Must be called before
  /// re-using the timer.
  void Reset()
  {
    memset(&timestamps_[0], 0, sizeof(uint64_t) * timestamps_.size());
  }

  /// Get the timestamp, in nanoseconds, for a kind.
  /// \param kind The timestamp kind.
  /// \return The timestamp in nanoseconds.
  uint64_t Timestamp(Kind kind) const { return timestamps_[(size_t)kind]; }

  /// Set a timestamp to the current time, in nanoseconds.
  /// \param kind The timestamp kind.
  /// \return The timestamp in nanoseconds.
  uint64_t CaptureTimestamp(Kind kind)
  {
    uint64_t& ts = timestamps_[(size_t)kind];
    ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
             .count();
    return ts;
  }

  /// Return the duration between start time point and end timepoint
  /// in nanosecond.
  /// \param start The start time point.
  /// \param end The end time point.
  /// \return Duration in nanosecond, or
  /// std::numeric_limits<uint64_t>::max to indicate that duration
  /// could not be calculated.
  uint64_t Duration(Kind start, Kind end) const
  {
    const uint64_t stime = timestamps_[(size_t)start];
    const uint64_t etime = timestamps_[(size_t)end];

    // If the start or end timestamp is 0 then can't calculate the
    // duration, so return max to indicate error.
    if ((stime == 0) || (etime == 0)) {
      return (std::numeric_limits<uint64_t>::max)();
    }

    return (stime > etime) ? (std::numeric_limits<uint64_t>::max)()
                           : etime - stime;
  }

 private:
  std::vector<uint64_t> timestamps_;
};

//==============================================================================

class InferOptionsImpl : public InferContext::Options {
 public:
  InferOptionsImpl()
      : flags_(0), batch_size_(0), priority_(0), timeout_ms_(0),
        correlation_id_(0)
  {
  }
  ~InferOptionsImpl() = default;

  bool Flag(InferRequestHeader::Flag flag) const override;
  void SetFlag(InferRequestHeader::Flag flag, bool value) override;
  uint32_t Flags() const override { return flags_; }
  void SetFlags(uint32_t flags) override { flags_ = flags; }

  size_t BatchSize() const override { return batch_size_; }
  void SetBatchSize(size_t batch_size) override { batch_size_ = batch_size; }

  uint32_t Priority() const override { return priority_; }
  void SetPriority(uint32_t priority) override { priority_ = priority; }

  uint64_t Timeout() const override { return timeout_ms_; }
  void SetTimeout(uint64_t timeout_ms) override { timeout_ms_ = timeout_ms; }

  CorrelationID CorrelationId() const override { return correlation_id_; }
  void SetCorrelationId(CorrelationID correlation_id) override
  {
    correlation_id_ = correlation_id;
  }

  Error AddRawResult(
      const std::shared_ptr<InferContext::Output>& output) override;
  Error AddClassResult(
      const std::shared_ptr<InferContext::Output>& output, uint64_t k) override;
  Error AddSharedMemoryResult(
      const std::shared_ptr<InferContext::Output>& output,
      const std::string& name, size_t offset, size_t byte_size) override;

  // Options for an output
  struct OutputOptions {
    OutputOptions(
        InferContext::Result::ResultFormat f, uint64_t n = 0,
        std::string name = "", size_t offset = 0, size_t byte_size = 0)
        : result_format(f), u64(n), shm_name(name), shm_offset(offset),
          shm_byte_size(byte_size)
    {
    }
    InferContext::Result::ResultFormat result_format;
    uint64_t u64;
    std::string shm_name;
    size_t shm_offset, shm_byte_size;
  };

  using OutputOptionsPair =
      std::pair<std::shared_ptr<InferContext::Output>, OutputOptions>;

  const std::deque<OutputOptionsPair>& Outputs() const { return outputs_; }

 private:
  uint32_t flags_;
  size_t batch_size_;
  uint32_t priority_;
  uint64_t timeout_ms_;
  CorrelationID correlation_id_;
  std::deque<OutputOptionsPair> outputs_;
};

//==============================================================================

class InputImpl : public InferContext::Input {
 public:
  InputImpl(const ModelInput& mio);
  InputImpl(const InputImpl& obj);
  ~InputImpl() = default;

  const std::string& Name() const override { return mio_.name(); }
  int64_t ByteSize() const override { return byte_size_; }
  size_t TotalByteSize() const override { return total_byte_size_; }
  size_t TotalSendByteSize() const { return total_send_byte_size_; }
  DataType DType() const override { return mio_.data_type(); }
  bool IsShapeTensor() const override { return mio_.is_shape_tensor(); }
  ModelInput::Format Format() const override { return mio_.format(); }
  const DimsList& Dims() const override { return mio_.dims(); }
  bool IsSharedMemory() const { return (io_type_ == SHARED_MEMORY); }
  const std::string& GetSharedMemoryName() const { return shm_name_; }
  size_t GetSharedMemoryOffset() const { return shm_offset_; }

  void SetBatchSize(size_t batch_size) { batch_size_ = batch_size; }

  const std::vector<int64_t>& Shape() const override { return shape_; }
  Error SetShape(const std::vector<int64_t>& dims) override;

  Error Reset() override;
  Error SetRaw(const std::vector<uint8_t>& input) override;
  Error SetRaw(const uint8_t* input, size_t input_byte_size) override;
  Error SetSharedMemory(
      const std::string& name, size_t offset, size_t byte_size) override;
  Error SetFromString(const std::vector<std::string>& input) override;

  // Copy into 'buf' up to 'size' bytes of this input's data. Return
  // the actual amount copied in 'input_bytes' and if the end of input
  // is reached in 'end_of_input'
  Error GetNext(
      uint8_t* buf, size_t size, size_t* input_bytes, bool* end_of_input);

  // Copy the pointer of the raw buffer at 'batch_idx' into 'buf'
  Error GetRaw(size_t batch_idx, const uint8_t** buf, size_t* byte_size) const;

  // Copy the shared memory key, offset and batch_byte_size
  Error GetSharedMemory(
      std::string* name, size_t* offset, size_t* batch_byte_size);

  // Prepare to send this input as part of a request.
  Error PrepareForRequest();

 private:
  const ModelInput mio_;

  int64_t byte_size_;
  size_t total_byte_size_;
  size_t total_send_byte_size_;

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

  // Used only if working with Shared Memory
  enum IOType { NONE, RAW, SHARED_MEMORY };
  IOType io_type_;
  std::string shm_name_;
  size_t shm_offset_;
};

//==============================================================================

class OutputImpl : public InferContext::Output {
 public:
  OutputImpl(const ModelOutput& mio)
      : mio_(mio), result_format_(InferContext::Result::ResultFormat::RAW)
  {
  }
  ~OutputImpl() = default;

  const std::string& Name() const override { return mio_.name(); }
  DataType DType() const override { return mio_.data_type(); }
  bool IsShapeTensor() const override { return mio_.is_shape_tensor(); }
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
  InferContext::Result::ResultFormat result_format_;
};

//==============================================================================

class ResultImpl : public InferContext::Result {
 public:
  ResultImpl(
      const std::shared_ptr<InferContext::Output>& output, uint64_t batch_size);
  virtual ~ResultImpl() = default;

  const std::string& ModelName() const override { return model_name_; }
  int64_t ModelVersion() const override { return model_version_; }
  bool UsesSharedMemory() const { return use_shm_; }

  const std::shared_ptr<InferContext::Output> GetOutput() const override
  {
    return output_;
  }

  Error GetRawShape(std::vector<int64_t>* shape) const override;
  Error GetRaw(
      size_t batch_idx, const std::vector<uint8_t>** buf) const override;
  Error GetRaw(
      size_t batch_idx, const uint8_t** buf, size_t* byte_size) const override;
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

  void SetBatchnByteSize(const size_t s)
  {
    has_fixed_batch1_byte_size_ = true;
    batch1_byte_size_ = s / batch_size_;
  }

  void SetUsesSharedMemory(const bool use_shm) { use_shm_ = use_shm; }

  void SetBatch1Shape(const DimsList& dims)
  {
    shape_.clear();
    for (auto d : dims) {
      shape_.push_back(d);
    }
    batch1_element_count_ = GetElementCount(dims);
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
  Error SetNextRawResult(
      const uint8_t* buf, size_t size, const bool inplace,
      size_t* result_bytes);

 private:
  Error SetBatchRawResult(
      const size_t batch1_byte_size, const uint8_t* buf, size_t size,
      size_t* result_bytes);

  const std::shared_ptr<InferContext::Output> output_;
  const InferContext::Result::ResultFormat result_format_;
  const size_t batch_size_;

  bool has_fixed_batch1_byte_size_;
  size_t batch1_byte_size_;
  size_t batch1_element_count_;
  std::vector<int64_t> shape_;
  bool use_shm_;

  // If using in-place buffer for the results then 'inplace_ptrs_'
  // point to the buffer for each batch (so batch-size n will have n
  // pointers). If not using in-place buffer then a copy of the
  // results is placed into 'buffers_'. For in-place 'buffers_' is
  // used if a copy of results must be created for GetRaw(size_t,
  // const std::vector<uint8_t>**) API.
  bool inplace_;
  std::vector<const uint8_t*> inplace_ptrs_;
  mutable std::vector<std::vector<uint8_t>> buffers_;

  size_t bufs_idx_;
  std::vector<size_t> bufs_pos_;
  std::vector<size_t> bufs_byte_size_;
  std::vector<uint8_t> pending_;

  std::string model_name_;
  int64_t model_version_;

  InferResponseHeader::Output class_result_;
  std::vector<size_t> class_pos_;
};

//==============================================================================

class RequestImpl : public InferContext::Request {
 public:
  RequestImpl(const uint64_t id, InferContext::OnCompleteFn callback = nullptr)
      : callback_(std::move(callback)), id_(id)
  {
  }
  virtual ~RequestImpl() = default;

  uint64_t Id() const override { return id_; };
  void SetId(uint64_t id) { id_ = id; }

  uintptr_t RunIndex() const { return run_index_; }
  void SetRunIndex(uintptr_t idx) { run_index_ = idx; }

  RequestTimers& Timer() { return timer_; }

  // Set non-RAW results from the inference response
  Error PostRunProcessing(
      const InferResponseHeader& infer_response,
      InferContext::ResultMap* results) const;

 protected:
  // Callback function to be called once the request is completed.
  InferContext::OnCompleteFn callback_;

 private:
  // Identifier seen by user
  uint64_t id_;

  // Internal identifier for asynchronous call
  uintptr_t run_index_;

  // The timers for infer request.
  RequestTimers timer_;
};

//==============================================================================

class InferContextImpl : public InferContext {
 public:
  using ResultMap = std::map<std::string, std::unique_ptr<Result>>;

  InferContextImpl(
      const std::string& model_name, int64_t model_version,
      CorrelationID correlation_id, bool verbose);
  virtual ~InferContextImpl() {}

  const std::string& ModelName() const override { return model_name_; }
  int64_t ModelVersion() const override { return model_version_; }
  uint64_t MaxBatchSize() const override { return max_batch_size_; }
  CorrelationID CorrelationId() const override { return correlation_id_; }

  bool UsesSharedMemory(const std::string& output_name) const
  {
    if (shm_outputs_.empty()) {
      return false;
    }
    return (
        std::find(shm_outputs_.begin(), shm_outputs_.end(), output_name) !=
        shm_outputs_.end());
  }

  const std::vector<std::shared_ptr<Input>>& Inputs() const override
  {
    return inputs_;
  }
  const std::vector<std::shared_ptr<Output>>& Outputs() const override
  {
    return outputs_;
  }

  Error GetInput(
      const std::string& name, std::shared_ptr<Input>* input) const override;
  Error GetOutput(
      const std::string& name, std::shared_ptr<Output>* output) const override;

  Error SetRunOptions(const Options& options) override;
  Error GetStat(Stat* stat) const override;
  int64_t ByteSize(const DimsList& dims, DataType dtype) const override;

 protected:
  Error Init(std::unique_ptr<ServerStatusContext> sctx);

  // Update the context stat with the given timer
  Error UpdateStat(const RequestTimers& timer);

  using AsyncReqMap = std::map<uintptr_t, std::shared_ptr<Request>>;

  // map to record ongoing asynchronous requests with pointer to easy handle
  // or tag id as key
  AsyncReqMap ongoing_async_requests_;

  // Model name
  const std::string model_name_;

  // Model version
  const int64_t model_version_;

  // Requested correlation id for the inference request
  CorrelationID correlation_id_;

  // If true print verbose output
  const bool verbose_;

  // Maximum batch size supported by this context. A maximum batch
  // size indicates that the context does not support batching and so
  // only a single inference at a time can be performed.
  uint64_t max_batch_size_;

  // Requested batch size for inference request
  uint64_t batch_size_;

  // Use to assign unique identifier for each asynchronous request
  uint64_t async_request_id_;

  // The inputs and outputs
  std::vector<std::shared_ptr<Input>> inputs_;
  std::vector<std::shared_ptr<Output>> outputs_;

  // The list of ouputs that use shared memory
  std::vector<std::string> shm_outputs_;

  // Settings generated by current option
  // InferRequestHeader protobuf describing the request
  InferRequestHeader infer_request_;

  // Standalone request context used for synchronous request
  std::shared_ptr<Request> sync_request_;

  // The statistic of the current context
  Stat context_stat_;

  // worker thread that will perform the asynchronous transfer
  std::thread worker_;

  // Avoid race condition between main thread and worker thread
  std::mutex mutex_;

  // Condition variable used for waiting on asynchronous request
  std::condition_variable cv_;

  // signal for worker thread to stop
  bool exiting_;
};

}}}  // namespace nvidia::inferenceserver::client
