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
#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include "src/core/infer_response.h"
#include "src/core/infer_stats.h"
#include "src/core/infer_trace.h"
#include "src/core/memory.h"
#include "src/core/model_config.h"
#include "src/core/response_allocator.h"
#include "src/core/status.h"
#include "src/core/tritonserver.h"

namespace nvidia { namespace inferenceserver {

class InferenceBackend;
class InferenceServer;
class MetricModelReporter;

//
// An inference request. A request can be used multiple times for
// inference but before each inference run, PrepareForInference() must
// be called to verify and prepare the request. Verification involves
// ensuring that any changes made since the last inference are
// valid. Preparing involves removing/resetting any state left over
// from the previous inference.
//
class InferenceRequest {
 public:
  // Input tensor
  class Input {
   public:
    Input();
    Input(
        const std::string& name, const DataType datatype,
        const std::vector<int64_t>& shape);
    Input(
        const std::string& name, const DataType datatype, const int64_t* shape,
        const uint64_t dim_count);

    // The name of the input tensor. There is no mutable operator for
    // the name because it is used in a InferenceRequest map and a
    // mutable method would allow it to get out-of-sync.
    const std::string& Name() const { return name_; }

    // Data type of the input tensor.
    DataType DType() const { return datatype_; }

    // The original shape of the input tensor.
    const std::vector<int64_t>& OriginalShape() const
    {
      return original_shape_;
    }

    // The shape of the input tensor after normalization. This shape
    // is the original shape modified as required/expected by
    // inference processing.
    const std::vector<int64_t>& Shape() const { return shape_; }
    std::vector<int64_t>* MutableShape() { return &shape_; }

    // The data for this input.
    const std::shared_ptr<Memory>& Data() const { return data_; }

    // Set the data for this input. Error if input already has some
    // data.
    Status SetData(const std::shared_ptr<Memory>& data);

    // Append a new buffer of data to this input.
    Status AppendData(
        const void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
        int64_t memory_type_id);

    // Remove all existing data for the input.
    Status RemoveAllData();

    // Get the number of buffers containing the input tensor data.
    size_t DataBufferCount() const { return data_->BufferCount(); }

    // Get the 'idx' buffer containing a contiguous chunk of bytes for
    // the input. Return error is 'idx' refers to a buffer that does
    // not exist. Return a pointer to the chunk in 'base' and the
    // size of the chunk in 'byte_size'. 'memory_type' acts as
    // both input and output. On input 'memory_type' is the buffer
    // memory type preferred by the function caller. On return
    // 'memory_type' gives the actual memory type of the chunk pointed
    // to by 'base'.  'memory_type_id' acts as both input and
    // output. On input 'memory_type_id' is the buffer memory type id
    // preferred by the function caller.  On return 'memory_type_id'
    // gives the actual memory type id of the chunk pointed to by
    // 'base'.
    Status DataBuffer(
        const size_t idx, const void** base, size_t* byte_size,
        TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id) const;

   private:
    DISALLOW_COPY_AND_ASSIGN(Input);
    friend std::ostream& operator<<(
        std::ostream& out, const InferenceRequest::Input& input);

    std::string name_;
    DataType datatype_;
    std::vector<int64_t> original_shape_;
    std::vector<int64_t> shape_;
    std::shared_ptr<Memory> data_;
  };

  // InferenceRequest
  //
  // The two constructors are identical except one takes backend as a
  // shared pointer and the other as a raw pointer. The shared pointer
  // version is the primary one and acts to keep the backend alive as
  // long as the request is in flight. The raw pointer version is used
  // only for cases where the backend itself is issuing a request
  // (e.g. warmup) and no shared pointer version of the backend exists
  // (because we aren't using shared_from_this).
  InferenceRequest(
      const std::shared_ptr<InferenceBackend>& backend,
      const int64_t requested_model_version)
      : InferenceRequest(backend.get(), requested_model_version)
  {
    backend_shared_ = backend;
  }

  InferenceRequest(
      InferenceBackend* backend, const int64_t requested_model_version)
      : needs_normalization_(true), backend_raw_(backend),
        requested_model_version_(requested_model_version), flags_(0),
        correlation_id_(0), batch_size_(0), priority_(0), timeout_us_(0)
  {
  }

  const std::string& ModelName() const;
  int64_t RequestedModelVersion() const { return requested_model_version_; }
  int64_t ActualModelVersion() const;

  const std::string& Id() const { return id_; }
  void SetId(const std::string& i) { id_ = i; }

  uint32_t Flags() const { return flags_; }
  void SetFlags(uint32_t f) { flags_ = f; }

  uint64_t CorrelationId() const { return correlation_id_; }
  void SetCorrelationId(uint64_t c) { correlation_id_ = c; }

  // The batch size of the request, as understood by Triton. A
  // batch-size of 0 indicates that the model doesn't support batching
  // in a way that Triton understands.  Batch size is not set
  // explicitly so there is no setter for it. It is set when the
  // request is normalized.
  uint32_t BatchSize() const { return batch_size_; }

  uint32_t Priority() const { return priority_; }
  void SetPriority(uint32_t p);

  uint64_t TimeoutMicroseconds() const { return timeout_us_; }
  void SetTimeoutMicroseconds(uint64_t t) { timeout_us_ = t; }

#ifdef TRTIS_ENABLE_TRACING
  const std::unique_ptr<InferenceTrace>& Trace() const { return trace_; }
  std::unique_ptr<InferenceTrace>* MutableTrace() { return &trace_; }
  void SetTrace(std::unique_ptr<InferenceTrace>&& trace)
  {
    trace_ = std::move(trace);
  }
#endif  // TRTIS_ENABLE_TRACING

  // The original inputs are the inputs added to the request before
  // the inference execution (that is before
  // TRITONSERVER_ServerInferAsync is called). Once execution has
  // started the original inputs should not be modified until
  // execution completes (and those modifications will apply to the
  // next inference execution).
  Status MutableOriginalInput(const std::string& name, Input** input);
  std::unordered_map<std::string, Input>* MutableOriginalInputs()
  {
    return &original_inputs_;
  }
  const std::unordered_map<std::string, Input>& OriginalInputs() const
  {
    return original_inputs_;
  }

  // The override inputs are the inputs added to the request after
  // inference execution has started (that is after
  // TRITONSERVER_ServerInferAsync or equivalent is called). During
  // inference processing, if Triton needs to change an original input
  // it will add an override instead of changing the original. Triton
  // will also use an override if it needs to add a new input to the
  // request. Overrides are recorded as shared_ptr so that the same
  // override can be used efficiently multiple times or even in
  // multiple requests simultaneously. Must be careful not to modify
  // an override input if it is being shared unless you want that
  // change to be reflected in all requests that hold that override
  // input. Override inputs within a specific request are not
  // persisted across inference calls.
  std::unordered_map<std::string, std::shared_ptr<Input>>*
  MutableOverrideInputs()
  {
    return &override_inputs_;
  }
  const std::unordered_map<std::string, std::shared_ptr<Input>>&
  OverrideInputs() const
  {
    return override_inputs_;
  }

  // Get an input taking into account both original inputs and
  // overrides. If an override input is available use it, otherwise
  // use the original input. Accessing inputs via this method is not
  // valid until after PrepareForInference is called.
  Status ImmutableInput(const std::string& name, const Input** input) const;
  const std::unordered_map<std::string, Input*>& ImmutableInputs() const
  {
    return inputs_;
  }

  // The original requested outputs are the requested outputs added to
  // the request before the inference execution (that is before
  // TRITONSERVER_ServerInferAsync is called). Once execution has
  // started the original requested outputs should not be modified
  // until execution completes (and those modifications will apply to
  // the next inference execution).
  const std::set<std::string>& OriginalRequestedOutputs() const
  {
    return original_requested_outputs_;
  }

  // Get the requested outputs that should be used during
  // inference. Accessing outputs via this method is not valid until
  // after PrepareForInference is called.
  const std::set<std::string>& ImmutableRequestedOutputs() const
  {
    return (requested_outputs_.empty()) ? original_requested_outputs_
                                        : requested_outputs_;
  }

  // Get the response factory.
  const InferenceResponseFactory& ResponseFactory() const
  {
    return response_factory_;
  }

  // Add an original input to the request. If 'input' is non-null
  // return a pointer to the newly added input.
  Status AddOriginalInput(
      const std::string& name, const DataType datatype, const int64_t* shape,
      const uint64_t dim_count, Input** input = nullptr);
  Status AddOriginalInput(
      const std::string& name, const DataType datatype,
      const std::vector<int64_t>& shape, Input** input = nullptr);

  // Remove a single original input or all inputs.
  Status RemoveOriginalInput(const std::string& name);
  Status RemoveAllOriginalInputs();

  // Add an override input to the request. If 'input' is non-null
  // return a pointer to the newly added input.
  Status AddOverrideInput(
      const std::string& name, const DataType datatype,
      const std::vector<int64_t>& shape,
      std::shared_ptr<Input>* input = nullptr);

  // Add an override input to the request.
  Status AddOverrideInput(const std::shared_ptr<Input>& input);

  // Request an original requested output.
  Status AddOriginalRequestedOutput(const std::string& name);

  // Remove a single original requested output or all requested
  // outputs.
  Status RemoveOriginalRequestedOutput(const std::string& name);
  Status RemoveAllOriginalRequestedOutputs();

  // Initialize the release callback for the request.
  Status SetReleaseCallback(
      TRITONSERVER_InferenceRequestReleaseFn_t release_fn, void* release_userp)
  {
    release_fn_ = release_fn;
    release_userp_ = release_userp;
    return Status::Success;
  }

  // Initialize the response factory that is to be used with any
  // responses produced for this request.
  Status SetResponseCallback(
      const ResponseAllocator* allocator, void* alloc_userp,
      TRITONSERVER_InferenceResponseCompleteFn_t response_fn,
      void* response_userp)
  {
    response_factory_ = InferenceResponseFactory(
        backend_shared_, id_, allocator, alloc_userp, response_fn,
        response_userp);
    return Status::Success;
  }

  // Add a callback to be invoked on releasing the request object from Triton.
  // Multile callbacks can be added by calling this function in order,
  // and they will be invoked in reversed order.
  Status AddInternalReleaseCallback(std::function<void()>&& callback)
  {
    release_callbacks_.emplace_back(std::move(callback));
    return Status::Success;
  }

  // Prepare this request for inference.
  Status PrepareForInference();

  // Run this inference request using the backend associated with the
  // request. If Status::Success is returned then the call has taken
  // ownership of the request object and so 'request' will be
  // nullptr. If non-success is returned then the caller still retains
  // ownership of 'request'.
  static Status Run(std::unique_ptr<InferenceRequest>& request);

  // Send an error response for this request. If 'status' is Success
  // then no response is sent and the request is not released (even if
  // 'release_request' is true). If 'release_request' is true then the
  // release callback is called for this request and ownership is
  // given to the callback. Thus, if 'release_request' is true
  // 'request' is returned as nullptr.
  static void RespondIfError(
      std::unique_ptr<InferenceRequest>& request, const Status& status,
      const bool release_request = false);

  // Send an error response to a set of 'requests'. If 'status' is
  // Success then no responses are sent and the requests are not
  // released (even if 'release_request' is true). If
  // 'release_request' is true then the release callback is called for
  // each request, and the request ownership is given to the
  // callback. Thus, if 'release_request' is true 'requests' is
  // returned with all nullptrs.
  static void RespondIfError(
      std::vector<std::unique_ptr<InferenceRequest>>& requests,
      const Status& status, const bool release_requests = false);

  // Release the request. Call the release callback and transfer
  // ownership of the request to the callback. On return 'request' is
  // nullptr.
  static void Release(std::unique_ptr<InferenceRequest>&& request);

  // Create a copy of 'from' suitable for use as a "null" request as
  // required for the direct sequence batcher. The returned copy will
  // contain only the minimum content required for a null request.
  static InferenceRequest* CopyAsNull(const InferenceRequest& from);

  uint64_t QueueStartNs() const { return queue_start_ns_; }
  uint64_t CaptureQueueStartNs()
  {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    queue_start_ns_ = TIMESPEC_TO_NANOS(ts);
    return queue_start_ns_;
  }

#ifdef TRTIS_ENABLE_STATS
  uint64_t RequestStartNs() const { return request_start_ns_; }
  uint64_t CaptureRequestStartNs()
  {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    request_start_ns_ = TIMESPEC_TO_NANOS(ts);
    return request_start_ns_;
  }

  // Report the statistics to stats collectors associated with the request.
  // Duration and timestamps provide two granularities for stats collectors.
  void ReportStatistics(
      MetricModelReporter* metric_reporter, bool success,
      const uint64_t compute_start_ns, const uint64_t compute_input_end_ns,
      const uint64_t compute_output_start_ns, const uint64_t compute_end_ns);

  // Statistics for each request are aggregated into the corresponding
  // backend's statistics. Optionally this function may be used to
  // add an additional aggregator where statistics are also aggregated.
  void SetSecondaryStatsAggregator(
      InferenceStatsAggregator* secondary_stats_aggregator)
  {
    secondary_stats_aggregator_ = secondary_stats_aggregator;
  }
#endif  // TRTIS_ENABLE_STATS

 private:
  DISALLOW_COPY_AND_ASSIGN(InferenceRequest);
  friend std::ostream& operator<<(
      std::ostream& out, const InferenceRequest& request);

  Status Normalize();

  // Has anything in the request potentially changed in a way that
  // causes normalization to be required when preparing the request
  // for inference.
  bool needs_normalization_;

  // The backend associated with this request. For most requests
  // backend_shared_ will be non-null and will act to keep the backend
  // alive as long as this request is live. In this case backend_raw_
  // will be the raw pointer from the shared pointer. For cases where
  // the backend itself created the request (like running requests for
  // warmup), backend_shared_ will be nullptr, but backend_raw_ will
  // still be defined. Thus backend_raw_ is always defined and should
  // always to used to access the backend.
  std::shared_ptr<InferenceBackend> backend_shared_;
  InferenceBackend* backend_raw_;

  // The model version as requested and based on version policy the
  // specific version that is actually used for inference.
  int64_t requested_model_version_;
  int64_t actual_model_version_;

  std::string id_;

  uint32_t flags_;
  uint64_t correlation_id_;
  uint32_t batch_size_;
  uint32_t priority_;
  uint64_t timeout_us_;

  std::unordered_map<std::string, Input> original_inputs_;
  std::unordered_map<std::string, std::shared_ptr<Input>> override_inputs_;
  std::unordered_map<std::string, Input*> inputs_;
  std::set<std::string> original_requested_outputs_;

  // requested_outputs_ is to be used post-normalization. It will be
  // empty unless it differs from original_requested_outputs_, so
  // typically should access it through ImmutableRequestedOutputs.
  std::set<std::string> requested_outputs_;

  // The release function and user pointer for this request.
  TRITONSERVER_InferenceRequestReleaseFn_t release_fn_;
  void* release_userp_;

  // Additional release callbacks invoked before 'release_fn_'.
  std::vector<std::function<void()>> release_callbacks_;

  // The response factory associated with this request.
  InferenceResponseFactory response_factory_;

  // Request timestamps. Queue start is needed for schedulers even
  // when statistics are not being collected.
  uint64_t queue_start_ns_;

#ifdef TRTIS_ENABLE_STATS
  uint64_t request_start_ns_;
  InferenceStatsAggregator* secondary_stats_aggregator_ = nullptr;
#endif  // TRTIS_ENABLE_STATS

#ifdef TRTIS_ENABLE_TRACING
  // Inference trace associated with this request.
  std::unique_ptr<InferenceTrace> trace_;
#endif  // TRTIS_ENABLE_TRACING
};

std::ostream& operator<<(std::ostream& out, const InferenceRequest& request);
std::ostream& operator<<(
    std::ostream& out, const InferenceRequest::Input& input);

}}  // namespace nvidia::inferenceserver
