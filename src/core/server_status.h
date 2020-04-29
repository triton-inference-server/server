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

#include <time.h>
#include <mutex>
#include "src/core/model_config.pb.h"
#include "src/core/server_status.pb.h"
#include "src/core/status.h"
#include "src/core/tracing.h"

namespace nvidia { namespace inferenceserver {

class MetricModelReporter;
class ServerStatusManager;
class OpaqueTraceManager;
class Trace;

// FIXME move the trace handling to infer request directly.
#if 0
// Stats collector for an inference request. If TRTIS_ENABLE_STATS is not
// defined, it will only records timestamps that may be used by other objects
// along the inference pipeline (i.e. scheduler)
class ModelInferStats {
 public:
  enum class TimestampKind {
    kRequestStart,        // Start request processing
    kQueueStart,          // Request enters the queue
    kComputeStart,        // Request leaves queue and starts compute
    kComputeInputEnd,     // Requests finishes preparing inputs
    kComputeOutputStart,  // Request starts processing outputs
    kComputeEnd,          // Request completes compute
    kRequestEnd,          // Done with request processing
    COUNT__
  };

 public:
  // Start model-specific timer for 'model_name' and a given status
  // 'kind'.
  ModelInferStats(
      const std::shared_ptr<ServerStatusManager>& status_manager,
      const std::string& model_name)
      : status_manager_(status_manager), model_name_(model_name),
        requested_model_version_(-1), batch_size_(0), gpu_device_(-1),
        failed_(false), execution_count_(0), extra_queue_duration_(0),
        extra_compute_duration_(0), extra_compute_input_duration_(0),
        extra_compute_infer_duration_(0), extra_compute_output_duration_(0),
        trace_manager_(nullptr), trace_(nullptr),
        timestamps_((size_t)TimestampKind::COUNT__)
  {
    memset(&timestamps_[0], 0, sizeof(struct timespec) * timestamps_.size());
  }

  // Set the trace manager associated with the inference.
  void SetTraceManager(OpaqueTraceManager* tm) { trace_manager_ = tm; }

  // Get the trace manager associated with the inference.
  OpaqueTraceManager* GetTraceManager() const { return trace_manager_; }

  // Create a trace object associated to the inference.
  // Optional 'parent' can be provided if the trace object has a parent.
  // Model name, model version, and trace manager should be set before calling
  // this function. And each ModelInferStats instance should not call this
  // function more than once.
  void NewTrace(Trace* parent = nullptr);

  // Get the trace object associated to the inference.
  // Return nullptr if the inference will not be traced or if NewTrace()
  // has not been called.
  Trace* GetTrace() const { return trace_; }


 private:

  // The trace manager associated with these stats. This object is not owned by
  // this ModelInferStats object and so is not destroyed by this object.
  OpaqueTraceManager* trace_manager_;

  // The trace associated with these stats. This object is not owned by
  // this ModelInferStats object and so is not destroyed by this object.
  Trace* trace_;
};
#endif

// FIXME remove the need for status manager. Model related info should be
// obtained from model manager directly.
// Manage access and updates to server status information.
class ServerStatusManager {
 public:
  // Create a manager for server status
  explicit ServerStatusManager() = default;

  // Initialize status for a model.
  Status InitForModel(
      const std::string& model_name, const ModelConfig& model_config);

  // Update model config for an existing model.
  Status UpdateConfigForModel(
      const std::string& model_name, const ModelConfig& model_config);

  // Update the version ready state and reason for an existing model.
  Status SetModelVersionReadyState(
      const std::string& model_name, int64_t version, ModelReadyState state,
      const ModelReadyStateReason& state_reason);

  // Get the entire server status, including status for all models.
  Status Get(ServerStatus* server_status) const;

  // Get the server status and the status for a single model.
  Status Get(ServerStatus* server_status, const std::string& model_name) const;

 private:
  mutable std::mutex mu_;
  ServerStatus server_status_;
};

// FIXME good place? move to backend maybe
// A stats aggregator used for one backend.
class StatsAggregator {
 public:
  struct InferStats {
    InferStats()
        : failure_count_(0), failure_duration_ns_(0), success_count_(0),
          request_duration_ns_(0), queue_duration_ns_(0),
          compute_input_duration_ns_(0), compute_infer_duration_ns_(0),
          compute_output_duration_ns_(0)
    {
    }
    uint64_t failure_count_;
    uint64_t failure_duration_ns_;

    uint64_t success_count_;
    uint64_t request_duration_ns_;
    uint64_t queue_duration_ns_;
    uint64_t compute_input_duration_ns_;
    uint64_t compute_infer_duration_ns_;
    uint64_t compute_output_duration_ns_;
  };

  struct InferBatchStats {
    InferBatchStats()
        : count_(0), compute_input_duration_ns_(0),
          compute_infer_duration_ns_(0), compute_output_duration_ns_(0)
    {
    }
    uint64_t count_;
    uint64_t compute_input_duration_ns_;
    uint64_t compute_infer_duration_ns_;
    uint64_t compute_output_duration_ns_;
  };

  // Create an aggregator for model statistics
  StatsAggregator() : last_inference_ms_(0) {}

  // Create an aggregator with metric reporter attached for model statistics
  StatsAggregator(const std::shared_ptr<MetricModelReporter>& metric_reporter)
      : last_inference_ms_(0), metric_reporter_(metric_reporter)
  {
  }

  uint64_t LastInferenceMs() const { return last_inference_ms_; }
  const InferStats& ImmutableInferStats() const { return infer_stats_; }
  const std::map<size_t, InferBatchStats>& ImmutableInferBatchStats() const
  {
    return batch_stats_;
  }

  // FIXME passing device is somewhat confusing here, that is for updating
  // metrics properly, but is here the good place for updating matrics?
  //
  // Add durations to Infer stats for a failed inference request.
  void UpdateFailedInferStats(
      int device, uint64_t last_timestamp_ms, uint64_t request_duration_ns);

  // Add durations to infer stats for a successful inference request.
  void UpdateSuccessInferStats(
      int device, uint64_t last_timestamp_ms, uint64_t request_duration_ns,
      uint64_t queue_duration_ns, uint64_t compute_input_duration_ns,
      uint64_t compute_infer_duration_ns, uint64_t compute_output_duration_ns);

  // Add durations to batch infer stats for a batch execution.
  // 'success_request_count' is the number of sucess requests in the batch that
  // have infer_stats attached.
  void UpdateInferBatchStats(
      int device, size_t batch_size, uint64_t compute_input_duration_ns,
      uint64_t compute_infer_duration_ns, uint64_t compute_output_duration_ns);

 private:
  std::mutex mu_;
  uint64_t last_inference_ms_;
  InferStats infer_stats_;
  std::map<size_t, InferBatchStats> batch_stats_;
  std::shared_ptr<MetricModelReporter> metric_reporter_;
};
}}  // namespace nvidia::inferenceserver
