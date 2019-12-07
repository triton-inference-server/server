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

namespace nvidia { namespace inferenceserver {

class MetricModelReporter;
class ServerStatusManager;

// Updates a server stat with duration measured by a C++ scope.
class ServerStatTimerScoped {
 public:
  enum Kind {
    // Stat for status request. Duration from request to response.
    STATUS,
    // Stat for health request. Duration from request to response.
    HEALTH,
    // Stat for model control request. Duration from request to
    // response.
    MODEL_CONTROL,
    // Stat for shared memory control request. Duration from request
    // to response.
    SHARED_MEMORY_CONTROL,
    // Stat for repository request. Duration from request to response.
    REPOSITORY
  };

  // Start server timer for a given status 'kind'.
  ServerStatTimerScoped(
      const std::shared_ptr<ServerStatusManager>& status_manager, Kind kind)
      : status_manager_(status_manager), kind_(kind), enabled_(true)
  {
    clock_gettime(CLOCK_MONOTONIC, &start_);
  }

  // Stop the timer and record the duration, unless reporting has been
  // disabled.
  ~ServerStatTimerScoped();

  // Enable/Disable reporting for this timer. By default reporting is
  // enabled and so the server status is updated when this object is
  // destructed. Reporting may be enabled/disabled multiple times
  // while the timer is running without affecting the duration.
  void SetEnabled(bool enabled) { enabled_ = enabled; }

 private:
  std::shared_ptr<ServerStatusManager> status_manager_;
  const Kind kind_;
  bool enabled_;
  struct timespec start_;
};

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
#ifdef TRTIS_ENABLE_STATS
  // Start model-specific timer for 'model_name' and a given status
  // 'kind'.
  ModelInferStats(
      const std::shared_ptr<ServerStatusManager>& status_manager,
      const std::string& model_name)
      : status_manager_(status_manager), model_name_(model_name),
        requested_model_version_(-1), batch_size_(0), gpu_device_(-1),
        failed_(false), execution_count_(0), extra_queue_duration_(0),
        extra_compute_duration_(0), trace_manager_(nullptr), trace_(nullptr),
        timestamps_((size_t)TimestampKind::COUNT__)
  {
    memset(&timestamps_[0], 0, sizeof(struct timespec) * timestamps_.size());
  }

  // Report collected statistics.
  void Report();

  // Mark inferencing request as failed / not-failed.
  void SetFailed(bool failed) { failed_ = failed; }

  // Set the model version explicitly requested for the inference, or
  // -1 if latest version was requested.
  void SetRequestedVersion(int64_t v) { requested_model_version_ = v; }

  // Set the metric reporter for the model.
  void SetMetricReporter(const std::shared_ptr<MetricModelReporter> m)
  {
    metric_reporter_ = m;
  }

  // Set batch size for the inference stats.
  void SetBatchSize(size_t bs) { batch_size_ = bs; }

  // Set CUDA GPU device index where inference was performed.
  void SetGPUDevice(int idx) { gpu_device_ = idx; }

  // Set the number of model executions that were performed for this
  // inference request. Can be zero if this request was dynamically
  // batched with another request (in dynamic batch case only one of
  // the batched requests will count the execution).
  void SetModelExecutionCount(uint32_t count) { execution_count_ = count; }

  // Set the trace manager associated with the inference.
  void SetTraceManager(TRTSERVER_TraceManager* tm) { trace_manager_ = tm; }

  // Get the trace manager associated with the inference.
  TRTSERVER_TraceManager* GetTraceManager() const { return trace_manager_; }

  // Create a trace object associated to the inference.
  // Optional 'parent' can be provided if the trace object has a parent.
  // Model name, model version, and trace manager should be set before calling
  // this function. And each ModelInferStats instance should not call this
  // function more than once.
  void NewTrace(TRTSERVER_Trace* parent = nullptr);

  // Get the trace object associated to the inference.
  // Return nullptr if the inference will not be traced or if NewTrace()
  // has not been called.
  TRTSERVER_Trace* GetTrace() const { return trace_; }

  // Include queue time from another stat into this stat's queue time.
  void IncrementQueueDuration(const ModelInferStats& other);

  // Include compute time from another stat into this stat's compute
  // time.
  void IncrementComputeDuration(const ModelInferStats& other);

#else
  // Start model-specific timer for 'model_name' and a given status
  // 'kind'.
  ModelInferStats() : timestamps_((size_t)TimestampKind::COUNT__)
  {
    memset(&timestamps_[0], 0, sizeof(struct timespec) * timestamps_.size());
  }

#endif  // TRTIS_ENABLE_STATS

  // Get the timestamp for a kind.
  const struct timespec& Timestamp(TimestampKind kind) const
  {
    return timestamps_[(size_t)kind];
  }

  // Set a timestamp to the current time. Return the timestamp.
  const struct timespec& CaptureTimestamp(TimestampKind kind)
  {
    struct timespec& ts = timestamps_[(size_t)kind];
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts;
  }

 private:
#ifdef TRTIS_ENABLE_STATS
  uint64_t Duration(
      ModelInferStats::TimestampKind start_kind,
      ModelInferStats::TimestampKind end_kind) const;

  std::shared_ptr<ServerStatusManager> status_manager_;
  std::shared_ptr<MetricModelReporter> metric_reporter_;
  const std::string model_name_;
  int64_t requested_model_version_;
  size_t batch_size_;
  int gpu_device_;
  bool failed_;

  uint32_t execution_count_;

  uint64_t extra_queue_duration_;
  uint64_t extra_compute_duration_;

  // The trace manager associated with these stats. This object is not owned by
  // this ModelInferStats object and so is not destroyed by this object.
  TRTSERVER_TraceManager* trace_manager_;

  // The trace associated with these stats. This object is not owned by
  // this ModelInferStats object and so is not destroyed by this object.
  TRTSERVER_Trace* trace_;
#endif  // TRTIS_ENABLE_STATS

  std::vector<struct timespec> timestamps_;
};

// Manage access and updates to server status information.
class ServerStatusManager {
 public:
  // Create a manager for server status
  explicit ServerStatusManager(const std::string& server_version);

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
  Status Get(
      ServerStatus* server_status, const std::string& server_id,
      ServerReadyState server_ready_state, uint64_t server_uptime_ns) const;

  // Get the server status and the status for a single model.
  Status Get(
      ServerStatus* server_status, const std::string& server_id,
      ServerReadyState server_ready_state, uint64_t server_uptime_ns,
      const std::string& model_name) const;

  // Add a duration to the Server Stat specified by 'kind'.
  void UpdateServerStat(uint64_t duration, ServerStatTimerScoped::Kind kind);

  // Add durations to Infer stats for a failed inference request.
  void UpdateFailedInferStats(
      const std::string& model_name, const int64_t model_version,
      size_t batch_size, uint64_t last_timestamp_ms,
      uint64_t request_duration_ns);

  // Add durations to Infer stats for a successful inference request.
  void UpdateSuccessInferStats(
      const std::string& model_name, const int64_t model_version,
      size_t batch_size, uint32_t execution_cnt, uint64_t last_timestamp_ms,
      uint64_t request_duration_ns, uint64_t queue_duration_ns,
      uint64_t compute_duration_ns);

 private:
  mutable std::mutex mu_;
  ServerStatus server_status_;
};
}}  // namespace nvidia::inferenceserver
