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

#include <time.h>
#include <map>
#include <mutex>
#include "src/core/constants.h"
#include "src/core/metric_model_reporter.h"
#include "src/core/status.h"
#include "src/core/tritonserver.h"

namespace nvidia { namespace inferenceserver {

#ifdef TRTIS_ENABLE_STATS

//
// InferenceStatsAggregator
//
// A statistics aggregator.
//
class InferenceStatsAggregator {
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
  InferenceStatsAggregator() : last_inference_ms_(0) {}

  // Set metric reporter for the model statistics
  void SetMetricReporter(
      const std::shared_ptr<MetricModelReporter>& metric_reporter)
  {
    metric_reporter_ = metric_reporter;
  }

  uint64_t LastInferenceMs() const { return last_inference_ms_; }
  const InferStats& ImmutableInferStats() const { return infer_stats_; }
  const std::map<size_t, InferBatchStats>& ImmutableInferBatchStats() const
  {
    return batch_stats_;
  }

  // Add durations to Infer stats for a failed inference request.
  void UpdateFailure(
      const size_t batch_size, const uint64_t request_start_ns,
      const uint64_t request_end_ns);

  // Add durations to infer stats for a successful inference request.
  void UpdateSuccess(
      const size_t batch_size, const uint64_t request_start_ns,
      const uint64_t queue_start_ns, const uint64_t compute_start_ns,
      const uint64_t compute_input_end_ns,
      const uint64_t compute_output_start_ns, const uint64_t compute_end_ns,
      const uint64_t request_end_ns);

  // Add durations to batch infer stats for a batch execution.
  // 'success_request_count' is the number of sucess requests in the
  // batch that have infer_stats attached.
  void UpdateInferBatchStats(
      const size_t batch_size, const uint64_t compute_start_ns,
      const uint64_t compute_input_end_ns,
      const uint64_t compute_output_start_ns, const uint64_t compute_end_ns);

 private:
  std::mutex mu_;
  uint64_t last_inference_ms_;
  InferStats infer_stats_;
  std::map<size_t, InferBatchStats> batch_stats_;
  std::shared_ptr<MetricModelReporter> metric_reporter_;
};

#endif  // TRTIS_ENABLE_STATS

//
// Macros to set infer stats.
//
#ifdef TRTIS_ENABLE_STATS
#define INFER_STATS_SET_TIMESTAMP(TS_NS) \
  {                                      \
    struct timespec ts;                  \
    clock_gettime(CLOCK_MONOTONIC, &ts); \
    TS_NS = TIMESPEC_TO_NANOS(ts);       \
  }
#define INFER_STATS_DECL_TIMESTAMP(TS_NS) \
  uint64_t TS_NS;                         \
  INFER_STATS_SET_TIMESTAMP(TS_NS);
#else
#define INFER_STATS_DECL_TIMESTAMP(TS_NS)
#define INFER_STATS_SET_TIMESTAMP(TS_NS)
#endif  // TRTIS_ENABLE_STATS

}}  // namespace nvidia::inferenceserver
