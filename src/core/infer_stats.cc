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

#include "src/core/infer_stats.h"

#include <time.h>
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/metric_model_reporter.h"
#include "src/core/metrics.h"
#include "src/core/tracing.h"

namespace nvidia { namespace inferenceserver {

#if 0
void
ModelInferStats::NewTrace(Trace* parent)
{
#ifdef TRTIS_ENABLE_TRACING
  if (trace_manager_ != nullptr) {
    auto ltrace_manager = reinterpret_cast<OpaqueTraceManager*>(trace_manager_);
    trace_ = nullptr;
    if (trace_manager_->using_triton_) {
      ltrace_manager->triton_create_fn_(
          reinterpret_cast<TRITONSERVER_Trace**>(&trace_), model_name_.c_str(),
          requested_model_version_, ltrace_manager->userp_);
    } else {
      ltrace_manager->create_fn_(
          reinterpret_cast<TRITONSERVER_Trace**>(&trace_), model_name_.c_str(),
          requested_model_version_, ltrace_manager->userp_);
    }
    if (trace_ != nullptr) {
      trace_->SetModelName(model_name_);
      trace_->SetModelVersion(requested_model_version_);
      if (parent != nullptr) {
        trace_->SetParentId(parent->Id());
      }
    }
  }
#endif  // TRTIS_ENABLE_TRACING
}
#endif

void
StatsAggregator::UpdateFailedInferStats(
    int device, uint64_t last_timestamp_ms, uint64_t request_duration_ns)
{
  std::lock_guard<std::mutex> lock(mu_);

  if (last_inference_ms_ < last_timestamp_ms) {
    last_inference_ms_ = last_timestamp_ms;
  }
  infer_stats_.failure_count_++;
  infer_stats_.failure_duration_ns_ += request_duration_ns;

#ifdef TRTIS_ENABLE_METRICS
  if (metric_reporter_ != nullptr) {
    metric_reporter_->MetricInferenceFailure(device).Increment();
  }
#endif  // TRTIS_ENABLE_METRICS
}

void
StatsAggregator::UpdateSuccessInferStats(
    int device, uint64_t last_timestamp_ms, uint64_t request_duration_ns,
    uint64_t queue_duration_ns, uint64_t compute_input_duration_ns,
    uint64_t compute_infer_duration_ns, uint64_t compute_output_duration_ns)
{
  std::lock_guard<std::mutex> lock(mu_);

  if (last_inference_ms_ < last_timestamp_ms) {
    last_inference_ms_ = last_timestamp_ms;
  }

  infer_stats_.success_count_++;
  infer_stats_.request_duration_ns_ += request_duration_ns;
  infer_stats_.queue_duration_ns_ += queue_duration_ns;
  infer_stats_.compute_input_duration_ns_ += compute_input_duration_ns;
  infer_stats_.compute_infer_duration_ns_ += compute_infer_duration_ns;
  infer_stats_.compute_output_duration_ns_ += compute_output_duration_ns;

#ifdef TRTIS_ENABLE_METRICS
  if (metric_reporter_ != nullptr) {
    auto compute_duration_ns = compute_input_duration_ns +
                               compute_infer_duration_ns +
                               compute_output_duration_ns;
    metric_reporter_->MetricInferenceSuccess(device).Increment();
    metric_reporter_->MetricInferenceCount(device).Increment(1);

    metric_reporter_->MetricInferenceRequestDuration(device).Increment(
        request_duration_ns / 1000);
    metric_reporter_->MetricInferenceComputeDuration(device).Increment(
        compute_duration_ns / 1000);
    metric_reporter_->MetricInferenceQueueDuration(device).Increment(
        queue_duration_ns / 1000);

    metric_reporter_->MetricInferenceLoadRatio(device).Observe(
        (double)request_duration_ns /
        std::max(1.0, (double)compute_duration_ns));
  }
#endif  // TRTIS_ENABLE_METRICS
}

void
StatsAggregator::UpdateInferBatchStats(
    int device, size_t batch_size, uint64_t compute_input_duration_ns,
    uint64_t compute_infer_duration_ns, uint64_t compute_output_duration_ns)
{
  std::lock_guard<std::mutex> lock(mu_);

  auto it = batch_stats_.find(batch_size);
  if (it == batch_stats_.end()) {
    it = batch_stats_.emplace(batch_size, InferBatchStats()).first;
  }
  it->second.count_++;
  it->second.compute_input_duration_ns_ += compute_input_duration_ns;
  it->second.compute_infer_duration_ns_ += compute_infer_duration_ns;
  it->second.compute_output_duration_ns_ += compute_output_duration_ns;

#ifdef TRTIS_ENABLE_METRICS
  if (metric_reporter_ != nullptr) {
    metric_reporter_->MetricInferenceExecutionCount(device).Increment(1);
  }
#endif  // TRTIS_ENABLE_METRICS
}

}}  // namespace nvidia::inferenceserver
