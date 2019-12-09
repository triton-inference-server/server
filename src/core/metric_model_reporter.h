// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "src/core/model_config.h"
#include "src/core/status.h"

#ifdef TRTIS_ENABLE_METRICS
#include "prometheus/registry.h"
#endif  // TRTIS_ENABLE_METRICS

namespace nvidia { namespace inferenceserver {

//
// Interface for a metric reporter for a given version of a model.
//
class MetricModelReporter {
 public:
  MetricModelReporter(
      const std::string& model_name, int64_t model_version,
      const MetricTagsMap& model_tags);

  // Get the name of model for which metrics are being reported.
  const std::string& ModelName() const { return model_name_; }

  // Get the version of model for which metrics are being reported.
  int64_t ModelVersion() const { return model_version_; }

  // Get a metric for the backend specialized for the given GPU index
  // (if -1 then return non-specialized version of the metric).
#ifdef TRTIS_ENABLE_METRICS
#ifdef TRTIS_ENABLE_STATS
  prometheus::Counter& MetricInferenceSuccess(int gpu_device) const;
  prometheus::Counter& MetricInferenceFailure(int gpu_device) const;
  prometheus::Counter& MetricInferenceCount(int gpu_device) const;
  prometheus::Counter& MetricInferenceExecutionCount(int gpu_device) const;
  prometheus::Counter& MetricInferenceRequestDuration(int gpu_device) const;
  prometheus::Counter& MetricInferenceComputeDuration(int gpu_device) const;
  prometheus::Counter& MetricInferenceQueueDuration(int gpu_device) const;
  prometheus::Histogram& MetricInferenceLoadRatio(int gpu_device) const;
#endif  // TRTIS_ENABLE_STATS
#endif  // TRTIS_ENABLE_METRICS

 private:
  const std::string model_name_;
  const int64_t model_version_;
  const MetricTagsMap model_tags_;

#ifdef TRTIS_ENABLE_METRICS
#ifdef TRTIS_ENABLE_STATS
  void GetMetricLabels(
      std::map<std::string, std::string>* labels, const int gpu_device) const;
  prometheus::Counter& GetCounterMetric(
      std::map<int, prometheus::Counter*>& metrics,
      prometheus::Family<prometheus::Counter>& family,
      const int gpu_device) const;

  mutable std::map<int, prometheus::Counter*> metric_inf_success_;
  mutable std::map<int, prometheus::Counter*> metric_inf_failure_;
  mutable std::map<int, prometheus::Counter*> metric_inf_count_;
  mutable std::map<int, prometheus::Counter*> metric_inf_exec_count_;
  mutable std::map<int, prometheus::Counter*> metric_inf_request_duration_us_;
  mutable std::map<int, prometheus::Counter*> metric_inf_compute_duration_us_;
  mutable std::map<int, prometheus::Counter*> metric_inf_queue_duration_us_;
  mutable std::map<int, prometheus::Histogram*> metric_inf_load_ratio_;
#endif  // TRTIS_ENABLE_STATS
#endif  // TRTIS_ENABLE_METRICS
};

}}  // namespace nvidia::inferenceserver
