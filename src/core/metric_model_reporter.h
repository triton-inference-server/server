// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifdef TRITON_ENABLE_METRICS
#include "prometheus/registry.h"
#endif  // TRITON_ENABLE_METRICS

namespace nvidia { namespace inferenceserver {

//
// Interface for a metric reporter for a given version of a model.
//
class MetricModelReporter {
 public:
#ifdef TRITON_ENABLE_METRICS
  static Status Create(
      const std::string& model_name, const int64_t model_version,
      const int device, const MetricTagsMap& model_tags,
      std::shared_ptr<MetricModelReporter>* metric_model_reporter);

  ~MetricModelReporter();

  // Get a metric for the backend specialized for the given model,
  // version and GPU index.
  prometheus::Counter& MetricInferenceSuccess() const
  {
    return *metric_inf_success_;
  }
  prometheus::Counter& MetricInferenceFailure() const
  {
    return *metric_inf_failure_;
  }
  prometheus::Counter& MetricInferenceCount() const
  {
    return *metric_inf_count_;
  }
  prometheus::Counter& MetricInferenceExecutionCount() const
  {
    return *metric_inf_exec_count_;
  }
  prometheus::Counter& MetricInferenceRequestDuration() const
  {
    return *metric_inf_request_duration_us_;
  }
  prometheus::Counter& MetricInferenceQueueDuration() const
  {
    return *metric_inf_queue_duration_us_;
  }
  prometheus::Counter& MetricInferenceComputeInputDuration() const
  {
    return *metric_inf_compute_input_duration_us_;
  }
  prometheus::Counter& MetricInferenceComputeInferDuration() const
  {
    return *metric_inf_compute_infer_duration_us_;
  }
  prometheus::Counter& MetricInferenceComputeOutputDuration() const
  {
    return *metric_inf_compute_output_duration_us_;
  }

 private:
  MetricModelReporter(
      const std::string& model_name, const int64_t model_version,
      const int device, const MetricTagsMap& model_tags);

  static void GetMetricLabels(
      std::map<std::string, std::string>* labels, const std::string& model_name,
      const int64_t model_version, const int device,
      const MetricTagsMap& model_tags);
  prometheus::Counter* CreateCounterMetric(
      prometheus::Family<prometheus::Counter>& family,
      const std::map<std::string, std::string>& labels);

  prometheus::Counter* metric_inf_success_;
  prometheus::Counter* metric_inf_failure_;
  prometheus::Counter* metric_inf_count_;
  prometheus::Counter* metric_inf_exec_count_;
  prometheus::Counter* metric_inf_request_duration_us_;
  prometheus::Counter* metric_inf_queue_duration_us_;
  prometheus::Counter* metric_inf_compute_input_duration_us_;
  prometheus::Counter* metric_inf_compute_infer_duration_us_;
  prometheus::Counter* metric_inf_compute_output_duration_us_;
#endif  // TRITON_ENABLE_METRICS
};

}}  // namespace nvidia::inferenceserver
