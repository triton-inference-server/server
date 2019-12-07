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

#include "src/core/metric_model_reporter.h"

#include "src/core/constants.h"
#include "src/core/metrics.h"

namespace nvidia { namespace inferenceserver {

MetricModelReporter::MetricModelReporter(
    const std::string& model_name, int64_t model_version,
    const MetricTagsMap& model_tags)
    : model_name_(model_name), model_version_(model_version),
      model_tags_(model_tags)
{
}

#ifdef TRTIS_ENABLE_METRICS
#ifdef TRTIS_ENABLE_STATS

void
MetricModelReporter::GetMetricLabels(
    std::map<std::string, std::string>* labels, const int gpu_device) const
{
  labels->insert(std::map<std::string, std::string>::value_type(
      std::string(kMetricsLabelModelName), ModelName()));
  labels->insert(std::map<std::string, std::string>::value_type(
      std::string(kMetricsLabelModelVersion), std::to_string(ModelVersion())));
  for (const auto& tag : model_tags_) {
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
MetricModelReporter::GetCounterMetric(
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
MetricModelReporter::MetricInferenceSuccess(int gpu_device) const
{
  return GetCounterMetric(
      metric_inf_success_, Metrics::FamilyInferenceSuccess(), gpu_device);
}

prometheus::Counter&
MetricModelReporter::MetricInferenceFailure(int gpu_device) const
{
  return GetCounterMetric(
      metric_inf_failure_, Metrics::FamilyInferenceFailure(), gpu_device);
}

prometheus::Counter&
MetricModelReporter::MetricInferenceCount(int gpu_device) const
{
  return GetCounterMetric(
      metric_inf_count_, Metrics::FamilyInferenceCount(), gpu_device);
}

prometheus::Counter&
MetricModelReporter::MetricInferenceExecutionCount(int gpu_device) const
{
  return GetCounterMetric(
      metric_inf_exec_count_, Metrics::FamilyInferenceExecutionCount(),
      gpu_device);
}

prometheus::Counter&
MetricModelReporter::MetricInferenceRequestDuration(int gpu_device) const
{
  return GetCounterMetric(
      metric_inf_request_duration_us_,
      Metrics::FamilyInferenceRequestDuration(), gpu_device);
}

prometheus::Counter&
MetricModelReporter::MetricInferenceComputeDuration(int gpu_device) const
{
  return GetCounterMetric(
      metric_inf_compute_duration_us_,
      Metrics::FamilyInferenceComputeDuration(), gpu_device);
}

prometheus::Counter&
MetricModelReporter::MetricInferenceQueueDuration(int gpu_device) const
{
  return GetCounterMetric(
      metric_inf_queue_duration_us_, Metrics::FamilyInferenceQueueDuration(),
      gpu_device);
}

prometheus::Histogram&
MetricModelReporter::MetricInferenceLoadRatio(int gpu_device) const
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

#endif  // TRTIS_ENABLE_STATS
#endif  // TRTIS_ENABLE_METRICS

}}  // namespace nvidia::inferenceserver
