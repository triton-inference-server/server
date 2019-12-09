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
//
#pragma once

#ifdef TRTIS_ENABLE_METRICS

#include <atomic>
#include <thread>
#include "prometheus/registry.h"
#include "prometheus/serializer.h"
#include "prometheus/text_serializer.h"

namespace nvidia { namespace inferenceserver {

class Metrics {
 public:
  // Enable reporting of GPU metrics
  static void EnableGPUMetrics();

  // Get the prometheus registry
  static std::shared_ptr<prometheus::Registry> GetRegistry();

  // Get serialized metrics
  static const std::string SerializedMetrics();

  // Get the UUID for a CUDA device. Return true and initialize 'uuid'
  // if a UUID is found, return false if a UUID cannot be returned.
  static bool UUIDForCudaDevice(int cuda_device, std::string* uuid);

#ifdef TRTIS_ENABLE_STATS
  // Metric family counting successful inference requests
  static prometheus::Family<prometheus::Counter>& FamilyInferenceSuccess()
  {
    return GetSingleton()->inf_success_family_;
  }

  // Metric family counting failed inference requests
  static prometheus::Family<prometheus::Counter>& FamilyInferenceFailure()
  {
    return GetSingleton()->inf_failure_family_;
  }

  // Metric family counting inferences performed, where a batch-size
  // 'n' inference request is counted as 'n' inferences
  static prometheus::Family<prometheus::Counter>& FamilyInferenceCount()
  {
    return GetSingleton()->inf_count_family_;
  }

  // Metric family counting inferences performed, where a batch-size
  // 'n' inference request is counted as 'n' inferences
  static prometheus::Family<prometheus::Counter>&
  FamilyInferenceExecutionCount()
  {
    return GetSingleton()->inf_count_exec_family_;
  }

  // Metric family of cumulative inference request duration, in
  // microseconds
  static prometheus::Family<prometheus::Counter>&
  FamilyInferenceRequestDuration()
  {
    return GetSingleton()->inf_request_duration_us_family_;
  }

  // Metric family of cumulative inference compute duration, in
  // microseconds
  static prometheus::Family<prometheus::Counter>&
  FamilyInferenceComputeDuration()
  {
    return GetSingleton()->inf_compute_duration_us_family_;
  }

  // Metric family of cumulative inference queuing duration, in
  // microseconds
  static prometheus::Family<prometheus::Counter>& FamilyInferenceQueueDuration()
  {
    return GetSingleton()->inf_queue_duration_us_family_;
  }

  // Metric family of load-ratio histogram
  static prometheus::Family<prometheus::Histogram>& FamilyInferenceLoadRatio()
  {
    return GetSingleton()->inf_load_ratio_family_;
  }
#endif  // TRTIS_ENABLE_STATS

 private:
  Metrics();
  virtual ~Metrics();
  static Metrics* GetSingleton();
  bool InitializeNvmlMetrics();

  std::shared_ptr<prometheus::Registry> registry_;
  std::unique_ptr<prometheus::Serializer> serializer_;

#ifdef TRTIS_ENABLE_STATS
  prometheus::Family<prometheus::Counter>& inf_success_family_;
  prometheus::Family<prometheus::Counter>& inf_failure_family_;
  prometheus::Family<prometheus::Counter>& inf_count_family_;
  prometheus::Family<prometheus::Counter>& inf_count_exec_family_;
  prometheus::Family<prometheus::Counter>& inf_request_duration_us_family_;
  prometheus::Family<prometheus::Counter>& inf_compute_duration_us_family_;
  prometheus::Family<prometheus::Counter>& inf_queue_duration_us_family_;
  prometheus::Family<prometheus::Histogram>& inf_load_ratio_family_;
#endif  // TRTIS_ENABLE_STATS
  prometheus::Family<prometheus::Gauge>& gpu_utilization_family_;
  prometheus::Family<prometheus::Gauge>& gpu_memory_total_family_;
  prometheus::Family<prometheus::Gauge>& gpu_memory_used_family_;
  prometheus::Family<prometheus::Gauge>& gpu_power_usage_family_;
  prometheus::Family<prometheus::Gauge>& gpu_power_limit_family_;
  prometheus::Family<prometheus::Counter>& gpu_energy_consumption_family_;

  std::vector<prometheus::Gauge*> gpu_utilization_;
  std::vector<prometheus::Gauge*> gpu_memory_total_;
  std::vector<prometheus::Gauge*> gpu_memory_used_;
  std::vector<prometheus::Gauge*> gpu_power_usage_;
  std::vector<prometheus::Gauge*> gpu_power_limit_;
  std::vector<prometheus::Counter*> gpu_energy_consumption_;

  bool gpu_metrics_enabled_;
  std::unique_ptr<std::thread> nvml_thread_;
  std::atomic<bool> nvml_thread_exit_;
};

}}  // namespace nvidia::inferenceserver

#endif  // TRTIS_ENABLE_METRICS
