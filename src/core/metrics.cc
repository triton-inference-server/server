// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifdef TRITON_ENABLE_METRICS

#include "src/core/metrics.h"

#include <thread>
#include "src/core/constants.h"
#include "src/core/logging.h"

#ifdef TRITON_ENABLE_METRICS_GPU
#include <cuda_runtime_api.h>
#include <dcgm_agent.h>
#endif  // TRITON_ENABLE_METRICS_GPU

namespace nvidia { namespace inferenceserver {

Metrics::Metrics()
    : registry_(std::make_shared<prometheus::Registry>()),
      serializer_(new prometheus::TextSerializer()),
      inf_success_family_(
          prometheus::BuildCounter()
              .Name("nv_inference_request_success")
              .Help("Number of successful inference requests, all batch sizes")
              .Register(*registry_)),
      inf_failure_family_(
          prometheus::BuildCounter()
              .Name("nv_inference_request_failure")
              .Help("Number of failed inference requests, all batch sizes")
              .Register(*registry_)),
      inf_count_family_(prometheus::BuildCounter()
                            .Name("nv_inference_count")
                            .Help("Number of inferences performed")
                            .Register(*registry_)),
      inf_count_exec_family_(prometheus::BuildCounter()
                                 .Name("nv_inference_exec_count")
                                 .Help("Number of model executions performed")
                                 .Register(*registry_)),
      inf_request_duration_us_family_(
          prometheus::BuildCounter()
              .Name("nv_inference_request_duration_us")
              .Help("Cummulative inference request duration in microseconds")
              .Register(*registry_)),
      inf_queue_duration_us_family_(
          prometheus::BuildCounter()
              .Name("nv_inference_queue_duration_us")
              .Help("Cummulative inference queuing duration in microseconds")
              .Register(*registry_)),
      inf_compute_input_duration_us_family_(
          prometheus::BuildCounter()
              .Name("nv_inference_compute_input_duration_us")
              .Help("Cummulative compute input duration in microseconds")
              .Register(*registry_)),
      inf_compute_infer_duration_us_family_(
          prometheus::BuildCounter()
              .Name("nv_inference_compute_infer_duration_us")
              .Help("Cummulative compute inference duration in microseconds")
              .Register(*registry_)),
      inf_compute_output_duration_us_family_(
          prometheus::BuildCounter()
              .Name("nv_inference_compute_output_duration_us")
              .Help("Cummulative inference compute output duration in "
                    "microseconds")
              .Register(*registry_)),
#ifdef TRITON_ENABLE_METRICS_GPU
      gpu_utilization_family_(prometheus::BuildGauge()
                                  .Name("nv_gpu_utilization")
                                  .Help("GPU utilization rate [0.0 - 1.0)")
                                  .Register(*registry_)),
      gpu_memory_total_family_(prometheus::BuildGauge()
                                   .Name("nv_gpu_memory_total_bytes")
                                   .Help("GPU total memory, in bytes")
                                   .Register(*registry_)),
      gpu_memory_used_family_(prometheus::BuildGauge()
                                  .Name("nv_gpu_memory_used_bytes")
                                  .Help("GPU used memory, in bytes")
                                  .Register(*registry_)),
      gpu_power_usage_family_(prometheus::BuildGauge()
                                  .Name("nv_gpu_power_usage")
                                  .Help("GPU power usage in watts")
                                  .Register(*registry_)),
      gpu_power_limit_family_(prometheus::BuildGauge()
                                  .Name("nv_gpu_power_limit")
                                  .Help("GPU power management limit in watts")
                                  .Register(*registry_)),
      gpu_energy_consumption_family_(
          prometheus::BuildCounter()
              .Name("nv_energy_consumption")
              .Help("GPU energy consumption in joules since the Triton Server "
                    "started")
              .Register(*registry_)),
#endif  // TRITON_ENABLE_METRICS_GPU
      metrics_enabled_(false), gpu_metrics_enabled_(false)
{
}

Metrics::~Metrics()
{
#ifdef TRITON_ENABLE_METRICS_GPU
  // Signal the dcgm thread to exit and then wait for it...
  if (dcgm_thread_ != nullptr) {
    dcgm_thread_exit_.store(true);
    dcgm_thread_->join();
  }
#endif  // TRITON_ENABLE_METRICS_GPU
}

bool
Metrics::Enabled()
{
  auto singleton = GetSingleton();
  return singleton->metrics_enabled_;
}

void
Metrics::EnableMetrics()
{
  auto singleton = GetSingleton();
  singleton->metrics_enabled_ = true;
}

void
Metrics::EnableGPUMetrics()
{
  auto singleton = GetSingleton();

  // Ensure thread-safe enabling of GPU Metrics
  std::lock_guard<std::mutex> lock(singleton->gpu_metrics_enabling_);
  if (singleton->gpu_metrics_enabled_) {
    return;
  }

  if (std::getenv("TRITON_SERVER_CPU_ONLY") == nullptr) {
    singleton->InitializeDcgmMetrics();
  }

  singleton->gpu_metrics_enabled_ = true;
}

bool
Metrics::InitializeDcgmMetrics()
{
#ifndef TRITON_ENABLE_METRICS_GPU
  return false;
#else
  dcgmReturn_t dcgmerr = dcgmInit();
  if (dcgmerr != DCGM_ST_OK) {
    LOG_WARNING
        << "failed to initialize DCGM, GPU metrics will not be available: "
        << errorString(dcgmerr);
    return false;
  }

  dcgmerr = dcgmStartEmbedded(DCGM_OPERATION_MODE_MANUAL, &dcgm_handle_);
  if (dcgmerr != DCGM_ST_OK) {
    LOG_WARNING << "Error: dcgmStartEmbedded returned \""
                << errorString(dcgmerr) << "(" << dcgmerr << ")\"";
    return false;
  }

  dcgmerr = dcgmUpdateAllFields(dcgm_handle_, 1);
  if (dcgmerr != DCGM_ST_OK) {
    LOG_WARNING << "DCGM failed to update all fields, GPU metrics will not "
                   "be available: "
                << errorString(dcgmerr);
    return false;
  }

  unsigned int all_gpu_ids[DCGM_MAX_NUM_DEVICES];
  int gpu_count;
  dcgmerr = dcgmGetAllDevices(dcgm_handle_, all_gpu_ids, &gpu_count);
  if (dcgmerr != DCGM_ST_OK) {
    LOG_WARNING << "failed to get device info and count, GPU metrics will not "
                   "be available: "
                << errorString(dcgmerr);
    return false;
  }

  // Get DCGM metrics for each GPU. Some devices may have problems using DCGM
  // API and thus device count/ids needs to be updated.
  std::vector<uint32_t> available_gpu_ids;
  dcgmDeviceAttributes_t gpu_attributes[DCGM_MAX_NUM_DEVICES];
  for (int i = 0; i < gpu_count; i++) {
    gpu_attributes[i].version = dcgmDeviceAttributes_version;
    dcgmerr = dcgmGetDeviceAttributes(
        dcgm_handle_, all_gpu_ids[i], &gpu_attributes[i]);
    if (dcgmerr != DCGM_ST_OK) {
      LOG_WARNING << "failed to get device properties for device "
                  << all_gpu_ids[i]
                  << ", GPU metrics will not be available for this device: "
                  << errorString(dcgmerr);
    } else {
      LOG_INFO << "Collecting metrics for GPU " << all_gpu_ids[i] << ": "
               << std::string(gpu_attributes[i].identifiers.deviceName);

      std::map<std::string, std::string> gpu_labels;
      gpu_labels.insert(std::map<std::string, std::string>::value_type(
          kMetricsLabelGpuUuid,
          std::string(gpu_attributes[i].identifiers.uuid)));

      gpu_utilization_.push_back(&gpu_utilization_family_.Add(gpu_labels));
      gpu_memory_total_.push_back(&gpu_memory_total_family_.Add(gpu_labels));
      gpu_memory_used_.push_back(&gpu_memory_used_family_.Add(gpu_labels));
      gpu_power_usage_.push_back(&gpu_power_usage_family_.Add(gpu_labels));
      gpu_power_limit_.push_back(&gpu_power_limit_family_.Add(gpu_labels));
      gpu_energy_consumption_.push_back(
          &gpu_energy_consumption_family_.Add(gpu_labels));
      available_gpu_ids.emplace_back(i);
    }
  }

  // Periodically send the DCGM metrics...
  if (available_gpu_ids.size() > 0) {
    dcgmHandle_t handle = dcgm_handle_;
    dcgm_thread_exit_.store(false);
    dcgm_thread_.reset(new std::thread([this, available_gpu_ids, handle] {
      int available_gpu_count = available_gpu_ids.size();
      // Stop attempting metrics if they fail multiple consecutive
      // times for a device.
      constexpr int fail_threshold = 3;
      std::vector<int> power_limit_fail_cnt(available_gpu_count);
      std::vector<int> power_usage_fail_cnt(available_gpu_count);
      std::vector<int> energy_fail_cnt(available_gpu_count);
      std::vector<int> util_fail_cnt(available_gpu_count);
      std::vector<int> mem_fail_cnt(available_gpu_count);

      unsigned long long last_energy[available_gpu_count];
      for (int didx = 0; didx < available_gpu_count; ++didx) {
        last_energy[didx] = 0;
      }

      while (!dcgm_thread_exit_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        size_t field_count = 6;
        unsigned short fields[field_count] = {
            DCGM_FI_DEV_POWER_MGMT_LIMIT,
            DCGM_FI_DEV_POWER_USAGE,
            DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION,
            DCGM_FI_DEV_GPU_UTIL,
            DCGM_FI_DEV_FB_USED,
            DCGM_FI_DEV_FB_TOTAL,
        };

        for (int didx = 0; didx < available_gpu_count; ++didx) {
          dcgmFieldValue_v1 field_values[field_count];
          dcgmReturn_t dcgmerr = dcgmGetLatestValuesForFields(
              handle, available_gpu_ids[didx], fields, field_count,
              field_values);
          if (dcgmerr != DCGM_ST_OK) {
            power_limit_fail_cnt[didx]++;
            power_usage_fail_cnt[didx]++;
            energy_fail_cnt[didx]++;
            util_fail_cnt[didx]++;
            mem_fail_cnt[didx]++;
            LOG_WARNING << "failed to get field values for device "
                        << available_gpu_ids[didx] << ": "
                        << errorString(dcgmerr);
          } else {
            // Power limit
            if (power_limit_fail_cnt[didx] < fail_threshold) {
              unsigned int power_limit = field_values[0].value.i64;
              if ((field_values[0].status == DCGM_ST_OK) &&
                  (!DCGM_INT64_IS_BLANK(power_limit))) {
                power_limit_fail_cnt[didx] = 0;
              } else {
                power_limit_fail_cnt[didx]++;
                power_limit = 0;
                LOG_WARNING << "failed to get power limit for GPU " << didx
                            << ": " << errorString(dcgmerr);
              }
              gpu_power_limit_[didx]->Set((double)power_limit * 0.001);
            }

            // Power usage
            if (power_usage_fail_cnt[didx] < fail_threshold) {
              unsigned int power_usage = field_values[1].value.i64;
              if ((field_values[1].status == DCGM_ST_OK) &&
                  (!DCGM_INT64_IS_BLANK(power_usage))) {
                power_usage_fail_cnt[didx] = 0;
              } else {
                power_usage_fail_cnt[didx]++;
                power_usage = 0;
                LOG_WARNING << "failed to get power usage for GPU " << didx
                            << ": " << errorString(dcgmerr);
              }
              gpu_power_usage_[didx]->Set((double)power_usage * 0.001);
            }

            // Energy Consumption
            if (energy_fail_cnt[didx] < fail_threshold) {
              unsigned int energy = field_values[2].value.i64;
              if ((field_values[2].status == DCGM_ST_OK) &&
                  (!DCGM_INT64_IS_BLANK(energy))) {
                energy_fail_cnt[didx] = 0;
                if (last_energy[didx] == 0) {
                  last_energy[didx] = energy;
                }
                gpu_energy_consumption_[didx]->Increment(
                    (double)(energy - last_energy[didx]) * 0.001);
                last_energy[didx] = energy;
              } else {
                energy_fail_cnt[didx]++;
                energy = 0;
                LOG_WARNING << "failed to get energy consumption for GPU "
                            << didx << ": " << errorString(dcgmerr);
              }
            }

            // Utilization
            if (util_fail_cnt[didx] < fail_threshold) {
              unsigned int util = field_values[3].value.i64;
              if ((field_values[3].status == DCGM_ST_OK) &&
                  (!DCGM_INT64_IS_BLANK(util))) {
                util_fail_cnt[didx] = 0;
              } else {
                util_fail_cnt[didx]++;
                util = 0;
                LOG_WARNING << "failed to get GPU utilization for GPU " << didx
                            << ": " << errorString(dcgmerr);
              }
              gpu_utilization_[didx]->Set((double)util * 0.01);
            }

            // Memory Usage
            if (mem_fail_cnt[didx] < fail_threshold) {
              unsigned int memory_used = field_values[4].value.i64;
              unsigned int memory_total = field_values[5].value.i64;
              if ((field_values[4].status == DCGM_ST_OK) &&
                  (!DCGM_INT64_IS_BLANK(memory_used)) &&
                  (field_values[5].status == DCGM_ST_OK) &&
                  (!DCGM_INT64_IS_BLANK(memory_total))) {
                mem_fail_cnt[didx] = 0;
              } else {
                memory_total = 0;
                memory_used = 0;
                mem_fail_cnt[didx]++;
                LOG_WARNING << "failed to get memory usage for GPU " << didx
                            << ": " << errorString(dcgmerr);
              }
              gpu_memory_total_[didx]->Set(memory_total);
              gpu_memory_used_[didx]->Set(memory_used);
            }
          }
        }
      }
    }));
  }

  return true;
#endif  // TRITON_ENABLE_METRICS_GPU
}

bool
Metrics::UUIDForCudaDevice(int cuda_device, std::string* uuid)
{
  // If metrics were not initialized then just silently fail since
  // with DCGM we can't get the CUDA device (and not worth doing
  // anyway since metrics aren't being reported).
  auto singleton = GetSingleton();
  if (!singleton->gpu_metrics_enabled_) {
    return false;
  }

  // If GPU metrics is not enabled just silently fail.
#ifndef TRITON_ENABLE_METRICS_GPU
  return false;
#else

  dcgmDeviceAttributes_t gpu_attributes;
  dcgmReturn_t dcgmerr = dcgmGetDeviceAttributes(
      singleton->dcgm_handle_, cuda_device, &gpu_attributes);
  if (dcgmerr != DCGM_ST_OK) {
    LOG_ERROR << "failed to get device UUID: DCGM_ERROR "
              << errorString(dcgmerr);
    return false;
  }

  *uuid = gpu_attributes.identifiers.uuid;
  return true;
#endif  // TRITON_ENABLE_METRICS_GPU
}

std::shared_ptr<prometheus::Registry>
Metrics::GetRegistry()
{
  auto singleton = Metrics::GetSingleton();
  return singleton->registry_;
}

const std::string
Metrics::SerializedMetrics()
{
  auto singleton = Metrics::GetSingleton();
  return singleton->serializer_->Serialize(
      singleton->registry_.get()->Collect());
}

Metrics*
Metrics::GetSingleton()
{
  static Metrics singleton;
  return &singleton;
}

}}  // namespace nvidia::inferenceserver

#endif  // TRITON_ENABLE_METRICS
