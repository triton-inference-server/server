// Copyright (c) 2018-2021, NVIDIA CORPORATION & AFFILIATES. All rights
// reserved.
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
#include "prometheus/detail/utils.h"
#include "src/core/constants.h"
#include "src/core/logging.h"

#ifdef TRITON_ENABLE_METRICS_GPU
#include <cuda_runtime_api.h>
#include <dcgm_agent.h>
#include <cstring>
#include <set>
#include <string>
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

size_t
Metrics::HashLabels(const std::map<std::string, std::string>& labels)
{
  return prometheus::detail::hash_labels(labels);
}

Metrics::~Metrics()
{
#ifdef TRITON_ENABLE_METRICS_GPU
  // Signal the DCGM thread to exit and then wait for it...
  if (dcgm_thread_ != nullptr) {
    dcgm_thread_exit_.store(true);
    dcgm_thread_->join();
    dcgmGroupDestroy(dcgm_handle_, groupId_);
    // Stop and shutdown DCGM
    dcgmReturn_t derr;
    if (standalone_) {
      derr = dcgmDisconnect(dcgm_handle_);
    } else {
      derr = dcgmStopEmbedded(dcgm_handle_);
    }
    if (derr != DCGM_ST_OK) {
      LOG_WARNING << "Unable to stop DCGM: " << errorString(derr);
    }

    derr = dcgmShutdown();
    if (derr != DCGM_ST_OK) {
      LOG_WARNING << "Unable to shutdown DCGM: " << errorString(derr);
    }
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
    LOG_WARNING << "error initializing DCGM, GPU metrics will not be "
                << "available: " << errorString(dcgmerr);
    return false;
  }

  if (standalone_) {
    char hostIpAddress[16] = {0};
    std::string ipAddress = "127.0.0.1";
    strncpy(hostIpAddress, ipAddress.c_str(), 15);
    dcgmerr = dcgmConnect(hostIpAddress, &dcgm_handle_);
  } else {
    dcgmerr = dcgmStartEmbedded(DCGM_OPERATION_MODE_MANUAL, &dcgm_handle_);
  }
  if (dcgmerr != DCGM_ST_OK) {
    LOG_WARNING << "DCGM unable to start: " << errorString(dcgmerr);
    return false;
  }

  if (standalone_) {
    dcgmerr = dcgmUpdateAllFields(dcgm_handle_, 1);
    if (dcgmerr != DCGM_ST_OK) {
      LOG_WARNING << "DCGM unable to update all fields, GPU metrics will "
                     "not be available: "
                  << errorString(dcgmerr);
      return false;
    }
  }

  unsigned int dcgm_gpu_ids[DCGM_MAX_NUM_DEVICES];
  int dcgm_gpu_count;
  dcgmerr = dcgmGetAllDevices(dcgm_handle_, dcgm_gpu_ids, &dcgm_gpu_count);
  if (dcgmerr != DCGM_ST_OK) {
    LOG_WARNING << "DCGM unable to get device info and count, GPU "
                   "metrics will not be available: "
                << errorString(dcgmerr);
    return false;
  }

  // Get PCI Bus ID to DCGM device Id map.
  // Some devices may have problems using DCGM API and
  // these devices needs to be ignored.
  std::map<std::string, size_t> pci_bus_id_to_dcgm_id;
  std::map<std::string, std::map<std::string, std::string> >
      pci_bus_id_to_gpu_labels;
  std::map<std::string, std::string> pci_bus_id_to_device_name;
  dcgmDeviceAttributes_t gpu_attributes[DCGM_MAX_NUM_DEVICES];
  for (int i = 0; i < dcgm_gpu_count; i++) {
    gpu_attributes[i].version = dcgmDeviceAttributes_version;
    dcgmerr = dcgmGetDeviceAttributes(
        dcgm_handle_, dcgm_gpu_ids[i], &gpu_attributes[i]);
    if (dcgmerr != DCGM_ST_OK) {
      LOG_WARNING << "DCGM unable to get device properties for DCGM device "
                  << dcgm_gpu_ids[i]
                  << ", GPU metrics will not be available for this device: "
                  << errorString(dcgmerr);
    } else {
      std::string pciBusId = gpu_attributes[i].identifiers.pciBusId;
      pci_bus_id_to_dcgm_id[pciBusId] = i;
      pci_bus_id_to_device_name[pciBusId] =
          std::string(gpu_attributes[i].identifiers.deviceName);
      std::map<std::string, std::string> gpu_labels;
      gpu_labels.insert(std::map<std::string, std::string>::value_type(
          kMetricsLabelGpuUuid,
          std::string(gpu_attributes[i].identifiers.uuid)));
      pci_bus_id_to_gpu_labels[pciBusId] = gpu_labels;
    }
  }


  // Get CUDA-visible PCI Bus Ids and get DCGM metrics for each CUDA-visible GPU
  std::map<uint32_t, uint32_t> cuda_ids_to_dcgm_ids;
  std::vector<uint32_t> available_cuda_gpu_ids;
  int cuda_gpu_count;
  cudaError_t cudaerr = cudaGetDeviceCount(&cuda_gpu_count);
  if (cudaerr != cudaSuccess) {
    LOG_WARNING
        << "Cannot get CUDA device count, GPU metrics will not be available";
    return false;
  }
  for (int i = 0; i < cuda_gpu_count; ++i) {
    std::string pci_bus_id = "0000";  // pad 0's for uniformity
    char pcibusid_str[64];
    cudaerr = cudaDeviceGetPCIBusId(pcibusid_str, sizeof(pcibusid_str) - 1, i);
    if (cudaerr == cudaSuccess) {
      pci_bus_id.append(pcibusid_str);
      if (pci_bus_id_to_dcgm_id.count(pci_bus_id) <= 0) {
        LOG_INFO << "Skipping GPU:" << i
                 << " since it's not CUDA enabled. This should never happen!";
        continue;
      }
      // Filter out CUDA visible GPUs from GPUs found by DCGM
      LOG_INFO << "Collecting metrics for GPU " << i << ": "
               << pci_bus_id_to_device_name[pci_bus_id];
      auto& gpu_labels = pci_bus_id_to_gpu_labels[pci_bus_id];
      gpu_utilization_.push_back(&gpu_utilization_family_.Add(gpu_labels));
      gpu_memory_total_.push_back(&gpu_memory_total_family_.Add(gpu_labels));
      gpu_memory_used_.push_back(&gpu_memory_used_family_.Add(gpu_labels));
      gpu_power_usage_.push_back(&gpu_power_usage_family_.Add(gpu_labels));
      gpu_power_limit_.push_back(&gpu_power_limit_family_.Add(gpu_labels));
      gpu_energy_consumption_.push_back(
          &gpu_energy_consumption_family_.Add(gpu_labels));
      uint32_t dcgm_id = pci_bus_id_to_dcgm_id[pci_bus_id];
      cuda_ids_to_dcgm_ids[i] = dcgm_id;
      available_cuda_gpu_ids.emplace_back(i);
    } else {
      LOG_WARNING << "GPU metrics will not be available for device:" << i;
    }
  }

  // create a gpu group
  char groupName[] = "dcgm_group";
  dcgmerr =
      dcgmGroupCreate(dcgm_handle_, DCGM_GROUP_DEFAULT, groupName, &groupId_);
  if (dcgmerr != DCGM_ST_OK) {
    LOG_WARNING << "Cannot make GPU group: " << errorString(dcgmerr);
  }

  // Periodically send the DCGM metrics...
  if (available_cuda_gpu_ids.size() > 0) {
    dcgmHandle_t handle = dcgm_handle_;
    dcgmGpuGrp_t groupId = groupId_;
    dcgm_thread_exit_.store(false);
    dcgm_thread_.reset(new std::thread([this, available_cuda_gpu_ids,
                                        cuda_ids_to_dcgm_ids, handle, groupId] {
      int available_cuda_gpu_count = available_cuda_gpu_ids.size();
      // Stop attempting metrics if they fail multiple consecutive
      // times for a device.
      constexpr int fail_threshold = 3;
      std::vector<int> power_limit_fail_cnt(available_cuda_gpu_count);
      std::vector<int> power_usage_fail_cnt(available_cuda_gpu_count);
      std::vector<int> energy_fail_cnt(available_cuda_gpu_count);
      std::vector<int> util_fail_cnt(available_cuda_gpu_count);
      std::vector<int> mem_fail_cnt(available_cuda_gpu_count);
      std::vector<int> cuda_available_cnt(available_cuda_gpu_count);

      unsigned long long last_energy[available_cuda_gpu_count];
      for (int didx = 0; didx < available_cuda_gpu_count; ++didx) {
        last_energy[didx] = 0;
      }
      size_t field_count = 6;
      unsigned short util_flag =
          standalone_ ? DCGM_FI_PROF_GR_ENGINE_ACTIVE : DCGM_FI_DEV_GPU_UTIL;
      unsigned short fields[field_count] = {
          DCGM_FI_DEV_POWER_MGMT_LIMIT,          // power limit, watts
          DCGM_FI_DEV_POWER_USAGE,               // power usage, watts
          DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION,  // Total energy consumption, mJ
          util_flag,                             // util ratio, 1 = 1%
          DCGM_FI_DEV_FB_USED,                   // Frame buffer used, MiB
          DCGM_FI_DEV_FB_TOTAL,                  // Frame buffer used, MiB
      };

      char fieldName[] = "field_group";
      dcgmFieldGrp_t fieldGroupId;
      dcgmReturn_t dcgmerr = dcgmFieldGroupCreate(
          handle, field_count, &fields[0], fieldName, &fieldGroupId);
      if (dcgmerr != DCGM_ST_OK) {
        LOG_WARNING << "Cannot make field group: " << errorString(dcgmerr);
      }
      dcgmerr = dcgmWatchFields(
          handle, groupId, fieldGroupId, 2000000 /*update period, usec*/,
          5.0 /*maxKeepAge, sec*/, 5 /*maxKeepSamples*/);
      if (dcgmerr != DCGM_ST_OK) {
        LOG_WARNING << "Cannot start watching fields: " << errorString(dcgmerr);
      } else {
        while (!dcgm_thread_exit_.load()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(2000));
          dcgmUpdateAllFields(handle, 1 /* wait for update*/);
          for (int didx = 0; didx < available_cuda_gpu_count; ++didx) {
            uint32_t cuda_id = available_cuda_gpu_ids[didx];
            if (cuda_ids_to_dcgm_ids.count(cuda_id) <= 0) {
              LOG_WARNING << "Cannot find DCGM id for CUDA id " << cuda_id;
              continue;
            }
            uint32_t dcgm_id = cuda_ids_to_dcgm_ids.at(cuda_id);
            dcgmFieldValue_v1 field_values[field_count];
            dcgmReturn_t dcgmerr = dcgmGetLatestValuesForFields(
                handle, dcgm_id, fields, field_count, field_values);

            if (dcgmerr != DCGM_ST_OK) {
              power_limit_fail_cnt[didx]++;
              power_usage_fail_cnt[didx]++;
              energy_fail_cnt[didx]++;
              util_fail_cnt[didx]++;
              mem_fail_cnt[didx]++;
              LOG_WARNING << "Unable to get field values for GPU ID " << cuda_id
                          << ": " << errorString(dcgmerr);
            } else {
              // Power limit
              if (power_limit_fail_cnt[didx] < fail_threshold) {
                double power_limit = field_values[0].value.dbl;
                if ((field_values[0].status == DCGM_ST_OK) &&
                    (!DCGM_FP64_IS_BLANK(power_limit))) {
                  power_limit_fail_cnt[didx] = 0;
                } else {
                  power_limit_fail_cnt[didx]++;
                  power_limit = 0;
                  LOG_WARNING << "Unable to get power limit for GPU " << cuda_id
                              << ": " << errorString(dcgmerr);
                }
                gpu_power_limit_[didx]->Set(power_limit);
              }

              // Power usage
              if (power_usage_fail_cnt[didx] < fail_threshold) {
                double power_usage = field_values[1].value.dbl;
                if ((field_values[1].status == DCGM_ST_OK) &&
                    (!DCGM_FP64_IS_BLANK(power_usage))) {
                  power_usage_fail_cnt[didx] = 0;
                } else {
                  power_usage_fail_cnt[didx]++;
                  power_usage = 0;
                  LOG_WARNING << "Unable to get power usage for GPU " << cuda_id
                              << ": " << errorString(dcgmerr);
                }
                gpu_power_usage_[didx]->Set(power_usage);
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
                  LOG_WARNING << "Unable to get energy consumption for "
                              << "GPU " << cuda_id << ": "
                              << errorString(dcgmerr);
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
                  LOG_WARNING << "Unable to get GPU utilization for GPU "
                              << cuda_id << ": " << errorString(dcgmerr);
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
                  LOG_WARNING << "Unable to get memory usage for GPU "
                              << cuda_id << ": " << errorString(dcgmerr);
                }
                gpu_memory_total_[didx]->Set(
                    memory_total * 1024 * 1024);  // bytes
                gpu_memory_used_[didx]->Set(
                    memory_used * 1024 * 1024);  // bytes
              }
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
  gpu_attributes.version = dcgmDeviceAttributes_version;
  dcgmReturn_t dcgmerr = dcgmGetDeviceAttributes(
      singleton->dcgm_handle_, cuda_device, &gpu_attributes);
  if (dcgmerr != DCGM_ST_OK) {
    LOG_ERROR << "Unable to get device UUID: " << errorString(dcgmerr);
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
