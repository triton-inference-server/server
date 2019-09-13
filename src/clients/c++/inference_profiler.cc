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

#include "src/clients/c++/inference_profiler.h"

#include <math.h>

namespace perfclient {

nic::Error
InferenceProfiler::Create(
    const bool verbose, const double stable_offset,
    const uint64_t measurement_window_ms, const size_t max_measurement_count,
    const int64_t percentile, std::shared_ptr<ContextFactory>& factory,
    std::unique_ptr<LoadManager> manager,
    std::unique_ptr<InferenceProfiler>* profiler)
{
  std::unique_ptr<nic::ServerStatusContext> status_ctx;
  RETURN_IF_ERROR(factory->CreateServerStatusContext(&status_ctx));

  std::unique_ptr<InferenceProfiler> local_profiler(new InferenceProfiler(
      verbose, stable_offset, measurement_window_ms, max_measurement_count,
      (percentile != -1), percentile, factory->SchedulerType(),
      factory->ModelName(), factory->ModelVersion(), std::move(status_ctx),
      std::move(manager)));

  if (local_profiler->scheduler_type_ == ContextFactory::ENSEMBLE) {
    ni::ServerStatus server_status;
    RETURN_IF_ERROR(
        local_profiler->status_ctx_->GetServerStatus(&server_status));
    RETURN_IF_ERROR(local_profiler->BuildComposingModelMap(server_status));
  }

  *profiler = std::move(local_profiler);
  return nic::Error::Success;
}

nic::Error
InferenceProfiler::BuildComposingModelMap(const ni::ServerStatus& server_status)
{
  RETURN_IF_ERROR(
      BuildComposingModelMap(model_name_, model_version_, server_status));
  return nic::Error::Success;
}

nic::Error
InferenceProfiler::BuildComposingModelMap(
    const std::string& model_name, const int64_t& model_version,
    const ni::ServerStatus& server_status)
{
  const auto& itr = server_status.model_status().find(model_name);
  if (itr == server_status.model_status().end()) {
    return nic::Error(
        ni::RequestStatusCode::INTERNAL,
        "unable to find status for model" + model_name);
  } else {
    if (itr->second.config().platform() == "ensemble") {
      for (const auto& step :
           itr->second.config().ensemble_scheduling().step()) {
        this->composing_models_map_[std::make_pair(model_name, model_version)]
            .emplace(step.model_name(), step.model_version());
        this->BuildComposingModelMap(
            step.model_name(), step.model_version(), server_status);
      }
    }
  }
  return nic::Error::Success;
}

InferenceProfiler::InferenceProfiler(
    const bool verbose, const double stable_offset,
    const int32_t measurement_window_ms, const size_t max_measurement_count,
    const bool extra_percentile, const size_t percentile,
    const ContextFactory::ModelSchedulerType scheduler_type,
    const std::string& model_name, const int64_t model_version,
    std::unique_ptr<nic::ServerStatusContext> status_ctx,
    std::unique_ptr<LoadManager> manager)
    : verbose_(verbose), measurement_window_ms_(measurement_window_ms),
      max_measurement_count_(max_measurement_count),
      extra_percentile_(extra_percentile), percentile_(percentile),
      scheduler_type_(scheduler_type), model_name_(model_name),
      model_version_(model_version), status_ctx_(std::move(status_ctx)),
      manager_(std::move(manager))
{
  load_parameters_.stable_offset = stable_offset;
  load_parameters_.stability_window = 3;
}

nic::Error
InferenceProfiler::Profile(
    const size_t concurrent_request_count, PerfStatus& status_summary)
{
  status_summary.concurrency = concurrent_request_count;

  RETURN_IF_ERROR(manager_->ChangeConcurrencyLevel(concurrent_request_count));

  // Start measurement
  bool is_stable = true;
  LoadStatus load_status;

  do {
    RETURN_IF_ERROR(manager_->CheckHealth());

    RETURN_IF_ERROR(Measure(status_summary));

    load_status.infer_per_sec.push_back(status_summary.client_infer_per_sec);
    load_status.latencies.push_back(status_summary.stabilizing_latency_ns);
    load_status.avg_ips += (double)load_status.infer_per_sec.back() /
                           load_parameters_.stability_window;
    load_status.avg_latency +=
        load_status.latencies.back() / load_parameters_.stability_window;

    if (verbose_) {
      std::cout << "  Pass [" << load_status.infer_per_sec.size()
                << "] throughput: " << load_status.infer_per_sec.back()
                << " infer/sec. ";
      if (extra_percentile_) {
        std::cout << "p" << percentile_ << " latency: "
                  << (status_summary.client_percentile_latency_ns
                          .find(percentile_)
                          ->second /
                      1000)
                  << " usec" << std::endl;
      } else {
        std::cout << "Avg latency: "
                  << (status_summary.client_avg_latency_ns / 1000)
                  << " usec (std " << status_summary.std_us << " usec)"
                  << std::endl;
      }
    }

    if (load_status.infer_per_sec.size() >= load_parameters_.stability_window) {
      size_t idx =
          load_status.infer_per_sec.size() - load_parameters_.stability_window;
      if (load_status.infer_per_sec.size() >
          load_parameters_.stability_window) {
        load_status.avg_ips -= (double)load_status.infer_per_sec[idx - 1] /
                               load_parameters_.stability_window;
        load_status.avg_latency -=
            load_status.latencies[idx - 1] / load_parameters_.stability_window;
      }
      is_stable = true;
      for (; idx < load_status.infer_per_sec.size(); idx++) {
        // We call it complete only if stability_window measurements are within
        // +/-(stable_offset)% of the average infer per second and latency
        if ((load_status.infer_per_sec[idx] <
             load_status.avg_ips * (1 - load_parameters_.stable_offset)) ||
            (load_status.infer_per_sec[idx] >
             load_status.avg_ips * (1 + load_parameters_.stable_offset))) {
          is_stable = false;
        }
        if ((load_status.latencies[idx] <
             load_status.avg_latency * (1 - load_parameters_.stable_offset)) ||
            (load_status.latencies[idx] >
             load_status.avg_latency * (1 + load_parameters_.stable_offset))) {
          is_stable = false;
        }
      }
      if (is_stable) {
        break;
      }
    }
  } while ((!early_exit) &&
           (load_status.infer_per_sec.size() < max_measurement_count_));
  if (early_exit) {
    return nic::Error(ni::RequestStatusCode::INTERNAL, "Received exit signal.");
  } else if (!is_stable) {
    std::cerr << "Failed to obtain stable measurement within "
              << max_measurement_count_
              << " measurement windows for concurrency "
              << concurrent_request_count << ". Please try to "
              << "increase the time window." << std::endl;
  }

  return nic::Error::Success;
}

nic::Error
InferenceProfiler::GetServerSideStatus(
    std::map<std::string, ni::ModelStatus>* model_status)
{
  model_status->clear();

  ni::ServerStatus server_status;
  RETURN_IF_ERROR(status_ctx_->GetServerStatus(&server_status));
  RETURN_IF_ERROR(GetServerSideStatus(
      server_status, std::make_pair(model_name_, model_version_),
      model_status));
  return nic::Error::Success;
}

nic::Error
InferenceProfiler::GetServerSideStatus(
    ni::ServerStatus& server_status, const ModelInfo model_info,
    std::map<std::string, ni::ModelStatus>* model_status)
{
  const auto& itr = server_status.model_status().find(model_info.first);
  if (itr == server_status.model_status().end()) {
    return nic::Error(
        ni::RequestStatusCode::INTERNAL,
        "unable to find status for model" + model_info.first);
  } else {
    model_status->emplace(model_info.first, itr->second);
  }

  // Also get status for composing models if any
  for (const auto& composing_model_info : composing_models_map_[model_info]) {
    if (composing_models_map_.find(composing_model_info) !=
        composing_models_map_.end()) {
      RETURN_IF_ERROR(GetServerSideStatus(
          server_status, composing_model_info, model_status));
    } else {
      const auto& itr =
          server_status.model_status().find(composing_model_info.first);
      if (itr == server_status.model_status().end()) {
        return nic::Error(
            ni::RequestStatusCode::INTERNAL,
            "unable to find status for composing model" +
                composing_model_info.first);
      } else {
        model_status->emplace(composing_model_info.first, itr->second);
      }
    }
  }
  return nic::Error::Success;
}

// Used for measurement
nic::Error
InferenceProfiler::Measure(PerfStatus& status_summary)
{
  std::map<std::string, ni::ModelStatus> start_status;
  std::map<std::string, ni::ModelStatus> end_status;
  nic::InferContext::Stat start_stat;
  nic::InferContext::Stat end_stat;

  RETURN_IF_ERROR(GetServerSideStatus(&start_status));
  RETURN_IF_ERROR(manager_->GetAccumulatedContextStat(&start_stat));

  // Wait for specified time interval in msec
  std::this_thread::sleep_for(
      std::chrono::milliseconds((uint64_t)(measurement_window_ms_ * 1.2)));

  RETURN_IF_ERROR(manager_->GetAccumulatedContextStat(&end_stat));

  // Get server status and then print report on difference between
  // before and after status.
  RETURN_IF_ERROR(GetServerSideStatus(&end_status));

  TimestampVector current_timestamps;
  RETURN_IF_ERROR(manager_->SwapTimestamps(current_timestamps));

  RETURN_IF_ERROR(Summarize(
      current_timestamps, start_status, end_status, start_stat, end_stat,
      status_summary));

  return nic::Error::Success;
}

nic::Error
InferenceProfiler::Summarize(
    const TimestampVector& timestamps,
    const std::map<std::string, ni::ModelStatus>& start_status,
    const std::map<std::string, ni::ModelStatus>& end_status,
    const nic::InferContext::Stat& start_stat,
    const nic::InferContext::Stat& end_stat, PerfStatus& summary)
{
  size_t valid_sequence_count = 0;

  // Get measurement from requests that fall within the time interval
  std::pair<uint64_t, uint64_t> valid_range = MeasurementTimestamp(timestamps);
  std::vector<uint64_t> latencies =
      ValidLatencyMeasurement(timestamps, valid_range, valid_sequence_count);

  RETURN_IF_ERROR(SummarizeLatency(latencies, summary));
  RETURN_IF_ERROR(SummarizeClientStat(
      start_stat, end_stat, valid_range.second - valid_range.first,
      latencies.size(), valid_sequence_count, summary));

  RETURN_IF_ERROR(
      SummarizeServerStats(start_status, end_status, &(summary.server_stats)));

  return nic::Error::Success;
}

std::pair<uint64_t, uint64_t>
InferenceProfiler::MeasurementTimestamp(const TimestampVector& timestamps)
{
  // finding the start time of the first request
  // and the end time of the last request in the timestamp queue
  uint64_t first_request_start_ns = 0;
  uint64_t last_request_end_ns = 0;
  for (auto& timestamp : timestamps) {
    uint64_t request_start_time = TIMESPEC_TO_NANOS(std::get<0>(timestamp));
    uint64_t request_end_time = TIMESPEC_TO_NANOS(std::get<1>(timestamp));
    if ((first_request_start_ns > request_start_time) ||
        (first_request_start_ns == 0)) {
      first_request_start_ns = request_start_time;
    }
    if ((last_request_end_ns < request_end_time) ||
        (last_request_end_ns == 0)) {
      last_request_end_ns = request_end_time;
    }
  }

  // Define the measurement window [client_start_ns, client_end_ns) to be
  // in the middle of the queue
  uint64_t measurement_window_ns = measurement_window_ms_ * 1000 * 1000;
  uint64_t offset = first_request_start_ns + measurement_window_ns;
  offset =
      (offset > last_request_end_ns) ? 0 : (last_request_end_ns - offset) / 2;

  uint64_t start_ns = first_request_start_ns + offset;
  uint64_t end_ns = start_ns + measurement_window_ns;

  return std::make_pair(start_ns, end_ns);
}

std::vector<uint64_t>
InferenceProfiler::ValidLatencyMeasurement(
    const TimestampVector& timestamps,
    const std::pair<uint64_t, uint64_t>& valid_range,
    size_t& valid_sequence_count)
{
  std::vector<uint64_t> valid_latencies;
  valid_sequence_count = 0;
  for (auto& timestamp : timestamps) {
    uint64_t request_start_ns = TIMESPEC_TO_NANOS(std::get<0>(timestamp));
    uint64_t request_end_ns = TIMESPEC_TO_NANOS(std::get<1>(timestamp));

    if (request_start_ns <= request_end_ns) {
      // Only counting requests that end within the time interval
      if ((request_end_ns >= valid_range.first) &&
          (request_end_ns <= valid_range.second)) {
        valid_latencies.push_back(request_end_ns - request_start_ns);
        if (std::get<2>(timestamp) & ni::InferRequestHeader::FLAG_SEQUENCE_END)
          valid_sequence_count++;
      }
    }
  }

  // Always sort measured latencies as percentile will be reported as default
  std::sort(valid_latencies.begin(), valid_latencies.end());

  return valid_latencies;
}

nic::Error
InferenceProfiler::SummarizeLatency(
    const std::vector<uint64_t>& latencies, PerfStatus& summary)
{
  if (latencies.size() == 0) {
    return nic::Error(
        ni::RequestStatusCode::INTERNAL,
        "No valid requests recorded within time interval."
        " Please use a larger time window.");
  }

  uint64_t tol_latency_ns = 0;
  uint64_t tol_square_latency_us = 0;

  for (const auto& latency : latencies) {
    tol_latency_ns += latency;
    tol_square_latency_us += (latency * latency) / (1000 * 1000);
  }

  summary.client_avg_latency_ns = tol_latency_ns / latencies.size();

  // retrieve other interesting percentile
  summary.client_percentile_latency_ns.clear();
  std::set<size_t> percentiles{50, 90, 95, 99};
  if (extra_percentile_) {
    percentiles.emplace(percentile_);
  }

  for (const auto percentile : percentiles) {
    size_t index = (percentile / 100.0) * (latencies.size() - 1) + 0.5;
    summary.client_percentile_latency_ns.emplace(percentile, latencies[index]);
  }

  if (extra_percentile_) {
    summary.stabilizing_latency_ns =
        summary.client_percentile_latency_ns.find(percentile_)->second;
  } else {
    summary.stabilizing_latency_ns = summary.client_avg_latency_ns;
  }

  // calculate standard deviation
  uint64_t expected_square_latency_us =
      tol_square_latency_us / latencies.size();
  uint64_t square_avg_latency_us =
      (summary.client_avg_latency_ns * summary.client_avg_latency_ns) /
      (1000 * 1000);
  uint64_t var_us = (expected_square_latency_us > square_avg_latency_us)
                        ? (expected_square_latency_us - square_avg_latency_us)
                        : 0;
  summary.std_us = (uint64_t)(sqrt(var_us));

  return nic::Error::Success;
}

nic::Error
InferenceProfiler::SummarizeClientStat(
    const nic::InferContext::Stat& start_stat,
    const nic::InferContext::Stat& end_stat, const uint64_t duration_ns,
    const size_t valid_request_count, const size_t valid_sequence_count,
    PerfStatus& summary)
{
  summary.on_sequence_model = (scheduler_type_ == ContextFactory::SEQUENCE);
  summary.batch_size = manager_->BatchSize();
  summary.client_request_count = valid_request_count;
  summary.client_sequence_count = valid_sequence_count;
  summary.client_duration_ns = duration_ns;
  float client_duration_sec =
      (float)summary.client_duration_ns / ni::NANOS_PER_SECOND;
  summary.client_sequence_per_sec =
      (int)(valid_sequence_count / client_duration_sec);
  summary.client_infer_per_sec =
      (int)(valid_request_count * summary.batch_size / client_duration_sec);

  size_t completed_count =
      end_stat.completed_request_count - start_stat.completed_request_count;
  uint64_t request_time_ns = end_stat.cumulative_total_request_time_ns -
                             start_stat.cumulative_total_request_time_ns;
  uint64_t send_time_ns =
      end_stat.cumulative_send_time_ns - start_stat.cumulative_send_time_ns;
  uint64_t receive_time_ns = end_stat.cumulative_receive_time_ns -
                             start_stat.cumulative_receive_time_ns;
  if (completed_count != 0) {
    summary.client_avg_request_time_ns = request_time_ns / completed_count;
    summary.client_avg_send_time_ns = send_time_ns / completed_count;
    summary.client_avg_receive_time_ns = receive_time_ns / completed_count;
  }

  return nic::Error::Success;
}

nic::Error
InferenceProfiler::SummarizeServerModelStats(
    const std::string& model_name, const int64_t model_version,
    const ni::ModelStatus& start_status, const ni::ModelStatus& end_status,
    ServerSideStats* server_stats)
{
  // If model_version is -1 then look in the end status to find the
  // latest (highest valued version) and use that as the version.
  int64_t status_model_version = 0;
  if (model_version < 0) {
    for (const auto& vp : end_status.version_status()) {
      status_model_version = std::max(status_model_version, vp.first);
    }
  } else {
    status_model_version = model_version;
  }

  const auto& vend_itr = end_status.version_status().find(status_model_version);
  if (vend_itr == end_status.version_status().end()) {
    return nic::Error(
        ni::RequestStatusCode::INTERNAL, "missing model version status");
  } else {
    const auto& end_itr =
        vend_itr->second.infer_stats().find(manager_->BatchSize());
    if (end_itr == vend_itr->second.infer_stats().end()) {
      return nic::Error(
          ni::RequestStatusCode::INTERNAL, "missing inference stats");
    } else {
      uint64_t start_cnt = 0;
      uint64_t start_cumm_time_ns = 0;
      uint64_t start_queue_time_ns = 0;
      uint64_t start_compute_time_ns = 0;

      const auto& vstart_itr =
          start_status.version_status().find(status_model_version);
      if (vstart_itr != start_status.version_status().end()) {
        const auto& start_itr =
            vstart_itr->second.infer_stats().find(manager_->BatchSize());
        if (start_itr != vstart_itr->second.infer_stats().end()) {
          start_cnt = start_itr->second.success().count();
          start_cumm_time_ns = start_itr->second.success().total_time_ns();
          start_queue_time_ns = start_itr->second.queue().total_time_ns();
          start_compute_time_ns = start_itr->second.compute().total_time_ns();
        }
      }

      server_stats->request_count =
          end_itr->second.success().count() - start_cnt;
      server_stats->cumm_time_ns =
          end_itr->second.success().total_time_ns() - start_cumm_time_ns;
      server_stats->queue_time_ns =
          end_itr->second.queue().total_time_ns() - start_queue_time_ns;
      server_stats->compute_time_ns =
          end_itr->second.compute().total_time_ns() - start_compute_time_ns;
    }
  }

  return nic::Error::Success;
}

nic::Error
InferenceProfiler::SummarizeServerStats(
    const ModelInfo model_info,
    const std::map<std::string, ni::ModelStatus>& start_status,
    const std::map<std::string, ni::ModelStatus>& end_status,
    ServerSideStats* server_stats)
{
  RETURN_IF_ERROR(SummarizeServerModelStats(
      model_info.first, model_info.second,
      start_status.find(model_info.first)->second,
      end_status.find(model_info.first)->second, server_stats));

  // Summarize the composing models, if any.
  for (const auto& composing_model_info : composing_models_map_[model_info]) {
    auto it = server_stats->composing_models_stat
                  .emplace(composing_model_info, ServerSideStats())
                  .first;
    if (composing_models_map_.find(composing_model_info) !=
        composing_models_map_.end()) {
      RETURN_IF_ERROR(SummarizeServerStats(
          composing_model_info, start_status, end_status, &(it->second)));
    } else {
      RETURN_IF_ERROR(SummarizeServerModelStats(
          composing_model_info.first, composing_model_info.second,
          start_status.find(composing_model_info.first)->second,
          end_status.find(composing_model_info.first)->second, &(it->second)));
    }
  }

  return nic::Error::Success;
}

nic::Error
InferenceProfiler::SummarizeServerStats(
    const std::map<std::string, ni::ModelStatus>& start_status,
    const std::map<std::string, ni::ModelStatus>& end_status,
    ServerSideStats* server_stats)
{
  RETURN_IF_ERROR(SummarizeServerStats(
      std::make_pair(model_name_, model_version_), start_status, end_status,
      server_stats));
  return nic::Error::Success;
}

}  // namespace perfclient
