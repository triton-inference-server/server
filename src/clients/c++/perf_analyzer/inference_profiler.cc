// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "src/clients/c++/perf_analyzer/inference_profiler.h"

#include <math.h>
#include <algorithm>
#include <limits>
#include <queue>


namespace perfanalyzer {

namespace {

inline uint64_t
AverageDurationInUs(const uint64_t total_time_in_ns, const uint64_t cnt)
{
  return total_time_in_ns / (cnt * 1000);
}

EnsembleDurations
GetTotalEnsembleDurations(const ServerSideStats& stats)
{
  EnsembleDurations result;
  for (const auto& model_stats : stats.composing_models_stat) {
    if (model_stats.second.composing_models_stat.empty()) {
      const uint64_t cnt = model_stats.second.success_count;
      if (cnt != 0) {
        result.total_queue_time_us +=
            AverageDurationInUs(model_stats.second.queue_time_ns, cnt);
        result.total_compute_time_us +=
            AverageDurationInUs(model_stats.second.compute_input_time_ns, cnt) +
            AverageDurationInUs(model_stats.second.compute_infer_time_ns, cnt) +
            AverageDurationInUs(model_stats.second.compute_output_time_ns, cnt);
      }
    } else {
      const auto this_ensemble_duration =
          GetTotalEnsembleDurations(model_stats.second);
      result.total_queue_time_us += this_ensemble_duration.total_queue_time_us;
      result.total_compute_time_us +=
          this_ensemble_duration.total_compute_time_us;
    }
  }
  return result;
}


size_t
GetOverheadDuration(size_t total_time, size_t queue_time, size_t compute_time)
{
  return (total_time > queue_time + compute_time)
             ? (total_time - queue_time - compute_time)
             : 0;
}

cb::Error
ReportServerSideStats(const ServerSideStats& stats, const int iteration)
{
  const std::string ident = std::string(2 * iteration, ' ');

  const uint64_t exec_cnt = stats.execution_count;
  const uint64_t infer_cnt = stats.inference_count;

  const uint64_t cnt = stats.success_count;
  if (cnt == 0) {
    std::cout << ident << "  Request count: " << cnt << std::endl;
    return cb::Error::Success;
  }

  const uint64_t cumm_avg_us = AverageDurationInUs(stats.cumm_time_ns, cnt);

  std::cout << ident << "  Inference count: " << infer_cnt << std::endl
            << ident << "  Execution count: " << exec_cnt << std::endl
            << ident << "  Successful request count: " << exec_cnt << std::endl
            << ident << "  Avg request latency: " << cumm_avg_us << " usec";
  if (stats.composing_models_stat.empty()) {
    const uint64_t queue_avg_us = AverageDurationInUs(stats.queue_time_ns, cnt);
    const uint64_t compute_input_avg_us =
        AverageDurationInUs(stats.compute_input_time_ns, cnt);
    const uint64_t compute_infer_avg_us =
        AverageDurationInUs(stats.compute_infer_time_ns, cnt);
    const uint64_t compute_output_avg_us =
        AverageDurationInUs(stats.compute_output_time_ns, cnt);
    const uint64_t compute_avg_us =
        compute_input_avg_us + compute_infer_avg_us + compute_output_avg_us;
    std::cout << " (overhead "
              << GetOverheadDuration(cumm_avg_us, queue_avg_us, compute_avg_us)
              << " usec + "
              << "queue " << queue_avg_us << " usec + "
              << "compute input " << compute_input_avg_us << " usec + "
              << "compute infer " << compute_infer_avg_us << " usec + "
              << "compute output " << compute_output_avg_us << " usec)"
              << std::endl
              << std::endl;
  } else {
    const auto ensemble_times = GetTotalEnsembleDurations(stats);
    std::cout << " (overhead "
              << GetOverheadDuration(
                     cumm_avg_us, ensemble_times.total_queue_time_us,
                     ensemble_times.total_compute_time_us)
              << " usec + "
              << "queue " << ensemble_times.total_queue_time_us << " usec + "
              << "compute " << ensemble_times.total_compute_time_us << " usec)"
              << std::endl
              << std::endl;

    std::cout << ident << "Composing models: " << std::endl;
    for (const auto& model_stats : stats.composing_models_stat) {
      const auto& model_identifier = model_stats.first;
      std::cout << ident << model_identifier.first
                << ", version: " << model_identifier.second << std::endl;
      ReportServerSideStats(model_stats.second, iteration + 1);
    }
  }

  return cb::Error::Success;
}

cb::Error
ReportClientSideStats(
    const ClientSideStats& stats, const int64_t percentile,
    const cb::ProtocolType protocol, const bool verbose,
    const bool on_sequence_model, const bool include_lib_stats)
{
  const uint64_t avg_latency_us = stats.avg_latency_ns / 1000;
  const uint64_t std_us = stats.std_us;

  const uint64_t avg_request_time_us = stats.avg_request_time_ns / 1000;
  const uint64_t avg_send_time_us = stats.avg_send_time_ns / 1000;
  const uint64_t avg_receive_time_us = stats.avg_receive_time_ns / 1000;
  const uint64_t avg_response_wait_time_us =
      avg_request_time_us - avg_send_time_us - avg_receive_time_us;

  std::string client_library_detail = "    ";
  if (include_lib_stats) {
    if (protocol == cb::ProtocolType::GRPC) {
      client_library_detail +=
          "Avg gRPC time: " + std::to_string(avg_request_time_us) + " usec (";
      if (!verbose) {
        client_library_detail +=
            "(un)marshal request/response " +
            std::to_string(avg_send_time_us + avg_receive_time_us) +
            " usec + response wait " +
            std::to_string(avg_response_wait_time_us) + " usec)";
      } else {
        client_library_detail += "marshal " + std::to_string(avg_send_time_us) +
                                 " usec + response wait " +
                                 std::to_string(avg_response_wait_time_us) +
                                 " usec + unmarshal " +
                                 std::to_string(avg_receive_time_us) + " usec)";
      }
    } else {
      client_library_detail +=
          "Avg HTTP time: " + std::to_string(avg_request_time_us) + " usec (";
      if (!verbose) {
        client_library_detail +=
            "send/recv " +
            std::to_string(avg_send_time_us + avg_receive_time_us) +
            " usec + response wait " +
            std::to_string(avg_response_wait_time_us) + " usec)";
      } else {
        client_library_detail += "send " + std::to_string(avg_send_time_us) +
                                 " usec + response wait " +
                                 std::to_string(avg_response_wait_time_us) +
                                 " usec + receive " +
                                 std::to_string(avg_receive_time_us) + " usec)";
      }
    }
  }

  std::cout << "    Request count: " << stats.request_count << std::endl;
  if (stats.delayed_request_count != 0) {
    std::cout << "    Delayed Request Count: " << stats.delayed_request_count
              << std::endl;
  }
  if (on_sequence_model) {
    std::cout << "    Sequence count: " << stats.sequence_count << " ("
              << stats.sequence_per_sec << " seq/sec)" << std::endl;
  }
  std::cout << "    Throughput: " << stats.infer_per_sec << " infer/sec"
            << std::endl;
  if (percentile == -1) {
    std::cout << "    Avg latency: " << avg_latency_us << " usec"
              << " (standard deviation " << std_us << " usec)" << std::endl;
  }
  for (const auto& percentile : stats.percentile_latency_ns) {
    std::cout << "    p" << percentile.first
              << " latency: " << (percentile.second / 1000) << " usec"
              << std::endl;
  }

  std::cout << client_library_detail << std::endl;

  return cb::Error::Success;
}

cb::Error
Report(
    const PerfStatus& summary, const int64_t percentile,
    const cb::ProtocolType protocol, const bool verbose,
    const bool include_lib_stats, const bool include_server_stats)
{
  std::cout << "  Client: " << std::endl;
  ReportClientSideStats(
      summary.client_stats, percentile, protocol, verbose,
      summary.on_sequence_model, include_lib_stats);

  if (include_server_stats) {
    std::cout << "  Server: " << std::endl;
    ReportServerSideStats(summary.server_stats, 1);
  }

  return cb::Error::Success;
}

}  // namespace

cb::Error
InferenceProfiler::Create(
    const bool verbose, const double stability_threshold,
    const uint64_t measurement_window_ms, const size_t max_trials,
    const int64_t percentile, const uint64_t latency_threshold_ms_,
    const cb::ProtocolType protocol, std::shared_ptr<ModelParser>& parser,
    std::unique_ptr<cb::ClientBackend> profile_backend,
    std::unique_ptr<LoadManager> manager,
    std::unique_ptr<InferenceProfiler>* profiler)
{
  std::unique_ptr<InferenceProfiler> local_profiler(new InferenceProfiler(
      verbose, stability_threshold, measurement_window_ms, max_trials,
      (percentile != -1), percentile, latency_threshold_ms_, protocol, parser,
      std::move(profile_backend), std::move(manager)));

  *profiler = std::move(local_profiler);
  return cb::Error::Success;
}

InferenceProfiler::InferenceProfiler(
    const bool verbose, const double stability_threshold,
    const int32_t measurement_window_ms, const size_t max_trials,
    const bool extra_percentile, const size_t percentile,
    const uint64_t latency_threshold_ms_, const cb::ProtocolType protocol,
    std::shared_ptr<ModelParser>& parser,
    std::unique_ptr<cb::ClientBackend> profile_backend,
    std::unique_ptr<LoadManager> manager)
    : verbose_(verbose), measurement_window_ms_(measurement_window_ms),
      max_trials_(max_trials), extra_percentile_(extra_percentile),
      percentile_(percentile), latency_threshold_ms_(latency_threshold_ms_),
      protocol_(protocol), parser_(parser),
      profile_backend_(std::move(profile_backend)), manager_(std::move(manager))
{
  load_parameters_.stability_threshold = stability_threshold;
  load_parameters_.stability_window = 3;
  if (profile_backend_->Kind() == cb::BackendKind::TRITON) {
    // Measure and report client library stats only when the model
    // is not decoupled.
    include_lib_stats_ = (!parser_->IsDecoupled());
    // Measure and report server statistics only when the server
    // supports the statistics extension.
    std::set<std::string> extensions;
    profile_backend_->ServerExtensions(&extensions);
    include_server_stats_ = (extensions.find("statistics") != extensions.end());
  } else {
    include_lib_stats_ = true;
    include_server_stats_ = false;
  }
}

cb::Error
InferenceProfiler::Profile(
    const size_t concurrent_request_count, std::vector<PerfStatus>& summary,
    bool* meets_threshold)
{
  cb::Error err;
  PerfStatus status_summary;

  status_summary.concurrency = concurrent_request_count;

  bool is_stable = false;
  *meets_threshold = true;

  RETURN_IF_ERROR(dynamic_cast<ConcurrencyManager*>(manager_.get())
                      ->ChangeConcurrencyLevel(concurrent_request_count));

  err = ProfileHelper(false /* clean_starts */, status_summary, &is_stable);
  if (err.IsOk()) {
    err = Report(
        status_summary, percentile_, protocol_, verbose_, include_lib_stats_,
        include_server_stats_);
    summary.push_back(status_summary);
    uint64_t stabilizing_latency_ms =
        status_summary.stabilizing_latency_ns / (1000 * 1000);
    if (!err.IsOk()) {
      std::cerr << err << std::endl;
      *meets_threshold = false;
    } else if (
        (stabilizing_latency_ms >= latency_threshold_ms_) &&
        (latency_threshold_ms_ != NO_LIMIT)) {
      std::cerr << "Measured latency went over the set limit of "
                << latency_threshold_ms_ << " msec. " << std::endl;
      *meets_threshold = false;
    } else if (!is_stable) {
      std::cerr << "Failed to obtain stable measurement within " << max_trials_
                << " measurement windows for concurrency "
                << concurrent_request_count << ". Please try to "
                << "increase the --measurement-interval." << std::endl;
      *meets_threshold = false;
    }
  } else {
    return err;
  }

  return cb::Error::Success;
}

cb::Error
InferenceProfiler::Profile(
    const double request_rate, std::vector<PerfStatus>& summary,
    bool* meets_threshold)
{
  cb::Error err;
  PerfStatus status_summary;

  status_summary.request_rate = request_rate;

  bool is_stable = false;
  *meets_threshold = true;

  RETURN_IF_ERROR(dynamic_cast<RequestRateManager*>(manager_.get())
                      ->ChangeRequestRate(request_rate));

  err = ProfileHelper(false /*clean_starts*/, status_summary, &is_stable);
  if (err.IsOk()) {
    err = Report(
        status_summary, percentile_, protocol_, verbose_, include_lib_stats_,
        include_server_stats_);
    summary.push_back(status_summary);
    uint64_t stabilizing_latency_ms =
        status_summary.stabilizing_latency_ns / (1000 * 1000);
    if (!err.IsOk()) {
      std::cerr << err << std::endl;
      *meets_threshold = false;
    } else if (
        (stabilizing_latency_ms >= latency_threshold_ms_) &&
        (latency_threshold_ms_ != NO_LIMIT)) {
      std::cerr << "Measured latency went over the set limit of "
                << latency_threshold_ms_ << " msec. " << std::endl;
      *meets_threshold = false;
    } else if (!is_stable) {
      std::cerr << "Failed to obtain stable measurement." << std::endl;
      *meets_threshold = false;
    }
  } else {
    return err;
  }

  return cb::Error::Success;
}

cb::Error
InferenceProfiler::Profile(
    std::vector<PerfStatus>& summary, bool* meets_threshold)
{
  cb::Error err;
  PerfStatus status_summary;

  RETURN_IF_ERROR(
      dynamic_cast<CustomLoadManager*>(manager_.get())->InitCustomIntervals());
  RETURN_IF_ERROR(dynamic_cast<CustomLoadManager*>(manager_.get())
                      ->GetCustomRequestRate(&status_summary.request_rate));

  bool is_stable = false;
  *meets_threshold = true;

  err = ProfileHelper(true /* clean_starts */, status_summary, &is_stable);
  if (err.IsOk()) {
    err = Report(
        status_summary, percentile_, protocol_, verbose_, include_lib_stats_,
        include_server_stats_);
    summary.push_back(status_summary);
    uint64_t stabilizing_latency_ms =
        status_summary.stabilizing_latency_ns / (1000 * 1000);
    if (!err.IsOk()) {
      std::cerr << err << std::endl;
      *meets_threshold = false;
    } else if (
        (stabilizing_latency_ms >= latency_threshold_ms_) &&
        (latency_threshold_ms_ != NO_LIMIT)) {
      std::cerr << "Measured latency went over the set limit of "
                << latency_threshold_ms_ << " msec. " << std::endl;
      *meets_threshold = false;
    } else if (!is_stable) {
      std::cerr << "Failed to obtain stable measurement." << std::endl;
      *meets_threshold = false;
    }
  } else {
    return err;
  }

  return cb::Error::Success;
}

cb::Error
InferenceProfiler::ProfileHelper(
    const bool clean_starts, PerfStatus& status_summary, bool* is_stable)
{
  // Start measurement
  LoadStatus load_status;
  size_t completed_trials = 0;
  std::queue<cb::Error> error;

  do {
    RETURN_IF_ERROR(manager_->CheckHealth());

    // Needed to obtain stable measurements
    if (clean_starts) {
      manager_->ResetWorkers();
    }

    error.push(Measure(status_summary));
    if (error.size() >= load_parameters_.stability_window) {
      error.pop();
    }

    if (error.back().IsOk()) {
      load_status.infer_per_sec.push_back(
          status_summary.client_stats.infer_per_sec);
      load_status.latencies.push_back(status_summary.stabilizing_latency_ns);
    } else {
      load_status.infer_per_sec.push_back(0);
      load_status.latencies.push_back(std::numeric_limits<uint64_t>::max());
    }

    load_status.avg_ips +=
        load_status.infer_per_sec.back() / load_parameters_.stability_window;
    load_status.avg_latency +=
        load_status.latencies.back() / load_parameters_.stability_window;
    if (verbose_) {
      if (error.back().IsOk()) {
        std::cout << "  Pass [" << (completed_trials + 1)
                  << "] throughput: " << load_status.infer_per_sec.back()
                  << " infer/sec. ";
        if (extra_percentile_) {
          std::cout << "p" << percentile_ << " latency: "
                    << (status_summary.client_stats.percentile_latency_ns
                            .find(percentile_)
                            ->second /
                        1000)
                    << " usec" << std::endl;
        } else {
          std::cout << "Avg latency: "
                    << (status_summary.client_stats.avg_latency_ns / 1000)
                    << " usec (std " << status_summary.client_stats.std_us
                    << " usec)" << std::endl;
        }
      } else {
        std::cout << "  Pass [" << (completed_trials + 1)
                  << "] cb::Error: " << error.back().Message() << std::endl;
      }
    }

    if (load_status.infer_per_sec.size() >= load_parameters_.stability_window) {
      size_t idx =
          load_status.infer_per_sec.size() - load_parameters_.stability_window;
      if (load_status.infer_per_sec.size() >
          load_parameters_.stability_window) {
        load_status.avg_ips -= load_status.infer_per_sec[idx - 1] /
                               load_parameters_.stability_window;
        load_status.avg_latency -=
            load_status.latencies[idx - 1] / load_parameters_.stability_window;
      }
      *is_stable = true;
      bool within_threshold = false;
      for (; idx < load_status.infer_per_sec.size(); idx++) {
        if (load_status.infer_per_sec[idx] == 0) {
          *is_stable = false;
        }
        if ((load_status.latencies[idx] <
             (latency_threshold_ms_ * 1000 * 1000))) {
          within_threshold = true;
        }
        // We call it complete only if stability_window measurements are within
        // +/-(stability_threshold)% of the average infer per second and latency
        if ((load_status.infer_per_sec[idx] <
             load_status.avg_ips *
                 (1 - load_parameters_.stability_threshold)) ||
            (load_status.infer_per_sec[idx] >
             load_status.avg_ips *
                 (1 + load_parameters_.stability_threshold))) {
          *is_stable = false;
        }
        if ((load_status.latencies[idx] <
             load_status.avg_latency *
                 (1 - load_parameters_.stability_threshold)) ||
            (load_status.latencies[idx] >
             load_status.avg_latency *
                 (1 + load_parameters_.stability_threshold))) {
          *is_stable = false;
        }
      }
      if (*is_stable) {
        break;
      }
      if ((!within_threshold) && (latency_threshold_ms_ != NO_LIMIT)) {
        break;
      }
    }
    completed_trials++;
  } while ((!early_exit) && (completed_trials < max_trials_));

  // return the appropriate error which might have occured in the
  // stability_window for its proper handling.
  while (!error.empty()) {
    if (!error.front().IsOk()) {
      return error.front();
    } else {
      error.pop();
    }
  }

  if (early_exit) {
    return cb::Error("Received exit signal.");
  }
  return cb::Error::Success;
}

cb::Error
InferenceProfiler::GetServerSideStatus(
    std::map<cb::ModelIdentifier, cb::ModelStatistics>* model_stats)
{
  if ((parser_->SchedulerType() == ModelParser::ENSEMBLE) ||
      (parser_->SchedulerType() == ModelParser::ENSEMBLE_SEQUENCE)) {
    RETURN_IF_ERROR(profile_backend_->ModelInferenceStatistics(model_stats));
  } else {
    RETURN_IF_ERROR(profile_backend_->ModelInferenceStatistics(
        model_stats, parser_->ModelName(), parser_->ModelVersion()));
  }
  return cb::Error::Success;
}

// Used for measurement
cb::Error
InferenceProfiler::Measure(PerfStatus& status_summary)
{
  std::map<cb::ModelIdentifier, cb::ModelStatistics> start_status;
  std::map<cb::ModelIdentifier, cb::ModelStatistics> end_status;
  cb::InferStat start_stat;
  cb::InferStat end_stat;

  if (include_server_stats_) {
    RETURN_IF_ERROR(GetServerSideStatus(&start_status));
  }
  RETURN_IF_ERROR(manager_->GetAccumulatedClientStat(&start_stat));

  // Wait for specified time interval in msec
  std::this_thread::sleep_for(
      std::chrono::milliseconds((uint64_t)(measurement_window_ms_ * 1.2)));

  RETURN_IF_ERROR(manager_->GetAccumulatedClientStat(&end_stat));

  // Get server status and then print report on difference between
  // before and after status.
  if (include_server_stats_) {
    RETURN_IF_ERROR(GetServerSideStatus(&end_status));
  }

  TimestampVector current_timestamps;
  RETURN_IF_ERROR(manager_->SwapTimestamps(current_timestamps));

  RETURN_IF_ERROR(Summarize(
      current_timestamps, start_status, end_status, start_stat, end_stat,
      status_summary));

  return cb::Error::Success;
}

cb::Error
InferenceProfiler::Summarize(
    const TimestampVector& timestamps,
    const std::map<cb::ModelIdentifier, cb::ModelStatistics>& start_status,
    const std::map<cb::ModelIdentifier, cb::ModelStatistics>& end_status,
    const cb::InferStat& start_stat, const cb::InferStat& end_stat,
    PerfStatus& summary)
{
  size_t valid_sequence_count = 0;
  size_t delayed_request_count = 0;

  // Get measurement from requests that fall within the time interval
  std::pair<uint64_t, uint64_t> valid_range;
  MeasurementTimestamp(timestamps, &valid_range);
  std::vector<uint64_t> latencies;
  ValidLatencyMeasurement(
      timestamps, valid_range, valid_sequence_count, delayed_request_count,
      &latencies);

  RETURN_IF_ERROR(SummarizeLatency(latencies, summary));
  RETURN_IF_ERROR(SummarizeClientStat(
      start_stat, end_stat, valid_range.second - valid_range.first,
      latencies.size(), valid_sequence_count, delayed_request_count, summary));

  if (include_server_stats_) {
    RETURN_IF_ERROR(SummarizeServerStats(
        start_status, end_status, &(summary.server_stats)));
  }

  return cb::Error::Success;
}

void
InferenceProfiler::MeasurementTimestamp(
    const TimestampVector& timestamps,
    std::pair<uint64_t, uint64_t>* valid_range)
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

  *valid_range = std::make_pair(start_ns, end_ns);
}

void
InferenceProfiler::ValidLatencyMeasurement(
    const TimestampVector& timestamps,
    const std::pair<uint64_t, uint64_t>& valid_range,
    size_t& valid_sequence_count, size_t& delayed_request_count,
    std::vector<uint64_t>* valid_latencies)
{
  valid_latencies->clear();
  valid_sequence_count = 0;
  for (auto& timestamp : timestamps) {
    uint64_t request_start_ns = TIMESPEC_TO_NANOS(std::get<0>(timestamp));
    uint64_t request_end_ns = TIMESPEC_TO_NANOS(std::get<1>(timestamp));

    if (request_start_ns <= request_end_ns) {
      // Only counting requests that end within the time interval
      if ((request_end_ns >= valid_range.first) &&
          (request_end_ns <= valid_range.second)) {
        valid_latencies->push_back(request_end_ns - request_start_ns);
        // Just add the sequence_end flag here.
        if (std::get<2>(timestamp)) {
          valid_sequence_count++;
        }
        if (std::get<3>(timestamp)) {
          delayed_request_count++;
        }
      }
    }
  }

  // Always sort measured latencies as percentile will be reported as default
  std::sort(valid_latencies->begin(), valid_latencies->end());
}

cb::Error
InferenceProfiler::SummarizeLatency(
    const std::vector<uint64_t>& latencies, PerfStatus& summary)
{
  if (latencies.size() == 0) {
    return cb::Error(
        "No valid requests recorded within time interval."
        " Please use a larger time window.");
  }

  uint64_t tol_latency_ns = 0;
  uint64_t tol_square_latency_us = 0;

  for (const auto& latency : latencies) {
    tol_latency_ns += latency;
    tol_square_latency_us += (latency * latency) / (1000 * 1000);
  }

  summary.client_stats.avg_latency_ns = tol_latency_ns / latencies.size();

  // retrieve other interesting percentile
  summary.client_stats.percentile_latency_ns.clear();
  std::set<size_t> percentiles{50, 90, 95, 99};
  if (extra_percentile_) {
    percentiles.emplace(percentile_);
  }

  for (const auto percentile : percentiles) {
    size_t index = (percentile / 100.0) * (latencies.size() - 1) + 0.5;
    summary.client_stats.percentile_latency_ns.emplace(
        percentile, latencies[index]);
  }

  if (extra_percentile_) {
    summary.stabilizing_latency_ns =
        summary.client_stats.percentile_latency_ns.find(percentile_)->second;
  } else {
    summary.stabilizing_latency_ns = summary.client_stats.avg_latency_ns;
  }

  // calculate standard deviation
  uint64_t expected_square_latency_us =
      tol_square_latency_us / latencies.size();
  uint64_t square_avg_latency_us = (summary.client_stats.avg_latency_ns *
                                    summary.client_stats.avg_latency_ns) /
                                   (1000 * 1000);
  uint64_t var_us = (expected_square_latency_us > square_avg_latency_us)
                        ? (expected_square_latency_us - square_avg_latency_us)
                        : 0;
  summary.client_stats.std_us = (uint64_t)(sqrt(var_us));

  return cb::Error::Success;
}

cb::Error
InferenceProfiler::SummarizeClientStat(
    const cb::InferStat& start_stat, const cb::InferStat& end_stat,
    const uint64_t duration_ns, const size_t valid_request_count,
    const size_t valid_sequence_count, const size_t delayed_request_count,
    PerfStatus& summary)
{
  summary.on_sequence_model =
      ((parser_->SchedulerType() == ModelParser::SEQUENCE) ||
       (parser_->SchedulerType() == ModelParser::ENSEMBLE_SEQUENCE));
  summary.batch_size = std::max(manager_->BatchSize(), (size_t)1);
  summary.client_stats.request_count = valid_request_count;
  summary.client_stats.sequence_count = valid_sequence_count;
  summary.client_stats.delayed_request_count = delayed_request_count;
  summary.client_stats.duration_ns = duration_ns;
  float client_duration_sec = (float)summary.client_stats.duration_ns /
                              nvidia::inferenceserver::NANOS_PER_SECOND;
  summary.client_stats.sequence_per_sec =
      valid_sequence_count / client_duration_sec;
  summary.client_stats.infer_per_sec =
      (valid_request_count * summary.batch_size) / client_duration_sec;

  if (include_lib_stats_) {
    size_t completed_count =
        end_stat.completed_request_count - start_stat.completed_request_count;
    uint64_t request_time_ns = end_stat.cumulative_total_request_time_ns -
                               start_stat.cumulative_total_request_time_ns;
    uint64_t send_time_ns =
        end_stat.cumulative_send_time_ns - start_stat.cumulative_send_time_ns;
    uint64_t receive_time_ns = end_stat.cumulative_receive_time_ns -
                               start_stat.cumulative_receive_time_ns;
    if (completed_count != 0) {
      summary.client_stats.avg_request_time_ns =
          request_time_ns / completed_count;
      summary.client_stats.avg_send_time_ns = send_time_ns / completed_count;
      summary.client_stats.avg_receive_time_ns =
          receive_time_ns / completed_count;
    }
  }

  return cb::Error::Success;
}

cb::Error
InferenceProfiler::SummarizeServerStatsHelper(
    const cb::ModelIdentifier& model_identifier,
    const std::map<cb::ModelIdentifier, cb::ModelStatistics>& start_status,
    const std::map<cb::ModelIdentifier, cb::ModelStatistics>& end_status,
    ServerSideStats* server_stats)
{
  // If model_version is an empty string then look in the end status to find the
  // latest (highest valued version) and use that as the version.
  int64_t status_model_version = -1;
  if (model_identifier.second.empty()) {
    for (const auto& id : end_status) {
      // Model name should match
      if (model_identifier.first.compare(id.first.first) == 0) {
        int64_t this_version = std::stoll(id.first.second);
        status_model_version = std::max(status_model_version, this_version);
      }
    }
  } else {
    status_model_version = std::stoll(model_identifier.second);
  }

  if (status_model_version == -1) {
    return cb::Error("failed to determine the requested model version");
  }

  const std::pair<std::string, std::string> this_id(
      model_identifier.first, std::to_string(status_model_version));

  const auto& end_itr = end_status.find(this_id);
  if (end_itr == end_status.end()) {
    return cb::Error("missing statistics for requested model");
  } else {
    uint64_t start_infer_cnt = 0;
    uint64_t start_exec_cnt = 0;
    uint64_t start_cnt = 0;
    uint64_t start_cumm_time_ns = 0;
    uint64_t start_queue_time_ns = 0;
    uint64_t start_compute_input_time_ns = 0;
    uint64_t start_compute_infer_time_ns = 0;
    uint64_t start_compute_output_time_ns = 0;

    const auto& start_itr = start_status.find(this_id);
    if (start_itr != start_status.end()) {
      start_infer_cnt = start_itr->second.inference_count_;
      start_exec_cnt = start_itr->second.execution_count_;
      start_cnt = start_itr->second.success_count_;
      start_cumm_time_ns = start_itr->second.cumm_time_ns_;
      start_queue_time_ns = start_itr->second.queue_time_ns_;
      start_compute_input_time_ns = start_itr->second.compute_input_time_ns_;
      start_compute_infer_time_ns = start_itr->second.compute_infer_time_ns_;
      start_compute_output_time_ns = start_itr->second.compute_output_time_ns_;
    }

    server_stats->inference_count =
        end_itr->second.inference_count_ - start_infer_cnt;
    server_stats->execution_count =
        end_itr->second.execution_count_ - start_exec_cnt;
    server_stats->success_count = end_itr->second.success_count_ - start_cnt;
    server_stats->cumm_time_ns =
        end_itr->second.cumm_time_ns_ - start_cumm_time_ns;
    server_stats->queue_time_ns =
        end_itr->second.queue_time_ns_ - start_queue_time_ns;
    server_stats->compute_input_time_ns =
        end_itr->second.compute_input_time_ns_ - start_compute_input_time_ns;
    server_stats->compute_infer_time_ns =
        end_itr->second.compute_infer_time_ns_ - start_compute_infer_time_ns;
    server_stats->compute_output_time_ns =
        end_itr->second.compute_output_time_ns_ - start_compute_output_time_ns;
  }

  return cb::Error::Success;
}

cb::Error
InferenceProfiler::SummarizeServerStats(
    const cb::ModelIdentifier& model_identifier,
    const std::map<cb::ModelIdentifier, cb::ModelStatistics>& start_status,
    const std::map<cb::ModelIdentifier, cb::ModelStatistics>& end_status,
    ServerSideStats* server_stats)
{
  RETURN_IF_ERROR(SummarizeServerStatsHelper(
      model_identifier, start_status, end_status, server_stats));

  // Summarize the composing models, if any.
  for (const auto& composing_model_identifier :
       (*parser_->GetComposingModelMap())[model_identifier.first]) {
    auto it = server_stats->composing_models_stat
                  .emplace(composing_model_identifier, ServerSideStats())
                  .first;
    RETURN_IF_ERROR(SummarizeServerStats(
        composing_model_identifier, start_status, end_status, &(it->second)));
  }

  return cb::Error::Success;
}

cb::Error
InferenceProfiler::SummarizeServerStats(
    const std::map<cb::ModelIdentifier, cb::ModelStatistics>& start_status,
    const std::map<cb::ModelIdentifier, cb::ModelStatistics>& end_status,
    ServerSideStats* server_stats)
{
  RETURN_IF_ERROR(SummarizeServerStats(
      std::make_pair(parser_->ModelName(), parser_->ModelVersion()),
      start_status, end_status, server_stats));
  return cb::Error::Success;
}

}  // namespace perfanalyzer
