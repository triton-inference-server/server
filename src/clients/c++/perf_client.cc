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

#include <getopt.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include "src/clients/c++/request_grpc.h"
#include "src/clients/c++/request_http.h"
#include "src/core/constants.h"

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

//==============================================================================
// Perf Client
//
// Perf client provides various metrics to measure the performance of
// the inference server. It can either be used to measure the throughput,
// latency and time distribution under specific setting (i.e. fixed batch size
// and fixed concurrent requests), or be used to generate throughput-latency
// data point under dynamic setting (i.e. collecting throughput-latency data
// under different load level).
//
// The following data is collected and used as part of the metrics:
// - Throughput (infer/sec):
//     The number of inference processed per second as seen by the client.
//     The number of inference is measured by the multiplication of the number
//     of requests and their batch size. And the total time is the time elapsed
//     from when the client starts sending requests to when the client received
//     all responses.
// - Latency (usec):
//     The average elapsed time between when a request is sent and
//     when the response for the request is received.
//
// There are two settings (see -d option) for the data collection:
// - Fixed concurrent request mode:
//     In this setting, the client will maintain a fixed number of concurrent
//     requests sent to the server (see -t option). See ConcurrencyManager for
//     more detail. The number of requests will be the total number of requests
//     sent within the time interval for measurement (see -p option) and
//     the latency will be the average latency across all requests.
//
//     Besides throughput and latency, which is measured in client side,
//     the following data measured by the server will also be reported
//     in this setting:
//     - Concurrent request: the number of concurrent requests as specified
//         in -t option
//     - Batch size: the batch size of each request as specified in -b option
//     - Inference count: batch size * number of inference requests
//     - Cumulative time: the total time between request received and
//         response sent on the requests sent by perf client.
//     - Average Cumulative time: cumulative time / number of inference requests
//     - Compute time: the total time it takes to run inferencing including time
//         copying input tensors to GPU memory, time executing the model,
//         and time copying output tensors from GPU memory for the requests
//         sent by perf client.
//     - Average compute time: compute time / number of inference requests
//     - Queue time: the total time it takes to wait for an available model
//         instance for the requests sent by perf client.
//     - Average queue time: queue time / number of inference requests
//
// - Dynamic concurrent request mode:
//     In this setting, the client will perform the following procedure:
//       1. Follows the procedure in fixed concurrent request mode using
//          k concurrent requests (k starts at 1).
//       2. Gathers data reported from step 1.
//       3. Increases k by 1 and repeats step 1 and 2 until latency from current
//          iteration exceeds latency threshold (see -l option)
//     At each iteration, the data mentioned in fixed concurrent request mode
//     will be reported. Besides that, after the procedure above, a collection
//     of "throughput, latency, concurrent request count" tuples will be
//     reported in increasing load level order.
//
// Options:
// -b: batch size for each request sent.
// -t: number of concurrent requests sent. If -d is set, -t indicate the number
//     of concurrent requests to start with ("starting concurrency" level).
// -d: enable dynamic concurrent request mode.
// -l: latency threshold in msec, will have no effect if -d is not set.
// -p: time interval for each measurement window in msec.
//
// For detail of the options not listed, please refer to the usage.
//

namespace {

volatile bool early_exit = false;

void
SignalHandler(int signum)
{
  std::cout << "Interrupt signal (" << signum << ") received." << std::endl
            << "Waiting for in-flight inferences to complete." << std::endl;

  early_exit = true;
}

typedef struct PerformanceStatusStruct {
  uint32_t concurrency;
  size_t batch_size;
  // Request count and elapsed time measured by server
  uint64_t server_request_count;
  uint64_t server_cumm_time_ns;
  uint64_t server_queue_time_ns;
  uint64_t server_compute_time_ns;

  // Request count and elapsed time measured by client
  uint64_t client_request_count;
  // Only record sequences that finish within the measurement window
  uint64_t client_sequence_count;
  uint64_t client_duration_ns;
  uint64_t client_min_latency_ns;
  uint64_t client_max_latency_ns;
  uint64_t client_avg_latency_ns;
  // Using usec to avoid square of large number (large in nsec)
  uint64_t std_us;
  uint64_t client_avg_request_time_ns;
  uint64_t client_avg_send_time_ns;
  uint64_t client_avg_receive_time_ns;
  // Per sec stat
  int client_infer_per_sec;
  int client_sequence_per_sec;
  bool on_sequence_model;
} PerfStatus;


enum ProtocolType { HTTP = 0, GRPC = 1 };

//==============================================================================
// Concurrency Manager
//
// An instance of concurrency manager will be created at the beginning of the
// perf client and it will be used to simulate different load level in respect
// to number of concurrent infer requests and to report the performance status.
// After the creation, perf client obtains performance status under the setting
// specified in command line options by calling Step() function.
//
// (Tentative usage)
// std::unique_ptr<ConcurrencyManager> manager;
// ConcurrencyManager::Create(&manager, ...);
// if (fixed_mode) {
//   PerfStatus status_summary;
//   manager->Step(status_summary, concurrent_request_count);
//   Report(status_summary, ...);
// } else {
//   PerfStatus status_summary;
//   for (count = 1;;count++) {
//     manager->Step(status_summary, count);
//     Report(status_summary, ...);
//     if (status_summary.avg_latency_us >= latency_threshold)
//       break;
//   }
// }
//
// Detail:
// Concurrency Manager will maintain the number of concurrent requests by using
// corresponding number of worker threads that keep sending randomly generated
// requests to the server. The worker threads will record the start time and end
// time of each request into a shared vector.
//
// The manager can adjust the number of concurrent requests by creating
// new threads or by reducing the concurrency that existing threads create
// (see threads_concurrency_).
// After the adjustment, the manager will actively measure the throughput until
// it is stable. Once stable, the manager update the 'status_summary' based on
// the most recent measurement.
//
// The measurement procedure:
// 1. Main thread gets start status from the server and records the start time.
// 2. After given time interval, main thread gets end status from the server and
//    records the end time.
// 3. From the shared vector, Main thread uses data that are generated between
//    recorded start time and end time to measure client side status and
//    update status_summary.

class ConcurrencyManager {
 public:
  ~ConcurrencyManager()
  {
    early_exit = true;
    // wake up all threads
    wake_signal_.notify_all();

    size_t cnt = 0;
    for (auto& thread : threads_) {
      thread.join();
      if (!threads_status_[cnt]->IsOk()) {
        std::cerr << "Thread [" << cnt
                  << "] had error: " << *(threads_status_[cnt]) << std::endl;
      }
      cnt++;
    }
  }

  static nic::Error Create(
      std::unique_ptr<ConcurrencyManager>* manager, const bool verbose,
      const bool profile, const int32_t batch_size, const double stable_offset,
      const uint64_t measurement_window_ms, const size_t max_measurement_count,
      const bool streaming, const size_t max_threads,
      const std::string& model_name, const int64_t model_version,
      const std::string& url, const ProtocolType protocol,
      const size_t sequence_length, const bool zero_input)
  {
    nic::Error err = nic::Error::Success;
    manager->reset(new ConcurrencyManager(
        verbose, profile, batch_size, stable_offset, measurement_window_ms,
        max_measurement_count, streaming, max_threads, model_name,
        model_version, url, protocol, sequence_length, zero_input, err));
    return err;
  }

  // Step will adjust the number of concurrent requests to be the same as
  // 'concurrent_request_count' (by creating threads or by pausing threads)
  // and it will actively measure throughput in every 'measurement_window' msec
  // until the throughput is stable. Once the throughput is stable, it summarize
  // the most recent measurement into 'status_summary'
  // NOTE: the requests are being sent regardless of the measurement, so the
  // data returned by the server (see struct PerforamnceStatusStruct) will
  // include more requests than what the client measures (we can't get the exact
  // server status right before the first request and right after the last
  // request).
  nic::Error Step(
      PerfStatus& status_summary, const size_t concurrent_request_count)
  {
    status_summary.concurrency = concurrent_request_count;

    // Always prefer to create new threads if the maximum limit has not been met
    while ((concurrent_request_count > threads_.size()) &&
           (threads_.size() < max_threads_)) {
      // Launch new thread for inferencing
      threads_status_.emplace_back(
          new nic::Error(ni::RequestStatusCode::SUCCESS));
      threads_contexts_stat_.emplace_back(
          new std::vector<nic::InferContext::Stat>());
      threads_concurrency_.emplace_back(new size_t(0));
      threads_.emplace_back(
          &ConcurrencyManager::AsyncInfer, this, threads_status_.back(),
          threads_contexts_stat_.back(), threads_concurrency_.back());
    }

    // Compute the new concurrency level for each thread (take floor)
    // and spread the remaining value
    size_t avg_concurrency = concurrent_request_count / threads_.size();
    size_t threads_add_one = concurrent_request_count % threads_.size();
    for (size_t i = 0; i < threads_concurrency_.size(); i++) {
      *(threads_concurrency_[i]) =
          avg_concurrency + (i < threads_add_one ? 1 : 0);
    }

    // Make sure all threads will check their updated concurrency level
    wake_signal_.notify_all();

    std::cout << "Request concurrency: " << concurrent_request_count
              << std::endl;

    // Start measurement
    nic::Error err(ni::RequestStatusCode::SUCCESS);

    size_t recent_k = 3;
    std::vector<int> infer_per_sec;
    std::vector<uint64_t> latencies;
    // Stable will only be changed if max_measurement_count >= recent_k
    bool stable = true;
    double avg_ips = 0;
    uint64_t avg_latency = 0;
    do {
      // Check thread status to make sure that the actual concurrency level is
      // consistent to the one being reported
      // If some thread return early, main thread will return and
      // the worker thread's error message will be reported
      // when ConcurrencyManager's destructor get called.
      for (auto& thread_status : threads_status_) {
        if (!thread_status->IsOk()) {
          return nic::Error(
              ni::RequestStatusCode::INTERNAL,
              "Failed to maintain concurrency level requested."
              " Worker thread(s) failed to generate concurrent requests.");
        }
      }

      err = Measure(status_summary);
      if (!err.IsOk()) {
        return err;
      }
      infer_per_sec.push_back(status_summary.client_infer_per_sec);
      latencies.push_back(status_summary.client_avg_latency_ns);
      avg_ips += (double)infer_per_sec.back() / recent_k;
      avg_latency += latencies.back() / recent_k;

      if (verbose_) {
        std::cout << "  Pass [" << infer_per_sec.size()
                  << "] throughput: " << infer_per_sec.back() << " infer/sec. "
                  << "Avg latency: "
                  << (status_summary.client_avg_latency_ns / 1000)
                  << " usec (std " << status_summary.std_us << " usec)"
                  << std::endl;
      }

      if (infer_per_sec.size() >= recent_k) {
        size_t idx = infer_per_sec.size() - recent_k;
        if (infer_per_sec.size() > recent_k) {
          avg_ips -= (double)infer_per_sec[idx - 1] / recent_k;
          avg_latency -= latencies[idx - 1] / recent_k;
        }
        stable = true;
        for (; idx < infer_per_sec.size(); idx++) {
          // We call it stable only if recent_k measurement are within
          // +/-(stable_offset_)% of the average infer per second and latency
          if ((infer_per_sec[idx] < avg_ips * (1 - stable_offset_)) ||
              (infer_per_sec[idx] > avg_ips * (1 + stable_offset_))) {
            stable = false;
            break;
          }
          if ((latencies[idx] < avg_latency * (1 - stable_offset_)) ||
              (latencies[idx] > avg_latency * (1 + stable_offset_))) {
            stable = false;
            break;
          }
        }
        if (stable) {
          break;
        }
      }
    } while ((!early_exit) && (infer_per_sec.size() < max_measurement_count_));
    if (early_exit) {
      return nic::Error(
          ni::RequestStatusCode::INTERNAL, "Received exit signal.");
    } else if (!stable) {
      std::cerr << "Failed to obtain stable measurement within "
                << max_measurement_count_
                << " measurement windows for concurrency "
                << concurrent_request_count << ". Please try to "
                << "increase the time window." << std::endl;
    }

    return err;
  }

 private:
  using TimestampVector =
      std::vector<std::tuple<struct timespec, struct timespec, uint32_t>>;

  ConcurrencyManager(
      const bool verbose, const bool profile, const int32_t batch_size,
      const double stable_offset, const int32_t measurement_window_ms,
      const size_t max_measurement_count, const bool streaming,
      const size_t max_threads, const std::string& model_name,
      const int64_t model_version, const std::string& url,
      const ProtocolType protocol, const size_t sequence_length,
      const bool zero_input, nic::Error& err)
      : verbose_(verbose), profile_(profile), batch_size_(batch_size),
        stable_offset_(stable_offset),
        measurement_window_ms_(measurement_window_ms),
        max_measurement_count_(max_measurement_count), streaming_(streaming),
        max_threads_(max_threads), model_name_(model_name),
        model_version_(model_version), url_(url), protocol_(protocol),
        sequence_length_(sequence_length), zero_input_(zero_input),
        on_sequence_model_(false), current_correlation_id_(0)
  {
    request_timestamps_.reset(new TimestampVector());

    ni::ModelStatus model_status;
    err = GetModelStatus(&model_status);
    if (err.IsOk()) {
      on_sequence_model_ = model_status.config().has_sequence_batching();
    }
  }

  nic::Error StartProfile()
  {
    std::unique_ptr<nic::ProfileContext> ctx;
    nic::Error err;
    if (protocol_ == ProtocolType::HTTP) {
      err = nic::ProfileHttpContext::Create(&ctx, url_, false);
    } else {
      err = nic::ProfileGrpcContext::Create(&ctx, url_, false);
    }
    if (!err.IsOk()) {
      return err;
    }

    return ctx->StartProfile();
  }

  nic::Error StopProfile()
  {
    std::unique_ptr<nic::ProfileContext> ctx;
    nic::Error err;
    if (protocol_ == ProtocolType::HTTP) {
      err = nic::ProfileHttpContext::Create(&ctx, url_, false);
    } else {
      err = nic::ProfileGrpcContext::Create(&ctx, url_, false);
    }
    if (!err.IsOk()) {
      return err;
    }

    return ctx->StopProfile();
  }

  nic::Error GetModelStatus(ni::ModelStatus* model_status)
  {
    std::unique_ptr<nic::ServerStatusContext> ctx;
    nic::Error err;
    if (protocol_ == ProtocolType::HTTP) {
      err =
          nic::ServerStatusHttpContext::Create(&ctx, url_, model_name_, false);
    } else {
      err =
          nic::ServerStatusGrpcContext::Create(&ctx, url_, model_name_, false);
    }
    if (err.IsOk()) {
      ni::ServerStatus server_status;
      err = ctx->GetServerStatus(&server_status);
      if (err.IsOk()) {
        const auto& itr = server_status.model_status().find(model_name_);
        if (itr == server_status.model_status().end()) {
          err = nic::Error(
              ni::RequestStatusCode::INTERNAL,
              "unable to find status for model");
        } else {
          model_status->CopyFrom(itr->second);
        }
      }
    }

    return err;
  }

  nic::Error PrepareInfer(
      std::unique_ptr<nic::InferContext>* ctx,
      std::unique_ptr<nic::InferContext::Options>* options,
      std::vector<uint8_t>& input_buffer)
  {
    nic::Error err;
    // Create the context for inference of the specified model,
    // make sure to use an unused correlation id on sequence model.
    ni::CorrelationID correlation_id = 0;
    {
      std::lock_guard<std::mutex> lock(correlation_id_mutex_);
      if (on_sequence_model_) {
        current_correlation_id_++;
        correlation_id = current_correlation_id_;
      }
    }
    if (streaming_) {
      err = nic::InferGrpcStreamContext::Create(
          ctx, correlation_id, url_, model_name_, model_version_, false);
    } else if (protocol_ == ProtocolType::HTTP) {
      err = nic::InferHttpContext::Create(
          ctx, correlation_id, url_, model_name_, model_version_, false);
    } else {
      err = nic::InferGrpcContext::Create(
          ctx, correlation_id, url_, model_name_, model_version_, false);
    }
    if (!err.IsOk()) {
      return err;
    }

    uint64_t max_batch_size = (*ctx)->MaxBatchSize();

    // Model specifying maximum batch size of 0 indicates that batching
    // is not supported and so the input tensors do not expect a "N"
    // dimension (and 'batch_size' should be 1 so that only a single
    // image instance is inferred at a time).
    if (max_batch_size == 0) {
      if (batch_size_ != 1) {
        return nic::Error(
            ni::RequestStatusCode::INVALID_ARG,
            "expecting batch size 1 for model '" + (*ctx)->ModelName() +
                "' which does not support batching");
      }
    } else if (batch_size_ > max_batch_size) {
      return nic::Error(
          ni::RequestStatusCode::INVALID_ARG,
          "expecting batch size <= " + std::to_string(max_batch_size) +
              " for model '" + (*ctx)->ModelName() + "'");
    }

    // Prepare context for 'batch_size' batches. Request that all
    // outputs be returned.
    // Only set options if it has not been created, otherwise,
    // assuming that the options for this model has been created previously
    if (*options == nullptr) {
      err = nic::InferContext::Options::Create(options);
      if (!err.IsOk()) {
        return err;
      }

      (*options)->SetBatchSize(batch_size_);
      for (const auto& output : (*ctx)->Outputs()) {
        (*options)->AddRawResult(output);
      }
    }

    err = (*ctx)->SetRunOptions(*(*options));
    if (!err.IsOk()) {
      return err;
    }

    // Create a zero or randomly (as indicated by zero_input_)
    // initialized buffer that is large enough to provide the largest
    // needed input. We (re)use this buffer for all input values.
    size_t max_input_byte_size = 0;
    for (const auto& input : (*ctx)->Inputs()) {
      const int64_t bs = input->ByteSize();
      if (bs < 0) {
        return nic::Error(
            ni::RequestStatusCode::INVALID_ARG,
            "input '" + input->Name() +
                "' has variable-size shape, unable to create input values for "
                "model '" +
                (*ctx)->ModelName() + "'");
      }

      max_input_byte_size =
          std::max(max_input_byte_size, (size_t)input->ByteSize());
    }

    if (input_buffer.size() == 0) {
      std::vector<uint8_t> input_buf(max_input_byte_size);
      for (size_t i = 0; i < input_buf.size(); ++i) {
        input_buf[i] = (zero_input_) ? 0 : rand();
      }
      input_buffer.swap(input_buf);
    }

    // Initialize inputs to use random values...
    for (const auto& input : (*ctx)->Inputs()) {
      err = input->Reset();
      if (!err.IsOk()) {
        return err;
      }

      for (size_t i = 0; i < batch_size_; ++i) {
        err = input->SetRaw(&input_buffer[0], (size_t)input->ByteSize());
        if (!err.IsOk()) {
          return err;
        }
      }
    }

    return err;
  }

  nic::Error GetAccumulatedContextStat(nic::InferContext::Stat* contexts_stat)
  {
    std::lock_guard<std::mutex> lk(status_report_mutex_);
    for (auto& thread_contexts_stat : threads_contexts_stat_) {
      for (auto& context_stat : (*thread_contexts_stat)) {
        contexts_stat->completed_request_count +=
            context_stat.completed_request_count;
        contexts_stat->cumulative_total_request_time_ns +=
            context_stat.cumulative_total_request_time_ns;
        contexts_stat->cumulative_send_time_ns +=
            context_stat.cumulative_send_time_ns;
        contexts_stat->cumulative_receive_time_ns +=
            context_stat.cumulative_receive_time_ns;
      }
    }
    return nic::Error::Success;
  }

  nic::Error Summarize(
      PerfStatus& summary, const ni::ModelStatus& start_status,
      const ni::ModelStatus& end_status,
      const nic::InferContext::Stat& start_stat,
      const nic::InferContext::Stat& end_stat)
  {
    nic::Error err(ni::RequestStatusCode::SUCCESS);

    //===============
    // Summarizing statistic measured by client

    // Get the requests in the shared vector
    TimestampVector current_timestamps;
    status_report_mutex_.lock();
    request_timestamps_->swap(current_timestamps);
    status_report_mutex_.unlock();

    // finding the start time of the first request
    // and the end time of the last request in the timestamp queue
    uint64_t first_request_start_ns = 0;
    uint64_t last_request_end_ns = 0;
    for (auto& timestamp : current_timestamps) {
      uint64_t request_start_time =
          std::get<0>(timestamp).tv_sec * ni::NANOS_PER_SECOND +
          std::get<0>(timestamp).tv_nsec;
      uint64_t request_end_time =
          std::get<1>(timestamp).tv_sec * ni::NANOS_PER_SECOND +
          std::get<1>(timestamp).tv_nsec;
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

    uint64_t client_start_ns = first_request_start_ns + offset;
    uint64_t client_end_ns = client_start_ns + measurement_window_ns;
    uint64_t client_duration_ns = client_end_ns - client_start_ns;

    // Get measurement from requests that fall within the time interval
    size_t valid_timestamp_count = 0;
    size_t valid_sequence_count = 0;
    uint64_t min_latency_ns = 0;
    uint64_t max_latency_ns = 0;
    uint64_t tol_latency_ns = 0;
    uint64_t tol_square_latency_us = 0;
    for (auto& timestamp : current_timestamps) {
      uint64_t request_start_ns =
          std::get<0>(timestamp).tv_sec * ni::NANOS_PER_SECOND +
          std::get<0>(timestamp).tv_nsec;
      uint64_t request_end_ns =
          std::get<1>(timestamp).tv_sec * ni::NANOS_PER_SECOND +
          std::get<1>(timestamp).tv_nsec;

      if (request_start_ns <= request_end_ns) {
        // Only counting requests that end within the time interval
        if ((request_end_ns >= client_start_ns) &&
            (request_end_ns <= client_end_ns)) {
          uint64_t request_latency = request_end_ns - request_start_ns;
          if ((request_latency < min_latency_ns) || (min_latency_ns == 0))
            min_latency_ns = request_latency;
          if ((request_latency > max_latency_ns) || (max_latency_ns == 0))
            max_latency_ns = request_latency;
          tol_latency_ns += request_latency;
          tol_square_latency_us +=
              (request_latency * request_latency) / (1000 * 1000);
          valid_timestamp_count++;
          if (std::get<2>(timestamp) &
              ni::InferRequestHeader::FLAG_SEQUENCE_END)
            valid_sequence_count++;
        }
      }
    }

    if (valid_timestamp_count == 0) {
      return nic::Error(
          ni::RequestStatusCode::INTERNAL,
          "No valid requests recorded within time interval."
          " Please use a larger time window.");
    }

    summary.on_sequence_model = on_sequence_model_;
    summary.batch_size = batch_size_;
    summary.client_request_count = valid_timestamp_count;
    summary.client_sequence_count = valid_sequence_count;
    summary.client_duration_ns = client_duration_ns;
    float client_duration_sec =
        (float)summary.client_duration_ns / ni::NANOS_PER_SECOND;
    summary.client_sequence_per_sec =
        (int)(valid_sequence_count / client_duration_sec);
    summary.client_infer_per_sec =
        (int)(valid_timestamp_count * summary.batch_size / client_duration_sec);
    summary.client_min_latency_ns = min_latency_ns;
    summary.client_max_latency_ns = max_latency_ns;
    summary.client_avg_latency_ns = tol_latency_ns / valid_timestamp_count;

    // calculate standard deviation
    uint64_t expected_square_latency_us =
        tol_square_latency_us / valid_timestamp_count;
    uint64_t square_avg_latency_us =
        (summary.client_avg_latency_ns * summary.client_avg_latency_ns) /
        (1000 * 1000);
    uint64_t var_us = (expected_square_latency_us > square_avg_latency_us)
                          ? (expected_square_latency_us - square_avg_latency_us)
                          : 0;
    summary.std_us = (uint64_t)(sqrt(var_us));

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

    //===============
    // Summarizing statistic measured by client

    // If model_version is -1 then look in the end status to find the
    // latest (highest valued version) and use that as the version.
    int64_t status_model_version = 0;
    if (model_version_ < 0) {
      for (const auto& vp : end_status.version_status()) {
        status_model_version = std::max(status_model_version, vp.first);
      }
    } else {
      status_model_version = model_version_;
    }

    const auto& vend_itr =
        end_status.version_status().find(status_model_version);
    if (vend_itr == end_status.version_status().end()) {
      err = nic::Error(
          ni::RequestStatusCode::INTERNAL, "missing model version status");
    } else {
      const auto& end_itr = vend_itr->second.infer_stats().find(batch_size_);
      if (end_itr == vend_itr->second.infer_stats().end()) {
        err = nic::Error(
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
              vstart_itr->second.infer_stats().find(batch_size_);
          if (start_itr != vstart_itr->second.infer_stats().end()) {
            start_cnt = start_itr->second.success().count();
            start_cumm_time_ns = start_itr->second.success().total_time_ns();
            start_queue_time_ns = start_itr->second.queue().total_time_ns();
            start_compute_time_ns = start_itr->second.compute().total_time_ns();
          }
        }

        summary.server_request_count =
            end_itr->second.success().count() - start_cnt;
        summary.server_cumm_time_ns =
            end_itr->second.success().total_time_ns() - start_cumm_time_ns;
        summary.server_queue_time_ns =
            end_itr->second.queue().total_time_ns() - start_queue_time_ns;
        summary.server_compute_time_ns =
            end_itr->second.compute().total_time_ns() - start_compute_time_ns;
      }
    }
    return err;
  }

  size_t GetRandomSequenceLength()
  {
    int random_offset =
        ((2.0 * rand() / double(RAND_MAX)) - 1.0) * 0.2 * sequence_length_;
    size_t res = sequence_length_ + random_offset;
    if (int(sequence_length_) + random_offset <= 0) {
      res = 1;
    }
    return res;
  }

  // Function for worker threads
  void AsyncInfer(
      std::shared_ptr<nic::Error> err,
      std::shared_ptr<std::vector<nic::InferContext::Stat>> stats,
      std::shared_ptr<size_t> concurrency)
  {
    std::vector<uint8_t> input_buf;
    std::unique_ptr<nic::InferContext::Options> options(nullptr);

    std::vector<std::unique_ptr<nic::InferContext>> ctxs;
    std::vector<bool> ctxs_working;
    std::vector<std::map<uint64_t, std::pair<struct timespec, uint32_t>>>
        requests_start_time;

    // run inferencing until receiving exit signal to maintain server load.
    do {
      // Create the context for inference of the specified model.
      size_t num_reqs = *concurrency;
      while (num_reqs > ctxs.size()) {
        ctxs.emplace_back();
        ctxs_working.push_back(false);
        stats->emplace_back();
        requests_start_time.emplace_back();
        *err = PrepareInfer(&(ctxs.back()), &options, input_buf);
        if (!err->IsOk()) {
          return;
        }
      }
      // Run inference to get output
      std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;
      std::shared_ptr<nic::InferContext::Request> request;

      // Create async requests such that the number of ongoing requests
      // matches the concurrency level (here is 'num_reqs')
      size_t seq_length = on_sequence_model_ ? GetRandomSequenceLength() : 1;
      for (size_t idx = 0; idx < num_reqs; idx++) {
        if (!ctxs_working[idx]) {
          // Create requests
          for (size_t i = 0; i < seq_length; i++) {
            uint32_t flags = 0;
            if (on_sequence_model_) {
              if (i == 0) {
                flags |= ni::InferRequestHeader::FLAG_SEQUENCE_START;
              }
              if (i == seq_length - 1) {
                flags |= ni::InferRequestHeader::FLAG_SEQUENCE_END;
              }
              options->SetFlag(
                  ni::InferRequestHeader::FLAG_SEQUENCE_START,
                  flags & ni::InferRequestHeader::FLAG_SEQUENCE_START);
              options->SetFlag(
                  ni::InferRequestHeader::FLAG_SEQUENCE_END,
                  flags & ni::InferRequestHeader::FLAG_SEQUENCE_END);
              ctxs[idx]->SetRunOptions(*options);
            }
            struct timespec start_time;
            clock_gettime(CLOCK_MONOTONIC, &start_time);
            *err = ctxs[idx]->AsyncRun(&request);
            if (!err->IsOk()) {
              return;
            }
            requests_start_time[idx].emplace(
                request->Id(), std::make_pair(start_time, flags));
          }
          ctxs_working[idx] = true;
        }
      }

      // Get any request that is completed and
      // record the end time of the request
      // [TODO] separate the send / recv to different threads
      bool keep_loop = true;
      while (keep_loop) {
        keep_loop = false;
        for (size_t idx = 0; idx < ctxs.size(); idx++) {
          if (ctxs_working[idx]) {
            bool is_ready;
            *err = ctxs[idx]->GetReadyAsyncRequest(&request, &is_ready, false);

            if (!err->IsOk()) {
              return;
            }

            if (!is_ready) {
              continue;
            }
            // keep the loop until no more ready requests in all contexts
            keep_loop = true;
            *err = ctxs[idx]->GetAsyncRunResults(
                &results, &is_ready, request, true);

            struct timespec end_time;
            clock_gettime(CLOCK_MONOTONIC, &end_time);

            if (!err->IsOk()) {
              return;
            }

            auto itr = requests_start_time[idx].find(request->Id());
            struct timespec start_time = itr->second.first;
            uint32_t flags = itr->second.second;
            requests_start_time[idx].erase(itr);

            if (!on_sequence_model_ ||
                (flags & ni::InferRequestHeader::FLAG_SEQUENCE_END)) {
              ctxs_working[idx] = false;
            }

            // Add the request timestamp to shared vector with proper locking
            status_report_mutex_.lock();
            // Critical section
            request_timestamps_->emplace_back(
                std::make_tuple(start_time, end_time, flags));
            // Update its InferContext statistic to shared Stat pointer
            ctxs[idx]->GetStat(&((*stats)[idx]));
            status_report_mutex_.unlock();
          }
        }
      }

      // Wait if no request should be sent and it is not exiting
      {
        std::unique_lock<std::mutex> lock(wake_mutex_);
        wake_signal_.wait(
            lock, [concurrency]() { return early_exit || (*concurrency > 0); });
      }

      // Stop inferencing if an early exit has been signaled.
    } while (!early_exit);
  }

  // Used for measurement
  nic::Error Measure(PerfStatus& status_summary)
  {
    nic::Error err(ni::RequestStatusCode::SUCCESS);

    ni::ModelStatus start_status;
    ni::ModelStatus end_status;
    nic::InferContext::Stat start_stat;
    nic::InferContext::Stat end_stat;

    err = GetModelStatus(&start_status);
    if (!err.IsOk()) {
      return err;
    }
    // Start profiling on the server if requested.
    if (profile_) {
      err = StartProfile();
      if (!err.IsOk()) {
        return err;
      }
    }

    err = GetAccumulatedContextStat(&start_stat);

    // Wait for specified time interval in msec
    std::this_thread::sleep_for(
        std::chrono::milliseconds((uint64_t)(measurement_window_ms_ * 1.2)));

    err = GetAccumulatedContextStat(&end_stat);

    // Stop profiling on the server if requested.
    if (profile_) {
      err = StopProfile();
      if (!err.IsOk()) {
        return err;
      }
    }

    // Get server status and then print report on difference between
    // before and after status.
    err = GetModelStatus(&end_status);
    if (!err.IsOk()) {
      return err;
    }

    err = Summarize(
        status_summary, start_status, end_status, start_stat, end_stat);
    if (!err.IsOk()) {
      return err;
    }

    return nic::Error(ni::RequestStatusCode::SUCCESS);
  }

  bool verbose_;
  bool profile_;
  size_t batch_size_;
  double stable_offset_;
  uint64_t measurement_window_ms_;
  size_t max_measurement_count_;
  bool streaming_;
  size_t max_threads_;
  std::string model_name_;
  int64_t model_version_;
  std::string url_;
  ProtocolType protocol_;
  size_t sequence_length_;
  bool zero_input_;

  bool on_sequence_model_;
  ni::CorrelationID current_correlation_id_;
  std::mutex correlation_id_mutex_;

  // Note: early_exit signal is kept global
  std::vector<std::thread> threads_;
  std::vector<std::shared_ptr<nic::Error>> threads_status_;
  std::vector<std::shared_ptr<std::vector<nic::InferContext::Stat>>>
      threads_contexts_stat_;
  std::vector<std::shared_ptr<size_t>> threads_concurrency_;

  // Use condition variable to pause/continue worker threads
  std::condition_variable wake_signal_;
  std::mutex wake_mutex_;

  // Pointer to a vector of request timestamps <start_time, end_time>
  // Request latency will be end_time - start_time
  std::shared_ptr<TimestampVector> request_timestamps_;
  // Mutex to avoid race condition on adding elements into the timestamp vector
  // and on updating context statistic.
  std::mutex status_report_mutex_;
};

ProtocolType
ParseProtocol(const std::string& str)
{
  std::string protocol(str);
  std::transform(protocol.begin(), protocol.end(), protocol.begin(), ::tolower);
  if (protocol == "http") {
    return ProtocolType::HTTP;
  } else if (protocol == "grpc") {
    return ProtocolType::GRPC;
  }

  std::cerr << "unexpected protocol type \"" << str
            << "\", expecting HTTP or gRPC" << std::endl;
  exit(1);

  return ProtocolType::HTTP;
}

nic::Error
Report(
    const PerfStatus& summary, const size_t concurrent_request_count,
    const ProtocolType protocol, const bool verbose)
{
  const uint64_t cnt = summary.server_request_count;

  const uint64_t cumm_time_us = summary.server_cumm_time_ns / 1000;
  const uint64_t cumm_avg_us = cumm_time_us / cnt;

  const uint64_t queue_time_us = summary.server_queue_time_ns / 1000;
  const uint64_t queue_avg_us = queue_time_us / cnt;

  const uint64_t compute_time_us = summary.server_compute_time_ns / 1000;
  const uint64_t compute_avg_us = compute_time_us / cnt;

  const uint64_t overhead = (cumm_avg_us > queue_avg_us + compute_avg_us)
                                ? (cumm_avg_us - queue_avg_us - compute_avg_us)
                                : 0;

  const uint64_t avg_latency_us = summary.client_avg_latency_ns / 1000;
  const uint64_t std_us = summary.std_us;

  const uint64_t avg_request_time_us =
      summary.client_avg_request_time_ns / 1000;
  const uint64_t avg_send_time_us = summary.client_avg_send_time_ns / 1000;
  const uint64_t avg_receive_time_us =
      summary.client_avg_receive_time_ns / 1000;
  const uint64_t avg_response_wait_time_us =
      avg_request_time_us - avg_send_time_us - avg_receive_time_us;

  std::string client_library_detail = "    ";
  if (protocol == ProtocolType::GRPC) {
    client_library_detail +=
        "Avg gRPC time: " +
        std::to_string(
            avg_send_time_us + avg_receive_time_us + avg_request_time_us) +
        " usec (";
    if (!verbose) {
      client_library_detail +=
          "(un)marshal request/response " +
          std::to_string(avg_send_time_us + avg_receive_time_us) +
          " usec + response wait " + std::to_string(avg_request_time_us) +
          " usec)";
    } else {
      client_library_detail +=
          "marshal " + std::to_string(avg_send_time_us) +
          " usec + response wait " + std::to_string(avg_request_time_us) +
          " usec + unmarshal " + std::to_string(avg_receive_time_us) + " usec)";
    }
  } else {
    client_library_detail +=
        "Avg HTTP time: " + std::to_string(avg_request_time_us) + " usec (";
    if (!verbose) {
      client_library_detail +=
          "send/recv " +
          std::to_string(avg_send_time_us + avg_receive_time_us) +
          " usec + response wait " + std::to_string(avg_response_wait_time_us) +
          " usec)";
    } else {
      client_library_detail +=
          "send " + std::to_string(avg_send_time_us) +
          " usec + response wait " + std::to_string(avg_response_wait_time_us) +
          " usec + receive " + std::to_string(avg_receive_time_us) + " usec)";
    }
  }

  std::cout << "  Client: " << std::endl
            << "    Request count: " << summary.client_request_count
            << std::endl;
  if (summary.on_sequence_model) {
    std::cout << "    Sequence count: " << summary.client_sequence_count << " ("
              << summary.client_sequence_per_sec << " seq/sec)" << std::endl;
  }
  std::cout << "    Throughput: " << summary.client_infer_per_sec
            << " infer/sec" << std::endl
            << "    Avg latency: " << avg_latency_us << " usec"
            << " (standard deviation " << std_us << " usec)" << std::endl
            << client_library_detail << std::endl
            << "  Server: " << std::endl
            << "    Request count: " << cnt << std::endl
            << "    Avg request latency: " << cumm_avg_us << " usec"
            << " (overhead " << overhead << " usec + "
            << "queue " << queue_avg_us << " usec + "
            << "compute " << compute_avg_us << " usec)" << std::endl
            << std::endl;

  return nic::Error(ni::RequestStatusCode::SUCCESS);
}

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-f <filename for storing report in csv format>" << std::endl;
  std::cerr << "\t-b <batch size>" << std::endl;
  std::cerr << "\t-t <number of concurrent requests>" << std::endl;
  std::cerr << "\t-d" << std::endl;
  std::cerr << "\t-a" << std::endl;
  std::cerr << "\t-z" << std::endl;
  std::cerr << "\t--streaming" << std::endl;
  std::cerr << "\t--max-threads <thread counts>" << std::endl;
  std::cerr << "\t-l <latency threshold (in msec)>" << std::endl;
  std::cerr << "\t-c <maximum concurrency>" << std::endl;
  std::cerr << "\t-s <deviation threshold for stable measurement"
            << " (in percentage)>" << std::endl;
  std::cerr << "\t-p <measurement window (in msec)>" << std::endl;
  std::cerr << "\t-r <maximum number of measurements for each profiling>"
            << std::endl;
  std::cerr << "\t-n" << std::endl;
  std::cerr << "\t-m <model name>" << std::endl;
  std::cerr << "\t-x <model version>" << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-i <Protocol used to communicate with inference service>"
            << std::endl;
  std::cerr << "\t--sequence-length <length>" << std::endl;
  std::cerr << std::endl;
  std::cerr
      << "The -d flag enables dynamic concurrent request count where the number"
      << " of concurrent requests will increase linearly until the request"
      << " latency is above the threshold set (see -l)." << std::endl;
  std::cerr << "The -a flag is deprecated. Enable it will not change"
            << "perf client behaviors." << std::endl;
  std::cerr << "The --streaming flag is only valid with gRPC protocol."
            << std::endl;
  std::cerr << "The --max-threads flag sets the maximum number of threads that"
            << " will be created for providing desired concurrency."
            << " Default is 16." << std::endl;
  std::cerr
      << "For -t, it indicates the number of starting concurrent requests if -d"
      << " flag is set." << std::endl;
  std::cerr
      << "For -s, it indicates the deviation threshold for the measurements. "
         "The measurement is considered as stable if the recent 3 measurements "
         "are within +/- (deviation threshold)% of their average in terms of "
         "both infer per second and latency. Default is 10(%)"
      << std::endl;
  std::cerr
      << "For -c, it indicates the maximum number of concurrent requests "
         "allowed if -d flag is set. Once the number of concurrent requests "
         "exceeds the maximum, the perf client will stop and exit regardless "
         "of the latency threshold. Default is 0 to indicate that no limit is "
         "set on the number of concurrent requests."
      << std::endl;
  std::cerr
      << "For -p, it indicates the time interval used for each measurement."
      << " The perf client will sample a time interval specified by -p and"
      << " take measurement over the requests completed"
      << " within that time interval." << std::endl;
  std::cerr << "For -r, it indicates the maximum number of measurements for "
               "each profiling setting. The perf client will take multiple "
               "measurements and report the measurement until it is stable. "
               "The perf client will abort if the measurement is still "
               "unstable after the maximum number of measuremnts."
            << std::endl;
  std::cerr << "For -l, it has no effect unless -d flag is set." << std::endl;
  std::cerr << "The -n flag enables profiling for the duration of the run"
            << std::endl;
  std::cerr
      << "If -x is not specified the most recent version (that is, the highest "
      << "numbered version) of the model will be used." << std::endl;
  std::cerr << "For -i, available protocols are gRPC and HTTP. Default is HTTP."
            << std::endl;
  std::cerr << "The -z flag causes input tensors to be initialized with zeros "
               "instead of random data"
            << std::endl;
  std::cerr
      << "For --sequence-length, it indicates the base length of a sequence"
      << " used for sequence models. A sequence with length x will be composed"
      << " of x requests to be sent as the elements in the sequence. The length"
      << " of the actual sequence will be within +/- 20% of the base length."
      << std::endl;

  exit(1);
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  bool profile = false;
  bool dynamic_concurrency_mode = false;
  bool streaming = false;
  bool zero_input = false;
  size_t max_threads = 16;
  // average length of a sentence
  size_t sequence_length = 20;
  uint64_t latency_threshold_ms = 0;
  int32_t batch_size = 1;
  int32_t concurrent_request_count = 1;
  size_t max_concurrency = 0;
  double stable_offset = 0.1;
  uint64_t measurement_window_ms = 0;
  size_t max_measurement_count = 10;
  std::string model_name;
  int64_t model_version = -1;
  std::string url("localhost:8000");
  std::string filename("");
  ProtocolType protocol = ProtocolType::HTTP;

  // {name, has_arg, *flag, val}
  static struct option long_options[] = {{"streaming", 0, 0, 0},
                                         {"max-threads", 1, 0, 1},
                                         {"sequence-length", 1, 0, 2},
                                         {0, 0, 0, 0}};

  // Parse commandline...
  int opt;
  while ((opt = getopt_long(
              argc, argv, "vndazc:u:m:x:b:t:p:i:l:r:s:f:", long_options,
              NULL)) != -1) {
    switch (opt) {
      case 0:
        streaming = true;
        break;
      case 1:
        max_threads = std::atoi(optarg);
        break;
      case 2:
        sequence_length = std::atoi(optarg);
        break;
      case 'v':
        verbose = true;
        break;
      case 'n':
        profile = true;
        break;
      case 'z':
        zero_input = true;
        break;
      case 'd':
        dynamic_concurrency_mode = true;
        break;
      case 'u':
        url = optarg;
        break;
      case 'm':
        model_name = optarg;
        break;
      case 'x':
        model_version = std::atoll(optarg);
        break;
      case 'b':
        batch_size = std::atoi(optarg);
        break;
      case 't':
        concurrent_request_count = std::atoi(optarg);
        break;
      case 'p':
        measurement_window_ms = std::atoi(optarg);
        break;
      case 'i':
        protocol = ParseProtocol(optarg);
        break;
      case 'l':
        latency_threshold_ms = std::atoi(optarg);
        break;
      case 'c':
        max_concurrency = std::atoi(optarg);
        break;
      case 'r':
        max_measurement_count = std::atoi(optarg);
        break;
      case 's':
        stable_offset = atof(optarg) / 100;
        break;
      case 'f':
        filename = optarg;
        break;
      case 'a':
        std::cerr << "WARNING: -a flag is deprecated. Enable it will not change"
                  << "perf client behaviors." << std::endl;
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  if (model_name.empty()) {
    Usage(argv, "-m flag must be specified");
  }
  if (batch_size <= 0) {
    Usage(argv, "batch size must be > 0");
  }
  if (measurement_window_ms <= 0) {
    Usage(argv, "measurement window must be > 0 in msec");
  }
  if (concurrent_request_count <= 0) {
    Usage(argv, "concurrent request count must be > 0");
  }
  if (dynamic_concurrency_mode && latency_threshold_ms < 0) {
    Usage(argv, "latency threshold must be >= 0 for dynamic concurrency mode");
  }
  if (streaming && protocol != ProtocolType::GRPC) {
    Usage(argv, "Streaming is only allowed with gRPC protocol");
  }
  if (max_threads == 0) {
    Usage(argv, "maximum number of threads must be > 0");
  }
  if (sequence_length == 0) {
    sequence_length = 20;
    std::cerr << "WARNING: using an invalid sequence length. Perf client will"
              << " use default value if it is measuring on sequence model."
              << std::endl;
  }

  // trap SIGINT to allow threads to exit gracefully
  signal(SIGINT, SignalHandler);

  nic::Error err(ni::RequestStatusCode::SUCCESS);
  std::unique_ptr<ConcurrencyManager> manager;
  err = ConcurrencyManager::Create(
      &manager, verbose, profile, batch_size, stable_offset,
      measurement_window_ms, max_measurement_count, streaming, max_threads,
      model_name, model_version, url, protocol, sequence_length, zero_input);
  if (!err.IsOk()) {
    std::cerr << err << std::endl;
    return 1;
  }

  // pre-run report
  std::cout << "*** Measurement Settings ***" << std::endl
            << "  Batch size: " << batch_size << std::endl
            << "  Measurement window: " << measurement_window_ms << " msec"
            << std::endl;
  if (dynamic_concurrency_mode) {
    std::cout << "  Latency limit: " << latency_threshold_ms << " msec"
              << std::endl;
    if (max_concurrency != 0) {
      std::cout << "  Concurrency limit: " << max_concurrency
                << " concurrent requests" << std::endl;
    }
  }
  std::cout << std::endl;

  PerfStatus status_summary;
  std::vector<PerfStatus> summary;
  if (!dynamic_concurrency_mode) {
    err = manager->Step(status_summary, concurrent_request_count);
    if (err.IsOk()) {
      err = Report(status_summary, concurrent_request_count, protocol, verbose);
    }
  } else {
    for (size_t count = concurrent_request_count;
         (count <= max_concurrency) || (max_concurrency == 0); count++) {
      err = manager->Step(status_summary, count);
      if (err.IsOk()) {
        err = Report(status_summary, count, protocol, verbose);
        summary.push_back(status_summary);
        uint64_t avg_latency_ms =
            status_summary.client_avg_latency_ns / (1000 * 1000);
        if ((avg_latency_ms >= latency_threshold_ms) || !err.IsOk()) {
          std::cerr << err << std::endl;
          break;
        }
      } else {
        break;
      }
    }
  }
  if (!err.IsOk()) {
    std::cerr << err << std::endl;
    return 1;
  }
  if (summary.size()) {
    std::ofstream ofs(filename, std::ofstream::out);
    // Can print more depending on verbose, but it seems too much information
    std::cout << "Inferences/Second vs. Client Average Batch Latency"
              << std::endl;
    if (!filename.empty()) {
      ofs << "Concurrency,Inferences/Second,Client Send,"
          << "Network+Server Send/Recv,Server Queue,"
          << "Server Compute,Client Recv" << std::endl;
    }

    for (PerfStatus& status : summary) {
      std::cout << "Concurrency: " << status.concurrency << ", "
                << status.client_infer_per_sec << " infer/sec, latency "
                << (status.client_avg_latency_ns / 1000) << " usec"
                << std::endl;
    }

    if (!filename.empty()) {
      // Sort summary results in order of increasing infer/sec.
      std::sort(
          summary.begin(), summary.end(),
          [](const PerfStatus& a, const PerfStatus& b) -> bool {
            return a.client_infer_per_sec < b.client_infer_per_sec;
          });

      for (PerfStatus& status : summary) {
        uint64_t avg_queue_ns =
            status.server_queue_time_ns / status.server_request_count;
        uint64_t avg_compute_ns =
            status.server_compute_time_ns / status.server_request_count;
        uint64_t avg_network_misc_ns =
            status.client_avg_latency_ns - avg_queue_ns - avg_compute_ns -
            status.client_avg_send_time_ns - status.client_avg_receive_time_ns;

        ofs << status.concurrency << "," << status.client_infer_per_sec << ","
            << (status.client_avg_send_time_ns / 1000) << ","
            << (avg_network_misc_ns / 1000) << "," << (avg_queue_ns / 1000)
            << "," << (avg_compute_ns / 1000) << ","
            << (status.client_avg_receive_time_ns / 1000) << std::endl;
      }
    }
    ofs.close();
  }
  return 0;
}
