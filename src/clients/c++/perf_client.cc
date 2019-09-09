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

#include "src/clients/c++/concurrency_manager.h"
#include "src/clients/c++/context_factory.h"
#include "src/clients/c++/inference_profiler.h"
#include "src/clients/c++/load_manager.h"
#include "src/clients/c++/perf_utils.h"


namespace perfclient {

volatile bool early_exit = false;

void
SignalHandler(int signum)
{
  std::cout << "Interrupt signal (" << signum << ") received." << std::endl;
  // Upon invoking the SignalHandler for the first time early_exit flag is
  // invoked and client waits for in-flight inferences to complete before
  // exiting. On the second invocation, the program exits immediately.
  if (!early_exit) {
    std::cout << "Waiting for in-flight inferences to complete." << std::endl;
    early_exit = true;
  } else {
    std::cout << "Exiting immediately..." << std::endl;
    exit(0);
  }
}

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
//     when the response for the request is received. If 'percentile' flag is
//     specified, the selected percentile value will be reported instead of
//     average value.
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

nic::Error
ReportServerSideStats(const ServerSideStats& stats, const int iteration)
{
  const std::string ident = std::string(2 * iteration, ' ');
  const uint64_t cnt = stats.request_count;
  if (cnt == 0) {
    std::cout << ident << "  Request count: " << cnt << std::endl;
    return nic::Error(ni::RequestStatusCode::SUCCESS);
  }

  const uint64_t cumm_time_us = stats.cumm_time_ns / 1000;
  const uint64_t cumm_avg_us = cumm_time_us / cnt;

  const uint64_t queue_time_us = stats.queue_time_ns / 1000;
  const uint64_t queue_avg_us = queue_time_us / cnt;

  const uint64_t compute_time_us = stats.compute_time_ns / 1000;
  const uint64_t compute_avg_us = compute_time_us / cnt;

  const uint64_t overhead = (cumm_avg_us > queue_avg_us + compute_avg_us)
                                ? (cumm_avg_us - queue_avg_us - compute_avg_us)
                                : 0;
  std::cout << ident << "  Request count: " << cnt << std::endl
            << ident << "  Avg request latency: " << cumm_avg_us << " usec";
  if (stats.composing_models_stat.empty()) {
    std::cout << " (overhead " << overhead << " usec + "
              << "queue " << queue_avg_us << " usec + "
              << "compute " << compute_avg_us << " usec)" << std::endl
              << std::endl;
  } else {
    std::cout << std::endl;
    std::cout << ident << "  Total avg compute time : " << compute_avg_us
              << " usec" << std::endl;
    std::cout << ident << "  Total avg queue time : " << queue_avg_us << " usec"
              << std::endl
              << std::endl;

    std::cout << ident << "Composing models: " << std::endl;
    for (const auto& model_stats : stats.composing_models_stat) {
      const auto& model_info = model_stats.first;
      std::cout << ident << model_info.first
                << ", version: " << model_info.second << std::endl;
      ReportServerSideStats(model_stats.second, iteration + 1);
    }
  }

  return nic::Error(ni::RequestStatusCode::SUCCESS);
}

nic::Error
Report(
    const PerfStatus& summary, const size_t concurrent_request_count,
    const int64_t percentile, const ProtocolType protocol, const bool verbose)
{
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
            << " infer/sec" << std::endl;
  if (percentile == -1) {
    std::cout << "    Avg latency: " << avg_latency_us << " usec"
              << " (standard deviation " << std_us << " usec)" << std::endl;
  }
  for (const auto& percentile : summary.client_percentile_latency_ns) {
    std::cout << "    p" << percentile.first
              << " latency: " << (percentile.second / 1000) << " usec"
              << std::endl;
  }
  std::cout << client_library_detail << std::endl;

  std::cout << "  Server: " << std::endl;
  ReportServerSideStats(summary.server_stats, 1);

  return nic::Error(ni::RequestStatusCode::SUCCESS);
}
}  // namespace perfclient

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
  std::cerr << "\t-H <HTTP header>" << std::endl;
  std::cerr << "\t--sequence-length <length>" << std::endl;
  std::cerr << "\t--percentile <percentile>" << std::endl;
  std::cerr << "\t--shape <name:shape>" << std::endl;
  std::cerr << "\t--data-directory <path>" << std::endl;
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
  std::cerr
      << "If -x is not specified the most recent version (that is, the highest "
      << "numbered version) of the model will be used." << std::endl;
  std::cerr << "For -i, available protocols are gRPC and HTTP. Default is HTTP."
            << std::endl;
  std::cerr
      << "For -H, the header will be added to HTTP requests (ignored for GRPC "
         "requests). The header must be specified as 'Header:Value'. -H may be "
         "specified multiple times to add multiple headers."
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
  std::cerr
      << "For --percentile, it indicates that the specified percentile in terms"
      << " of latency will also be reported and used to detemine if the"
      << " measurement is stable instead of average latency."
      << " Default is -1 to indicate no percentile will be used." << std::endl;
  std::cerr << "For --shape, the shape used for the specified input. The"
               " argument must be specified as 'name:shape' where the shape is"
               " a comma-separated list for dimension sizes, for example"
               " '--shape input_name:1,2,3'. --shape may be specified multiple"
               " times to specify shapes for different inputs."
            << std::endl;
  std::cerr
      << "For --data-directory, it indicates that the perf client will use user"
      << " provided data instead of synthetic data for model inputs. There must"
      << " be a binary file for each input required by the model."
      << " The file must be named the same as the input and must contain data"
      << " required for sending the input in a batch-1 request. The perf client"
      << " will reuse the data to match the specified batch size."
      << " Note that the files should contain only the raw binary"
      << " representation of the data in row major order." << std::endl;

  exit(1);
}

int
main(int argc, char** argv)
{
  bool verbose = false;
  bool dynamic_concurrency_mode = false;
  bool streaming = false;
  bool zero_input = false;
  size_t max_threads = 16;
  // average length of a sentence
  size_t sequence_length = 20;
  int32_t percentile = -1;
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
  std::string data_directory("");
  perfclient::ProtocolType protocol = perfclient::ProtocolType::HTTP;
  std::map<std::string, std::string> http_headers;
  std::unordered_map<std::string, std::vector<int64_t>> input_shapes;

  // {name, has_arg, *flag, val}
  static struct option long_options[] = {{"streaming", 0, 0, 0},
                                         {"max-threads", 1, 0, 1},
                                         {"sequence-length", 1, 0, 2},
                                         {"percentile", 1, 0, 3},
                                         {"data-directory", 1, 0, 4},
                                         {"shape", 1, 0, 5},
                                         {0, 0, 0, 0}};

  // Parse commandline...
  int opt;
  while ((opt = getopt_long(
              argc, argv, "vdazc:u:m:x:b:t:p:i:H:l:r:s:f:", long_options,
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
      case 3:
        percentile = std::atoi(optarg);
        break;
      case 4:
        data_directory = optarg;
        break;
      case 5: {
        std::string arg = optarg;
        std::string name = arg.substr(0, arg.rfind(":"));
        std::string shape_str = arg.substr(name.size() + 1);
        size_t pos = 0;
        std::vector<int64_t> shape;
        try {
          while (pos != std::string::npos) {
            size_t comma_pos = shape_str.find(",", pos);
            int64_t dim;
            if (comma_pos == std::string::npos) {
              dim = std::stoll(shape_str.substr(pos, comma_pos));
              pos = comma_pos;
            } else {
              dim = std::stoll(shape_str.substr(pos, comma_pos - pos));
              pos = comma_pos + 1;
            }
            if (dim <= 0) {
              Usage(argv, "input shape must be > 0");
            }
            shape.emplace_back(dim);
          }
        }
        catch (const std::invalid_argument& ia) {
          Usage(argv, "failed to parse input shape: " + std::string(optarg));
        }
        input_shapes[name] = shape;
        break;
      }
      case 'v':
        verbose = true;
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
        protocol = perfclient::ParseProtocol(optarg);
        break;
      case 'H': {
        std::string arg = optarg;
        std::string header = arg.substr(0, arg.find(":"));
        http_headers[header] = arg.substr(header.size() + 1);
        break;
      }
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
  if (protocol == perfclient::ProtocolType::UNKNOWN) {
    Usage(argv, "protocol should be either HTTP or gRPC");
  }
  if (streaming && (protocol != perfclient::ProtocolType::GRPC)) {
    Usage(argv, "streaming is only allowed with gRPC protocol");
  }
  if (!http_headers.empty() && (protocol != perfclient::ProtocolType::HTTP)) {
    std::cerr << "WARNING: HTTP headers specified with -H are ignored when "
                 "using non-HTTP protocol."
              << std::endl;
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
  if (percentile != -1 && (percentile > 99 || percentile < 1)) {
    Usage(argv, "percentile must be -1 for not reporting or in range (0, 100)");
  }
  if (zero_input && !data_directory.empty()) {
    Usage(argv, "zero input can't be set when data directory is provided");
  }

  // trap SIGINT to allow threads to exit gracefully
  signal(SIGINT, perfclient::SignalHandler);

  nic::Error err;
  std::shared_ptr<perfclient::ContextFactory> factory;
  std::unique_ptr<perfclient::LoadManager> manager;
  std::unique_ptr<perfclient::InferenceProfiler> profiler;
  err = perfclient::ContextFactory::Create(
      url, protocol, http_headers, streaming, model_name, model_version,
      &factory);
  if (!err.IsOk()) {
    std::cerr << err << std::endl;
    return 1;
  }
  err = perfclient::ConcurrencyManager::Create(
      batch_size, max_threads, sequence_length, zero_input, input_shapes,
      data_directory, factory, &manager);
  if (!err.IsOk()) {
    std::cerr << err << std::endl;
    return 1;
  }
  err = perfclient::InferenceProfiler::Create(
      verbose, stable_offset, measurement_window_ms, max_measurement_count,
      percentile, factory, std::move(manager), &profiler);
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
  if (percentile == -1) {
    std::cout << "  Stabilizing using average latency" << std::endl;
  } else {
    std::cout << "  Stabilizing using p" << percentile << " latency"
              << std::endl;
  }
  std::cout << std::endl;

  perfclient::PerfStatus status_summary;
  std::vector<perfclient::PerfStatus> summary;

  if (!dynamic_concurrency_mode) {
    err = profiler->Profile(concurrent_request_count, status_summary);
    if (err.IsOk()) {
      err = perfclient::Report(
          status_summary, concurrent_request_count, percentile, protocol,
          verbose);
      summary.push_back(status_summary);
    }
  } else {
    for (size_t count = concurrent_request_count;
         (count <= max_concurrency) || (max_concurrency == 0); count++) {
      err = profiler->Profile(count, status_summary);
      if (err.IsOk()) {
        err = perfclient::Report(
            status_summary, count, percentile, protocol, verbose);
        summary.push_back(status_summary);
        uint64_t stabilizing_latency_ms =
            status_summary.stabilizing_latency_ns / (1000 * 1000);
        if ((stabilizing_latency_ms >= latency_threshold_ms) || !err.IsOk()) {
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
    // In the case of early_exit, the thread does not return and continues to
    // report the summary
    if (!perfclient::early_exit) {
      return 1;
    }
  }
  if (summary.size()) {
    // Can print more depending on verbose, but it seems too much information
    std::cout << "Inferences/Second vs. Client ";
    if (percentile == -1) {
      std::cout << "Average Batch Latency" << std::endl;
    } else {
      std::cout << "p" << percentile << " Batch Latency" << std::endl;
    }

    for (perfclient::PerfStatus& status : summary) {
      std::cout << "Concurrency: " << status.concurrency << ", "
                << status.client_infer_per_sec << " infer/sec, latency "
                << (status.stabilizing_latency_ns / 1000) << " usec"
                << std::endl;
    }

    if (!filename.empty()) {
      std::ofstream ofs(filename, std::ofstream::out);

      ofs << "Concurrency,Inferences/Second,Client Send,"
          << "Network+Server Send/Recv,Server Queue,"
          << "Server Compute,Client Recv";
      for (const auto& percentile : summary[0].client_percentile_latency_ns) {
        ofs << ",p" << percentile.first << " latency";
      }
      ofs << std::endl;

      // Sort summary results in order of increasing infer/sec.
      std::sort(
          summary.begin(), summary.end(),
          [](const perfclient::PerfStatus& a,
             const perfclient::PerfStatus& b) -> bool {
            return a.client_infer_per_sec < b.client_infer_per_sec;
          });

      for (perfclient::PerfStatus& status : summary) {
        uint64_t avg_queue_ns = status.server_stats.queue_time_ns /
                                status.server_stats.request_count;
        uint64_t avg_compute_ns = status.server_stats.compute_time_ns /
                                  status.server_stats.request_count;
        uint64_t avg_client_wait_ns = status.client_avg_latency_ns -
                                      status.client_avg_send_time_ns -
                                      status.client_avg_receive_time_ns;
        // Network misc is calculated by subtracting data from different
        // measurements (server v.s. client), so the result needs to be capped
        // at 0
        uint64_t avg_network_misc_ns =
            avg_client_wait_ns > (avg_queue_ns + avg_compute_ns)
                ? avg_client_wait_ns - (avg_queue_ns + avg_compute_ns)
                : 0;

        ofs << status.concurrency << "," << status.client_infer_per_sec << ","
            << (status.client_avg_send_time_ns / 1000) << ","
            << (avg_network_misc_ns / 1000) << "," << (avg_queue_ns / 1000)
            << "," << (avg_compute_ns / 1000) << ","
            << (status.client_avg_receive_time_ns / 1000);
        for (const auto& percentile : status.client_percentile_latency_ns) {
          ofs << "," << (percentile.second / 1000);
        }
        ofs << std::endl;
      }
      ofs.close();

      // Record composing model stat in a separate file
      if (!summary.front().server_stats.composing_models_stat.empty()) {
        // For each of the composing model, generate CSV file in the same format
        // as the one for ensemble.
        for (const auto& model_info :
             summary[0].server_stats.composing_models_stat) {
          const auto& name = model_info.first.first;
          const auto& version = model_info.first.second;
          const auto name_ver = name + "_v" + std::to_string(version);

          std::ofstream ofs(name_ver + "." + filename, std::ofstream::out);
          ofs << "Concurrency,Inferences/Second,Client Send,"
              << "Network+Server Send/Recv,Server Queue,"
              << "Server Compute,Client Recv" << std::endl;

          for (perfclient::PerfStatus& status : summary) {
            auto it = status.server_stats.composing_models_stat.find(
                model_info.first);
            const auto& stats = it->second;
            uint64_t avg_queue_ns = stats.queue_time_ns / stats.request_count;
            uint64_t avg_compute_ns =
                stats.compute_time_ns / stats.request_count;
            uint64_t avg_overhead_ns = stats.cumm_time_ns / stats.request_count;
            avg_overhead_ns =
                (avg_overhead_ns > (avg_queue_ns + avg_compute_ns))
                    ? (avg_overhead_ns - avg_queue_ns - avg_compute_ns)
                    : 0;
            // infer / sec of the composing model is calculated using the
            // request count ratio between the composing model and the ensemble
            double infer_ratio =
                1.0 * stats.request_count / status.server_stats.request_count;
            int infer_per_sec = infer_ratio * status.client_infer_per_sec;
            ofs << status.concurrency << "," << infer_per_sec << ",0,"
                << (avg_overhead_ns / 1000) << "," << (avg_queue_ns / 1000)
                << "," << (avg_compute_ns / 1000) << ",0" << std::endl;
          }
        }
      }
    }
  }
  return 0;
}
