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

#include "src/clients/c++/perf_client/concurrency_manager.h"
#include "src/clients/c++/perf_client/context_factory.h"
#include "src/clients/c++/perf_client/inference_profiler.h"
#include "src/clients/c++/perf_client/load_manager.h"
#include "src/clients/c++/perf_client/perf_utils.h"


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
//     requests sent to the server (see --concurrency-range option). See
//     ConcurrencyManager for more detail. The number of requests will be the
//     total number of requests sent within the time interval for measurement
//     (see --measurement-interval option) and the latency will be the average
//     latency across all requests.
//
//     Besides throughput and latency, which is measured in client side,
//     the following data measured by the server will also be reported
//     in this setting:
//     - Concurrent request: the number of concurrent requests as specified
//         in --concurrency-range option. Note, for fixed concurrent request
//         mode, user must specify --concurrency-range <'start'>,
//         omitting 'end' and 'step' values.
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
//          k concurrent requests (k starts at 'start').
//       2. Gathers data reported from step 1.
//       3. Increases k by 'step' and repeats step 1 and 2 until latency from
//          current iteration exceeds latency threshold (see --latency-threshold
//          option) or concurrency level reaches 'end'. Note, by setting
//          --latency-threshold or 'end' to 0 the effect of each threshold can
//          be removed. However, both can not be 0 simultaneously.
//     At each iteration, the data mentioned in fixed concurrent request mode
//     will be reported. Besides that, after the procedure above, a collection
//     of "throughput, latency, concurrent request count" tuples will be
//     reported in increasing load level order.
//
// Options:
// -b: batch size for each request sent.
// --concurrency-range: The range of concurrency levels perf_client will use.
//     A concurrency level indicates the number of concurrent requests in queue.
// --latency-threshold: latency threshold in msec.
// --measurement-interval: time interval for each measurement window in msec.
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
        "Avg gRPC time: " + std::to_string(avg_request_time_us) + " usec (";
    if (!verbose) {
      client_library_detail +=
          "(un)marshal request/response " +
          std::to_string(avg_send_time_us + avg_receive_time_us) +
          " usec + response wait " + std::to_string(avg_response_wait_time_us) +
          " usec)";
    } else {
      client_library_detail +=
          "marshal " + std::to_string(avg_send_time_us) +
          " usec + response wait " + std::to_string(avg_response_wait_time_us) +
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

// Used to format the usage message
std::string
FormatMessage(std::string str, int offset)
{
  int width = 60;
  int current_pos = offset;
  while (current_pos + width < int(str.length())) {
    int n = str.rfind(' ', current_pos + width);
    if (n != int(std::string::npos)) {
      str.replace(n, 1, "\n\t ");
      current_pos += (width + 10);
    }
  }
  return str;
}

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "==== SYNOPSIS ====\n \n";
  std::cerr << "\t-m <model name>" << std::endl;
  std::cerr << "\t-x <model version>" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << std::endl;
  std::cerr << "I. MEASUREMENT PARAMETERS: " << std::endl;
  std::cerr << "\t--measurement-interval (-p) <measurement window (in msec)>"
            << std::endl;
  std::cerr << "\t--concurrency-range <start:end:step>" << std::endl;
  std::cerr << "\t--latency-threshold (-l) <latency threshold (in msec)>"
            << std::endl;
  std::cerr << "\t--max-threads <thread counts>" << std::endl;
  std::cerr << "\t--stability-percentage (-s) <deviation threshold for stable "
               "measurement (in percentage)>"
            << std::endl;
  std::cerr << "\t--max-trials (-r)  <maximum number of measurements for each "
               "profiling>"
            << std::endl;
  std::cerr << "\t--percentile <percentile>" << std::endl;
  std::cerr << "\tDEPRECATED OPTIONS" << std::endl;
  std::cerr << "\t-t <number of concurrent requests>" << std::endl;
  std::cerr << "\t-c <maximum concurrency>" << std::endl;
  std::cerr << "\t-d" << std::endl;
  std::cerr << "\t-a" << std::endl;
  std::cerr << std::endl;
  std::cerr << "II. INPUT DATA OPTIONS: " << std::endl;
  std::cerr << "\t-b <batch size>" << std::endl;
  std::cerr << "\t--input-data <\"zero\"|\"random\"|<path>>" << std::endl;
  std::cerr << "\t--shape <name:shape>" << std::endl;
  std::cerr << "\t--sequence-length <length>" << std::endl;
  std::cerr << "\t--string-length <length>" << std::endl;
  std::cerr << "\t--string-data <string>" << std::endl;
  std::cerr << "\tDEPRECATED OPTIONS" << std::endl;
  std::cerr << "\t-z" << std::endl;
  std::cerr << "\t--data-directory <path>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "III. SERVER DETAILS: " << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-i <Protocol used to communicate with inference service>"
            << std::endl;
  std::cerr << std::endl;
  std::cerr << "IV. OTHER OPTIONS: " << std::endl;
  std::cerr << "\t-f <filename for storing report in csv format>" << std::endl;
  std::cerr << "\t-H <HTTP header>" << std::endl;
  std::cerr << "\t--streaming" << std::endl;
  std::cerr << std::endl;
  std::cerr << "==== OPTIONS ==== \n \n";

  std::cerr
      << std::setw(9) << std::left << " -m: "
      << FormatMessage(
             "This is a required argument and is used to specify the model"
             " against which to run perf_client.",
             9)
      << std::endl;
  std::cerr << std::setw(9) << std::left << " -x: "
            << FormatMessage(
                   "The version of the above model to be used. If not specified"
                   " the most recent version (that is, the highest numbered"
                   " version) of the model will be used.",
                   9)
            << std::endl;
  std::cerr << std::setw(9) << std::left
            << " -v: " << FormatMessage("Enables the verbose mode.", 9)
            << std::endl;
  std::cerr << std::endl;
  std::cerr << "I. MEASUREMENT PARAMETERS: " << std::endl;
  std::cerr
      << FormatMessage(
             " --measurement-interval (-p): Indicates the time interval used "
             "for each measurement in milliseconds. The perf client will "
             "sample a time interval specified by -p and take measurement over "
             "the requests completed within that time interval. The default "
             "value is 5000 msec.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --concurrency-range <start:end:step>: Determines the range of "
             "concurrency levels covered by the perf_client. The perf_client "
             "will start from the concurrency level of 'start' and go till "
             "'end' with a stride of 'step'. The default value of 'end' and "
             "'step' are 1. If 'end' is not specified then perf_client will "
             "run for a single concurrency level determined by 'start'. If "
             "'end' is set as 0, then the concurrency limit will be "
             "incremented by 'step' till latency threshold is met. 'end' and "
             "--latency-threshold can not be both 0 simultaneously.",
             18)
      << std::endl;
  std::cerr << FormatMessage(
                   " --latency-threshold (-l): Sets the limit on the observed "
                   "latency. Client will terminate the concurrency search once "
                   "the measured latency exceeds this threshold. By default, "
                   "latency threshold is set 0 and the perf_client will run "
                   "for entire --concurrency-range.",
                   18)
            << std::endl;
  std::cerr
      << FormatMessage(
             " --max-threads: Sets the maximum number of threads that will be "
             "created for providing desired concurrency. Default is 16.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --stability-percentage (-s): Indicates the allowed variation in "
             "latency measurements when determining if a result is stable. The "
             "measurement is considered as stable if the recent 3 measurements "
             "are within +/- (stability percentage)% of their average in terms "
             "of both infer per second and latency. Default is 10(%).",
             18)
      << std::endl;
  std::cerr << FormatMessage(
                   " --max-trials (-r): Indicates the maximum number of "
                   "measurements for each concurrency level visited during "
                   "search. The perf client will take multiple measurements "
                   "and report the measurement until it is stable. The perf "
                   "client will abort if the measurement is still unstable "
                   "after the maximum number of measurements. The default "
                   "value is 10.",
                   18)
            << std::endl;
  std::cerr
      << FormatMessage(
             " --percentile: Indicates the confidence value as a percentile "
             "that will be used to determine if a measurement is stable. For "
             "example, a value of 85 indicates that the 85th percentile "
             "latency will be used to determine stability. The percentile will "
             "also be reported in the results. The default is -1 indicating "
             "that the average latency is used to determine stability",
             18)
      << std::endl;
  std::cerr << std::endl;
  std::cerr << "II. INPUT DATA OPTIONS: " << std::endl;
  std::cerr << std::setw(9) << std::left
            << " -b: " << FormatMessage("Batch size for each request sent.", 9)
            << std::endl;
  std::cerr << FormatMessage(
                   " --input-data: Select the type of data that will be used "
                   "for input in inference requests. The available "
                   "options are \"zero\", \"random\" or path to a directory. "
                   "If the option is path to a directory then the directory "
                   "must contain a binary file for each input, named the same "
                   "as the input. Each file must contain the data required for "
                   "that input for a batch-1 request. Each file should contain "
                   "the raw binary representation of the input in row-major "
                   "order. Default is \"random\".",
                   18)
            << std::endl;
  std::cerr << FormatMessage(
                   " --shape: The shape used for the specified input. The "
                   "argument must be specified as 'name:shape' where the shape "
                   "is a comma-separated list for dimension sizes, for example "
                   "'--shape input_name:1,2,3' indicate tensor shape [ 1, 2, 3 "
                   "]. --shape may be specified multiple times to specify "
                   "shapes for different inputs.",
                   18)
            << std::endl;
  std::cerr << FormatMessage(
                   " --sequence-length: Indicates the base length of a "
                   "sequence used for sequence models. A sequence with length "
                   "x will be composed of x requests to be sent as the "
                   "elements in the sequence. The length of the actual "
                   "sequence will be within +/- 20% of the base length.",
                   18)
            << std::endl;
  std::cerr << FormatMessage(
                   " --string-length: Specifies the length of the random "
                   "strings to be generated by the client for string input. "
                   "This option is ignored if --input-data points to a "
                   "directory. Default is 128.",
                   18)
            << std::endl;
  std::cerr << FormatMessage(
                   " --string-data: If provided, client will use this string "
                   "to initialize string input buffers. The perf client will "
                   "replicate the given string to build tensors of required "
                   "shape. --string-length will not have any effect. This "
                   "option is ignored if --input-data points to a directory.",
                   18)
            << std::endl;
  std::cerr << std::endl;
  std::cerr << "III. SERVER DETAILS: " << std::endl;
  std::cerr << std::setw(9) << std::left << " -u: "
            << FormatMessage(
                   "Specify URL to the server. Default is \"localhost:8000\" "
                   "if using HTTP and \"localhost:8001\" if using gRPC. ",
                   9)
            << std::endl;
  std::cerr << std::setw(9) << std::left << " -i: "
            << FormatMessage(
                   "The communication protocol to use. The available protocols "
                   "are gRPC and HTTP. Default is HTTP.",
                   9)
            << std::endl;
  std::cerr << std::endl;
  std::cerr << "IV. OTHER OPTIONS: " << std::endl;
  std::cerr
      << std::setw(9) << std::left << " -f: "
      << FormatMessage(
             "The latency report will be stored in the file named by "
             "this option. By default, the result is not recorded in a file.",
             9)
      << std::endl;
  std::cerr
      << std::setw(9) << std::left << " -H: "
      << FormatMessage(
             "The header will be added to HTTP requests (ignored for GRPC "
             "requests). The header must be specified as 'Header:Value'. -H "
             "may be specified multiple times to add multiple headers.",
             9)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --streaming: Enables the use of streaming API. This flag is "
             "only valid with gRPC protocol. By default, it is set false.",
             18)
      << std::endl;

  exit(1);
}

enum CONCURRENCY_RANGE { kSTART = 0, kEND = 1, kSTEP = 2 };
const uint64_t NO_LIMIT = 0;

int
main(int argc, char** argv)
{
  bool verbose = false;
  bool streaming = false;
  size_t max_threads = 16;
  // average length of a sentence
  size_t sequence_length = 20;
  int32_t percentile = -1;
  uint64_t latency_threshold_ms = NO_LIMIT;
  int32_t batch_size = 1;
  uint64_t concurrency_range[3] = {1, 1, 1};
  double stability_threshold = 0.1;
  uint64_t measurement_window_ms = 5000;
  size_t max_trials = 10;
  std::string model_name;
  int64_t model_version = -1;
  std::string url("localhost:8000");
  std::string filename("");
  perfclient::ProtocolType protocol = perfclient::ProtocolType::HTTP;
  std::map<std::string, std::string> http_headers;
  std::unordered_map<std::string, std::vector<int64_t>> input_shapes;
  size_t string_length = 128;
  std::string string_data;
  std::string data_directory("");
  bool zero_input = false;
  int32_t concurrent_request_count = 1;
  size_t max_concurrency = 0;
  bool dynamic_concurrency_mode = false;

  // Required for detecting the use of conflicting options
  bool using_concurrency_range = false;
  bool using_old_options = false;
  bool url_specified = false;


  // {name, has_arg, *flag, val}
  static struct option long_options[] = {{"streaming", 0, 0, 0},
                                         {"max-threads", 1, 0, 1},
                                         {"sequence-length", 1, 0, 2},
                                         {"percentile", 1, 0, 3},
                                         {"data-directory", 1, 0, 4},
                                         {"shape", 1, 0, 5},
                                         {"measurement-interval", 1, 0, 6},
                                         {"concurrency-range", 1, 0, 7},
                                         {"latency-threshold", 1, 0, 8},
                                         {"stability-percentage", 1, 0, 9},
                                         {"max-trials", 1, 0, 10},
                                         {"input-data", 1, 0, 11},
                                         {"string-length", 1, 0, 12},
                                         {"string-data", 1, 0, 13},
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
      case 6: {
        measurement_window_ms = std::atoi(optarg);
        break;
      }
      case 7: {
        using_concurrency_range = true;
        std::string arg = optarg;
        size_t pos = 0;
        int index = 0;
        try {
          while (pos != std::string::npos) {
            size_t colon_pos = arg.find(":", pos);
            if (index > 2) {
              Usage(
                  argv,
                  "option concurrency-range can have maximum of three "
                  "elements");
            }
            if (colon_pos == std::string::npos) {
              concurrency_range[index] = std::stoll(arg.substr(pos, colon_pos));
              pos = colon_pos;
            } else {
              concurrency_range[index] =
                  std::stoll(arg.substr(pos, colon_pos - pos));
              pos = colon_pos + 1;
              index++;
            }
          }
        }
        catch (const std::invalid_argument& ia) {
          Usage(argv, "failed to parse search range: " + std::string(optarg));
        }
        break;
      }
      case 8: {
        latency_threshold_ms = std::atoi(optarg);
        break;
      }
      case 9: {
        stability_threshold = atof(optarg) / 100;
        break;
      }
      case 10: {
        max_trials = std::atoi(optarg);
        break;
      }
      case 11: {
        std::string arg = optarg;
        // Check whether the argument is a directory
        if (perfclient::IsDirectory(arg)) {
          data_directory = optarg;
        } else if (arg.compare("zero") == 0) {
          zero_input = true;
        } else if (arg.compare("random") == 0) {
          break;
        } else {
          Usage(argv, "unsupported input data provided " + std::string(optarg));
        }
        break;
      }
      case 12: {
        string_length = std::atoi(optarg);
        break;
      }
      case 13: {
        string_data = optarg;
        break;
      }
      case 'v':
        verbose = true;
        break;
      case 'z':
        zero_input = true;
        break;
      case 'd':
        using_old_options = true;
        dynamic_concurrency_mode = true;
        break;
      case 'u':
        url_specified = true;
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
        using_old_options = true;
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
        using_old_options = true;
        max_concurrency = std::atoi(optarg);
        break;
      case 'r':
        max_trials = std::atoi(optarg);
        break;
      case 's':
        stability_threshold = atof(optarg) / 100;
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
  if (concurrency_range[CONCURRENCY_RANGE::kSTART] <= 0 ||
      concurrent_request_count < 0) {
    Usage(argv, "The start of the search range must be > 0");
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


  if (using_concurrency_range && using_old_options) {
    Usage(argv, "can not use deprecated options with concurrency_range");
  } else if (using_old_options) {
    if (dynamic_concurrency_mode) {
      concurrency_range[CONCURRENCY_RANGE::kEND] = max_concurrency;
    }
    concurrency_range[CONCURRENCY_RANGE::kSTART] = concurrent_request_count;
  }

  if ((concurrency_range[CONCURRENCY_RANGE::kEND] == NO_LIMIT) &&
      (latency_threshold_ms == NO_LIMIT)) {
    Usage(
        argv,
        "The end of the search range and the latency limit can not be both 0 "
        "simultaneously");
  }

  if (!url_specified && (protocol == perfclient::ProtocolType::GRPC)) {
    url = "localhost:8001";
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
      batch_size, max_threads, sequence_length, string_length, string_data,
      zero_input, input_shapes, data_directory, factory, &manager);
  if (!err.IsOk()) {
    std::cerr << err << std::endl;
    return 1;
  }
  err = perfclient::InferenceProfiler::Create(
      verbose, stability_threshold, measurement_window_ms, max_trials,
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
  if (concurrency_range[CONCURRENCY_RANGE::kEND] != 1) {
    std::cout << "  Latency limit: " << latency_threshold_ms << " msec"
              << std::endl;
    if (concurrency_range[CONCURRENCY_RANGE::kEND] != NO_LIMIT) {
      std::cout << "  Concurrency limit: "
                << std::max(
                       concurrency_range[CONCURRENCY_RANGE::kSTART],
                       concurrency_range[CONCURRENCY_RANGE::kEND])
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

  size_t count = concurrency_range[CONCURRENCY_RANGE::kSTART];
  do {
    err = profiler->Profile(count, status_summary);
    if (err.IsOk()) {
      err = perfclient::Report(
          status_summary, count, percentile, protocol, verbose);
      summary.push_back(status_summary);
      if (latency_threshold_ms != NO_LIMIT) {
        uint64_t stabilizing_latency_ms =
            status_summary.stabilizing_latency_ns / (1000 * 1000);
        if (!err.IsOk()) {
          std::cerr << err << std::endl;
          break;
        } else if (stabilizing_latency_ms >= latency_threshold_ms) {
          std::cerr << "Aborting execution as measured latency went over "
                       "the set limit of "
                    << latency_threshold_ms << " msec. " << std::endl;
          break;
        }
      }
    } else {
      break;
    }
    count += concurrency_range[CONCURRENCY_RANGE::kSTEP];
  } while ((count <= concurrency_range[CONCURRENCY_RANGE::kEND]) ||
           (concurrency_range[CONCURRENCY_RANGE::kEND] == NO_LIMIT));


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
            double infer_per_sec = infer_ratio * status.client_infer_per_sec;
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
