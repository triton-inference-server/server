// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "src/clients/c++/perf_client/custom_load_manager.h"
#include "src/clients/c++/perf_client/inference_profiler.h"
#include "src/clients/c++/perf_client/model_parser.h"
#include "src/clients/c++/perf_client/perf_utils.h"
#include "src/clients/c++/perf_client/request_rate_manager.h"

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
// There are broadly three ways to load server for the data collection using
// perf_client:
// - Maintaining Target Concurrency:
//     In this setting, the client will maintain a target number of concurrent
//     requests sent to the server (see --concurrency-range option) while
//     taking measurements.
//     The number of requests will be the total number of requests sent within
//     the time interval for measurement (see --measurement-interval option) and
//     the latency will be the average latency across all requests.
//
//     Besides throughput and latency, which is measured on client side,
//     the following data measured by the server will also be reported
//     in this setting:
//     - Concurrent request: the number of concurrent requests as specified
//         in --concurrency-range option. Note, for running perf client for
//         a single concurrency, user must specify --concurrency-range
//         <'start'>, omitting 'end' and 'step' values.
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
//     If all fields of --concurrency-range are specified, the client will
//     perform the following procedure:
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
// - Maintaining Target Request Rate:
//     This mode is enabled only when --request-rate-range option is specified.
//     Unlike above, here the client will try to maintain a target rate of
//     requests issued to the server while taking measurements. Rest of the
//     behaviour of client is identical as above. It is important to note that
//     even though over a  sufficiently large interval the rate of requests
//     will tend to the target request rate, the actual request rate for a small
//     time interval will depend upon the selected request distribution
//     (--request-distribution). For 'constant' request distribution the time
//     interval between successive requests is maintained to be constant, hence
//     request rate is constant over time. However, 'poisson' request
//     distribution varies the time interval between successive requests such
//     that there are periods of bursts and nulls in request generation.
//     Additionally, 'poisson' distribution mimics the real-world traffic and
//     can be used to obtain measurements for a realistic-load.
//     With each request-rate, the client also reports the 'Delayed Request
//     Count' which gives an idea of how many requests missed their schedule as
//     specified by the distribution. Users can use --max-threads to increase
//     the number of threads which might help in dispatching requests as per
//     the schedule. Also note that a very large number of threads might be
//     counter-productive with most of the time being spent on context-switching
//     the threads.
//
// - Following User Provided Request Delivery Schedule:
//     This mode is enabled only when --request-intervals option is specified.
//     In this case, client will try to dispatch the requests to the server with
//     time intervals between successive requests specified in a user provided
//     file. This file should contain time intervals in microseconds in each new
//     line. Client will loop around the values to produce a consistent load for
//     measurements. Once, the readings are stabilized then the final statistics
//     will be reported. The statistics will include 'Delayed Request Count' for
//     the requests that missed their schedule. As described before, users can
//     tune --max-threads to allow client in keeping up with the schedule. This
//     mode will help user in analyzing the performance of the server under
//     different custom settings which may be of interest.
//
// By default, perf_client will maintain target concurrency while measuring the
// performance.
//
// Options:
// -b: batch size for each request sent.
// --concurrency-range: The range of concurrency levels perf_client will use.
//    A concurrency level indicates the number of concurrent requests in queue.
// --request-rate-range: The range of request rates perf_client will use to load
//    the server.
// --request-intervals: File containing time intervals (in microseconds) to use
//    between successive requests.
// --latency-threshold: latency threshold in msec.
// --measurement-interval: time interval for each measurement window in msec.
// --async: Enables Asynchronous inference calls.
// --binary-search: Enables binary search within the specified range.
// --request-distribution: Allows user to specify the distribution for selecting
//     the time intervals between the request dispatch.
//
// For detail of the options not listed, please refer to the usage.
//

namespace {

enum SEARCH_RANGE { kSTART = 0, kEND = 1, kSTEP = 2 };

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
  std::cerr << "\t--async (-a)" << std::endl;
  std::cerr << "\t--sync" << std::endl;
  std::cerr << "\t--measurement-interval (-p) <measurement window (in msec)>"
            << std::endl;
  std::cerr << "\t--concurrency-range <start:end:step>" << std::endl;
  std::cerr << "\t--request-rate-range <start:end:step>" << std::endl;
  std::cerr << "\t--request-distribution <\"poisson\"|\"constant\">"
            << std::endl;
  std::cerr << "\t--request-intervals <path to file containing time intervals "
               "in microseconds>"
            << std::endl;
  std::cerr << "\t--binary-search" << std::endl;
  std::cerr << "\t--num-of-sequences <number of concurrent sequences>"
            << std::endl;
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
  std::cerr << std::endl;
  std::cerr << "II. INPUT DATA OPTIONS: " << std::endl;
  std::cerr << "\t-b <batch size>" << std::endl;
  std::cerr << "\t--input-data <\"zero\"|\"random\"|<path>>" << std::endl;
  std::cerr << "\t--shared-memory <\"system\"|\"cuda\"|\"none\">" << std::endl;
  std::cerr << "\t--output-shared-memory-size <size in bytes>" << std::endl;
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
            << " -v: " << FormatMessage("Enables verbose mode.", 9)
            << std::endl;
  std::cerr << std::setw(9) << std::left
            << " -v -v: " << FormatMessage("Enables extra verbose mode.", 9)
            << std::endl;
  std::cerr << std::endl;
  std::cerr << "I. MEASUREMENT PARAMETERS: " << std::endl;
  std::cerr
      << FormatMessage(
             " --async (-a): Enables asynchronous mode in perf_client. "
             "By default, perf_client will use synchronous API to "
             "request inference. However, if the model is sequential "
             "then default mode is asynchronous. Specify --sync to "
             "operate sequential models in synchronous mode. In synchronous "
             "mode, perf_client will start threads equal to the concurrency "
             "level. Use asynchronous mode to limit the number of threads, yet "
             "maintain the concurrency.",
             18)
      << std::endl;
  std::cerr << FormatMessage(
                   " --sync: Force enables synchronous mode in perf_client. "
                   "Can be used to operate perf_client with sequential model "
                   "in synchronous mode.",
                   18)
            << std::endl;
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
             "--latency-threshold can not be both 0 simultaneously. 'end' can "
             "not be 0 for sequence models while using asynchronous mode.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --request-rate-range <start:end:step>: Determines the range of "
             "request rates for load generated by client. This option can take "
             "floating-point values. The search along the request rate range "
             "is enabled only when using this option. If not specified, then "
             "client will search along the concurrency-range. The perf_client "
             "will start from the request rate of 'start' and go till "
             "'end' with a stride of 'step'. The default values of 'start', "
             "'end' and 'step' are all 1.0. If 'end' is not specified then "
             "perf_client will run for a single request rate as determined by "
             "'start'. If 'end' is set as 0.0, then the request rate will be "
             "incremented by 'step' till latency threshold is met. 'end' and "
             "--latency-threshold can not be both 0 simultaneously.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --request-distribution <\"poisson\"|\"constant\">: Specifies "
             "the time interval distribution between dispatching inference "
             "requests to the server. Poisson distribution closely mimics the "
             "real-world work load on a server. This option is ignored if not "
             "using --request-rate-range. By default, this option is set to be "
             "constant.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --request-intervals: Specifies a path to a file containing time "
             "intervals in microseconds. Each time interval should be in a new "
             "line. The client will try to maintain time intervals between "
             "successive generated requests to be as close as possible in this "
             "file. This option can be used to apply custom load to server "
             "with a certain pattern of interest. The client will loop around "
             "the file if the duration of execution exceeds to that accounted "
             "for by the intervals. This option can not be used with "
             "--request-rate-range or --concurrency-range.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             "--binary-search: Enables the binary search on the specified "
             "search range. This option requires 'start' and 'end' to be "
             "expilicitly specified in the --concurrency-range or "
             "--request-rate-range. When using this option, 'step' is more "
             "like the precision. Lower the 'step', more the number of "
             "iterations along the search path to find suitable convergence. "
             "By default, linear search is used.",
             18)
      << std::endl;

  std::cerr << FormatMessage(
                   "--num-of-sequences: Sets the number of concurrent "
                   "sequences for sequence models. This option is ignored when "
                   "--request-rate-range is not specified. By default, its "
                   "value is 4.",
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
             "created for providing desired concurrency or request rate. "
             "However, when running"
             "in synchronous mode with concurrency-range having explicit 'end' "
             "specification,"
             "this value will be ignored. Default is 4 if --request-rate-range "
             "is specified otherwise default is 16.",
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
  std::cerr
      << FormatMessage(
             " --input-data: Select the type of data that will be used "
             "for input in inference requests. The available options are "
             "\"zero\", \"random\", path to a directory or a json file. If the "
             "option is path to a directory then the directory must "
             "contain a binary/text file for each non-string/string input "
             "respectively, named the same as the input. Each "
             "file must contain the data required for that input for a batch-1 "
             "request. Each binary file should contain the raw binary "
             "representation of the input in row-major order for non-string "
             "inputs. The text file should contain all strings needed by "
             "batch-1, each in a new line, listed in row-major order. When "
             "pointing to a json file, user must adhere to the format "
             "described in /docs/perf_client.rst. By specifying json data, "
             "users can control data used with every request. Multiple data "
             "streams can be specified for a sequence model and the client "
             "will select a data stream in a round-robin fashion for every new "
             "sequence. Muliple json files can also be provided (--input-data "
             "json_file1 --input-data json-file2 and so on) and the client "
             "will append data streams from each file. Default is \"random\".",
             18)
      << std::endl;
  std::cerr << FormatMessage(
                   " --shared-memory <\"system\"|\"cuda\"|\"none\">: Specifies "
                   "the type of the shared memory to use for input and output "
                   "data. Default is none.",
                   18)
            << std::endl;

  std::cerr
      << FormatMessage(
             " --output-shared-memory-size: The size in bytes of the shared "
             "memory region to allocate per output tensor. Only needed when "
             "one or more of the outputs are of string type and/or variable "
             "shape. The value should be larger than the size of the largest "
             "output tensor the model is expected to return. The client will "
             "use the following formula to calculate the total shared memory "
             "to allocate: output_shared_memory_size * number_of_outputs * "
             "batch_size. Defaults to 100KB.",
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

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  bool extra_verbose = false;
  bool streaming = false;
  size_t max_threads = 4;
  // average length of a sentence
  size_t sequence_length = 20;
  int32_t percentile = -1;
  uint64_t latency_threshold_ms = NO_LIMIT;
  int32_t batch_size = 1;
  uint64_t concurrency_range[3] = {1, 1, 1};
  double request_rate_range[3] = {1.0, 1.0, 1.0};
  double stability_threshold = 0.1;
  uint64_t measurement_window_ms = 5000;
  size_t max_trials = 10;
  std::string model_name;
  std::string model_version;
  std::string url("localhost:8000");
  std::string filename("");
  ProtocolType protocol = ProtocolType::HTTP;
  std::shared_ptr<nic::Headers> http_headers(new nic::Headers());
  SharedMemoryType shared_memory_type = SharedMemoryType::NO_SHARED_MEMORY;
  size_t output_shm_size = 100 * 1024;
  std::unordered_map<std::string, std::vector<int64_t>> input_shapes;
  size_t string_length = 128;
  std::string string_data;
  std::vector<std::string> user_data;
  bool zero_input = false;
  int32_t concurrent_request_count = 1;
  size_t max_concurrency = 0;
  uint32_t num_of_sequences = 4;
  bool dynamic_concurrency_mode = false;
  bool async = false;
  bool forced_sync = false;

  bool using_concurrency_range = false;
  bool using_request_rate_range = false;
  bool using_custom_intervals = false;
  SearchMode search_mode = SearchMode::LINEAR;
  Distribution request_distribution = Distribution::CONSTANT;
  std::string request_intervals_file("");

  // Required for detecting the use of conflicting options
  bool using_old_options = false;
  bool url_specified = false;
  bool max_threads_specified = false;


  // {name, has_arg, *flag, val}
  static struct option long_options[] = {
      {"streaming", 0, 0, 0},
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
      {"async", 0, 0, 14},
      {"sync", 0, 0, 15},
      {"request-rate-range", 1, 0, 16},
      {"num-of-sequences", 1, 0, 17},
      {"binary-search", 0, 0, 18},
      {"request-distribution", 1, 0, 19},
      {"request-intervals", 1, 0, 20},
      {"shared-memory", 1, 0, 21},
      {"output-shared-memory-size", 1, 0, 22},
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
        max_threads_specified = true;
        break;
      case 2:
        sequence_length = std::atoi(optarg);
        break;
      case 3:
        percentile = std::atoi(optarg);
        break;
      case 4:
        user_data.push_back(optarg);
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
          Usage(
              argv,
              "failed to parse concurrency range: " + std::string(optarg));
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
        if (IsDirectory(arg) || IsFile(arg)) {
          user_data.push_back(optarg);
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
      case 14: {
        async = true;
        break;
      }
      case 15: {
        forced_sync = true;
        break;
      }
      case 16: {
        using_request_rate_range = true;
        std::string arg = optarg;
        size_t pos = 0;
        int index = 0;
        try {
          while (pos != std::string::npos) {
            size_t colon_pos = arg.find(":", pos);
            if (index > 2) {
              Usage(
                  argv,
                  "option request_rate_range can have maximum of three "
                  "elements");
            }
            if (colon_pos == std::string::npos) {
              request_rate_range[index] = std::stod(arg.substr(pos, colon_pos));
              pos = colon_pos;
            } else {
              request_rate_range[index] =
                  std::stod(arg.substr(pos, colon_pos - pos));
              pos = colon_pos + 1;
              index++;
            }
          }
        }
        catch (const std::invalid_argument& ia) {
          Usage(
              argv,
              "failed to parse request rate range: " + std::string(optarg));
        }
        break;
      }
      case 17: {
        num_of_sequences = std::atoi(optarg);
        break;
      }
      case 18: {
        search_mode = SearchMode::BINARY;
        break;
      }
      case 19: {
        std::string arg = optarg;
        if (arg.compare("poisson") == 0) {
          request_distribution = Distribution::POISSON;
        } else if (arg.compare("constant") == 0) {
          request_distribution = Distribution::CONSTANT;
        } else {
          Usage(
              argv, "unsupported request distribution provided " +
                        std::string(optarg));
        }
        break;
      }
      case 20:
        using_custom_intervals = true;
        request_intervals_file = optarg;
        break;
      case 21: {
        std::string arg = optarg;
        if (arg.compare("system") == 0) {
          shared_memory_type = SharedMemoryType::SYSTEM_SHARED_MEMORY;
        } else if (arg.compare("cuda") == 0) {
#ifdef TRITON_ENABLE_GPU
          shared_memory_type = SharedMemoryType::CUDA_SHARED_MEMORY;
#else
          Usage(
              argv,
              "cuda shared memory is not supported when TRITON_ENABLE_GPU=0");
#endif  // TRITON_ENABLE_GPU
        }
        break;
      }
      case 22:
        output_shm_size = std::atoi(optarg);
        break;
      case 'v':
        extra_verbose = verbose;
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
        model_version = optarg;
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
        protocol = ParseProtocol(optarg);
        break;
      case 'H': {
        std::string arg = optarg;
        std::string header = arg.substr(0, arg.find(":"));
        (*http_headers)[header] = arg.substr(header.size() + 1);
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
        async = true;
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
  if (concurrency_range[SEARCH_RANGE::kSTART] <= 0 ||
      concurrent_request_count < 0) {
    Usage(argv, "The start of the search range must be > 0");
  }
  if (request_rate_range[SEARCH_RANGE::kSTART] <= 0) {
    Usage(argv, "The start of the search range must be > 0");
  }
  if (protocol == ProtocolType::UNKNOWN) {
    Usage(argv, "protocol should be either HTTP or gRPC");
  }
  if (streaming && (protocol != ProtocolType::GRPC)) {
    Usage(argv, "streaming is only allowed with gRPC protocol");
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
  if (zero_input && !user_data.empty()) {
    Usage(argv, "zero input can't be set when data directory is provided");
  }
  if (async && forced_sync) {
    Usage(argv, "Both --async and --sync can not be specified simultaneously.");
  }


  if (using_concurrency_range && using_old_options) {
    Usage(argv, "can not use deprecated options with --concurrency-range");
  } else if (using_old_options) {
    if (dynamic_concurrency_mode) {
      concurrency_range[SEARCH_RANGE::kEND] = max_concurrency;
    }
    concurrency_range[SEARCH_RANGE::kSTART] = concurrent_request_count;
  }

  if (using_request_rate_range && using_old_options) {
    Usage(argv, "can not use concurrency options with --request-rate-range");
  }

  if (using_request_rate_range && using_concurrency_range) {
    Usage(
        argv,
        "can not specify concurrency_range and request_rate_range "
        "simultaneously");
  }

  if (using_custom_intervals && using_old_options) {
    Usage(argv, "can not use deprecated options with --request-intervals");
  }

  if ((using_custom_intervals) &&
      (using_request_rate_range || using_concurrency_range)) {
    Usage(
        argv,
        "can not use --concurrency-range or --request-rate-range "
        "along with --request-intervals");
  }

  if (((concurrency_range[SEARCH_RANGE::kEND] == NO_LIMIT) ||
       (request_rate_range[SEARCH_RANGE::kEND] ==
        static_cast<double>(NO_LIMIT))) &&
      (latency_threshold_ms == NO_LIMIT)) {
    Usage(
        argv,
        "The end of the search range and the latency limit can not be both 0 "
        "(or 0.0) simultaneously");
  }

  if (((concurrency_range[SEARCH_RANGE::kEND] == NO_LIMIT) ||
       (request_rate_range[SEARCH_RANGE::kEND] ==
        static_cast<double>(NO_LIMIT))) &&
      (search_mode == SearchMode::BINARY)) {
    Usage(
        argv,
        "The end of the range can not be 0 (or 0.0) for binary search mode.");
  }

  if ((search_mode == SearchMode::BINARY) &&
      (latency_threshold_ms == NO_LIMIT)) {
    Usage(argv, "The latency threshold can not be 0 for binary search mode.");
  }

  if (((concurrency_range[SEARCH_RANGE::kEND] <
        concurrency_range[SEARCH_RANGE::kSTART]) ||
       (request_rate_range[SEARCH_RANGE::kEND] <
        request_rate_range[SEARCH_RANGE::kSTART])) &&
      (search_mode == SearchMode::BINARY)) {
    Usage(
        argv,
        "The end of the range can not be less than start of the range for "
        "binary search mode.");
  }

  if (!url_specified && (protocol == ProtocolType::GRPC)) {
    url = "localhost:8001";
  }

  bool target_concurrency =
      (using_concurrency_range || using_old_options ||
       !(using_request_rate_range || using_custom_intervals));


  // Overriding the max_threads default for request_rate search
  if (!max_threads_specified && target_concurrency) {
    max_threads = 16;
  }

  // trap SIGINT to allow threads to exit gracefully
  signal(SIGINT, SignalHandler);

  std::shared_ptr<TritonClientFactory> factory;
  // Disabling verbosity in the clients to prevent huge dump of
  // messages.
  FAIL_IF_ERR(
      TritonClientFactory::Create(
          url, protocol, http_headers, extra_verbose, &factory),
      "failed to create client factory");

  std::unique_ptr<TritonClientWrapper> triton_client_wrapper;
  FAIL_IF_ERR(
      factory->CreateTritonClient(&triton_client_wrapper),
      "failed to create triton client");

  std::shared_ptr<ModelParser> parser = std::make_shared<ModelParser>();
  if (protocol == ProtocolType::HTTP) {
    rapidjson::Document model_metadata;
    FAIL_IF_ERR(
        triton_client_wrapper->ModelMetadata(
            &model_metadata, model_name, model_version),
        "failed to get model metadata");
    rapidjson::Document model_config;
    FAIL_IF_ERR(
        triton_client_wrapper->ModelConfig(
            &model_config, model_name, model_version),
        "failed to get model config");
    FAIL_IF_ERR(
        parser->Init(
            model_metadata, model_config, model_version, input_shapes,
            triton_client_wrapper),
        "failed to create model parser");
  } else {
    inference::ModelMetadataResponse model_metadata;
    FAIL_IF_ERR(
        triton_client_wrapper->ModelMetadata(
            &model_metadata, model_name, model_version),
        "failed to get model metadata");

    inference::ModelConfigResponse model_config;
    FAIL_IF_ERR(
        triton_client_wrapper->ModelConfig(
            &model_config, model_name, model_version),
        "failed to get model config");
    FAIL_IF_ERR(
        parser->Init(
            model_metadata, model_config.config(), model_version, input_shapes,
            triton_client_wrapper),
        "failed to create model parser");
  }

  if ((parser->MaxBatchSize() == 0) && batch_size > 1) {
    std::cerr << "can not specify batch size > 1 as the model does not support "
                 "batching"
              << std::endl;
    return 1;
  }

  // Change the default value for the --async option for sequential models
  if ((parser->SchedulerType() == ModelParser::SEQUENCE) ||
      (parser->SchedulerType() == ModelParser::ENSEMBLE_SEQUENCE)) {
    if (!async) {
      async = forced_sync ? false : true;
    }
    // Validate the batch_size specification
    if (batch_size > 1) {
      std::cerr << "can not specify batch size > 1 when using a sequence model"
                << std::endl;
      return 1;
    }
  }

  if (streaming) {
    if (forced_sync) {
      std::cerr << "can not use streaming with synchronous API" << std::endl;
      return 1;
    }
    async = true;
  }

  std::unique_ptr<LoadManager> manager;

  if (target_concurrency) {
    if ((parser->SchedulerType() == ModelParser::SEQUENCE) ||
        (parser->SchedulerType() == ModelParser::ENSEMBLE_SEQUENCE)) {
      if (concurrency_range[SEARCH_RANGE::kEND] == NO_LIMIT && async) {
        std::cerr << "The 'end' concurrency can not be 0 for sequence "
                     "models when using asynchronous API."
                  << std::endl;
        return 1;
      }
    }
    max_concurrency = std::max(
        concurrency_range[SEARCH_RANGE::kSTART],
        concurrency_range[SEARCH_RANGE::kEND]);


    if (!async) {
      if (concurrency_range[SEARCH_RANGE::kEND] == NO_LIMIT) {
        std::cerr
            << "WARNING: The maximum attainable concurrency will be limited by "
               "max_threads specification."
            << std::endl;
        concurrency_range[SEARCH_RANGE::kEND] = max_threads;
      } else {
        // As only one synchronous request can be generated from a thread at a
        // time, to maintain the requested concurrency, that many threads need
        // to be generated.
        if (max_threads_specified) {
          std::cerr
              << "WARNING: Overriding max_threads specification to ensure "
                 "requested concurrency range."
              << std::endl;
        }
        max_threads = std::max(
            concurrency_range[SEARCH_RANGE::kSTART],
            concurrency_range[SEARCH_RANGE::kEND]);
      }
    }
    FAIL_IF_ERR(
        ConcurrencyManager::Create(
            async, streaming, batch_size, max_threads, max_concurrency,
            sequence_length, string_length, string_data, zero_input, user_data,
            shared_memory_type, output_shm_size, parser, factory, &manager),
        "failed to create concurrency manager");

  } else if (using_request_rate_range) {
    FAIL_IF_ERR(
        RequestRateManager::Create(
            async, streaming, measurement_window_ms, request_distribution,
            batch_size, max_threads, num_of_sequences, sequence_length,
            string_length, string_data, zero_input, user_data,
            shared_memory_type, output_shm_size, parser, factory, &manager),
        "failed to create request rate manager");

  } else {
    FAIL_IF_ERR(
        CustomLoadManager::Create(
            async, streaming, measurement_window_ms, request_intervals_file,
            batch_size, max_threads, num_of_sequences, sequence_length,
            string_length, string_data, zero_input, user_data,
            shared_memory_type, output_shm_size, parser, factory, &manager),
        "failed to create custom load manager");
  }

  std::unique_ptr<InferenceProfiler> profiler;
  FAIL_IF_ERR(
      InferenceProfiler::Create(
          verbose, stability_threshold, measurement_window_ms, max_trials,
          percentile, latency_threshold_ms, protocol, parser,
          std::move(triton_client_wrapper), std::move(manager), &profiler),
      "failed to create profiler");

  // pre-run report
  std::cout << "*** Measurement Settings ***" << std::endl
            << "  Batch size: " << batch_size << std::endl
            << "  Measurement window: " << measurement_window_ms << " msec"
            << std::endl;
  if (concurrency_range[SEARCH_RANGE::kEND] != 1) {
    std::cout << "  Latency limit: " << latency_threshold_ms << " msec"
              << std::endl;
    if (concurrency_range[SEARCH_RANGE::kEND] != NO_LIMIT) {
      std::cout << "  Concurrency limit: "
                << std::max(
                       concurrency_range[SEARCH_RANGE::kSTART],
                       concurrency_range[SEARCH_RANGE::kEND])
                << " concurrent requests" << std::endl;
    }
  }
  if (request_rate_range[SEARCH_RANGE::kEND] != 1.0) {
    std::cout << "  Latency limit: " << latency_threshold_ms << " msec"
              << std::endl;
    if (request_rate_range[SEARCH_RANGE::kEND] !=
        static_cast<double>(NO_LIMIT)) {
      std::cout << "  Request Rate limit: "
                << std::max(
                       request_rate_range[SEARCH_RANGE::kSTART],
                       request_rate_range[SEARCH_RANGE::kEND])
                << " requests per seconds" << std::endl;
    }
  }
  if (using_request_rate_range) {
    if (request_distribution == Distribution::POISSON) {
      std::cout << "  Using poisson distribution on request generation"
                << std::endl;
    } else {
      std::cout << "  Using uniform distribution on request generation"
                << std::endl;
    }
  }
  if (search_mode == SearchMode::BINARY) {
    std::cout << "  Using Binary Search algorithm" << std::endl;
  }
  if (async) {
    std::cout << "  Using asynchronous calls for inference" << std::endl;
  } else {
    std::cout << "  Using synchronous calls for inference" << std::endl;
  }
  if (parser->IsDecoupled()) {
    std::cout << "  Detected decoupled model, using the first response for "
                 "measuring latency"
              << std::endl;
  }
  if (percentile == -1) {
    std::cout << "  Stabilizing using average latency" << std::endl;
  } else {
    std::cout << "  Stabilizing using p" << percentile << " latency"
              << std::endl;
  }
  std::cout << std::endl;

  std::vector<PerfStatus> summary;

  if (using_custom_intervals) {
    // Will be using user-provided time intervals, hence no control variable.
    search_mode = SearchMode::NONE;
  }

  nic::Error err;
  if (target_concurrency) {
    err = profiler->Profile<size_t>(
        concurrency_range[SEARCH_RANGE::kSTART],
        concurrency_range[SEARCH_RANGE::kEND],
        concurrency_range[SEARCH_RANGE::kSTEP], search_mode, summary);
  } else {
    err = profiler->Profile<double>(
        request_rate_range[SEARCH_RANGE::kSTART],
        request_rate_range[SEARCH_RANGE::kEND],
        request_rate_range[SEARCH_RANGE::kSTEP], search_mode, summary);
  }

  if (!err.IsOk()) {
    std::cerr << err << std::endl;
    // In the case of early_exit, the thread does not return and continues to
    // report the summary
    if (!early_exit) {
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

    for (PerfStatus& status : summary) {
      if (target_concurrency) {
        std::cout << "Concurrency: " << status.concurrency << ", ";
      } else {
        std::cout << "Request Rate: " << status.request_rate << ", ";
      }
      std::cout << "throughput: " << status.client_stats.infer_per_sec
                << " infer/sec, latency "
                << (status.stabilizing_latency_ns / 1000) << " usec"
                << std::endl;
    }

    if (!filename.empty()) {
      std::ofstream ofs(filename, std::ofstream::out);
      if (target_concurrency) {
        ofs << "Concurrency,";
      } else {
        ofs << "Request Rate,";
      }
      ofs << "Inferences/Second,Client Send,"
          << "Network+Server Send/Recv,Server Queue,"
          << "Server Compute Input,Server Compute Infer,"
          << "Server Compute Output,Client Recv";
      for (const auto& percentile :
           summary[0].client_stats.percentile_latency_ns) {
        ofs << ",p" << percentile.first << " latency";
      }
      ofs << std::endl;

      // Sort summary results in order of increasing infer/sec.
      std::sort(
          summary.begin(), summary.end(),
          [](const PerfStatus& a, const PerfStatus& b) -> bool {
            return a.client_stats.infer_per_sec < b.client_stats.infer_per_sec;
          });

      for (PerfStatus& status : summary) {
        uint64_t avg_queue_ns = status.server_stats.queue_time_ns /
                                status.server_stats.success_count;
        uint64_t avg_compute_input_ns =
            status.server_stats.compute_input_time_ns /
            status.server_stats.success_count;
        uint64_t avg_compute_infer_ns =
            status.server_stats.compute_infer_time_ns /
            status.server_stats.success_count;
        uint64_t avg_compute_output_ns =
            status.server_stats.compute_output_time_ns /
            status.server_stats.success_count;

        uint64_t avg_compute_ns =
            avg_compute_input_ns + avg_compute_infer_ns + avg_compute_output_ns;

        uint64_t avg_client_wait_ns = status.client_stats.avg_latency_ns -
                                      status.client_stats.avg_send_time_ns -
                                      status.client_stats.avg_receive_time_ns;
        // Network misc is calculated by subtracting data from different
        // measurements (server v.s. client), so the result needs to be capped
        // at 0
        uint64_t avg_network_misc_ns =
            avg_client_wait_ns > (avg_queue_ns + avg_compute_ns)
                ? avg_client_wait_ns - (avg_queue_ns + avg_compute_ns)
                : 0;
        if (target_concurrency) {
          ofs << status.concurrency << ",";
        } else {
          ofs << status.request_rate << ",";
        }

        ofs << status.client_stats.infer_per_sec << ","
            << (status.client_stats.avg_send_time_ns / 1000) << ","
            << (avg_network_misc_ns / 1000) << "," << (avg_queue_ns / 1000)
            << "," << (avg_compute_input_ns / 1000) << ","
            << (avg_compute_infer_ns / 1000) << ","
            << (avg_compute_output_ns / 1000) << ","
            << (status.client_stats.avg_receive_time_ns / 1000);
        for (const auto& percentile :
             status.client_stats.percentile_latency_ns) {
          ofs << "," << (percentile.second / 1000);
        }
        ofs << std::endl;
      }
      ofs.close();

      // Record composing model stat in a separate file
      if (!summary.front().server_stats.composing_models_stat.empty()) {
        // For each of the composing model, generate CSV file in the same format
        // as the one for ensemble.
        for (const auto& model_identifier :
             summary[0].server_stats.composing_models_stat) {
          const auto& name = model_identifier.first.first;
          const auto& version = model_identifier.first.second;
          const auto name_ver = name + "_v" + version;

          std::ofstream ofs(name_ver + "." + filename, std::ofstream::out);
          if (target_concurrency) {
            ofs << "Concurrency,";
          } else {
            ofs << "Request Rate,";
          }
          ofs << "Inferences/Second,Client Send,"
              << "Network+Server Send/Recv,Server Queue,"
              << "Server Compute Input,Server Compute Infer,"
              << "Server Compute Output,Client Recv";

          for (PerfStatus& status : summary) {
            auto it = status.server_stats.composing_models_stat.find(
                model_identifier.first);
            const auto& stats = it->second;
            uint64_t avg_queue_ns = stats.queue_time_ns / stats.success_count;
            uint64_t avg_compute_input_ns =
                stats.compute_input_time_ns / stats.success_count;
            uint64_t avg_compute_infer_ns =
                stats.compute_infer_time_ns / stats.success_count;
            uint64_t avg_compute_output_ns =
                stats.compute_output_time_ns / stats.success_count;
            uint64_t avg_compute_ns = avg_compute_input_ns +
                                      avg_compute_infer_ns +
                                      avg_compute_output_ns;
            uint64_t avg_overhead_ns = stats.cumm_time_ns / stats.success_count;
            avg_overhead_ns =
                (avg_overhead_ns > (avg_queue_ns + avg_compute_ns))
                    ? (avg_overhead_ns - avg_queue_ns - avg_compute_ns)
                    : 0;
            // infer / sec of the composing model is calculated using the
            // request count ratio between the composing model and the ensemble
            double infer_ratio =
                1.0 * stats.success_count / status.server_stats.success_count;
            double infer_per_sec =
                infer_ratio * status.client_stats.infer_per_sec;
            if (target_concurrency) {
              ofs << status.concurrency << ",";
            } else {
              ofs << status.request_rate << ",";
            }
            ofs << infer_per_sec << ",0," << (avg_overhead_ns / 1000) << ","
                << (avg_queue_ns / 1000) << "," << (avg_compute_input_ns / 1000)
                << "," << (avg_compute_infer_ns / 1000) << ","
                << (avg_compute_output_ns / 1000) << ",0" << std::endl;
          }
        }
      }
    }
  }
  return 0;
}
