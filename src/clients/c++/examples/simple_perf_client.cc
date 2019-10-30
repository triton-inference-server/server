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

#include <semaphore.h>
#include <unistd.h>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include "src/clients/c++/library/request_grpc.h"
#include "src/clients/c++/library/request_http.h"

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

constexpr uint64_t NANOS_PER_SECOND = 1000000000;
#define TIMESPEC_TO_NANOS(TS) ((TS).tv_sec * NANOS_PER_SECOND + (TS).tv_nsec)

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    nic::Error err = (X);                                          \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

#define FAIL_IF(X, MSG)                             \
  {                                                 \
    if (X) {                                        \
      std::cerr << "error: " << (MSG) << std::endl; \
      exit(1);                                      \
  }}

namespace {

//
// C++11 doesn't have a barrier so we implement our own.
//
class Barrier {
 public:
  explicit Barrier(size_t cnt) : threshold_(cnt), count_(cnt), generation_(0) {}

  void Wait()
  {
    std::unique_lock<std::mutex> lock(mu_);
    auto lgen = generation_;
    if (--count_ == 0) {
      generation_++;
      count_ = threshold_;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [this, lgen] { return lgen != generation_; });
    }
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  const size_t threshold_;
  size_t count_;
  size_t generation_;
};

class InferContextFactory {
 public:
  InferContextFactory(
      const std::string& protocol, const std::string& url,
      const std::string& model_name, uint32_t batch_size, uint32_t tensor_size,
      bool verbose)
      : protocol_(protocol), url_(url), model_name_(model_name),
        batch_size_(batch_size), tensor_size_(tensor_size), verbose_(verbose)
  {
    input_data_.resize(tensor_size_);
  }

  // Create an inference context.
  void Create(std::unique_ptr<nic::InferContext>* ctx) const
  {
    nic::Error err;

    if (protocol_ == "http") {
      err = nic::InferHttpContext::Create(
          ctx, url_, model_name_, -1 /* model_version */, verbose_);
    } else if (protocol_ == "grpc") {
      err = nic::InferGrpcContext::Create(
          ctx, url_, model_name_, -1 /* model_version */, verbose_);
    } else {
      std::cerr << "error: unknown protocol '" + protocol_ + "'" << std::endl;
      exit(1);
    }

    if (!err.IsOk()) {
      std::cerr << "error: unable to create inference context: " << err
                << std::endl;
      exit(1);
    }

    // Set the context options to specified batch-size and request
    // size. Request that all output tensors be returned.
    std::unique_ptr<nic::InferContext::Options> options;
    FAIL_IF_ERR(
        nic::InferContext::Options::Create(&options),
        "unable to create inference options");

    options->SetBatchSize(batch_size_);
    for (const auto& output : (*ctx)->Outputs()) {
      options->AddRawResult(output);
    }

    FAIL_IF_ERR(
        (*ctx)->SetRunOptions(*options), "unable to set inference options");

    const std::vector<std::shared_ptr<nic::InferContext::Input>>& inputs =
        (*ctx)->Inputs();
    FAIL_IF(inputs.size() != 1, "expected 1 model input");
    std::shared_ptr<nic::InferContext::Input> input = inputs[0];
    FAIL_IF_ERR(input->Reset(), "unable to reset INPUT0");

    std::vector<int64_t> input_shape{tensor_size_};
    FAIL_IF_ERR(input->SetShape(input_shape), "unable to set shape for input");
    for (uint32_t b = 0; b < batch_size_; ++b) {
      FAIL_IF_ERR(
          input->SetRaw(
              reinterpret_cast<const uint8_t*>(&input_data_[0]),
              input_data_.size() * sizeof(float)),
          "unable to set data for input");
    }
  }

 private:
  const std::string protocol_;
  const std::string url_;
  const std::string model_name_;
  const uint32_t batch_size_;
  const uint32_t tensor_size_;
  const bool verbose_;

  // The memory block for the input data. We don't bother to
  // initialize it. Set every input tensor to use this data
  std::vector<float> input_data_;
};

void
RunSyncSerial(
    nic::InferContext* ctx, const uint32_t iters, const uint32_t start_window,
    const uint32_t end_window, uint64_t* t_total_duration_ns,
    std::vector<uint64_t>* request_duration_ns)
{
  struct timespec start, end;

  for (uint32_t iter = 0; iter < iters; iter++) {
    std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;

    if (iter == start_window) {
      clock_gettime(CLOCK_MONOTONIC, &start);
    }

    struct timespec request_start;
    clock_gettime(CLOCK_MONOTONIC, &request_start);

    FAIL_IF_ERR(ctx->Run(&results), "unable to run");

    struct timespec request_end;
    clock_gettime(CLOCK_MONOTONIC, &request_end);

    if (iter == end_window) {
      clock_gettime(CLOCK_MONOTONIC, &end);
    }

    // We expect there to be 1 result.
    FAIL_IF(
        results.size() != 1,
        "expected 1 result, got " + std::to_string(results.size()));

    if (request_duration_ns != nullptr && iter >= start_window &&
        iter <= end_window) {
      uint64_t start_ns = TIMESPEC_TO_NANOS(request_start);
      uint64_t end_ns = TIMESPEC_TO_NANOS(request_end);
      request_duration_ns->push_back(end_ns - start_ns);
    }
  }

  uint64_t start_ns = TIMESPEC_TO_NANOS(start);
  uint64_t end_ns = TIMESPEC_TO_NANOS(end);
  *t_total_duration_ns = end_ns - start_ns;
}

void
RunSyncConcurrent(
    const InferContextFactory& factory, const uint32_t iters,
    const uint32_t concurrency, std::vector<uint64_t>* total_duration_ns,
    std::vector<uint64_t>* request_duration_ns)
{
  // Create 'concurrency' threads, each of which will send 'iters'
  // requests. Use a barrier so that all threads start working at the
  // same time.
  auto barrier = std::make_shared<Barrier>(concurrency);

  std::vector<std::unique_ptr<std::thread>> threads;
  std::vector<std::unique_ptr<std::vector<uint64_t>>> threads_duration_ns(
      concurrency);
  std::vector<uint64_t> threads_total_duration_ns(concurrency);

  // We will be selecting 'iters/concurrency' readings from the center of
  // each thread to finally report the stable results.
  uint64_t start_window = (iters * (concurrency - 1)) / (2 * concurrency);
  uint64_t end_window = start_window + (iters / concurrency) - 1;

  for (uint32_t t = 0; t < concurrency; ++t) {
    std::vector<uint64_t>* t_duration_ns = new std::vector<uint64_t>();
    threads_duration_ns[t].reset(t_duration_ns);

    uint64_t* t_total_duration_ns = &threads_total_duration_ns[t];
    *t_total_duration_ns = 0;

    threads.emplace_back(
        new std::thread([barrier, factory, iters, start_window, end_window,
                         t_total_duration_ns, t_duration_ns] {
          std::unique_ptr<nic::InferContext> ctx;
          factory.Create(&ctx);

          barrier->Wait();

          RunSyncSerial(
              ctx.get(), iters, start_window, end_window, t_total_duration_ns,
              t_duration_ns);
        }));
  }

  // Wait for all threads to finish.
  for (const auto& thread : threads) {
    thread->join();
  }

  // Collect all request durations.
  if (request_duration_ns != nullptr) {
    for (const auto& td : threads_duration_ns) {
      request_duration_ns->insert(
          request_duration_ns->end(), td->begin(), td->end());
    }
  }

  if (total_duration_ns != nullptr) {
    total_duration_ns->insert(
        total_duration_ns->end(), threads_total_duration_ns.begin(),
        threads_total_duration_ns.end());
  }
}

void
RunAsyncComplete(
    nic::InferContext* ctx,
    const std::shared_ptr<nic::InferContext::Request>& request, sem_t* sem,
    std::vector<uint64_t>::iterator iter)
{
  // We include getting the results in the timing since that is
  // included in the sync case as well.
  std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;
  ctx->GetAsyncRunResults(request, &results);

  FAIL_IF(
      results.size() != 1,
      "expected 1 result, got " + std::to_string(results.size()));

  struct timespec end;
  clock_gettime(CLOCK_MONOTONIC, &end);
  uint64_t end_ns = TIMESPEC_TO_NANOS(end);
  *iter = end_ns - *iter;

  sem_post(sem);
}

void
RunAsyncConcurrent(
    const InferContextFactory& factory, const uint32_t iters,
    const uint32_t concurrency, std::vector<uint64_t>* total_duration_ns,
    std::vector<uint64_t>* request_duration_ns)
{
  std::unique_ptr<nic::InferContext> ctx;
  factory.Create(&ctx);

  sem_t lsem;
  sem_t* sem = &lsem;
  sem_init(sem, 0, concurrency);

  struct timespec total_start, total_end;
  std::vector<uint64_t> request_duration_buffer_ns(iters * concurrency);

  // We will be selecting 'iters' readings from the center to finally report the
  // stable results with the given concurrency.
  uint64_t start_window = (iters * (concurrency - 1)) / 2;
  uint64_t end_window = start_window + iters - 1;
  bool signal_end;

  uint64_t index = 0;
  for (std::vector<uint64_t>::iterator iter =
           request_duration_buffer_ns.begin();
       iter != request_duration_buffer_ns.end(); ++iter) {
    if (index == start_window) {
      clock_gettime(CLOCK_MONOTONIC, &total_start);
    }

    signal_end = (index == end_window);

    // Wait so that only 'concurrency' requests are in flight at any
    // given time.
    sem_wait(sem);

    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    *iter = TIMESPEC_TO_NANOS(start);

    FAIL_IF_ERR(
        ctx->AsyncRun(
            [sem, iter, signal_end, &total_end](
                nic::InferContext* ctx,
                const std::shared_ptr<nic::InferContext::Request>& request) {
              RunAsyncComplete(ctx, request, sem, iter);
              if (signal_end) {
                clock_gettime(CLOCK_MONOTONIC, &total_end);
              }
            }),
        "unable to async run");

    index += 1;
  }

  // Wait for all the in-flight requests to complete.
  while (true) {
    int sem_value;
    sem_getvalue(sem, &sem_value);
    if (sem_value == (int)concurrency) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // extract the readings from the target window
  request_duration_ns->assign(
      request_duration_buffer_ns.begin() + start_window,
      request_duration_buffer_ns.begin() + end_window);

  if (total_duration_ns != nullptr) {
    uint64_t total_start_ns = TIMESPEC_TO_NANOS(total_start);
    uint64_t total_end_ns = TIMESPEC_TO_NANOS(total_end);
    total_duration_ns->push_back(total_end_ns - total_start_ns);
  }

  sem_destroy(sem);
}

// Return stddev of ns values. Returned value is in
// microseconds. Optionally return the mean in microseconds in
// 'r_mean_us'.
uint64_t
StdDev(const std::vector<uint64_t>& values_ns, uint64_t* r_mean_us = nullptr)
{
  uint64_t sum_ns = 0;
  for (const auto ns : values_ns) {
    sum_ns += ns;
  }

  const uint64_t sum_us = sum_ns / 1000;
  const uint64_t mean_us = sum_us / values_ns.size();
  uint64_t var_us = 0;
  for (size_t n = 0; n < values_ns.size(); n++) {
    uint64_t diff_us = (values_ns[n] / 1000) - mean_us;
    var_us += diff_us * diff_us;
  }

  if (r_mean_us != nullptr) {
    *r_mean_us = mean_us;
  }

  var_us /= values_ns.size();
  return std::sqrt(var_us);
}

void
ShowResults(
    const std::vector<uint64_t>& total_duration_ns,
    const std::vector<uint64_t>& request_duration_ns, const std::string& name,
    const std::string& framework, const std::string& model_name,
    const std::string& protocol, const uint32_t iters,
    const uint32_t concurrency, const uint32_t dynamic_batch_size,
    const uint32_t batch_size, const uint32_t tensor_size,
    const uint32_t instance_count, const bool verbose)
{
  uint64_t request_duration_mean_us;
  uint64_t request_duration_stddev_us =
      StdDev(request_duration_ns, &request_duration_mean_us);

  double total_infer_per_us = 0.0;
  for (const auto ns : total_duration_ns) {
    total_infer_per_us += (iters / total_duration_ns.size()) / (ns / 1000.0);
  }

  std::cout << "{\"s_benchmark_kind\":\"simple_perf\",";
  std::cout << "\"s_benchmark_name\":\"" << name << "\",";
  std::cout << "\"s_protocol\":\"" << protocol << "\",";
  std::cout << "\"s_framework\":\"" << framework << "\",";
  std::cout << "\"s_model\":\"" << model_name << "\",";
  std::cout << "\"l_concurrency\":" << concurrency << ",";
  std::cout << "\"l_dynamic_batch_size\":" << dynamic_batch_size << ",";
  std::cout << "\"l_batch_size\":" << batch_size << ",";
  std::cout << "\"l_instance_count\":" << instance_count << ",";
  std::cout << "\"l_iters\":" << iters << ",";
  std::cout << "\"l_size\":" << tensor_size << ",";
  std::cout << "\"d_infer_per_sec\":" << (total_infer_per_us * 1000.0 * 1000.0)
            << ",";
  std::cout << "\"d_latency_avg_ms\":" << (request_duration_mean_us / 1000.0)
            << ",";
  std::cout << "\"d_latency_avg_stddev_ms\":"
            << (request_duration_stddev_us / 1000.0) << "}";
}

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-i <Protocol used to communicate with inference service>"
            << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-l <benchmark name>" << std::endl;
  std::cerr << "\t-f <model framework>" << std::endl;
  std::cerr << "\t-m <model name>" << std::endl;
  std::cerr << "\t-c <concurrency>" << std::endl;
  std::cerr << "\t-d <dynamic batch size>" << std::endl;
  std::cerr << "\t-x <instance count>" << std::endl;
  std::cerr << "\t-s <inference size>" << std::endl;
  std::cerr << "\t-w <warmup iterations>" << std::endl;
  std::cerr << "\t-n <measurement iterations>" << std::endl;
  std::cerr << "\t-a" << std::endl;
  std::cerr << std::endl;
  std::cerr
      << "For -i, available protocols are 'grpc' and 'http'. Default is 'http."
      << std::endl;
  std::cerr
      << "For -s, specify the input size in fp32 elements. So a value of 8 "
         "indicates that input will be a tensor of 8 fp32 elements. Output "
         "tensor size equals input tensor size. Default is 1."
      << std::endl;

  exit(1);
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  bool async = false;
  std::string benchmark_name;
  std::string protocol = "http";
  std::string url("localhost:8000");
  std::string model_name;
  std::string framework;
  uint32_t concurrency = 1;
  uint32_t dynamic_batch_size = 1;
  uint32_t instance_count = 1;
  uint32_t tensor_size = 1;
  uint32_t warmup_iters = 10;
  uint32_t measure_iters = 10;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "avl:i:u:m:f:c:d:x:s:w:n:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
        break;
      case 'a':
        async = true;
        break;
      case 'l':
        benchmark_name = optarg;
        break;
      case 'i':
        protocol = optarg;
        break;
      case 'u':
        url = optarg;
        break;
      case 'm':
        model_name = optarg;
        break;
      case 'f':
        framework = optarg;
        break;
      case 'c':
        concurrency = std::stoul(optarg);
        break;
      case 'd':
        dynamic_batch_size = std::stoul(optarg);
        break;
      case 'x':
        instance_count = std::stoul(optarg);
        break;
      case 's':
        tensor_size = std::stoul(optarg);
        break;
      case 'w':
        warmup_iters = std::stoul(optarg);
        break;
      case 'n':
        measure_iters = std::stoul(optarg);
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  nic::Error err;
  const uint32_t batch_size = 1;

  if (benchmark_name.empty()) {
    Usage(argv, "-l <benchmark name> must be specified");
  }
  if (model_name.empty()) {
    Usage(argv, "-m <model name> must be specified");
  }

  InferContextFactory factory(
      protocol, url, model_name, batch_size, tensor_size, verbose);

  std::vector<uint64_t> total_duration_ns;
  std::vector<uint64_t> request_duration_ns;

  // Warmup
  RunSyncConcurrent(
      factory, warmup_iters, 1 /* concurrency */,
      nullptr /* total_duration_ns */, nullptr /* request_duration_ns */);

  if (!async) {
    // sync
    total_duration_ns.clear();
    request_duration_ns.clear();
    RunSyncConcurrent(
        factory, measure_iters, concurrency, &total_duration_ns,
        &request_duration_ns);
    ShowResults(
        total_duration_ns, request_duration_ns, benchmark_name, framework,
        model_name, protocol, measure_iters, concurrency, dynamic_batch_size,
        batch_size, tensor_size, instance_count, verbose);
  } else {
    // async
    total_duration_ns.clear();
    request_duration_ns.clear();
    RunAsyncConcurrent(
        factory, measure_iters, concurrency, &total_duration_ns,
        &request_duration_ns);
    ShowResults(
        total_duration_ns, request_duration_ns, benchmark_name, framework,
        model_name, protocol, measure_iters, concurrency, dynamic_batch_size,
        batch_size, tensor_size, instance_count, verbose);
  }

  return 0;
}
