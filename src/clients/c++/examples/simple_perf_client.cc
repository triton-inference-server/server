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

#include <unistd.h>
#include <cmath>
#include <future>
#include <iostream>
#include <string>
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
    }                                               \
  }

namespace {

void
RunSyncSerial(
    nic::InferContext* ctx, const uint32_t iters,
    std::vector<uint64_t>* duration_ns)
{
  if (duration_ns != nullptr) {
    duration_ns->clear();
  }

  for (uint32_t iter = 0; iter < iters; iter++) {
    std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;

    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);

    FAIL_IF_ERR(ctx->Run(&results), "unable to run");

    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);

    // We expect there to be 1 result.
    FAIL_IF(
        results.size() != 1,
        "expected 1 result, got " + std::to_string(results.size()));

    if (duration_ns != nullptr) {
      uint64_t start_ns = TIMESPEC_TO_NANOS(start);
      uint64_t end_ns = TIMESPEC_TO_NANOS(end);
      duration_ns->push_back(end_ns - start_ns);
    }
  }
}

void
RunAsyncComplete(
    nic::InferContext* ctx, std::shared_ptr<nic::InferContext::Request> request,
    std::promise<uint64_t>* p)
{
  // We include getting the results in the timing since that is
  // included in the sync case as well.
  bool is_ready = false;
  std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;
  ctx->GetAsyncRunResults(&results, &is_ready, request, false);
  FAIL_IF(!is_ready, "callback invoked when request is not ready");

  FAIL_IF(
      results.size() != 1,
      "expected 1 result, got " + std::to_string(results.size()));

  struct timespec end;
  clock_gettime(CLOCK_MONOTONIC, &end);
  uint64_t end_ns = TIMESPEC_TO_NANOS(end);

  p->set_value(end_ns);
  delete p;
}

void
RunAsyncSerial(
    nic::InferContext* ctx, const uint32_t iters,
    std::vector<uint64_t>* duration_ns)
{
  if (duration_ns != nullptr) {
    duration_ns->clear();
  }

  for (uint32_t iter = 0; iter < iters; iter++) {
    auto p = new std::promise<uint64_t>();
    std::future<uint64_t> completed = p->get_future();

    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);

    FAIL_IF_ERR(
        ctx->AsyncRun([p](nic::InferContext* ctx,
                          std::shared_ptr<nic::InferContext::Request> request) {
          RunAsyncComplete(ctx, request, p);
        }),
        "unable to async run");

    uint64_t end_ns = completed.get();
    if (duration_ns != nullptr) {
      uint64_t start_ns = TIMESPEC_TO_NANOS(start);
      duration_ns->push_back(end_ns - start_ns);
    }
  }
}

void
ShowResults(
    const std::vector<uint64_t>& duration_ns, const std::string& name,
    const std::string& protocol, const bool verbose)
{
  uint64_t sum_ns = 0;
  for (const auto ns : duration_ns) {
    if (verbose) {
      std::cout << ((ns / 1000) / 1000.0) << " ms" << std::endl;
    }

    sum_ns += ns;
  }

  const uint64_t sum_us = sum_ns / 1000;
  const uint64_t mean_us = sum_us / duration_ns.size();
  uint64_t var_us = 0;
  for (size_t n = 0; n < duration_ns.size(); n++) {
    uint64_t diff_us = (duration_ns[n] / 1000) - mean_us;
    var_us += diff_us * diff_us;
  }

  var_us /= duration_ns.size();
  uint64_t stddev_us = std::sqrt(var_us);

  std::cout << name << " (" << protocol << ")" << std::endl;
  std::cout << "Total time: " << (sum_us / 1000.0) << " ms" << std::endl;
  std::cout << "Mean time: " << (mean_us / 1000.0) << " ms" << std::endl;
  std::cout << "Stddev: " << (stddev_us / 1000.0) << " ms" << std::endl;
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
  std::cerr << "\t-m <model name>" << std::endl;
  std::cerr << "\t-b <batch size>" << std::endl;
  std::cerr << "\t-s <inference size>" << std::endl;
  std::cerr << "\t-w <warmup iterations>" << std::endl;
  std::cerr << "\t-n <measurement iterations>" << std::endl;
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
  std::string url("localhost:8000");
  std::string protocol = "http";
  std::string model_name;
  uint32_t batch_size = 1;
  uint32_t tensor_size = 1;
  uint32_t warmup_iters = 10;
  uint32_t measure_iters = 10;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vi:u:m:b:s:w:n:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
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
      case 'b':
        batch_size = std::stoul(optarg);
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

  if (model_name.empty()) {
    Usage(argv, "-m <model name> must be specified");
  }

  // Create the inference context for the model.
  std::unique_ptr<nic::InferContext> ctx;
  if (protocol == "http") {
    err = nic::InferHttpContext::Create(
        &ctx, url, model_name, -1 /* model_version */, verbose);
  } else if (protocol == "grpc") {
    err = nic::InferGrpcContext::Create(
        &ctx, url, model_name, -1 /* model_version */, verbose);
  } else {
    Usage(argv, "unknown protocol '" + protocol + "'");
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

  options->SetBatchSize(batch_size);
  for (const auto& output : ctx->Outputs()) {
    options->AddRawResult(output);
  }

  FAIL_IF_ERR(ctx->SetRunOptions(*options), "unable to set inference options");

  // Create a memory block for the input data. We don't bother to
  // initialize it. Set the input tensor to use that data.
  std::vector<float> input_data(tensor_size);

  const std::vector<std::shared_ptr<nic::InferContext::Input>>& inputs =
      ctx->Inputs();
  FAIL_IF(inputs.size() != 1, "expected 1 model input");
  std::shared_ptr<nic::InferContext::Input> input = inputs[0];
  FAIL_IF_ERR(input->Reset(), "unable to reset INPUT0");

  std::vector<int64_t> input_shape{tensor_size};
  FAIL_IF_ERR(input->SetShape(input_shape), "unable to set shape for input");
  FAIL_IF_ERR(
      input->SetRaw(
          reinterpret_cast<uint8_t*>(&input_data[0]),
          input_data.size() * sizeof(float)),
      "unable to set data for input");

  std::vector<uint64_t> duration_ns;

  // Warmup
  RunSyncSerial(ctx.get(), warmup_iters, nullptr /* duration_ns */);

  // Test sync serial
  RunSyncSerial(ctx.get(), measure_iters, &duration_ns);
  ShowResults(duration_ns, "Sync Serial Run", protocol, verbose);

  // Test async serial
  RunAsyncSerial(ctx.get(), measure_iters, &duration_ns);
  ShowResults(duration_ns, "Async Serial Run", protocol, verbose);

  return 0;
}
