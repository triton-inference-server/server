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

#include "src/clients/c++/request.h"

#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    nic::Error err = (X);                                          \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

namespace {

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-r" << std::endl;
  std::cerr << "\t-a" << std::endl;
  std::cerr << "\t-u <URL for inference service and its gRPC port>"
            << std::endl;
  std::cerr << std::endl;
  std::cerr << "For -r, the client will run non-streaming context first."
            << std::endl;
  std::cerr << "For -a, the client will send asynchronous requests."
            << std::endl;

  exit(1);
}

int32_t
Send(
    const std::unique_ptr<nic::InferContext>& ctx, int32_t control,
    int32_t value)
{
  // Initialize the inputs with the data.
  std::shared_ptr<nic::InferContext::Input> icontrol, ivalue;
  FAIL_IF_ERR(ctx->GetInput("CONTROL", &icontrol), "unable to get CONTROL");
  FAIL_IF_ERR(ctx->GetInput("INPUT", &ivalue), "unable to get INPUT");

  FAIL_IF_ERR(icontrol->Reset(), "unable to reset CONTROL");
  FAIL_IF_ERR(ivalue->Reset(), "unable to reset INPUT");

  FAIL_IF_ERR(
      icontrol->SetRaw(reinterpret_cast<uint8_t*>(&control), sizeof(int32_t)),
      "unable to set data for CONTROL");
  FAIL_IF_ERR(
      ivalue->SetRaw(reinterpret_cast<uint8_t*>(&value), sizeof(int32_t)),
      "unable to set data for INPUT");

  // Send inference request to the inference server.
  std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;
  FAIL_IF_ERR(ctx->Run(&results), "unable to run model");

  // We expect there to be 1 result value, return it...
  if (results.size() != 1) {
    std::cerr << "error: expected 1 result, got " << results.size()
              << std::endl;
  }

  int32_t r = 0;
  FAIL_IF_ERR(
      results["OUTPUT"]->GetRawAtCursor(0 /* batch idx */, &r),
      "unable to get OUTPUT result");

  return r;
}

std::shared_ptr<nic::InferContext::Request>
AsyncSend(
    const std::unique_ptr<nic::InferContext>& ctx, int32_t control,
    int32_t value)
{
  std::shared_ptr<nic::InferContext::Request> request;
  // Initialize the inputs with the data.
  std::shared_ptr<nic::InferContext::Input> icontrol, ivalue;
  FAIL_IF_ERR(ctx->GetInput("CONTROL", &icontrol), "unable to get CONTROL");
  FAIL_IF_ERR(ctx->GetInput("INPUT", &ivalue), "unable to get INPUT");

  FAIL_IF_ERR(icontrol->Reset(), "unable to reset CONTROL");
  FAIL_IF_ERR(ivalue->Reset(), "unable to reset INPUT");

  FAIL_IF_ERR(
      icontrol->SetRaw(reinterpret_cast<uint8_t*>(&control), sizeof(int32_t)),
      "unable to set data for CONTROL");
  FAIL_IF_ERR(
      ivalue->SetRaw(reinterpret_cast<uint8_t*>(&value), sizeof(int32_t)),
      "unable to set data for INPUT");

  // Send inference request to the inference server.
  FAIL_IF_ERR(ctx->AsyncRun(&request), "unable to run model");

  return std::move(request);
}

int32_t
AsyncReceive(
    const std::unique_ptr<nic::InferContext>& ctx,
    std::shared_ptr<nic::InferContext::Request> request)
{
  // Retrieve result of the inference request from the inference server.
  std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;
  FAIL_IF_ERR(
      ctx->GetAsyncRunResults(&results, request, true),
      "unable to get results");

  // We expect there to be 1 result value, return it...
  if (results.size() != 1) {
    std::cerr << "error: expected 1 result, got " << results.size()
              << std::endl;
  }

  int32_t r = 0;
  FAIL_IF_ERR(
      results["OUTPUT"]->GetRawAtCursor(0 /* batch idx */, &r),
      "unable to get OUTPUT result");

  return r;
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  bool async = false;
  bool reverse = false;
  std::string url("localhost:8001");
  std::string protocol = "grpc";

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vrau:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
        break;
      case 'r':
        reverse = true;
        break;
      case 'a':
        async = true;
        break;
      case 'u':
        url = optarg;
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  nic::Error err;

  // We use the custom "sequence" model which takes 2 inputs, one
  // control and one the actual input value. The output is the
  // accumulated value of the inputs. See src/custom/sequence.
  std::string model_name = "simple_sequence";

  // Create 2 inference context with different correlation ID. We will
  // use these to send to sequences of inference requests. Must use a
  // non-zero correlation ID since zero indicates no correlation ID.
  std::unique_ptr<nic::InferContext> ctx0, ctx1, warmup_ctx;
  const ni::CorrelationID correlation_id0 = 1;
  const ni::CorrelationID correlation_id1 = 2;
  const ni::CorrelationID warmup_correlation_id = 3;

  // Create two different contexts, one is using streaming while the other
  // isn't. Then we can compare their difference in sync/async runs
  err = nic::InferGrpcStreamContext::Create(
      &ctx0, correlation_id0, url, model_name, -1 /* model_version */, verbose);
  if (err.IsOk()) {
    err = nic::InferGrpcContext::Create(
        &ctx1, correlation_id1, url, model_name, -1 /* model_version */,
        verbose);
  }
  if (err.IsOk()) {
    err = nic::InferGrpcContext::Create(
        &warmup_ctx, warmup_correlation_id, url, model_name,
        -1 /* model_version */, false);
  }

  if (!err.IsOk()) {
    std::cerr << "error: unable to create inference contexts: " << err
              << std::endl;
    exit(1);
  }

  // Set the context options to do batch-size 1 requests. Also request
  // that the output tensor be returned.
  std::unique_ptr<nic::InferContext::Options> options;
  FAIL_IF_ERR(
      nic::InferContext::Options::Create(&options),
      "unable to create inference options");

  options->SetBatchSize(1);
  for (const auto& output : ctx0->Outputs()) {
    options->AddRawResult(output);
  }

  FAIL_IF_ERR(ctx0->SetRunOptions(*options), "unable to set context 0 options");
  FAIL_IF_ERR(ctx1->SetRunOptions(*options), "unable to set context 1 options");
  FAIL_IF_ERR(
      warmup_ctx->SetRunOptions(*options),
      "unable to set warmup context options");

  // Now send the inference sequences.. FIXME, for now must send the
  // proper control values since TRTIS is not yet doing it.
  //
  std::vector<int32_t> values{11, 7, 5, 3, 1, 0};
  std::vector<int32_t> result0_list;
  std::vector<int32_t> result1_list;
  uint64_t seq0_ns;
  uint64_t seq1_ns;

  // Warmup the server to avoid time difference due to run order
  Send(warmup_ctx, 1, 0);
  for (int32_t v : values) {
    Send(warmup_ctx, 0, v);
  }

  // Record the time of the sequence
  struct timespec start_time;
  struct timespec end_time;
  uint64_t NANOS = 1000000000;

  std::vector<std::unique_ptr<nic::InferContext>> ctxs;
  if (!reverse) {
    ctxs.emplace_back(std::move(ctx0));
    ctxs.emplace_back(std::move(ctx1));
  } else {
    ctxs.emplace_back(std::move(ctx1));
    ctxs.emplace_back(std::move(ctx0));
  }

  if (async) {
    std::vector<std::shared_ptr<nic::InferContext::Request>> request0_list;
    std::vector<std::shared_ptr<nic::InferContext::Request>> request1_list;

    clock_gettime(CLOCK_MONOTONIC, &start_time);
    // Send requests, first reset accumulator for the sequence.
    request0_list.emplace_back(AsyncSend(ctxs[0], 1, 0));
    // Now send a sequence of values...
    for (int32_t v : values) {
      request0_list.emplace_back(AsyncSend(ctxs[0], 0, v));
    }
    // Get results
    for (size_t i = 0; i < request0_list.size(); i++) {
      result0_list.push_back(AsyncReceive(ctxs[0], request0_list[i]));
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    seq0_ns = (end_time.tv_sec * NANOS + end_time.tv_nsec) -
              (start_time.tv_sec * NANOS + start_time.tv_nsec);

    // Repeat above but in a different Inference Context
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    request1_list.emplace_back(AsyncSend(ctxs[1], 1, 0));
    for (int32_t v : values) {
      request1_list.emplace_back(AsyncSend(ctxs[1], 0, v));
    }
    for (size_t i = 0; i < request1_list.size(); i++) {
      result1_list.push_back(AsyncReceive(ctxs[1], request1_list[i]));
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    seq1_ns = (end_time.tv_sec * NANOS + end_time.tv_nsec) -
              (start_time.tv_sec * NANOS + start_time.tv_nsec);
  } else {
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    // Send requests, first reset accumulator for the sequence.
    result0_list.push_back(Send(ctxs[0], 1, 0));
    // Now send a sequence of values...
    for (int32_t v : values) {
      result0_list.push_back(Send(ctxs[0], 0, v));
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    seq0_ns = (end_time.tv_sec * NANOS + end_time.tv_nsec) -
              (start_time.tv_sec * NANOS + start_time.tv_nsec);

    // Repeat above but in a different Inference Context
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    result1_list.push_back(Send(ctxs[1], 1, 0));
    for (int32_t v : values) {
      result1_list.push_back(Send(ctxs[1], 0, v));
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    seq1_ns = (end_time.tv_sec * NANOS + end_time.tv_nsec) -
              (start_time.tv_sec * NANOS + start_time.tv_nsec);
  }

  if (!reverse) {
    std::cout << "streaming : non-streaming" << std::endl;
  } else {
    std::cout << "non-streaming : streaming" << std::endl;
  }
  std::cout << seq0_ns << " ns : " << seq1_ns << " ns" << std::endl;
  std::cout << "Results" << std::endl;
  for (size_t i = 0; i < result0_list.size(); i++) {
    std::cout << "[" << i << "] " << result0_list[i] << " : " << result1_list[i]
              << std::endl;
  }

  return 0;
}
