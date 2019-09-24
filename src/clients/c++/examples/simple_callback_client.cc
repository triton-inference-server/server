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
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <string>
#include "src/clients/c++/library/request_grpc.h"
#include "src/clients/c++/library/request_http.h"

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
  std::cerr << "\t-i <Protocol used to communicate with inference service>"
            << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << std::endl;
  std::cerr
      << "For -i, available protocols are 'grpc' and 'http'. Default is 'http."
      << std::endl;

  exit(1);
}

void
ValidateResults(
    nic::InferContext* ctx,
    const std::shared_ptr<nic::InferContext::Request>& request,
    const std::vector<int32_t>& input0_data,
    const std::vector<int32_t>& input1_data)
{
  bool is_ready = false;
  std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;
  ctx->GetAsyncRunResults(&results, &is_ready, request, false);
  if (is_ready == false) {
    std::cerr << "Callback is called while request is not ready" << std::endl;
    exit(1);
  }
  // We expect there to be 2 results. Walk over all 16 result elements
  // and print the sum and difference calculated by the model.
  if (results.size() != 2) {
    std::cerr << "error: expected 2 results, got " << results.size()
              << std::endl;
    exit(1);
  }

  for (size_t i = 0; i < 16; ++i) {
    int32_t r0, r1;
    FAIL_IF_ERR(
        results["OUTPUT0"]->GetRawAtCursor(0 /* batch idx */, &r0),
        "unable to get OUTPUT0 result at idx " + std::to_string(i));
    FAIL_IF_ERR(
        results["OUTPUT1"]->GetRawAtCursor(0 /* batch idx */, &r1),
        "unable to get OUTPUT1 result at idx " + std::to_string(i));
    std::cout << input0_data[i] << " + " << input1_data[i] << " = " << r0
              << std::endl;
    std::cout << input0_data[i] << " - " << input1_data[i] << " = " << r1
              << std::endl;

    if ((input0_data[i] + input1_data[i]) != r0) {
      std::cerr << "error: incorrect sum" << std::endl;
      exit(1);
    }
    if ((input0_data[i] - input1_data[i]) != r1) {
      std::cerr << "error: incorrect difference" << std::endl;
      exit(1);
    }
  }
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  std::string url("localhost:8000");
  std::string protocol = "http";

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vi:u:")) != -1) {
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
      case '?':
        Usage(argv);
        break;
    }
  }

  nic::Error err;

  // We use a simple model that takes 2 input tensors of 16 integers
  // each and returns 2 output tensors of 16 integers each. One output
  // tensor is the element-wise sum of the inputs and one output is
  // the element-wise difference.
  std::string model_name = "simple";

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

  // Set the context options to do batch-size 1 requests. Also request
  // that all output tensors be returned.
  std::unique_ptr<nic::InferContext::Options> options;
  FAIL_IF_ERR(
      nic::InferContext::Options::Create(&options),
      "unable to create inference options");

  options->SetBatchSize(1);
  for (const auto& output : ctx->Outputs()) {
    options->AddRawResult(output);
  }

  FAIL_IF_ERR(ctx->SetRunOptions(*options), "unable to set inference options");

  // Create the data for the two input tensors. Initialize the first
  // to unique integers and the second to all ones.
  std::vector<int32_t> input0_data(16);
  std::vector<int32_t> input1_data(16);
  for (size_t i = 0; i < 16; ++i) {
    input0_data[i] = i;
    input1_data[i] = 1;
  }

  // Initialize the inputs with the data.
  std::shared_ptr<nic::InferContext::Input> input0, input1;
  FAIL_IF_ERR(ctx->GetInput("INPUT0", &input0), "unable to get INPUT0");
  FAIL_IF_ERR(ctx->GetInput("INPUT1", &input1), "unable to get INPUT1");

  FAIL_IF_ERR(input0->Reset(), "unable to reset INPUT0");
  FAIL_IF_ERR(input1->Reset(), "unable to reset INPUT1");

  FAIL_IF_ERR(
      input0->SetRaw(
          reinterpret_cast<uint8_t*>(&input0_data[0]),
          input0_data.size() * sizeof(int32_t)),
      "unable to set data for INPUT0");
  FAIL_IF_ERR(
      input1->SetRaw(
          reinterpret_cast<uint8_t*>(&input1_data[0]),
          input1_data.size() * sizeof(int32_t)),
      "unable to set data for INPUT1");

  // Send inference request to the inference server.
  std::mutex mtx;
  std::condition_variable cv;
  size_t repeat_cnt = 2;
  size_t done_cnt = 0;
  for (size_t i = 0; i < repeat_cnt; i++) {
    FAIL_IF_ERR(
        ctx->AsyncRun([&, i](
                          nic::InferContext* ctx,
                          std::shared_ptr<nic::InferContext::Request> request) {
          std::lock_guard<std::mutex> lk(mtx);
          std::cout << "Callback no." << i << " is called" << std::endl;
          done_cnt++;
          ValidateResults(ctx, request, input0_data, input1_data);
          cv.notify_all();
          return;
        }),
        "unable to run model");
  }

  // Wait until all callbacks are invoked
  {
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [&]() {
      if (done_cnt >= repeat_cnt) {
        return true;
      } else {
        return false;
      }
    });
  }
  if (done_cnt == repeat_cnt) {
    std::cout << "All done" << std::endl;
  } else {
    std::cerr << "Done cnt: " << done_cnt
              << " does not match repeat cnt: " << repeat_cnt << std::endl;
    exit(1);
  }

  // Send another AsyncRun with callback which will defer the completed request
  // to another thread (main thread) to handle
  bool callback_invoked = false;
  std::shared_ptr<nic::InferContext::Request> request_placeholder;
  FAIL_IF_ERR(
      ctx->AsyncRun([&](nic::InferContext* ctx,
                        std::shared_ptr<nic::InferContext::Request> request) {
        // Defer the response retrieval to main thread
        std::lock_guard<std::mutex> lk(mtx);
        callback_invoked = true;
        request_placeholder = std::move(request);
        cv.notify_all();
        return;
      }),
      "unable to run model");

  std::shared_ptr<nic::InferContext::Request> request;
  bool is_ready = false;
  nic::Error error = ctx->GetReadyAsyncRequest(&request, &is_ready, false);
  if (error.IsOk()) {
    std::cerr << "Expecting error on GetReadyAsyncRequest" << std::endl;
    exit(1);
  } else if (
      error.Message() !=
      "No asynchronous requests can be returned, all outstanding requests "
      "will signal completion via their callback function") {
    std::cerr
        << "Expecting different error message on GetReadyAsyncRequest, got: "
        << error.Message() << std::endl;
    exit(1);
  }

  // Ensure callback is completed
  {
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [&]() { return callback_invoked; });
  }

  // Get deferred response
  std::cout << "Getting results from deferred response" << std::endl;
  ValidateResults(ctx.get(), request_placeholder, input0_data, input1_data);

  // Check again, should return different error message
  error = ctx->GetReadyAsyncRequest(&request, &is_ready, false);
  if (error.IsOk()) {
    std::cerr << "Expecting error on GetReadyAsyncRequest" << std::endl;
    exit(1);
  } else if (error.Message() != "No asynchronous requests have been sent") {
    std::cerr
        << "Expecting different error message on GetReadyAsyncRequest, got: "
        << error.Message() << std::endl;
    exit(1);
  }

  return 0;
}
