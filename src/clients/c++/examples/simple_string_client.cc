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
#include <iostream>
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

  // We use a simple model that takes 2 input tensors of 16 strings
  // each and returns 2 output tensors of 16 strings each. The input
  // strings must represent integers. One output tensor is the
  // element-wise sum of the inputs and one output is the element-wise
  // difference.
  std::string model_name = "simple_string";

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
  // to unique integers and the second to all ones. The input tensors
  // are the string representation of these values.
  std::vector<std::string> input0_data(16);
  std::vector<std::string> input1_data(16);
  std::vector<int32_t> expected_sum(16);
  std::vector<int32_t> expected_diff(16);
  for (size_t i = 0; i < 16; ++i) {
    input0_data[i] = std::to_string(i);
    input1_data[i] = std::to_string(1);
    expected_sum[i] = i + 1;
    expected_diff[i] = i - 1;
  }

  // Initialize the inputs with the data.
  std::shared_ptr<nic::InferContext::Input> input0, input1;
  FAIL_IF_ERR(ctx->GetInput("INPUT0", &input0), "unable to get INPUT0");
  FAIL_IF_ERR(ctx->GetInput("INPUT1", &input1), "unable to get INPUT1");

  FAIL_IF_ERR(input0->Reset(), "unable to reset INPUT0");
  FAIL_IF_ERR(input1->Reset(), "unable to reset INPUT1");

  FAIL_IF_ERR(
      input0->SetFromString(input0_data), "unable to set data for INPUT0");
  FAIL_IF_ERR(
      input1->SetFromString(input1_data), "unable to set data for INPUT1");

  // Send inference request to the inference server.
  std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;
  FAIL_IF_ERR(ctx->Run(&results), "unable to run model");

  // We expect there to be 2 results. Walk over all 16 result elements
  // and print the sum and difference calculated by the model.
  if (results.size() != 2) {
    std::cerr << "error: expected 2 results, got " << results.size()
              << std::endl;
  }

  for (size_t i = 0; i < 16; ++i) {
    // Read the output values (they are strings).
    std::string sr0, sr1;
    FAIL_IF_ERR(
        results["OUTPUT0"]->GetRawAtCursor(0 /* batch idx */, &sr0),
        "unable to get OUTPUT0 result at idx " + std::to_string(i));
    FAIL_IF_ERR(
        results["OUTPUT1"]->GetRawAtCursor(0 /* batch idx */, &sr1),
        "unable to get OUTPUT1 result at idx " + std::to_string(i));

    std::cout << input0_data[i] << " + " << input1_data[i] << " = " << sr0
              << std::endl;
    std::cout << input0_data[i] << " - " << input1_data[i] << " = " << sr1
              << std::endl;

    // Check correctness by converting output strings to integers.
    const int32_t r0 = std::stoi(sr0);
    const int32_t r1 = std::stoi(sr1);

    if (expected_sum[i] != r0) {
      std::cerr << "error: incorrect sum" << std::endl;
      exit(1);
    }
    if (expected_diff[i] != r1) {
      std::cerr << "error: incorrect difference" << std::endl;
      exit(1);
    }
  }

  return 0;
}
