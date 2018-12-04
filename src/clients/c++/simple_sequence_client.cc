// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
  std::vector<std::unique_ptr<nic::InferContext::Result>> results;
  FAIL_IF_ERR(ctx->Run(&results), "unable to run model");

  // We expect there to be 1 result value, return it...
  if (results.size() != 1) {
    std::cerr << "error: expected 1 result, got " << results.size()
              << std::endl;
  }

  int32_t r;
  FAIL_IF_ERR(
      results[0]->GetRawAtCursor(0 /* batch idx */, &r),
      "unable to get OUTPUT result");

  return r;
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

  // We use the custom "sequence" model which takes 2 inputs, one
  // control and one the actual input value. The output is the
  // accumulated value of the inputs. See src/custom/sequence.
  std::string model_name = "simple_sequence";

  // Create 2 inference context with different correlation ID. We will
  // use these to send to sequences of inference requests. Must use a
  // non-zero correlation ID since zero indicates no correlation ID.
  std::unique_ptr<nic::InferContext> ctx0, ctx1;
  const ni::CorrelationID correlation_id0 = 1;
  const ni::CorrelationID correlation_id1 = 2;

  if (protocol == "http") {
    err = nic::InferHttpContext::Create(
        &ctx0, correlation_id0, url, model_name, -1 /* model_version */,
        verbose);
    if (err.IsOk()) {
      err = nic::InferHttpContext::Create(
          &ctx1, correlation_id1, url, model_name, -1 /* model_version */,
          verbose);
    }
  } else if (protocol == "grpc") {
    err = nic::InferGrpcContext::Create(
        &ctx0, correlation_id0, url, model_name, -1 /* model_version */,
        verbose);
    if (err.IsOk()) {
      err = nic::InferGrpcContext::Create(
          &ctx1, correlation_id1, url, model_name, -1 /* model_version */,
          verbose);
    }
  } else {
    Usage(argv, "unknown protocol '" + protocol + "'");
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

  // Now send the inference sequences.. FIXME, for now must send the
  // proper control values since TRTIS is not yet doing it.
  //
  // First reset accumulator for both sequences.
  int32_t result0 = Send(ctx0, 1, 0);
  int32_t result1 = Send(ctx1, 1, 100);

  // Now send a sequence of values...
  for (int32_t v : {11, 7, 5, 3, 1, 0}) {
    result0 = Send(ctx0, 0, v);
    result1 = Send(ctx1, 0, -v);
    std::cout << "sequence0 = " << result0 << std::endl;
    std::cout << "sequence1 = " << result1 << std::endl;
  }

  return 0;
}
