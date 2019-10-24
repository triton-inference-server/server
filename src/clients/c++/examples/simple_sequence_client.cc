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

#include <unistd.h>
#include <condition_variable>
#include <iostream>
#include <string>
#include <vector>
#include "src/clients/c++/library/request_grpc.h"
#include "src/clients/c++/library/request_http.h"

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

using RequestList = std::vector<std::shared_ptr<nic::InferContext::Request>>;

// Global mutex to synchronize the threads
std::mutex mutex_;
std::condition_variable cv_;

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
  std::cerr << "\t-o <offset for correlation ID>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "For -r, the client will run non-streaming context first."
            << std::endl;
  std::cerr << "For -a, the client will send asynchronous requests."
            << std::endl;
  std::cerr << "For -o, the client will use correlation ID <1 + 2 * offset> "
            << "and <2 + 2 * offset>. Default offset is 0." << std::endl;

  exit(1);
}

int32_t
Send(
    const std::unique_ptr<nic::InferContext>& ctx, int32_t value,
    const uint64_t correlation_id = 0, bool start_of_sequence = false,
    bool end_of_sequence = false)
{
  // Set the context options to do batch-size 1 requests. Also request
  // that all output tensors be returned.
  std::unique_ptr<nic::InferContext::Options> options;
  FAIL_IF_ERR(
      nic::InferContext::Options::Create(&options),
      "unable to create inference options");

  options->SetFlags(0);
  if (start_of_sequence) {
    options->SetFlag(ni::InferRequestHeader::FLAG_SEQUENCE_START, true);
  }
  if (end_of_sequence) {
    options->SetFlag(ni::InferRequestHeader::FLAG_SEQUENCE_END, true);
  }

  options->SetBatchSize(1);
  options->SetCorrelationId(correlation_id);
  for (const auto& output : ctx->Outputs()) {
    options->AddRawResult(output);
  }

  FAIL_IF_ERR(ctx->SetRunOptions(*options), "unable to set context 0 options");

  // Initialize the inputs with the data.
  std::shared_ptr<nic::InferContext::Input> ivalue;
  FAIL_IF_ERR(ctx->GetInput("INPUT", &ivalue), "unable to get INPUT");
  FAIL_IF_ERR(ivalue->Reset(), "unable to reset INPUT");
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

void
AsyncSend(
    const std::unique_ptr<nic::InferContext>& ctx, int32_t value,
    const uint64_t correlation_id, bool start_of_sequence, bool end_of_sequence,
    RequestList& request_list)
{
  std::shared_ptr<nic::InferContext::Request> request;

  // Set the context options to do batch-size 1 requests. Also request
  // that all output tensors be returned.
  std::unique_ptr<nic::InferContext::Options> options;
  FAIL_IF_ERR(
      nic::InferContext::Options::Create(&options),
      "unable to create inference options");

  options->SetFlags(0);
  if (start_of_sequence) {
    options->SetFlag(ni::InferRequestHeader::FLAG_SEQUENCE_START, true);
  }
  if (end_of_sequence) {
    options->SetFlag(ni::InferRequestHeader::FLAG_SEQUENCE_END, true);
  }

  options->SetBatchSize(1);
  options->SetCorrelationId(correlation_id);
  for (const auto& output : ctx->Outputs()) {
    options->AddRawResult(output);
  }

  FAIL_IF_ERR(ctx->SetRunOptions(*options), "unable to set context 0 options");

  // Initialize the inputs with the data.
  std::shared_ptr<nic::InferContext::Input> ivalue;
  FAIL_IF_ERR(ctx->GetInput("INPUT", &ivalue), "unable to get INPUT");
  FAIL_IF_ERR(ivalue->Reset(), "unable to reset INPUT");
  FAIL_IF_ERR(
      ivalue->SetRaw(reinterpret_cast<uint8_t*>(&value), sizeof(int32_t)),
      "unable to set data for INPUT");

  // Send inference request to the inference server.
  FAIL_IF_ERR(
      ctx->AsyncRun(
          [&](nic::InferContext* ctx,
              const std::shared_ptr<nic::InferContext::Request>& request) {
            {
              std::lock_guard<std::mutex> lk(mutex_);
              request_list.push_back(request);
            }
            cv_.notify_all();
          }),
      "unable to run model");
}

int32_t
AsyncReceive(
    const std::unique_ptr<nic::InferContext>& ctx,
    std::shared_ptr<nic::InferContext::Request> request)
{
  // Retrieve result of the inference request from the inference server.
  std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;
  FAIL_IF_ERR(
      ctx->GetAsyncRunResults(request, &results), "unable to get results");

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
  int correlation_id_offset = 0;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vrau:o:")) != -1) {
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
      case 'o':
        correlation_id_offset = std::stoi(optarg);
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  nic::Error err;

  // We use the custom "sequence" model which takes 1 input value. The
  // output is the accumulated value of the inputs. See
  // src/custom/sequence.
  std::string model_name = "simple_sequence";

  // Create 2 inference context with different correlation ID. We will
  // use these to send to sequences of inference requests. Must use a
  // non-zero correlation ID since zero indicates no correlation ID.
  std::unique_ptr<nic::InferContext> ctx0, ctx1;
  const ni::CorrelationID correlation_id0 = 1 + correlation_id_offset * 2;
  const ni::CorrelationID correlation_id1 = 2 + correlation_id_offset * 2;
  std::cout << "sequence 0 correlation ID " << correlation_id0 << " : "
            << "sequence 1 correlation ID " << correlation_id1 << std::endl;

  // Create two different contexts, in the sync case we can use one
  // streaming and one not streaming. In the async case, use a
  // single streaming context since async+non-streaming means that
  // order of requests reaching inference server is not guaranteed.
  err = nic::InferGrpcStreamContext::Create(
      &ctx0, correlation_id0, url, model_name, -1 /* model_version */, verbose);
  if (err.IsOk()) {
    if (!async) {
      err = nic::InferGrpcContext::Create(
          &ctx1, correlation_id1, url, model_name, -1 /* model_version */,
          verbose);
    }
  }

  if (!err.IsOk()) {
    std::cerr << "error: unable to create inference contexts: " << err
              << std::endl;
    exit(1);
  }

  // Now send the inference sequences..
  //
  std::vector<int32_t> values{11, 7, 5, 3, 2, 0, 1};
  std::vector<int32_t> result0_list;
  std::vector<int32_t> result1_list;

  if (async) {
    RequestList request0_list;
    RequestList request1_list;

    // Send requests, first reset accumulator for the sequence.
    AsyncSend(
        ctx0, 0, correlation_id0, true /* start-of-sequence */,
        false /* end-of-sequence */, request0_list);
    AsyncSend(
        ctx0, 100, correlation_id1, true /* start-of-sequence */,
        false /* end-of-sequence */, request1_list);

    // Now send a sequence of values...
    for (int32_t v : values) {
      AsyncSend(
          ctx0, v, correlation_id0, false /* start-of-sequence */,
          (v == 1) /* end-of-sequence */, request0_list);
      AsyncSend(
          ctx0, -v, correlation_id1, false /* start-of-sequence */,
          (v == 1) /* end-of-sequence */, request1_list);
    }

    // Wait until all callbacks are invoked
    {
      std::unique_lock<std::mutex> lk(mutex_);
      cv_.wait(lk, [&]() {
        if (request0_list.size() > values.size() &&
            request1_list.size() > values.size()) {
          return true;
        } else {
          return false;
        }
      });
    }
    // Get results
    for (size_t i = 0; i < request0_list.size(); i++) {
      result0_list.push_back(AsyncReceive(ctx0, request0_list[i]));
    }
    for (size_t i = 0; i < request1_list.size(); i++) {
      result1_list.push_back(AsyncReceive(ctx0, request1_list[i]));
    }
  } else {
    std::vector<std::unique_ptr<nic::InferContext>> ctxs;

    if (!reverse) {
      ctxs.emplace_back(std::move(ctx0));
      ctxs.emplace_back(std::move(ctx1));
    } else {
      ctxs.emplace_back(std::move(ctx1));
      ctxs.emplace_back(std::move(ctx0));
    }

    // Send requests, first reset accumulator for the sequence.
    result0_list.push_back(Send(ctxs[0], 0, 0, true /* start-of-sequence */));

    // Now send a sequence of values...
    for (int32_t v : values) {
      result0_list.push_back(Send(
          ctxs[0], v, 0, false /* start-of-sequence */,
          (v == 1) /* end-of-sequence */));
    }

    // Different from asynchronous setting, requests are sent sequence by
    // sequence because the client will be blocked on each request. In the case
    // where one sequence is waiting for available slot while the other sequence
    // has started the sequence, the other sequence may be terminated due to
    // idleness.
    result1_list.push_back(Send(ctxs[1], 100, 0, true /* start-of-sequence */));
    for (int32_t v : values) {
      result1_list.push_back(Send(
          ctxs[1], -v, 0, false /* start-of-sequence */,
          (v == 1) /* end-of-sequence */));
    }
  }

  if (async) {
    std::cout << "streaming : streaming" << std::endl;
  } else if (!reverse) {
    std::cout << "streaming : non-streaming" << std::endl;
  } else {
    std::cout << "non-streaming : streaming" << std::endl;
  }

  int32_t seq0_expected = 0;
  int32_t seq1_expected = 100;

  for (size_t i = 0; i < result0_list.size(); i++) {
    std::cout << "[" << i << "] " << result0_list[i] << " : " << result1_list[i]
              << std::endl;

    if ((seq0_expected != result0_list[i]) ||
        (seq1_expected != result1_list[i])) {
      std::cout << "[ expected ] " << seq0_expected << " : " << seq1_expected
                << std::endl;
      return 1;
    }

    if (i < values.size()) {
      seq0_expected += values[i];
      seq1_expected -= values[i];
    }
  }

  return 0;
}
