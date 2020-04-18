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

#include <unistd.h>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <string>
#include "src/clients/c++/experimental_api_v2/library/grpc_client.h"

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
ValidateShapeAndDatatype(
    const std::string& name, std::shared_ptr<nic::InferResult> result)
{
  std::vector<int64_t> shape;
  FAIL_IF_ERR(result->Shape(name, &shape), "unable to get shape for " + name);
  // Validate shape
  if ((shape.size() != 2) || (shape[0] != 1) || (shape[1] != 16)) {
    std::cerr << "error: received incorrect shapes for " << name << std::endl;
    exit(1);
  }
  std::string datatype;
  FAIL_IF_ERR(
      result->Datatype(name, &datatype), "unable to get datatype for " + name);
  // Validate datatype
  if (datatype.compare("INT32") != 0) {
    std::cerr << "error: received incorrect datatype for " << name << ": "
              << datatype << std::endl;
    exit(1);
  }
}

void
ValidateResult(
    const std::shared_ptr<nic::InferResult> result,
    std::vector<int32_t>& input0_data, std::vector<int32_t>& input1_data)
{
  // Validate the results...
  ValidateShapeAndDatatype("OUTPUT0", result);
  ValidateShapeAndDatatype("OUTPUT1", result);

  // Get pointers to the result returned...
  int32_t* output0_data;
  size_t output0_byte_size;
  FAIL_IF_ERR(
      result->RawData(
          "OUTPUT0", (const uint8_t**)&output0_data, &output0_byte_size),
      "unable to get datatype for OUTPUT0");
  if (output0_byte_size != 64) {
    std::cerr << "error: received incorrect byte size for OUTPUT0: "
              << output0_byte_size << std::endl;
    exit(1);
  }

  int32_t* output1_data;
  size_t output1_byte_size;
  FAIL_IF_ERR(
      result->RawData(
          "OUTPUT1", (const uint8_t**)&output1_data, &output1_byte_size),
      "unable to get datatype for OUTPUT1");
  if (output0_byte_size != 64) {
    std::cerr << "error: received incorrect byte size for OUTPUT1: "
              << output0_byte_size << std::endl;
    exit(1);
  }

  for (size_t i = 0; i < 16; ++i) {
    std::cout << input0_data[i] << " + " << input1_data[i] << " = "
              << *(output0_data + i) << std::endl;
    std::cout << input0_data[i] << " - " << input1_data[i] << " = "
              << *(output1_data + i) << std::endl;

    if ((input0_data[i] + input1_data[i]) != *(output0_data + i)) {
      std::cerr << "error: incorrect sum" << std::endl;
      exit(1);
    }
    if ((input0_data[i] - input1_data[i]) != *(output1_data + i)) {
      std::cerr << "error: incorrect difference" << std::endl;
      exit(1);
    }
  }

  // Get full response
  std::cout << result->DebugString() << std::endl;
}

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-H <HTTP header>" << std::endl;
  std::cerr << std::endl;
  std::cerr
      << "For -H, header must be 'Header:Value'. May be given multiple times."
      << std::endl;

  exit(1);
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  std::string url("localhost:8001");
  nic::Headers http_headers;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vu:H:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
        break;
      case 'u':
        url = optarg;
        break;
      case 'H': {
        std::string arg = optarg;
        std::string header = arg.substr(0, arg.find(":"));
        http_headers[header] = arg.substr(header.size() + 1);
        break;
      }
      case '?':
        Usage(argv);
        break;
    }
  }

  // We use a simple model that takes 2 input tensors of 16 integers
  // each and returns 2 output tensors of 16 integers each. One output
  // tensor is the element-wise sum of the inputs and one output is
  // the element-wise difference.
  std::string model_name = "simple";
  std::string model_version = "";

  // Create a InferenceServerGrpcClient instance to communicate with the
  // server using gRPC protocol.
  std::unique_ptr<nic::InferenceServerGrpcClient> client;
  FAIL_IF_ERR(
      nic::InferenceServerGrpcClient::Create(&client, url, verbose),
      "unable to create grpc client");

  // Create the data for the two input tensors. Initialize the first
  // to unique integers and the second to all ones.
  std::vector<int32_t> input0_data(16);
  std::vector<int32_t> input1_data(16);
  for (size_t i = 0; i < 16; ++i) {
    input0_data[i] = i;
    input1_data[i] = 1;
  }

  std::vector<int64_t> shape{1, 16};

  // Initialize the inputs with the data.
  nic::InferInput* input0;
  nic::InferInput* input1;

  FAIL_IF_ERR(
      nic::InferInput::Create(&input0, "INPUT0", shape, "INT32"),
      "unable to get INPUT0");
  std::shared_ptr<nic::InferInput> input0_ptr;
  input0_ptr.reset(input0);
  FAIL_IF_ERR(
      nic::InferInput::Create(&input1, "INPUT1", shape, "INT32"),
      "unable to get INPUT1");
  std::shared_ptr<nic::InferInput> input1_ptr;
  input1_ptr.reset(input1);

  FAIL_IF_ERR(
      input0_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&input0_data[0]),
          input0_data.size() * sizeof(int32_t)),
      "unable to set data for INPUT0");
  FAIL_IF_ERR(
      input1_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&input1_data[0]),
          input1_data.size() * sizeof(int32_t)),
      "unable to set data for INPUT1");

  // Generate the outputs to be requested.
  nic::InferRequestedOutput* output0;
  nic::InferRequestedOutput* output1;

  FAIL_IF_ERR(
      nic::InferRequestedOutput::Create(&output0, "OUTPUT0"),
      "unable to get OUTPUT0");
  std::shared_ptr<nic::InferRequestedOutput> output0_ptr;
  output0_ptr.reset(output0);
  FAIL_IF_ERR(
      nic::InferRequestedOutput::Create(&output1, "OUTPUT1"),
      "unable to get OUTPUT1");
  std::shared_ptr<nic::InferRequestedOutput> output1_ptr;
  output1_ptr.reset(output1);


  // The inference settings. Will be using default for now.
  nic::InferOptions options(model_name);
  options.model_version_ = model_version;

  std::vector<nic::InferInput*> inputs = {input0_ptr.get(), input1_ptr.get()};
  std::vector<const nic::InferRequestedOutput*> outputs = {output0_ptr.get(),
                                                           output1_ptr.get()};

  // Send inference request to the inference server.
  std::mutex mtx;
  std::condition_variable cv;
  size_t repeat_cnt = 2;
  size_t done_cnt = 0;
  for (size_t i = 0; i < repeat_cnt; i++) {
    FAIL_IF_ERR(
        client->AsyncInfer(
            [&, i](nic::InferResult* result) {
              {
                std::shared_ptr<nic::InferResult> result_ptr;
                result_ptr.reset(result);
                std::lock_guard<std::mutex> lk(mtx);
                std::cout << "Callback no." << i << " is called" << std::endl;
                done_cnt++;
                if (result_ptr->RequestStatus().IsOk()) {
                  ValidateResult(result_ptr, input0_data, input1_data);
                } else {
                  std::cerr << "error: Inference failed: "
                            << result_ptr->RequestStatus() << std::endl;
                  exit(1);
                }
              }
              cv.notify_all();
            },
            options, inputs, outputs, http_headers),
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

  // Send another AsyncInfer whose callback defers the completed request
  // to another thread (main thread) to handle
  bool callback_invoked = false;
  std::shared_ptr<nic::InferResult> result_placeholder;
  FAIL_IF_ERR(
      client->AsyncInfer(
          [&](nic::InferResult* result) {
            {
              std::shared_ptr<nic::InferResult> result_ptr;
              result_ptr.reset(result);
              // Defer the response retrieval to main thread
              std::lock_guard<std::mutex> lk(mtx);
              callback_invoked = true;
              result_placeholder = std::move(result_ptr);
            }
            cv.notify_all();
          },
          options, inputs, outputs, http_headers),
      "unable to run model");

  // Ensure callback is completed
  {
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [&]() { return callback_invoked; });
  }

  // Get deferred response
  std::cout << "Getting results from deferred response" << std::endl;
  if (result_placeholder->RequestStatus().IsOk()) {
    ValidateResult(result_placeholder, input0_data, input1_data);
  } else {
    std::cerr << "error: Inference failed: "
              << result_placeholder->RequestStatus() << std::endl;
    exit(1);
  }

  nic::InferStat infer_stat;
  client->ClientInferStat(&infer_stat);
  std::cout << "completed_request_count " << infer_stat.completed_request_count
            << std::endl;
  std::cout << "cumulative_total_request_time_ns "
            << infer_stat.cumulative_total_request_time_ns << std::endl;
  std::cout << "cumulative_send_time_ns " << infer_stat.cumulative_send_time_ns
            << std::endl;
  std::cout << "cumulative_receive_time_ns "
            << infer_stat.cumulative_receive_time_ns << std::endl;

  std::cout << "PASS : Async Infer" << std::endl;

  return 0;
}
