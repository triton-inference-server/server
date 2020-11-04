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
#include "src/clients/c++/library/grpc_client.h"
#include "src/clients/c++/library/http_client.h"

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
  FAIL_IF_ERR(
      result->Shape(name, &shape), "unable to get shape for '" + name + "'");
  // Validate shape
  if ((shape.size() != 2) || (shape[0] != 1) || (shape[1] != 16)) {
    std::cerr << "error: received incorrect shapes for '" << name << "'"
              << std::endl;
    exit(1);
  }
  std::string datatype;
  FAIL_IF_ERR(
      result->Datatype(name, &datatype),
      "unable to get datatype for '" + name + "'");
  // Validate datatype
  if (datatype.compare("INT32") != 0) {
    std::cerr << "error: received incorrect datatype for '" << name
              << "': " << datatype << std::endl;
    exit(1);
  }
}

void
ValidateResult(
    const std::shared_ptr<nic::InferResult> result,
    std::vector<int32_t>& input0_data)
{
  // Validate the results...
  ValidateShapeAndDatatype("OUTPUT0", result);

  // Get pointers to the result returned...
  int32_t* output0_data;
  size_t output0_byte_size;
  FAIL_IF_ERR(
      result->RawData(
          "OUTPUT0", (const uint8_t**)&output0_data, &output0_byte_size),
      "unable to get result data for 'OUTPUT0'");
  if (output0_byte_size != 64) {
    std::cerr << "error: received incorrect byte size for 'OUTPUT0': "
              << output0_byte_size << std::endl;
    exit(1);
  }

  for (size_t i = 0; i < 16; ++i) {
    if ((input0_data[i]) != *(output0_data + i)) {
      std::cerr << "error: incorrect output" << std::endl;
      exit(1);
    }
  }

  // Get full response
  std::cout << result->DebugString() << std::endl;
}


void
RunSynchronousInference(
    std::unique_ptr<nic::InferenceServerGrpcClient>& grpc_client,
    std::unique_ptr<nic::InferenceServerHttpClient>& http_client,
    uint32_t client_timeout, std::vector<nic::InferInput*>& inputs,
    std::vector<const nic::InferRequestedOutput*>& outputs,
    nic::InferOptions& options, std::vector<int32_t>& input0_data)
{
  options.client_timeout_ = client_timeout;
  nic::InferResult* results;
  if (grpc_client.get() != nullptr) {
    FAIL_IF_ERR(
        grpc_client->Infer(&results, options, inputs, outputs),
        "unable to run model");
  } else {
    FAIL_IF_ERR(
        http_client->Infer(&results, options, inputs, outputs),
        "unable to run model");
  }
  std::shared_ptr<nic::InferResult> results_ptr;
  results_ptr.reset(results);

  // Validate results
  if (results_ptr->RequestStatus().IsOk()) {
    ValidateResult(results_ptr, input0_data);
  } else {
    std::cerr << "error: Inference failed: " << results_ptr->RequestStatus()
              << std::endl;
    exit(1);
  }
}

void
RunAsynchronousInference(
    std::unique_ptr<nic::InferenceServerGrpcClient>& grpc_client,
    std::unique_ptr<nic::InferenceServerHttpClient>& http_client,
    uint32_t client_timeout, std::vector<nic::InferInput*>& inputs,
    std::vector<const nic::InferRequestedOutput*>& outputs,
    nic::InferOptions& options, std::vector<int32_t>& input0_data)
{
  std::mutex mtx;
  std::condition_variable cv;
  bool done = false;

  auto callback = [&](nic::InferResult* result) {
    {
      std::shared_ptr<nic::InferResult> result_ptr;
      result_ptr.reset(result);
      std::lock_guard<std::mutex> lk(mtx);
      std::cout << "Callback called" << std::endl;
      done = true;
      if (result_ptr->RequestStatus().IsOk()) {
        ValidateResult(result_ptr, input0_data);
      } else {
        std::cerr << "error: Inference failed: " << result_ptr->RequestStatus()
                  << std::endl;
        exit(1);
      }
    }
    cv.notify_all();
  };

  options.client_timeout_ = client_timeout;
  if (grpc_client.get() != nullptr) {
    FAIL_IF_ERR(
        grpc_client->AsyncInfer(callback, options, inputs, outputs),
        "unable to run model");
  } else {
    FAIL_IF_ERR(
        http_client->AsyncInfer(callback, options, inputs, outputs),
        "unable to run model");
  }

  // Wait until all callbacks are invoked
  {
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [&]() { return done; });
  }
}

void
RunStreamingInference(
    std::unique_ptr<nic::InferenceServerGrpcClient>& grpc_client,
    uint32_t client_timeout, std::vector<nic::InferInput*>& inputs,
    std::vector<const nic::InferRequestedOutput*>& outputs,
    nic::InferOptions& options, std::vector<int32_t>& input0_data)
{
  std::mutex mtx;
  std::condition_variable cv;
  std::vector<std::shared_ptr<nic::InferResult>> result_list;

  FAIL_IF_ERR(
      grpc_client->StartStream(
          [&](nic::InferResult* result) {
            {
              std::shared_ptr<nic::InferResult> result_ptr(result);
              std::lock_guard<std::mutex> lk(mtx);
              result_list.push_back(result_ptr);
            }
            cv.notify_all();
          },
          false /*ship_stats*/, client_timeout),
      "Failed to start the stream");

  FAIL_IF_ERR(
      grpc_client->AsyncStreamInfer(options, inputs), "unable to run model");

  auto timeout = std::chrono::microseconds(client_timeout);
  // Wait until all callbacks are invoked or the timeout expires
  {
    std::unique_lock<std::mutex> lk(mtx);
    if (!cv.wait_for(lk, timeout, [&]() { return (result_list.size() > 0); })) {
      std::cerr << "Stream has been closed" << std::endl;
      exit(1);
    }
  }

  // Validate results
  if (result_list.size() != 1) {
    std::cerr << "error: expected a single response, got " << result_list.size()
              << std::endl;
    exit(1);
  }
  if (result_list[0]->RequestStatus().IsOk()) {
    ValidateResult(result_list[0], input0_data);
  } else {
    std::cerr << "error: Inference failed: " << result_list[0]->RequestStatus()
              << std::endl;
    exit(1);
  }
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
  std::cerr << "\t-t <client timeout in microseconds>" << std::endl;
  std::cerr << std::endl;

  exit(1);
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  std::string protocol = "http";
  std::string url;
  bool async = false;
  bool streaming = false;
  uint32_t client_timeout = 0;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vi:u:ast:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
        break;
      case 'i': {
        std::string p(optarg);
        std::transform(p.begin(), p.end(), p.begin(), ::tolower);
        if (p == "grpc" || p == "http") {
          protocol = p;
        } else {
          protocol = "unknown";
        }
        break;
      }
      case 'u':
        url = optarg;
        break;
      case 'a':
        async = true;
        break;
      case 's':
        streaming = true;
        break;
      case 't':
        client_timeout = std::stoi(optarg);
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  // Option validations
  if (protocol == "unknown") {
    std::cerr << "Supports only http and grpc protocols" << std::endl;
    Usage(argv);
  }

  if (streaming && (protocol != "grpc")) {
    std::cerr << "Supports only http and grpc protocols" << std::endl;
    Usage(argv);
  }

  std::string model_name = "custom_identity_int32";
  std::string model_version = "";

  // Create a InferenceServerGrpcClient instance to communicate with the
  // server using gRPC protocol.
  std::unique_ptr<nic::InferenceServerGrpcClient> grpc_client;
  std::unique_ptr<nic::InferenceServerHttpClient> http_client;

  if (protocol == "grpc") {
    if (url.empty()) {
      url = "localhost:8001";
    }
    FAIL_IF_ERR(
        nic::InferenceServerGrpcClient::Create(&grpc_client, url, verbose),
        "unable to create grpc client");
  } else {
    if (url.empty()) {
      url = "localhost:8000";
    }
    FAIL_IF_ERR(
        nic::InferenceServerHttpClient::Create(&http_client, url, verbose),
        "unable to create grpc client");
  }

  // Initialize the tensor data
  std::vector<int32_t> input0_data(16);
  for (size_t i = 0; i < 16; ++i) {
    input0_data[i] = i;
  }

  std::vector<int64_t> shape{1, 16};

  // Initialize the inputs with the data.
  nic::InferInput* input0;

  FAIL_IF_ERR(
      nic::InferInput::Create(&input0, "INPUT0", shape, "INT32"),
      "unable to get INPUT0");
  std::shared_ptr<nic::InferInput> input0_ptr;
  input0_ptr.reset(input0);

  FAIL_IF_ERR(
      input0_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&input0_data[0]),
          input0_data.size() * sizeof(int32_t)),
      "unable to set data for INPUT0");

  // Generate the outputs to be requested.
  nic::InferRequestedOutput* output0;
  FAIL_IF_ERR(
      nic::InferRequestedOutput::Create(&output0, "OUTPUT0"),
      "unable to get 'OUTPUT0'");
  std::shared_ptr<nic::InferRequestedOutput> output0_ptr;
  output0_ptr.reset(output0);


  // The inference settings. Will be using default for now.
  nic::InferOptions options(model_name);
  options.model_version_ = model_version;
  options.client_timeout_ = client_timeout;

  std::vector<nic::InferInput*> inputs = {input0_ptr.get()};
  std::vector<const nic::InferRequestedOutput*> outputs = {output0_ptr.get()};

  // Send inference request to the inference server.
  if (streaming) {
    RunStreamingInference(
        grpc_client, client_timeout, inputs, outputs, options, input0_data);
  } else if (async) {
    RunAsynchronousInference(
        grpc_client, http_client, client_timeout, inputs, outputs, options,
        input0_data);
  } else {
    RunSynchronousInference(
        grpc_client, http_client, client_timeout, inputs, outputs, options,
        input0_data);
  }

  return 0;
}
