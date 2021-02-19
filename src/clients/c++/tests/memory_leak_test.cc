// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#define INPUT_DIM 16
#define INT32_BYTE_SIZE 4

namespace {

void
ValidateShapeAndDatatype(
    const std::string& name, std::shared_ptr<nic::InferResult> result)
{
  std::vector<int64_t> shape;
  FAIL_IF_ERR(
      result->Shape(name, &shape), "unable to get shape for '" + name + "'");
  // Validate shape
  if ((shape.size() != 2) || (shape[0] != 1) || (shape[1] != INPUT_DIM)) {
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
  if (output0_byte_size != INPUT_DIM * INT32_BYTE_SIZE) {
    std::cerr << "error: received incorrect byte size for 'OUTPUT0': "
              << output0_byte_size << std::endl;
    exit(1);
  }

  for (size_t i = 0; i < INPUT_DIM; ++i) {
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
    std::vector<nic::InferInput*>& inputs,
    std::vector<const nic::InferRequestedOutput*>& outputs,
    nic::InferOptions& options, std::vector<int32_t>& input0_data, bool reuse,
    std::string url, bool verbose, std::string protocol, uint32_t repetitions)
{
  // If re-use is enabled then use these client objects else use new objects for
  // each inference request.
  std::unique_ptr<nic::InferenceServerGrpcClient> grpc_client_reuse;
  std::unique_ptr<nic::InferenceServerHttpClient> http_client_reuse;

  for (size_t i = 0; i < repetitions; ++i) {
    nic::InferResult* results;
    if (!reuse) {
      if (protocol == "grpc") {
        std::unique_ptr<nic::InferenceServerGrpcClient> grpc_client;
        FAIL_IF_ERR(
            nic::InferenceServerGrpcClient::Create(&grpc_client, url, verbose),
            "unable to create grpc client");
        FAIL_IF_ERR(
            grpc_client->Infer(&results, options, inputs, outputs),
            "unable to run model");
      } else {
        std::unique_ptr<nic::InferenceServerHttpClient> http_client;
        FAIL_IF_ERR(
            nic::InferenceServerHttpClient::Create(&http_client, url, verbose),
            "unable to create http client");
        FAIL_IF_ERR(
            http_client->Infer(&results, options, inputs, outputs),
            "unable to run model");
      }
    } else {
      if (protocol == "grpc") {
        FAIL_IF_ERR(
            nic::InferenceServerGrpcClient::Create(
                &grpc_client_reuse, url, verbose),
            "unable to create grpc client");
        FAIL_IF_ERR(
            grpc_client_reuse->Infer(&results, options, inputs, outputs),
            "unable to run model");
      } else {
        FAIL_IF_ERR(
            nic::InferenceServerHttpClient::Create(
                &http_client_reuse, url, verbose),
            "unable to create http client");
        FAIL_IF_ERR(
            http_client_reuse->Infer(&results, options, inputs, outputs),
            "unable to run model");
      }
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
}

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-i <http/grpc>" << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-t <client timeout in microseconds>" << std::endl;
  std::cerr << "\t-r <number of repetitions for inference> default is 100."
            << std::endl;
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
  bool reuse = false;
  uint32_t repetitions = 100;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vi:u:r:R")) != -1) {
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
      case 'r':
        repetitions = std::stoi(optarg);
        break;
      case 'R':
        reuse = true;
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

  std::string model_name = "custom_identity_int32";
  std::string model_version = "";

  if (protocol == "grpc") {
    if (url.empty()) {
      url = "localhost:8001";
    }
  } else {
    if (url.empty()) {
      url = "localhost:8000";
    }
    protocol = "http";
  }

  // Initialize the tensor data
  std::vector<int32_t> input0_data(INPUT_DIM);
  for (size_t i = 0; i < INPUT_DIM; ++i) {
    input0_data[i] = i;
  }

  std::vector<int64_t> shape{1, INPUT_DIM};

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

  std::vector<nic::InferInput*> inputs = {input0_ptr.get()};
  std::vector<const nic::InferRequestedOutput*> outputs = {output0_ptr.get()};

  // Send 'repetitions' number of inference requests to the inference server.
  RunSynchronousInference(
      inputs, outputs, options, input0_data, reuse, url, verbose, protocol,
      repetitions);

  return 0;
}
