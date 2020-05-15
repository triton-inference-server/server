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
#include <iostream>
#include <string>
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
  if (datatype.compare("BYTES") != 0) {
    std::cerr << "error: received incorrect datatype for " << name << ": "
              << datatype << std::endl;
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
  std::string url("localhost:8000");
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

  // We use a simple model that takes 2 input tensors of 16 strings
  // each and returns 2 output tensors of 16 strings each. The input
  // strings must represent integers. One output tensor is the
  // element-wise sum of the inputs and one output is the element-wise
  // difference.
  std::string model_name = "simple_string";
  std::string model_version = "";

  // Create a InferenceServerHttpClient instance to communicate with the
  // server using http protocol.
  std::unique_ptr<nic::InferenceServerHttpClient> client;
  FAIL_IF_ERR(
      nic::InferenceServerHttpClient::Create(&client, url, verbose),
      "unable to create http client");

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

  std::vector<int64_t> shape{1, 16};

  // Initialize the inputs with the data.
  nic::InferInput* input0;
  nic::InferInput* input1;

  FAIL_IF_ERR(
      nic::InferInput::Create(&input0, "INPUT0", shape, "BYTES"),
      "unable to get INPUT0");
  std::shared_ptr<nic::InferInput> input0_ptr;
  input0_ptr.reset(input0);
  FAIL_IF_ERR(
      nic::InferInput::Create(&input1, "INPUT1", shape, "BYTES"),
      "unable to get INPUT1");
  std::shared_ptr<nic::InferInput> input1_ptr;
  input1_ptr.reset(input1);

  FAIL_IF_ERR(
      input0_ptr->AppendFromString(input0_data),
      "unable to set data for INPUT0");
  FAIL_IF_ERR(
      input1_ptr->AppendFromString(input1_data),
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

  nic::InferResult* results;
  FAIL_IF_ERR(
      client->Infer(&results, options, inputs, outputs, http_headers),
      "unable to run model");
  std::shared_ptr<nic::InferResult> results_ptr;
  results_ptr.reset(results);

  // Validate the results...
  ValidateShapeAndDatatype("OUTPUT0", results_ptr);
  ValidateShapeAndDatatype("OUTPUT1", results_ptr);

  // Get the result data
  std::vector<std::string> result0_data;
  std::vector<std::string> result1_data;
  FAIL_IF_ERR(
      results_ptr->StringData("OUTPUT0", &result0_data),
      "unable to get data for OUTPUT0");
  if (result0_data.size() != 16) {
    std::cerr << "error: received incorrect number of strings for OUTPUT0: "
              << result0_data.size() << std::endl;
    exit(1);
  }

  FAIL_IF_ERR(
      results_ptr->StringData("OUTPUT1", &result1_data),
      "unable to get data for OUTPUT1");
  if (result1_data.size() != 16) {
    std::cerr << "error: received incorrect number of strings for OUTPUT1: "
              << result1_data.size() << std::endl;
    exit(1);
  }


  for (size_t i = 0; i < 16; ++i) {
    std::cout << input0_data[i] << " + " << input0_data[i] << " = "
              << result0_data[i] << std::endl;
    std::cout << input0_data[i] << " - " << input1_data[i] << " = "
              << result1_data[i] << std::endl;

    if (expected_sum[i] != std::stoi(result0_data[i])) {
      std::cerr << "error: incorrect sum" << std::endl;
      exit(1);
    }
    if (expected_diff[i] != std::stoi(result1_data[i])) {
      std::cerr << "error: incorrect difference" << std::endl;
      exit(1);
    }
  }

  // Get full response
  std::cout << results_ptr->DebugString() << std::endl;

  std::cout << "PASS : String Infer" << std::endl;

  return 0;
}
