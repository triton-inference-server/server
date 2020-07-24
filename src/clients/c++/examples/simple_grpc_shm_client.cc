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
#include "src/clients/c++/examples/shm_utils.h"
#include "src/clients/c++/library/grpc_client.h"

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

  // Unregistering all shared memory regions for a clean
  // start.
  FAIL_IF_ERR(
      client->UnregisterSystemSharedMemory(),
      "unable to unregister all system shared memory regions");
  FAIL_IF_ERR(
      client->UnregisterCudaSharedMemory(),
      "unable to unregister all cuda shared memory regions");

  std::vector<int64_t> shape{1, 16};
  size_t input_byte_size = 64;
  size_t output_byte_size = 64;

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

  // Create Input0 and Input1 in Shared Memory. Initialize Input0 to unique
  // integers and Input1 to all ones.
  std::string shm_key = "/input_simple";
  int shm_fd_ip, *input0_shm;
  FAIL_IF_ERR(
      nic::CreateSharedMemoryRegion(shm_key, input_byte_size * 2, &shm_fd_ip),
      "");
  FAIL_IF_ERR(
      nic::MapSharedMemory(
          shm_fd_ip, 0, input_byte_size * 2, (void**)&input0_shm),
      "");
  FAIL_IF_ERR(nic::CloseSharedMemory(shm_fd_ip), "");
  int* input1_shm = (int*)(input0_shm + 16);
  for (size_t i = 0; i < 16; ++i) {
    *(input0_shm + i) = i;
    *(input1_shm + i) = 1;
  }

  FAIL_IF_ERR(
      client->RegisterSystemSharedMemory(
          "input_data", "/input_simple", input_byte_size * 2),
      "failed to register input shared memory region");

  FAIL_IF_ERR(
      input0_ptr->SetSharedMemory(
          "input_data", input_byte_size, 0 /* offset */),
      "unable to set shared memory for INPUT0");
  FAIL_IF_ERR(
      input1_ptr->SetSharedMemory(
          "input_data", input_byte_size, input_byte_size /* offset */),
      "unable to set shared memory for INPUT1");

  // Generate the outputs to be requested.
  nic::InferRequestedOutput* output0;
  nic::InferRequestedOutput* output1;

  FAIL_IF_ERR(
      nic::InferRequestedOutput::Create(&output0, "OUTPUT0"),
      "unable to get 'OUTPUT0'");
  std::shared_ptr<nic::InferRequestedOutput> output0_ptr;
  output0_ptr.reset(output0);
  FAIL_IF_ERR(
      nic::InferRequestedOutput::Create(&output1, "OUTPUT1"),
      "unable to get 'OUTPUT1'");
  std::shared_ptr<nic::InferRequestedOutput> output1_ptr;
  output1_ptr.reset(output1);

  // Create Output0 and Output1 in Shared Memory
  shm_key = "/output_simple";
  int shm_fd_op;
  int* output0_shm;
  FAIL_IF_ERR(
      nic::CreateSharedMemoryRegion(shm_key, output_byte_size * 2, &shm_fd_op),
      "");
  FAIL_IF_ERR(
      nic::MapSharedMemory(
          shm_fd_op, 0, output_byte_size * 2, (void**)&output0_shm),
      "");
  FAIL_IF_ERR(nic::CloseSharedMemory(shm_fd_op), "");
  int* output1_shm = (int*)(output0_shm + 16);


  FAIL_IF_ERR(
      client->RegisterSystemSharedMemory(
          "output_data", "/output_simple", output_byte_size * 2),
      "failed to register output shared memory region");

  FAIL_IF_ERR(
      output0_ptr->SetSharedMemory(
          "output_data", output_byte_size, 0 /* offset */),
      "unable to set shared memory for 'OUTPUT0'");
  FAIL_IF_ERR(
      output1_ptr->SetSharedMemory(
          "output_data", output_byte_size, output_byte_size /* offset */),
      "unable to set shared memory for 'OUTPUT1'");


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

  for (size_t i = 0; i < 16; ++i) {
    std::cout << input0_shm[i] << " + " << input1_shm[i] << " = "
              << output0_shm[i] << std::endl;
    std::cout << input0_shm[i] << " - " << input1_shm[i] << " = "
              << output1_shm[i] << std::endl;

    if ((input0_shm[i] + input1_shm[i]) != output0_shm[i]) {
      std::cerr << "error: incorrect sum" << std::endl;
      exit(1);
    }
    if ((input0_shm[i] - input1_shm[i]) != output1_shm[i]) {
      std::cerr << "error: incorrect difference" << std::endl;
      exit(1);
    }
  }

  // Get shared memory regions active/registered within triton
  inference::SystemSharedMemoryStatusResponse status;
  FAIL_IF_ERR(
      client->SystemSharedMemoryStatus(&status),
      "failed to get shared memory status");
  std::cout << "Shared Memory Status:\n" << status.DebugString() << "\n";

  // Unregister shared memory
  FAIL_IF_ERR(
      client->UnregisterSystemSharedMemory("input_data"),
      "unable to unregister shared memory input region");
  FAIL_IF_ERR(
      client->UnregisterSystemSharedMemory("output_data"),
      "unable to unregister shared memory output region");

  // Cleanup shared memory
  FAIL_IF_ERR(nic::UnmapSharedMemory(input0_shm, input_byte_size * 2), "");
  FAIL_IF_ERR(nic::UnlinkSharedMemoryRegion("/input_simple"), "");
  FAIL_IF_ERR(nic::UnmapSharedMemory(output0_shm, output_byte_size * 2), "");
  FAIL_IF_ERR(nic::UnlinkSharedMemoryRegion("/output_simple"), "");

  std::cout << "PASS : System Shared Memory " << std::endl;

  return 0;
}
