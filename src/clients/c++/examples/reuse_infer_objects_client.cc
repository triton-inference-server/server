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

union TritonClient {
  TritonClient()
  {
    new (&http_client_) std::unique_ptr<nic::InferenceServerHttpClient>{};
  }
  ~TritonClient() {}

  std::unique_ptr<nic::InferenceServerHttpClient> http_client_;
  std::unique_ptr<nic::InferenceServerGrpcClient> grpc_client_;
};

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
InferAndValidate(
    const bool use_shared_memory, TritonClient& triton_client,
    const std::string& protocol, const nic::InferOptions& options,
    const nic::Headers& http_headers, std::vector<nic::InferInput*>& inputs,
    const size_t input_byte_size,
    std::vector<nic::InferRequestedOutput*>& outputs,
    const size_t output_byte_size, std::vector<int*>& shm_ptrs)
{
  std::vector<int32_t> input0_data(16);
  std::vector<int32_t> input1_data(16);

  int32_t* input0_data_ptr;
  int32_t* input1_data_ptr;
  int32_t* output0_data_ptr;
  int32_t* output1_data_ptr;

  FAIL_IF_ERR(inputs[0]->Reset(), "unable to reset input 'INPUT0'");
  FAIL_IF_ERR(inputs[1]->Reset(), "unable to reset input 'INPUT1'");

  if (use_shared_memory) {
    input0_data_ptr = shm_ptrs[0];
    input1_data_ptr = shm_ptrs[1];

    FAIL_IF_ERR(
        inputs[0]->SetSharedMemory(
            "input_data", input_byte_size, 0 /* offset */),
        "unable to set shared memory for INPUT0");
    FAIL_IF_ERR(
        inputs[1]->SetSharedMemory(
            "input_data", input_byte_size, input_byte_size /* offset */),
        "unable to set shared memory for INPUT1");

    FAIL_IF_ERR(
        outputs[0]->SetSharedMemory(
            "output_data", output_byte_size, 0 /* offset */),
        "unable to set shared memory for 'OUTPUT0'");
    FAIL_IF_ERR(
        outputs[1]->SetSharedMemory(
            "output_data", output_byte_size, output_byte_size /* offset */),
        "unable to set shared memory for 'OUTPUT1'");

  } else {
    input0_data_ptr = &input0_data[0];
    input1_data_ptr = &input1_data[0];
    // Create the data for the two input tensors. Initialize the first
    // to unique integers and the second to all twos. We use twos instead
    // of ones in input1_data to validate whether inputs were set correctly.
    for (size_t i = 0; i < 16; ++i) {
      input0_data[i] = i;
      input1_data[i] = 2;
    }

    FAIL_IF_ERR(
        inputs[0]->AppendRaw(
            reinterpret_cast<uint8_t*>(&input0_data[0]),
            input0_data.size() * sizeof(int32_t)),
        "unable to set data for 'INPUT0'");
    FAIL_IF_ERR(
        inputs[1]->AppendRaw(
            reinterpret_cast<uint8_t*>(&input1_data[0]),
            input1_data.size() * sizeof(int32_t)),
        "unable to set data for 'INPUT1'");

    FAIL_IF_ERR(
        outputs[0]->UnsetSharedMemory(),
        "unable to unset shared memory for 'OUTPUT0'");
    FAIL_IF_ERR(
        outputs[1]->UnsetSharedMemory(),
        "unable to unset shared memory for 'OUTPUT1'");
  }

  std::vector<const nic::InferRequestedOutput*> routputs = {outputs[0],
                                                            outputs[1]};

  nic::InferResult* results;
  if (protocol == "http") {
    FAIL_IF_ERR(
        triton_client.http_client_->Infer(
            &results, options, inputs, routputs, http_headers),
        "unable to run model");
  } else {
    FAIL_IF_ERR(
        triton_client.grpc_client_->Infer(
            &results, options, inputs, routputs, http_headers),
        "unable to run model");
  }
  std::shared_ptr<nic::InferResult> results_ptr;
  results_ptr.reset(results);

  // Validate the results...
  ValidateShapeAndDatatype("OUTPUT0", results_ptr);
  ValidateShapeAndDatatype("OUTPUT1", results_ptr);

  if (use_shared_memory) {
    std::cout << "\n\n======== SHARED_MEMORY ========\n";
    output0_data_ptr = shm_ptrs[2];
    output1_data_ptr = shm_ptrs[3];
  } else {
    std::cout << "\n\n======== NO_SHARED_MEMORY ========\n";
    // Get pointers to the result returned...
    size_t recv_output0_byte_size;
    FAIL_IF_ERR(
        results_ptr->RawData(
            "OUTPUT0", (const uint8_t**)&output0_data_ptr,
            &recv_output0_byte_size),
        "unable to get result data for 'OUTPUT0'");
    if (recv_output0_byte_size != output_byte_size) {
      std::cerr << "error: received incorrect byte size for 'OUTPUT0': "
                << recv_output0_byte_size << std::endl;
      exit(1);
    }

    size_t recv_output1_byte_size;
    FAIL_IF_ERR(
        results_ptr->RawData(
            "OUTPUT1", (const uint8_t**)&output1_data_ptr,
            &recv_output1_byte_size),
        "unable to get result data for 'OUTPUT1'");
    if (recv_output1_byte_size != output_byte_size) {
      std::cerr << "error: received incorrect byte size for 'OUTPUT1': "
                << recv_output1_byte_size << std::endl;
      exit(1);
    }
  }
  for (size_t i = 0; i < 16; ++i) {
    std::cout << input0_data_ptr[i] << " + " << input1_data_ptr[i] << " = "
              << output0_data_ptr[i] << std::endl;
    std::cout << input0_data_ptr[i] << " - " << input1_data_ptr[i] << " = "
              << output1_data_ptr[i] << std::endl;

    if ((input0_data_ptr[i] + input1_data_ptr[i]) != output0_data_ptr[i]) {
      std::cerr << "error: incorrect sum" << std::endl;
      exit(1);
    }
    if ((input0_data_ptr[i] - input1_data_ptr[i]) != output1_data_ptr[i]) {
      std::cerr << "error: incorrect difference" << std::endl;
      exit(1);
    }
  }

  std::cout << "\n======== END ========\n\n";
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

// Tests whether the same InferInput and InferRequestedOutput objects can be
// successfully used repeatedly for different inferences using/not-using
// shared memory.
int
main(int argc, char** argv)
{
  bool verbose = false;
  std::string url("localhost:8000");
  bool url_specified = false;
  nic::Headers http_headers;
  std::string protocol("http");

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vu:i:H:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
        break;
      case 'u':
        url = optarg;
        url_specified = true;
        break;
      case 'i':
        protocol = optarg;
        std::transform(
            protocol.begin(), protocol.end(), protocol.begin(), ::tolower);
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

  // Create the inference client for the server. From it
  // extract and validate that the model meets the requirements for
  // image classification.
  TritonClient triton_client;
  nic::Error err;
  if (protocol == "http") {
    err = nic::InferenceServerHttpClient::Create(
        &triton_client.http_client_, url, verbose);
  } else if (protocol == "grpc") {
    if (!url_specified) {
      url = "localhost:8001";
    }
    err = nic::InferenceServerGrpcClient::Create(
        &triton_client.grpc_client_, url, verbose);
  } else {
    std::cerr
        << "error: unsupported protocol provided: only supports grpc or http."
        << std::endl;
    exit(1);
  }
  if (!err.IsOk()) {
    std::cerr << "error: unable to create client for inference: " << err
              << std::endl;
    exit(1);
  }

  // Unregistering all shared memory regions for a clean
  // start.
  if (protocol == "http") {
    FAIL_IF_ERR(
        triton_client.http_client_->UnregisterSystemSharedMemory(),
        "unable to unregister all system shared memory regions");
    FAIL_IF_ERR(
        triton_client.http_client_->UnregisterCudaSharedMemory(),
        "unable to unregister all cuda shared memory regions");
  } else {
    FAIL_IF_ERR(
        triton_client.grpc_client_->UnregisterSystemSharedMemory(),
        "unable to unregister all system shared memory regions");
    FAIL_IF_ERR(
        triton_client.grpc_client_->UnregisterCudaSharedMemory(),
        "unable to unregister all cuda shared memory regions");
  }

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

  if (protocol == "http") {
    FAIL_IF_ERR(
        triton_client.http_client_->RegisterSystemSharedMemory(
            "input_data", "/input_simple", input_byte_size * 2),
        "failed to register input shared memory region");
  } else {
    FAIL_IF_ERR(
        triton_client.grpc_client_->RegisterSystemSharedMemory(
            "input_data", "/input_simple", input_byte_size * 2),
        "failed to register input shared memory region");
  }

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

  if (protocol == "http") {
    FAIL_IF_ERR(
        triton_client.http_client_->RegisterSystemSharedMemory(
            "output_data", "/output_simple", output_byte_size * 2),
        "failed to register output shared memory region");
  } else {
    FAIL_IF_ERR(
        triton_client.grpc_client_->RegisterSystemSharedMemory(
            "output_data", "/output_simple", output_byte_size * 2),
        "failed to register output shared memory region");
  }

  std::vector<nic::InferInput*> inputs = {input0_ptr.get(), input1_ptr.get()};
  std::vector<nic::InferRequestedOutput*> outputs = {output0_ptr.get(),
                                                     output1_ptr.get()};

  std::vector<int*> shm_ptrs = {input0_shm, input1_shm, output0_shm,
                                output1_shm};

  // The inference settings. Will be using default for now.
  nic::InferOptions options(model_name);
  options.model_version_ = model_version;

  // Issue inference using shared memory
  InferAndValidate(
      true /* use_shared_memory */, triton_client, protocol, options,
      http_headers, inputs, input_byte_size, outputs, output_byte_size,
      shm_ptrs);

  // Issue inference without using shared memory
  InferAndValidate(
      false /* use_shared_memory */, triton_client, protocol, options,
      http_headers, inputs, input_byte_size, outputs, output_byte_size,
      shm_ptrs);

  // Issue inference using shared memory
  InferAndValidate(
      true /* use_shared_memory */, triton_client, protocol, options,
      http_headers, inputs, input_byte_size, outputs, output_byte_size,
      shm_ptrs);

  // Unregister shared memory
  if (protocol == "http") {
    FAIL_IF_ERR(
        triton_client.http_client_->UnregisterSystemSharedMemory("input_data"),
        "unable to unregister shared memory input region");
    FAIL_IF_ERR(
        triton_client.http_client_->UnregisterSystemSharedMemory("output_data"),
        "unable to unregister shared memory output region");
  } else {
    FAIL_IF_ERR(
        triton_client.grpc_client_->UnregisterSystemSharedMemory("input_data"),
        "unable to unregister shared memory input region");
    FAIL_IF_ERR(
        triton_client.grpc_client_->UnregisterSystemSharedMemory("output_data"),
        "unable to unregister shared memory output region");
  }

  // Cleanup shared memory
  FAIL_IF_ERR(nic::UnmapSharedMemory(input0_shm, input_byte_size * 2), "");
  FAIL_IF_ERR(nic::UnlinkSharedMemoryRegion("/input_simple"), "");
  FAIL_IF_ERR(nic::UnmapSharedMemory(output0_shm, output_byte_size * 2), "");
  FAIL_IF_ERR(nic::UnlinkSharedMemoryRegion("/output_simple"), "");

  return 0;
}
