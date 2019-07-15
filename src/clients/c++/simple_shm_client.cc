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

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include "src/clients/c++/request_grpc.h"
#include "src/clients/c++/request_http.h"

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
  std::cerr << "\t-H <HTTP header>" << std::endl;
  std::cerr << std::endl;
  std::cerr
      << "For -i, available protocols are 'grpc' and 'http'. Default is 'http."
      << std::endl;
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
  std::string protocol = "http";
  std::map<std::string, std::string> http_headers;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vi:u:H:")) != -1) {
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

  nic::Error err;

  // We use a simple model that takes 2 input tensors of 16 integers
  // each and returns 2 output tensors of 16 integers each. One output
  // tensor is the element-wise sum of the inputs and one output is
  // the element-wise difference.
  std::string model_name = "simple";

  // Create a health context and get the ready and live state of the
  // server.
  std::unique_ptr<nic::ServerHealthContext> health_ctx;
  if (protocol == "http") {
    err = nic::ServerHealthHttpContext::Create(
        &health_ctx, url, http_headers, verbose);
  } else if (protocol == "grpc") {
    err = nic::ServerHealthGrpcContext::Create(&health_ctx, url, verbose);
  } else {
    Usage(argv, "unknown protocol '" + protocol + "'");
  }

  if (!err.IsOk()) {
    std::cerr << "error: unable to create health context: " << err << std::endl;
    exit(1);
  }

  bool live, ready;
  err = health_ctx->GetLive(&live);
  if (!err.IsOk()) {
    std::cerr << "error: unable to get server liveness: " << err << std::endl;
    exit(1);
  }

  err = health_ctx->GetReady(&ready);
  if (!err.IsOk()) {
    std::cerr << "error: unable to get server readiness: " << err << std::endl;
    exit(1);
  }

  std::cout << "Health for model " << model_name << ":" << std::endl;
  std::cout << "Live: " << live << std::endl;
  std::cout << "Ready: " << ready << std::endl;

  // Create a status context and get the status of the model.
  std::unique_ptr<nic::ServerStatusContext> status_ctx;
  if (protocol == "http") {
    err = nic::ServerStatusHttpContext::Create(
        &status_ctx, url, http_headers, model_name, verbose);
  } else if (protocol == "grpc") {
    err = nic::ServerStatusGrpcContext::Create(
        &status_ctx, url, model_name, verbose);
  } else {
    Usage(argv, "unknown protocol '" + protocol + "'");
  }

  if (!err.IsOk()) {
    std::cerr << "error: unable to create status context: " << err << std::endl;
    exit(1);
  }

  ni::ServerStatus server_status;
  err = status_ctx->GetServerStatus(&server_status);
  if (!err.IsOk()) {
    std::cerr << "error: unable to get status: " << err << std::endl;
    exit(1);
  }

  std::cout << "Status for model " << model_name << ":" << std::endl;
  std::cout << server_status.DebugString() << std::endl;

  // Create the inference context for the model.
  std::unique_ptr<nic::InferContext> infer_ctx;
  if (protocol == "http") {
    err = nic::InferHttpContext::Create(
        &infer_ctx, url, http_headers, model_name, -1 /* model_version */,
        verbose);
  } else if (protocol == "grpc") {
    err = nic::InferGrpcContext::Create(
        &infer_ctx, url, model_name, -1 /* model_version */, verbose);
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
  for (const auto& output : infer_ctx->Outputs()) {
    options->AddRawResult(output);
  }

  FAIL_IF_ERR(
      infer_ctx->SetRunOptions(*options), "unable to set inference options");

  // Create the data for the two input tensors. Initialize the first
  // to unique integers and the second to all ones.
  std::vector<int32_t> input0_data(16);
  std::vector<int32_t> input1_data(16);
  std::vector<int32_t> output0_data(16);
  std::vector<int32_t> output1_data(16);
  for (size_t i = 0; i < 16; ++i) {
    input0_data[i] = i;
    input1_data[i] = 1;
  }

  // Initialize the inputs with the data.
  std::shared_ptr<nic::InferContext::Input> input0, input1;
  std::shared_ptr<nic::InferContext::Output> output0, output1;
  FAIL_IF_ERR(infer_ctx->GetInput("INPUT0", &input0), "unable to get INPUT0");
  FAIL_IF_ERR(infer_ctx->GetInput("INPUT1", &input1), "unable to get INPUT1");
  FAIL_IF_ERR(
      infer_ctx->GetOutput("OUTPUT0", &output0), "unable to get OUTPUT0");
  FAIL_IF_ERR(
      infer_ctx->GetOutput("OUTPUT1", &output1), "unable to get OUTPUT1");

  FAIL_IF_ERR(input0->Reset(), "unable to reset INPUT0");
  FAIL_IF_ERR(input1->Reset(), "unable to reset INPUT1");

  size_t input_byte_size = 16 * sizeof(int32_t);
  size_t output_byte_size = 16 * sizeof(int32_t);
  std::string shm_key;
  int shm_fd, res;
  void* shm_addr;

  // Create  Input0 and Input1 in Shared Memory
  // get shared memory file descriptor (NOT a file)
  for (size_t n = 0; n < 2; n++) {
    shm_key = "/input" + std::to_string(n) + "_simple";
    shm_fd = shm_open(shm_key.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (shm_fd == -1) {
      std::cerr << "error: unable to get input shared memory descriptor";
      exit(1);
    }
    // extend shared memory object as by default it's initialized with size 0
    res = ftruncate(shm_fd, input_byte_size);
    if (res == -1) {
      std::cerr << "error: unable to get initialize size";
      exit(1);
    }
    // map shared memory to process address space
    shm_addr = mmap(NULL, input_byte_size, PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_addr == MAP_FAILED) {
      std::cerr << "error: unable to process address space";
      exit(1);
    }
  }

  // Create  Output0 and Output1 in Shared Memory
  // get shared memory file descriptor (NOT a file)
  for (size_t n = 0; n < 2; n++) {
    shm_key = "/output" + std::to_string(n) + "_simple";
    shm_fd = shm_open(shm_key.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (shm_fd == -1) {
      std::cerr << "error: unable to get output shared memory descriptor";
      exit(1);
    }
    // extend shared memory object as by default it's initialized with size 0
    res = ftruncate(shm_fd, output_byte_size);
    if (res == -1) {
      std::cerr << "error: unable to get initialize size";
      exit(1);
    }
    // map shared memory to process address space
    shm_addr = mmap(NULL, output_byte_size, PROT_READ, MAP_SHARED, shm_fd, 0);
    if (shm_addr == MAP_FAILED) {
      std::cerr << "error: unable to process address space";
      exit(1);
    }
  }

  // Write Input0 and Input1 to shared memory
  for (size_t n = 0; n < 2; n++) {
    shm_key = "/input" + std::to_string(n) + "_simple";
    shm_fd = shm_open(shm_key.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
    if (shm_fd == -1) {
      std::cerr << "error: unable to get shared memory descriptor";
      exit(1);
    }
    shm_addr = mmap(NULL, input_byte_size, PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_addr == MAP_FAILED) {
      std::cerr << "error: unable to process address space";
      exit(1);
    }
    if (n == 0)
      memcpy(shm_addr, &input0_data[0], input_byte_size);
    else
      memcpy(shm_addr, &input1_data[0], input_byte_size);
  }

  err = input0->SetSharedMemory("/input0_simple", 0, input_byte_size);
  if (!err.IsOk()) {
    std::cerr << "failed setting shared memory input: " << err << std::endl;
    exit(1);
  }
  err = input1->SetSharedMemory("/input1_simple", 0, input_byte_size);
  if (!err.IsOk()) {
    std::cerr << "failed setting shared memory input: " << err << std::endl;
    exit(1);
  }

  err = output0->SetSharedMemory("/output0_simple", 0, output_byte_size);
  if (!err.IsOk()) {
    std::cerr << "failed setting shared memory output: " << err << std::endl;
    exit(1);
  }
  err = output1->SetSharedMemory("/output1_simple", 0, output_byte_size);
  if (!err.IsOk()) {
    std::cerr << "failed setting shared memory output: " << err << std::endl;
    exit(1);
  }


  // Send inference request to the inference server.
  std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;
  FAIL_IF_ERR(infer_ctx->Run(&results), "unable to run model");

  // We expect there to be 2 results and them to be written by TRTIS into the
  // corresponding shared memory location provided. The 16 result elements
  // in both outputs are the sum and difference calculated by the model.
  for (size_t n = 0; n < 2; n++) {
    shm_key = "/output" + std::to_string(n) + "_simple";
    shm_fd = shm_open(shm_key.c_str(), O_RDONLY, S_IRUSR | S_IWUSR);
    if (shm_fd == -1) {
      std::cerr << "error: unable to get shared memory descriptor";
      exit(1);
    }
    shm_addr = mmap(NULL, output_byte_size, PROT_READ, MAP_SHARED, shm_fd, 0);
    if (shm_addr == MAP_FAILED) {
      std::cerr << "error: unable to process address space";
      exit(1);
    }

    // read output from memory
    if (n == 0)
      memcpy(&output0_data[0], (float*)(shm_addr), output_byte_size);
    else
      memcpy(&output1_data[0], (float*)(shm_addr), output_byte_size);
  }

  // Validate result from model
  for (size_t i = 0; i < 16; ++i) {
    std::cout << input0_data[i] << " + " << input1_data[i] << " = "
              << output0_data[i] << std::endl;
    std::cout << input0_data[i] << " - " << input1_data[i] << " = "
              << output1_data[i] << std::endl;

    if ((input0_data[i] + input1_data[i]) != output0_data[i]) {
      std::cerr << "error: incorrect sum" << std::endl;
      exit(1);
    }
    if ((input0_data[i] - input1_data[i]) != output1_data[i]) {
      std::cerr << "error: incorrect difference" << std::endl;
      exit(1);
    }
  }

  return 0;
}
