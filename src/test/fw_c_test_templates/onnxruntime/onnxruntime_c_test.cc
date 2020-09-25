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

#include <onnxruntime_c_api.h>
#include <unistd.h>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#ifdef TRITON_ENABLE_GPU
#include <cuda_provider_factory.h>
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

// Check Status of Ort function and exit on error
void
CheckStatus(OrtStatus* status)
{
  if (status != nullptr) {
    OrtErrorCode code = ort_api->GetErrorCode(status);
    const char* msg = ort_api->GetErrorMessage(status);
    std::cerr << "ONNXRuntime error " << code << ": " << msg;
    ort_api->ReleaseStatus(status);
    exit(1);
  }
}

void
Usage(char** argv)
{
  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-m <model_path>" << std::endl;
  std::cerr << std::endl;

  exit(1);
}

int
main(int argc, char* argv[])
{
  std::string model_path = "model.onnx";

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "m:")) != -1) {
    switch (opt) {
      case 'm':
        model_path = optarg;
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  // Initialize  enviroment
  OrtEnv* env;
  CheckStatus(ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

  // Initialize session options
  OrtSessionOptions* session_options;
  CheckStatus(ort_api->CreateSessionOptions(&session_options));
  CheckStatus(ort_api->SetIntraOpNumThreads(session_options, 1));

  // Set graph optimization level
  GraphOptimizationLevel optimization_level =
      GraphOptimizationLevel::ORT_ENABLE_ALL;
  CheckStatus(ort_api->SetSessionGraphOptimizationLevel(
      session_options, optimization_level));

  // Optionally add more execution providers via session_options
#ifdef TRITON_ENABLE_GPU
  size_t gpu_device_id = 0;
  CheckStatus(OrtSessionOptionsAppendExecutionProvider_CUDA(
      session_options, gpu_device_id));
#endif  // TRITON_ENABLE_GPU

  printf("Test using native Onnxruntime C API\n");
  OrtSession* session;
  CheckStatus(ort_api->CreateSession(
      env, model_path.c_str(), session_options, &session));

  OrtAllocator* allocator;
  CheckStatus(ort_api->GetAllocatorWithDefaultOptions(&allocator));
  const OrtMemoryInfo* allocator_info;
  CheckStatus(ort_api->AllocatorGetInfo(allocator, &allocator_info));

  // This model has only 2 inputs and 1 output node.
  // input__0: {4096, 13} FP16
  // input__1: {4096, 26} INT64
  // output__0: {4096, 1} FP16

  // Use OrtSession to get metadata of inputs and outputs if not known
  std::vector<const char*> input_names = {"input__0", "input__1"};
  std::vector<const char*> output_names = {"output__0"};
  std::vector<std::vector<int64_t>> input_dims = {{4096, 13}, {4096, 26}};
  std::vector<std::vector<int64_t>> output_dims = {{4096, 1}};
  std::vector<ONNXTensorElementDataType> input_types = {
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64};
  std::vector<ONNXTensorElementDataType> output_types = {
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16};

  // initialize input data with 0s
  int64_t input0_elements = accumulate(
      input_dims[0].begin(), input_dims[0].end(), 1,
      std::multiplies<int64_t>());
  int64_t input1_elements = accumulate(
      input_dims[1].begin(), input_dims[1].end(), 1,
      std::multiplies<int64_t>());
  // Use char since float16 does not exist natively
  std::vector<char> input0_data(input0_elements * 2, 0);
  std::vector<int64_t> input1_data(input1_elements, 0);

  // Create input tensor objects from data vectors
  std::vector<OrtValue*> input_tensors;
  input_tensors.emplace_back(nullptr);
  CheckStatus(ort_api->CreateTensorWithDataAsOrtValue(
      allocator_info, input0_data.data(),
      input0_elements * 2 /*sizeof(float16)*/, input_dims[0].data(),
      input_dims[0].size(), input_types[0], &input_tensors.back()));

  input_tensors.emplace_back(nullptr);
  CheckStatus(ort_api->CreateTensorWithDataAsOrtValue(
      allocator_info, input1_data.data(), input1_elements * sizeof(int64_t),
      input_dims[1].data(), input_dims[1].size(), input_types[1],
      &input_tensors.back()));

  // Run model with input tensors and get back output tensor
  OrtValue* output_tensor = nullptr;
  CheckStatus(ort_api->Run(
      session, nullptr, input_names.data(),
      (const OrtValue* const*)input_tensors.data(), input_tensors.size(),
      output_names.data(), output_names.size(), &output_tensor));

  // Add in your own validation script for outputs.

  ort_api->ReleaseValue(input_tensors[0]);
  ort_api->ReleaseValue(input_tensors[1]);
  ort_api->ReleaseSession(session);
  ort_api->ReleaseSessionOptions(session_options);
  ort_api->ReleaseEnv(env);
  printf("Done!\n");
  return 0;
}