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

#include <unistd.h>
#include <chrono>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include "src/core/api.pb.h"
#include "src/core/server_status.pb.h"
#include "src/core/trtserver.h"
#include "src/servers/common.h"

#ifdef TRTIS_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRTIS_ENABLE_GPU

namespace ni = nvidia::inferenceserver;

namespace {

bool use_gpu_memory = false;

#ifdef TRTIS_ENABLE_GPU
#define FAIL_IF_CUDA_ERR(X, MSG)                                          \
  do {                                                                    \
    cudaError_t err = (X);                                                \
    if (err != cudaSuccess) {                                             \
      LOG_ERROR << "error: " << (MSG) << ": " << cudaGetErrorString(err); \
      exit(1);                                                            \
    }                                                                     \
  } while (false)


static auto gpu_data_deleter = [](void* data) {
  if (data != nullptr) {
    auto err = cudaFree(data);
    if (err != cudaSuccess) {
      LOG_ERROR << "error: failed to cudaFree " << data << ": "
                << cudaGetErrorString(err);
    }
  }
};
#endif  // TRTIS_ENABLE_GPU

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    LOG_ERROR << msg;
  }

  LOG_ERROR << "Usage: " << argv[0] << " [options]";
  LOG_ERROR << "\t-g Use GPU memory for input and output tensors";
  LOG_ERROR << "\t-r [model repository absolute path]";

  exit(1);
}

std::string
MemoryTypeString(TRTSERVER_Memory_Type memory_type)
{
  return (memory_type == TRTSERVER_MEMORY_CPU) ? "CPU memory" : "GPU memory";
}

TRTSERVER_Error*
ResponseAlloc(
    TRTSERVER_ResponseAllocator* allocator, void** buffer, void** buffer_userp,
    const char* tensor_name, size_t byte_size,
    TRTSERVER_Memory_Type memory_type, int64_t memory_type_id, void* userp)
{
  // Pass the tensor name with buffer_userp so we can show it when
  // releasing the buffer.

  // If 'byte_size' is zero just return 'buffer'==nullptr, we don't
  // need to do any other book-keeping.
  if (byte_size == 0) {
    *buffer = nullptr;
    *buffer_userp = nullptr;
    LOG_INFO << "allocated " << byte_size << " bytes for result tensor "
             << tensor_name;
  } else {
    void* allocated_ptr = nullptr;
    if (memory_type == TRTSERVER_MEMORY_CPU) {
      allocated_ptr = malloc(byte_size);
#ifdef TRTIS_ENABLE_GPU
    } else if (use_gpu_memory) {
      auto err = cudaMalloc(&allocated_ptr, byte_size);
      if (err != cudaSuccess) {
        LOG_INFO << "cudaMalloc failed: " << cudaGetErrorString(err);
        allocated_ptr = nullptr;
      }
#endif  // TRTIS_ENABLE_GPU
    }

    if (allocated_ptr != nullptr) {
      *buffer = allocated_ptr;
      *buffer_userp = new std::string(tensor_name);
      LOG_INFO << "allocated " << byte_size << " bytes in "
               << MemoryTypeString(memory_type) << " for result tensor "
               << tensor_name;
    }
  }

  return nullptr;  // Success
}

TRTSERVER_Error*
ResponseRelease(
    TRTSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRTSERVER_Memory_Type memory_type, int64_t memory_type_id)
{
  std::string* name = nullptr;
  if (buffer_userp != nullptr) {
    name = reinterpret_cast<std::string*>(buffer_userp);
  } else {
    name = new std::string("<unknown>");
  }

  LOG_INFO << "Releasing buffer " << buffer << " of size " << byte_size
           << " in " << MemoryTypeString(memory_type) << " for result '"
           << *name << "'";
  if (memory_type == TRTSERVER_MEMORY_CPU) {
    free(buffer);
#ifdef TRTIS_ENABLE_GPU
  } else if (use_gpu_memory) {
    auto err = cudaFree(buffer);
    if (err != cudaSuccess) {
      LOG_ERROR << "error: failed to cudaFree " << buffer << ": "
                << cudaGetErrorString(err);
    }
#endif  // TRTIS_ENABLE_GPU
  } else {
    LOG_ERROR << "error: unexpected buffer allocated in GPU memory";
  }

  delete name;

  return nullptr;  // Success
}

void
InferComplete(
    TRTSERVER_Server* server, TRTSERVER_InferenceResponse* response,
    void* userp)
{
  std::promise<TRTSERVER_InferenceResponse*>* p =
      reinterpret_cast<std::promise<TRTSERVER_InferenceResponse*>*>(userp);
  p->set_value(response);
  delete p;
}

TRTSERVER_Error*
ParseModelConfig(
    const ni::ModelConfig& config, bool* is_int, bool* is_torch_model)
{
  auto data_type = ni::TYPE_INVALID;
  for (const auto& input : config.input()) {
    if ((input.data_type() != ni::TYPE_INT32) &&
        (input.data_type() != ni::TYPE_FP32)) {
      return TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_UNSUPPORTED,
          "simple lib example only supports model with data type INT32 or "
          "FP32");
    }
    if (data_type == ni::TYPE_INVALID) {
      data_type = input.data_type();
    } else if (input.data_type() != data_type) {
      return TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_INVALID_ARG,
          "the inputs and outputs of 'simple' model must have the data type");
    }
  }
  for (const auto& output : config.output()) {
    if ((output.data_type() != ni::TYPE_INT32) &&
        (output.data_type() != ni::TYPE_FP32)) {
      return TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_UNSUPPORTED,
          "simple lib example only supports model with data type INT32 or "
          "FP32");
    } else if (output.data_type() != data_type) {
      return TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_INVALID_ARG,
          "the inputs and outputs of 'simple' model must have the data type");
    }
  }

  *is_int = (data_type == ni::TYPE_INT32);
  *is_torch_model = (config.platform() == "pytorch_libtorch");
  return nullptr;
}

template <typename T>
void
GenerateInputData(
    std::vector<char>* input0_data, std::vector<char>* input1_data)
{
  input0_data->resize(16 * sizeof(T));
  input1_data->resize(16 * sizeof(T));
  for (size_t i = 0; i < 16; ++i) {
    ((T*)input0_data->data())[i] = i;
    ((T*)input1_data->data())[i] = 1;
  }
}

template <typename T>
void
CompareResult(
    const std::string& output0_name, const std::string& output1_name,
    const void* input0, const void* input1, const void* output0,
    const void* output1)
{
  for (size_t i = 0; i < 16; ++i) {
    LOG_INFO << ((T*)input0)[i] << " + " << ((T*)input1)[i] << " = "
             << ((T*)output0)[i];
    LOG_INFO << ((T*)input0)[i] << " - " << ((T*)input1)[i] << " = "
             << ((T*)output1)[i];

    if ((((T*)input0)[i] + ((T*)input1)[i]) != ((T*)output0)[i]) {
      FAIL("incorrect sum in " + output0_name);
    }
    if ((((T*)input0)[i] - ((T*)input1)[i]) != ((T*)output1)[i]) {
      FAIL("incorrect difference in " + output1_name);
    }
  }
}

}  // namespace

int
main(int argc, char** argv)
{
  std::string model_repository_path;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "gr:")) != -1) {
    switch (opt) {
      case 'g':
        use_gpu_memory = true;
        break;
      case 'r':
        model_repository_path = optarg;
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  if (model_repository_path.empty()) {
    Usage(argv, "-r must be used to specify model repository path");
  }
#ifndef TRTIS_ENABLE_GPU
  if (use_gpu_memory) {
    Usage(argv, "-g can not be used without enabling GPU");
  }
#endif  // TRTIS_ENABLE_GPU

  // Create the server...
  TRTSERVER_ServerOptions* server_options = nullptr;
  FAIL_IF_ERR(
      TRTSERVER_ServerOptionsNew(&server_options), "creating server options");
  FAIL_IF_ERR(
      TRTSERVER_ServerOptionsSetModelRepositoryPath(
          server_options, model_repository_path.c_str()),
      "setting model repository path");

  TRTSERVER_Server* server_ptr = nullptr;
  FAIL_IF_ERR(
      TRTSERVER_ServerNew(&server_ptr, server_options), "creating server");
  FAIL_IF_ERR(
      TRTSERVER_ServerOptionsDelete(server_options), "deleting server options");

  std::shared_ptr<TRTSERVER_Server> server(server_ptr, TRTSERVER_ServerDelete);

  // Wait until the server is both live and ready.
  size_t health_iters = 0;
  while (true) {
    bool live, ready;
    FAIL_IF_ERR(
        TRTSERVER_ServerIsLive(server.get(), &live),
        "unable to get server liveness");
    FAIL_IF_ERR(
        TRTSERVER_ServerIsReady(server.get(), &ready),
        "unable to get server readiness");
    LOG_INFO << "Server Health: live " << live << ", ready " << ready;
    if (live && ready) {
      break;
    }

    if (++health_iters >= 10) {
      FAIL("failed to find healthy inference server");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // Print status of the server.
  {
    TRTSERVER_Protobuf* server_status_protobuf;
    FAIL_IF_ERR(
        TRTSERVER_ServerStatus(server.get(), &server_status_protobuf),
        "unable to get server status protobuf");
    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(
        TRTSERVER_ProtobufSerialize(
            server_status_protobuf, &buffer, &byte_size),
        "unable to serialize server status protobuf");

    ni::ServerStatus server_status;
    if (!server_status.ParseFromArray(buffer, byte_size)) {
      FAIL("error: failed to parse server status");
    }

    LOG_INFO << "Server Status:";
    LOG_INFO << server_status.DebugString();

    FAIL_IF_ERR(
        TRTSERVER_ProtobufDelete(server_status_protobuf),
        "deleting status protobuf");
  }

  // Wait for the simple model to become available.
  bool is_torch_model = false;
  bool is_int = true;
  while (true) {
    TRTSERVER_Protobuf* model_status_protobuf;
    FAIL_IF_ERR(
        TRTSERVER_ServerModelStatus(
            server.get(), "simple", &model_status_protobuf),
        "unable to get model status protobuf");
    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(
        TRTSERVER_ProtobufSerialize(model_status_protobuf, &buffer, &byte_size),
        "unable to serialize model status protobuf");

    ni::ServerStatus model_status;
    if (!model_status.ParseFromArray(buffer, byte_size)) {
      FAIL("error: failed to parse model status");
    }

    auto itr = model_status.model_status().find("simple");
    if (itr == model_status.model_status().end()) {
      FAIL("unable to find status for model 'simple'");
    }

    auto vitr = itr->second.version_status().find(1);
    if (vitr == itr->second.version_status().end()) {
      FAIL("unable to find version 1 status for model 'simple'");
    }

    LOG_INFO << "'simple' model is "
             << ni::ModelReadyState_Name(vitr->second.ready_state());
    if (vitr->second.ready_state() == ni::ModelReadyState::MODEL_READY) {
      FAIL_IF_ERR(
          ParseModelConfig(itr->second.config(), &is_int, &is_torch_model),
          "parsing model config");
      break;
    }

    // [TODO] do so before break
    FAIL_IF_ERR(
        TRTSERVER_ProtobufDelete(model_status_protobuf),
        "deleting status protobuf");

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // Create the allocator that will be used to allocate buffers for
  // the result tensors.
  TRTSERVER_ResponseAllocator* allocator = nullptr;
  FAIL_IF_ERR(
      TRTSERVER_ResponseAllocatorNew(
          &allocator, ResponseAlloc, ResponseRelease),
      "creating response allocator");

  // The inference request provides meta-data with an
  // InferRequestHeader and the actual data via a provider.
  const std::string model_name("simple");
  int64_t model_version = -1;  // latest

  ni::InferRequestHeader request_header;
  request_header.set_id(123);
  request_header.set_batch_size(1);

  auto input0 = request_header.add_input();
  input0->set_name(is_torch_model ? "INPUT__0" : "INPUT0");
  auto input1 = request_header.add_input();
  input1->set_name(is_torch_model ? "INPUT__1" : "INPUT1");

  auto output0 = request_header.add_output();
  output0->set_name(is_torch_model ? "OUTPUT__0" : "OUTPUT0");
  auto output1 = request_header.add_output();
  output1->set_name(is_torch_model ? "OUTPUT__1" : "OUTPUT1");

  std::string request_header_serialized;
  request_header.SerializeToString(&request_header_serialized);

  // Create the inference request provider which provides all the
  // input information needed for an inference.
  TRTSERVER_InferenceRequestProvider* request_provider = nullptr;
  FAIL_IF_ERR(
      TRTSERVER_InferenceRequestProviderNew(
          &request_provider, server.get(), model_name.c_str(), model_version,
          request_header_serialized.c_str(), request_header_serialized.size()),
      "creating inference request provider");

  // Create the data for the two input tensors. Initialize the first
  // to unique integers and the second to all ones.
  std::vector<char> input0_data;
  std::vector<char> input1_data;
  if (is_int) {
    GenerateInputData<int32_t>(&input0_data, &input1_data);
  } else {
    GenerateInputData<float>(&input0_data, &input1_data);
  }

  size_t input0_size = input0_data.size();
  size_t input1_size = input1_data.size();

  const void* input0_base = &input0_data[0];
  const void* input1_base = &input1_data[0];
  auto memory_type = TRTSERVER_MEMORY_CPU;
#ifdef TRTIS_ENABLE_GPU
  std::unique_ptr<void, decltype(gpu_data_deleter)> input0_gpu(
      nullptr, gpu_data_deleter);
  std::unique_ptr<void, decltype(gpu_data_deleter)> input1_gpu(
      nullptr, gpu_data_deleter);
  if (use_gpu_memory) {
    void* dst;
    FAIL_IF_CUDA_ERR(
        cudaMalloc(&dst, input0_size), "allocating GPU memory for INPUT0 data");
    input0_gpu.reset(dst);
    FAIL_IF_CUDA_ERR(
        cudaMemcpy(dst, &input0_data[0], input0_size, cudaMemcpyHostToDevice),
        "setting INPUT0 data in GPU memory");
    FAIL_IF_CUDA_ERR(
        cudaMalloc(&dst, input1_size), "allocating GPU memory for INPUT1 data");
    input1_gpu.reset(dst);
    FAIL_IF_CUDA_ERR(
        cudaMemcpy(dst, &input1_data[0], input1_size, cudaMemcpyHostToDevice),
        "setting INPUT1 data in GPU memory");
  }

  input0_base = use_gpu_memory ? input0_gpu.get() : &input0_data[0];
  input1_base = use_gpu_memory ? input1_gpu.get() : &input1_data[0];
  memory_type = use_gpu_memory ? TRTSERVER_MEMORY_GPU : TRTSERVER_MEMORY_CPU;
#endif  // TRTIS_ENABLE_GPU

  FAIL_IF_ERR(
      TRTSERVER_InferenceRequestProviderSetInputData(
          request_provider, input0->name().c_str(), input0_base, input0_size,
          memory_type),
      "assigning INPUT0 data");
  FAIL_IF_ERR(
      TRTSERVER_InferenceRequestProviderSetInputData(
          request_provider, input1->name().c_str(), input1_base, input1_size,
          memory_type),
      "assigning INPUT1 data");

  // Perform inference...
  auto p = new std::promise<TRTSERVER_InferenceResponse*>();
  std::future<TRTSERVER_InferenceResponse*> completed = p->get_future();

  FAIL_IF_ERR(
      TRTSERVER_ServerInferAsync(
          server.get(), request_provider, allocator,
          nullptr /* response_allocator_userp */, InferComplete,
          reinterpret_cast<void*>(p)),
      "running inference");

  // The request provider can be deleted immediately after the
  // ServerInferAsync call returns.
  FAIL_IF_ERR(
      TRTSERVER_InferenceRequestProviderDelete(request_provider),
      "deleting inference request provider");

  // Wait for the inference response and check the status.
  TRTSERVER_InferenceResponse* response = completed.get();
  FAIL_IF_ERR(TRTSERVER_InferenceResponseStatus(response), "response");

  // Print the response header metadata.
  {
    TRTSERVER_Protobuf* response_protobuf;
    FAIL_IF_ERR(
        TRTSERVER_InferenceResponseHeader(response, &response_protobuf),
        "unable to get response header protobuf");
    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(
        TRTSERVER_ProtobufSerialize(response_protobuf, &buffer, &byte_size),
        "unable to serialize response header protobuf");

    ni::InferResponseHeader response_header;
    if (!response_header.ParseFromArray(buffer, byte_size)) {
      FAIL("error: failed to parse response header");
    }

    LOG_INFO << "Model \"simple\" response header:";
    LOG_INFO << response_header.DebugString();

    FAIL_IF_ERR(
        TRTSERVER_ProtobufDelete(response_protobuf),
        "deleting response protobuf");
  }

  // Check the output tensor values...
  // Note that depending on whether the backend supports outputs in GPU memory,
  // the output tensor may be in CPU memory even if -g flag is set.
  const void* output0_content;
  size_t output0_byte_size;
  TRTSERVER_Memory_Type output0_memory_type;
  FAIL_IF_ERR(
      TRTSERVER_InferenceResponseOutputData(
          response, output0->name().c_str(), &output0_content,
          &output0_byte_size, &output0_memory_type),
      "getting output0 result");
  if (output0_byte_size != input0_size) {
    FAIL(
        "unexpected output0 byte-size, expected " +
        std::to_string(input0_size) + ", got " +
        std::to_string(output0_byte_size));
  } else if (
      (!use_gpu_memory) && (output0_memory_type == TRTSERVER_MEMORY_GPU)) {
    FAIL(
        "unexpected output0 memory type, expected to be allocated "
        "in " +
        MemoryTypeString(TRTSERVER_MEMORY_CPU) + ", got " +
        MemoryTypeString(output0_memory_type));
  }

  const void* output1_content;
  size_t output1_byte_size;
  TRTSERVER_Memory_Type output1_memory_type;
  FAIL_IF_ERR(
      TRTSERVER_InferenceResponseOutputData(
          response, output1->name().c_str(), &output1_content,
          &output1_byte_size, &output1_memory_type),
      "getting output1 result");
  if (output1_byte_size != input1_size) {
    FAIL(
        "unexpected output1 byte-size, expected " +
        std::to_string(input1_size) + ", got " +
        std::to_string(output1_byte_size));
  } else if (
      (!use_gpu_memory) && (output1_memory_type == TRTSERVER_MEMORY_GPU)) {
    FAIL(
        "unexpected output1 memory type, expected to be allocated "
        "in " +
        MemoryTypeString(TRTSERVER_MEMORY_CPU) + ", got " +
        MemoryTypeString(output1_memory_type));
  }

  const void* output0_result = output0_content;
  const void* output1_result = output1_content;

#ifdef TRTIS_ENABLE_GPU
  // Different from CPU memory, outputs in GPU memory must be copied to CPU
  // memory to be read directly.
  std::vector<char> output0_data(output0_byte_size);
  std::vector<char> output1_data(output1_byte_size);
  if (output0_memory_type == TRTSERVER_MEMORY_CPU) {
    LOG_INFO << "OUTPUT0 are stored in CPU memory";
  } else {
    LOG_INFO << "OUTPUT0 are stored in GPU memory";
    FAIL_IF_CUDA_ERR(
        cudaMemcpy(
            &output0_data[0], output0_content, output0_byte_size,
            cudaMemcpyDeviceToHost),
        "setting INPUT0 data in GPU memory");
    output0_result = &output0_data[0];
  }

  if (output1_memory_type == TRTSERVER_MEMORY_CPU) {
    LOG_INFO << "OUTPUT1 are stored in CPU memory";
  } else {
    LOG_INFO << "OUTPUT1 are stored in GPU memory";
    FAIL_IF_CUDA_ERR(
        cudaMemcpy(
            &output1_data[0], output1_content, output1_byte_size,
            cudaMemcpyDeviceToHost),
        "setting INPUT0 data in GPU memory");
    output1_result = &output1_data[0];
  }
#endif  // TRTIS_ENABLE_GPU

  if (is_int) {
    CompareResult<int32_t>(
        output0->name(), output1->name(), &input0_data[0], &input1_data[0],
        output0_result, output1_result);
  } else {
    CompareResult<float>(
        output0->name(), output1->name(), &input0_data[0], &input1_data[0],
        output0_result, output1_result);
  }

  FAIL_IF_ERR(
      TRTSERVER_InferenceResponseDelete(response),
      "deleting inference response");

  FAIL_IF_ERR(
      TRTSERVER_ResponseAllocatorDelete(allocator),
      "deleting response allocator");

  return 0;
}
