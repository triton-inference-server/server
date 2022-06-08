// Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "common.h"
#include "triton/core/tritonserver.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace ni = triton::server;

namespace {

bool enforce_memory_type = false;
TRITONSERVER_MemoryType requested_memory_type;

#ifdef TRITON_ENABLE_GPU
static auto cuda_data_deleter = [](void* data) {
  if (data != nullptr) {
    cudaPointerAttributes attr;
    auto cuerr = cudaPointerGetAttributes(&attr, data);
    if (cuerr != cudaSuccess) {
      std::cerr << "error: failed to get CUDA pointer attribute of " << data
                << ": " << cudaGetErrorString(cuerr) << std::endl;
    }
    if (attr.type == cudaMemoryTypeDevice) {
      cuerr = cudaFree(data);
    } else if (attr.type == cudaMemoryTypeHost) {
      cuerr = cudaFreeHost(data);
    }
    if (cuerr != cudaSuccess) {
      std::cerr << "error: failed to release CUDA pointer " << data << ": "
                << cudaGetErrorString(cuerr) << std::endl;
    }
  }
};
#endif  // TRITON_ENABLE_GPU

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-m <\"system\"|\"pinned\"|gpu>"
            << " Enforce the memory type for input and output tensors."
            << " If not specified, inputs will be in system memory and outputs"
            << " will be based on the model's preferred type." << std::endl;
  std::cerr << "\t-v Enable verbose logging" << std::endl;
  std::cerr << "\t-r [model repository absolute path]" << std::endl;
  std::cerr << "\t-t Thread count." << std::endl;
  std::cerr << "\t-l Number of loops to run in each thread." << std::endl;

  exit(1);
}

TRITONSERVER_Error*
ResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  // Initially attempt to make the actual memory type and id that we
  // allocate be the same as preferred memory type
  *actual_memory_type = preferred_memory_type;
  *actual_memory_type_id = preferred_memory_type_id;

  // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
  // need to do any other book-keeping.
  if (byte_size == 0) {
    *buffer = nullptr;
    *buffer_userp = nullptr;
    std::cout << "allocated " << byte_size << " bytes for result tensor "
              << tensor_name << std::endl;
  } else {
    void* allocated_ptr = nullptr;
    if (enforce_memory_type) {
      *actual_memory_type = requested_memory_type;
    }

    switch (*actual_memory_type) {
#ifdef TRITON_ENABLE_GPU
      case TRITONSERVER_MEMORY_CPU_PINNED: {
        auto err = cudaSetDevice(*actual_memory_type_id);
        if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
            (err != cudaErrorInsufficientDriver)) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "unable to recover current CUDA device: " +
                  std::string(cudaGetErrorString(err)))
                  .c_str());
        }

        err = cudaHostAlloc(&allocated_ptr, byte_size, cudaHostAllocPortable);
        if (err != cudaSuccess) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "cudaHostAlloc failed: " +
                  std::string(cudaGetErrorString(err)))
                  .c_str());
        }
        break;
      }

      case TRITONSERVER_MEMORY_GPU: {
        auto err = cudaSetDevice(*actual_memory_type_id);
        if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
            (err != cudaErrorInsufficientDriver)) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "unable to recover current CUDA device: " +
                  std::string(cudaGetErrorString(err)))
                  .c_str());
        }

        err = cudaMalloc(&allocated_ptr, byte_size);
        if (err != cudaSuccess) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "cudaMalloc failed: " + std::string(cudaGetErrorString(err)))
                  .c_str());
        }
        break;
      }
#endif  // TRITON_ENABLE_GPU

      // Use CPU memory if the requested memory type is unknown
      // (default case).
      case TRITONSERVER_MEMORY_CPU:
      default: {
        *actual_memory_type = TRITONSERVER_MEMORY_CPU;
        allocated_ptr = malloc(byte_size);
        break;
      }
    }

    // Pass the tensor name with buffer_userp so we can show it when
    // releasing the buffer.
    if (allocated_ptr != nullptr) {
      *buffer = allocated_ptr;
      *buffer_userp = new std::string(tensor_name);
      std::cout << "allocated " << byte_size << " bytes in "
                << TRITONSERVER_MemoryTypeString(*actual_memory_type)
                << " for result tensor " << tensor_name << std::endl;
    }
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
ResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  std::string* name = nullptr;
  if (buffer_userp != nullptr) {
    name = reinterpret_cast<std::string*>(buffer_userp);
  } else {
    name = new std::string("<unknown>");
  }

  std::cout << "Releasing buffer " << buffer << " of size " << byte_size
            << " in " << TRITONSERVER_MemoryTypeString(memory_type)
            << " for result '" << *name << "'" << std::endl;
  switch (memory_type) {
    case TRITONSERVER_MEMORY_CPU:
      free(buffer);
      break;
#ifdef TRITON_ENABLE_GPU
    case TRITONSERVER_MEMORY_CPU_PINNED: {
      auto err = cudaSetDevice(memory_type_id);
      if (err == cudaSuccess) {
        err = cudaFreeHost(buffer);
      }
      if (err != cudaSuccess) {
        std::cerr << "error: failed to cudaFree " << buffer << ": "
                  << cudaGetErrorString(err) << std::endl;
      }
      break;
    }
    case TRITONSERVER_MEMORY_GPU: {
      auto err = cudaSetDevice(memory_type_id);
      if (err == cudaSuccess) {
        err = cudaFree(buffer);
      }
      if (err != cudaSuccess) {
        std::cerr << "error: failed to cudaFree " << buffer << ": "
                  << cudaGetErrorString(err) << std::endl;
      }
      break;
    }
#endif  // TRITON_ENABLE_GPU
    default:
      std::cerr << "error: unexpected buffer allocated in CUDA managed memory"
                << std::endl;
      break;
  }

  delete name;

  return nullptr;  // Success
}

void
InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  // We reuse the request so we don't delete it here.
}

void
InferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  if (response != nullptr) {
    // Send 'response' to the future.
    std::promise<TRITONSERVER_InferenceResponse*>* p =
        reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
    p->set_value(response);
    delete p;
  }
}

TRITONSERVER_Error*
ParseModelMetadata(const rapidjson::Document& model_metadata)
{
  std::string seen_data_type;
  for (const auto& input : model_metadata["inputs"].GetArray()) {
    if (strcmp(input["datatype"].GetString(), "FP32")) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "multi-server example only supports model with data type FP32");
    }
    if (seen_data_type.empty()) {
      seen_data_type = input["datatype"].GetString();
    } else if (strcmp(seen_data_type.c_str(), input["datatype"].GetString())) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "the inputs and outputs of model must have the data type");
    }
  }
  for (const auto& output : model_metadata["outputs"].GetArray()) {
    if (strcmp(output["datatype"].GetString(), "FP32")) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "multi-server example only supports model with data type FP32");
    } else if (strcmp(seen_data_type.c_str(), output["datatype"].GetString())) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "the inputs and outputs of model must have the data type");
    }
  }

  return nullptr;
}

void
GenerateInputData(
    std::vector<float>* input0_data, std::vector<float>* input1_data)
{
  input0_data->resize(16);
  input1_data->resize(16);
  for (size_t i = 0; i < 16; ++i) {
    input0_data->data()[i] = i;
    input1_data->data()[i] = 1;
  }
}

void
CompareResult(
    const std::string& output0_name, const std::string& output1_name,
    const float* input0, const float* input1, const float* output0,
    const float* output1)
{
  for (size_t i = 0; i < 16; ++i) {
    std::cout << input0[i] << " + " << input1[i] << " = " << output0[i]
              << std::endl;
    std::cout << input0[i] << " - " << input1[i] << " = " << output1[i]
              << std::endl;

    if ((input0[i] + input1[i]) != output0[i]) {
      FAIL("incorrect sum in " + output0_name);
    }
    if ((input0[i] - input1[i]) != output1[i]) {
      FAIL("incorrect difference in " + output1_name);
    }
  }
}

void
Check(
    TRITONSERVER_InferenceResponse* response,
    const std::vector<float>& input0_data,
    const std::vector<float>& input1_data, const std::string& output0,
    const std::string& output1, const size_t expected_byte_size,
    const TRITONSERVER_DataType expected_datatype)
{
  std::unordered_map<std::string, std::vector<float>> output_data;

  uint32_t output_count;
  FAIL_IF_ERR(
      TRITONSERVER_InferenceResponseOutputCount(response, &output_count),
      "getting number of response outputs");
  if (output_count != 2) {
    FAIL("expecting 2 response outputs, got " + std::to_string(output_count));
  }

  for (uint32_t idx = 0; idx < output_count; ++idx) {
    const char* cname;
    TRITONSERVER_DataType datatype;
    const int64_t* shape;
    uint64_t dim_count;
    const void* base;
    size_t byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    void* userp;

    FAIL_IF_ERR(
        TRITONSERVER_InferenceResponseOutput(
            response, idx, &cname, &datatype, &shape, &dim_count, &base,
            &byte_size, &memory_type, &memory_type_id, &userp),
        "getting output info");

    if (cname == nullptr) {
      FAIL("unable to get output name");
    }

    std::string name(cname);
    if ((name != output0) && (name != output1)) {
      FAIL("unexpected output '" + name + "'");
    }

    if ((dim_count != 2) || (shape[0] != 1) || (shape[1] != 16)) {
      FAIL("unexpected shape for '" + name + "'");
    }

    if (datatype != expected_datatype) {
      FAIL(
          "unexpected datatype '" +
          std::string(TRITONSERVER_DataTypeString(datatype)) + "' for '" +
          name + "'");
    }

    if (byte_size != expected_byte_size) {
      FAIL(
          "unexpected byte-size, expected " +
          std::to_string(expected_byte_size) + ", got " +
          std::to_string(byte_size) + " for " + name);
    }

    if (enforce_memory_type && (memory_type != requested_memory_type)) {
      FAIL(
          "unexpected memory type, expected to be allocated in " +
          std::string(TRITONSERVER_MemoryTypeString(requested_memory_type)) +
          ", got " + std::string(TRITONSERVER_MemoryTypeString(memory_type)) +
          ", id " + std::to_string(memory_type_id) + " for " + name);
    }

    // We make a copy of the data here... which we could avoid for
    // performance reasons but ok for this example.
    std::vector<float>& odata = output_data[name];
    switch (memory_type) {
      case TRITONSERVER_MEMORY_CPU: {
        std::cout << name << " is stored in system memory" << std::endl;
        const float* cbase = reinterpret_cast<const float*>(base);
        odata.assign(cbase, cbase + (byte_size / sizeof(float)));
        break;
      }

      case TRITONSERVER_MEMORY_CPU_PINNED: {
        std::cout << name << " is stored in pinned memory" << std::endl;
        const float* cbase = reinterpret_cast<const float*>(base);
        odata.assign(cbase, cbase + (byte_size / sizeof(float)));
        break;
      }

#ifdef TRITON_ENABLE_GPU
      case TRITONSERVER_MEMORY_GPU: {
        std::cout << name << " is stored in GPU memory" << std::endl;
        odata.reserve(byte_size);
        FAIL_IF_CUDA_ERR(
            cudaMemcpy(&odata[0], base, byte_size, cudaMemcpyDeviceToHost),
            "getting " + name + " data from GPU memory");
        break;
      }
#endif

      default:
        FAIL("unexpected memory type");
    }
  }

  CompareResult(
      output0, output1, &input0_data[0], &input1_data[0],
      output_data[output0].data(), output_data[output1].data());
}

}  // namespace

void
SetServerOptions(
    TRITONSERVER_ServerOptions** server_options, bool verbose_level,
    std::string model_repository_path1, std::string model_repository_path2)
{
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsNew(server_options), "creating server options");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetModelRepositoryPath(
          *server_options, model_repository_path1.c_str()),
      "setting model repository path");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetModelRepositoryPath(
          *server_options, model_repository_path2.c_str()),
      "setting model repository path");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetLogVerbose(*server_options, verbose_level),
      "setting verbose logging level");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetMetrics(*server_options, true),
      "failed to enable metrics");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetStrictReadiness(*server_options, true),
      "failed to set strict readiness");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetStrictModelConfig(*server_options, true),
      "failed to set strict model config");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetModelControlMode(
          *server_options, TRITONSERVER_MODEL_CONTROL_EXPLICIT),
      "failed to set model control mode to explicit");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetBackendDirectory(
          *server_options, "/opt/tritonserver/backends"),
      "setting backend directory");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
          *server_options, "/opt/tritonserver/repoagents"),
      "setting repository agent directory");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetStrictModelConfig(*server_options, true),
      "setting strict model configuration");
#ifdef TRITON_ENABLE_GPU
  double min_compute_capability = TRITON_MIN_COMPUTE_CAPABILITY;
#else
  double min_compute_capability = 0;
#endif  // TRITON_ENABLE_GPU
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
          *server_options, min_compute_capability),
      "setting minimum supported CUDA compute capability");
}

void
CheckServerLiveAndReady(std::shared_ptr<TRITONSERVER_Server> server)
{
  size_t wait_seconds = 0;
  while (true) {
    bool live, ready;
    FAIL_IF_ERR(
        TRITONSERVER_ServerIsLive(server.get(), &live),
        "unable to get server liveness");
    FAIL_IF_ERR(
        TRITONSERVER_ServerIsReady(server.get(), &ready),
        "unable to get server readiness");
    std::cout << "Server Health: live " << live << ", ready " << ready
              << std::endl;
    if (live && ready) {
      break;
    }

    if (++wait_seconds >= 10) {
      FAIL("failed to find healthy inference server");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
}

void
PrintServerStatus(std::shared_ptr<TRITONSERVER_Server> server)
{
  TRITONSERVER_Message* server_metadata_message;
  FAIL_IF_ERR(
      TRITONSERVER_ServerMetadata(server.get(), &server_metadata_message),
      "unable to get server metadata message");
  const char* buffer;
  size_t byte_size;
  FAIL_IF_ERR(
      TRITONSERVER_MessageSerializeToJson(
          server_metadata_message, &buffer, &byte_size),
      "unable to serialize server metadata message");

  std::cout << "Server Status:" << std::endl;
  std::cout << std::string(buffer, byte_size) << std::endl;

  FAIL_IF_ERR(
      TRITONSERVER_MessageDelete(server_metadata_message),
      "deleting status metadata");
}

void
AwaitModelReady(
    std::shared_ptr<TRITONSERVER_Server> server, const std::string model_name)
{
  bool is_ready = false;
  size_t wait_seconds = 0;
  while (!is_ready) {
    FAIL_IF_ERR(
        TRITONSERVER_ServerModelIsReady(
            server.get(), model_name.c_str(), 1, &is_ready),
        "unable to get model readiness");
    if (!is_ready) {
      if (++wait_seconds >= 5) {
        FAIL("model failed to be ready in 5 seconds");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      continue;
    }

    TRITONSERVER_Message* model_metadata_message;
    FAIL_IF_ERR(
        TRITONSERVER_ServerModelMetadata(
            server.get(), model_name.c_str(), 1, &model_metadata_message),
        "unable to get model metadata message");
    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(
        TRITONSERVER_MessageSerializeToJson(
            model_metadata_message, &buffer, &byte_size),
        "unable to serialize model status protobuf");

    rapidjson::Document model_metadata;
    model_metadata.Parse(buffer, byte_size);
    if (model_metadata.HasParseError()) {
      FAIL(
          "error: failed to parse model metadata from JSON: " +
          std::string(GetParseError_En(model_metadata.GetParseError())) +
          " at " + std::to_string(model_metadata.GetErrorOffset()));
    }

    FAIL_IF_ERR(
        TRITONSERVER_MessageDelete(model_metadata_message),
        "deleting status protobuf");

    if (strcmp(model_metadata["name"].GetString(), model_name.c_str())) {
      FAIL("unable to find metadata for model");
    }

    bool found_version = false;
    if (model_metadata.HasMember("versions")) {
      for (const auto& version : model_metadata["versions"].GetArray()) {
        if (strcmp(version.GetString(), "1") == 0) {
          found_version = true;
          break;
        }
      }
    }
    if (!found_version) {
      FAIL("unable to find version 1 status for model");
    }

    FAIL_IF_ERR(ParseModelMetadata(model_metadata), "parsing model metadata");
  }
}

void
RunInferenceAndValidate(
    std::shared_ptr<TRITONSERVER_Server> server,
    TRITONSERVER_ResponseAllocator* allocator, const std::string model_name)
{
  TRITONSERVER_InferenceRequest* irequest = nullptr;
  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestNew(
          &irequest, server.get(), model_name.c_str(), -1 /* model_version */),
      "creating inference request");

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestSetId(irequest, "my_request_id"),
      "setting ID for the request");

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestSetReleaseCallback(
          irequest, InferRequestComplete, nullptr /* request_release_userp */),
      "setting request release callback");

  // Inputs
  auto input0 = "INPUT0";
  auto input1 = "INPUT1";

  std::vector<int64_t> input0_shape({1, 16});
  std::vector<int64_t> input1_shape({1, 16});

  const TRITONSERVER_DataType datatype = TRITONSERVER_TYPE_FP32;

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAddInput(
          irequest, input0, datatype, &input0_shape[0], input0_shape.size()),
      "setting input 0 meta-data for the request");
  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAddInput(
          irequest, input1, datatype, &input1_shape[0], input1_shape.size()),
      "setting input 1 meta-data for the request");

  auto output0 = "OUTPUT0";
  auto output1 = "OUTPUT1";

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output0),
      "requesting output 0 for the request");
  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output1),
      "requesting output 1 for the request");

  // Create the data for the two input tensors. Initialize the first
  // to unique values and the second to all ones.
  std::vector<float> input0_data;
  std::vector<float> input1_data;
  GenerateInputData(&input0_data, &input1_data);

  size_t input0_size = input0_data.size() * 4;
  size_t input1_size = input1_data.size() * 4;

  const void* input0_base = &input0_data[0];
  const void* input1_base = &input1_data[0];
#ifdef TRITON_ENABLE_GPU
  std::unique_ptr<void, decltype(cuda_data_deleter)> input0_gpu(
      nullptr, cuda_data_deleter);
  std::unique_ptr<void, decltype(cuda_data_deleter)> input1_gpu(
      nullptr, cuda_data_deleter);
  bool use_cuda_memory =
      (enforce_memory_type &&
       (requested_memory_type != TRITONSERVER_MEMORY_CPU));
  if (use_cuda_memory) {
    FAIL_IF_CUDA_ERR(cudaSetDevice(0), "setting CUDA device to device 0");
    if (requested_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
      void* dst;
      FAIL_IF_CUDA_ERR(
          cudaMalloc(&dst, input0_size),
          "allocating GPU memory for INPUT0 data");
      input0_gpu.reset(dst);
      FAIL_IF_CUDA_ERR(
          cudaMemcpy(dst, &input0_data[0], input0_size, cudaMemcpyHostToDevice),
          "setting INPUT0 data in GPU memory");
      FAIL_IF_CUDA_ERR(
          cudaMalloc(&dst, input1_size),
          "allocating GPU memory for INPUT1 data");
      input1_gpu.reset(dst);
      FAIL_IF_CUDA_ERR(
          cudaMemcpy(dst, &input1_data[0], input1_size, cudaMemcpyHostToDevice),
          "setting INPUT1 data in GPU memory");
    } else {
      void* dst;
      FAIL_IF_CUDA_ERR(
          cudaHostAlloc(&dst, input0_size, cudaHostAllocPortable),
          "allocating pinned memory for INPUT0 data");
      input0_gpu.reset(dst);
      FAIL_IF_CUDA_ERR(
          cudaMemcpy(dst, &input0_data[0], input0_size, cudaMemcpyHostToHost),
          "setting INPUT0 data in pinned memory");
      FAIL_IF_CUDA_ERR(
          cudaHostAlloc(&dst, input1_size, cudaHostAllocPortable),
          "allocating pinned memory for INPUT1 data");
      input1_gpu.reset(dst);
      FAIL_IF_CUDA_ERR(
          cudaMemcpy(dst, &input1_data[0], input1_size, cudaMemcpyHostToHost),
          "setting INPUT1 data in pinned memory");
    }
  }

  input0_base = use_cuda_memory ? input0_gpu.get() : &input0_data[0];
  input1_base = use_cuda_memory ? input1_gpu.get() : &input1_data[0];
#endif  // TRITON_ENABLE_GPU

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAppendInputData(
          irequest, input0, input0_base, input0_size, requested_memory_type,
          0 /* memory_type_id */),
      "assigning INPUT0 data");
  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAppendInputData(
          irequest, input1, input1_base, input1_size, requested_memory_type,
          0 /* memory_type_id */),
      "assigning INPUT1 data");

  // Perform inference...
  {
    auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
    std::future<TRITONSERVER_InferenceResponse*> completed = p->get_future();

    FAIL_IF_ERR(
        TRITONSERVER_InferenceRequestSetResponseCallback(
            irequest, allocator, nullptr /* response_allocator_userp */,
            InferResponseComplete, reinterpret_cast<void*>(p)),
        "setting response callback");

    FAIL_IF_ERR(
        TRITONSERVER_ServerInferAsync(
            server.get(), irequest, nullptr /* trace */),
        "running inference");

    // Wait for the inference to complete.
    TRITONSERVER_InferenceResponse* completed_response = completed.get();

    FAIL_IF_ERR(
        TRITONSERVER_InferenceResponseError(completed_response),
        "response status");

    Check(
        completed_response, input0_data, input1_data, output0, output1,
        input0_size, datatype);

    FAIL_IF_ERR(
        TRITONSERVER_InferenceResponseDelete(completed_response),
        "deleting inference response");
  }

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestDelete(irequest),
      "deleting inference request");
}

void
PrintModelStats(
    std::shared_ptr<TRITONSERVER_Server> server, const std::string model_name)
{
  TRITONSERVER_Message* model_stats_message = nullptr;

  FAIL_IF_ERR(
      TRITONSERVER_ServerModelStatistics(
          server.get(), model_name.c_str(), -1 /* model_version */,
          &model_stats_message),
      "unable to get model stats message");
  const char* buffer;
  size_t byte_size;
  FAIL_IF_ERR(
      TRITONSERVER_MessageSerializeToJson(
          model_stats_message, &buffer, &byte_size),
      "unable to serialize server metadata message");

  std::cout << "Model '" << model_name << "' Stats:" << std::endl;
  std::cout << std::string(buffer, byte_size) << std::endl;

  FAIL_IF_ERR(
      TRITONSERVER_MessageDelete(model_stats_message),
      "deleting model stats message");
}

void
CreateAndRunTritonserverInstance(
    std::vector<std::string> model_repository_paths, size_t thread_id,
    bool verbose_level)
{
  TRITONSERVER_ServerOptions* server_options = nullptr;

  SetServerOptions(
      &server_options, verbose_level, model_repository_paths[0],
      model_repository_paths[thread_id]);

  TRITONSERVER_Server* server_ptr = nullptr;
  FAIL_IF_ERR(
      TRITONSERVER_ServerNew(&server_ptr, server_options),
      "creating server instance no. " + std::to_string(thread_id));
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsDelete(server_options),
      "deleting server options");

  std::shared_ptr<TRITONSERVER_Server> server(
      server_ptr, TRITONSERVER_ServerDelete);

  // Wait and until the servers are both live and ready.
  CheckServerLiveAndReady(server);

  // Print status of the servers.
  PrintServerStatus(server);
  std::string model1 = "simple1",
              model2 = "simple" + std::to_string(thread_id + 1);

  // Load models in server.
  FAIL_IF_ERR(
      TRITONSERVER_ServerLoadModel(server.get(), model1.c_str()),
      "failed to load model");
  FAIL_IF_ERR(
      TRITONSERVER_ServerLoadModel(server.get(), model2.c_str()),
      "failed to load model");

  // Wait for the models to become available.
  AwaitModelReady(server, model1.c_str());
  AwaitModelReady(server, model2.c_str());

  // Create the allocator that will be used to allocate buffers for
  // the result tensors.
  TRITONSERVER_ResponseAllocator* allocator = nullptr;
  FAIL_IF_ERR(
      TRITONSERVER_ResponseAllocatorNew(
          &allocator, ResponseAlloc, ResponseRelease, nullptr /* start_fn */),
      "creating response allocator");

  // Inference
  RunInferenceAndValidate(server, allocator, model1.c_str());
  RunInferenceAndValidate(server, allocator, model2.c_str());

  FAIL_IF_ERR(
      TRITONSERVER_ResponseAllocatorDelete(allocator),
      "deleting response allocator");

  // Print Model Statistics for all models
  PrintModelStats(server, model1.c_str());
  PrintModelStats(server, model2.c_str());

  // Unload models in both servers.
  FAIL_IF_ERR(
      TRITONSERVER_ServerUnloadModel(server.get(), model1.c_str()),
      "failed to unload model");
  FAIL_IF_ERR(
      TRITONSERVER_ServerUnloadModel(server.get(), model2.c_str()),
      "failed to unload model");

  std::string wrong_model;
  if (thread_id == (model_repository_paths.size() + 1)) {
    wrong_model = "simple2";
  } else {
    wrong_model = "simple" + std::to_string(thread_id + 2);
  }

  // Try to load wrong model. Expected to fail
  TRITONSERVER_Error* err =
      TRITONSERVER_ServerLoadModel(server.get(), wrong_model.c_str());
  if (err == nullptr) {
    FAIL("Success when expected to failed to load wrong model");
  }
}

static volatile std::atomic<int> counter(0);
static std::mutex mutex;
std::condition_variable cv;

void
RepeatedlyCreateAndRunInstance(
    std::vector<std::string> model_repository_paths, size_t thread_id,
    size_t loops, bool verbose_level)
{
  std::unique_lock<std::mutex> lock(mutex);
  counter++;
  cv.wait(lock);

  for (size_t i = 0; i < loops; i++) {
    CreateAndRunTritonserverInstance(
        model_repository_paths, thread_id, verbose_level);
  }
}

int
main(int argc, char** argv)
{
  std::vector<std::string> model_repository_paths;
  int verbose_level = 0;
  int thread_count = 2;
  int loops = 1;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vm:r:t:l:")) != -1) {
    switch (opt) {
      case 'm': {
        enforce_memory_type = true;
        if (!strcmp(optarg, "system")) {
          requested_memory_type = TRITONSERVER_MEMORY_CPU;
        } else if (!strcmp(optarg, "pinned")) {
          requested_memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
        } else if (!strcmp(optarg, "gpu")) {
          requested_memory_type = TRITONSERVER_MEMORY_GPU;
        } else {
          Usage(
              argv,
              "-m must be used to specify one of the following types:"
              " <\"system\"|\"pinned\"|gpu>");
        }
        break;
      }
      case 'r':
        model_repository_paths.push_back(optarg);
        break;
      case 't':
        thread_count = std::stoi(optarg);
        break;
      case 'l':
        loops = std::stoi(optarg);
        break;
      case 'v':
        verbose_level = 1;
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  if ((thread_count < 1) && (loops < 1)) {
    Usage(argv, "thread_count and loops must be > 1");
  }

  // model repository paths must be 'thread_count' + 1
  if (int(model_repository_paths.size() - 1) != thread_count) {
    Usage(
        argv, "-r must be used to specify " + std::to_string(thread_count + 1) +
                  " model repository paths, " + std::to_string(thread_count) +
                  " unique paths and 1 common");
  }

  for (const auto& repo_path : model_repository_paths) {
    if (repo_path.empty()) {
      Usage(argv, "model repository paths must not be empty");
    }
  }
#ifndef TRITON_ENABLE_GPU
  if (enforce_memory_type && requested_memory_type != TRITONSERVER_MEMORY_CPU) {
    Usage(argv, "-m can only be set to \"system\" without enabling GPU");
  }
#endif  // TRITON_ENABLE_GPU

  // Check API version.
  uint32_t api_version_major, api_version_minor;
  FAIL_IF_ERR(
      TRITONSERVER_ApiVersion(&api_version_major, &api_version_minor),
      "getting Triton API version");
  if ((TRITONSERVER_API_VERSION_MAJOR != api_version_major) ||
      (TRITONSERVER_API_VERSION_MINOR > api_version_minor)) {
    FAIL("triton server API version mismatch");
  }

  // Create 'thread_count' number of instances of the server with 1 common and 1
  // unique repo each
  std::thread tritonservers[thread_count];
  for (int i = 0; i < thread_count; i++) {
    tritonservers[i] = std::thread(
        &RepeatedlyCreateAndRunInstance, model_repository_paths, size_t(i + 1),
        size_t(loops), verbose_level);
  }
  while (counter < thread_count) {
    usleep(50);
  }

  {
    std::unique_lock<std::mutex> lock(mutex);
    cv.notify_all();
  }

  for (int i = 0; i < thread_count; ++i) {
    tritonservers[i].join();
  }

  return 0;
}
