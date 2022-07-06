// Copyright 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime_api.h>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <unistd.h>
#include <chrono>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include "common.h"
#include "triton/core/tritonserver.h"

static_assert(
    TRITON_MIN_COMPUTE_CAPABILITY >= 1.0,
    "Invalid TRITON_MIN_COMPUTE_CAPABILITY specified");

namespace ni = triton::server;

namespace {

struct IOSpec {
  TRITONSERVER_MemoryType input_type_;
  int64_t input_type_id_;

  TRITONSERVER_MemoryType output_type_;
  int64_t output_type_id_;
};

// Meta data used for preparing input data and validate output data
IOSpec io_spec;

static auto gpu_data_deleter = [](void* data) {
  if (data != nullptr) {
    FAIL_IF_CUDA_ERR(
        cudaSetDevice(io_spec.input_type_id_),
        "setting CUDA device to release GPU memory on " +
            std::to_string(io_spec.input_type_id_));
    FAIL_IF_CUDA_ERR(cudaFree(data), "releasing GPU memory");
  }
};

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-i [input device ID]" << std::endl;
  std::cerr << "\t-out [output device ID]" << std::endl;
  std::cerr << "\t-v Enable verbose logging" << std::endl;
  std::cerr << "\t-r [model repository absolute path]" << std::endl;
  std::cerr << "\t-m [model name to be tested]" << std::endl;
  std::cerr << "\t-h [host policy name]" << std::endl;
  std::cerr << "\tFor '-h', if specify, the input will be set with different "
            << "host policy names, given that the specified value is the "
            << "host policy that the model under test is associated with."
            << std::endl;
  std::cerr << "\tFor device ID, -1 is used to stand for CPU device, "
            << "non-negative value is for GPU device." << std::endl;

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
  // If 'byte_size' is zero just return 'buffer'==nullptr, we don't
  // need to do any other book-keeping.
  if (byte_size == 0) {
    *buffer = nullptr;
    *buffer_userp = nullptr;
    std::cout << "allocated " << byte_size << " bytes for result tensor "
              << tensor_name << std::endl;
  } else {
    void* allocated_ptr = nullptr;
    if (io_spec.output_type_ == TRITONSERVER_MEMORY_CPU) {
      allocated_ptr = malloc(byte_size);
    } else {
      auto err = cudaSetDevice(io_spec.output_type_id_);
      if (err == cudaSuccess) {
        err = cudaMalloc(&allocated_ptr, byte_size);
      }
      if (err != cudaSuccess) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "failed to allocate CUDA memory: " +
                std::string(cudaGetErrorString(err)))
                .c_str());
      }
    }

    if (allocated_ptr == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          std::string(
              "failed to allocate " + std::to_string(byte_size) + " bytes in " +
              TRITONSERVER_MemoryTypeString(io_spec.output_type_) +
              " for result tensor " + tensor_name)
              .c_str());
    }

    // Pass the tensor name with buffer_userp so we can show it when
    // releasing the buffer.
    *buffer = allocated_ptr;
    *buffer_userp = new std::string(tensor_name);
    std::cout << "allocated " << byte_size << " bytes in "
              << TRITONSERVER_MemoryTypeString(io_spec.output_type_)
              << " for result tensor " << tensor_name << std::endl;
  }

  *actual_memory_type = io_spec.output_type_;
  *actual_memory_type_id = io_spec.output_type_id_;
  return nullptr;  // Success
}

TRITONSERVER_Error*
ResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  std::unique_ptr<std::string> name;
  if (buffer_userp != nullptr) {
    name.reset(reinterpret_cast<std::string*>(buffer_userp));
  } else {
    name.reset(new std::string("<unknown>"));
  }

  std::cout << "Releasing buffer " << buffer << " of size " << byte_size
            << " in " << TRITONSERVER_MemoryTypeString(memory_type)
            << " for result '" << *name << "'" << std::endl;
  if (memory_type == TRITONSERVER_MEMORY_CPU) {
    free(buffer);
  } else {
    auto err = cudaSetDevice(memory_type_id);
    if (err == cudaSuccess) {
      err = cudaFree(buffer);
    }
    if (err != cudaSuccess) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, std::string(
                                           "failed to release CUDA memory: " +
                                           std::string(cudaGetErrorString(err)))
                                           .c_str());
    }
  }

  return nullptr;  // Success
}

void
InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  if ((flags & TRITONSERVER_REQUEST_RELEASE_ALL) != 0) {
    TRITONSERVER_InferenceRequestDelete(request);
  }
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

uint32_t
OutputIndex(TRITONSERVER_InferenceResponse* response, const std::string& name)
{
  uint32_t output_count;
  FAIL_IF_ERR(
      TRITONSERVER_InferenceResponseOutputCount(response, &output_count),
      "getting number of response outputs");

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

    if (name == std::string(cname)) {
      return idx;
    }
  }

  FAIL("can't found output '" + name + "'");
  return 0;
}

TRITONSERVER_Error*
ParseModelConfig(
    const rapidjson::Document& model_metadata, TRITONSERVER_DataType* dtype,
    bool* is_torch_model)
{
  *dtype = TRITONSERVER_TYPE_INVALID;
  for (const auto& input : model_metadata["inputs"].GetArray()) {
    if (strcmp(input["datatype"].GetString(), "INT32") &&
        strcmp(input["datatype"].GetString(), "FP32") &&
        strcmp(input["datatype"].GetString(), "BYTES")) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "IO test utility only supports model with data type INT32, "
          "FP32 or BYTES");
    }

    if (*dtype == TRITONSERVER_TYPE_INVALID) {
      *dtype = TRITONSERVER_StringToDataType(input["datatype"].GetString());
    } else {
      auto dt = TRITONSERVER_StringToDataType(input["datatype"].GetString());
      if (dt != *dtype) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "the model inputs must have the same data type");
      }
    }
  }

  for (const auto& output : model_metadata["outputs"].GetArray()) {
    if (strcmp(output["datatype"].GetString(), "INT32") &&
        strcmp(output["datatype"].GetString(), "FP32") &&
        strcmp(output["datatype"].GetString(), "BYTES")) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "IO test utility only supports model with data type INT32, "
          "FP32 or BYTES");
    } else {
      auto dt = TRITONSERVER_StringToDataType(output["datatype"].GetString());
      if (dt != *dtype) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "the model inputs and outputs must have the same data type");
      }
    }
  }

  *is_torch_model = (model_metadata["platform"] == "pytorch_libtorch");
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

void
GenerateStringInputData(
    std::vector<char>* input0_data, std::vector<char>* input1_data)
{
  std::string input0_str = "";
  std::string input1_str = "";
  for (size_t i = 0; i < 16; ++i) {
    std::string i0 = std::to_string(i + 1);
    uint32_t i0_len = i0.size();
    input0_str.append(reinterpret_cast<const char*>(&i0_len), sizeof(uint32_t));
    input0_str.append(i0);
    std::string i1 = std::to_string(1);
    uint32_t i1_len = i1.size();
    input1_str.append(reinterpret_cast<const char*>(&i1_len), sizeof(uint32_t));
    input1_str.append(i1);
  }

  std::copy(
      input0_str.begin(), input0_str.end(), std::back_inserter(*input0_data));
  std::copy(
      input1_str.begin(), input1_str.end(), std::back_inserter(*input1_data));
}

void
GenerateStringOutputData(
    std::vector<char>* output0_data, std::vector<char>* output1_data)
{
  std::string output0_str = "";
  std::string output1_str = "";
  for (size_t i = 0; i < 16; ++i) {
    std::string o0 = std::to_string(i + 2);
    uint32_t o0_len = o0.size();
    output0_str.append(
        reinterpret_cast<const char*>(&o0_len), sizeof(uint32_t));
    output0_str.append(o0);
    std::string o1 = std::to_string(i);
    uint32_t o1_len = o1.size();
    output1_str.append(
        reinterpret_cast<const char*>(&o1_len), sizeof(uint32_t));
    output1_str.append(o1);
  }

  std::copy(
      output0_str.begin(), output0_str.end(),
      std::back_inserter(*output0_data));
  std::copy(
      output1_str.begin(), output1_str.end(),
      std::back_inserter(*output1_data));
}

template <typename T>
void
CompareResult(
    const std::string& output0_name, const std::string& output1_name,
    const void* input0, const void* input1, const void* output0,
    const void* output1)
{
  for (size_t i = 0; i < 16; ++i) {
    std::cout << ((T*)input0)[i] << " + " << ((T*)input1)[i] << " = "
              << ((T*)output0)[i] << std::endl;
    std::cout << ((T*)input0)[i] << " - " << ((T*)input1)[i] << " = "
              << ((T*)output1)[i] << std::endl;

    if ((((T*)input0)[i] + ((T*)input1)[i]) != ((T*)output0)[i]) {
      FAIL("incorrect sum in " + output0_name);
    }
    if ((((T*)input0)[i] - ((T*)input1)[i]) != ((T*)output1)[i]) {
      FAIL("incorrect difference in " + output1_name);
    }
  }
}

void
CompareStringResult(
    const std::string& output0_name, const std::string& output1_name,
    const void* input0, const void* input1, const void* output0,
    const void* output1)
{
  // preprocess results from serialized buffer to integers
  std::vector<int> output0_numbers;
  std::vector<int> output1_numbers;
  size_t buf_offset0 = 0, buf_offset1 = 0;
  const uint8_t* base0 = reinterpret_cast<const uint8_t*>(output0);
  const uint8_t* base1 = reinterpret_cast<const uint8_t*>(output1);
  for (size_t i = 0; i < 16; ++i) {
    const uint32_t len0 =
        *(reinterpret_cast<const uint32_t*>(base0 + buf_offset0));
    std::string o0_tmp(
        reinterpret_cast<const char*>(base0 + buf_offset0 + sizeof(len0)),
        len0);
    output0_numbers.push_back(std::atoi(o0_tmp.c_str()));
    buf_offset0 += sizeof(len0) + len0;

    const uint32_t len1 =
        *(reinterpret_cast<const uint32_t*>(base1 + buf_offset1));
    std::string o1_tmp(
        reinterpret_cast<const char*>(base1 + buf_offset1 + sizeof(len1)),
        len1);
    output1_numbers.push_back(std::atoi(o1_tmp.c_str()));
    buf_offset1 += sizeof(len1) + len1;
  }

  for (int i = 0; i < 16; ++i) {
    std::cout << (i + 1) << " + " << 1 << " = " << output0_numbers[i]
              << std::endl;
    std::cout << (i + 1) << " - " << 1 << " = " << output1_numbers[i]
              << std::endl;

    if (((i + 1) + 1) != output0_numbers[i]) {
      FAIL("incorrect sum in " + output0_name);
    }
    if (((i + 1) - 1) != output1_numbers[i]) {
      FAIL("incorrect difference in " + output1_name);
    }
  }
}

}  // namespace

int
main(int argc, char** argv)
{
  std::string model_repository_path;
  std::string model_name;
  int verbose_level = 0;

  io_spec.input_type_ = TRITONSERVER_MEMORY_CPU;
  io_spec.input_type_id_ = 0;
  io_spec.output_type_ = TRITONSERVER_MEMORY_CPU;
  io_spec.output_type_id_ = 0;

  const char* host_policy_cstr = nullptr;
  std::string host_policy;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vi:o:r:m:h:")) != -1) {
    switch (opt) {
      case 'i': {
        int64_t raw_id = std::stoll(optarg);
        if (raw_id < 0) {
          io_spec.input_type_ = TRITONSERVER_MEMORY_CPU;
          io_spec.input_type_id_ = 0;
        } else {
          io_spec.input_type_ = TRITONSERVER_MEMORY_GPU;
          io_spec.input_type_id_ = raw_id;
        }
        break;
      }
      case 'o': {
        int64_t raw_id = std::stoll(optarg);
        if (raw_id < 0) {
          io_spec.output_type_ = TRITONSERVER_MEMORY_CPU;
          io_spec.output_type_id_ = 0;
        } else {
          io_spec.output_type_ = TRITONSERVER_MEMORY_GPU;
          io_spec.output_type_id_ = raw_id;
        }
        break;
      }
      case 'h': {
        host_policy = optarg;
        host_policy_cstr = host_policy.c_str();
        break;
      }
      case 'r':
        model_repository_path = optarg;
        break;
      case 'm':
        model_name = optarg;
        break;
      case 'v':
        verbose_level = 1;
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  if (model_repository_path.empty()) {
    Usage(argv, "-r must be used to specify model repository path");
  }
  if (model_name.empty()) {
    Usage(argv, "-m must be used to specify model being test");
  }

  // Create the server...
  TRITONSERVER_ServerOptions* server_options = nullptr;
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsNew(&server_options),
      "creating server options");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetModelRepositoryPath(
          server_options, model_repository_path.c_str()),
      "setting model repository path");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetModelControlMode(
          server_options, TRITONSERVER_MODEL_CONTROL_EXPLICIT),
      "setting model control mode");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetStartupModel(
          server_options, model_name.c_str()),
      "setting model to load");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetLogVerbose(server_options, verbose_level),
      "setting verbose logging level");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetBackendDirectory(
          server_options, "/opt/tritonserver/backends"),
      "setting backend directory");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
          server_options, "/opt/tritonserver/repoagents"),
      "setting repository agent directory");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetStrictModelConfig(server_options, true),
      "setting strict model configuration");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
          server_options, TRITON_MIN_COMPUTE_CAPABILITY),
      "setting minimum supported CUDA compute capability");

  TRITONSERVER_Server* server_ptr = nullptr;
  FAIL_IF_ERR(
      TRITONSERVER_ServerNew(&server_ptr, server_options), "creating server");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsDelete(server_options),
      "deleting server options");

  std::shared_ptr<TRITONSERVER_Server> server(
      server_ptr, TRITONSERVER_ServerDelete);

  // Wait until the server is both live and ready.
  size_t health_iters = 0;
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

    if (++health_iters >= 10) {
      FAIL("failed to find healthy inference server");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // Print status of the server.
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

  // Wait for the model to become available.
  bool is_torch_model = false;
  TRITONSERVER_DataType dtype = TRITONSERVER_TYPE_INT32;
  bool is_ready = false;
  health_iters = 0;
  while (!is_ready) {
    FAIL_IF_ERR(
        TRITONSERVER_ServerModelIsReady(
            server.get(), model_name.c_str(), 1, &is_ready),
        "unable to get model readiness");
    if (!is_ready) {
      if (++health_iters >= 10) {
        FAIL("model failed to be ready in 10 iterations");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
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

    FAIL_IF_ERR(
        ParseModelConfig(model_metadata, &dtype, &is_torch_model),
        "parsing model metadata");
  }

  // Create the allocator that will be used to allocate buffers for
  // the result tensors.
  TRITONSERVER_ResponseAllocator* allocator = nullptr;
  FAIL_IF_ERR(
      TRITONSERVER_ResponseAllocatorNew(
          &allocator, ResponseAlloc, ResponseRelease, nullptr /* start_fn */),
      "creating response allocator");

  TRITONSERVER_InferenceRequest* irequest = nullptr;
  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestNew(
          &irequest, server.get(), model_name.c_str(), -1 /* model_version */),
      "creating inference request");

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestSetId(irequest, "123"),
      "setting ID for the request");

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestSetReleaseCallback(
          irequest, InferRequestComplete, nullptr /* request_release_userp */),
      "setting request release callback");

  // Create 0 data that shouldn't be selected and used to test host policy
  // functionality
  std::vector<uint32_t> zero_data(16);

  // Create the data for the two input tensors. Initialize the first
  // to unique integers and the second to all ones.
  std::vector<char> input0_data;
  std::vector<char> input1_data;
  if (dtype == TRITONSERVER_TYPE_INT32) {
    GenerateInputData<int32_t>(&input0_data, &input1_data);
  } else if (dtype == TRITONSERVER_TYPE_FP32) {
    GenerateInputData<float>(&input0_data, &input1_data);
  } else {
    GenerateStringInputData(&input0_data, &input1_data);
  }

  auto input0 = "INPUT0";
  auto input1 = "INPUT1";

  // Get the size of the input tensors
  size_t input0_size = input0_data.size();
  size_t input1_size = input1_data.size();

  std::vector<int64_t> input0_shape({1, 16});
  std::vector<int64_t> input1_shape({1, 16});

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAddInput(
          irequest, input0, dtype, &input0_shape[0], input0_shape.size()),
      "setting input 0 meta-data for the request");
  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAddInput(
          irequest, input1, dtype, &input1_shape[0], input1_shape.size()),
      "setting input 1 meta-data for the request");

  auto output0 = is_torch_model ? "OUTPUT__0" : "OUTPUT0";
  auto output1 = is_torch_model ? "OUTPUT__1" : "OUTPUT1";

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output0),
      "requesting output 0 for the request");
  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output1),
      "requesting output 1 for the request");

  const void* input0_base = &input0_data[0];
  const void* input1_base = &input1_data[0];
  bool gpu_input = (io_spec.input_type_ == TRITONSERVER_MEMORY_GPU);
  std::unique_ptr<void, decltype(gpu_data_deleter)> input0_gpu(
      nullptr, gpu_data_deleter);
  std::unique_ptr<void, decltype(gpu_data_deleter)> input1_gpu(
      nullptr, gpu_data_deleter);
  if (gpu_input) {
    FAIL_IF_CUDA_ERR(
        cudaSetDevice(io_spec.input_type_id_),
        "setting CUDA device to device " +
            std::to_string(io_spec.input_type_id_));
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

  input0_base = gpu_input ? input0_gpu.get() : &input0_data[0];
  input1_base = gpu_input ? input1_gpu.get() : &input1_data[0];


  if (host_policy_cstr == nullptr) {
    FAIL_IF_ERR(
        TRITONSERVER_InferenceRequestAppendInputData(
            irequest, input0, input0_base, input0_size, io_spec.input_type_,
            io_spec.input_type_id_),
        "assigning INPUT0 data");
    FAIL_IF_ERR(
        TRITONSERVER_InferenceRequestAppendInputData(
            irequest, input1, input1_base, input1_size, io_spec.input_type_,
            io_spec.input_type_id_),
        "assigning INPUT1 data");

    FAIL_IF_ERR(
        TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
            irequest, input0, zero_data.data(),
            zero_data.size() * sizeof(uint32_t), TRITONSERVER_MEMORY_CPU, 0,
            "fake_host_policy_name"),
        "assigning zero INPUT0 data with host policy 'fake_host_policy_name'");
    FAIL_IF_ERR(
        TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
            irequest, input1, zero_data.data(),
            zero_data.size() * sizeof(uint32_t), TRITONSERVER_MEMORY_CPU, 0,
            "fake_host_policy_name"),
        "assigning zero INPUT1 data with host policy 'fake_host_policy_name'");
  } else {
    FAIL_IF_ERR(
        TRITONSERVER_InferenceRequestAppendInputData(
            irequest, input0, zero_data.data(),
            zero_data.size() * sizeof(uint32_t), TRITONSERVER_MEMORY_CPU, 0),
        "assigning zero INPUT0 data");
    FAIL_IF_ERR(
        TRITONSERVER_InferenceRequestAppendInputData(
            irequest, input1, zero_data.data(),
            zero_data.size() * sizeof(uint32_t), TRITONSERVER_MEMORY_CPU, 0),
        "assigning zero INPUT1 data");
    FAIL_IF_ERR(
        TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
            irequest, input0, input0_base, input0_size, io_spec.input_type_,
            io_spec.input_type_id_, host_policy_cstr),
        "assigning INPUT0 data to provided host policy");
    FAIL_IF_ERR(
        TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
            irequest, input1, input1_base, input1_size, io_spec.input_type_,
            io_spec.input_type_id_, host_policy_cstr),
        "assigning INPUT1 data to provided host policy");
  }

  // Perform inference...
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

  // Wait for the inference response and check the status.
  TRITONSERVER_InferenceResponse* response = completed.get();
  FAIL_IF_ERR(TRITONSERVER_InferenceResponseError(response), "response status");

  // Create the expected data for the two output tensors.
  std::vector<char> expected0_data;
  std::vector<char> expected1_data;
  if (dtype == TRITONSERVER_TYPE_BYTES) {
    GenerateStringOutputData(&expected0_data, &expected1_data);
  }

  // Check the output tensor values...
  // Note that depending on whether the backend supports outputs in GPU memory,
  // the output tensor may be in CPU memory even if -g flag is set.

  const void* output0_content;
  size_t output0_byte_size;
  TRITONSERVER_MemoryType output0_memory_type;
  int64_t output0_memory_type_id;
  {
    const char* cname;
    TRITONSERVER_DataType datatype;
    const int64_t* shape;
    uint64_t dim_count;
    void* userp;

    FAIL_IF_ERR(
        TRITONSERVER_InferenceResponseOutput(
            response, OutputIndex(response, output0), &cname, &datatype, &shape,
            &dim_count, &output0_content, &output0_byte_size,
            &output0_memory_type, &output0_memory_type_id, &userp),
        "getting output0 info");

    if (dtype == TRITONSERVER_TYPE_BYTES) {
      size_t expected0_size = expected0_data.size();
      if (expected0_size != output0_byte_size) {
        FAIL(
            "unexpected output0 byte-size, expected " +
            std::to_string(expected0_size) + ", got " +
            std::to_string(output0_byte_size));
      }
    } else if (output0_byte_size != input0_size) {
      FAIL(
          "unexpected output0 byte-size, expected " +
          std::to_string(input0_size) + ", got " +
          std::to_string(output0_byte_size));
    } else if (
        (io_spec.output_type_ != output0_memory_type) ||
        (io_spec.output_type_id_ != output0_memory_type_id)) {
      FAIL(
          std::string("unexpected output0 memory type (id), expected to be "
                      "allocated in ") +
          TRITONSERVER_MemoryTypeString(io_spec.output_type_) + " with id " +
          std::to_string(io_spec.output_type_id_) + ", got " +
          TRITONSERVER_MemoryTypeString(output0_memory_type) + " with id " +
          std::to_string(output0_memory_type_id));
    }
  }

  const void* output1_content;
  size_t output1_byte_size;
  TRITONSERVER_MemoryType output1_memory_type;
  int64_t output1_memory_type_id;
  {
    const char* cname;
    TRITONSERVER_DataType datatype;
    const int64_t* shape;
    uint64_t dim_count;
    void* userp;

    FAIL_IF_ERR(
        TRITONSERVER_InferenceResponseOutput(
            response, OutputIndex(response, output1), &cname, &datatype, &shape,
            &dim_count, &output1_content, &output1_byte_size,
            &output1_memory_type, &output1_memory_type_id, &userp),
        "getting output1 info");

    if (dtype == TRITONSERVER_TYPE_BYTES) {
      size_t expected1_size = expected1_data.size();
      if (expected1_size != output1_byte_size) {
        FAIL(
            "unexpected output1 byte-size, expected " +
            std::to_string(expected1_size) + ", got " +
            std::to_string(output1_byte_size));
      }
    } else if (output1_byte_size != input1_size) {
      FAIL(
          "unexpected output1 byte-size, expected " +
          std::to_string(input1_size) + ", got " +
          std::to_string(output1_byte_size));
    } else if (
        (io_spec.output_type_ != output1_memory_type) ||
        (io_spec.output_type_id_ != output1_memory_type_id)) {
      FAIL(
          std::string("unexpected output1 memory type (id), expected to be "
                      "allocated in ") +
          TRITONSERVER_MemoryTypeString(io_spec.output_type_) + " with id " +
          std::to_string(io_spec.output_type_id_) + ", got " +
          TRITONSERVER_MemoryTypeString(output1_memory_type) + " with id " +
          std::to_string(output1_memory_type_id));
    }
  }

  const void* output0_result = output0_content;
  const void* output1_result = output1_content;

  // Different from CPU memory, outputs in GPU memory must be copied to CPU
  // memory to be read directly.
  std::vector<char> output0_data(output0_byte_size);
  std::vector<char> output1_data(output1_byte_size);
  if (output0_memory_type == TRITONSERVER_MEMORY_CPU) {
    std::cout << "OUTPUT0 are stored in CPU memory" << std::endl;
  } else {
    std::cout << "OUTPUT0 are stored in GPU memory" << std::endl;
    FAIL_IF_CUDA_ERR(
        cudaMemcpy(
            &output0_data[0], output0_content, output0_byte_size,
            cudaMemcpyDeviceToHost),
        "setting INPUT0 data in GPU memory");
    output0_result = &output0_data[0];
  }

  if (output1_memory_type == TRITONSERVER_MEMORY_CPU) {
    std::cout << "OUTPUT1 are stored in CPU memory" << std::endl;
  } else {
    std::cout << "OUTPUT1 are stored in GPU memory" << std::endl;
    FAIL_IF_CUDA_ERR(
        cudaMemcpy(
            &output1_data[0], output1_content, output1_byte_size,
            cudaMemcpyDeviceToHost),
        "setting INPUT0 data in GPU memory");
    output1_result = &output1_data[0];
  }

  if (dtype == TRITONSERVER_TYPE_INT32) {
    CompareResult<int32_t>(
        output0, output1, &input0_data[0], &input1_data[0], output0_result,
        output1_result);
  } else if (dtype == TRITONSERVER_TYPE_FP32) {
    CompareResult<float>(
        output0, output1, &input0_data[0], &input1_data[0], output0_result,
        output1_result);
  } else {
    CompareStringResult(
        output0, output1, &input0_data[0], &input1_data[0], output0_result,
        output1_result);
  }

  FAIL_IF_ERR(
      TRITONSERVER_InferenceResponseDelete(response),
      "deleting inference response");

  FAIL_IF_ERR(
      TRITONSERVER_ResponseAllocatorDelete(allocator),
      "deleting response allocator");

  return 0;
}
