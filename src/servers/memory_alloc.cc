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

#include <cuda_runtime_api.h>
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

namespace ni = nvidia::inferenceserver;

namespace {

struct IOSpec {
  TRTSERVER_Memory_Type input_type_;
  int64_t input_type_id_;

  TRTSERVER_Memory_Type output_type_;
  int64_t output_type_id_;
};

// Meta data used for preparing input data and validate output data
IOSpec io_spec;

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
    LOG_ERROR << msg;
  }

  LOG_ERROR << "Usage: " << argv[0] << " [options]";
  LOG_ERROR << "\t-i [input device ID]";
  LOG_ERROR << "\t-out [output device ID]";
  LOG_ERROR << "\t-v Enable verbose logging";
  LOG_ERROR << "\t-r [model repository absolute path]";
  LOG_ERROR << "\t-m [model name to be tested]";
  LOG_ERROR << "\tFor device ID, -1 is used to stand for CPU device, "
            << "non-negative value is for GPU device.";

  exit(1);
}

std::string
MemoryTypeString(TRTSERVER_Memory_Type memory_type)
{
  return (memory_type == TRTSERVER_MEMORY_CPU) ? "CPU memory" : "GPU memory";
}

TRTSERVER_Error*
ResponseAlloc(
    TRTSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRTSERVER_Memory_Type preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRTSERVER_Memory_Type* actual_memory_type,
    int64_t* actual_memory_type_id)
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
    if (io_spec.output_type_ == TRTSERVER_MEMORY_CPU) {
      allocated_ptr = malloc(byte_size);
    } else {
      auto err = cudaSetDevice(io_spec.output_type_id_);
      if (err == cudaSuccess) {
        err = cudaMalloc(&allocated_ptr, byte_size);
      }
      if (err != cudaSuccess) {
        return TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INTERNAL, std::string(
                                          "failed to allocate CUDA memory: " +
                                          std::string(cudaGetErrorString(err)))
                                          .c_str());
      }
    }

    if (allocated_ptr == nullptr) {
      return TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_INTERNAL,
          std::string(
              "failed to allocate " + std::to_string(byte_size) + " bytes in " +
              MemoryTypeString(io_spec.output_type_) + " for result tensor " +
              tensor_name)
              .c_str());
    }

    *buffer = allocated_ptr;
    *buffer_userp = new std::string(tensor_name);
    LOG_INFO << "allocated " << byte_size << " bytes in "
             << MemoryTypeString(io_spec.output_type_) << " for result tensor "
             << tensor_name;
  }

  *actual_memory_type = io_spec.output_type_;
  *actual_memory_type_id = io_spec.output_type_id_;
  return nullptr;  // Success
}

TRTSERVER_Error*
ResponseRelease(
    TRTSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRTSERVER_Memory_Type memory_type, int64_t memory_type_id)
{
  std::unique_ptr<std::string> name;
  if (buffer_userp != nullptr) {
    name.reset(reinterpret_cast<std::string*>(buffer_userp));
  } else {
    name.reset(new std::string("<unknown>"));
  }

  LOG_INFO << "Releasing buffer " << buffer << " of size " << byte_size
           << " in " << MemoryTypeString(memory_type) << " for result '"
           << *name << "'";
  if (memory_type == TRTSERVER_MEMORY_CPU) {
    free(buffer);
  } else {
    auto err = cudaSetDevice(memory_type_id);
    if (err == cudaSuccess) {
      err = cudaFree(buffer);
    }
    if (err != cudaSuccess) {
      return TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_INTERNAL, std::string(
                                        "failed to release CUDA memory: " +
                                        std::string(cudaGetErrorString(err)))
                                        .c_str());
    }
  }

  return nullptr;  // Success
}

void
InferComplete(
    TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
    TRTSERVER_InferenceResponse* response, void* userp)
{
  std::promise<TRTSERVER_InferenceResponse*>* p =
      reinterpret_cast<std::promise<TRTSERVER_InferenceResponse*>*>(userp);
  p->set_value(response);
  delete p;

  TRTSERVER_TraceManagerDelete(trace_manager);
}

TRTSERVER_Error*
ParseModelConfig(
    const ni::ModelConfig& config, ni::DataType* dtype, bool* is_torch_model)
{
  auto data_type = ni::TYPE_INVALID;
  for (const auto& input : config.input()) {
    if ((input.data_type() != ni::TYPE_INT32) &&
        (input.data_type() != ni::TYPE_FP32) &&
        (input.data_type() != ni::TYPE_STRING)) {
      return TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_UNSUPPORTED,
          "IO test utility only supports model with data type INT32, "
          "FP32 or STRING");
    }
    if (data_type == ni::TYPE_INVALID) {
      data_type = input.data_type();
    } else if (input.data_type() != data_type) {
      auto err_str = "the inputs of '" + config.name() +
                     "' model must have the same data type";
      return TRTSERVER_ErrorNew(TRTSERVER_ERROR_INVALID_ARG, err_str.c_str());
    }
  }
  for (const auto& output : config.output()) {
    if ((output.data_type() != ni::TYPE_INT32) &&
        (output.data_type() != ni::TYPE_FP32) &&
        (output.data_type() != ni::TYPE_STRING)) {
      return TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_UNSUPPORTED,
          "IO test utility only supports model with data type INT32, "
          "FP32 or STRING");
    } else if (output.data_type() != data_type) {
      auto err_str = "the inputs and outputs of '" + config.name() +
                     "' model must have the same data type";
      return TRTSERVER_ErrorNew(TRTSERVER_ERROR_INVALID_ARG, err_str.c_str());
    }
  }

  *dtype = data_type;
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
    LOG_INFO << (i + 1) << " + " << 1 << " = " << output0_numbers[i];
    LOG_INFO << (i + 1) << " - " << 1 << " = " << output1_numbers[i];

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

  io_spec.input_type_ = TRTSERVER_MEMORY_CPU;
  io_spec.input_type_id_ = 0;
  io_spec.output_type_ = TRTSERVER_MEMORY_CPU;
  io_spec.output_type_id_ = 0;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vi:o:r:m:")) != -1) {
    switch (opt) {
      case 'i': {
        int64_t raw_id = std::stoll(optarg);
        if (raw_id < 0) {
          io_spec.input_type_ = TRTSERVER_MEMORY_CPU;
          io_spec.input_type_id_ = 0;
        } else {
          io_spec.input_type_ = TRTSERVER_MEMORY_GPU;
          io_spec.input_type_id_ = raw_id;
        }
        break;
      }
      case 'o': {
        int64_t raw_id = std::stoll(optarg);
        if (raw_id < 0) {
          io_spec.output_type_ = TRTSERVER_MEMORY_CPU;
          io_spec.output_type_id_ = 0;
        } else {
          io_spec.output_type_ = TRTSERVER_MEMORY_GPU;
          io_spec.output_type_id_ = raw_id;
        }
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
  TRTSERVER_ServerOptions* server_options = nullptr;
  FAIL_IF_ERR(
      TRTSERVER_ServerOptionsNew(&server_options), "creating server options");
  FAIL_IF_ERR(
      TRTSERVER_ServerOptionsSetModelRepositoryPath(
          server_options, model_repository_path.c_str()),
      "setting model repository path");
  FAIL_IF_ERR(
      TRTSERVER_ServerOptionsSetModelControlMode(
          server_options, TRTSERVER_MODEL_CONTROL_EXPLICIT),
      "setting model control mode");
  FAIL_IF_ERR(
      TRTSERVER_ServerOptionsSetStartupModel(
          server_options, model_name.c_str()),
      "setting model to load");
  FAIL_IF_ERR(
      TRTSERVER_ServerOptionsSetLogVerbose(server_options, verbose_level),
      "setting verbose logging level");

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

  // Wait for the model to become available.
  bool is_torch_model = false;
  ni::DataType dtype = ni::TYPE_INT32;
  while (true) {
    TRTSERVER_Protobuf* model_status_protobuf;
    FAIL_IF_ERR(
        TRTSERVER_ServerModelStatus(
            server.get(), model_name.c_str(), &model_status_protobuf),
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

    FAIL_IF_ERR(
        TRTSERVER_ProtobufDelete(model_status_protobuf),
        "deleting status protobuf");

    auto itr = model_status.model_status().find(model_name);
    if (itr == model_status.model_status().end()) {
      FAIL("unable to find status for model '" + model_name + "'");
    }

    auto vitr = itr->second.version_status().find(1);
    if (vitr == itr->second.version_status().end()) {
      FAIL("unable to find version 1 status for model '" + model_name + "'");
    }

    LOG_INFO << "'" + model_name + "' model is "
             << ni::ModelReadyState_Name(vitr->second.ready_state());
    if (vitr->second.ready_state() == ni::ModelReadyState::MODEL_READY) {
      FAIL_IF_ERR(
          ParseModelConfig(itr->second.config(), &dtype, &is_torch_model),
          "parsing model config");
      break;
    }

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

  // Create the data for the two input tensors. Initialize the first
  // to unique integers and the second to all ones.
  std::vector<char> input0_data;
  std::vector<char> input1_data;
  if (dtype == ni::TYPE_INT32) {
    GenerateInputData<int32_t>(&input0_data, &input1_data);
  } else if (dtype == ni::TYPE_FP32) {
    GenerateInputData<float>(&input0_data, &input1_data);
  } else {
    GenerateStringInputData(&input0_data, &input1_data);
  }

  // Get the size of the input tensors
  size_t input0_size = input0_data.size();
  size_t input1_size = input1_data.size();

  // For string we need to set batch byte size explicitly
  if (dtype == ni::TYPE_STRING) {
    input0->set_batch_byte_size(input0_size);
    input1->set_batch_byte_size(input1_size);
  }

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

  const void* input0_base = &input0_data[0];
  const void* input1_base = &input1_data[0];
  bool gpu_input = (io_spec.input_type_ == TRTSERVER_MEMORY_GPU);
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

  FAIL_IF_ERR(
      TRTSERVER_InferenceRequestProviderSetInputData(
          request_provider, input0->name().c_str(), input0_base, input0_size,
          io_spec.input_type_, io_spec.input_type_id_),
      "assigning INPUT0 data");
  FAIL_IF_ERR(
      TRTSERVER_InferenceRequestProviderSetInputData(
          request_provider, input1->name().c_str(), input1_base, input1_size,
          io_spec.input_type_, io_spec.input_type_id_),
      "assigning INPUT1 data");

  // Perform inference...
  auto p = new std::promise<TRTSERVER_InferenceResponse*>();
  std::future<TRTSERVER_InferenceResponse*> completed = p->get_future();

  FAIL_IF_ERR(
      TRTSERVER_ServerInferAsync(
          server.get(), nullptr /* trace_manager */, request_provider,
          allocator, nullptr /* response_allocator_userp */, InferComplete,
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

    LOG_INFO << "Model \"" << model_name << "\" response header:";
    LOG_INFO << response_header.DebugString();

    FAIL_IF_ERR(
        TRTSERVER_ProtobufDelete(response_protobuf),
        "deleting response protobuf");
  }

  // Create the expected data for the two output tensors.
  std::vector<char> expected0_data;
  std::vector<char> expected1_data;
  if (dtype == ni::TYPE_STRING) {
    GenerateStringOutputData(&expected0_data, &expected1_data);
  }

  // Check the output tensor values...
  // Note that depending on whether the backend supports outputs in GPU memory,
  // the output tensor may be in CPU memory even if -g flag is set.
  const void* output0_content;
  size_t output0_byte_size;
  TRTSERVER_Memory_Type output0_memory_type;
  int64_t output0_memory_type_id;
  FAIL_IF_ERR(
      TRTSERVER_InferenceResponseOutputData(
          response, output0->name().c_str(), &output0_content,
          &output0_byte_size, &output0_memory_type, &output0_memory_type_id),
      "getting output0 result");
  if (dtype == ni::TYPE_STRING) {
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
        "unexpected output0 memory type (id), expected to be allocated "
        "in " +
        MemoryTypeString(io_spec.output_type_) + " with id " +
        std::to_string(io_spec.output_type_id_) + ", got " +
        MemoryTypeString(output0_memory_type) + " with id " +
        std::to_string(output0_memory_type_id));
  }

  const void* output1_content;
  size_t output1_byte_size;
  TRTSERVER_Memory_Type output1_memory_type;
  int64_t output1_memory_type_id;
  FAIL_IF_ERR(
      TRTSERVER_InferenceResponseOutputData(
          response, output1->name().c_str(), &output1_content,
          &output1_byte_size, &output1_memory_type, &output1_memory_type_id),
      "getting output1 result");
  if (dtype == ni::TYPE_STRING) {
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
        "unexpected output1 memory type (id), expected to be allocated "
        "in " +
        MemoryTypeString(io_spec.output_type_) + " with id " +
        std::to_string(io_spec.output_type_id_) + ", got " +
        MemoryTypeString(output1_memory_type) + " with id " +
        std::to_string(output1_memory_type_id));
  }

  const void* output0_result = output0_content;
  const void* output1_result = output1_content;

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

  if (dtype == ni::TYPE_INT32) {
    CompareResult<int32_t>(
        output0->name(), output1->name(), &input0_data[0], &input1_data[0],
        output0_result, output1_result);
  } else if (dtype == ni::TYPE_FP32) {
    CompareResult<float>(
        output0->name(), output1->name(), &input0_data[0], &input1_data[0],
        output0_result, output1_result);
  } else {
    CompareStringResult(
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
