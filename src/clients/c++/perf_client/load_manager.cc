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

#include "src/clients/c++/perf_client/load_manager.h"
#include "src/clients/c++/examples/shm_utils.h"
#include "src/core/model_config.h"

#ifdef TRTIS_ENABLE_GPU
#include <cuda_runtime_api.h>

#define RETURN_IF_CUDA_ERR(FUNC)                               \
  {                                                            \
    const cudaError_t result = FUNC;                           \
    if (result != cudaSuccess) {                               \
      return nic::Error(                                       \
          ni::RequestStatusCode::INTERNAL,                     \
          "CUDA exception (line " + std::to_string(__LINE__) + \
              "): " + cudaGetErrorName(result) + " (" +        \
              cudaGetErrorString(result) + ")");               \
    }                                                          \
  }

#endif  // TRTIS_ENABLE_GPU

namespace {

#ifdef TRTIS_ENABLE_GPU
nic::Error
CreateCUDAIPCHandle(
    cudaIpcMemHandle_t* cuda_handle, void* input_d_ptr, int device_id = 0)
{
  // Set the GPU device to the desired GPU
  RETURN_IF_CUDA_ERR(cudaSetDevice(device_id));

  //  Create IPC handle for data on the gpu
  RETURN_IF_CUDA_ERR(cudaIpcGetMemHandle(cuda_handle, input_d_ptr));

  return nic::Error::Success;
}
#endif  // TRTIS_ENABLE_GPU

void
SerializeStringTensor(
    std::vector<std::string> string_tensor, std::vector<char>* serialized_data)
{
  std::string serialized = "";
  for (auto s : string_tensor) {
    uint32_t len = s.size();
    serialized.append(reinterpret_cast<const char*>(&len), sizeof(uint32_t));
    serialized.append(s);
  }

  std::copy(
      serialized.begin(), serialized.end(),
      std::back_inserter(*serialized_data));
}

}  // namespace

LoadManager::~LoadManager()
{
  nic::Error err;
  if (shared_memory_ctx_ != nullptr) {
    err = shared_memory_ctx_->UnregisterAllSharedMemory();
    if (!err.IsOk()) {
      std::cerr << "Unable to unregister all shared memory regions"
                << std::endl;
    }
    if (shared_memory_type_ == SharedMemoryType::SYSTEM_SHARED_MEMORY) {
      for (auto region : shared_memory_regions_) {
        err = nic::UnmapSharedMemory(
            shared_memory_regions_[region.first].first,
            shared_memory_regions_[region.first].second);
        if (!err.IsOk()) {
          std::cerr << "Unable to unmap shared memory with key ("
                    << region.first << "): Starting: "
                    << static_cast<void*>(
                           shared_memory_regions_[region.first].first)
                    << ", size: " << shared_memory_regions_[region.first].second
                    << std::endl;
        }
        err = nic::UnlinkSharedMemoryRegion(region.first);
        if (!err.IsOk()) {
          std::cerr << "Unable to unlink shared memory with key: "
                    << region.first << std::endl;
        }
      }
    } else if (shared_memory_type_ == SharedMemoryType::CUDA_SHARED_MEMORY) {
#ifdef TRTIS_ENABLE_GPU
      for (auto region : shared_memory_regions_) {
        cudaError_t cuda_err =
            cudaFree(shared_memory_regions_[region.first].first);
        if (cuda_err != cudaSuccess) {
          std::cerr << "Unable to free cuda shared memory for " << region.first
                    << ": Starting: "
                    << static_cast<void*>(
                           shared_memory_regions_[region.first].first)
                    << ", size: " << shared_memory_regions_[region.first].second
                    << std::endl;
        }
      }
#endif  // TRTIS_ENABLE_GPU
    }
  }
}

nic::Error
LoadManager::CheckHealth()
{
  // Check thread status to make sure that the actual concurrency level is
  // consistent to the one being reported
  // If some thread return early, main thread will return and
  // the worker thread's error message will be reported
  // when ConcurrencyManager's destructor get called.
  for (auto& thread_stat : threads_stat_) {
    if (!thread_stat->status_.IsOk()) {
      return nic::Error(
          ni::RequestStatusCode::INTERNAL,
          "Failed to maintain concurrency level requested."
          " Worker thread(s) failed to generate concurrent requests.");
    }
  }
  return nic::Error::Success;
}

nic::Error
LoadManager::SwapTimestamps(TimestampVector& new_timestamps)
{
  TimestampVector total_timestamp;
  // Gather request timestamps with proper locking from all the worker
  // threads
  for (auto& thread_stat : threads_stat_) {
    std::lock_guard<std::mutex> lock(thread_stat->mu_);
    total_timestamp.insert(
        total_timestamp.end(), thread_stat->request_timestamps_.begin(),
        thread_stat->request_timestamps_.end());
    thread_stat->request_timestamps_.clear();
  }
  // Swap the results
  total_timestamp.swap(new_timestamps);
  return nic::Error::Success;
}

nic::Error
LoadManager::GetAccumulatedContextStat(nic::InferContext::Stat* contexts_stat)
{
  for (auto& thread_stat : threads_stat_) {
    std::lock_guard<std::mutex> lock(thread_stat->mu_);
    for (auto& context_stat : thread_stat->contexts_stat_) {
      contexts_stat->completed_request_count +=
          context_stat.completed_request_count;
      contexts_stat->cumulative_total_request_time_ns +=
          context_stat.cumulative_total_request_time_ns;
      contexts_stat->cumulative_send_time_ns +=
          context_stat.cumulative_send_time_ns;
      contexts_stat->cumulative_receive_time_ns +=
          context_stat.cumulative_receive_time_ns;
    }
  }
  return nic::Error::Success;
}


LoadManager::LoadManager(
    const bool async,
    const std::unordered_map<std::string, std::vector<int64_t>>& input_shapes,
    const int32_t batch_size, const size_t max_threads,
    const size_t sequence_length, const SharedMemoryType shared_memory_type,
    const size_t output_shm_size,
    const std::shared_ptr<ContextFactory>& factory)
    : async_(async), input_shapes_(input_shapes), batch_size_(batch_size),
      max_threads_(max_threads), sequence_length_(sequence_length),
      shared_memory_type_(shared_memory_type),
      output_shm_size_(output_shm_size), factory_(factory)
{
  on_sequence_model_ =
      ((factory_->SchedulerType() == ContextFactory::SEQUENCE) ||
       (factory_->SchedulerType() == ContextFactory::ENSEMBLE_SEQUENCE));
}

nic::Error
LoadManager::InitManagerInputs(
    const size_t string_length, const std::string& string_data,
    const bool zero_input, const std::string& data_directory)
{
  std::unique_ptr<nic::InferContext> ctx;
  RETURN_IF_ERROR(factory_->CreateInferContext(&ctx));

  size_t max_input_byte_size = 0;

  for (const auto& input : ctx->Inputs()) {
    size_t batch1_num_strings = 1;
    // Validate user provided shape
    if (!input_shapes_.empty()) {
      auto it = input_shapes_.find(input->Name());
      if (it != input_shapes_.end()) {
        const auto& dims = it->second;
        const auto& config_dims = input->Dims();
        if (!ni::CompareDimsWithWildcard(config_dims, dims)) {
          return nic::Error(
              ni::RequestStatusCode::INVALID_ARG,
              "input '" + input->Name() + "' expects shape " +
                  ni::DimsListToString(config_dims) +
                  " and user supplied shape " + ni::DimsListToString(dims));
        }
      }
    }

    // For variable shape, set the shape if specified
    if (input->Shape().empty()) {
      auto it = input_shapes_.find(input->Name());
      if (it != input_shapes_.end()) {
        input->SetShape(it->second);
      }
    }

    const int64_t bs = input->ByteSize();
    if (bs < 0 && input->DType() != ni::DataType::TYPE_STRING) {
      if (input->Shape().empty()) {
        return nic::Error(
            ni::RequestStatusCode::INVALID_ARG,
            "input '" + input->Name() +
                "' has variable-size shape and the shape to be used is not "
                "specified, unable to create input values for model '" +
                ctx->ModelName() + "'");
      }
    }

    // Validate the shape specification for TYPE_STRING
    if (input->DType() == ni::DataType::TYPE_STRING) {
      bool is_variable_shape = false;
      for (const auto dim : input->Dims()) {
        if (dim == -1) {
          is_variable_shape = true;
          break;
        }
      }
      if (is_variable_shape && input->Shape().empty()) {
        return nic::Error(
            ni::RequestStatusCode::INVALID_ARG,
            "input '" + input->Name() +
                "' has variable-size shape and the shape to be used is "
                "not specified, unable to create input values for "
                "model '" +
                ctx->ModelName() + "'");
      }

      // Get the number of strings needed for this input batch-1
      batch1_num_strings = 1;
      if (!input->Shape().empty()) {
        for (const auto dim : input->Shape()) {
          batch1_num_strings *= dim;
        }
      } else {
        for (const auto dim : input->Dims()) {
          batch1_num_strings *= dim;
        }
      }
    }

    // Read provided data
    if (!data_directory.empty()) {
      if (input->DType() != ni::DataType::TYPE_STRING) {
        const auto file_path = data_directory + "/" + input->Name();
        auto it = input_data_.emplace(input->Name(), std::vector<char>()).first;
        RETURN_IF_ERROR(ReadFile(file_path, &it->second));
        size_t batch1_size = input->ByteSize();
        if (batch1_size != it->second.size()) {
          return nic::Error(
              ni::RequestStatusCode::INVALID_ARG,
              "input '" + input->Name() + "' requires " +
                  std::to_string(batch1_size) +
                  " bytes for each batch, but provided data has " +
                  std::to_string(it->second.size()) + " bytes");
        }
      } else {
        const auto file_path = data_directory + "/" + input->Name();
        std::vector<std::string> input_string_data;
        RETURN_IF_ERROR(ReadTextFile(file_path, &input_string_data));
        if (input_string_data.size() != batch1_num_strings) {
          return nic::Error(
              ni::RequestStatusCode::INVALID_ARG,
              "input '" + input->Name() + "' requires " +
                  std::to_string(batch1_num_strings) +
                  " strings for each batch, but provided data has " +
                  std::to_string(input_string_data.size()) + " strings.");
        }
        auto it = input_string_data_.emplace(input->Name(), std::vector<char>())
                      .first;
        SerializeStringTensor(input_string_data, &it->second);
      }
    } else {
      if (input->DType() != ni::DataType::TYPE_STRING) {
        max_input_byte_size =
            std::max(max_input_byte_size, (size_t)input->ByteSize());
      } else {
        // Generate string input and store it into map
        std::vector<std::string> input_string_data;
        input_string_data.resize(batch1_num_strings);
        if (!string_data.empty()) {
          for (size_t i = 0; i < batch1_num_strings; i++) {
            input_string_data[i] = string_data;
          }
        } else {
          for (size_t i = 0; i < batch1_num_strings; i++) {
            input_string_data[i] = GetRandomString(string_length);
          }
        }
        auto it = input_string_data_.emplace(input->Name(), std::vector<char>())
                      .first;
        SerializeStringTensor(input_string_data, &it->second);
      }
    }
  }

  // Create a zero or randomly (as indicated by zero_input_)
  // initialized buffer that is large enough to provide the largest
  // needed input. We (re)use this buffer for all input values.
  if (max_input_byte_size > 0) {
    if (zero_input) {
      input_buf_.resize(max_input_byte_size, 0);
    } else {
      input_buf_.resize(max_input_byte_size);
      for (auto& byte : input_buf_) {
        byte = rand();
      }
    }
  }

  // Reserve the required vector space
  threads_stat_.reserve(max_threads_);

  return nic::Error::Success;
}

nic::Error
LoadManager::InitSharedMemory()
{
  nic::Error err;

  RETURN_IF_ERROR(
      factory_->CreateSharedMemoryControlContext(&shared_memory_ctx_));

  // Calling this function for the clean start
  shared_memory_ctx_->UnregisterAllSharedMemory();

  std::unique_ptr<nic::InferContext> ctx;
  RETURN_IF_ERROR(factory_->CreateInferContext(&ctx));

  // Allocate the shared memory for outputs
  for (const auto& output : ctx->Outputs()) {
    int64_t batch1_bytesize = ctx->ByteSize(output->Dims(), output->DType());
    if (batch1_bytesize < 0) {
      batch1_bytesize = output_shm_size_;
    }
    uint8_t* output_shm_ptr;
    size_t alloc_size = batch1_bytesize * batch_size_;
    if (shared_memory_type_ == SharedMemoryType::SYSTEM_SHARED_MEMORY) {
      std::string shm_key("/" + output->Name());
      int shm_fd_op;
      RETURN_IF_ERROR(
          nic::CreateSharedMemoryRegion(shm_key, alloc_size, &shm_fd_op));
      RETURN_IF_ERROR(nic::MapSharedMemory(
          shm_fd_op, 0, alloc_size, (void**)&output_shm_ptr));

      shared_memory_regions_[output->Name()] =
          std::pair<uint8_t*, size_t>(output_shm_ptr, alloc_size);

      RETURN_IF_ERROR(shared_memory_ctx_->RegisterSharedMemory(
          output->Name(), shm_key, 0, alloc_size));
    } else {
#ifdef TRTIS_ENABLE_GPU
      cudaError_t cuda_err = cudaMalloc((void**)&output_shm_ptr, alloc_size);
      if (cuda_err != cudaSuccess) {
        return nic::Error(
            ni::RequestStatusCode::INTERNAL,
            "unable to allocate memory of " + std::to_string(alloc_size) +
                "bytes on gpu for output " + output->Name());
      }
      shared_memory_regions_[output->Name()] =
          std::pair<uint8_t*, size_t>(output_shm_ptr, alloc_size);

      cudaIpcMemHandle_t cuda_handle;
      RETURN_IF_ERROR(CreateCUDAIPCHandle(&cuda_handle, (void*)output_shm_ptr));
      // Using GPU with device id 0
      RETURN_IF_ERROR(shared_memory_ctx_->RegisterCudaSharedMemory(
          output->Name(), cuda_handle, alloc_size, 0));
#endif  // TRTIS_ENABLE_GPU
    }
  }

  // Set the provided shape for variable shape tensor
  for (const auto& input : ctx->Inputs()) {
    if (input->Shape().empty()) {
      auto it = input_shapes_.find(input->Name());
      if (it != input_shapes_.end()) {
        input->SetShape(it->second);
      }
    }
  }

  for (const auto& input : ctx->Inputs()) {
    const uint8_t* data_ptr;
    size_t batch1_bytesize;
    RETURN_IF_ERROR(GetInputData(input, &data_ptr, &batch1_bytesize));

    // create the shared memory region for the input
    uint8_t* input_shm_ptr;
    size_t alloc_size = batch1_bytesize * batch_size_;

    if (shared_memory_type_ == SharedMemoryType::SYSTEM_SHARED_MEMORY) {
      std::string shm_key("/" + input->Name());
      int shm_fd_ip;
      RETURN_IF_ERROR(
          nic::CreateSharedMemoryRegion(shm_key, alloc_size, &shm_fd_ip));
      RETURN_IF_ERROR(nic::MapSharedMemory(
          shm_fd_ip, 0, alloc_size, (void**)&input_shm_ptr));
      shared_memory_regions_[input->Name()] =
          std::pair<uint8_t*, size_t>(input_shm_ptr, alloc_size);

      // Populate the region with data
      size_t count = 0;
      while (count < batch_size_) {
        memcpy(
            input_shm_ptr + (count * batch1_bytesize), data_ptr,
            batch1_bytesize);
        count++;
      }

      // Register the region with TRTIS
      RETURN_IF_ERROR(shared_memory_ctx_->RegisterSharedMemory(
          input->Name(), shm_key, 0, alloc_size));
    } else {
#ifdef TRTIS_ENABLE_GPU
      cudaError_t cuda_err = cudaMalloc((void**)&input_shm_ptr, alloc_size);
      if (cuda_err != cudaSuccess) {
        return nic::Error(
            ni::RequestStatusCode::INTERNAL,
            "unable to allocate memory of " + std::to_string(alloc_size) +
                "bytes on gpu for input " + input->Name());
      }

      shared_memory_regions_[input->Name()] =
          std::pair<uint8_t*, size_t>(input_shm_ptr, alloc_size);

      // Populate the region with data
      size_t count = 0;
      while (count < batch_size_) {
        cudaError_t cuda_err = cudaMemcpy(
            (void*)(input_shm_ptr + (count * batch1_bytesize)), (void*)data_ptr,
            batch1_bytesize, cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) {
          return nic::Error(
              ni::RequestStatusCode::INTERNAL,
              "Failed to copy data to cuda shared memory for " + input->Name());
        }
        count++;
      }

      cudaIpcMemHandle_t cuda_handle;
      RETURN_IF_ERROR(CreateCUDAIPCHandle(&cuda_handle, (void*)input_shm_ptr));

      // Register the region with TRTIS
      RETURN_IF_ERROR(shared_memory_ctx_->RegisterCudaSharedMemory(
          input->Name(), cuda_handle, alloc_size, 0));
#endif  // TRTIS_ENABLE_GPU
    }
  }
  return nic::Error::Success;
}

nic::Error
LoadManager::PrepareInfer(
    std::unique_ptr<nic::InferContext>* ctx,
    std::unique_ptr<nic::InferContext::Options>* options)
{
  RETURN_IF_ERROR(factory_->CreateInferContext(ctx));

  uint64_t max_batch_size = (*ctx)->MaxBatchSize();

  // Model specifying maximum batch size of 0 indicates that batching
  // is not supported and so the input tensors do not expect a "N"
  // dimension (and 'batch_size' should be 1 so that only a single
  // image instance is inferred at a time).
  if (max_batch_size == 0) {
    if (batch_size_ != 1) {
      return nic::Error(
          ni::RequestStatusCode::INVALID_ARG,
          "expecting batch size 1 for model '" + (*ctx)->ModelName() +
              "' which does not support batching");
    }
  } else if (batch_size_ > max_batch_size) {
    return nic::Error(
        ni::RequestStatusCode::INVALID_ARG,
        "expecting batch size <= " + std::to_string(max_batch_size) +
            " for model '" + (*ctx)->ModelName() + "'");
  }

  // Prepare context for 'batch_size' batches. Request that all
  // outputs be returned.
  // Only set options if it has not been created, otherwise,
  // assuming that the options for this model has been created previously
  if (*options == nullptr) {
    RETURN_IF_ERROR(nic::InferContext::Options::Create(options));

    (*options)->SetBatchSize(batch_size_);
    for (const auto& output : (*ctx)->Outputs()) {
      (*options)->AddRawResult(output);
    }
  }

  RETURN_IF_ERROR((*ctx)->SetRunOptions(*(*options)));

  // Set the provided shape for variable shape tensor
  for (const auto& input : (*ctx)->Inputs()) {
    if (input->Shape().empty()) {
      auto it = input_shapes_.find(input->Name());
      if (it != input_shapes_.end()) {
        input->SetShape(it->second);
      }
    }
  }

  // Initialize inputs
  for (const auto& input : (*ctx)->Inputs()) {
    RETURN_IF_ERROR(input->Reset());

    const uint8_t* data_ptr;
    size_t batch1_bytesize;
    RETURN_IF_ERROR(GetInputData(input, &data_ptr, &batch1_bytesize));

    for (size_t i = 0; i < batch_size_; ++i) {
      RETURN_IF_ERROR(input->SetRaw(data_ptr, batch1_bytesize));
    }
  }

  return nic::Error::Success;
}

nic::Error
LoadManager::PrepareSharedMemoryInfer(
    std::unique_ptr<nic::InferContext>* ctx,
    std::unique_ptr<nic::InferContext::Options>* options)
{
  nic::Error err;
  RETURN_IF_ERROR(factory_->CreateInferContext(ctx));

  uint64_t max_batch_size = (*ctx)->MaxBatchSize();

  // Model specifying maximum batch size of 0 indicates that batching
  // is not supported and so the input tensors do not expect a "N"
  // dimension (and 'batch_size' should be 1 so that only a single
  // image instance is inferred at a time).
  if (max_batch_size == 0) {
    if (batch_size_ != 1) {
      return nic::Error(
          ni::RequestStatusCode::INVALID_ARG,
          "expecting batch size 1 for model '" + (*ctx)->ModelName() +
              "' which does not support batching");
    }
  } else if (batch_size_ > max_batch_size) {
    return nic::Error(
        ni::RequestStatusCode::INVALID_ARG,
        "expecting batch size <= " + std::to_string(max_batch_size) +
            " for model '" + (*ctx)->ModelName() + "'");
  }

  // Only set options if it has not been created, otherwise,
  // assuming that the options for this model has been created previously
  if (*options == nullptr) {
    RETURN_IF_ERROR(nic::InferContext::Options::Create(options));
    (*options)->SetBatchSize(batch_size_);
    for (const auto& output : (*ctx)->Outputs()) {
      (*options)->AddSharedMemoryResult(
          output, output->Name(), 0, output_shm_size_);
    }
  }

  RETURN_IF_ERROR((*ctx)->SetRunOptions(*(*options)));

  for (const auto& input : (*ctx)->Inputs()) {
    RETURN_IF_ERROR(input->SetSharedMemory(
        input->Name(), 0, shared_memory_regions_[input->Name()].second));
    if (input->Shape().empty()) {
      auto it = input_shapes_.find(input->Name());
      if (it != input_shapes_.end()) {
        input->SetShape(it->second);
      }
    }
  }

  return nic::Error::Success;
}

nic::Error
LoadManager::GetInputData(
    std::shared_ptr<nic::InferContext::Input> input, const uint8_t** data,
    size_t* batch1_size)
{
  if (input->DType() != ni::DataType::TYPE_STRING) {
    // if available, use provided data instead
    auto it = input_data_.find(input->Name());
    if (it != input_data_.end()) {
      *data = (const uint8_t*)&(it->second)[0];
    } else if (input_buf_.size() != 0) {
      *batch1_size = (size_t)input->ByteSize();
      *data = &input_buf_[0];
    } else {
      return nic::Error(
          ni::RequestStatusCode::INVALID_ARG,
          "unable to find data for input '" + input->Name() + "'.");
    }
  } else {
    std::vector<char>* string_data;
    auto it = input_string_data_.find(input->Name());
    if (it != input_string_data_.end()) {
      string_data = &it->second;
      *batch1_size = string_data->size();
      *data = (const uint8_t*)&it->second[0];
    } else {
      return nic::Error(
          ni::RequestStatusCode::INVALID_ARG,
          "unable to find data for input '" + input->Name() + "'.");
    }
  }
  return nic::Error::Success;
}

size_t
LoadManager::GetRandomLength(double offset_ratio)
{
  int random_offset = ((2.0 * rand() / double(RAND_MAX)) - 1.0) * offset_ratio *
                      sequence_length_;
  if (int(sequence_length_) + random_offset <= 0) {
    return 1;
  }
  return sequence_length_ + random_offset;
}

void
LoadManager::StopWorkerThreads()
{
  early_exit = true;
  // wake up all threads
  wake_signal_.notify_all();

  size_t cnt = 0;
  for (auto& thread : threads_) {
    thread.join();
    if (!threads_stat_[cnt]->status_.IsOk()) {
      std::cerr << "Thread [" << cnt
                << "] had error: " << (threads_stat_[cnt]->status_)
                << std::endl;
    }
    cnt++;
  }
}
