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
      output_shm_size_(output_shm_size), factory_(factory),
      using_json_data_(false), next_corr_id_(1)
{
  on_sequence_model_ =
      ((factory_->SchedulerType() == ContextFactory::SEQUENCE) ||
       (factory_->SchedulerType() == ContextFactory::ENSEMBLE_SEQUENCE));

  data_loader_.reset(new DataLoader(batch_size));
}

nic::Error
LoadManager::InitManagerInputs(
    const size_t string_length, const std::string& string_data,
    const bool zero_input, std::vector<std::string>& user_data)
{
  std::unique_ptr<nic::InferContext> ctx;
  RETURN_IF_ERROR(factory_->CreateInferContext(&ctx));

  for (const auto& input : ctx->Inputs()) {
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
    }
  }

  // Read provided data
  if (!user_data.empty()) {
    if (IsDirectory(user_data[0])) {
      RETURN_IF_ERROR(
          data_loader_->ReadDataFromDir(ctx->Inputs(), user_data[0]));
    } else {
      using_json_data_ = true;
      for (const auto& json_file : user_data) {
        RETURN_IF_ERROR(
            data_loader_->ReadDataFromJSON(ctx->Inputs(), json_file));
      }
      std::cout << " Successfully read data for "
                << data_loader_->GetDataStreamsCount() << " stream/streams";
      if (data_loader_->GetDataStreamsCount() == 1) {
        std::cout << " with " << data_loader_->GetTotalSteps(0)
                  << " step/steps";
      }
      std::cout << "." << std::endl;
    }
  } else {
    RETURN_IF_ERROR(data_loader_->GenerateData(
        ctx->Inputs(), zero_input, string_length, string_data));
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
    for (int i = 0; i < (int)data_loader_->GetDataStreamsCount(); i++) {
      for (int j = 0; j < (int)data_loader_->GetTotalSteps(i);
           j += batch_size_) {
        // Extract the data for requested batch size
        std::vector<const uint8_t*> data_ptrs;
        std::vector<size_t> byte_size;
        size_t alloc_size = 0;
        size_t count = 0;
        while (count < batch_size_) {
          const uint8_t* data_ptr;
          size_t batch1_bytesize;
          RETURN_IF_ERROR(data_loader_->GetInputData(
              input, i, (j + count) % data_loader_->GetTotalSteps(i), &data_ptr,
              &batch1_bytesize));
          data_ptrs.push_back(data_ptr);
          byte_size.push_back(batch1_bytesize);
          alloc_size += batch1_bytesize;
          count++;
        }

        // Generate the shared memory region name
        std::string key_name(
            input->Name() + "_" + std::to_string(i) + "_" + std::to_string(j));

        uint8_t* input_shm_ptr;
        if (shared_memory_type_ == SharedMemoryType::SYSTEM_SHARED_MEMORY) {
          std::string shm_key("/" + key_name);
          int shm_fd_ip;
          RETURN_IF_ERROR(
              nic::CreateSharedMemoryRegion(shm_key, alloc_size, &shm_fd_ip));
          RETURN_IF_ERROR(nic::MapSharedMemory(
              shm_fd_ip, 0, alloc_size, (void**)&input_shm_ptr));
          shared_memory_regions_[key_name] =
              std::pair<uint8_t*, size_t>(input_shm_ptr, alloc_size);

          // Populate the region with data
          size_t count = 0;
          size_t offset = 0;
          while (count < batch_size_) {
            memcpy(input_shm_ptr + offset, data_ptrs[count], byte_size[count]);
            offset += byte_size[count];
            count++;
          }

          // Register the region with TRTIS
          RETURN_IF_ERROR(shared_memory_ctx_->RegisterSharedMemory(
              key_name, shm_key, 0, alloc_size));
        } else {
#ifdef TRTIS_ENABLE_GPU
          cudaError_t cuda_err = cudaMalloc((void**)&input_shm_ptr, alloc_size);
          if (cuda_err != cudaSuccess) {
            return nic::Error(
                ni::RequestStatusCode::INTERNAL,
                "unable to allocate memory of " + std::to_string(alloc_size) +
                    "bytes on gpu for input " + key_name);
          }

          shared_memory_regions_[key_name] =
              std::pair<uint8_t*, size_t>(input_shm_ptr, alloc_size);

          // Populate the region with data
          size_t count = 0;
          size_t offset = 0;
          while (count < batch_size_) {
            cudaError_t cuda_err = cudaMemcpy(
                (void*)(input_shm_ptr + offset), (void*)data_ptrs[count],
                byte_size[count], cudaMemcpyHostToDevice);
            if (cuda_err != cudaSuccess) {
              return nic::Error(
                  ni::RequestStatusCode::INTERNAL,
                  "Failed to copy data to cuda shared memory for " + key_name);
            }
            offset += byte_size[count];
            count++;
          }

          cudaIpcMemHandle_t cuda_handle;
          RETURN_IF_ERROR(
              CreateCUDAIPCHandle(&cuda_handle, (void*)input_shm_ptr));

          // Register the region with TRTIS
          RETURN_IF_ERROR(shared_memory_ctx_->RegisterCudaSharedMemory(
              key_name, cuda_handle, alloc_size, 0));
#endif  // TRTIS_ENABLE_GPU
        }
      }
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
    RETURN_IF_ERROR(
        data_loader_->GetInputData(input, 0, 0, &data_ptr, &batch1_bytesize));

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
    std::string key_name(
        input->Name() + "_" + std::to_string(0) + "_" + std::to_string(0));
    RETURN_IF_ERROR(input->SetSharedMemory(
        key_name, 0, shared_memory_regions_[key_name].second));
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
LoadManager::UpdateInputs(
    const std::vector<std::shared_ptr<nic::InferContext::Input>>& inputs,
    int stream_index, int step_index)
{
  // Validate update parameters here
  size_t data_stream_count = data_loader_->GetDataStreamsCount();
  if (stream_index < 0 || stream_index >= (int)data_stream_count) {
    return nic::Error(
        ni::RequestStatusCode::INTERNAL,
        "stream_index for retrieving the data should be less than " +
            std::to_string(data_stream_count) + ", got " +
            std::to_string(stream_index));
  }
  size_t step_count = data_loader_->GetTotalSteps(stream_index);
  if (step_index < 0 || step_index >= (int)step_count) {
    return nic::Error(
        ni::RequestStatusCode::INTERNAL,
        "step_id for retrieving the data should be less than " +
            std::to_string(step_count) + ", got " + std::to_string(step_index));
  }

  if (shared_memory_type_ == SharedMemoryType::NO_SHARED_MEMORY) {
    RETURN_IF_ERROR(SetInputs(inputs, stream_index, step_index));
  } else {
    RETURN_IF_ERROR(SetInputsSharedMemory(inputs, stream_index, step_index));
  }

  return nic::Error::Success;
}

nic::Error
LoadManager::SetInputs(
    const std::vector<std::shared_ptr<nic::InferContext::Input>>& inputs,
    const int stream_index, const int step_index)
{
  for (const auto& input : inputs) {
    RETURN_IF_ERROR(input->Reset());

    const uint8_t* data_ptr;
    size_t batch1_bytesize;

    if (!on_sequence_model_) {
      for (size_t i = 0; i < batch_size_; ++i) {
        RETURN_IF_ERROR(data_loader_->GetInputData(
            input, 0, (step_index + i) % data_loader_->GetTotalSteps(0),
            &data_ptr, &batch1_bytesize));
        RETURN_IF_ERROR(input->SetRaw(data_ptr, batch1_bytesize));
      }
    } else {
      // Sequence models only support single batch_size_
      RETURN_IF_ERROR(data_loader_->GetInputData(
          input, stream_index, step_index, &data_ptr, &batch1_bytesize));
      RETURN_IF_ERROR(input->SetRaw(data_ptr, batch1_bytesize));
    }
  }
  return nic::Error::Success;
}


nic::Error
LoadManager::SetInputsSharedMemory(
    const std::vector<std::shared_ptr<nic::InferContext::Input>>& inputs,
    const int stream_index, const int step_index)
{
  for (const auto& input : inputs) {
    RETURN_IF_ERROR(input->Reset());

    std::string region_name(
        input->Name() + '_' + std::to_string(stream_index) + "_" +
        std::to_string(step_index));

    RETURN_IF_ERROR(input->SetSharedMemory(
        region_name, 0, shared_memory_regions_[region_name].second));
  }
  return nic::Error::Success;
}

void
LoadManager::InitNewSequence(int sequence_id)
{
  sequence_stat_[sequence_id]->corr_id_ = next_corr_id_++;
  if (!using_json_data_) {
    size_t new_length = GetRandomLength(0.2);
    sequence_stat_[sequence_id]->remaining_queries_ =
        new_length == 0 ? 1 : new_length;
  } else {
    // Selecting next available data stream in a round-robin fashion.
    // TODO: A mode to randomly pick data stream for new sequences.
    sequence_stat_[sequence_id]->data_stream_id_ =
        sequence_stat_[sequence_id]->corr_id_ %
        data_loader_->GetDataStreamsCount();
    sequence_stat_[sequence_id]->remaining_queries_ =
        data_loader_->GetTotalSteps(
            sequence_stat_[sequence_id]->data_stream_id_);
  }
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
