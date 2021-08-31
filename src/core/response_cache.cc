// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "src/core/response_cache.h"
#include "src/core/logging.h"

// TODO: Figure out how to use Triton log macros in unit test
#define DEBUG(x)                               \
  do {                                         \
    std::cerr << "[DEBUG] " << x << std::endl; \
  } while (0)

namespace nvidia { namespace inferenceserver {

RequestResponseCache::RequestResponseCache(const uint64_t size)
{
  // Allocate buffer
  buffer_ = malloc(size);
  // Exit early if buffer allocation failed
  if (buffer_ == nullptr) {
    throw std::runtime_error("failed to allocate buffer");
  }

  // Create cache as managed buffer
  managed_buffer_ = boost::interprocess::managed_external_buffer(
      boost::interprocess::create_only_t{}, buffer_, size);
}

RequestResponseCache::~RequestResponseCache()
{
  // Deallocate each chunk from managed buffer
  for (auto& iter : cache_) {
    auto& entry = iter.second;
    for (auto& output : entry.outputs_) {
      if (output.buffer_ != nullptr) {
        managed_buffer_.deallocate(output.buffer_);
      }
    }
  }

  // Validate we freed all underlying memory managed by cache
  if (!managed_buffer_.all_memory_deallocated()) {
    // Destructors can't throw exceptions
    LOG_ERROR << "failed to free managed cache memory";
  }

  // Free total cache buffer
  if (buffer_ != nullptr) {
    free(buffer_);
  }
}

Status
RequestResponseCache::Hash(const InferenceRequest& request, uint64_t* key)
{
  Status status = Status::Success;
  std::size_t seed = 0;
  // Add request model name to hash
  boost::hash_combine(seed, request.ModelName());
  // Add request model version to hash
  // TODO: RequestedModelVersion or ActualModelVersion ?
  //boost::hash_combine(seed, request.RequestedModelVersion());
  boost::hash_combine(seed, request.ActualModelVersion());

  // TODO: Use PrepareForInference + ImmutableInputs?
  const auto& inputs = request.OriginalInputs();
  for (const auto& input : inputs) {
    // Add input name to hash
    boost::hash_combine(seed, input.second.Name());
    // Fetch input buffer for hashing raw data
    const void* buffer = nullptr;
    uint64_t buffer_byte_size = 0;
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;
    const uint32_t index = 0;
    status = input.second.DataBuffer(
        index, &buffer, &buffer_byte_size, &memory_type, &memory_type_id);
    if (!status.IsOk()) {
      buffer = nullptr;
      buffer_byte_size = 0;
      return status;
    }
    // Add each byte of input buffer to hash
    const unsigned char* tmp = static_cast<const unsigned char*>(buffer);
    for (uint64_t byte = 0; byte < buffer_byte_size; byte++) {
      boost::hash_combine(seed, tmp[byte]);
    }
  }

  *key = static_cast<uint64_t>(seed);
  return Status::Success;
}

Status
RequestResponseCache::Lookup(const uint64_t key, InferenceResponse* ptr)
{
  auto iter = cache_.find(key);
  if (iter == cache_.end()) {
    return Status(Status::Code::INTERNAL, "key not found in cache");
  }
  // Populate passed-in "ptr" from cache entry
  auto entry = iter->second;
  // Build InferenceResponse from CacheEntry
  auto status = BuildInferenceResponse(entry, ptr);
  if (!status.IsOk()) {
    return status;
  }

  // Update this key to front of LRU list
  UpdateLRU(iter);

  return Status::Success;
}

Status
RequestResponseCache::Insert(
    const uint64_t key, const InferenceResponse& response)
{
  // Exit early if key already exists in cache
  auto iter = cache_.find(key);
  if (iter != cache_.end()) {
    return Status(
        Status::Code::INTERNAL,
        "key [" + std::to_string(key) + "] already exists in cache");
  }

  // Construct cache entry from response
  auto entry = CacheEntry();
  auto status = BuildCacheEntry(response, &entry);
  if (!status.IsOk()) {
    return status;
  }

  // Insert entry into cache
  DEBUG("Inserting key [" + std::to_string(key) + "] into cache.");
  auto cache_pair = cache_.insert({key, entry});
  bool ok = cache_pair.second;
  // Exit early if cache insertion failed
  if (!ok) {
    return Status(Status::Code::INTERNAL, "Cache insertion failed");
  }
  // Update LRU with new cache entry
  auto cache_iter = cache_pair.first;
  UpdateLRU(cache_iter);
  return Status::Success;
}

Status
RequestResponseCache::BuildCacheEntry(
    const InferenceResponse& response, CacheEntry* entry)
{
  auto status = Status::Success;

  // Build cache entry data from response outputs
  for (const auto& response_output : response.Outputs()) {
    auto cache_output = Output();

    // Fetch output buffer details
    const void* response_buffer = nullptr;
    size_t response_byte_size = 0;
    TRITONSERVER_MemoryType response_memory_type;
    int64_t response_memory_type_id;
    void* userp;
    // TODO: How to handle different memory types? GPU vs CPU vs Pinned, etc.
    status = response_output.DataBuffer(
        &response_buffer, &response_byte_size, &response_memory_type,
        &response_memory_type_id, &userp);

    // Exit early if we fail to get output buffer from response
    if (!status.IsOk()) {
      return status;
    }

    // Exit early if response buffer from output is invalid
    if (response_buffer == nullptr) {
      return Status(
          Status::Code::INTERNAL, "Response buffer from output was nullptr");
    }

    // Exit early if cache entry will be larger than available cache size
    if (response_byte_size > managed_buffer_.get_size()) {
      return Status(
          Status::Code::INTERNAL,
          "Cache entry is larger than total cache size");
    }

    // If cache doesn't enough space, evict until enough is available
    while (response_byte_size > managed_buffer_.get_free_memory()) {
      status = Evict();
      if (!status.IsOk()) {
        return status;
      }
    }

    // Set output metadata
    cache_output.name_ = response_output.Name();
    cache_output.dtype_ = response_output.DType();
    cache_output.shape_ = response_output.Shape();
    cache_output.buffer_size_ = static_cast<uint64_t>(response_byte_size);

    // Allocate buffer for response output in cache entry
    cache_output.buffer_ =
        managed_buffer_.allocate(response_byte_size, std::nothrow_t{});
    // Exit early if we fail to allocate from managed buffer
    if (cache_output.buffer_ == nullptr) {
      return Status(
          Status::Code::INTERNAL,
          "Failed to allocate buffer from managed buffer");
    }

    // Copy data from response buffer to cache entry output buffer
    // TODO: How to differently handle different memory types?
    //       GPU vs. CPU memory, etc.
    std::memcpy(cache_output.buffer_, response_buffer, response_byte_size);
    // Add each output to cache entry
    entry->outputs_.push_back(cache_output);
  }

  return Status::Success;
}


Status
RequestResponseCache::BuildInferenceResponse(
    const CacheEntry& entry, InferenceResponse* response)
{
  auto status = Status::Success;
  if (response == nullptr) {
    return Status(Status::Code::INTERNAL, "invalid response ptr passed in");
  }

  for (auto& cache_output : entry.outputs_) {
    InferenceResponse::Output* response_output = nullptr;
    status = response->AddOutput(
        cache_output.name_, cache_output.dtype_, cache_output.shape_,
        &response_output);
    if (!status.IsOk()) {
      return status;
    }

    if (response_output == nullptr) {
      return Status(
          Status::Code::INTERNAL,
          "InferenceResponse::Output pointer as nullptr");
    }

    // TODO: Assuming CPU memory only for now
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;

    void* vp = cache_output.buffer_;
    status = response_output->AllocateDataBuffer(
        &vp, cache_output.buffer_size_, &memory_type, &memory_type_id);
    if (!status.IsOk()) {
      return status;
    }

    // TODO: Add field to InferenceResponse to indicate this was from cache
    // response.cached = true;
  }

  return status;
}

// LRU
Status
RequestResponseCache::Evict()
{
  // Least recently used key in back of LRU list
  uint64_t lru_key = lru_.back();
  DEBUG("Evicting key [" + std::to_string(lru_key) + "] from cache.");

  // Find cache entry for least recently used key
  auto iter = cache_.find(lru_key);
  // Error check if key isn't in cache, but this shouldn't happen in evict
  // and probably indicates a bug
  if (iter == cache_.end()) {
    return Status(
        Status::Code::INTERNAL,
        "key [" + std::to_string(lru_key) +
            "] not found in cache during eviction: this indicates a bug in the "
            "code");
  }
  // Get size of cache entry being evicted to update available size
  auto entry = iter->second;
  // Free managed memory used in cache entry's outputs
  for (auto& output : entry.outputs_) {
    managed_buffer_.deallocate(output.buffer_);
  }

  // Remove LRU entry from cache
  cache_.erase(lru_key);
  // Remove LRU key from LRU list
  lru_.pop_back();
  // Increment number of evictions
  num_evictions_++;

  return Status::Success;
}

// LRU
void
RequestResponseCache::UpdateLRU(
    std::unordered_map<uint64_t, CacheEntry>::iterator& cache_iter)
{
  const auto& key = cache_iter->first;
  auto& cache_entry = cache_iter->second;
  // Remove key from LRU list if it was already in there
  auto lru_iter = std::find(lru_.begin(), lru_.end(), key);
  if (lru_iter != lru_.end()) {
    lru_.erase(lru_iter);
  }
  // Add key to front of LRU list since it's most recently used
  lru_.push_front(key);
  // Set CacheEntry LRU iterator to new LRU key location
  cache_entry.lru_iter_ = lru_.begin();
}

}}  // namespace nvidia::inferenceserver
