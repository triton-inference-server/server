// Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "src/core/infer_stats.h"
#include "src/core/logging.h"

namespace {

std::string
PointerToString(void* ptr)
{
  std::stringstream ss;
  ss << ptr;
  return ss.str();
}

}  // namespace

namespace nvidia { namespace inferenceserver {

Status
RequestResponseCache::Create(
    uint64_t cache_size, std::unique_ptr<RequestResponseCache>* cache)
{
  cache->reset(new RequestResponseCache(cache_size));

  return Status::Success;
}

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

  LOG_INFO << "Response Cache is created at '" << PointerToString(buffer_)
           << "' with size " << size;
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
RequestResponseCache::Lookup(
    const uint64_t key, InferenceResponse* ptr, InferenceRequest* request)
{
  // Lock on cache lookup
  std::lock_guard<std::recursive_mutex> lk(cache_mtx_);

  if (request == nullptr) {
    return Status(
        Status::Code::INTERNAL, "Cache Lookup passed a nullptr request");
  }

  // Capture start lookup latency
  request->CaptureCacheLookupStartNs();

  num_lookups_++;
  LOG_VERBOSE(1) << "Looking up key [" + std::to_string(key) + "] in cache.";

  // Search cache for request hash key
  auto iter = cache_.find(key);
  if (iter == cache_.end()) {
    num_misses_++;
    LOG_VERBOSE(1) << "MISS for key [" + std::to_string(key) + "] in cache.";
    CaptureLookupEndTime(request);
    return Status(Status::Code::INTERNAL, "key not found in cache");
  }

  // If find succeeds, it's a cache hit
  num_hits_++;
  LOG_VERBOSE(1) << "HIT for key [" + std::to_string(key) + "] in cache.";

  // Populate passed-in "ptr" from cache entry
  auto entry = iter->second;
  // Build InferenceResponse from CacheEntry
  auto status = BuildInferenceResponse(entry, ptr);
  if (!status.IsOk()) {
    CaptureLookupEndTime(request);
    return status;
  }

  // Update this key to front of LRU list
  UpdateLRU(iter);
  LOG_VERBOSE(1) << "Using cached response for key [" + std::to_string(key) +
                        "].";
  CaptureLookupEndTime(request);
  return Status::Success;
}

Status
RequestResponseCache::Insert(
    const uint64_t key, const InferenceResponse& response,
    InferenceRequest* request)
{
  // Lock on cache insertion
  std::lock_guard<std::recursive_mutex> lk(cache_mtx_);

  // Capture start insertion latency
  request->CaptureCacheInsertionStartNs();

  // Exit early if key already exists in cache
  auto iter = cache_.find(key);
  if (iter != cache_.end()) {
    CaptureInsertionEndTime(request);
    return Status(
        Status::Code::ALREADY_EXISTS,
        "key [" + std::to_string(key) + "] already exists in cache");
  }

  // Construct cache entry from response
  auto entry = CacheEntry();
  auto status = BuildCacheEntry(response, &entry);
  if (!status.IsOk()) {
    CaptureInsertionEndTime(request);
    return status;
  }

  // Insert entry into cache
  LOG_VERBOSE(1) << "Inserting key [" + std::to_string(key) + "] into cache.";
  auto cache_pair = cache_.insert({key, entry});
  // Exit early if cache insertion failed
  if (!cache_pair.second) {
    CaptureInsertionEndTime(request);
    return Status(Status::Code::INTERNAL, "Cache insertion failed");
  }
  // Update LRU with new cache entry
  auto cache_iter = cache_pair.first;
  UpdateLRU(cache_iter);

  CaptureInsertionEndTime(request);
  return Status::Success;
}

// LRU
Status
RequestResponseCache::Evict()
{
  // Lock on cache eviction
  std::lock_guard<std::recursive_mutex> lk(cache_mtx_);

  // Nothing to evict if cache is empty
  if (NumEntries() == 0) {
    return Status(Status::Code::INTERNAL, "Cache is empty, nothing to evict.");
  }

  // Least recently used key in back of LRU list
  uint64_t lru_key = lru_.back();
  LOG_VERBOSE(1) << "Evicting key [" + std::to_string(lru_key) +
                        "] from cache.";

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
    // Lock on buffer deallocation
    std::lock_guard<std::recursive_mutex> lk(buffer_mtx_);
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

// Helpers
void
RequestResponseCache::UpdateLRU(
    std::unordered_map<uint64_t, CacheEntry>::iterator& cache_iter)
{
  // Lock on cache update
  std::lock_guard<std::recursive_mutex> lk(cache_mtx_);

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

Status
RequestResponseCache::BuildCacheEntry(
    const InferenceResponse& response, CacheEntry* entry)
{
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
    RETURN_IF_ERROR(response_output.DataBuffer(
        &response_buffer, &response_byte_size, &response_memory_type,
        &response_memory_type_id, &userp));

    // TODO: Handle other memory types
    if (response_memory_type != TRITONSERVER_MEMORY_CPU &&
        response_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
      return Status(
          Status::Code::INTERNAL,
          "Only input buffers in CPU memory are allowed in cache currently");
    }

    // Exit early if response buffer from output is invalid
    if (response_buffer == nullptr) {
      return Status(
          Status::Code::INTERNAL, "Response buffer from output was nullptr");
    }

    // Lock on managed buffer references
    {
      std::lock_guard<std::recursive_mutex> lk(buffer_mtx_);

      // Exit early if cache entry will be larger than available cache size
      if (response_byte_size > managed_buffer_.get_size()) {
        return Status(
            Status::Code::INTERNAL,
            "Cache entry is larger than total cache size");
      }

      // If cache doesn't have enough space, evict until enough space available
      // NOTE: FreeBytes() doesn't account for allocator overhead so allocation
      //       may fail even if response_byte_size is less than FreeBytes()
      while (response_byte_size > FreeBytes()) {
        LOG_VERBOSE(1) << "EVICT: Response larger than remaining available "
                          "memory, attempting to evict from cache.";
        RETURN_IF_ERROR(Evict());
      }

      // Attempt to allocate buffer until success or eviction from cache fails
      while (cache_output.buffer_ == nullptr) {
        // Allocate buffer for response output in cache entry
        cache_output.buffer_ =
            managed_buffer_.allocate(response_byte_size, std::nothrow_t{});
        // Attempt to evict if allocation fails
        if (cache_output.buffer_ == nullptr) {
          LOG_VERBOSE(1) << "FAILED to allocate buffer in cache. Attempting to "
                            "evict an entry.";
          // Exit out if Eviction fails
          RETURN_IF_ERROR(Evict());
        }
      }

      // Copy data from response buffer to cache entry output buffer
      // TODO: Handle different memory types: GPU, SHM, Pinned, etc.
      std::memcpy(cache_output.buffer_, response_buffer, response_byte_size);

      // Set output metadata
      cache_output.name_ = response_output.Name();
      cache_output.dtype_ = response_output.DType();
      cache_output.shape_ = response_output.Shape();
      cache_output.buffer_size_ = static_cast<uint64_t>(response_byte_size);
    }

    // Add each output to cache entry
    entry->outputs_.push_back(cache_output);
  }

  return Status::Success;
}


Status
RequestResponseCache::BuildInferenceResponse(
    const CacheEntry& entry, InferenceResponse* response)
{
  if (response == nullptr) {
    return Status(Status::Code::INTERNAL, "invalid response ptr passed in");
  }

  // Lock on cache references
  {
    std::lock_guard<std::recursive_mutex> lk(cache_mtx_);

    // TODO: What should we do if [response] already contains
    //       some outputs? Currently it will just append outputs
    if (response->Outputs().size() != 0) {
      return Status(
          Status::Code::INTERNAL,
          "InferenceResponse already contains some outputs");
    }

    for (auto& cache_output : entry.outputs_) {
      InferenceResponse::Output* response_output = nullptr;
      RETURN_IF_ERROR(response->AddOutput(
          cache_output.name_, cache_output.dtype_, cache_output.shape_,
          &response_output));

      if (response_output == nullptr) {
        return Status(
            Status::Code::INTERNAL,
            "InferenceResponse::Output pointer as nullptr");
      }

      // TODO: Assuming CPU memory only for now
      TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t memory_type_id = 0;

      // Allocate buffer for inference response
      void* buffer;
      RETURN_IF_ERROR(response_output->AllocateDataBuffer(
          &buffer, cache_output.buffer_size_, &memory_type, &memory_type_id));

      // TODO: Handle other memory types
      if (memory_type != TRITONSERVER_MEMORY_CPU &&
          memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
        return Status(
            Status::Code::INTERNAL,
            "Only input buffers in CPU memory are allowed in cache currently");
      }

      if (buffer == nullptr) {
        return Status(
            Status::Code::INTERNAL, "failed to allocate buffer for output '" +
                                        cache_output.name_ + "'");
      }

      // TODO: No out of scope issue here? With underlying
      //       allocated_buffer_ == buffer ?
      std::memcpy(buffer, cache_output.buffer_, cache_output.buffer_size_);

      // TODO: Add field to InferenceResponse to indicate this was from cache
      // response.cached = true;
    }
  }

  return Status::Success;
}

Status
RequestResponseCache::HashInputBuffers(
    const InferenceRequest::Input* input, size_t* seed)
{
  // Iterate over each data buffer in input in case of non-contiguous memory
  for (size_t idx = 0; idx < input->DataBufferCount(); ++idx) {
    const void* src_buffer;
    size_t src_byte_size;
    TRITONSERVER_MemoryType src_memory_type;
    int64_t src_memory_type_id;

    RETURN_IF_ERROR(input->DataBuffer(
        idx, &src_buffer, &src_byte_size, &src_memory_type,
        &src_memory_type_id));

    // TODO: Handle other memory types
    if (src_memory_type != TRITONSERVER_MEMORY_CPU &&
        src_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
      return Status(
          Status::Code::INTERNAL,
          "Only input buffers in CPU memory are allowed in cache currently");
    }

    // Add each byte of input buffer chunk to hash
    const unsigned char* tmp = static_cast<const unsigned char*>(src_buffer);
    for (uint64_t byte = 0; byte < src_byte_size; byte++) {
      boost::hash_combine(*seed, tmp[byte]);
    }
  }

  return Status::Success;
}


Status
RequestResponseCache::HashInputs(const InferenceRequest& request, size_t* seed)
{
  const auto& inputs = request.ImmutableInputs();
  // Convert inputs to ordered map for consistency in hashing
  // inputs sorted by key (input) name
  std::map<std::string, InferenceRequest::Input*> ordered_inputs(
      inputs.begin(), inputs.end());
  for (const auto& input : ordered_inputs) {
    // Add input name to hash
    boost::hash_combine(*seed, input.second->Name());
    // Fetch input buffer for hashing raw data
    RETURN_IF_ERROR(HashInputBuffers(input.second, seed));
  }

  return Status::Success;
}


Status
RequestResponseCache::Hash(const InferenceRequest& request, uint64_t* key)
{
  std::size_t seed = 0;
  // Add request model name to hash
  boost::hash_combine(seed, request.ModelName());
  // Add request model version to hash
  boost::hash_combine(seed, request.ActualModelVersion());
  RETURN_IF_ERROR(HashInputs(request, &seed));
  *key = static_cast<uint64_t>(seed);
  return Status::Success;
}

void
RequestResponseCache::CaptureLookupEndTime(InferenceRequest* request)
{
  // Capture end lookup latency on hit and update total latency
  request->CaptureCacheLookupEndNs();
  total_lookup_latency_ns_ +=
      (request->CacheLookupEndNs() - request->CacheLookupStartNs());
}

void
RequestResponseCache::CaptureInsertionEndTime(InferenceRequest* request)
{
  // Capture end insertion latency on miss and update total latency
  request->CaptureCacheInsertionEndNs();
  total_insertion_latency_ns_ +=
      (request->CacheInsertionEndNs() - request->CacheInsertionStartNs());
}

}}  // namespace nvidia::inferenceserver
