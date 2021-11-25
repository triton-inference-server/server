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

#pragma once

#include <list>
#include <string>
#include <unordered_map>

#include "src/core/backend.h"
#include "src/core/infer_request.h"
#include "src/core/infer_response.h"
#include "src/core/status.h"

#include <boost/functional/hash.hpp>
#include <boost/interprocess/managed_external_buffer.hpp>

namespace nvidia { namespace inferenceserver {

// Assuming CPU memory only for now
struct Output {
  // Output tensor data buffer
  void* buffer_;
  // Size of "buffer" above
  uint64_t buffer_size_ = 0;
  // Name of the output
  std::string name_;
  // Datatype of the output
  inference::DataType dtype_;
  // Shape of the output
  std::vector<int64_t> shape_;
};

struct CacheEntry {
  explicit CacheEntry() {}
  // Point to key in LRU list for maintaining LRU order
  std::list<uint64_t>::iterator lru_iter_;
  // each output buffer = managed_buffer.allocate(size, ...)
  std::vector<Output> outputs_;
};

class RequestResponseCache {
 public:
  ~RequestResponseCache();
  // Create the request/response cache object
  static Status Create(
      uint64_t cache_size, std::unique_ptr<RequestResponseCache>* cache);
  // Hash inference request to access cache and store it in "key"
  // Return Status object indicating success or failure.
  Status Hash(const InferenceRequest& request, uint64_t* key);
  // Lookup 'key' in cache and return the inference response in 'ptr' on cache
  // hit or nullptr on cache miss Return Status object indicating success or
  // failure.
  Status Lookup(const uint64_t key, InferenceResponse* ptr, InferenceRequest* request);
  // Insert response into cache, evict entries to make space if necessary
  // Return Status object indicating success or failure.
  Status Insert(const uint64_t key, const InferenceResponse& response);
  // Evict entry from cache based on policy
  // Return Status object indicating success or failure.
  Status Evict();
  // Returns number of items in cache
  size_t NumEntries()
  {
    std::lock_guard<std::recursive_mutex> lk(cache_mtx_);
    return cache_.size();
  }
  // Returns number of items evicted in cache lifespan
  size_t NumEvictions()
  {
    std::lock_guard<std::recursive_mutex> lk(cache_mtx_);
    return num_evictions_;
  }
  // Returns number of lookups in cache lifespan, should sum to hits + misses
  size_t NumLookups()
  {
    std::lock_guard<std::recursive_mutex> lk(cache_mtx_);
    return num_lookups_;
  }
  // Returns number of cache hits in cache lifespan
  size_t NumHits()
  {
    std::lock_guard<std::recursive_mutex> lk(cache_mtx_);
    return num_hits_;
  }
  // Returns number of cache hits in cache lifespan
  size_t NumMisses()
  {
    std::lock_guard<std::recursive_mutex> lk(cache_mtx_);
    return num_misses_;
  }
  // Returns the total lookup latency (nanoseconds) of all lookups in cache
  // lifespan
  uint64_t TotalLookupLatencyNs()
  {
    std::lock_guard<std::recursive_mutex> lk(cache_mtx_);
    return total_lookup_latency_ns_;
  }
  // Returns total number of bytes allocated for cache
  size_t TotalBytes()
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_mtx_);
    return managed_buffer_.get_size();
  }
  // Returns number of free bytes in cache
  size_t FreeBytes()
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_mtx_);
    return managed_buffer_.get_free_memory();
  }
  // Returns number of bytes in use by cache
  size_t AllocatedBytes()
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_mtx_);
    return managed_buffer_.get_size() - managed_buffer_.get_free_memory();
  }
  // Returns fraction of bytes allocated over total cache size between [0, 1]
  double TotalUtilization()
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_mtx_);
    return static_cast<double>(AllocatedBytes()) /
           static_cast<double>(TotalBytes());
  }

 private:
  explicit RequestResponseCache(const uint64_t cache_size);
  // Update LRU ordering on lookup
  void UpdateLRU(std::unordered_map<uint64_t, CacheEntry>::iterator&);
  // Build CacheEntry from InferenceResponse
  Status BuildCacheEntry(const InferenceResponse& response, CacheEntry* entry);
  // Build InferenceResponse from CacheEntry
  Status BuildInferenceResponse(
      const CacheEntry& entry, InferenceResponse* response);
  // Helper function to hash data buffers used by "input"
  Status HashInputBuffers(const InferenceRequest::Input* input, size_t* seed);
  // Helper function to hash each input in "request"
  Status HashInputs(const InferenceRequest& request, size_t* seed);

  // Cache buffer
  void* buffer_;
  // Managed buffer
  boost::interprocess::managed_external_buffer managed_buffer_;
  // key -> CacheEntry containing values and list iterator for LRU management
  std::unordered_map<uint64_t, CacheEntry> cache_;
  // List of keys sorted from most to least recently used
  std::list<uint64_t> lru_;
  // Cache metrics
  size_t num_evictions_ = 0;
  size_t num_lookups_ = 0;
  size_t num_hits_ = 0;
  size_t num_misses_ = 0;
  uint64_t total_lookup_latency_ns_ = 0;
  // Mutex for buffer synchronization
  std::recursive_mutex buffer_mtx_;
  // Mutex for cache synchronization
  std::recursive_mutex cache_mtx_;
};

}}  // namespace nvidia::inferenceserver
