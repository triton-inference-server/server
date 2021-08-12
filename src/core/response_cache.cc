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
    // TODO: Error check creation of managed_buffer_ ?
}

RequestResponseCache::~RequestResponseCache()
{
    // TODO: Free each managed buffer allocated to cache entries,
    //       or is it enough to free full buffer?
    for (auto& iter : cache_) {
        auto& entry = iter.second;
        for (auto& output : entry.outputs) {
            managed_buffer_.deallocate(output.buffer);
        }
    }

    // Free total cache buffer
    if (buffer_ != nullptr) {
        free(buffer_);
    }
}

uint64_t RequestResponseCache::Hash(const InferenceRequest& request) {
    std::size_t seed = 0;
    // TODO: Linker ERROR
    //boost::hash_combine(seed, request.ModelName()); 
    // TODO: RequestedModelVersion or ActualModelVersion ?
    boost::hash_combine(seed, request.RequestedModelVersion());

    // TODO: OriginalInputs, OverrideInputs, ImmutableInputs ?
    const auto& inputs = request.OriginalInputs();
    for (const auto& input : inputs) {
        // Add input name to hash
        boost::hash_combine(seed, input.second.Name());
        // Add input data to hash
        // TODO: Consider hashing pointer as-is vs. actual raw input bytes
        // Scenario: What about the same input values stored in a different memory address?
        boost::hash_combine(seed, input.second.Data());
    }

    uint64_t key = static_cast<uint64_t>(seed);
    return key;
}

Status RequestResponseCache::Lookup(const uint64_t key, InferenceResponse** ptr) {
    auto iter = cache_.find(key);
    if (iter == cache_.end()) {
        return Status(
            Status::Code::INTERNAL, "key not found in cache"
        );
    }
    // Update this key to front of LRU list
    UpdateLRU(iter);
    // Populate passed-in "ptr" from cache entry
    auto entry = iter->second;
    // Build InferenceResponse from CacheEntry
    auto status = BuildInferenceResponse(entry, ptr);
    if (!status.IsOk()) {
        return status;
    }

    return Status::Success;
}

Status RequestResponseCache::Insert(const uint64_t key, const InferenceResponse& response) {
    // Exit early if key already exists in cache
    auto iter = cache_.find(key);
    if (iter != cache_.end()) {
        return Status(
            Status::Code::INTERNAL, "key already exists in cache"
        );
    }
    // Construct cache entry from response
    auto entry = CacheEntry();
    auto status = BuildCacheEntry(entry, response);
    if (!status.IsOk()) {
        return status;
    }

    // If cache doesn't have room for new entry, evict until enough size is available
    while (entry.size > available_size_) {
        auto status = Evict();
        // If evict fails for some reason, exit with its failure status
        if (!status.IsOk()) {
            return status;
        }
    }
    // Insert entry into cache
    auto cache_pair = cache_.insert({key, entry});
    bool ok = cache_pair.second;
    // Exit early if cache insertion failed
    if (!ok) {
        return Status(
            Status::Code::INTERNAL, "Cache insertion failed"
        );
    }
    // Update LRU with new cache entry
    auto cache_iter = cache_pair.first;
    UpdateLRU(cache_iter);
    // Update available cache size
    available_size_ -= entry.size;
    // Error checking cache size management
    if (available_size_ > total_size_) {
        return Status(
            Status::Code::INTERNAL,
            "Available size exceeded total size: this indicates a bug in the code"
        );
    }

    return Status::Success;
}

Status RequestResponseCache::BuildCacheEntry(CacheEntry& entry, const InferenceResponse& response) {
    // Build cache entry data from response outputs
    //for (const InferenceResponse::Output& response_output : response.Outputs()) {
    for (const auto& response_output : response.Outputs()) {
        auto cache_output = Output();

        // Fetch output buffer details
        const void* response_buffer = nullptr;
        size_t response_byte_size = 0;
        /*
        TRITONSERVER_MemoryType response_memory_type;
        int64_t response_memory_type_id = 0;
        void* userp = nullptr;
        // TODO: How to differently handle different memory types?
        //       GPU vs. CPU memory, etc.
        // TODO: Linker error on calling DataBuffer() here
        Status status = response_output.DataBuffer(
            &response_buffer, &response_byte_size, &response_memory_type,
            &response_memory_type_id, &userp);
        */
        // TODO: Temp WAR
        auto status = Status::Success;

        // Exit early if we fail to get output buffer from response
        if (!status.IsOk()) {
            return Status(
                Status::Code::INTERNAL, "Failed to get output buffer for " + response_output.Name()
            );
        }

        // Exit early if response buffer from output is invalid
        if (response_buffer == nullptr) {
            return Status(
                Status::Code::INTERNAL, "Response buffer from output was nullptr"
            );
        }

        // Exit early if cache entry will be larger than total cache size
        if (response_byte_size > total_size_) {
            return Status(
                Status::Code::INTERNAL, "Cache entry is larger than total cache size"
            );
        } 

        // Set output metadata
        cache_output.name = response_output.Name();
        cache_output.dtype = response_output.DType();
        cache_output.size = static_cast<uint64_t>(response_byte_size);

        // Allocate buffer for response output in cache entry
        cache_output.buffer = managed_buffer_.allocate(response_byte_size, std::nothrow_t{});
        // Exit early if we fail to allocate from managed buffer
        if (cache_output.buffer == nullptr) {
            return Status(
                Status::Code::INTERNAL, "Failed to allocate buffer from managed buffer"
            );
        }

        // Copy data from response buffer to cache entry output buffer
        // TODO: How to differently handle different memory types?
        //       GPU vs. CPU memory, etc.
        std::memcpy(cache_output.buffer, response_buffer, response_byte_size);
        // Sum up output sizes for total cache entry size
        entry.size += cache_output.size;
        // Add each output to cache entry
        entry.outputs.push_back(cache_output);
    }

    return Status::Success;
}


Status RequestResponseCache::BuildInferenceResponse(const CacheEntry& entry, InferenceResponse** response) {
    // TODO
    return Status(
        Status::Code::INTERNAL, "NotImplementedError"
    );
}

// LRU
Status RequestResponseCache::Evict() {
    // Least recently used key in back of LRU list
    uint64_t lru_key = lru_.back();
    // Find cache entry for least recently used key
    auto iter = cache_.find(lru_key);
    // Error check if key isn't in cache, but this shouldn't happen in evict
    // and probably indicates a bug
    if (iter == cache_.end()) {
        return Status(
            Status::Code::INTERNAL,
            "key not found in cache during eviction: this indicates a bug in the code"
        );
    }
    // Get size of cache entry being evicted to update available size
    auto entry = iter->second;
    // Free managed memory used in cache entry's outputs
    for (auto& output : entry.outputs) {
        managed_buffer_.deallocate(output.buffer);
    }

    // Remove LRU entry from cache
    cache_.erase(lru_key);
    // Remove LRU key from LRU list
    lru_.pop_back();
    // Update available cache size
    available_size_ += entry.size;
    // Error checking cache size management
    if (available_size_ > total_size_) {
        return Status(
            Status::Code::INTERNAL,
            "Available size exceeded total size: this indicates a bug in the code"
        );
    }
    return Status::Success;
}

// LRU
void RequestResponseCache::UpdateLRU(std::unordered_map<uint64_t, CacheEntry>::iterator& cache_iter) {
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
    cache_entry.lru_iter = lru_.begin();
}

}}  // namespace nvidia::inferenceserver
