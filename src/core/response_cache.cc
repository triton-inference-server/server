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
    // Exit early if managed buffer allocation failed
    if (managed_buffer_ == nullptr) {
        throw("failed to create managed external buffer");
    }
}

RequestResponseCache::~RequestResponseCache()
{
    if (buffer_ != nullptr) {
        free(buffer_);
    }
}

uint64_t Hash(const InferenceRequest& request) {
    // TODO: OriginalInputs, OverrideInputs, ImmutableInputs ?
    auto inputs = request->OriginalInputs();
    // Create hash function
    std::hash<vector<std::string>> hash_function;
    // Setup vector of strings for hashing from request parameters
    std::vector<std::string> hash_inputs;
    hash_inputs.push_back(request->Name());
    hash_inputs.push_back(request->ModelName());
    // TODO: RequestedModelVersion or ActualModelVersion ?
    hash_inputs.push_back(request->RequestedModelVersion());
    // Setup vector of input tensor values for hashing
    for (auto const& input : inputs) {
        hash_inputs.push_back(input->Name());
        // TODO: Example of accessing/iterating over input data?
        // TODO: Can we cast any type to string here? Or hash based on defined dtype?
        //       Can we hash just the raw bits/bytes for any input buffer, or is this too collision prone?
    }

    // Hash together the various request fields
    uint64_t key = static_cast<uint64_t>(hash_function(hash_inputs));
    return key;
}

Status Lookup(const uint64_t key, InferenceResponse** ptr) {
    auto iter = cache_.find(key);
    if (iter == cache_.end()) {
        return Status(
            Status::Code::INTERNAL, "key not found in cache"
        );
    }
    // Update this key to front of LRU list
    Update(iter);
    // Populate passed-in "ptr" from cache entry
    auto entry = iter->second;
    // TODO: Copy contents from CacheEntry ptr to passed-in ptr
    // *ptr = ...
    if (*ptr == nullptr) {
        return Status(
            Status::Code::INTERNAL, "InferenceResponse ptr in cache was invalid nullptr"
        );
    }
    return Status::Success;
}

Status Insert(const uint64_t key, const InferenceResponse& response) {
    // Exit early if key already exists in cache
    auto iter = cache_.find(key);
    if (iter != cache_.end()) {
        return Status(
            Status::Code::INTERNAL, "key already exists in cache"
        );
    }
    // TODO: Construct cache entry from response
    auto entry = CacheEntry();
    // TODO: update cache entry size
    // entry.size = ...
    // TODO: request buffer from managed_buffer
    // *ptr = managed_buffer_.allocate(entry.size, ...)
    // cache.ptr = ptr

    // If cache entry is larger than total cache size, exit with failure
    if (entry.size > total_size_) {
        return Status(
            Status::Code::INTERNAL, "Cache entry is larger than total cache size"
        );
    } 
    // If cache doesn't have room for new entry, evict until enough size is available
    while (entry.size > available_size_) {
        auto status = Evict();
        // If evict fails for some reason, exit with its failure status
        if (!status.IsOk()) {
            return status;
        }
    }
    // Now that we have room for new entry, update LRU and insert it in cache
    lru_.push_front(key);
    cache_.insert({key, entry});
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

// LRU
Status Evict() {
    auto lru_key = lru_.back();
    auto iter = cache_.find(lru_key);
    // Error check if key isn't in cache, but this shouldn't happen in evict
    // and probably indicates a bug
    if (iter == cache_.end()) {
        return Status(
            Status::Code::INTERNAL,
            "key not found in cache during eviction: this indicates a bug in the code"
        );
    }
    // TODO: free managed memory used in cache entry as well?
    // managed_buffer_.deallocate(...)
    // Get size of cache entry being evicted to update available size
    auto entry = iter->second;
    uint64_t entry_size = entry.size;
    // Remove LRU entry from cache
    cache_.erase(lru_key);
    // Remove LRU key from LRU list
    lru_.pop_back();
    // Update available cache size
    available_size_ += entry_size;
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
void Update(std::unordered_map<uint64_t, CacheEntry>::iterator& cache_iter) {
    // Remove key from LRU list at it's current location
    lru_.erase(cache_iter->second.lru_iter);
    // Add key to front of LRU list since it's most recently used
    lru_.push_front(cache_iter->first);
    // Set CacheEntry iterator to new LRU key location
    cache_iter->second.lru_iter = lru_.begin();
}

}}  // namespace nvidia::inferenceserver
