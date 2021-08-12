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

#include "src/core/status.h"
#include "src/core/infer_request.h"
#include "src/core/infer_response.h"

#include <boost/functional/hash.hpp>
#include <boost/interprocess/managed_external_buffer.hpp>

namespace nvidia { namespace inferenceserver {

struct Output {
    // Output tensor data buffer
    void* buffer;
    // Size of "buffer" above
    uint64_t size;
    // Name of the output
    std::string name;
    // Datatype of the output
    inference::DataType dtype;
};

struct CacheEntry {
    explicit CacheEntry() {}
    // Point to key in LRU list for maintaining LRU order
    std::list<uint64_t>::iterator lru_iter;
    // each output buffer = managed_buffer.allocate(size, ...)
    std::vector<Output> outputs;
    // sum of output sizes
    uint64_t size;
};

class RequestResponseCache {
    public:
        RequestResponseCache(const uint64_t cache_size);
        ~RequestResponseCache();
        // Hash inference request to access cache
        uint64_t Hash(const InferenceRequest& request);
        // Lookup 'key' in cache and return the inference response in 'ptr' on cache hit or nullptr on cache miss
        // Return Status object indicating success or failure.
        Status Lookup(const uint64_t key, InferenceResponse** ptr);
        // Insert response into cache, evict entries to make space if necessary
        // Return Status object indicating success or failure.
        Status Insert(const uint64_t key, const InferenceResponse& response);
        // Evict entry from cache based on policy
        // Return Status object indicating success or failure.
        Status Evict();

    private:
        // Cache buffer
        void* buffer_;
        // Managed buffer
        boost::interprocess::managed_external_buffer managed_buffer_;
        // Total size of cache, will evict if a new item will exceed the total/available size
        uint64_t total_size_;
        // Remaining size left in cache, updated on insertion and eviction
        uint64_t available_size_;
        // key -> CacheEntry containing values and list iterator for LRU management
        std::unordered_map<uint64_t, CacheEntry> cache_;
        // list of keys sorted from most to least recently used
        std::list<uint64_t> lru_;

        // Update LRU ordering on lookup
        void UpdateLRU(std::unordered_map<uint64_t, CacheEntry>::iterator&);
        // Build CacheEntry from InferenceResponse
        Status BuildCacheEntry(CacheEntry& entry, const InferenceResponse& response);
        // Build InferenceResponse from CacheEntry
        Status BuildInferenceResponse(const CacheEntry& entry, InferenceResponse** response);
};

}}  // namespace nvidia::inferenceserver
