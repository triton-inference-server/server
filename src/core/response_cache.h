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

#include <boost/interprocess/managed_external_buffer.hpp>
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class RequestResponseCache {
    RequestResponseCache(const uint64_t cache_size);
    ~RequestResponseCache();
    // Hash inference request to access cache
    uint64_t Hash(const InferenceRequest& request);
    // Lookup key in cache, request used for strict exact matching on collisions
    // Return InferenceResponse if found in cache
    // Q: Doc describes returning handle/ptr, how does this work if ptr evicted
    //    after it's returned but before it's used/de-referenced?
    InferenceResponse Lookup(const uint64_t key, const InferenceRequest& request);
    // Insert response into cache, evict entries to make space if necessary
    // Return Status object indicating success or failure.
    Status Insert(const uint64_t key, const InferenceResponse& response);
    // Eviction handler to call corresponding function based on policy
    Status Evict();
    // Eviction function for LRU policy
    Status EvictLRU();

    // Cache buffer
    char* buffer_;
    // Managed buffer
    boost::interprocess::managed_external_buffer cache_;
    // Eviction policy
    // TODO: Enum / typedef this
    std::string eviction_policy_ = "LRU";

};

}}  // namespace nvidia::inferenceserver
