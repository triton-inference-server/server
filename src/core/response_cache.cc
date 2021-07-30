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
    cache_ = boost::interprocess::managed_external_buffer(
        boost::interprocess::create_only_t{}, this->buffer, size);
    // Exit early if managed buffer allocation failed
    if (cache_ == nullptr) {
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

// TODO: Doc describes returning handle/ptr, how does this work if ptr evicted
//    after it's returned but before it's used/de-referenced?
InferenceResponse Lookup(const uint64_t key, const InferenceRequest& request) {
    // TODO
}

Status Insert(const uint64_t key, const InferenceResponse& response) {
    // TODO
}

Status Evict(const uint64_t size) {
    // TODO: Use enum/typedef over strings for cache_policy_
    switch(cache_policy_) {
        case: "LRU":
            return EvictLRU(size);
        default:
            return Status(Status::Code::INTERNAL, "Unsupported Cache Policy");

    }

}

Status EvictLRU(const uint64_t size) {
    auto status = Status::Success;
    // TODO: Evict entries with LRU policy until "size" bytes is available
    return status;
}


}}  // namespace nvidia::inferenceserver
