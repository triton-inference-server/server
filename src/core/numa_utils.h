// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <map>
#include <thread>
#include <vector>
#include "src/core/model_config.h"
#include "src/core/status.h"
#include "src/core/tritonserver_apis.h"

namespace nvidia { namespace inferenceserver {

// Helper function to set memory policy and thread affinity on current thread
Status SetNumaConfigOnThread(const HostPolicyCmdlineConfig& host_policy);

// Restrict the memory allocation to specific NUMA node.
Status SetNumaMemoryPolicy(const HostPolicyCmdlineConfig& host_policy);

// Retrieve the node mask used to set memory policy for the current thread
Status GetNumaMemoryPolicyNodeMask(unsigned long* node_mask);

// Reset the memory allocation setting.
Status ResetNumaMemoryPolicy();

// Set a thread affinity to be on specific cpus.
Status SetNumaThreadAffinity(
    std::thread::native_handle_type thread,
    const HostPolicyCmdlineConfig& host_policy);


}}  // namespace nvidia::inferenceserver
