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
#include "numa_utils.h"

#ifndef _WIN32
#include <numa.h>
#include <numaif.h>
#endif
#include "src/core/logging.h"

namespace {
std::string
VectorToString(const std::vector<int>& vec)
{
  std::string str("[");
  for (const auto& element : vec) {
    str += std::to_string(element);
    str += ",";
  }

  str += "]";
  return str;
}

}  // namespace

namespace nvidia { namespace inferenceserver {

// NUMA setting will be ignored on Windows platform
#ifdef _WIN32
Status
SetNumaConfigOnThread(
    const NumaConfig& numa_config,
    const TRITONSERVER_InstanceGroupKind device_kind, const int numa_id)
{
  return Status::Success;
}

Status
SetNumaMemoryPolicy(
    const NumaConfig& numa_config,
    const TRITONSERVER_InstanceGroupKind device_kind, const int numa_id)
{
  return Status::Success;
}

Status
ResetNumaMemoryPolicy()
{
  return Status::Success;
}

Status
SetNumaThreadAffinity(
    std::thread::native_handle_type thread, const NumaConfig& numa_config,
    const TRITONSERVER_InstanceGroupKind device_kind, const int numa_id)
{
  return Status::Success;
}
#else
// Use variable to make sure no NUMA related function is actually called
// if Triton is not running with NUMA awareness. i.e. Extra docker permission
// is needed to call the NUMA functions and this ensures backward compatibility.
thread_local bool numa_set = false;

Status
SetNumaConfigOnThread(
    const NumaConfig& numa_config,
    const TRITONSERVER_InstanceGroupKind device_kind, const int numa_id)
{
  // Set thread affinity
  RETURN_IF_ERROR(
      SetNumaThreadAffinity(pthread_self(), numa_config, device_kind, numa_id));

  // Set memory policy
  RETURN_IF_ERROR(SetNumaMemoryPolicy(numa_config, device_kind, numa_id));

  return Status::Success;
}

Status
SetNumaMemoryPolicy(
    const NumaConfig& numa_config,
    const TRITONSERVER_InstanceGroupKind device_kind, const int numa_id)
{
  const auto it = numa_config.find(std::make_pair(device_kind, numa_id));
  if (it != numa_config.end()) {
    LOG_VERBOSE(1) << "Device thread ("
                   << TRITONSERVER_InstanceGroupKindString(device_kind) << "_"
                   << numa_id << ") is binding to NUMA node "
                   << it->second.first
                   << ". Max NUMA node count: " << numa_max_node();
    numa_set = true;
    unsigned long node_mask = 1UL << it->second.first;
    if (set_mempolicy(MPOL_BIND, &node_mask, numa_max_node() + 1) != 0) {
      return Status(
          Status::Code::INTERNAL,
          std::string("Unable to set NUMA memory policy: ") + strerror(errno));
    }
  }
  return Status::Success;
}

Status
ResetNumaMemoryPolicy()
{
  if (numa_set && (set_mempolicy(MPOL_DEFAULT, nullptr, 0) != 0)) {
    return Status(
        Status::Code::INTERNAL,
        std::string("Unable to reset NUMA memory policy: ") + strerror(errno));
  }
  numa_set = false;
  return Status::Success;
}

Status
SetNumaThreadAffinity(
    std::thread::native_handle_type thread, const NumaConfig& numa_config,
    const TRITONSERVER_InstanceGroupKind device_kind, const int numa_id)
{
  const auto it = numa_config.find(std::make_pair(device_kind, numa_id));
  if (it != numa_config.end()) {
    LOG_VERBOSE(1) << "Device thread ("
                   << TRITONSERVER_InstanceGroupKindString(device_kind) << "_"
                   << numa_id << ") is binding to one of the CPUs: "
                   << VectorToString(it->second.second);
    numa_set = true;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int cpu : it->second.second) {
      CPU_SET(cpu, &cpuset);
    }
    if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset) != 0) {
      return Status(
          Status::Code::INTERNAL,
          std::string("Unable to set NUMA thread affinity: ") +
              strerror(errno));
    }
  }
  return Status::Success;
}
#endif

}}  // namespace nvidia::inferenceserver
