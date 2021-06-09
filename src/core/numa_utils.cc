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

namespace nvidia { namespace inferenceserver {

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

Status
ParseIntOption(const std::string& msg, const std::string& arg, int* value)
{
  try {
    *value = std::stoi(arg);
  }
  catch (const std::invalid_argument& ia) {
    return Status(
        Status::Code::INVALID_ARG,
        msg + ": Can't parse '" + arg + "' to integer");
  }
  return Status::Success;
}

}  // namespace

// NUMA setting will be ignored on Windows platform
#ifdef _WIN32
Status
SetNumaConfigOnThread(const HostPolicyCmdlineConfig& host_policy)
{
  return Status::Success;
}

Status
SetNumaMemoryPolicy(const HostPolicyCmdlineConfig& host_policy)
{
  return Status::Success;
}

Status
GetNumaMemoryPolicyNodeMask(unsigned long* node_mask)
{
  *node_mask = 0;
  return Status::Success;
}

Status
ResetNumaMemoryPolicy()
{
  return Status::Success;
}

Status
SetNumaThreadAffinity(
    std::thread::native_handle_type thread,
    const HostPolicyCmdlineConfig& host_policy)
{
  return Status::Success;
}
#else
// Use variable to make sure no NUMA related function is actually called
// if Triton is not running with NUMA awareness. i.e. Extra docker permission
// is needed to call the NUMA functions and this ensures backward compatibility.
thread_local bool numa_set = false;

Status
SetNumaConfigOnThread(const HostPolicyCmdlineConfig& host_policy)
{
  // Set thread affinity
  RETURN_IF_ERROR(SetNumaThreadAffinity(pthread_self(), host_policy));

  // Set memory policy
  RETURN_IF_ERROR(SetNumaMemoryPolicy(host_policy));

  return Status::Success;
}

Status
SetNumaMemoryPolicy(const HostPolicyCmdlineConfig& host_policy)
{
  const auto it = host_policy.find("numa-node");
  if (it != host_policy.end()) {
    int node_id;
    RETURN_IF_ERROR(
        ParseIntOption("Parsing 'numa-node' value", it->second, &node_id));
    LOG_VERBOSE(1) << "Thread is binding to NUMA node " << it->second
                   << ". Max NUMA node count: " << (numa_max_node() + 1);
    numa_set = true;
    unsigned long node_mask = 1UL << node_id;
    if (set_mempolicy(MPOL_BIND, &node_mask, (numa_max_node() + 1) + 1) != 0) {
      return Status(
          Status::Code::INTERNAL,
          std::string("Unable to set NUMA memory policy: ") + strerror(errno));
    }
  }
  return Status::Success;
}

Status
GetNumaMemoryPolicyNodeMask(unsigned long* node_mask)
{
  *node_mask = 0;
  int mode;
  if (numa_set &&
      get_mempolicy(&mode, node_mask, numa_max_node() + 1, NULL, 0) != 0) {
    return Status(
        Status::Code::INTERNAL,
        std::string("Unable to get NUMA node for current thread: ") +
            strerror(errno));
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
    std::thread::native_handle_type thread,
    const HostPolicyCmdlineConfig& host_policy)
{
  const auto it = host_policy.find("cpu-cores");
  if (it != host_policy.end()) {
    // Parse CPUs
    std::vector<int> cpus;
    {
      const auto& cpu_str = it->second;
      auto delim_cpus = cpu_str.find(",");
      int current_pos = 0;
      while (true) {
        auto delim_range = cpu_str.find("-", current_pos);
        if (delim_range == std::string::npos) {
          return Status(
              Status::Code::INVALID_ARG,
              std::string("host policy setting 'cpu-cores' format is "
                          "'<lower_cpu_core_id>-<upper_cpu_core_id>'. Got ") +
                  cpu_str.substr(
                      current_pos, ((delim_cpus == std::string::npos)
                                        ? (cpu_str.length() + 1)
                                        : delim_cpus) -
                                       current_pos));
        }
        int lower, upper;
        RETURN_IF_ERROR(ParseIntOption(
            "Parsing 'cpu-cores' value",
            cpu_str.substr(current_pos, delim_range - current_pos), &lower));
        RETURN_IF_ERROR(ParseIntOption(
            "Parsing 'cpu-cores' value",
            (delim_cpus == std::string::npos)
                ? cpu_str.substr(delim_range + 1)
                : cpu_str.substr(
                      delim_range + 1, delim_cpus - (delim_range + 1)),
            &upper));
        for (; lower <= upper; ++lower) {
          cpus.push_back(lower);
        }
        // break if the processed range is the last specified range
        if (delim_cpus != std::string::npos) {
          current_pos = delim_cpus + 1;
          delim_cpus = cpu_str.find(",", current_pos);
        } else {
          break;
        }
      }
    }

    LOG_VERBOSE(1) << "Thread is binding to one of the CPUs: "
                   << VectorToString(cpus);
    numa_set = true;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int cpu : cpus) {
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
