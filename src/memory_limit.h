// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <string>

#include "common.h"

namespace triton { namespace server {

// Where a detected memory limit came from. Used in the startup log line.
enum class MemoryLimitSource { CGROUP_V2, CGROUP_V1, PHYSICAL_RAM };

struct MemoryLimit {
  // Limit in bytes. 0 only if no cgroup limit was found and physical RAM
  // could not be determined either.
  uint64_t bytes{0};
  MemoryLimitSource source{MemoryLimitSource::PHYSICAL_RAM};
  // The cgroup file the limit was read from, or why the physical RAM
  // fallback was used.
  std::string detail;
  // True when cgroup files were found but the cgroup of this process could
  // not be resolved, so the physical RAM fallback may be too high.
  bool detection_failed{false};
};

// Returns the memory limit that applies to this process: the smallest cgroup
// memory limit on the path from this process's own cgroup up to the top of
// the cgroup mount (cgroup v1 or v2), or physical RAM if no limit is set, it
// cannot be read, or the platform has no cgroups. The cgroup path is mapped
// through /proc/self/mountinfo, so this works inside a container (private or
// host cgroup namespace) and outside one. Never fails: a read problem only
// causes the physical RAM fallback.
MemoryLimit DetectMemoryLimit();

// Same as DetectMemoryLimit(), for tests. 'root' is prepended to every path
// that is read, so a fake /proc and /sys tree can be used, and physical RAM is
// passed in.
MemoryLimit DetectMemoryLimit(
    const std::string& root, uint64_t physical_ram_bytes);

// Total physical RAM in bytes, or 0 if it cannot be determined.
uint64_t PhysicalRamBytes();

// The HTTP parse memory budget shared by all HTTP endpoints.
struct ParseMemoryBudget {
  // False when the budget is turned off.
  bool enabled{true};
  uint64_t bytes{0};
  // How the budget was picked, for the startup log line.
  std::string description;
  // True when the startup log line should be a warning.
  bool warn{false};
};

// Works out the parse memory budget from the value of
// --http-parse-memory-budget: a byte count, 0 to turn the budget off, or
// HTTP_PARSE_MEMORY_BUDGET_AUTO to use HTTP_PARSE_MEMORY_BUDGET_PERCENT of
// 'limit'.
ParseMemoryBudget ResolveParseMemoryBudget(
    int64_t flag_bytes, const MemoryLimit& limit);

// Same as above, with the limit from DetectMemoryLimit().
ParseMemoryBudget ResolveParseMemoryBudget(int64_t flag_bytes);

}}  // namespace triton::server
