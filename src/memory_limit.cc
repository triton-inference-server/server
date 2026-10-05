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

#include "memory_limit.h"

#include <fstream>
#include <sstream>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace triton { namespace server {

namespace {

// cgroup v1 reports "no limit" as a huge page-aligned number. Anything at or
// above this is treated as no limit even when physical RAM is unknown.
constexpr uint64_t kNoLimitThreshold = 1ULL << 60;

bool
ReadFile(const std::string& path, std::string* contents)
{
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }
  std::stringstream ss;
  ss << file.rdbuf();
  *contents = ss.str();
  return true;
}

bool
FileExists(const std::string& path)
{
  std::ifstream file(path);
  return file.is_open();
}

std::string
Trim(const std::string& s)
{
  const char* whitespace = " \t\r\n";
  const size_t begin = s.find_first_not_of(whitespace);
  if (begin == std::string::npos) {
    return "";
  }
  const size_t end = s.find_last_not_of(whitespace);
  return s.substr(begin, end - begin + 1);
}

std::vector<std::string>
Split(const std::string& s, char delimiter)
{
  std::vector<std::string> parts;
  std::string part;
  std::istringstream in(s);
  while (std::getline(in, part, delimiter)) {
    parts.push_back(part);
  }
  return parts;
}

// True if 'path' is 'root' or below it. Paths with a ".." component are
// rejected, since they point outside our cgroup namespace.
bool
IsPathUnder(const std::string& path, const std::string& root)
{
  if (path.empty() || path[0] != '/') {
    return false;
  }
  for (const auto& component : Split(path, '/')) {
    if (component == "..") {
      return false;
    }
  }
  if (root == "/") {
    return true;
  }
  return (path.compare(0, root.size(), root) == 0) &&
         ((path.size() == root.size()) || (path[root.size()] == '/'));
}

// This process's cgroup paths, from /proc/self/cgroup. Each line is
// "hierarchy-ID:controller-list:cgroup-path".
struct ProcCgroup {
  bool has_v2{false};
  std::string v2_path;  // the "0::<path>" line
  bool has_v1_memory{false};
  std::string v1_memory_path;  // the line whose controllers include "memory"
};

ProcCgroup
ParseProcCgroup(const std::string& contents)
{
  ProcCgroup result;
  std::istringstream in(contents);
  std::string line;
  while (std::getline(in, line)) {
    line = Trim(line);
    const size_t first = line.find(':');
    if (first == std::string::npos) {
      continue;
    }
    const size_t second = line.find(':', first + 1);
    if (second == std::string::npos) {
      continue;
    }
    const std::string id = line.substr(0, first);
    const std::string controllers = line.substr(first + 1, second - first - 1);
    const std::string path = line.substr(second + 1);
    if ((id == "0") && controllers.empty()) {
      result.has_v2 = true;
      result.v2_path = path;
      continue;
    }
    for (const auto& controller : Split(controllers, ',')) {
      if (controller == "memory") {
        result.has_v1_memory = true;
        result.v1_memory_path = path;
        break;
      }
    }
  }
  return result;
}

struct CgroupMount {
  std::string root;         // mountinfo field 4: root of the mount
  std::string mount_point;  // mountinfo field 5: where it is mounted
};

// Finds the cgroup mount that contains 'cgroup_path'. For v1 only mounts that
// carry the memory controller are considered. Each mountinfo line is
// "id parent major:minor root mount-point options [optional...] - fstype
// source super-options".
bool
FindMount(
    const std::string& mountinfo, bool v2, const std::string& cgroup_path,
    CgroupMount* mount)
{
  std::istringstream in(mountinfo);
  std::string line;
  while (std::getline(in, line)) {
    const size_t separator = line.find(" - ");
    if (separator == std::string::npos) {
      continue;
    }
    const std::vector<std::string> fields =
        Split(Trim(line.substr(0, separator)), ' ');
    const std::vector<std::string> fs_fields =
        Split(Trim(line.substr(separator + 3)), ' ');
    if ((fields.size() < 5) || fs_fields.empty()) {
      continue;
    }
    const std::string& fstype = fs_fields[0];
    if (v2) {
      if (fstype != "cgroup2") {
        continue;
      }
    } else {
      if ((fstype != "cgroup") || (fs_fields.size() < 3)) {
        continue;
      }
      bool has_memory = false;
      for (const auto& option : Split(fs_fields[2], ',')) {
        if (option == "memory") {
          has_memory = true;
          break;
        }
      }
      if (!has_memory) {
        continue;
      }
    }
    if (IsPathUnder(cgroup_path, fields[3])) {
      mount->root = fields[3];
      mount->mount_point = fields[4];
      return true;
    }
  }
  return false;
}

// Parses one limit file. Returns false for "max", for values that mean no
// limit (at or above physical RAM, or the v1 "no limit" number), and for
// anything that cannot be parsed.
bool
ParseLimit(const std::string& contents, uint64_t physical_ram, uint64_t* bytes)
{
  const std::string value = Trim(contents);
  if (value.empty() || (value == "max")) {
    return false;
  }
  for (const char c : value) {
    if ((c < '0') || (c > '9')) {
      return false;
    }
  }
  uint64_t limit = 0;
  try {
    limit = std::stoull(value);
  }
  catch (...) {
    return false;
  }
  if ((limit == 0) || (limit >= kNoLimitThreshold) ||
      ((physical_ram > 0) && (limit >= physical_ram))) {
    return false;
  }
  *bytes = limit;
  return true;
}

}  // namespace

uint64_t
PhysicalRamBytes()
{
#ifdef _WIN32
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  if (GlobalMemoryStatusEx(&status)) {
    return status.ullTotalPhys;
  }
  return 0;
#else
  const long pages = sysconf(_SC_PHYS_PAGES);
  const long page_size = sysconf(_SC_PAGE_SIZE);
  if ((pages <= 0) || (page_size <= 0)) {
    return 0;
  }
  return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
#endif
}

MemoryLimit
DetectMemoryLimit(const std::string& root, uint64_t physical_ram_bytes)
{
  MemoryLimit fallback;
  fallback.bytes = physical_ram_bytes;
  fallback.source = MemoryLimitSource::PHYSICAL_RAM;

  std::string proc_cgroup, mountinfo;
  if (!ReadFile(root + "/proc/self/cgroup", &proc_cgroup) ||
      !ReadFile(root + "/proc/self/mountinfo", &mountinfo)) {
    fallback.detail = "no cgroup information";
    return fallback;
  }

  // Prefer cgroup v1 when this process is in a v1 memory hierarchy: that is
  // where the memory controller lives on v1 and on hybrid hosts. Otherwise
  // use the cgroup v2 hierarchy.
  const ProcCgroup self = ParseProcCgroup(proc_cgroup);
  bool v2 = false;
  std::string cgroup_path;
  CgroupMount mount;
  if (self.has_v1_memory &&
      FindMount(mountinfo, false /* v2 */, self.v1_memory_path, &mount)) {
    cgroup_path = self.v1_memory_path;
  } else if (
      self.has_v2 &&
      FindMount(mountinfo, true /* v2 */, self.v2_path, &mount)) {
    v2 = true;
    cgroup_path = self.v2_path;
  } else {
    fallback.detail = "no cgroup memory mount found for this process";
    return fallback;
  }

  // Map our cgroup path onto the mount: drop the mount root, then join the
  // rest to the mount point. On cgroup v1 the mount root is often our own
  // cgroup, so a plain join would point at the wrong directory.
  std::string relative =
      (mount.root == "/") ? cgroup_path : cgroup_path.substr(mount.root.size());
  if (relative == "/") {
    relative.clear();
  }
  const std::string mount_dir = root + mount.mount_point;

  // Our own cgroup directory must exist. If it does not, the mapping failed;
  // use the fallback rather than report a wrong limit.
  if (!FileExists(mount_dir + relative + "/cgroup.procs")) {
    fallback.detail =
        "cgroup directory not found: " + mount.mount_point + relative;
    return fallback;
  }

  // Walk from our cgroup up to the top of the mount and take the smallest
  // limit. A missing limit file at a level means no limit at that level (for
  // example, the cgroup v2 root has no memory.max).
  const std::string limit_file = v2 ? "/memory.max" : "/memory.limit_in_bytes";
  uint64_t smallest = 0;
  std::string smallest_path;
  std::string dir = relative;
  while (true) {
    std::string contents;
    uint64_t limit = 0;
    if (ReadFile(mount_dir + dir + limit_file, &contents) &&
        ParseLimit(contents, physical_ram_bytes, &limit) &&
        ((smallest == 0) || (limit < smallest))) {
      smallest = limit;
      smallest_path = mount.mount_point + dir + limit_file;
    }
    if (dir.empty()) {
      break;
    }
    dir = dir.substr(0, dir.find_last_of('/'));
  }

  if (smallest == 0) {
    fallback.detail = "no cgroup memory limit set";
    return fallback;
  }

  MemoryLimit result;
  result.bytes = smallest;
  result.source =
      v2 ? MemoryLimitSource::CGROUP_V2 : MemoryLimitSource::CGROUP_V1;
  result.detail = smallest_path;
  return result;
}

MemoryLimit
DetectMemoryLimit()
{
  return DetectMemoryLimit("", PhysicalRamBytes());
}

}}  // namespace triton::server
