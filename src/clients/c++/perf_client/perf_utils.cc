// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <string>

#include "src/clients/c++/perf_client/perf_utils.h"

ProtocolType
ParseProtocol(const std::string& str)
{
  std::string protocol(str);
  std::transform(protocol.begin(), protocol.end(), protocol.begin(), ::tolower);
  if (protocol == "http") {
    return ProtocolType::HTTP;
  } else if (protocol == "grpc") {
    return ProtocolType::GRPC;
  }
  return ProtocolType::UNKNOWN;
}

nic::Error
ReadFile(const std::string& path, std::vector<char>* contents)
{
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in) {
    return nic::Error(
        ni::RequestStatusCode::INVALID_ARG,
        "failed to open file '" + path + "'");
  }

  in.seekg(0, std::ios::end);

  int file_size = in.tellg();
  if (file_size > 0) {
    contents->resize(file_size);
    in.seekg(0, std::ios::beg);
    in.read(&(*contents)[0], contents->size());
  }

  in.close();

  // If size is invalid, report after ifstream is closed
  if (file_size < 0) {
    return nic::Error(
        ni::RequestStatusCode::INVALID_ARG,
        "failed to get size for file '" + path + "'");
  } else if (file_size == 0) {
    return nic::Error(
        ni::RequestStatusCode::INVALID_ARG, "file '" + path + "' is empty");
  }

  return nic::Error::Success;
}

nic::Error
ReadTextFile(const std::string& path, std::vector<std::string>* contents)
{
  std::ifstream in(path);
  if (!in) {
    return nic::Error(
        ni::RequestStatusCode::INVALID_ARG,
        "failed to open file '" + path + "'");
  }

  std::string current_string;
  while (std::getline(in, current_string)) {
    contents->push_back(current_string);
  }
  in.close();

  if (contents->size() == 0) {
    return nic::Error(
        ni::RequestStatusCode::INVALID_ARG, "file '" + path + "' is empty");
  }
  return nic::Error::Success;
}

nic::Error
ReadTimeIntervalsFile(
    const std::string& path, std::vector<std::chrono::nanoseconds>* contents)
{
  std::ifstream in(path);
  if (!in) {
    return nic::Error(
        ni::RequestStatusCode::INVALID_ARG,
        "failed to open file '" + path + "'");
  }

  std::string current_string;
  while (std::getline(in, current_string)) {
    std::chrono::nanoseconds curent_time_interval_ns(
        std::stol(current_string) * 1000);
    contents->push_back(curent_time_interval_ns);
  }
  in.close();

  if (contents->size() == 0) {
    return nic::Error(
        ni::RequestStatusCode::INVALID_ARG, "file '" + path + "' is empty");
  }
  return nic::Error::Success;
}

bool
IsDirectory(const std::string& path)
{
  struct stat s;
  if (stat(path.c_str(), &s) == 0 && (s.st_mode & S_IFDIR)) {
    return true;
  } else {
    return false;
  }
}

std::string
GetRandomString(const int string_length)
{
  std::mt19937_64 gen{std::random_device()()};
  std::uniform_int_distribution<size_t> dist{0, character_set.length() - 1};
  std::string random_string;
  std::generate_n(std::back_inserter(random_string), string_length, [&] {
    return character_set[dist(gen)];
  });
  return random_string;
}

template <>
std::function<std::chrono::nanoseconds(std::mt19937&)>
ScheduleDistribution<Distribution::POISSON>(const double request_rate)
{
  std::exponential_distribution<> dist =
      std::exponential_distribution<>(request_rate);
  return [dist](std::mt19937& gen) mutable {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(dist(gen)));
  };
}

template <>
std::function<std::chrono::nanoseconds(std::mt19937&)>
ScheduleDistribution<Distribution::CONSTANT>(const double request_rate)
{
  std::chrono::nanoseconds period =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(1.0 / request_rate));
  return [period](std::mt19937& /*gen*/) { return period; };
}

nic::Error
CreateSharedMemoryRegion(std::string shm_key, size_t byte_size, int* shm_fd)
{
  // get shared memory region descriptor
  *shm_fd = shm_open(shm_key.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
  if (*shm_fd == -1) {
    return nic::Error(
        ni::RequestStatusCode::INVALID_ARG,
        "unable to get shared memory descriptor for shared-memory key '" +
            shm_key + "'");
  }
  // extend shared memory object as by default it's initialized with size 0
  int res = ftruncate(*shm_fd, byte_size);
  if (res == -1) {
    return nic::Error(
        ni::RequestStatusCode::INVALID_ARG,
        "unable to initialize shared-memory key '" + shm_key +
            "' to requested size: " + std::to_string(byte_size) + " bytes");
  }

  return nic::Error::Success;
}

nic::Error
MapSharedMemory(int shm_fd, size_t offset, size_t byte_size, void** shm_addr)
{
  // map shared memory to process address space
  *shm_addr =
      mmap(NULL, byte_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, offset);
  if (*shm_addr == MAP_FAILED) {
    return nic::Error(
        ni::RequestStatusCode::INVALID_ARG,
        "unable to process address space or shared-memory descriptor: " +
            std::to_string(shm_fd));
  }

  return nic::Error::Success;
}

nic::Error
UnlinkSharedMemoryRegion(std::string shm_key)
{
  int shm_fd = shm_unlink(shm_key.c_str());
  if (shm_fd == -1) {
    return nic::Error(
        ni::RequestStatusCode::INVALID_ARG,
        "unable to unlink shared memory for key '" + shm_key + "'");
  }

  return nic::Error::Success;
}

nic::Error
UnmapSharedMemory(void* shm_addr, size_t byte_size)
{
  int tmp_fd = munmap(shm_addr, byte_size);
  if (tmp_fd == -1) {
    return nic::Error(
        ni::RequestStatusCode::INVALID_ARG,
        "unable to munmap shared memory region");
  }

  return nic::Error::Success;
}
