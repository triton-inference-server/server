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

// Keep-alive HTTP load generator for the identity model used by
// qa/L0_metrics/run_latency_histogram_load_test.sh. One thread per in-flight
// request, matching the concurrency from triton issue 9003.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

std::string g_host = "127.0.0.1";
int g_port = 8000;
std::string g_model = "identity";
int g_concurrency = 1000;
int g_seconds = 20;

std::atomic<bool> g_run{true};
std::atomic<uint64_t> g_success{0};
std::atomic<uint64_t> g_failure{0};

bool
SendAll(int fd, const std::string& data)
{
  size_t sent = 0;
  while (sent < data.size()) {
    const ssize_t n = ::send(fd, data.data() + sent, data.size() - sent, 0);
    if (n <= 0) {
      return false;
    }
    sent += static_cast<size_t>(n);
  }
  return true;
}

bool
ReadSome(int fd, std::string* buf)
{
  char tmp[4096];
  const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
  if (n <= 0) {
    return false;
  }
  buf->append(tmp, static_cast<size_t>(n));
  return true;
}

// Returns true when a full HTTP response was consumed and its status is 200.
bool
ReadResponse(int fd, std::string* buf)
{
  for (;;) {
    const auto header_end = buf->find("\r\n\r\n");
    if (header_end != std::string::npos) {
      const std::string headers = buf->substr(0, header_end);
      const auto status_end = headers.find("\r\n");
      const std::string status =
          headers.substr(0, status_end == std::string::npos ? headers.size()
                                                            : status_end);
      const bool ok = status.find(" 200 ") != std::string::npos;
      auto cl = headers.find("Content-Length:");
      if (cl == std::string::npos) {
        cl = headers.find("content-length:");
      }
      size_t body = 0;
      if (cl != std::string::npos) {
        body = static_cast<size_t>(std::strtoul(
            headers.c_str() + cl + std::strlen("Content-Length:"), nullptr, 10));
      }
      const size_t total = header_end + 4 + body;
      while (buf->size() < total) {
        if (!ReadSome(fd, buf)) {
          return false;
        }
      }
      buf->erase(0, total);
      return ok;
    }
    if (!ReadSome(fd, buf)) {
      return false;
    }
  }
}

int
Connect()
{
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return -1;
  }
  const int yes = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
  timeval tv{};
  tv.tv_sec = 5;
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(g_port));
  if (::inet_pton(AF_INET, g_host.c_str(), &addr.sin_addr) != 1) {
    ::close(fd);
    return -1;
  }
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    return -1;
  }
  return fd;
}

std::string
BuildRequest()
{
  const std::string body =
      "{\"inputs\":[{\"name\":\"INPUT0\",\"datatype\":\"FP32\",\"shape\":[1,16],"
      "\"data\":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}]}";
  std::ostringstream req;
  req << "POST /v2/models/" << g_model << "/infer HTTP/1.1\r\n"
      << "Host: " << g_host << ":" << g_port << "\r\n"
      << "Content-Type: application/json\r\n"
      << "Connection: keep-alive\r\n"
      << "Content-Length: " << body.size() << "\r\n\r\n"
      << body;
  return req.str();
}

void
Worker(const std::string& request)
{
  int fd = -1;
  std::string buf;
  while (g_run.load(std::memory_order_relaxed)) {
    if (fd < 0) {
      fd = Connect();
      buf.clear();
      if (fd < 0) {
        g_failure.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
    }
    if (!SendAll(fd, request) || !ReadResponse(fd, &buf)) {
      ::close(fd);
      fd = -1;
      g_failure.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    g_success.fetch_add(1, std::memory_order_relaxed);
  }
  if (fd >= 0) {
    ::close(fd);
  }
}

void
Usage(const char* argv0)
{
  std::cerr << "Usage: " << argv0
            << " [--host H] [--port P] [--model M] [--concurrency N] [--seconds S]\n";
}

}  // namespace

int
main(int argc, char** argv)
{
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need = [&]() {
      if (i + 1 >= argc) {
        Usage(argv[0]);
        std::exit(2);
      }
      return std::string(argv[++i]);
    };
    if (arg == "--host") {
      g_host = need();
    } else if (arg == "--port") {
      g_port = std::stoi(need());
    } else if (arg == "--model") {
      g_model = need();
    } else if (arg == "--concurrency") {
      g_concurrency = std::stoi(need());
    } else if (arg == "--seconds") {
      g_seconds = std::stoi(need());
    } else {
      Usage(argv[0]);
      return 2;
    }
  }
  if (g_concurrency < 1 || g_seconds < 1) {
    Usage(argv[0]);
    return 2;
  }

  const std::string request = BuildRequest();
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(g_concurrency));
  for (int i = 0; i < g_concurrency; ++i) {
    threads.emplace_back(Worker, std::cref(request));
  }

  const auto start = std::chrono::steady_clock::now();
  uint64_t last = 0;
  for (int elapsed = 5; elapsed <= g_seconds; elapsed += 5) {
    std::this_thread::sleep_until(start + std::chrono::seconds(elapsed));
    const uint64_t now = g_success.load(std::memory_order_relaxed);
    const double window = static_cast<double>(now - last) / 5.0;
    std::cout << "window_rps " << static_cast<uint64_t>(window) << std::endl;
    last = now;
  }
  const auto remaining = std::chrono::seconds(g_seconds) -
                         (std::chrono::steady_clock::now() - start);
  if (remaining.count() > 0) {
    std::this_thread::sleep_for(remaining);
  }
  g_run.store(false, std::memory_order_relaxed);
  for (auto& thread : threads) {
    thread.join();
  }

  const double seconds = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - start)
                             .count();
  const uint64_t success = g_success.load();
  std::cout << "success " << success << "\n"
            << "failure " << g_failure.load() << "\n"
            << "seconds " << seconds << "\n"
            << "rps " << (static_cast<double>(success) / seconds) << std::endl;
  return success == 0 ? 1 : 0;
}
