// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>

namespace triton { namespace server {

// Configure a socket to not generate SIGPIPE on macOS
// Returns 0 on success, -1 on failure
inline int ConfigureMacOSSocket(int sockfd) {
  int set = 1;
  return setsockopt(sockfd, SOL_SOCKET, SO_NOSIGPIPE, (void*)&set, sizeof(int));
}

// Wrapper for send() that handles SIGPIPE on macOS
// On Linux, MSG_NOSIGNAL prevents SIGPIPE, but macOS doesn't support this flag
inline ssize_t SafeSend(int sockfd, const void* buf, size_t len, int flags) {
#ifdef MSG_NOSIGNAL
  // Linux supports MSG_NOSIGNAL
  return send(sockfd, buf, len, flags | MSG_NOSIGNAL);
#else
  // macOS doesn't support MSG_NOSIGNAL, rely on SO_NOSIGPIPE set on socket
  ssize_t ret = send(sockfd, buf, len, flags);
  if (ret == -1 && errno == EPIPE) {
    // Handle broken pipe without terminating the process
    errno = EPIPE;  // Preserve the error
  }
  return ret;
#endif
}

}}  // namespace triton::server

#endif  // __APPLE__