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

#include "src/clients/c++/perf_client/perf_utils.h"

namespace perfclient {

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

}  // namespace perfclient
