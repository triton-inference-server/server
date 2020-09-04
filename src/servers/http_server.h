// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include <memory>
#include <string>
#include "src/servers/shared_memory_manager.h"
#include "src/servers/tracer.h"
#include "triton/common/tritonserver.h"

namespace nvidia { namespace inferenceserver {

class HTTPServer {
 public:
  static TRITONSERVER_Error* CreateAPIServer(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      nvidia::inferenceserver::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& smb_manager,
      const int32_t port, const int thread_cnt,
      std::unique_ptr<HTTPServer>* http_server);

  static TRITONSERVER_Error* CreateMetricsServer(
      const std::shared_ptr<TRITONSERVER_Server>& server, int32_t port,
      int thread_cnt, std::unique_ptr<HTTPServer>* metrics_server);

  virtual ~HTTPServer() = default;

  virtual TRITONSERVER_Error* Start() = 0;
  virtual TRITONSERVER_Error* Stop() = 0;
};

}}  // namespace nvidia::inferenceserver
