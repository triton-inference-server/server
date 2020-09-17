// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <atomic>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include "triton/core/tritonserver.h"

namespace nvidia { namespace inferenceserver {

//
// Manager for tracing to a file.
//
class TraceManager {
 public:
  // Create a trace manager that appends trace information
  // to a specified file.
  static TRITONSERVER_Error* Create(
      TraceManager** manager, const TRITONSERVER_InferenceTraceLevel level,
      const uint32_t rate, const std::string& filepath);

  ~TraceManager();

  // Return a trace that should be used to collected trace activities
  // for an inference request. Return nullptr if no tracing should
  // occur.
  TRITONSERVER_InferenceTrace* SampleTrace();

  // Capture a timestamp generated outside of triton and associate it
  // with a trace id. If 'timestamp_ns' is 0 the current timestamp
  // will be used.
  void CaptureTimestamp(
      const uint64_t trace_id, TRITONSERVER_InferenceTraceLevel level,
      const std::string& name, uint64_t timestamp_ns = 0);

 private:
  TraceManager(
      const TRITONSERVER_InferenceTraceLevel level, const uint32_t rate,
      std::unique_ptr<std::ofstream>&& trace_file);

  void WriteTrace(const std::stringstream& ss);

  static void TraceRelease(TRITONSERVER_InferenceTrace* trace, void* userp);
  static void TraceActivity(
      TRITONSERVER_InferenceTrace* trace,
      TRITONSERVER_InferenceTraceActivity activity, uint64_t timestamp_ns,
      void* userp);

  const TRITONSERVER_InferenceTraceLevel level_;
  const uint32_t rate_;
  std::unique_ptr<std::ofstream> trace_file_;

  std::mutex mu_;
  uint32_t trace_cnt_;

  // Atomically incrementing counter used to implement sampling rate.
  std::atomic<uint64_t> sample_;
};

}}  // namespace nvidia::inferenceserver
