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
#pragma once

#include <atomic>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include "src/core/trtserver.h"

namespace nvidia { namespace inferenceserver {

class Tracer;
class TraceManager;

struct TraceMetaData {
  // just a pointer to the manager, 'tracer_' objects hold it's shared pointer.
  TraceManager* manager_;
  std::atomic<bool> trace_set_;
  std::unique_ptr<Tracer> tracer_;
};

//
// Manager for tracing to a file.
//
class TraceManager : public std::enable_shared_from_this<TraceManager> {
 public:
  // Create a trace manager that appends trace information
  // to a specified file.
  static TRTSERVER_Error* Create(
      std::shared_ptr<TraceManager>* manager, const std::string& filepath);

  ~TraceManager();

  // Set the trace level and sampling rate.
  TRTSERVER_Error* SetLevel(TRTSERVER_Trace_Level level);
  TRTSERVER_Error* SetRate(uint32_t rate);

  // Return a trace meta data object that should be used to collected trace
  // activities for an inference request. Return nullptr if no tracing
  // should occur.
  TraceMetaData* SampleTrace();

  // Create a trace object that should be used to collected trace
  // activities for the model execution. Return nullptr if no tracing
  // should occur.
  static void CreateTrace(
      TRTSERVER_Trace** trace, const char* model_name, int64_t version,
      void* userp);

  static void ReleaseTrace(
      TRTSERVER_Trace* trace, void* activity_userp, void* userp);

  // Write to the trace file.
  void WriteTrace(const std::stringstream& ss);

 private:
  TraceManager(std::unique_ptr<std::ofstream> trace_file);

  // Helper function to create a new trace object.
  void NewTrace(TRTSERVER_Trace** trace);

  std::mutex mu_;
  std::unique_ptr<std::ofstream> trace_file_;
  uint32_t trace_cnt_;

  TRTSERVER_Trace_Level level_;
  uint32_t rate_;

  // Atomically incrementing counter used to implement sampling rate.
  std::atomic<uint64_t> sample_;
};

//
// A single trace
//
class Tracer {
 public:
  Tracer(
      const std::shared_ptr<TraceManager>& manager,
      TRTSERVER_Trace_Level level);
  ~Tracer();

  static void TraceActivity(
      TRTSERVER_Trace* trace, TRTSERVER_Trace_Activity activity,
      uint64_t timestamp_ns, void* userp);

  void SetModel(const std::string& model_name, int64_t model_version)
  {
    model_name_ = model_name;
    model_version_ = model_version;
  }

  void SetId(int64_t id, int64_t parent_id)
  {
    id_ = id;
    parent_id_ = parent_id;
  }

  void SetServerTrace(TRTSERVER_Trace* trace) { trace_ = trace; }
  TRTSERVER_Trace* ServerTrace() const { return trace_; }

  // Capture a named timestamp using a nanosecond precision time. If
  // the time is not given (or is given as zero) then the current time
  // will be used.
  void CaptureTimestamp(
      TRTSERVER_Trace_Level level, const std::string& name,
      uint64_t timestamp_ns = 0);

 private:
  std::shared_ptr<TraceManager> manager_;
  const TRTSERVER_Trace_Level level_;

  std::string model_name_;
  int64_t model_version_;

  int64_t id_;
  int64_t parent_id_;

  std::stringstream tout_;
  uint32_t timestamp_cnt_;

  TRTSERVER_Trace* trace_;
};

}}  // namespace nvidia::inferenceserver
