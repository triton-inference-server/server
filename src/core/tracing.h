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

#ifdef TRTIS_ENABLE_TRACING

#include <atomic>
#include <memory>
#include <vector>
#include "src/core/server_status.h"
#include "src/core/status.h"
#include "src/core/trtserver.h"

namespace nvidia { namespace inferenceserver {

//
// A light-weight structure to store user-provided trace manager.
//
struct OpaqueTraceManager {
  TRTSERVER_TraceManagerCreateTraceFn_t create_fn_;
  TRTSERVER_TraceManagerReleaseTraceFn_t release_fn_;
  void* userp_;
};

//
// A trace.
//
class Trace {
 public:
  static Status Create(
      TRTSERVER_Trace_Level level, TRTSERVER_TraceActivityFn_t activity_fn,
      void* activity_userp, std::unique_ptr<Trace>* trace)
  {
    trace->reset(new Trace(level, activity_fn, activity_userp));
    return Status::Success;
  }

  void SetModelName(const std::string& n) { model_name_ = n; }
  void SetModelVersion(int64_t v) { model_version_ = v; }
  void SetParentId(int64_t pid) { parent_id_ = pid; }

  void* ActivityUserp() const { return activity_userp_; }
  const char* ModelName() const { return model_name_.c_str(); }
  int64_t ModelVersion() const { return model_version_; }
  int64_t Id() const { return id_; }
  int64_t ParentId() const { return parent_id_; }

  void Report(const ModelInferStats* infer_stats);

 private:
  Trace(
      TRTSERVER_Trace_Level level, TRTSERVER_TraceActivityFn_t activity_fn,
      void* activity_userp)
      : level_(level), activity_fn_(activity_fn),
        activity_userp_(activity_userp), id_(next_id_++), parent_id_(-1)
  {
  }

  const TRTSERVER_Trace_Level level_;
  TRTSERVER_TraceActivityFn_t activity_fn_;
  void* activity_userp_;

  std::string model_name_;
  int64_t model_version_;

  // unique id will be assigned when the trace object is being created
  int64_t id_;
  int64_t parent_id_;

  // Maintain next id statically so that trace id is unique even across traces
  static std::atomic<int64_t> next_id_;
};

}}  // namespace nvidia::inferenceserver

#endif  // TRTIS_ENABLE_TRACING
