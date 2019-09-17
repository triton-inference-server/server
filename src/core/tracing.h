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

#include <vector>
#include "src/core/server_status.h"
#include "src/core/status.h"
#include "src/core/trtserver.h"

namespace nvidia { namespace inferenceserver {

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

  void Report(const std::shared_ptr<ModelInferStats>& infer_stats);

 private:
  Trace(
      TRTSERVER_Trace_Level level, TRTSERVER_TraceActivityFn_t activity_fn,
      void* activity_userp)
      : level_(level), activity_fn_(activity_fn),
        activity_userp_(activity_userp)
  {
  }

  const TRTSERVER_Trace_Level level_;
  TRTSERVER_TraceActivityFn_t activity_fn_;
  void* activity_userp_;
};

}}  // namespace nvidia::inferenceserver

#endif  // TRTIS_ENABLE_TRACING
