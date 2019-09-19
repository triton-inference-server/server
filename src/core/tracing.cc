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

#ifdef TRTIS_ENABLE_TRACING

#include "src/core/tracing.h"

#include "src/core/constants.h"

namespace nvidia { namespace inferenceserver {

void
Trace::Report(const std::shared_ptr<ModelInferStats>& infer_stats)
{
  if (level_ != TRTSERVER_TRACE_LEVEL_DISABLED) {
    activity_fn_(
        reinterpret_cast<TRTSERVER_Trace*>(this), TRTSERVER_TRACE_REQUEST_START,
        TIMESPEC_TO_NANOS(infer_stats->Timestamp(
            ModelInferStats::TimestampKind::kRequestStart)),
        activity_userp_);
    activity_fn_(
        reinterpret_cast<TRTSERVER_Trace*>(this), TRTSERVER_TRACE_QUEUE_START,
        TIMESPEC_TO_NANOS(infer_stats->Timestamp(
            ModelInferStats::TimestampKind::kQueueStart)),
        activity_userp_);
    activity_fn_(
        reinterpret_cast<TRTSERVER_Trace*>(this), TRTSERVER_TRACE_COMPUTE_START,
        TIMESPEC_TO_NANOS(infer_stats->Timestamp(
            ModelInferStats::TimestampKind::kComputeStart)),
        activity_userp_);
    activity_fn_(
        reinterpret_cast<TRTSERVER_Trace*>(this), TRTSERVER_TRACE_COMPUTE_END,
        TIMESPEC_TO_NANOS(infer_stats->Timestamp(
            ModelInferStats::TimestampKind::kComputeEnd)),
        activity_userp_);
    activity_fn_(
        reinterpret_cast<TRTSERVER_Trace*>(this), TRTSERVER_TRACE_REQUEST_END,
        TIMESPEC_TO_NANOS(infer_stats->Timestamp(
            ModelInferStats::TimestampKind::kRequestEnd)),
        activity_userp_);
  }

  if (level_ == TRTSERVER_TRACE_LEVEL_MAX) {
    activity_fn_(
        reinterpret_cast<TRTSERVER_Trace*>(this),
        TRTSERVER_TRACE_COMPUTE_INPUT_END,
        TIMESPEC_TO_NANOS(infer_stats->Timestamp(
            ModelInferStats::TimestampKind::kComputeInputEnd)),
        activity_userp_);
    activity_fn_(
        reinterpret_cast<TRTSERVER_Trace*>(this),
        TRTSERVER_TRACE_COMPUTE_OUTPUT_START,
        TIMESPEC_TO_NANOS(infer_stats->Timestamp(
            ModelInferStats::TimestampKind::kComputeOutputStart)),
        activity_userp_);
  }
}

}}  // namespace nvidia::inferenceserver

#endif  // TRTIS_ENABLE_TRACING
