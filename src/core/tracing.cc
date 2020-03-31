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

#include "src/core/tracing.h"

#include "src/core/constants.h"

namespace nvidia { namespace inferenceserver {

namespace {

TRTSERVER_Trace_Level
TritonTraceLevelToTrt(TRITONSERVER_Trace_Level level)
{
  switch (level) {
    case TRITONSERVER_TRACE_LEVEL_DISABLED:
      return TRTSERVER_TRACE_LEVEL_DISABLED;
      break;
    case TRITONSERVER_TRACE_LEVEL_MIN:
      return TRTSERVER_TRACE_LEVEL_MIN;
      break;
    default:
      return TRTSERVER_TRACE_LEVEL_MAX;
      break;
  }
}

#define REPORT_TIMESTAMP(ACTIVITY_TYPE)                                    \
  do {                                                                     \
    if (using_triton_) {                                                   \
      triton_activity_fn_(                                                 \
          reinterpret_cast<TRITONSERVER_Trace*>(this),                     \
          TRITONSERVER_TRACE_##ACTIVITY_TYPE, timestamp, activity_userp_); \
    } else {                                                               \
      activity_fn_(                                                        \
          reinterpret_cast<TRTSERVER_Trace*>(this),                        \
          TRTSERVER_TRACE_##ACTIVITY_TYPE, timestamp, activity_userp_);    \
    }                                                                      \
  } while (false)


}  // namespace
std::atomic<int64_t> Trace::next_id_(0);

Trace::Trace(
    TRITONSERVER_Trace_Level level, TRITONSERVER_TraceActivityFn_t activity_fn,
    void* activity_userp)
    : using_triton_(true), level_(TritonTraceLevelToTrt(level)),
      activity_fn_(nullptr), triton_activity_fn_(activity_fn),
      activity_userp_(activity_userp), id_(next_id_++), parent_id_(-1)
{
}

void
Trace::Report(const ModelInferStats* infer_stats)
{
  // InferStats that is not captured should not be reported (i.e. ensemble
  // only have valid timestamp for request start and end)
  uint64_t timestamp = 0;
  if (level_ != TRTSERVER_TRACE_LEVEL_DISABLED) {
    timestamp = TIMESPEC_TO_NANOS(
        infer_stats->Timestamp(ModelInferStats::TimestampKind::kRequestStart));
    if (timestamp != 0) {
      REPORT_TIMESTAMP(REQUEST_START);
    }
    timestamp = TIMESPEC_TO_NANOS(
        infer_stats->Timestamp(ModelInferStats::TimestampKind::kQueueStart));
    if (timestamp != 0) {
      REPORT_TIMESTAMP(QUEUE_START);
    }
    timestamp = TIMESPEC_TO_NANOS(
        infer_stats->Timestamp(ModelInferStats::TimestampKind::kComputeStart));
    if (timestamp != 0) {
      REPORT_TIMESTAMP(COMPUTE_START);
    }
    timestamp = TIMESPEC_TO_NANOS(
        infer_stats->Timestamp(ModelInferStats::TimestampKind::kComputeEnd));
    if (timestamp != 0) {
      REPORT_TIMESTAMP(COMPUTE_END);
    }
    timestamp = TIMESPEC_TO_NANOS(
        infer_stats->Timestamp(ModelInferStats::TimestampKind::kRequestEnd));
    if (timestamp != 0) {
      REPORT_TIMESTAMP(REQUEST_END);
    }
  }

  if (level_ == TRTSERVER_TRACE_LEVEL_MAX) {
    timestamp = TIMESPEC_TO_NANOS(infer_stats->Timestamp(
        ModelInferStats::TimestampKind::kComputeInputEnd));
    if (timestamp != 0) {
      REPORT_TIMESTAMP(COMPUTE_INPUT_END);
    }
    timestamp = TIMESPEC_TO_NANOS(infer_stats->Timestamp(
        ModelInferStats::TimestampKind::kComputeOutputStart));
    if (timestamp != 0) {
      REPORT_TIMESTAMP(COMPUTE_OUTPUT_START);
    }
  }
}

}}  // namespace nvidia::inferenceserver
