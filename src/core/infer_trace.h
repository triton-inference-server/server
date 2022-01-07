// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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
#include <chrono>
#include <memory>
#include "src/core/constants.h"
#include "src/core/status.h"
#include "src/core/tritonserver_apis.h"

namespace nvidia { namespace inferenceserver {

#ifdef TRITON_ENABLE_TRACING

//
// InferenceTrace
//
// Interface to TRITONSERVER_InferenceTrace to report trace events.
//
class InferenceTrace {
 public:
  InferenceTrace(
      const TRITONSERVER_InferenceTraceLevel level, const uint64_t parent_id,
      TRITONSERVER_InferenceTraceActivityFn_t activity_fn,
      TRITONSERVER_InferenceTraceTensorActivityFn_t tensor_activity_fn,
      TRITONSERVER_InferenceTraceReleaseFn_t release_fn, void* userp)
      : level_(level), id_(next_id_++), parent_id_(parent_id),
        activity_fn_(activity_fn), tensor_activity_fn_(tensor_activity_fn),
        release_fn_(release_fn), userp_(userp)
  {
  }

  InferenceTrace* SpawnChildTrace();

  int64_t Id() const { return id_; }
  int64_t ParentId() const { return parent_id_; }

  const std::string& ModelName() const { return model_name_; }
  int64_t ModelVersion() const { return model_version_; }

  void SetModelName(const std::string& n) { model_name_ = n; }
  void SetModelVersion(int64_t v) { model_version_ = v; }

  // Report trace activity.
  void Report(
      const TRITONSERVER_InferenceTraceActivity activity, uint64_t timestamp_ns)
  {
    if (level_ < TRITONSERVER_TRACE_LEVEL_TIMESTAMPS) {
      return;
    }

    activity_fn_(
        reinterpret_cast<TRITONSERVER_InferenceTrace*>(this), activity,
        timestamp_ns, userp_);
  }

  // Report trace activity at the current time.
  void ReportNow(const TRITONSERVER_InferenceTraceActivity activity)
  {
    if (level_ < TRITONSERVER_TRACE_LEVEL_TIMESTAMPS) {
      return;
    }

    Report(
        activity, std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
  }

  // Report tensor trace activity.
  void ReportTensor(
      const TRITONSERVER_InferenceTraceActivity activity, const char* name,
      TRITONSERVER_DataType datatype, const void* base, size_t byte_size,
      const int64_t* shape, uint64_t dim_count,
      TRITONSERVER_MemoryType memory_type, int64_t memory_type_id)
  {
    if (level_ < TRITONSERVER_TRACE_LEVEL_TENSORS) {
      return;
    }

    tensor_activity_fn_(
        reinterpret_cast<TRITONSERVER_InferenceTrace*>(this), activity, name,
        datatype, base, byte_size, shape, dim_count, memory_type,
        memory_type_id, userp_);
  }

  // Release the trace. Call the trace release callback.
  void Release();

 private:
  const TRITONSERVER_InferenceTraceLevel level_;
  const uint64_t id_;
  const uint64_t parent_id_;

  TRITONSERVER_InferenceTraceActivityFn_t activity_fn_;
  TRITONSERVER_InferenceTraceTensorActivityFn_t tensor_activity_fn_;
  TRITONSERVER_InferenceTraceReleaseFn_t release_fn_;
  void* userp_;

  std::string model_name_;
  int64_t model_version_;

  // Maintain next id statically so that trace id is unique even
  // across traces
  static std::atomic<uint64_t> next_id_;
};

//
// InferenceTraceProxy
//
// Object attached as shared_ptr to InferenceRequest and
// InferenceResponse(s) being traced as part of a single inference
// request.
//
class InferenceTraceProxy {
 public:
  InferenceTraceProxy(InferenceTrace* trace) : trace_(trace) {}
  ~InferenceTraceProxy() { trace_->Release(); }
  int64_t Id() const { return trace_->Id(); }
  int64_t ParentId() const { return trace_->ParentId(); }
  const std::string& ModelName() const { return trace_->ModelName(); }
  int64_t ModelVersion() const { return trace_->ModelVersion(); }
  void SetModelName(const std::string& n) { trace_->SetModelName(n); }
  void SetModelVersion(int64_t v) { trace_->SetModelVersion(v); }

  void Report(
      const TRITONSERVER_InferenceTraceActivity activity, uint64_t timestamp_ns)
  {
    trace_->Report(activity, timestamp_ns);
  }

  void ReportNow(const TRITONSERVER_InferenceTraceActivity activity)
  {
    trace_->ReportNow(activity);
  }

  void ReportTensor(
      const TRITONSERVER_InferenceTraceActivity activity, const char* name,
      TRITONSERVER_DataType datatype, const void* base, size_t byte_size,
      const int64_t* shape, uint64_t dim_count,
      TRITONSERVER_MemoryType memory_type, int64_t memory_type_id)
  {
    trace_->ReportTensor(
        activity, name, datatype, base, byte_size, shape, dim_count,
        memory_type, memory_type_id);
  }

  std::shared_ptr<InferenceTraceProxy> SpawnChildTrace();

 private:
  InferenceTrace* trace_;
};

#endif  // TRITON_ENABLE_TRACING

//
// Macros to generate trace activity
//
#ifdef TRITON_ENABLE_TRACING
#define INFER_TRACE_ACTIVITY(T, A, TS_NS) \
  {                                       \
    const auto& trace = (T);              \
    const auto ts_ns = (TS_NS);           \
    if (trace != nullptr) {               \
      trace->Report(A, ts_ns);            \
    }                                     \
  }
#define INFER_TRACE_ACTIVITY_NOW(T, A) \
  {                                    \
    const auto& trace = (T);           \
    if (trace != nullptr) {            \
      trace->ReportNow(A);             \
    }                                  \
  }
#define INFER_TRACE_TENSOR_ACTIVITY(T, A, N, D, BA, BY, S, DI, MT, MTI) \
  {                                                                     \
    const auto& trace = (T);                                            \
    if (trace != nullptr) {                                             \
      trace->ReportTensor(A, N, D, BA, BY, S, DI, MT, MTI);             \
    }                                                                   \
  }
#else
#define INFER_TRACE_ACTIVITY(T, A, TS_NS)
#define INFER_TRACE_ACTIVITY_NOW(T, A)
#define INFER_TRACE_TENSOR_ACTIVITY(T, A, N, D, BA, BY, S, DI, MT, MTI)
#endif  // TRITON_ENABLE_TRACING
}}      // namespace nvidia::inferenceserver
