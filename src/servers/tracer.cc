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

#include "src/servers/tracer.h"

#include <unordered_map>
#include "src/core/constants.h"
#include "src/core/logging.h"

namespace nvidia { namespace inferenceserver {

namespace {
struct TraceStreams {
  TraceStreams(TraceManager* manager) : manager_(manager) {}
  TraceManager* manager_;
  std::mutex mtx_;
  std::unordered_map<uint64_t, std::unique_ptr<std::stringstream>> streams_;
};
}  // namespace

TRITONSERVER_Error*
TraceManager::Create(
    TraceManager** manager, const TRITONSERVER_InferenceTraceLevel level,
    const uint32_t rate, const std::string& filepath)
{
  if (filepath.empty()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "trace configuration requires a non-empty file path");
  }

  try {
    std::unique_ptr<std::ofstream> trace_file(new std::ofstream);
    trace_file->open(filepath);

    LOG_INFO << "Configure trace: " << filepath;
    *manager = new TraceManager(level, rate, std::move(trace_file));
  }
  catch (const std::ofstream::failure& e) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("failed creating trace file: " + std::string(e.what()))
            .c_str());
  }
  catch (...) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "failed creating trace file: reason unknown");
  }

  return nullptr;  // success
}

TraceManager::TraceManager(
    const TRITONSERVER_InferenceTraceLevel level, const uint32_t rate,
    std::unique_ptr<std::ofstream>&& trace_file)
    : level_(level), rate_(rate), trace_file_(std::move(trace_file)),
      trace_cnt_(0), sample_(1)
{
}

TraceManager::~TraceManager()
{
  if (trace_cnt_ > 0) {
    *trace_file_ << "]";
  }

  trace_file_->close();
}

TRITONSERVER_InferenceTrace*
TraceManager::SampleTrace()
{
  uint64_t s = sample_.fetch_add(1);
  if ((s % rate_) != 0) {
    return nullptr;
  }

  // userp is a pair of the trace manager and a string buffer where
  // the trace collects its trace activity output.
  std::unique_ptr<TraceStreams> userp(new TraceStreams(this));

  TRITONSERVER_InferenceTrace* trace;
  TRITONSERVER_Error* err = TRITONSERVER_InferenceTraceNew(
      &trace, level_, 0 /* parent_id */, TraceActivity, TraceRelease,
      userp.get());
  if (err != nullptr) {
    LOG_TRITONSERVER_ERROR(err, "creating inference trace object");
    return nullptr;
  }

  userp.release();

  return trace;
}

void
TraceManager::WriteTrace(const std::stringstream& ss)
{
  std::lock_guard<std::mutex> lock(mu_);

  if (trace_cnt_ == 0) {
    *trace_file_ << "[";
  } else {
    *trace_file_ << ",";
  }

  trace_cnt_++;

  *trace_file_ << ss.rdbuf();
}

void
TraceManager::CaptureTimestamp(
    const uint64_t trace_id, TRITONSERVER_InferenceTraceLevel level,
    const std::string& name, uint64_t timestamp_ns)
{
  if ((trace_id != 0) && (level <= level_)) {
    if (timestamp_ns == 0) {
      timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
    }

    std::stringstream ss;
    ss << "{\"id\":" << trace_id << ",\"timestamps\":["
       << "{\"name\":\"" << name << "\",\"ns\":" << timestamp_ns << "}]}";

    WriteTrace(ss);
  }
}

void
TraceManager::TraceRelease(TRITONSERVER_InferenceTrace* trace, void* userp)
{
  auto ts = reinterpret_cast<TraceStreams*>(userp);
  std::stringstream* ss = nullptr;
  {
    uint64_t id;
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceTraceId(trace, &id), "getting trace id");
    std::lock_guard<std::mutex> lk(ts->mtx_);
    ss = ts->streams_[id].get();
  }
  ts->manager_->WriteTrace(*ss);

  uint64_t parent_id;
  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceTraceParentId(trace, &parent_id),
      "getting trace parent id");
  // The userp will be shared with the trace children, so only delete it
  // if the root trace is being released
  if (parent_id == 0) {
    delete ts;
  }

  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceTraceDelete(trace), "deleting trace");
}

void
TraceManager::TraceActivity(
    TRITONSERVER_InferenceTrace* trace,
    TRITONSERVER_InferenceTraceActivity activity, uint64_t timestamp_ns,
    void* userp)
{
  uint64_t id;
  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceTraceId(trace, &id), "getting trace id");

  // The function may be called with different traces but the same 'userp',
  // group the activity of the same trace together for more readable output.
  auto ts = reinterpret_cast<TraceStreams*>(userp);
  std::stringstream* ss = nullptr;
  {
    if (activity == TRITONSERVER_TRACE_REQUEST_START) {
      std::unique_ptr<std::stringstream> stream(new std::stringstream());
      ss = stream.get();
      std::lock_guard<std::mutex> lk(ts->mtx_);
      ts->streams_.emplace(id, std::move(stream));
    } else {
      std::lock_guard<std::mutex> lk(ts->mtx_);
      ss = ts->streams_[id].get();
    }
  }

  // If 'activity' is TRITONSERVER_TRACE_REQUEST_START then collect
  // and serialize trace details.
  if (activity == TRITONSERVER_TRACE_REQUEST_START) {
    const char* model_name;
    int64_t model_version;
    uint64_t parent_id;

    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceTraceModelName(trace, &model_name),
        "getting model name");
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceTraceModelVersion(trace, &model_version),
        "getting model version");
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceTraceParentId(trace, &parent_id),
        "getting trace parent id");

    *ss << "{\"id\":" << id << ",\"model_name\":\"" << model_name
        << "\",\"model_version\":" << model_version;
    if (parent_id != 0) {
      *ss << ",\"parent_id\":" << parent_id;
    }
    *ss << "}";
  }

  *ss << ",{\"id\":" << id << ",\"timestamps\":["
      << "{\"name\":\"" << TRITONSERVER_InferenceTraceActivityString(activity)
      << "\",\"ns\":" << timestamp_ns << "}]}";
}

}}  // namespace nvidia::inferenceserver
