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

#include "src/servers/tracer.h"

#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/servers/common.h"

namespace nvidia { namespace inferenceserver {

TRTSERVER_Error*
TraceManager::Create(
    std::shared_ptr<TraceManager>* manager, const std::string& filepath)
{
  if (filepath.empty()) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INVALID_ARG,
        "trace configuration requires a non-empty file path");
  }

  try {
    std::unique_ptr<std::ofstream> trace_file(new std::ofstream);
    trace_file->open(filepath);

    LOG_INFO << "Configure trace: " << filepath;
    manager->reset(new TraceManager(std::move(trace_file)));
  }
  catch (const std::ofstream::failure& e) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INVALID_ARG,
        std::string("failed creating trace file: " + std::string(e.what()))
            .c_str());
  }
  catch (...) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INVALID_ARG,
        "failed creating trace file: reason unknown");
  }

  return nullptr;  // success
}

TraceManager::TraceManager(std::unique_ptr<std::ofstream> trace_file)
    : trace_file_(std::move(trace_file)), trace_cnt_(0),
      level_(TRTSERVER_TRACE_LEVEL_DISABLED), rate_(1000), sample_(1)
{
}

TraceManager::~TraceManager()
{
  LOG_INFO << "Close trace";

  if (trace_cnt_ > 0) {
    *trace_file_ << "]";
  }

  trace_file_->close();
}

TRTSERVER_Error*
TraceManager::SetLevel(TRTSERVER_Trace_Level level)
{
  // We don't bother with a mutex here since this is the only writer.
  level_ = level;

  LOG_INFO << "Setting trace level: " << level;

  return nullptr;  // success
}

TRTSERVER_Error*
TraceManager::SetRate(uint32_t rate)
{
  // We don't bother with a mutex here since this is the only writer.
  rate_ = rate;

  LOG_INFO << "Setting trace rate: " << rate;

  return nullptr;  // success
}

Tracer*
TraceManager::SampleTrace()
{
  uint64_t s = sample_.fetch_add(1);
  if ((s % rate_) != 0) {
    return nullptr;
  }

  Tracer* tracer = new Tracer(shared_from_this(), level_);

  TRTSERVER_Trace* trace = nullptr;
  TRTSERVER_Error* err = TRTSERVER_TraceNew(
      &trace, level_, Tracer::TraceActivity, tracer /* userp */);
  if (err != nullptr) {
    delete tracer;

    LOG_ERROR << "error creating trace: " << TRTSERVER_ErrorCodeString(err)
              << " - " << TRTSERVER_ErrorMessage(err);
    TRTSERVER_ErrorDelete(err);

    return nullptr;
  }

  tracer->SetServerTrace(trace);

  return tracer;
}

void
TraceManager::WriteTrace(const std::stringstream& ss)
{
  std::lock_guard<std::mutex> lock(mu_);

  if (trace_file_ != nullptr) {
    if (trace_cnt_ == 0) {
      *trace_file_ << "[";
    } else {
      *trace_file_ << ",";
    }

    *trace_file_ << ss.rdbuf();
  }

  trace_cnt_++;
}

Tracer::Tracer(
    const std::shared_ptr<TraceManager>& manager, TRTSERVER_Trace_Level level)
    : manager_(manager), level_(level), model_version_(-1), timestamp_cnt_(0)
{
  tout_ << "{ \"timestamps\": [";
}

Tracer::~Tracer()
{
  tout_ << "], \"model_name\": \"" << model_name_
        << "\", \"model_version\": " << model_version_ << " }";
  manager_->WriteTrace(tout_);

  LOG_IF_ERR(TRTSERVER_TraceDelete(trace_), "deleting trace");
}

void
Tracer::CaptureTimestamp(
    TRTSERVER_Trace_Level level, const std::string& name, uint64_t timestamp_ns)
{
  if (level <= level_) {
    if (timestamp_ns == 0) {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      timestamp_ns = TIMESPEC_TO_NANOS(ts);
    }

    if (timestamp_cnt_ != 0) {
      tout_ << ",";
    }

    tout_ << "{\"name\":\"" << name << "\", \"ns\":" << timestamp_ns << "}";
    timestamp_cnt_++;
  }
}

void
Tracer::TraceActivity(
    TRTSERVER_Trace* trace, TRTSERVER_Trace_Activity activity,
    uint64_t timestamp_ns, void* userp)
{
  Tracer* tracer = reinterpret_cast<Tracer*>(userp);

  const char* activity_name = "<unknown>";
  switch (activity) {
    case TRTSERVER_TRACE_REQUEST_START:
      activity_name = "request handler start";
      break;
    case TRTSERVER_TRACE_QUEUE_START:
      activity_name = "queue start";
      break;
    case TRTSERVER_TRACE_COMPUTE_START:
      activity_name = "compute start";
      break;
    case TRTSERVER_TRACE_COMPUTE_INPUT_END:
      activity_name = "compute input end";
      break;
    case TRTSERVER_TRACE_COMPUTE_OUTPUT_START:
      activity_name = "compute output start";
      break;
    case TRTSERVER_TRACE_COMPUTE_END:
      activity_name = "compute end";
      break;
    case TRTSERVER_TRACE_REQUEST_END:
      activity_name = "request handler end";
      break;
  }

  tracer->CaptureTimestamp(tracer->level_, activity_name, timestamp_ns);
}

}}  // namespace nvidia::inferenceserver
