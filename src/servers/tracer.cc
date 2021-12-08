// Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <stdlib.h>
#include <unordered_map>
#include "src/core/constants.h"
#include "src/core/logging.h"
#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#include "src/servers/common.h"
#endif  // TRITON_ENABLE_GPU

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
TraceManager::SampleTrace(void** userp)
{
  uint64_t s = sample_.fetch_add(1);
  if ((s % rate_) != 0) {
    return nullptr;
  }

  // userp is a pair of the trace manager and a string buffer where
  // the trace collects its trace activity output.
  std::unique_ptr<TraceStreams> luserp(new TraceStreams(this));

  TRITONSERVER_InferenceTrace* trace;
  TRITONSERVER_Error* err = TRITONSERVER_InferenceTraceTensorNew(
      &trace, level_, 0 /* parent_id */, TraceActivity, TraceTensorActivity,
      TraceStreamRelease, luserp.get());
  if (err != nullptr) {
    LOG_TRITONSERVER_ERROR(err, "creating inference trace object");
    return nullptr;
  }

  if (userp != nullptr) {
    *userp = luserp.get();
  }
  luserp.release();

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
  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceTraceDelete(trace), "deleting trace");
}

void
TraceManager::TraceStreamRelease(
    TRITONSERVER_InferenceTrace* trace, void* userp)
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
  if (ss != nullptr) {
    ts->manager_->WriteTrace(*ss);
  }

  uint64_t parent_id;
  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceTraceParentId(trace, &parent_id),
      "getting trace parent id");
  // The userp will be shared with the trace children, so only delete it
  // if the root trace is being released
  if (parent_id == 0) {
    delete ts;
  }
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
  std::lock_guard<std::mutex> lk(ts->mtx_);

  std::stringstream* ss = nullptr;
  {
    if (activity == TRITONSERVER_TRACE_REQUEST_START) {
      std::unique_ptr<std::stringstream> stream(new std::stringstream());
      ss = stream.get();
      ts->streams_.emplace(id, std::move(stream));
    } else {
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

void
TraceManager::TraceTensorActivity(
    TRITONSERVER_InferenceTrace* trace,
    TRITONSERVER_InferenceTraceActivity activity, const char* name,
    TRITONSERVER_DataType datatype, const void* base, size_t byte_size,
    const int64_t* shape, uint64_t dim_count,
    TRITONSERVER_MemoryType memory_type, int64_t memory_type_id, void* userp)
{
  if ((activity != TRITONSERVER_TRACE_TENSOR_QUEUE_INPUT) &&
      (activity != TRITONSERVER_TRACE_TENSOR_BACKEND_INPUT) &&
      (activity != TRITONSERVER_TRACE_TENSOR_BACKEND_OUTPUT)) {
    LOG_ERROR << "Unsupported activity: "
              << TRITONSERVER_InferenceTraceActivityString(activity);
    return;
  }

  void* buffer_base = const_cast<void*>(base);
  if (memory_type == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
    buffer_base = malloc(byte_size);
    if (buffer_base == nullptr) {
      LOG_ERROR << "Failed to malloc CPU buffer";
      return;
    }
    FAIL_IF_CUDA_ERR(
        cudaMemcpy(buffer_base, base, byte_size, cudaMemcpyDeviceToHost),
        "copying buffer into CPU memory");
#else
    LOG_ERROR << "GPU buffer is unsupported";
    return;
#endif  // TRITON_ENABLE_GPU
  }

  uint64_t id;
  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceTraceId(trace, &id), "getting trace id");

  // The function may be called with different traces but the same 'userp',
  // group the activity of the same trace together for more readable output.
  auto ts = reinterpret_cast<TraceStreams*>(userp);
  std::lock_guard<std::mutex> lk(ts->mtx_);

  std::stringstream* ss = nullptr;
  {
    if (ts->streams_.find(id) == ts->streams_.end()) {
      std::unique_ptr<std::stringstream> stream(new std::stringstream());
      ss = stream.get();
      ts->streams_.emplace(id, std::move(stream));
    } else {
      ss = ts->streams_[id].get();
    }
  }

  // collect and serialize trace details.
  *ss << ",{\"id\":" << id << ",\"activity\":\""
      << TRITONSERVER_InferenceTraceActivityString(activity) << "\"";
  // collect tensor
  *ss << ",\"tensor\":{";
  // collect tensor name
  *ss << "\"name\":\"" << std::string(name) << "\"";
  // collect tensor data
  *ss << ",\"data\":\"";
  size_t element_count = 1;
  for (uint64_t i = 0; i < dim_count; i++) {
    element_count *= shape[i];
  }
  switch (datatype) {
    case TRITONSERVER_TYPE_BOOL: {
      const uint8_t* bool_base = reinterpret_cast<const uint8_t*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << ((bool_base[e] == 0) ? false : true);
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_UINT8: {
      const uint8_t* cbase = reinterpret_cast<const uint8_t*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << cbase[e];
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_UINT16: {
      const uint16_t* cbase = reinterpret_cast<const uint16_t*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << cbase[e];
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_UINT32: {
      const uint32_t* cbase = reinterpret_cast<const uint32_t*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << cbase[e];
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_UINT64: {
      const uint64_t* cbase = reinterpret_cast<const uint64_t*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << cbase[e];
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_INT8: {
      const int8_t* cbase = reinterpret_cast<const int8_t*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << cbase[e];
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_INT16: {
      const int16_t* cbase = reinterpret_cast<const int16_t*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << cbase[e];
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_INT32: {
      const int32_t* cbase = reinterpret_cast<const int32_t*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << cbase[e];
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_INT64: {
      const int64_t* cbase = reinterpret_cast<const int64_t*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << cbase[e];
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_FP16: {
      break;
    }
    case TRITONSERVER_TYPE_FP32: {
      const float* cbase = reinterpret_cast<const float*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << cbase[e];
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_FP64: {
      const double* cbase = reinterpret_cast<const double*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << cbase[e];
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_BYTES: {
      const char* cbase = reinterpret_cast<const char*>(buffer_base);
      size_t offset = 0;
      for (size_t e = 0; e < element_count; ++e) {
        if ((offset + sizeof(uint32_t)) > byte_size) {
          return;
        }
        const size_t len = *(reinterpret_cast<const uint32_t*>(cbase + offset));
        offset += sizeof(uint32_t);
        if ((offset + len) > byte_size) {
          return;
        }
        std::string str(cbase + offset, len);
        *ss << "\"" << str << "\"";
        offset += len;

        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_INVALID: {
      return;
    }
  }
  *ss << "\",\"shape\":\"";
  for (uint64_t i = 0; i < dim_count; i++) {
    *ss << shape[i];
    if (i < (dim_count - 1)) {
      *ss << ",";
    }
  }
  *ss << "\",\"dtype\":\"" << TRITONSERVER_DataTypeString(datatype) << "\"}";
  *ss << "}";

  if (memory_type == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
    if (buffer_base != nullptr) {
      free(buffer_base);
    }
#endif  // TRITON_ENABLE_GPU
  }
}
}}  // namespace nvidia::inferenceserver
