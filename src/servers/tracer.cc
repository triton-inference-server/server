// Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TRITONSERVER_Error*
TraceManager::Create(
    TraceManager** manager, const TRITONSERVER_InferenceTraceLevel level,
    const uint32_t rate, const uint32_t log_frequency,
    const std::string& filepath)
{
  // Always create TraceManager regardless of the global setting as they
  // can be updated at runtime even if tracing is not enable at start.
  // No trace should be sampled if the setting is not valid.
  *manager = new TraceManager(level, rate, log_frequency, filepath);
  return nullptr;  // success
}

TraceManager::TraceManager(
    const TRITONSERVER_InferenceTraceLevel level, const uint32_t rate,
    const uint32_t log_frequency, const std::string& filepath)
{
  std::shared_ptr<TraceFile> file(new TraceFile(filepath));
  global_setting_.reset(new TraceSetting(level, rate, log_frequency, file));
  trace_files_.emplace(filepath);
}

void
TraceManager::ClearTraceSetting(const std::string& model_name)
{
  std::lock_guard<std::mutex> w_lk(w_mu_);
  if (!model_name.empty()) {
    auto it = model_settings_.find(model_name);
    if (it != model_settings_.end()) {
      trace_files_.erase(it->second->file_->FileName());
      std::lock_guard<std::mutex> r_lk(r_mu_);
      model_settings_.erase(model_name);
    }
  }
}

TRITONSERVER_Error*
TraceManager::UpdateTraceSetting(
    const std::string& model_name,
    const TRITONSERVER_InferenceTraceLevel* level, const uint32_t* rate,
    const uint32_t* log_frequency, const std::string& filepath)
{
  std::lock_guard<std::mutex> w_lk(w_mu_);

  // First try to get the previous setting, if 'ref == nullptr',
  // this is adding new model setting
  std::shared_ptr<TraceSetting> ref;
  auto it = model_settings_.find(model_name);
  if (it != model_settings_.end()) {
    ref = it->second;
  } else {
    ref = global_setting_;
  }

  // check if there is filename collision and prepare TraceFile object
  std::shared_ptr<TraceFile> file;
  if (!filepath.empty()) {
    auto f_it = trace_files_.find(filepath);
    // There may be a collision, check if the updating setting is the one
    // with the same filename
    if (f_it != trace_files_.end()) {
      // If it is updating the setting and the same file name will be used,
      // then the collision is false alarm.
      if ((ref != nullptr) && (ref->file_->FileName() == filepath)) {
        file = ref->file_;
      } else {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("Trace file name '") + filepath +
             "' has been used by another trace setting")
                .c_str());
      }
    } else {
      file.reset(new TraceFile(filepath));
    }
  }

  // Prepare the updated setting
  std::shared_ptr<TraceSetting> lts(
      new TraceSetting(level, rate, log_frequency, file, *ref));
  // The only invalid setting allowed is if it is turned off explicitly
  if ((!lts->Valid()) &&
      ((level == nullptr) || (*level != TRITONSERVER_TRACE_LEVEL_DISABLED))) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Attempting to set invalid trace setting :") +
         lts->Reason())
            .c_str());
  }

  // Update / Init the setting in read lock to exclude reader access
  {
    std::lock_guard<std::mutex> r_lk(r_mu_);
    if (model_name.empty()) {
      // global update
      trace_files_.erase(global_setting_->file_->FileName());
      global_setting_ = std::move(lts);
    } else {
      auto it = model_settings_.find(model_name);
      if (it != model_settings_.end()) {
        // Model update
        trace_files_.erase(it->second->file_->FileName());
        it->second = std::move(lts);
      } else {
        // Model init
        model_settings_.emplace(model_name, std::move(lts));
      }
    }
  }

  trace_files_.emplace(file->FileName());
  return nullptr;
}

void
TraceManager::GetTraceSetting(
    const std::string& model_name, TRITONSERVER_InferenceTraceLevel* level,
    uint32_t* rate, uint32_t* log_frequency, std::string* filepath)
{
  std::shared_ptr<TraceSetting> trace_setting;
  {
    std::lock_guard<std::mutex> r_lk(r_mu_);
    auto m_it = model_settings_.find(model_name);
    trace_setting =
        (m_it == model_settings_.end()) ? global_setting_ : m_it->second;
  }

  *level = trace_setting->level_;
  *rate = trace_setting->rate_;
  *log_frequency = trace_setting->log_frequency_;
  *filepath = trace_setting->file_->FileName();
}

std::unique_ptr<TraceManager::Trace>
TraceManager::SampleTrace(const std::string& model_name)
{
  std::shared_ptr<TraceSetting> trace_setting;
  {
    std::lock_guard<std::mutex> r_lk(r_mu_);
    auto m_it = model_settings_.find(model_name);
    trace_setting =
        (m_it == model_settings_.end()) ? global_setting_ : m_it->second;
  }
  if (!trace_setting->Valid()) {
    return nullptr;
  }
  std::unique_ptr<Trace> ts = trace_setting->SampleTrace();
  if (ts != nullptr) {
    ts->setting_ = trace_setting;
  }
  return ts;
}

TraceManager::Trace::~Trace()
{
  // Write trace now
  setting_->WriteTrace(streams_);
}

void
TraceManager::Trace::CaptureTimestamp(
    const std::string& name, uint64_t timestamp_ns)
{
  if (setting_->level_ & TRITONSERVER_TRACE_LEVEL_TIMESTAMPS) {
    std::lock_guard<std::mutex> lk(mtx_);
    std::stringstream* ss = nullptr;
    {
      if (streams_.find(trace_id_) == streams_.end()) {
        std::unique_ptr<std::stringstream> stream(new std::stringstream());
        ss = stream.get();
        streams_.emplace(trace_id_, std::move(stream));
      } else {
        ss = streams_[trace_id_].get();
        // If the string stream is not newly created, add "," as there is
        // already content in the string stream
        *ss << ",";
      }
    }
    *ss << "{\"id\":" << trace_id_ << ",\"timestamps\":["
        << "{\"name\":\"" << name << "\",\"ns\":" << timestamp_ns << "}]}";
  }
}

void
TraceManager::TraceRelease(TRITONSERVER_InferenceTrace* trace, void* userp)
{
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
  auto ts = reinterpret_cast<Trace*>(userp);

  std::lock_guard<std::mutex> lk(ts->mtx_);
  std::stringstream* ss = nullptr;
  {
    if (ts->streams_.find(id) == ts->streams_.end()) {
      std::unique_ptr<std::stringstream> stream(new std::stringstream());
      ss = stream.get();
      ts->streams_.emplace(id, std::move(stream));
    } else {
      ss = ts->streams_[id].get();
      // If the string stream is not newly created, add "," as there is
      // already content in the string stream
      *ss << ",";
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
    *ss << "},";
  }

  *ss << "{\"id\":" << id << ",\"timestamps\":["
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
  auto ts = reinterpret_cast<Trace*>(userp);

  std::lock_guard<std::mutex> lk(ts->mtx_);
  std::stringstream* ss = nullptr;
  {
    if (ts->streams_.find(id) == ts->streams_.end()) {
      std::unique_ptr<std::stringstream> stream(new std::stringstream());
      ss = stream.get();
      ts->streams_.emplace(id, std::move(stream));
    } else {
      ss = ts->streams_[id].get();
      // If the string stream is not newly created, add "," as there is
      // already content in the string stream
      *ss << ",";
    }
  }

  // collect and serialize trace details.
  *ss << "{\"id\":" << id << ",\"activity\":\""
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

TraceManager::TraceFile::~TraceFile()
{
  if (!first_write_) {
    trace_file_ << "]";
  }
}

void
TraceManager::TraceFile::SaveTraces(
    std::stringstream& trace_stream, const bool to_index_file)
{
  if (to_index_file) {
    std::string file_name = file_name_ + std::to_string(index_.fetch_add(1));
    // Don't need lock because unique index ensure exclusive access
    // [WIP] exception handling
    std::ofstream file_stream;
    file_stream.open(file_name);
    file_stream << "[";
    file_stream << trace_stream.rdbuf();
    file_stream << "]";
  } else {
    std::lock_guard<std::mutex> lock(mu_);
    if (first_write_) {
      // [FIXME] may raise exception so need to catch it
      trace_file_.open(file_name_);
      trace_file_ << "[";
      first_write_ = false;
    } else {
      trace_file_ << ",";
    }
    trace_file_ << trace_stream.rdbuf();
  }
}

std::unique_ptr<TraceManager::Trace>
TraceManager::TraceSetting::SampleTrace()
{
  bool create_trace = false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    create_trace = (((++sample_) % rate_) == 0);
  }
  if (create_trace) {
    std::unique_ptr<TraceManager::Trace> lts(new Trace());
    TRITONSERVER_InferenceTrace* trace;
    TRITONSERVER_Error* err = TRITONSERVER_InferenceTraceTensorNew(
        &trace, level_, 0 /* parent_id */, TraceActivity, TraceTensorActivity,
        TraceRelease, lts.get());
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "creating inference trace object");
      return nullptr;
    }
    lts->trace_.reset(trace);
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceTraceId(trace, &lts->trace_id_),
        "getting trace id");
    return lts;
  }
  return nullptr;
}

void
TraceManager::TraceSetting::WriteTrace(
    const std::unordered_map<uint64_t, std::unique_ptr<std::stringstream>>&
        streams)
{
  std::unique_lock<std::mutex> lock(mu_);

  if (count_ != 0) {
    trace_stream_ << ",";
  }
  count_++;

  size_t stream_count = 0;
  for (const auto& stream : streams) {
    trace_stream_ << stream.second->rdbuf();
    // Need to add ',' unless it is the last trace in the group
    ++stream_count;
    if (stream_count != streams.size()) {
      trace_stream_ << ",";
    }
  }
  if ((log_frequency_ != 0) && (count_ >= log_frequency_)) {
    // Reset variables and release lock before saving to file
    count_ = 0;
    std::stringstream stream;
    trace_stream_.swap(stream);
    lock.unlock();

    file_->SaveTraces(stream, true /* to_index_file */);
  }
}

TraceManager::TraceSetting::TraceSetting(
    const TRITONSERVER_InferenceTraceLevel level, const uint32_t rate,
    const uint32_t log_frequency, const std::shared_ptr<TraceFile>& file)
    : level_(level), rate_(rate), log_frequency_(log_frequency), file_(file),
      sample_(0), count_(0)
{
  if (level_ == TRITONSERVER_TRACE_LEVEL_DISABLED) {
    invalid_reason_ = "tracing is disabled";
  } else if (rate_ == 0) {
    invalid_reason_ = "sample rate must be non-zero";
  } else if (file_->FileName().empty()) {
    invalid_reason_ = "trace file name is not given";
  }
}

TraceManager::TraceSetting::TraceSetting(
    const TRITONSERVER_InferenceTraceLevel* level, const uint32_t* rate,
    const uint32_t* log_frequency, const std::shared_ptr<TraceFile>& file,
    const TraceSetting& ref)
    : sample_(0), count_(0)
{
  level_ = (level == nullptr) ? ref.level_ : *level;
  rate_ = (rate == nullptr) ? ref.rate_ : *rate;
  log_frequency_ =
      (log_frequency == nullptr) ? ref.log_frequency_ : *log_frequency;
  file_ = (file == nullptr) ? ref.file_ : file;
  if (level_ == TRITONSERVER_TRACE_LEVEL_DISABLED) {
    invalid_reason_ = "tracing is disabled";
  } else if (rate_ == 0) {
    invalid_reason_ = "sample rate must be non-zero";
  } else if (file_->FileName().empty()) {
    invalid_reason_ = "trace file name is not given";
  }
}

TraceManager::TraceSetting::~TraceSetting()
{
  // If log frequency is set, should log the remaining traces to indexed file.
  if (count_ != 0) {
    file_->SaveTraces(trace_stream_, (log_frequency_ != 0));
  }
}

}}  // namespace nvidia::inferenceserver
