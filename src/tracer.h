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
#pragma once

#include <atomic>
#include <condition_variable>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include "triton/core/tritonserver.h"

namespace triton { namespace server {

//
// Manager for tracing to a file.
//
class TraceManager {
 private:
  class TraceSetting;

 public:
  // The new field values for a setting, 'clear_xxx_' indicates
  // whether to clear the previously specified filed value.
  // If false, 'xxx_' will be used as the new field value.
  // If 'xxx_' is nullptr, the field value will not be updated.
  struct NewSetting {
    NewSetting()
        : clear_level_(false), level_(nullptr), clear_rate_(false),
          rate_(nullptr), clear_count_(false), count_(nullptr),
          clear_log_frequency_(false), log_frequency_(nullptr),
          clear_filepath_(false), filepath_(nullptr)
    {
    }
    bool clear_level_;
    const TRITONSERVER_InferenceTraceLevel* level_;

    bool clear_rate_;
    const uint32_t* rate_;

    bool clear_count_;
    const int32_t* count_;

    bool clear_log_frequency_;
    const uint32_t* log_frequency_;

    bool clear_filepath_;
    const std::string* filepath_;
  };

  struct Trace;
  // Create a trace manager that appends trace information
  // to a specified file as global setting.
  static TRITONSERVER_Error* Create(
      TraceManager** manager, const TRITONSERVER_InferenceTraceLevel level,
      const uint32_t rate, const int32_t count, const uint32_t log_frequency,
      const std::string& filepath);

  ~TraceManager() = default;

  // Return a trace that should be used to collected trace activities
  // for an inference request. Return nullptr if no tracing should occur.
  std::shared_ptr<Trace> SampleTrace(const std::string& model_name);

  // Update global setting if 'model_name' is empty, otherwise, model setting is
  // updated.
  TRITONSERVER_Error* UpdateTraceSetting(
      const std::string& model_name, const NewSetting& new_setting);

  void GetTraceSetting(
      const std::string& model_name, TRITONSERVER_InferenceTraceLevel* level,
      uint32_t* rate, int32_t* count, uint32_t* log_frequency,
      std::string* filepath);

  // Return the current timestamp.
  static uint64_t CaptureTimestamp()
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  static void TraceRelease(TRITONSERVER_InferenceTrace* trace, void* userp);

  struct Trace {
    Trace() : trace_(nullptr), trace_id_(0) {}
    ~Trace();
    std::shared_ptr<TraceSetting> setting_;
    // Group the spawned traces by trace ID for better formatting
    std::mutex mtx_;
    std::unordered_map<uint64_t, std::unique_ptr<std::stringstream>> streams_;
    // Triton trace object that this trace is assosicated with,
    // 'Trace' object does not take ownership of 'trace_'. The caller of
    // SampleTrace() must call TraceManager::TraceRelease() with 'trace_userp_'
    // to properly release the resources if 'trace_' is not passed to a
    // TRITONSERVER_ServerInferAsync() call.
    TRITONSERVER_InferenceTrace* trace_;
    void* trace_userp_;

    uint64_t trace_id_;

    // Capture a timestamp generated outside of triton and associate it
    // with this trace.
    void CaptureTimestamp(const std::string& name, uint64_t timestamp_ns);
  };

 private:
  TraceManager(
      const TRITONSERVER_InferenceTraceLevel level, const uint32_t rate,
      const int32_t count, const uint32_t log_frequency,
      const std::string& filepath);

  static void TraceActivity(
      TRITONSERVER_InferenceTrace* trace,
      TRITONSERVER_InferenceTraceActivity activity, uint64_t timestamp_ns,
      void* userp);

  static void TraceTensorActivity(
      TRITONSERVER_InferenceTrace* trace,
      TRITONSERVER_InferenceTraceActivity activity, const char* name,
      TRITONSERVER_DataType datatype, const void* base, size_t byte_size,
      const int64_t* shape, uint64_t dim_count,
      TRITONSERVER_MemoryType memory_type, int64_t memory_type_id, void* userp);

  // Helper function for UpdateTraceSetting() as recursive update may be needed
  // if global setting is being updated
  TRITONSERVER_Error* UpdateTraceSettingInternal(
      const std::string& model_name, const NewSetting& new_setting);

  class TraceFile {
   public:
    TraceFile(const std::string& file_name)
        : file_name_(file_name), index_(0), first_write_(true)
    {
    }
    ~TraceFile();

    // Save the traces stored in 'trace_stream' into the file. 'to_index_file'
    // specifies whether the file name should be indexed, if true, the traces
    // will be written to 'file_name.index' where index will be incremented
    // every time the traces are written to a file with index. If false, the
    // trace will be written to 'file_name'.
    void SaveTraces(std::stringstream& trace_stream, const bool to_index_file);

    const std::string& FileName() { return file_name_; }

   private:
    const std::string file_name_;
    // The file index for the next index file write.
    std::atomic<uint32_t> index_;

    // Multiple traces may be finished and write to the trace file at the same
    // time
    std::mutex mu_;
    std::ofstream trace_file_;
    bool first_write_;
  };

  class TraceSetting {
   public:
    TraceSetting()
        : level_(TRITONSERVER_TRACE_LEVEL_DISABLED), rate_(0), count_(-1),
          log_frequency_(0), level_specified_(false), rate_specified_(false),
          count_specified_(false), log_frequency_specified_(false),
          filepath_specified_(false), sample_(0), created_(0), collected_(0),
          sample_in_stream_(0)
    {
      invalid_reason_ = "Setting hasn't been initialized";
    }
    TraceSetting(
        const TRITONSERVER_InferenceTraceLevel level, const uint32_t rate,
        const int32_t count, const uint32_t log_frequency,
        const std::shared_ptr<TraceFile>& file, const bool level_specified,
        const bool rate_specified, const bool count_specified,
        const bool log_frequency_specified, const bool filepath_specified);

    ~TraceSetting();

    bool Valid() { return invalid_reason_.empty() && (count_ != 0); }
    const std::string& Reason() { return invalid_reason_; }

    void WriteTrace(
        const std::unordered_map<uint64_t, std::unique_ptr<std::stringstream>>&
            streams);

    std::shared_ptr<Trace> SampleTrace();

    const TRITONSERVER_InferenceTraceLevel level_;
    const uint32_t rate_;
    int32_t count_;
    const uint32_t log_frequency_;
    const std::shared_ptr<TraceFile> file_;

    // Whether the field value is specified or mirror from upper level setting
    const bool level_specified_;
    const bool rate_specified_;
    const bool count_specified_;
    const bool log_frequency_specified_;
    const bool filepath_specified_;

   private:
    std::string invalid_reason_;

    std::mutex mu_;

    // use to sample a trace based on sampling rate.
    uint64_t sample_;

    // use to track the status of trace count feature
    uint64_t created_;
    uint64_t collected_;

    // Tracking traces that haven't been saved to file
    uint32_t sample_in_stream_;
    std::stringstream trace_stream_;
  };

  // Trace settings
  // Note that 'global_default_' doesn't use for actual trace sampling,
  // it is used to revert the field values when clearing fields in
  // 'global_setting_'
  std::unique_ptr<TraceSetting> global_default_;
  std::shared_ptr<TraceSetting> global_setting_;
  std::unordered_map<std::string, std::shared_ptr<TraceSetting>>
      model_settings_;
  // The collection of models that have their own trace setting while
  // some of the fields are mirroring global setting.
  std::set<std::string> fallback_used_models_;

  // The collection of files that are used in trace settings, use to
  // avoid creating duplicate TraceFile objects for the same file path.
  std::unordered_map<std::string, std::weak_ptr<TraceFile>> trace_files_;

  // lock for accessing trace setting. 'w_mu_' for write and
  // 'r_mu_' for read / write
  std::mutex w_mu_;
  std::mutex r_mu_;
};

}}  // namespace triton::server
