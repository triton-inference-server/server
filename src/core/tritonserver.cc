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

#include "src/core/tritonserver.h"

#include <google/protobuf/util/json_util.h>
#include <string>
#include <vector>
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "src/core/backend.h"
#include "src/core/infer_request.h"
#include "src/core/logging.h"
#include "src/core/metrics.h"
#include "src/core/model_config_utils.h"
#include "src/core/nvtx.h"
#include "src/core/server.h"
#include "src/core/status.h"
#include "src/core/tracing.h"

namespace ni = nvidia::inferenceserver;

namespace {

void
SetDurationStats(
    const nvidia::inferenceserver::StatDuration& stat,
    rapidjson::MemoryPoolAllocator<>& allocator,
    rapidjson::Value* duration_stat)
{
  duration_stat->SetObject();
  duration_stat->AddMember(
      "count", rapidjson::Value(stat.count()).Move(), allocator);
  duration_stat->AddMember(
      "ns", rapidjson::Value(stat.total_time_ns()).Move(), allocator);
}

//
// TritonServerError
//
// Implementation for TRITONSERVER_Error.
//
class TritonServerError {
 public:
  static TRITONSERVER_Error* Create(
      TRITONSERVER_Error_Code code, const char* msg);
  static TRITONSERVER_Error* Create(
      TRITONSERVER_Error_Code code, const std::string& msg);
  static TRITONSERVER_Error* Create(const ni::Status& status);

  TRITONSERVER_Error_Code Code() const { return code_; }
  const std::string& Message() const { return msg_; }

 private:
  TritonServerError(TRITONSERVER_Error_Code code, const std::string& msg)
      : code_(code), msg_(msg)
  {
  }
  TritonServerError(TRITONSERVER_Error_Code code, const char* msg)
      : code_(code), msg_(msg)
  {
  }

  TRITONSERVER_Error_Code code_;
  const std::string msg_;
};

TRITONSERVER_Error*
TritonServerError::Create(TRITONSERVER_Error_Code code, const char* msg)
{
  return reinterpret_cast<TRITONSERVER_Error*>(
      new TritonServerError(code, msg));
}

TRITONSERVER_Error*
TritonServerError::Create(TRITONSERVER_Error_Code code, const std::string& msg)
{
  return reinterpret_cast<TRITONSERVER_Error*>(
      new TritonServerError(code, msg));
}

TRITONSERVER_Error*
TritonServerError::Create(const ni::Status& status)
{
  // If 'status' is success then return nullptr as that indicates
  // success
  if (status.IsOk()) {
    return nullptr;
  }

  return Create(
      StatusCodeToTritonServerCode(status.StatusCode()), status.Message());
}

#define RETURN_IF_STATUS_ERROR(S)                 \
  do {                                            \
    const ni::Status& status__ = (S);             \
    if (!status__.IsOk()) {                       \
      return TritonServerError::Create(status__); \
    }                                             \
  } while (false)

//
// TritonServerResponseAllocator
//
// Implementation for TRITONSERVER_ResponseAllocator.
//
class TritonServerResponseAllocator {
 public:
  explicit TritonServerResponseAllocator(
      TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn,
      TRITONSERVER_ResponseAllocatorReleaseFn_t release_fn);

  TRITONSERVER_ResponseAllocatorAllocFn_t AllocFn() const { return alloc_fn_; }
  TRITONSERVER_ResponseAllocatorReleaseFn_t ReleaseFn() const
  {
    return release_fn_;
  }

 private:
  TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn_;
  TRITONSERVER_ResponseAllocatorReleaseFn_t release_fn_;
};

TritonServerResponseAllocator::TritonServerResponseAllocator(
    TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn,
    TRITONSERVER_ResponseAllocatorReleaseFn_t release_fn)
    : alloc_fn_(alloc_fn), release_fn_(release_fn)
{
}

//
// TritonServerMessage
//
// Implementation for TRITONSERVER_Message.
//
class TritonServerMessage {
 public:
  TritonServerMessage(const rapidjson::Document& msg);
  void Serialize(const char** base, size_t* byte_size) const;

 private:
  rapidjson::StringBuffer serialized_;
};

TritonServerMessage::TritonServerMessage(const rapidjson::Document& msg)
{
  serialized_.Clear();
  rapidjson::Writer<rapidjson::StringBuffer> writer(serialized_);
  msg.Accept(writer);
}

void
TritonServerMessage::Serialize(const char** base, size_t* byte_size) const
{
  *base = serialized_.GetString();
  *byte_size = serialized_.GetSize();
}

//
// TritonServerMetrics
//
// Implementation for TRITONSERVER_Metrics.
//
class TritonServerMetrics {
 public:
  TritonServerMetrics() = default;
  TRITONSERVER_Error* Serialize(const char** base, size_t* byte_size);

 private:
  std::string serialized_;
};

TRITONSERVER_Error*
TritonServerMetrics::Serialize(const char** base, size_t* byte_size)
{
#ifdef TRTIS_ENABLE_METRICS
  serialized_ = ni::Metrics::SerializedMetrics();
  *base = serialized_.c_str();
  *byte_size = serialized_.size();
  return nullptr;  // Success
#else
  *base = nullptr;
  *byte_size = 0;
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRTIS_ENABLE_METRICS
}

//
// TritonServerOptions
//
// Implementation for TRITONSERVER_ServerOptions.
//
class TritonServerOptions {
 public:
  TritonServerOptions();

  const std::string& ServerId() const { return server_id_; }
  void SetServerId(const char* id) { server_id_ = id; }

  const std::set<std::string>& ModelRepositoryPaths() const
  {
    return repo_paths_;
  }
  void SetModelRepositoryPath(const char* p) { repo_paths_.insert(p); }

  ni::ModelControlMode ModelControlMode() const { return model_control_mode_; }
  void SetModelControlMode(ni::ModelControlMode m) { model_control_mode_ = m; }

  const std::set<std::string>& StartupModels() const { return models_; }
  void SetStartupModel(const char* m) { models_.insert(m); }

  bool ExitOnError() const { return exit_on_error_; }
  void SetExitOnError(bool b) { exit_on_error_ = b; }

  bool StrictModelConfig() const { return strict_model_config_; }
  void SetStrictModelConfig(bool b) { strict_model_config_ = b; }

  uint64_t PinnedMemoryPoolByteSize() const { return pinned_memory_pool_size_; }
  void SetPinnedMemoryPoolByteSize(uint64_t s) { pinned_memory_pool_size_ = s; }


  const std::map<int, uint64_t>& CudaMemoryPoolByteSize() const
  {
    return cuda_memory_pool_size_;
  }
  void SetCudaMemoryPoolByteSize(int id, uint64_t s)
  {
    cuda_memory_pool_size_[id] = s;
  }

  double MinSupportedComputeCapability() const
  {
    return min_compute_capability_;
  }
  void SetMinSupportedComputeCapability(double c)
  {
    min_compute_capability_ = c;
  }

  bool StrictReadiness() const { return strict_readiness_; }
  void SetStrictReadiness(bool b) { strict_readiness_ = b; }

  unsigned int ExitTimeout() const { return exit_timeout_; }
  void SetExitTimeout(unsigned int t) { exit_timeout_ = t; }

  bool Metrics() const { return metrics_; }
  void SetMetrics(bool b) { metrics_ = b; }

  bool GpuMetrics() const { return gpu_metrics_; }
  void SetGpuMetrics(bool b) { gpu_metrics_ = b; }

  bool TensorFlowSoftPlacement() const { return tf_soft_placement_; }
  void SetTensorFlowSoftPlacement(bool b) { tf_soft_placement_ = b; }

  float TensorFlowGpuMemoryFraction() const { return tf_gpu_mem_fraction_; }
  void SetTensorFlowGpuMemoryFraction(float f) { tf_gpu_mem_fraction_ = f; }

  const std::map<int, std::pair<int, uint64_t>>& TensorFlowVgpuMemoryLimits()
      const
  {
    return tf_vgpu_memory_limits_;
  }
  void AddTensorFlowVgpuMemoryLimits(
      int gpu_device, int num_vgpus, uint64_t per_vgpu_memory_mbytes)
  {
    tf_vgpu_memory_limits_[gpu_device] =
        std::make_pair(num_vgpus, per_vgpu_memory_mbytes);
  }

 private:
  std::string server_id_;
  uint32_t server_protocol_version_;
  std::set<std::string> repo_paths_;
  ni::ModelControlMode model_control_mode_;
  std::set<std::string> models_;
  bool exit_on_error_;
  bool strict_model_config_;
  bool strict_readiness_;
  bool metrics_;
  bool gpu_metrics_;
  unsigned int exit_timeout_;
  uint64_t pinned_memory_pool_size_;
  std::map<int, uint64_t> cuda_memory_pool_size_;
  double min_compute_capability_;

  bool tf_soft_placement_;
  float tf_gpu_mem_fraction_;
  std::map<int, std::pair<int, uint64_t>> tf_vgpu_memory_limits_;
};

TritonServerOptions::TritonServerOptions()
    : server_id_("inference:0"), server_protocol_version_(1),
      model_control_mode_(ni::MODE_POLL), exit_on_error_(true),
      strict_model_config_(true), strict_readiness_(true), metrics_(true),
      gpu_metrics_(true), exit_timeout_(30), pinned_memory_pool_size_(1 << 28),
#ifdef TRTIS_ENABLE_GPU
      min_compute_capability_(TRTIS_MIN_COMPUTE_CAPABILITY),
#else
      min_compute_capability_(0),
#endif  // TRTIS_ENABLE_GPU
      tf_soft_placement_(true), tf_gpu_mem_fraction_(0)
{
#ifndef TRTIS_ENABLE_METRICS
  metrics_ = false;
  gpu_metrics_ = false;
#endif  // TRTIS_ENABLE_METRICS

#ifndef TRTIS_ENABLE_METRICS_GPU
  gpu_metrics_ = false;
#endif  // TRTIS_ENABLE_METRICS_GPU
}

//
// TritonInferenceRequest
//
class TritonInferenceRequest {
 public:
  TritonInferenceRequest(
      const std::shared_ptr<ni::InferenceBackend>& backend,
      ni::InferenceRequest* request)
      : backend_(backend), request_(request), status_(ni::Status::Success)
  {
  }

  const std::shared_ptr<ni::InferenceBackend>& Backend() const
  {
    return backend_;
  }

  const std::shared_ptr<ni::InferenceRequest>& Request() const
  {
    return request_;
  }

  const ni::Status& RequestStatus() const { return status_; }
  void SetRequestStatus(const ni::Status& s) { status_ = s; }

  const std::shared_ptr<ni::InferResponseProvider>& Response() const
  {
    return response_provider_;
  }
  void SetResponse(const std::shared_ptr<ni::InferResponseProvider>& r)
  {
    response_provider_ = r;
  }

 private:
  std::shared_ptr<ni::InferenceBackend> backend_;
  std::shared_ptr<ni::InferenceRequest> request_;
  ni::Status status_;
  std::shared_ptr<ni::InferResponseProvider> response_provider_;
};

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

//
// TRITONSERVER_Error
//
TRITONSERVER_Error*
TRITONSERVER_ErrorNew(TRITONSERVER_Error_Code code, const char* msg)
{
  return reinterpret_cast<TRITONSERVER_Error*>(
      TritonServerError::Create(code, msg));
}

void
TRITONSERVER_ErrorDelete(TRITONSERVER_Error* error)
{
  TritonServerError* lerror = reinterpret_cast<TritonServerError*>(error);
  delete lerror;
}

TRITONSERVER_Error_Code
TRITONSERVER_ErrorCode(TRITONSERVER_Error* error)
{
  TritonServerError* lerror = reinterpret_cast<TritonServerError*>(error);
  return lerror->Code();
}

const char*
TRITONSERVER_ErrorCodeString(TRITONSERVER_Error* error)
{
  TritonServerError* lerror = reinterpret_cast<TritonServerError*>(error);
  return ni::Status::CodeString(
      ni::TritonServerCodeToStatusCode(lerror->Code()));
}

const char*
TRITONSERVER_ErrorMessage(TRITONSERVER_Error* error)
{
  TritonServerError* lerror = reinterpret_cast<TritonServerError*>(error);
  return lerror->Message().c_str();
}

//
// TRITONSERVER_ResponseAllocator
//
TRITONSERVER_Error*
TRITONSERVER_ResponseAllocatorNew(
    TRITONSERVER_ResponseAllocator** allocator,
    TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn,
    TRITONSERVER_ResponseAllocatorReleaseFn_t release_fn)
{
  *allocator = reinterpret_cast<TRITONSERVER_ResponseAllocator*>(
      new TritonServerResponseAllocator(alloc_fn, release_fn));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ResponseAllocatorDelete(TRITONSERVER_ResponseAllocator* allocator)
{
  TritonServerResponseAllocator* lalloc =
      reinterpret_cast<TritonServerResponseAllocator*>(allocator);
  delete lalloc;
  return nullptr;  // Success
}

//
// TRITONSERVER_Message
//
TRITONSERVER_Error*
TRITONSERVER_MessageDelete(TRITONSERVER_Message* message)
{
  TritonServerMessage* lmessage =
      reinterpret_cast<TritonServerMessage*>(message);
  delete lmessage;
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_MessageSerializeToJson(
    TRITONSERVER_Message* protobuf, const char** base, size_t* byte_size)
{
  TritonServerMessage* lprotobuf =
      reinterpret_cast<TritonServerMessage*>(protobuf);
  lprotobuf->Serialize(base, byte_size);
  return nullptr;  // Success
}

//
// TRITONSERVER_Metrics
//
TRITONSERVER_Error*
TRITONSERVER_MetricsDelete(TRITONSERVER_Metrics* metrics)
{
  TritonServerMetrics* lmetrics =
      reinterpret_cast<TritonServerMetrics*>(metrics);
  delete lmetrics;
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_MetricsFormatted(
    TRITONSERVER_Metrics* metrics, TRITONSERVER_Metric_Format format,
    const char** base, size_t* byte_size)
{
  TritonServerMetrics* lmetrics =
      reinterpret_cast<TritonServerMetrics*>(metrics);

  switch (format) {
    case TRITONSERVER_METRIC_PROMETHEUS: {
      return lmetrics->Serialize(base, byte_size);
    }

    default:
      break;
  }

  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("unknown metrics format '" + std::to_string(format) + "'")
          .c_str());
}

//
// TRITONSERVER_Trace
//
TRITONSERVER_Error*
TRITONSERVER_TraceNew(
    TRITONSERVER_Trace** trace, TRITONSERVER_Trace_Level level,
    TRITONSERVER_TraceActivityFn_t activity_fn, void* activity_userp)
{
#ifdef TRTIS_ENABLE_TRACING
  std::unique_ptr<ni::Trace> ltrace;
  RETURN_IF_STATUS_ERROR(
      ni::Trace::Create(level, activity_fn, activity_userp, &ltrace));
  *trace = reinterpret_cast<TRITONSERVER_Trace*>(ltrace.release());
  return nullptr;  // Success
#else
  *trace = nullptr;
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "tracing not supported");
#endif  // TRTIS_ENABLE_TRACING
}

TRITONSERVER_Error*
TRITONSERVER_TraceDelete(TRITONSERVER_Trace* trace)
{
#ifdef TRTIS_ENABLE_TRACING
  ni::Trace* ltrace = reinterpret_cast<ni::Trace*>(trace);
  delete ltrace;
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "tracing not supported");
#endif  // TRTIS_ENABLE_TRACING
}

TRITONSERVER_Error*
TRITONSERVER_TraceModelName(TRITONSERVER_Trace* trace, const char** model_name)
{
#ifdef TRTIS_ENABLE_TRACING
  ni::Trace* ltrace = reinterpret_cast<ni::Trace*>(trace);
  *model_name = ltrace->ModelName();
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "tracing not supported");
#endif  // TRTIS_ENABLE_TRACING
}

TRITONSERVER_Error*
TRITONSERVER_TraceModelVersion(
    TRITONSERVER_Trace* trace, int64_t* model_version)
{
#ifdef TRTIS_ENABLE_TRACING
  ni::Trace* ltrace = reinterpret_cast<ni::Trace*>(trace);
  *model_version = ltrace->ModelVersion();
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "tracing not supported");
#endif  // TRTIS_ENABLE_TRACING
}

TRITONSERVER_Error*
TRITONSERVER_TraceId(TRITONSERVER_Trace* trace, int64_t* id)
{
#ifdef TRTIS_ENABLE_TRACING
  ni::Trace* ltrace = reinterpret_cast<ni::Trace*>(trace);
  *id = ltrace->Id();
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "tracing not supported");
#endif  // TRTIS_ENABLE_TRACING
}

TRITONSERVER_Error*
TRITONSERVER_TraceParentId(TRITONSERVER_Trace* trace, int64_t* parent_id)
{
#ifdef TRTIS_ENABLE_TRACING
  ni::Trace* ltrace = reinterpret_cast<ni::Trace*>(trace);
  *parent_id = ltrace->ParentId();
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "tracing not supported");
#endif  // TRTIS_ENABLE_TRACING
}

TRITONSERVER_Error*
TRITONSERVER_TraceManagerNew(
    TRITONSERVER_TraceManager** trace_manager,
    TRITONSERVER_TraceManagerCreateTraceFn_t create_fn,
    TRITONSERVER_TraceManagerReleaseTraceFn_t release_fn, void* userp)
{
#ifdef TRTIS_ENABLE_TRACING
  std::unique_ptr<ni::OpaqueTraceManager> ltrace_manager(
      new ni::OpaqueTraceManager);
  ltrace_manager->triton_create_fn_ = create_fn;
  ltrace_manager->triton_release_fn_ = release_fn;
  ltrace_manager->using_triton_ = true;
  ltrace_manager->userp_ = userp;
  *trace_manager =
      reinterpret_cast<TRITONSERVER_TraceManager*>(ltrace_manager.release());
  return nullptr;  // Success
#else
  *trace_manager = nullptr;
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "tracing not supported");
#endif  // TRTIS_ENABLE_TRACING
}

TRITONSERVER_Error*
TRITONSERVER_TraceManagerDelete(TRITONSERVER_TraceManager* trace_manager)
{
#ifdef TRTIS_ENABLE_TRACING
  ni::OpaqueTraceManager* ltrace_manager =
      reinterpret_cast<ni::OpaqueTraceManager*>(trace_manager);
  delete ltrace_manager;
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "tracing not supported");
#endif  // TRTIS_ENABLE_TRACING
}

//
// TRITONSERVER_ServerOptions
//
TRITONSERVER_Error*
TRITONSERVER_ServerOptionsNew(TRITONSERVER_ServerOptions** options)
{
  *options =
      reinterpret_cast<TRITONSERVER_ServerOptions*>(new TritonServerOptions());
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsDelete(TRITONSERVER_ServerOptions* options)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  delete loptions;
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetServerId(
    TRITONSERVER_ServerOptions* options, const char* server_id)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetServerId(server_id);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetModelRepositoryPath(
    TRITONSERVER_ServerOptions* options, const char* model_repository_path)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetModelRepositoryPath(model_repository_path);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetModelControlMode(
    TRITONSERVER_ServerOptions* options, TRITONSERVER_Model_Control_Mode mode)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);

  // convert mode from TRITONSERVER_ to nvidia::inferenceserver
  switch (mode) {
    case TRITONSERVER_MODEL_CONTROL_NONE: {
      loptions->SetModelControlMode(ni::MODE_NONE);
      break;
    }
    case TRITONSERVER_MODEL_CONTROL_POLL: {
      loptions->SetModelControlMode(ni::MODE_POLL);
      break;
    }
    case TRITONSERVER_MODEL_CONTROL_EXPLICIT: {
      loptions->SetModelControlMode(ni::MODE_EXPLICIT);
      break;
    }
    default: {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string("unknown control mode '" + std::to_string(mode) + "'")
              .c_str());
    }
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetStartupModel(
    TRITONSERVER_ServerOptions* options, const char* model_name)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetStartupModel(model_name);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetExitOnError(
    TRITONSERVER_ServerOptions* options, bool exit)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetExitOnError(exit);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetStrictModelConfig(
    TRITONSERVER_ServerOptions* options, bool strict)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetStrictModelConfig(strict);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(
    TRITONSERVER_ServerOptions* options, uint64_t size)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetPinnedMemoryPoolByteSize(size);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize(
    TRITONSERVER_ServerOptions* options, int gpu_device, uint64_t size)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetCudaMemoryPoolByteSize(gpu_device, size);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
    TRITONSERVER_ServerOptions* options, double cc)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetMinSupportedComputeCapability(cc);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetStrictReadiness(
    TRITONSERVER_ServerOptions* options, bool strict)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetStrictReadiness(strict);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetExitTimeout(
    TRITONSERVER_ServerOptions* options, unsigned int timeout)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetExitTimeout(timeout);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetLogInfo(
    TRITONSERVER_ServerOptions* options, bool log)
{
#ifdef TRTIS_ENABLE_LOGGING
  // Logging is global for now...
  LOG_ENABLE_INFO(log);
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "logging not supported");
#endif  // TRTIS_ENABLE_LOGGING
}

// Enable or disable warning level logging.
TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetLogWarn(
    TRITONSERVER_ServerOptions* options, bool log)
{
#ifdef TRTIS_ENABLE_LOGGING
  // Logging is global for now...
  LOG_ENABLE_WARNING(log);
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "logging not supported");
#endif  // TRTIS_ENABLE_LOGGING
}

// Enable or disable error level logging.
TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetLogError(
    TRITONSERVER_ServerOptions* options, bool log)
{
#ifdef TRTIS_ENABLE_LOGGING
  // Logging is global for now...
  LOG_ENABLE_ERROR(log);
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "logging not supported");
#endif  // TRTIS_ENABLE_LOGGING
}

// Set verbose logging level. Level zero disables verbose logging.
TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetLogVerbose(
    TRITONSERVER_ServerOptions* options, int level)
{
#ifdef TRTIS_ENABLE_LOGGING
  // Logging is global for now...
  LOG_SET_VERBOSE(level);
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "logging not supported");
#endif             // TRTIS_ENABLE_LOGGING
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetMetrics(
    TRITONSERVER_ServerOptions* options, bool metrics)
{
#ifdef TRTIS_ENABLE_METRICS
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetMetrics(metrics);
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRTIS_ENABLE_METRICS
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetGpuMetrics(
    TRITONSERVER_ServerOptions* options, bool gpu_metrics)
{
#ifdef TRTIS_ENABLE_METRICS
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetGpuMetrics(gpu_metrics);
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRTIS_ENABLE_METRICS
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetTensorFlowSoftPlacement(
    TRITONSERVER_ServerOptions* options, bool soft_placement)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetTensorFlowSoftPlacement(soft_placement);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetTensorFlowGpuMemoryFraction(
    TRITONSERVER_ServerOptions* options, float fraction)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetTensorFlowGpuMemoryFraction(fraction);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsAddTensorFlowVgpuMemoryLimits(
    TRITONSERVER_ServerOptions* options, int gpu_device, int num_vgpus,
    uint64_t per_vgpu_memory_mbytes)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->AddTensorFlowVgpuMemoryLimits(
      gpu_device, num_vgpus, per_vgpu_memory_mbytes);
  return nullptr;  // Success
}

//
// TRITONSERVER_Server
//
TRITONSERVER_Error*
TRITONSERVER_ServerNew(
    TRITONSERVER_Server** server, TRITONSERVER_ServerOptions* options)
{
  ni::InferenceServer* lserver = new ni::InferenceServer();
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);

  NVTX_INITIALIZE;

#ifdef TRTIS_ENABLE_METRICS_GPU
  if (loptions->Metrics() && loptions->GpuMetrics()) {
    ni::Metrics::EnableGPUMetrics();
  }
#endif  // TRTIS_ENABLE_METRICS_GPU

  lserver->SetId(loptions->ServerId());
  lserver->SetProtocolVersion(2 /* protocol_version */);
  lserver->SetModelRepositoryPaths(loptions->ModelRepositoryPaths());
  lserver->SetModelControlMode(loptions->ModelControlMode());
  lserver->SetStartupModels(loptions->StartupModels());
  lserver->SetStrictModelConfigEnabled(loptions->StrictModelConfig());
  lserver->SetPinnedMemoryPoolByteSize(loptions->PinnedMemoryPoolByteSize());
  lserver->SetCudaMemoryPoolByteSize(loptions->CudaMemoryPoolByteSize());
  lserver->SetMinSupportedComputeCapability(
      loptions->MinSupportedComputeCapability());
  lserver->SetStrictReadinessEnabled(loptions->StrictReadiness());
  lserver->SetExitTimeoutSeconds(loptions->ExitTimeout());
  lserver->SetTensorFlowSoftPlacementEnabled(
      loptions->TensorFlowSoftPlacement());
  lserver->SetTensorFlowGPUMemoryFraction(
      loptions->TensorFlowGpuMemoryFraction());
  lserver->SetTensorFlowVGPUMemoryLimits(
      loptions->TensorFlowVgpuMemoryLimits());

  ni::Status status = lserver->Init();
  if (!status.IsOk()) {
    if (loptions->ExitOnError()) {
      delete lserver;
      RETURN_IF_STATUS_ERROR(status);
    }

    LOG_ERROR << status.AsString();
  }

  *server = reinterpret_cast<TRITONSERVER_Server*>(lserver);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerDelete(TRITONSERVER_Server* server)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  if (lserver != nullptr) {
    RETURN_IF_STATUS_ERROR(lserver->Stop());
  }
  delete lserver;
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerStop(TRITONSERVER_Server* server)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  if (lserver != nullptr) {
    RETURN_IF_STATUS_ERROR(lserver->Stop());
  }
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerPollModelRepository(TRITONSERVER_Server* server)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  RETURN_IF_STATUS_ERROR(lserver->PollModelRepository());
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerIsLive(TRITONSERVER_Server* server, bool* live)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::HEALTH);
#endif  // TRTIS_ENABLE_STATS

  RETURN_IF_STATUS_ERROR(lserver->IsLive(live));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerIsReady(TRITONSERVER_Server* server, bool* ready)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::HEALTH);
#endif  // TRTIS_ENABLE_STATS

  RETURN_IF_STATUS_ERROR(lserver->IsReady(ready));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerModelIsReady(
    TRITONSERVER_Server* server, const char* model_name,
    const char* model_version, bool* ready)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::HEALTH);
#endif  // TRTIS_ENABLE_STATS

  int64_t model_int_version = -1;
  if (model_version != nullptr) {
    RETURN_IF_STATUS_ERROR(
        ni::GetModelVersionFromString(model_version, &model_int_version));
  }

  RETURN_IF_STATUS_ERROR(
      lserver->ModelIsReady(model_name, model_int_version, ready));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerMetadata(
    TRITONSERVER_Server* server, TRITONSERVER_Message** server_metadata)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::STATUS);
#endif  // TRTIS_ENABLE_STATS

  rapidjson::Document metadata;
  metadata.SetObject();
  // Just store string reference in JSON object since it will be serialized to
  // another buffer.
  metadata.AddMember(
      "name", rapidjson::StringRef(lserver->Id().c_str()),
      metadata.GetAllocator());
  metadata.AddMember(
      "version", rapidjson::StringRef(lserver->Version().c_str()),
      metadata.GetAllocator());

  rapidjson::Value extensions(rapidjson::kArrayType);
  const std::vector<const char*>& exts = lserver->Extensions();
  for (const auto ext : exts) {
    extensions.PushBack(rapidjson::StringRef(ext), metadata.GetAllocator());
  }
  metadata.AddMember("extensions", extensions, metadata.GetAllocator());

  *server_metadata = reinterpret_cast<TRITONSERVER_Message*>(
      new TritonServerMessage(metadata));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerModelMetadata(
    TRITONSERVER_Server* server, const char* model_name,
    const char* model_version, TRITONSERVER_Message** model_metadata)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::STATUS);
#endif  // TRTIS_ENABLE_STATS

  int64_t model_version_int = -1;
  if (model_version != nullptr) {
    RETURN_IF_STATUS_ERROR(
        ni::GetModelVersionFromString(model_version, &model_version_int));
  }

  std::shared_ptr<ni::InferenceBackend> backend;
  RETURN_IF_STATUS_ERROR(
      lserver->GetInferenceBackend(model_name, model_version_int, &backend));

  std::vector<int64_t> ready_versions;
  RETURN_IF_STATUS_ERROR(
      lserver->ModelReadyVersions(model_name, &ready_versions));

  rapidjson::Document metadata;
  metadata.SetObject();
  // Just store string reference in JSON object since it will be serialized to
  // another buffer.
  metadata.AddMember(
      "name", rapidjson::StringRef(model_name), metadata.GetAllocator());

  rapidjson::Value versions(rapidjson::kArrayType);
  for (const auto v : ready_versions) {
    auto version_str = std::to_string(v);
    versions.PushBack(
        rapidjson::Value(version_str.c_str(), version_str.size()),
        metadata.GetAllocator());
  }
  metadata.AddMember("versions", versions, metadata.GetAllocator());

  const auto& model_config = backend->Config();
  metadata.AddMember(
      "platform", rapidjson::StringRef(model_config.platform().c_str()),
      metadata.GetAllocator());

  rapidjson::Value inputs(rapidjson::kArrayType);
  for (const auto& io : model_config.input()) {
    rapidjson::Value io_metadata;
    io_metadata.SetObject();
    io_metadata.AddMember(
        "name", rapidjson::StringRef(io.name().c_str()),
        metadata.GetAllocator());
    io_metadata.AddMember(
        "datatype",
        rapidjson::StringRef(ni::DataTypeToProtocolString(io.data_type())),
        metadata.GetAllocator());

    rapidjson::Value io_metadata_shape(rapidjson::kArrayType);
    for (const auto d : io.dims()) {
      io_metadata_shape.PushBack(d, metadata.GetAllocator());
    }
    io_metadata.AddMember("shape", io_metadata_shape, metadata.GetAllocator());

    inputs.PushBack(io_metadata, metadata.GetAllocator());
  }
  metadata.AddMember("inputs", inputs, metadata.GetAllocator());

  rapidjson::Value outputs(rapidjson::kArrayType);
  for (const auto& io : model_config.output()) {
    rapidjson::Value io_metadata;
    io_metadata.SetObject();
    io_metadata.AddMember(
        "name", rapidjson::StringRef(io.name().c_str()),
        metadata.GetAllocator());
    io_metadata.AddMember(
        "datatype",
        rapidjson::StringRef(ni::DataTypeToProtocolString(io.data_type())),
        metadata.GetAllocator());

    rapidjson::Value io_metadata_shape(rapidjson::kArrayType);
    for (const auto d : io.dims()) {
      io_metadata_shape.PushBack(d, metadata.GetAllocator());
    }
    io_metadata.AddMember("shape", io_metadata_shape, metadata.GetAllocator());

    outputs.PushBack(io_metadata, metadata.GetAllocator());
  }
  metadata.AddMember("outputs", outputs, metadata.GetAllocator());

  *model_metadata = reinterpret_cast<TRITONSERVER_Message*>(
      new TritonServerMessage(metadata));
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONSERVER_ServerModelStatistics(
    TRITONSERVER_Server* server, const char* model_name,
    const char* model_version, TRITONSERVER_Message** model_stats)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::STATUS);
#endif  // TRTIS_ENABLE_STATS

  auto model_name_string = std::string(model_name);
  int64_t model_version_int = -1;
  if (model_version != nullptr) {
    RETURN_IF_STATUS_ERROR(
        ni::GetModelVersionFromString(model_version, &model_version_int));
  }

  ni::ServerStatus server_status;
  RETURN_IF_STATUS_ERROR(lserver->GetStatus(&server_status, model_name_string));
  if ((!model_name_string.empty()) &&
      (server_status.model_status().find(model_name_string) ==
       server_status.model_status().end())) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("requested model " + model_name_string + " not found")
            .c_str());
  }


  rapidjson::Document metadata;
  auto& allocator = metadata.GetAllocator();
  metadata.SetObject();

  rapidjson::Value model_stats_json(rapidjson::kArrayType);
  for (const auto& m : server_status.model_status()) {
    std::cout << m.first << std::endl;
    if (model_name_string.empty() ||
        (m.first.compare(model_name_string) == 0)) {
      if ((model_version_int != -1) &&
          (m.second.version_status().find(model_version_int) ==
           m.second.version_status().end())) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "requested model version is not found for the model");
      }

      rapidjson::Value model_stat(rapidjson::kObjectType);
      for (const auto& v : m.second.version_status()) {
        if ((model_version_int == -1) || (v.first == model_version_int)) {
          rapidjson::Value inference_stats(rapidjson::kObjectType);
          const auto& ir = v.second.infer_stats().find(1);
          if (ir == v.second.infer_stats().end()) {
            static nvidia::inferenceserver::StatDuration zero_duration;
            rapidjson::Value duration_stats;
            SetDurationStats(zero_duration, allocator, &duration_stats);
            // Explicit use rapidjson's copy semantics to avoid calling
            // SetDurationStats()
            inference_stats.AddMember(
                "success", rapidjson::Value(duration_stats, allocator),
                allocator);
            inference_stats.AddMember(
                "fail", rapidjson::Value(duration_stats, allocator), allocator);
            inference_stats.AddMember(
                "queue", rapidjson::Value(duration_stats, allocator),
                allocator);
            inference_stats.AddMember(
                "compute_input", rapidjson::Value(duration_stats, allocator),
                allocator);
            inference_stats.AddMember(
                "compute_infer", rapidjson::Value(duration_stats, allocator),
                allocator);
            inference_stats.AddMember(
                "compute_output", rapidjson::Value(duration_stats, allocator),
                allocator);
          } else {
            rapidjson::Value duration_stats;
            SetDurationStats(ir->second.success(), allocator, &duration_stats);
            inference_stats.AddMember("success", duration_stats, allocator);

            SetDurationStats(ir->second.failed(), allocator, &duration_stats);
            inference_stats.AddMember("fail", duration_stats, allocator);

            SetDurationStats(ir->second.queue(), allocator, &duration_stats);
            inference_stats.AddMember("queue", duration_stats, allocator);

            SetDurationStats(
                ir->second.compute_input(), allocator, &duration_stats);
            inference_stats.AddMember(
                "compute_input", duration_stats, allocator);

            SetDurationStats(
                ir->second.compute_infer(), allocator, &duration_stats);
            inference_stats.AddMember(
                "compute_infer", duration_stats, allocator);

            SetDurationStats(
                ir->second.compute_output(), allocator, &duration_stats);
            inference_stats.AddMember(
                "compute_output", duration_stats, allocator);
          }
          rapidjson::Value model_stat(rapidjson::kObjectType);
          rapidjson::Value version_stats(rapidjson::kObjectType);
          auto version_str = std::to_string(v.first);
          model_stat.AddMember(
              "name", rapidjson::Value(m.first.c_str(), allocator).Move(),
              allocator);
          model_stat.AddMember(
              "version",
              rapidjson::Value(version_str.c_str(), allocator).Move(),
              allocator);
          model_stat.AddMember("inference_stats", inference_stats, allocator);
          model_stats_json.PushBack(model_stat, allocator);
        }
      }
    }
  }
  metadata.AddMember("model_stats", model_stats_json, allocator);
  *model_stats = reinterpret_cast<TRITONSERVER_Message*>(
      new TritonServerMessage(metadata));
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONSERVER_ServerModelConfig(
    TRITONSERVER_Server* server, const char* model_name,
    const char* model_version, TRITONSERVER_Message** model_config)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::STATUS);
#endif  // TRTIS_ENABLE_STATS

  int64_t model_version_int = -1;
  if (model_version != nullptr) {
    RETURN_IF_STATUS_ERROR(
        ni::GetModelVersionFromString(model_version, &model_version_int));
  }

  std::shared_ptr<ni::InferenceBackend> backend;
  RETURN_IF_STATUS_ERROR(
      lserver->GetInferenceBackend(model_name, model_version_int, &backend));

  std::string model_config_json;
  ::google::protobuf::util::MessageToJsonString(
      backend->Config(), &model_config_json);

  // Extra copies.. But this simplify TritonServerMessage class
  rapidjson::Document document;
  document.Parse(model_config_json.data(), model_config_json.size());
  if (document.HasParseError()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "failed to parse the request JSON buffer: " +
            std::string(GetParseError_En(document.GetParseError())) + " at " +
            std::to_string(document.GetErrorOffset()))
            .c_str());
  }

  *model_config = reinterpret_cast<TRITONSERVER_Message*>(
      new TritonServerMessage(document));
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONSERVER_ServerModelIndex(
    TRITONSERVER_Server* server, TRITONSERVER_Message** repository_index)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::REPOSITORY);
#endif  // TRTIS_ENABLE_STATS

  ni::ModelRepositoryIndex model_repository_index;
  RETURN_IF_STATUS_ERROR(
      lserver->GetModelRepositoryIndex(&model_repository_index));

  rapidjson::Document repository_index_json(rapidjson::kArrayType);
  for (const auto& model : model_repository_index.models()) {
    rapidjson::Value model_index;
    model_index.SetObject();
    model_index.AddMember(
        "name", rapidjson::StringRef(model.name().c_str()),
        repository_index_json.GetAllocator());
    repository_index_json.PushBack(
        model_index, repository_index_json.GetAllocator());
  }
  *repository_index = reinterpret_cast<TRITONSERVER_Message*>(
      new TritonServerMessage(repository_index_json));

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONSERVER_ServerLoadModel(
    TRITONSERVER_Server* server, const char* model_name)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::MODEL_CONTROL);
#endif  // TRTIS_ENABLE_STATS

  RETURN_IF_STATUS_ERROR(lserver->LoadModel(std::string(model_name)));

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONSERVER_ServerUnloadModel(
    TRITONSERVER_Server* server, const char* model_name)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::MODEL_CONTROL);
#endif  // TRTIS_ENABLE_STATS

  RETURN_IF_STATUS_ERROR(lserver->UnloadModel(std::string(model_name)));

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONSERVER_ServerMetrics(
    TRITONSERVER_Server* server, TRITONSERVER_Metrics** metrics)
{
#ifdef TRTIS_ENABLE_METRICS
  TritonServerMetrics* lmetrics = new TritonServerMetrics();
  *metrics = reinterpret_cast<TRITONSERVER_Metrics*>(lmetrics);
  return nullptr;  // Success
#else
  *metrics = nullptr;
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRTIS_ENABLE_METRICS
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestNew(
    TRITONSERVER_InferenceRequest** inference_request,
    TRITONSERVER_Server* server, const char* model_name,
    const char* model_version)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  int64_t model_int_version = -1;
  if (model_version != nullptr) {
    RETURN_IF_STATUS_ERROR(
        ni::GetModelVersionFromString(model_version, &model_int_version));
  }

  std::shared_ptr<ni::InferenceBackend> backend;
  RETURN_IF_STATUS_ERROR(
      lserver->GetInferenceBackend(model_name, model_int_version, &backend));

  std::unique_ptr<ni::InferenceRequest> request(new ni::InferenceRequest(
      model_name, model_int_version, backend->Version(),
      2 /* protocol_version */));

  *inference_request = reinterpret_cast<TRITONSERVER_InferenceRequest*>(
      new TritonInferenceRequest(backend, request.release()));

  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestDelete(
    TRITONSERVER_InferenceRequest* inference_request)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  delete lrequest;
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestId(
    TRITONSERVER_InferenceRequest* inference_request, const char** id)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  *id = lrequest->Request()->IdStr().c_str();
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetId(
    TRITONSERVER_InferenceRequest* inference_request, const char* id)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  lrequest->Request()->SetIdStr(id);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestFlags(
    TRITONSERVER_InferenceRequest* inference_request, uint32_t* flags)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  *flags = lrequest->Request()->Flags();
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetFlags(
    TRITONSERVER_InferenceRequest* inference_request, uint32_t flags)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  lrequest->Request()->SetFlags(flags);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestCorrelationId(
    TRITONSERVER_InferenceRequest* inference_request, uint64_t* correlation_id)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  *correlation_id = lrequest->Request()->CorrelationId();
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetCorrelationId(
    TRITONSERVER_InferenceRequest* inference_request, uint64_t correlation_id)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  lrequest->Request()->SetCorrelationId(correlation_id);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestPriority(
    TRITONSERVER_InferenceRequest* inference_request, uint32_t* priority)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  *priority = lrequest->Request()->Priority();
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetPriority(
    TRITONSERVER_InferenceRequest* inference_request, uint32_t priority)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  lrequest->Request()->SetPriority(priority);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestTimeoutMicroseconds(
    TRITONSERVER_InferenceRequest* inference_request, uint64_t* timeout_us)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  *timeout_us = lrequest->Request()->TimeoutMicroseconds();
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(
    TRITONSERVER_InferenceRequest* inference_request, uint64_t timeout_us)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  lrequest->Request()->SetTimeoutMicroseconds(timeout_us);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestAddInput(
    TRITONSERVER_InferenceRequest* inference_request, const char* name,
    const char* datatype, const int64_t* shape, uint64_t dim_count)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->Request()->AddOriginalInput(
      name, ni::ProtocolStringToDataType(datatype), shape, dim_count));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveInput(
    TRITONSERVER_InferenceRequest* inference_request, const char* name)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->Request()->RemoveOriginalInput(name));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveAllInputs(
    TRITONSERVER_InferenceRequest* inference_request)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->Request()->RemoveAllOriginalInputs());
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestAppendInputData(
    TRITONSERVER_InferenceRequest* inference_request, const char* name,
    const void* base, size_t byte_size, TRITONSERVER_Memory_Type memory_type,
    int64_t memory_type_id)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);

  ni::InferenceRequest::Input* input;
  RETURN_IF_STATUS_ERROR(
      lrequest->Request()->MutableOriginalInput(name, &input));
  RETURN_IF_STATUS_ERROR(
      input->AppendData(base, byte_size, memory_type, memory_type_id));

  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveAllInputData(
    TRITONSERVER_InferenceRequest* inference_request, const char* name)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);

  ni::InferenceRequest::Input* input;
  RETURN_IF_STATUS_ERROR(
      lrequest->Request()->MutableOriginalInput(name, &input));
  RETURN_IF_STATUS_ERROR(input->RemoveAllData());

  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestAddRequestedOutput(
    TRITONSERVER_InferenceRequest* inference_request, const char* name)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->Request()->AddRequestedOutput(name));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestOutputCount(
    TRITONSERVER_InferenceRequest* inference_request, uint64_t* output_count)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  auto requested_outputs = lrequest->Request()->RequestedOutputs();
  *output_count = requested_outputs.size();

  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestOutputName(
    TRITONSERVER_InferenceRequest* inference_request, uint64_t index,
    const char** name)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  auto iter = lrequest->Request()->RequestedOutputs().begin();
  std::advance(iter, index);
  *name = iter->first.c_str();
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveRequestedOutput(
    TRITONSERVER_InferenceRequest* inference_request, const char* name)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->Request()->RemoveRequestedOutput(name));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveAllRequestedOutputs(
    TRITONSERVER_InferenceRequest* inference_request)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->Request()->RemoveAllRequestedOutputs());
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetRequestedOutputClassificationCount(
    TRITONSERVER_InferenceRequest* inference_request, const char* name,
    uint32_t count)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);

  ni::InferenceRequest::RequestedOutput* requested;
  RETURN_IF_STATUS_ERROR(
      lrequest->Request()->MutableRequestedOutput(name, &requested));
  requested->SetClassificationCount(count);

  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestError(
    TRITONSERVER_InferenceRequest* inference_request)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->RequestStatus());
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestOutputData(
    TRITONSERVER_InferenceRequest* inference_request, const char* name,
    const void** base, size_t* byte_size, TRITONSERVER_Memory_Type* memory_type,
    int64_t* memory_type_id)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->Response()->OutputBufferContents(
      name, base, byte_size, memory_type, memory_type_id));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestOutputDataType(
    TRITONSERVER_InferenceRequest* inference_request, const char* name,
    const char** datatype)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  const auto& response_header = lrequest->Response()->ResponseHeader();
  for (const auto& output : response_header.output()) {
    if (output.name() == name) {
      *datatype = ni::DataTypeToProtocolString(output.data_type());
      return nullptr;  // Success
    }
  }

  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INVALID_ARG, "unknown output");
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestOutputShape(
    TRITONSERVER_InferenceRequest* inference_request, const char* name,
    const int64_t** shape, uint64_t* dim_count)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(
      lrequest->Response()->OutputShape(name, shape, dim_count));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveAllOutputs(
    TRITONSERVER_InferenceRequest* inference_request)
{
  TritonInferenceRequest* lrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  lrequest->SetResponse(nullptr);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerInferAsync(
    TRITONSERVER_Server* server, TRITONSERVER_TraceManager* trace_manager,
    TRITONSERVER_InferenceRequest* inference_request,
    TRITONSERVER_ResponseAllocator* response_allocator,
    void* response_allocator_userp,
    TRITONSERVER_InferenceCompleteFn_t complete_fn, void* complete_userp)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  TritonInferenceRequest* ltrtrequest =
      reinterpret_cast<TritonInferenceRequest*>(inference_request);
  TritonServerResponseAllocator* lresponsealloc =
      reinterpret_cast<TritonServerResponseAllocator*>(response_allocator);

  const auto& lrequest = ltrtrequest->Request();
  const auto& lbackend = ltrtrequest->Backend();

  ltrtrequest->SetResponse(nullptr);
  RETURN_IF_STATUS_ERROR(lrequest->PrepareForInference(*lbackend));

#ifdef TRTIS_ENABLE_STATS
  auto infer_stats = std::make_shared<ni::ModelInferStats>(
      lserver->StatusManager(), lrequest->ModelName());
  infer_stats->CaptureTimestamp(
      ni::ModelInferStats::TimestampKind::kRequestStart);
  infer_stats->SetRequestedVersion(lrequest->RequestedModelVersion());
  infer_stats->SetMetricReporter(lbackend->MetricReporter());
  infer_stats->SetBatchSize(lrequest->BatchSize());
  infer_stats->SetFailed(true);
  infer_stats->SetTraceManager(
      reinterpret_cast<ni::OpaqueTraceManager*>(trace_manager));
  infer_stats->NewTrace();
#else
  auto infer_stats = std::make_shared<ni::ModelInferStats>();
#endif  // TRTIS_ENABLE_STATS

  std::shared_ptr<ni::InferResponseProvider> infer_response_provider;
  {
    std::shared_ptr<ni::InferResponseProvider> del_response_provider;
    RETURN_IF_STATUS_ERROR(ni::InferResponseProvider::Create(
        lrequest, lbackend->GetLabelProvider(), response_allocator,
        lresponsealloc->AllocFn(), response_allocator_userp,
        lresponsealloc->ReleaseFn(), 2 /* protocol_version */,
        &del_response_provider));
    infer_response_provider = std::move(del_response_provider);
  }

  lserver->InferAsync(
      lbackend, lrequest, infer_response_provider, infer_stats,
      [infer_stats, trace_manager, ltrtrequest, infer_response_provider, server,
       complete_fn, complete_userp](const ni::Status& status) mutable {
        if (!status.IsOk()) {
          LOG_VERBOSE(1) << "Infer failed: " << status.Message();
        }

#ifdef TRTIS_ENABLE_STATS
        infer_stats->SetFailed(!status.IsOk());
        infer_stats->CaptureTimestamp(
            ni::ModelInferStats::TimestampKind::kRequestEnd);

        // We must explicitly update the inference stats before
        // sending the response... otherwise it is possible that the
        // client will be able to query the stats after the response
        // is received but before they've been updated for the request
        // (this is especially important for testing).
        infer_stats->Report();
#endif  // TRTIS_ENABLE_STATS

        // FIXMEV2 status should live in InferenceRequest instead of
        // being a callback arg.
        ltrtrequest->SetRequestStatus(status);

        ltrtrequest->SetResponse(infer_response_provider);

        complete_fn(
            server, trace_manager,
            reinterpret_cast<TRITONSERVER_InferenceRequest*>(ltrtrequest),
            complete_userp);
      });

  return nullptr;  // Success
}

#ifdef __cplusplus
}
#endif
