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

#include "src/core/trtserver.h"

#include <string>
#include <vector>
#include "src/core/backend.h"
#include "src/core/infer_request.h"
#include "src/core/logging.h"
#include "src/core/metrics.h"
#include "src/core/model_config_utils.h"
#include "src/core/nvtx.h"
#include "src/core/request_status.pb.h"
#include "src/core/server.h"
#include "src/core/server_status.h"
#include "src/core/status.h"
#include "src/core/tracing.h"
#include "src/core/trtserver2.h"

namespace ni = nvidia::inferenceserver;

namespace {

const char*
GetDataTypeProtocolString(const ni::DataType dtype)
{
  switch (dtype) {
    case ni::DataType::TYPE_BOOL:
      return "BOOL";
    case ni::DataType::TYPE_UINT8:
      return "UINT8";
    case ni::DataType::TYPE_UINT16:
      return "UINT16";
    case ni::DataType::TYPE_UINT32:
      return "UINT32";
    case ni::DataType::TYPE_UINT64:
      return "UINT64";
    case ni::DataType::TYPE_INT8:
      return "INT8";
    case ni::DataType::TYPE_INT16:
      return "INT16";
    case ni::DataType::TYPE_INT32:
      return "INT32";
    case ni::DataType::TYPE_INT64:
      return "INT64";
    case ni::DataType::TYPE_FP16:
      return "FP16";
    case ni::DataType::TYPE_FP32:
      return "FP32";
    case ni::DataType::TYPE_FP64:
      return "FP64";
    case ni::DataType::TYPE_STRING:
      return "BYTES";
    default:
      break;
  }

  return "";
}

//
// TrtServerError
//
// Implementation for TRTSERVER_Error.
//
class TrtServerError {
 public:
  static TRTSERVER_Error* Create(TRTSERVER_Error_Code code, const char* msg);
  static TRTSERVER_Error* Create(
      ni::RequestStatusCode status_code, const std::string& msg);
  static TRTSERVER_Error* Create(const ni::Status& status);

  ni::RequestStatusCode StatusCode() const { return status_code_; }
  const std::string& Message() const { return msg_; }

 private:
  TrtServerError(ni::RequestStatusCode status_code, const std::string& msg);
  TrtServerError(ni::RequestStatusCode status_code, const char* msg);

  ni::RequestStatusCode status_code_;
  const std::string msg_;
};

TRTSERVER_Error*
TrtServerError::Create(TRTSERVER_Error_Code code, const char* msg)
{
  return reinterpret_cast<TRTSERVER_Error*>(
      new TrtServerError(ni::TrtServerCodeToRequestStatus(code), msg));
}

TRTSERVER_Error*
TrtServerError::Create(
    ni::RequestStatusCode status_code, const std::string& msg)
{
  // If 'status_code' is success then return nullptr as that indicates
  // success
  if (status_code == ni::RequestStatusCode::SUCCESS) {
    return nullptr;
  }

  return reinterpret_cast<TRTSERVER_Error*>(
      new TrtServerError(status_code, msg));
}

TRTSERVER_Error*
TrtServerError::Create(const ni::Status& status)
{
  return Create(status.Code(), status.Message());
}

TrtServerError::TrtServerError(
    ni::RequestStatusCode status_code, const std::string& msg)
    : status_code_(status_code), msg_(msg)
{
}

TrtServerError::TrtServerError(
    ni::RequestStatusCode status_code, const char* msg)
    : status_code_(status_code), msg_(msg)
{
}

#define RETURN_IF_STATUS_ERROR(S)              \
  do {                                         \
    const ni::Status& status__ = (S);          \
    if (!status__.IsOk()) {                    \
      return TrtServerError::Create(status__); \
    }                                          \
  } while (false)

//
// TrtServerResponseAllocator
//
// Implementation for TRTSERVER_ResponseAllocator.
//
class TrtServerResponseAllocator {
 public:
  explicit TrtServerResponseAllocator(
      TRTSERVER_ResponseAllocatorAllocFn_t alloc_fn,
      TRTSERVER_ResponseAllocatorReleaseFn_t release_fn);

  TRTSERVER_ResponseAllocatorAllocFn_t AllocFn() const { return alloc_fn_; }
  TRTSERVER_ResponseAllocatorReleaseFn_t ReleaseFn() const
  {
    return release_fn_;
  }

 private:
  TRTSERVER_ResponseAllocatorAllocFn_t alloc_fn_;
  TRTSERVER_ResponseAllocatorReleaseFn_t release_fn_;
};

TrtServerResponseAllocator::TrtServerResponseAllocator(
    TRTSERVER_ResponseAllocatorAllocFn_t alloc_fn,
    TRTSERVER_ResponseAllocatorReleaseFn_t release_fn)
    : alloc_fn_(alloc_fn), release_fn_(release_fn)
{
}

//
// TrtServerProtobuf
//
// Implementation for TRTSERVER_Protobuf.
//
class TrtServerProtobuf {
 public:
  TrtServerProtobuf(const google::protobuf::MessageLite& msg);
  void Serialize(const char** base, size_t* byte_size) const;

 private:
  std::string serialized_;
};

TrtServerProtobuf::TrtServerProtobuf(const google::protobuf::MessageLite& msg)
{
  msg.SerializeToString(&serialized_);
}

void
TrtServerProtobuf::Serialize(const char** base, size_t* byte_size) const
{
  *base = serialized_.c_str();
  *byte_size = serialized_.size();
}

//
// TrtServerMetrics
//
// Implementation for TRTSERVER_Metrics.
//
class TrtServerMetrics {
 public:
  TrtServerMetrics() = default;
  TRTSERVER_Error* Serialize(const char** base, size_t* byte_size);

 private:
  std::string serialized_;
};

TRTSERVER_Error*
TrtServerMetrics::Serialize(const char** base, size_t* byte_size)
{
#ifdef TRTIS_ENABLE_METRICS
  serialized_ = ni::Metrics::SerializedMetrics();
  *base = serialized_.c_str();
  *byte_size = serialized_.size();
  return nullptr;  // Success
#else
  *base = nullptr;
  *byte_size = 0;
  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRTIS_ENABLE_METRICS
}

//
// TrtServerOptions
//
// Implementation for TRTSERVER_ServerOptions.
//
class TrtServerOptions {
 public:
  TrtServerOptions();

  const std::string& ServerId() const { return server_id_; }
  void SetServerId(const char* id) { server_id_ = id; }

  uint32_t ServerProtocolVersion() const { return server_protocol_version_; }
  void SetServerProtocolVersion(const uint32_t v)
  {
    server_protocol_version_ = v;
  }

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

TrtServerOptions::TrtServerOptions()
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
// TrtServerRequestOptions
//
// Implementation for TRTSERVER_InferenceRequestOptions.
//
class TrtServerRequestOptions {
 public:
  TrtServerRequestOptions(const char* model_name, int64_t model_version);
  TrtServerRequestOptions(
      const char* model_name, int64_t model_version,
      const std::shared_ptr<ni::InferRequestHeader>& request_header);

  TRTSERVER_Error* SetId(uint64_t id);
#ifdef TRTIS_ENABLE_GRPC_V2
  TRTSERVER_Error* SetIdStr(const char* id);
#endif  // TRTIS_ENABLE_GRPC_V2
  TRTSERVER_Error* SetFlags(uint32_t flags);
  TRTSERVER_Error* SetCorrelationId(uint64_t correlation_id);
  TRTSERVER_Error* SetBatchSize(uint64_t batch_size);
  TRTSERVER_Error* SetPriority(uint32_t priority);
  TRTSERVER_Error* SetTimeoutMicroseconds(uint64_t timeout_us);

  TRTSERVER_Error* AddInput(
      const char* input_name, const int64_t* dims, uint64_t dim_count,
      uint64_t batch_byte_size);
  TRTSERVER_Error* AddOutput(const char* output_name);
  TRTSERVER_Error* AddOutput(const char* output_name, uint32_t count);

  const std::string& ModelName() const { return model_name_; }
  int64_t ModelVersion() const { return model_version_; }
  ni::InferRequestHeader* InferRequestHeader() const;
#ifdef TRTIS_ENABLE_GRPC_V2
  const std::string& IdStr() const { return id_str_; }
#endif  // TRTIS_ENABLE_GRPC_V2

 private:
  const std::string model_name_;
  const int64_t model_version_;

#ifdef TRTIS_ENABLE_GRPC_V2
  std::string id_str_;
#endif  // TRTIS_ENABLE_GRPC_V2

  std::shared_ptr<ni::InferRequestHeader> request_header_;

  std::mutex mtx_;
};

TrtServerRequestOptions::TrtServerRequestOptions(
    const char* model_name, int64_t model_version)
    : model_name_(model_name), model_version_(model_version)
{
  request_header_ = std::make_shared<ni::InferRequestHeader>();
}

TrtServerRequestOptions::TrtServerRequestOptions(
    const char* model_name, int64_t model_version,
    const std::shared_ptr<ni::InferRequestHeader>& request_header)
    : model_name_(model_name), model_version_(model_version),
      request_header_(request_header)
{
}

TRTSERVER_Error*
TrtServerRequestOptions::SetId(uint64_t id)
{
  std::lock_guard<std::mutex> lk(mtx_);
  request_header_->set_id(id);
  return nullptr;  // Success
}

#ifdef TRTIS_ENABLE_GRPC_V2
TRTSERVER_Error*
TrtServerRequestOptions::SetIdStr(const char* id)
{
  std::lock_guard<std::mutex> lk(mtx_);
  id_str_ = id;
  return nullptr;  // Success
}
#endif  // TRTIS_ENABLE_GRPC_V2

TRTSERVER_Error*
TrtServerRequestOptions::SetFlags(uint32_t flags)
{
  std::lock_guard<std::mutex> lk(mtx_);
  request_header_->set_flags(flags);
  return nullptr;  // Success
}

TRTSERVER_Error*
TrtServerRequestOptions::SetCorrelationId(uint64_t correlation_id)
{
  std::lock_guard<std::mutex> lk(mtx_);
  request_header_->set_correlation_id(correlation_id);
  return nullptr;  // Success
}

TRTSERVER_Error*
TrtServerRequestOptions::SetBatchSize(uint64_t batch_size)
{
  std::lock_guard<std::mutex> lk(mtx_);
  request_header_->set_batch_size(batch_size);
  return nullptr;  // Success
}

TRTSERVER_Error*
TrtServerRequestOptions::SetPriority(uint32_t priority)
{
  std::lock_guard<std::mutex> lk(mtx_);
  request_header_->set_priority(priority);
  return nullptr;  // Success
}

TRTSERVER_Error*
TrtServerRequestOptions::SetTimeoutMicroseconds(uint64_t timeout_us)
{
  std::lock_guard<std::mutex> lk(mtx_);
  request_header_->set_timeout_microseconds(timeout_us);
  return nullptr;  // Success
}

TRTSERVER_Error*
TrtServerRequestOptions::AddInput(
    const char* input_name, const int64_t* dims, uint64_t dim_count,
    uint64_t batch_byte_size)
{
  std::lock_guard<std::mutex> lk(mtx_);
  auto rinput = request_header_->add_input();
  rinput->set_name(input_name);
  if (dims != nullptr) {
    for (size_t idx = 0; idx < dim_count; idx++) {
      rinput->add_dims(dims[idx]);
    }
  }
  rinput->set_batch_byte_size(batch_byte_size);
  return nullptr;  // Success
}

TRTSERVER_Error*
TrtServerRequestOptions::AddOutput(const char* output_name)
{
  std::lock_guard<std::mutex> lk(mtx_);
  auto routput = request_header_->add_output();
  routput->set_name(output_name);
  return nullptr;  // Success
}

TRTSERVER_Error*
TrtServerRequestOptions::AddOutput(const char* output_name, uint32_t count)
{
  std::lock_guard<std::mutex> lk(mtx_);
  auto routput = request_header_->add_output();
  routput->set_name(output_name);
  routput->mutable_cls()->set_count(count);
  return nullptr;  // Success
}

ni::InferRequestHeader*
TrtServerRequestOptions::InferRequestHeader() const
{
  return request_header_.get();
}

//
// TrtInferenceRequest
//
class TrtInferenceRequest {
 public:
  TrtInferenceRequest(
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

//
// TrtServerResponse
//
// Implementation for TRTSERVER_InferenceResponse.
//
class TrtServerResponse {
 public:
  TrtServerResponse(
      const ni::Status& infer_status, const std::string& id_str,
      const std::shared_ptr<ni::InferResponseProvider>& provider);
  TRTSERVER_Error* Status() const;
#ifdef TRTIS_ENABLE_GRPC_V2
  const std::string& IdStr() const { return id_str_; }
#endif  // TRTIS_ENABLE_GRPC_V2
  const ni::InferResponseHeader& Header() const;
  TRTSERVER_Error* OutputData(
      const char* name, const void** base, size_t* byte_size,
      TRTSERVER_Memory_Type* memory_type, int64_t* memory_type_id) const;

 private:
  const ni::Status infer_status_;
  const std::string id_str_;
  std::shared_ptr<ni::InferResponseProvider> response_provider_;
};

TrtServerResponse::TrtServerResponse(
    const ni::Status& infer_status, const std::string& id_str,
    const std::shared_ptr<ni::InferResponseProvider>& provider)
    : infer_status_(infer_status), id_str_(id_str), response_provider_(provider)
{
}

TRTSERVER_Error*
TrtServerResponse::Status() const
{
  return TrtServerError::Create(infer_status_);
}

const ni::InferResponseHeader&
TrtServerResponse::Header() const
{
  return response_provider_->ResponseHeader();
}

TRTSERVER_Error*
TrtServerResponse::OutputData(
    const char* name, const void** base, size_t* byte_size,
    TRTSERVER_Memory_Type* memory_type, int64_t* memory_type_id) const
{
  RETURN_IF_STATUS_ERROR(response_provider_->OutputBufferContents(
      name, base, byte_size, memory_type, memory_type_id));
  return nullptr;  // Success
}

//
// TrtServerModelIndex
//
// Implementation for TRTSERVER2_ModelIndex.
//
class TrtServerModelIndex {
 public:
  TrtServerModelIndex(const ni::ModelRepositoryIndex& model_repository_index);
  TRTSERVER_Error* GetModelNames(
      const char* const** models, uint64_t* models_count);

 private:
  ni::ModelRepositoryIndex model_repository_index_;
  std::vector<const char*> index_;
};

TrtServerModelIndex::TrtServerModelIndex(
    const ni::ModelRepositoryIndex& model_repository_index)
    : model_repository_index_(model_repository_index)
{
  {
    for (const auto& model : model_repository_index_.models())
      index_.push_back(model.name().c_str());
  }
}

TRTSERVER_Error*
TrtServerModelIndex::GetModelNames(
    const char* const** models, uint64_t* models_count)
{
  if (index_.empty()) {
    *models_count = 0;
    *models = nullptr;
  } else {
    *models_count = index_.size();
    *models = &(index_[0]);
  }

  return nullptr;
}


}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

//
// TRTSERVER_Error
//
TRTSERVER_Error*
TRTSERVER_ErrorNew(TRTSERVER_Error_Code code, const char* msg)
{
  return reinterpret_cast<TRTSERVER_Error*>(TrtServerError::Create(code, msg));
}

void
TRTSERVER_ErrorDelete(TRTSERVER_Error* error)
{
  TrtServerError* lerror = reinterpret_cast<TrtServerError*>(error);
  delete lerror;
}

TRTSERVER_Error_Code
TRTSERVER_ErrorCode(TRTSERVER_Error* error)
{
  TrtServerError* lerror = reinterpret_cast<TrtServerError*>(error);
  return ni::RequestStatusToTrtServerCode(lerror->StatusCode());
}

const char*
TRTSERVER_ErrorCodeString(TRTSERVER_Error* error)
{
  TrtServerError* lerror = reinterpret_cast<TrtServerError*>(error);
  return ni::RequestStatusCode_Name(lerror->StatusCode()).c_str();
}

const char*
TRTSERVER_ErrorMessage(TRTSERVER_Error* error)
{
  TrtServerError* lerror = reinterpret_cast<TrtServerError*>(error);
  return lerror->Message().c_str();
}

//
// TRTSERVER_ResponseAllocator
//
TRTSERVER_Error*
TRTSERVER_ResponseAllocatorNew(
    TRTSERVER_ResponseAllocator** allocator,
    TRTSERVER_ResponseAllocatorAllocFn_t alloc_fn,
    TRTSERVER_ResponseAllocatorReleaseFn_t release_fn)
{
  *allocator = reinterpret_cast<TRTSERVER_ResponseAllocator*>(
      new TrtServerResponseAllocator(alloc_fn, release_fn));
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ResponseAllocatorDelete(TRTSERVER_ResponseAllocator* allocator)
{
  TrtServerResponseAllocator* lalloc =
      reinterpret_cast<TrtServerResponseAllocator*>(allocator);
  delete lalloc;
  return nullptr;  // Success
}

//
// TRTSERVER_Protobuf
//
TRTSERVER_Error*
TRTSERVER_ProtobufDelete(TRTSERVER_Protobuf* protobuf)
{
  TrtServerProtobuf* lprotobuf = reinterpret_cast<TrtServerProtobuf*>(protobuf);
  delete lprotobuf;
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ProtobufSerialize(
    TRTSERVER_Protobuf* protobuf, const char** base, size_t* byte_size)
{
  TrtServerProtobuf* lprotobuf = reinterpret_cast<TrtServerProtobuf*>(protobuf);
  lprotobuf->Serialize(base, byte_size);
  return nullptr;  // Success
}

//
// TRTSERVER_Metrics
//
TRTSERVER_Error*
TRTSERVER_MetricsDelete(TRTSERVER_Metrics* metrics)
{
  TrtServerMetrics* lmetrics = reinterpret_cast<TrtServerMetrics*>(metrics);
  delete lmetrics;
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_MetricsFormatted(
    TRTSERVER_Metrics* metrics, TRTSERVER_Metric_Format format,
    const char** base, size_t* byte_size)
{
  TrtServerMetrics* lmetrics = reinterpret_cast<TrtServerMetrics*>(metrics);

  switch (format) {
    case TRTSERVER_METRIC_PROMETHEUS: {
      return lmetrics->Serialize(base, byte_size);
    }

    default:
      break;
  }

  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_INVALID_ARG,
      std::string("unknown metrics format '" + std::to_string(format) + "'")
          .c_str());
}

//
// TRTSERVER_Trace
//
TRTSERVER_Error*
TRTSERVER_TraceNew(
    TRTSERVER_Trace** trace, TRTSERVER_Trace_Level level,
    TRTSERVER_TraceActivityFn_t activity_fn, void* activity_userp)
{
#ifdef TRTIS_ENABLE_TRACING
  std::unique_ptr<ni::Trace> ltrace;
  RETURN_IF_STATUS_ERROR(
      ni::Trace::Create(level, activity_fn, activity_userp, &ltrace));
  *trace = reinterpret_cast<TRTSERVER_Trace*>(ltrace.release());
  return nullptr;  // Success
#else
  *trace = nullptr;
  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNSUPPORTED, "tracing not supported");
#endif  // TRTIS_ENABLE_TRACING
}

TRTSERVER_Error*
TRTSERVER_TraceDelete(TRTSERVER_Trace* trace)
{
#ifdef TRTIS_ENABLE_TRACING
  ni::Trace* ltrace = reinterpret_cast<ni::Trace*>(trace);
  delete ltrace;
  return nullptr;  // Success
#else
  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNSUPPORTED, "tracing not supported");
#endif  // TRTIS_ENABLE_TRACING
}

TRTSERVER_Error*
TRTSERVER_TraceModelName(TRTSERVER_Trace* trace, const char** model_name)
{
#ifdef TRTIS_ENABLE_TRACING
  ni::Trace* ltrace = reinterpret_cast<ni::Trace*>(trace);
  *model_name = ltrace->ModelName();
  return nullptr;  // Success
#else
  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNSUPPORTED, "tracing not supported");
#endif  // TRTIS_ENABLE_TRACING
}

TRTSERVER_Error*
TRTSERVER_TraceModelVersion(TRTSERVER_Trace* trace, int64_t* model_version)
{
#ifdef TRTIS_ENABLE_TRACING
  ni::Trace* ltrace = reinterpret_cast<ni::Trace*>(trace);
  *model_version = ltrace->ModelVersion();
  return nullptr;  // Success
#else
  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNSUPPORTED, "tracing not supported");
#endif  // TRTIS_ENABLE_TRACING
}

TRTSERVER_Error*
TRTSERVER_TraceId(TRTSERVER_Trace* trace, int64_t* id)
{
#ifdef TRTIS_ENABLE_TRACING
  ni::Trace* ltrace = reinterpret_cast<ni::Trace*>(trace);
  *id = ltrace->Id();
  return nullptr;  // Success
#else
  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNSUPPORTED, "tracing not supported");
#endif  // TRTIS_ENABLE_TRACING
}

TRTSERVER_Error*
TRTSERVER_TraceParentId(TRTSERVER_Trace* trace, int64_t* parent_id)
{
#ifdef TRTIS_ENABLE_TRACING
  ni::Trace* ltrace = reinterpret_cast<ni::Trace*>(trace);
  *parent_id = ltrace->ParentId();
  return nullptr;  // Success
#else
  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNSUPPORTED, "tracing not supported");
#endif  // TRTIS_ENABLE_TRACING
}

TRTSERVER_Error*
TRTSERVER_TraceManagerNew(
    TRTSERVER_TraceManager** trace_manager,
    TRTSERVER_TraceManagerCreateTraceFn_t create_fn,
    TRTSERVER_TraceManagerReleaseTraceFn_t release_fn, void* userp)
{
#ifdef TRTIS_ENABLE_TRACING
  std::unique_ptr<ni::OpaqueTraceManager> ltrace_manager(
      new ni::OpaqueTraceManager);
  ltrace_manager->create_fn_ = create_fn;
  ltrace_manager->release_fn_ = release_fn;
  ltrace_manager->userp_ = userp;
  *trace_manager =
      reinterpret_cast<TRTSERVER_TraceManager*>(ltrace_manager.release());
  return nullptr;  // Success
#else
  *trace_manager = nullptr;
  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNSUPPORTED, "tracing not supported");
#endif  // TRTIS_ENABLE_TRACING
}

TRTSERVER_Error*
TRTSERVER_TraceManagerDelete(TRTSERVER_TraceManager* trace_manager)
{
#ifdef TRTIS_ENABLE_TRACING
  ni::OpaqueTraceManager* ltrace_manager =
      reinterpret_cast<ni::OpaqueTraceManager*>(trace_manager);
  delete ltrace_manager;
  return nullptr;  // Success
#else
  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNSUPPORTED, "tracing not supported");
#endif  // TRTIS_ENABLE_TRACING
}

//
// TRTSERVER_InferenceRequestOptions
//
TRTSERVER_Error*
TRTSERVER_InferenceRequestOptionsNew(
    TRTSERVER_InferenceRequestOptions** request_options, const char* model_name,
    int64_t model_version)
{
  TrtServerRequestOptions* options =
      new TrtServerRequestOptions(model_name, model_version);
  *request_options =
      reinterpret_cast<TRTSERVER_InferenceRequestOptions*>(options);

  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_InferenceRequestOptionsSetId(
    TRTSERVER_InferenceRequestOptions* request_options, uint64_t id)
{
  TrtServerRequestOptions* loptions =
      reinterpret_cast<TrtServerRequestOptions*>(request_options);
  loptions->SetId(id);
  return nullptr;  // Success
}

#ifdef TRTIS_ENABLE_GRPC_V2
TRTSERVER_Error*
TRTSERVER_InferenceRequestOptionsSetIdStr(
    TRTSERVER_InferenceRequestOptions* request_options, const char* id)
{
  TrtServerRequestOptions* loptions =
      reinterpret_cast<TrtServerRequestOptions*>(request_options);
  loptions->SetIdStr(id);
  return nullptr;  // Success
}
#endif  // TRTIS_ENABLE_GRPC_V2

TRTSERVER_Error*
TRTSERVER_InferenceRequestOptionsSetFlags(
    TRTSERVER_InferenceRequestOptions* request_options, uint32_t flags)
{
  TrtServerRequestOptions* loptions =
      reinterpret_cast<TrtServerRequestOptions*>(request_options);
  loptions->SetFlags(flags);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_InferenceRequestOptionsSetCorrelationId(
    TRTSERVER_InferenceRequestOptions* request_options, uint64_t correlation_id)
{
  TrtServerRequestOptions* loptions =
      reinterpret_cast<TrtServerRequestOptions*>(request_options);
  loptions->SetCorrelationId(correlation_id);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_InferenceRequestOptionsSetBatchSize(
    TRTSERVER_InferenceRequestOptions* request_options, uint32_t batch_size)
{
  TrtServerRequestOptions* loptions =
      reinterpret_cast<TrtServerRequestOptions*>(request_options);
  loptions->SetBatchSize(batch_size);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_InferenceRequestOptionsSetPriority(
    TRTSERVER_InferenceRequestOptions* request_options, uint32_t priority)
{
  TrtServerRequestOptions* loptions =
      reinterpret_cast<TrtServerRequestOptions*>(request_options);
  loptions->SetPriority(priority);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_InferenceRequestOptionsSetTimeoutMicroseconds(
    TRTSERVER_InferenceRequestOptions* request_options, uint64_t timeout_us)
{
  TrtServerRequestOptions* loptions =
      reinterpret_cast<TrtServerRequestOptions*>(request_options);
  loptions->SetTimeoutMicroseconds(timeout_us);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_InferenceRequestOptionsAddInput(
    TRTSERVER_InferenceRequestOptions* request_options, const char* input_name,
    const int64_t* dims, uint64_t dim_count, uint64_t batch_byte_size)
{
  TrtServerRequestOptions* loptions =
      reinterpret_cast<TrtServerRequestOptions*>(request_options);
  loptions->AddInput(input_name, dims, dim_count, batch_byte_size);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_InferenceRequestOptionsAddOutput(
    TRTSERVER_InferenceRequestOptions* request_options, const char* output_name)
{
  TrtServerRequestOptions* loptions =
      reinterpret_cast<TrtServerRequestOptions*>(request_options);
  loptions->AddOutput(output_name);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_InferenceRequestOptionsAddClassificationOutput(
    TRTSERVER_InferenceRequestOptions* request_options, const char* output_name,
    uint32_t count)
{
  TrtServerRequestOptions* loptions =
      reinterpret_cast<TrtServerRequestOptions*>(request_options);
  loptions->AddOutput(output_name, count);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_InferenceRequestOptionsDelete(
    TRTSERVER_InferenceRequestOptions* request_options)
{
  TrtServerRequestOptions* loptions =
      reinterpret_cast<TrtServerRequestOptions*>(request_options);
  delete loptions;
  return nullptr;  // Success
}

//
// TRTSERVER_InferenceRequestProvider
//
TRTSERVER_Error*
TRTSERVER_InferenceRequestProviderNew(
    TRTSERVER_InferenceRequestProvider** request_provider,
    TRTSERVER_Server* server, const char* model_name, int64_t model_version,
    const char* request_header_base, size_t request_header_byte_size)
{
  std::shared_ptr<ni::InferRequestHeader> request_header =
      std::make_shared<ni::InferRequestHeader>();
  if (!request_header->ParseFromArray(
          request_header_base, request_header_byte_size)) {
    return TrtServerError::Create(
        ni::RequestStatusCode::INVALID_ARG,
        "failed to parse InferRequestHeader");
  }

  std::unique_ptr<TrtServerRequestOptions> request_options(
      new TrtServerRequestOptions(model_name, model_version, request_header));

  return TRTSERVER_InferenceRequestProviderNewV2(
      request_provider, server,
      reinterpret_cast<TRTSERVER_InferenceRequestOptions*>(
          request_options.get()));
}

TRTSERVER_Error*
TRTSERVER_InferenceRequestProviderNewV2(
    TRTSERVER_InferenceRequestProvider** request_provider,
    TRTSERVER_Server* server,
    TRTSERVER_InferenceRequestOptions* request_options)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  TrtServerRequestOptions* loptions =
      reinterpret_cast<TrtServerRequestOptions*>(request_options);

  std::shared_ptr<ni::InferenceBackend> backend;
  RETURN_IF_STATUS_ERROR(lserver->GetInferenceBackend(
      loptions->ModelName(), loptions->ModelVersion(), &backend));

  std::unique_ptr<ni::InferenceRequest> request(new ni::InferenceRequest(
      loptions->ModelName(), loptions->ModelVersion(), backend->Version(),
      lserver->ProtocolVersion()));
  request->SetId(loptions->InferRequestHeader()->id());
#ifdef TRTIS_ENABLE_GRPC_V2
  request->SetIdStr(loptions->IdStr());
#endif  // TRTIS_ENABLE_GRPC_V2
  request->SetFlags(loptions->InferRequestHeader()->flags());
  request->SetCorrelationId(loptions->InferRequestHeader()->correlation_id());
  request->SetBatchSize(loptions->InferRequestHeader()->batch_size());
  request->SetPriority(loptions->InferRequestHeader()->priority());
  request->SetTimeoutMicroseconds(
      loptions->InferRequestHeader()->timeout_microseconds());
  for (const auto& io : loptions->InferRequestHeader()->input()) {
    RETURN_IF_STATUS_ERROR(
        request->AddOriginalInput(io.name(), io.dims(), io.batch_byte_size()));
  }

  for (const auto& io : loptions->InferRequestHeader()->output()) {
    uint32_t cls_cnt = io.has_cls() ? io.cls().count() : 0;
    RETURN_IF_STATUS_ERROR(request->AddRequestedOutput(io.name(), cls_cnt));
  }

  RETURN_IF_STATUS_ERROR(request->PrepareForInference(*backend));

  *request_provider = reinterpret_cast<TRTSERVER_InferenceRequestProvider*>(
      new TrtInferenceRequest(backend, request.release()));

  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_InferenceRequestProviderDelete(
    TRTSERVER_InferenceRequestProvider* request_provider)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(request_provider);
  delete lrequest;
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_InferenceRequestProviderInputBatchByteSize(
    TRTSERVER_InferenceRequestProvider* request_provider, const char* name,
    uint64_t* byte_size)
{
  TrtInferenceRequest* ltrtrequest =
      reinterpret_cast<TrtInferenceRequest*>(request_provider);
  const auto& lrequest = ltrtrequest->Request();

  for (const auto& pr : lrequest->OriginalInputs()) {
    if (pr.first == std::string(name)) {
      *byte_size = pr.second.BatchByteSize();
      return nullptr;  // Success
    }
  }

  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_INVALID_ARG,
      std::string(
          "batch byte-size requested for unknown input tensor '" +
          std::string(name) + "', in model '" + lrequest->ModelName() + "'")
          .c_str());
}

TRTSERVER_Error*
TRTSERVER_InferenceRequestProviderSetInputData(
    TRTSERVER_InferenceRequestProvider* request_provider,
    const char* input_name, const void* base, size_t byte_size,
    TRTSERVER_Memory_Type memory_type, int64_t memory_type_id)
{
  TrtInferenceRequest* ltrtrequest =
      reinterpret_cast<TrtInferenceRequest*>(request_provider);
  const auto& lrequest = ltrtrequest->Request();

  ni::InferenceRequest::Input* input;
  RETURN_IF_STATUS_ERROR(lrequest->MutableOriginalInput(input_name, &input));
  input->AppendData(base, byte_size, memory_type, memory_type_id);

  return nullptr;  // Success
}

//
// TRTSERVER_InferenceResponse
//
TRTSERVER_Error*
TRTSERVER_InferenceResponseDelete(TRTSERVER_InferenceResponse* response)
{
  TrtServerResponse* lresponse = reinterpret_cast<TrtServerResponse*>(response);
  delete lresponse;
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_InferenceResponseStatus(TRTSERVER_InferenceResponse* response)
{
  TrtServerResponse* lresponse = reinterpret_cast<TrtServerResponse*>(response);
  return lresponse->Status();
}

#ifdef TRTIS_ENABLE_GRPC_V2
TRTSERVER_Error*
TRTSERVER_InferenceResponseIdStr(
    TRTSERVER_InferenceResponse* response, const char** id)
{
  TrtServerResponse* lresponse = reinterpret_cast<TrtServerResponse*>(response);
  *id = lresponse->IdStr().c_str();
  return nullptr;  // Success
}
#endif  // TRTIS_ENABLE_GRPC_V2

TRTSERVER_Error*
TRTSERVER_InferenceResponseHeader(
    TRTSERVER_InferenceResponse* response, TRTSERVER_Protobuf** header)
{
  TrtServerResponse* lresponse = reinterpret_cast<TrtServerResponse*>(response);
  TRTSERVER_Error* status = lresponse->Status();
  if (status != nullptr) {
    return status;
  }

  TrtServerProtobuf* protobuf = new TrtServerProtobuf(lresponse->Header());
  *header = reinterpret_cast<TRTSERVER_Protobuf*>(protobuf);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_InferenceResponseOutputData(
    TRTSERVER_InferenceResponse* response, const char* name, const void** base,
    size_t* byte_size, TRTSERVER_Memory_Type* memory_type,
    int64_t* memory_type_id)
{
  TrtServerResponse* lresponse = reinterpret_cast<TrtServerResponse*>(response);
  return lresponse->OutputData(
      name, base, byte_size, memory_type, memory_type_id);
}

//
// TRTSERVER_ServerOptions
//
TRTSERVER_Error*
TRTSERVER_ServerOptionsNew(TRTSERVER_ServerOptions** options)
{
  *options = reinterpret_cast<TRTSERVER_ServerOptions*>(new TrtServerOptions());
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerOptionsDelete(TRTSERVER_ServerOptions* options)
{
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);
  delete loptions;
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerOptionsSetServerId(
    TRTSERVER_ServerOptions* options, const char* server_id)
{
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);
  loptions->SetServerId(server_id);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerOptionsSetServerProtocolVersion(
    TRTSERVER_ServerOptions* options, const uint32_t server_protocol_version)
{
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);
  loptions->SetServerProtocolVersion(server_protocol_version);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerOptionsSetModelRepositoryPath(
    TRTSERVER_ServerOptions* options, const char* model_repository_path)
{
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);
  loptions->SetModelRepositoryPath(model_repository_path);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerOptionsSetModelControlMode(
    TRTSERVER_ServerOptions* options, TRTSERVER_Model_Control_Mode mode)
{
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);

  // convert mode from TRTSERVER_ to nvidia::inferenceserver
  switch (mode) {
    case TRTSERVER_MODEL_CONTROL_NONE: {
      loptions->SetModelControlMode(ni::MODE_NONE);
      break;
    }
    case TRTSERVER_MODEL_CONTROL_POLL: {
      loptions->SetModelControlMode(ni::MODE_POLL);
      break;
    }
    case TRTSERVER_MODEL_CONTROL_EXPLICIT: {
      loptions->SetModelControlMode(ni::MODE_EXPLICIT);
      break;
    }
    default: {
      return TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_INVALID_ARG,
          std::string("unknown control mode '" + std::to_string(mode) + "'")
              .c_str());
    }
  }

  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerOptionsSetStartupModel(
    TRTSERVER_ServerOptions* options, const char* model_name)
{
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);
  loptions->SetStartupModel(model_name);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerOptionsSetExitOnError(
    TRTSERVER_ServerOptions* options, bool exit)
{
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);
  loptions->SetExitOnError(exit);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerOptionsSetStrictModelConfig(
    TRTSERVER_ServerOptions* options, bool strict)
{
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);
  loptions->SetStrictModelConfig(strict);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(
    TRTSERVER_ServerOptions* options, uint64_t size)
{
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);
  loptions->SetPinnedMemoryPoolByteSize(size);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerOptionsSetCudaMemoryPoolByteSize(
    TRTSERVER_ServerOptions* options, int gpu_device, uint64_t size)
{
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);
  loptions->SetCudaMemoryPoolByteSize(gpu_device, size);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerOptionsSetMinSupportedComputeCapability(
    TRTSERVER_ServerOptions* options, double cc)
{
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);
  loptions->SetMinSupportedComputeCapability(cc);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerOptionsSetStrictReadiness(
    TRTSERVER_ServerOptions* options, bool strict)
{
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);
  loptions->SetStrictReadiness(strict);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerOptionsSetExitTimeout(
    TRTSERVER_ServerOptions* options, unsigned int timeout)
{
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);
  loptions->SetExitTimeout(timeout);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerOptionsSetLogInfo(TRTSERVER_ServerOptions* options, bool log)
{
#ifdef TRTIS_ENABLE_LOGGING
  // Logging is global for now...
  LOG_ENABLE_INFO(log);
  return nullptr;  // Success
#else
  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNSUPPORTED, "logging not supported");
#endif  // TRTIS_ENABLE_LOGGING
}

// Enable or disable warning level logging.
TRTSERVER_Error*
TRTSERVER_ServerOptionsSetLogWarn(TRTSERVER_ServerOptions* options, bool log)
{
#ifdef TRTIS_ENABLE_LOGGING
  // Logging is global for now...
  LOG_ENABLE_WARNING(log);
  return nullptr;  // Success
#else
  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNSUPPORTED, "logging not supported");
#endif  // TRTIS_ENABLE_LOGGING
}

// Enable or disable error level logging.
TRTSERVER_Error*
TRTSERVER_ServerOptionsSetLogError(TRTSERVER_ServerOptions* options, bool log)
{
#ifdef TRTIS_ENABLE_LOGGING
  // Logging is global for now...
  LOG_ENABLE_ERROR(log);
  return nullptr;  // Success
#else
  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNSUPPORTED, "logging not supported");
#endif  // TRTIS_ENABLE_LOGGING
}

// Set verbose logging level. Level zero disables verbose logging.
TRTSERVER_Error*
TRTSERVER_ServerOptionsSetLogVerbose(
    TRTSERVER_ServerOptions* options, int level)
{
#ifdef TRTIS_ENABLE_LOGGING
  // Logging is global for now...
  LOG_SET_VERBOSE(level);
#else
  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNSUPPORTED, "logging not supported");
#endif             // TRTIS_ENABLE_LOGGING
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerOptionsSetMetrics(
    TRTSERVER_ServerOptions* options, bool metrics)
{
#ifdef TRTIS_ENABLE_METRICS
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);
  loptions->SetMetrics(metrics);
  return nullptr;  // Success
#else
  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRTIS_ENABLE_METRICS
}

TRTSERVER_Error*
TRTSERVER_ServerOptionsSetGpuMetrics(
    TRTSERVER_ServerOptions* options, bool gpu_metrics)
{
#ifdef TRTIS_ENABLE_METRICS
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);
  loptions->SetGpuMetrics(gpu_metrics);
  return nullptr;  // Success
#else
  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRTIS_ENABLE_METRICS
}

TRTSERVER_Error*
TRTSERVER_ServerOptionsSetTensorFlowSoftPlacement(
    TRTSERVER_ServerOptions* options, bool soft_placement)
{
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);
  loptions->SetTensorFlowSoftPlacement(soft_placement);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerOptionsSetTensorFlowGpuMemoryFraction(
    TRTSERVER_ServerOptions* options, float fraction)
{
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);
  loptions->SetTensorFlowGpuMemoryFraction(fraction);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerOptionsAddTensorFlowVgpuMemoryLimits(
    TRTSERVER_ServerOptions* options, int gpu_device, int num_vgpus,
    uint64_t per_vgpu_memory_mbytes)
{
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);
  loptions->AddTensorFlowVgpuMemoryLimits(
      gpu_device, num_vgpus, per_vgpu_memory_mbytes);
  return nullptr;  // Success
}

//
// TRTSERVER_Server
//
TRTSERVER_Error*
TRTSERVER_ServerNew(TRTSERVER_Server** server, TRTSERVER_ServerOptions* options)
{
  ni::InferenceServer* lserver = new ni::InferenceServer();
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);

  NVTX_INITIALIZE;

#ifdef TRTIS_ENABLE_METRICS_GPU
  if (loptions->Metrics() && loptions->GpuMetrics()) {
    ni::Metrics::EnableGPUMetrics();
  }
#endif  // TRTIS_ENABLE_METRICS_GPU

  lserver->SetId(loptions->ServerId());
  lserver->SetProtocolVersion(loptions->ServerProtocolVersion());
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

  *server = reinterpret_cast<TRTSERVER_Server*>(lserver);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerDelete(TRTSERVER_Server* server)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  if (lserver != nullptr) {
    RETURN_IF_STATUS_ERROR(lserver->Stop());
  }
  delete lserver;
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerStop(TRTSERVER_Server* server)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  if (lserver != nullptr) {
    RETURN_IF_STATUS_ERROR(lserver->Stop());
  }
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerId(TRTSERVER_Server* server, const char** id)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  *id = lserver->Id().c_str();
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerVersion(TRTSERVER_Server* server, const char** version)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  *version = lserver->Version().c_str();
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerExtensions(
    TRTSERVER_Server* server, const char* const** extensions,
    uint64_t* extensions_count)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  const std::vector<const char*>& exts = lserver->Extensions();
  if (exts.empty()) {
    *extensions_count = 0;
    *extensions = nullptr;
  } else {
    *extensions_count = exts.size();
    *extensions = &(lserver->Extensions()[0]);
  }

  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerPollModelRepository(TRTSERVER_Server* server)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  RETURN_IF_STATUS_ERROR(lserver->PollModelRepository());
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerIsLive(TRTSERVER_Server* server, bool* live)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::HEALTH);
#endif  // TRTIS_ENABLE_STATS

  RETURN_IF_STATUS_ERROR(lserver->IsLive(live));
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerIsReady(TRTSERVER_Server* server, bool* ready)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::HEALTH);
#endif  // TRTIS_ENABLE_STATS

  RETURN_IF_STATUS_ERROR(lserver->IsReady(ready));
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerStatus(TRTSERVER_Server* server, TRTSERVER_Protobuf** status)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::STATUS);
#endif  // TRTIS_ENABLE_STATS

  ni::ServerStatus server_status;
  RETURN_IF_STATUS_ERROR(lserver->GetStatus(&server_status, std::string()));

  TrtServerProtobuf* protobuf = new TrtServerProtobuf(server_status);
  *status = reinterpret_cast<TRTSERVER_Protobuf*>(protobuf);

  return nullptr;  // success
}

TRTSERVER_Error*
TRTSERVER_ServerModelStatus(
    TRTSERVER_Server* server, const char* model_name,
    TRTSERVER_Protobuf** status)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::STATUS);
#endif  // TRTIS_ENABLE_STATS

  ni::ServerStatus server_status;
  RETURN_IF_STATUS_ERROR(
      lserver->GetStatus(&server_status, std::string(model_name)));

  TrtServerProtobuf* protobuf = new TrtServerProtobuf(server_status);
  *status = reinterpret_cast<TRTSERVER_Protobuf*>(protobuf);

  return nullptr;  // success
}

TRTSERVER_Error*
TRTSERVER_ServerModelRepositoryIndex(
    TRTSERVER_Server* server, TRTSERVER_Protobuf** repository_index)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::REPOSITORY);
#endif  // TRTIS_ENABLE_STATS

  ni::ModelRepositoryIndex model_repository_index;
  RETURN_IF_STATUS_ERROR(
      lserver->GetModelRepositoryIndex(&model_repository_index));

  TrtServerProtobuf* protobuf = new TrtServerProtobuf(model_repository_index);
  *repository_index = reinterpret_cast<TRTSERVER_Protobuf*>(protobuf);

  return nullptr;  // success
}

TRTSERVER_Error*
TRTSERVER2_ServerModelIndex(
    TRTSERVER_Server* server, TRTSERVER2_ModelIndex** model_index)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::REPOSITORY);
#endif  // TRTIS_ENABLE_STATS

  ni::ModelRepositoryIndex model_repository_index;
  RETURN_IF_STATUS_ERROR(
      lserver->GetModelRepositoryIndex(&model_repository_index));

  TrtServerModelIndex* index = new TrtServerModelIndex(model_repository_index);
  *model_index = reinterpret_cast<TRTSERVER2_ModelIndex*>(index);

  return nullptr;  // success
}

TRTSERVER_Error*
TRTSERVER2_ModelIndexNames(
    TRTSERVER2_ModelIndex* model_index, const char* const** models,
    uint64_t* models_count)
{
  TrtServerModelIndex* index =
      reinterpret_cast<TrtServerModelIndex*>(model_index);
  index->GetModelNames(models, models_count);
  return nullptr;
}

TRTSERVER_Error*
TRTSERVER2_ModelIndexDelete(TRTSERVER2_ModelIndex* model_index)
{
  TrtServerModelIndex* index =
      reinterpret_cast<TrtServerModelIndex*>(model_index);
  delete index;
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerLoadModel(TRTSERVER_Server* server, const char* model_name)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::MODEL_CONTROL);
#endif  // TRTIS_ENABLE_STATS

  RETURN_IF_STATUS_ERROR(lserver->LoadModel(std::string(model_name)));

  return nullptr;  // success
}

TRTSERVER_Error*
TRTSERVER_ServerUnloadModel(TRTSERVER_Server* server, const char* model_name)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::MODEL_CONTROL);
#endif  // TRTIS_ENABLE_STATS

  RETURN_IF_STATUS_ERROR(lserver->UnloadModel(std::string(model_name)));

  return nullptr;  // success
}

TRTSERVER_Error*
TRTSERVER_ServerMetrics(TRTSERVER_Server* server, TRTSERVER_Metrics** metrics)
{
#ifdef TRTIS_ENABLE_METRICS
  TrtServerMetrics* lmetrics = new TrtServerMetrics();
  *metrics = reinterpret_cast<TRTSERVER_Metrics*>(lmetrics);
  return nullptr;  // Success
#else
  *metrics = nullptr;
  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRTIS_ENABLE_METRICS
}

TRTSERVER_Error*
TRTSERVER_ServerInferAsync(
    TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
    TRTSERVER_InferenceRequestProvider* request_provider,
    TRTSERVER_ResponseAllocator* response_allocator,
    void* response_allocator_userp, TRTSERVER_InferenceCompleteFn_t complete_fn,
    void* complete_userp)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  TrtInferenceRequest* ltrtrequest =
      reinterpret_cast<TrtInferenceRequest*>(request_provider);
  TrtServerResponseAllocator* lresponsealloc =
      reinterpret_cast<TrtServerResponseAllocator*>(response_allocator);

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

  std::shared_ptr<ni::InferRequestProvider> infer_request_provider;
  RETURN_IF_STATUS_ERROR(
      ni::InferRequestProvider::Create(lrequest, &infer_request_provider));

  std::shared_ptr<ni::InferResponseProvider> infer_response_provider;
  {
    std::shared_ptr<ni::InferResponseProvider> del_response_provider;
    RETURN_IF_STATUS_ERROR(ni::InferResponseProvider::Create(
        lrequest, lbackend->GetLabelProvider(), response_allocator,
        lresponsealloc->AllocFn(), response_allocator_userp,
        lresponsealloc->ReleaseFn(), &del_response_provider));
    infer_response_provider = std::move(del_response_provider);
  }

#ifdef TRTIS_ENABLE_GRPC_V2
  const std::string& id_str = lrequest->IdStr();
#else
  const std::string id_str;
#endif  // TRTIS_ENABLE_GRPC_V2

  lserver->InferAsync(
      lbackend, infer_request_provider, infer_response_provider, infer_stats,
      [infer_stats, id_str, trace_manager, infer_response_provider, server,
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

        TrtServerResponse* response =
            new TrtServerResponse(status, id_str, infer_response_provider);
        complete_fn(
            server, trace_manager,
            reinterpret_cast<TRTSERVER_InferenceResponse*>(response),
            complete_userp);
      });

  return nullptr;  // Success
}


//
// TRTSERVER2
//

TRTSERVER_Error*
TRTSERVER2_InferenceRequestNew(
    TRTSERVER2_InferenceRequest** inference_request, TRTSERVER_Server* server,
    const char* model_name, const char* model_version)
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
      lserver->ProtocolVersion()));

  *inference_request = reinterpret_cast<TRTSERVER2_InferenceRequest*>(
      new TrtInferenceRequest(backend, request.release()));

  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestDelete(
    TRTSERVER2_InferenceRequest* inference_request)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  delete lrequest;
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestId(
    TRTSERVER2_InferenceRequest* inference_request, const char** id)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  *id = lrequest->Request()->IdStr().c_str();
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestSetId(
    TRTSERVER2_InferenceRequest* inference_request, const char* id)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  lrequest->Request()->SetIdStr(id);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestFlags(
    TRTSERVER2_InferenceRequest* inference_request, uint32_t* flags)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  *flags = lrequest->Request()->Flags();
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestSetFlags(
    TRTSERVER2_InferenceRequest* inference_request, uint32_t flags)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  lrequest->Request()->SetFlags(flags);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestCorrelationId(
    TRTSERVER2_InferenceRequest* inference_request, uint64_t* correlation_id)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  *correlation_id = lrequest->Request()->CorrelationId();
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestSetCorrelationId(
    TRTSERVER2_InferenceRequest* inference_request, uint64_t correlation_id)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  lrequest->Request()->SetCorrelationId(correlation_id);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestPriority(
    TRTSERVER2_InferenceRequest* inference_request, uint32_t* priority)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  *priority = lrequest->Request()->Priority();
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestSetPriority(
    TRTSERVER2_InferenceRequest* inference_request, uint32_t priority)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  lrequest->Request()->SetPriority(priority);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestTimeoutMicroseconds(
    TRTSERVER2_InferenceRequest* inference_request, uint64_t* timeout_us)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  *timeout_us = lrequest->Request()->TimeoutMicroseconds();
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestSetTimeoutMicroseconds(
    TRTSERVER2_InferenceRequest* inference_request, uint64_t timeout_us)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  lrequest->Request()->SetTimeoutMicroseconds(timeout_us);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestAddInput(
    TRTSERVER2_InferenceRequest* inference_request, const char* name,
    const char* datatype, const int64_t* shape, uint64_t dim_count)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(
      lrequest->Request()->AddOriginalInput(name, datatype, shape, dim_count));
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestRemoveInput(
    TRTSERVER2_InferenceRequest* inference_request, const char* name)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->Request()->RemoveOriginalInput(name));
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestRemoveAllInputs(
    TRTSERVER2_InferenceRequest* inference_request)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->Request()->RemoveAllOriginalInputs());
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestAppendInputData(
    TRTSERVER2_InferenceRequest* inference_request, const char* name,
    const void* base, size_t byte_size, TRTSERVER_Memory_Type memory_type,
    int64_t memory_type_id)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);

  ni::InferenceRequest::Input* input;
  RETURN_IF_STATUS_ERROR(
      lrequest->Request()->MutableOriginalInput(name, &input));
  RETURN_IF_STATUS_ERROR(
      input->AppendData(base, byte_size, memory_type, memory_type_id));

  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestRemoveAllInputData(
    TRTSERVER2_InferenceRequest* inference_request, const char* name)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);

  ni::InferenceRequest::Input* input;
  RETURN_IF_STATUS_ERROR(
      lrequest->Request()->MutableOriginalInput(name, &input));
  RETURN_IF_STATUS_ERROR(input->RemoveAllData());

  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestAddRequestedOutput(
    TRTSERVER2_InferenceRequest* inference_request, const char* name)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->Request()->AddRequestedOutput(name));
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestRemoveRequestedOutput(
    TRTSERVER2_InferenceRequest* inference_request, const char* name)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->Request()->RemoveRequestedOutput(name));
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestRemoveAllRequestedOutputs(
    TRTSERVER2_InferenceRequest* inference_request)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->Request()->RemoveAllRequestedOutputs());
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestSetRequestedOutputClassificationCount(
    TRTSERVER2_InferenceRequest* inference_request, const char* name,
    uint32_t count)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);

  ni::InferenceRequest::RequestedOutput* requested;
  RETURN_IF_STATUS_ERROR(
      lrequest->Request()->MutableRequestedOutput(name, &requested));
  requested->SetClassificationCount(count);

  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestError(TRTSERVER2_InferenceRequest* inference_request)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->RequestStatus());
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestOutputData(
    TRTSERVER2_InferenceRequest* inference_request, const char* name,
    const void** base, size_t* byte_size, TRTSERVER_Memory_Type* memory_type,
    int64_t* memory_type_id)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->Response()->OutputBufferContents(
      name, base, byte_size, memory_type, memory_type_id));
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestOutputDataType(
    TRTSERVER2_InferenceRequest* inference_request, const char* name,
    const char** datatype)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  const auto& response_header = lrequest->Response()->ResponseHeader();
  for (const auto& output : response_header.output()) {
    if (output.name() == name) {
      *datatype = GetDataTypeProtocolString(output.data_type());
      return nullptr;  // Success
    }
  }

  return TRTSERVER_ErrorNew(TRTSERVER_ERROR_INVALID_ARG, "unknown output");
}

// TEMPORARY: will be removed as part of V1->V2 transition
TRTSERVER_Error*
TRTSERVER2_InferenceRequestOutputClassBatchSize(
    TRTSERVER2_InferenceRequest* inference_request, const char* name,
    uint64_t* batch_size)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  const auto& response_header = lrequest->Response()->ResponseHeader();
  for (const auto& output : response_header.output()) {
    if (output.name() == name) {
      *batch_size = output.batch_classes().size();
      return nullptr;  // Success
    }
  }

  return TRTSERVER_ErrorNew(TRTSERVER_ERROR_INVALID_ARG, "unknown output");
}

// TEMPORARY: will be removed as part of V1->V2 transition
TRTSERVER_Error*
TRTSERVER2_InferenceRequestOutputClasses(
    TRTSERVER2_InferenceRequest* inference_request, const char* name,
    int32_t* idx, float* value, char** label)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  const auto& response_header = lrequest->Response()->ResponseHeader();
  for (const auto& output : response_header.output()) {
    if (output.name() == name) {
      int index = 0;
      uint64_t batch_size = output.batch_classes().size();
      for (uint64_t batch_id = 0; batch_id < batch_size; batch_id++) {
        auto& bcls = output.batch_classes(0);
        for (int i = 0; i < bcls.cls().size(); i++) {
          auto& cls = bcls.cls(i);
          idx[index] = cls.idx();
          value[index] = cls.value();
          label[index] = const_cast<char*>(cls.label().c_str());
          index++;
        }
      }
      return nullptr;  // Success
    }
  }

  return TRTSERVER_ErrorNew(TRTSERVER_ERROR_INVALID_ARG, "unknown output");
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestOutputShape(
    TRTSERVER2_InferenceRequest* inference_request, const char* name,
    int64_t* shape, uint64_t* dim_count)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  const auto& response_header = lrequest->Response()->ResponseHeader();
  for (const auto& output : response_header.output()) {
    if (output.name() == name) {
      if (!output.has_raw()) {
        return TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INVALID_ARG,
            "output shape not available for classification");
      }

      if ((uint64_t)output.raw().dims_size() > *dim_count) {
        return TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INVALID_ARG,
            std::string(
                "output shape has " + std::to_string(output.raw().dims_size()) +
                " dimensions, shape buffer too small")
                .c_str());
      }

      *dim_count = output.raw().dims_size();
      for (int d = 0; d < output.raw().dims_size(); ++d) {
        shape[d] = output.raw().dims(d);
      }

      return nullptr;  // Success
    }
  }

  return TRTSERVER_ErrorNew(TRTSERVER_ERROR_INVALID_ARG, "unknown output");
}

TRTSERVER_Error*
TRTSERVER2_InferenceRequestRemoveAllOutputs(
    TRTSERVER2_InferenceRequest* inference_request)
{
  TrtInferenceRequest* lrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  lrequest->SetResponse(nullptr);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER2_ServerInferAsync(
    TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
    TRTSERVER2_InferenceRequest* inference_request,
    TRTSERVER_ResponseAllocator* response_allocator,
    void* response_allocator_userp,
    TRTSERVER2_InferenceCompleteFn_t complete_fn, void* complete_userp)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  TrtInferenceRequest* ltrtrequest =
      reinterpret_cast<TrtInferenceRequest*>(inference_request);
  TrtServerResponseAllocator* lresponsealloc =
      reinterpret_cast<TrtServerResponseAllocator*>(response_allocator);

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

  std::shared_ptr<ni::InferRequestProvider> infer_request_provider;
  RETURN_IF_STATUS_ERROR(
      ni::InferRequestProvider::Create(lrequest, &infer_request_provider));

  std::shared_ptr<ni::InferResponseProvider> infer_response_provider;
  {
    std::shared_ptr<ni::InferResponseProvider> del_response_provider;
    RETURN_IF_STATUS_ERROR(ni::InferResponseProvider::Create(
        lrequest, lbackend->GetLabelProvider(), response_allocator,
        lresponsealloc->AllocFn(), response_allocator_userp,
        lresponsealloc->ReleaseFn(), &del_response_provider));
    infer_response_provider = std::move(del_response_provider);
  }

  lserver->InferAsync(
      lbackend, infer_request_provider, infer_response_provider, infer_stats,
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
            reinterpret_cast<TRTSERVER2_InferenceRequest*>(ltrtrequest),
            complete_userp);
      });

  return nullptr;  // Success
}

#ifdef __cplusplus
}
#endif
