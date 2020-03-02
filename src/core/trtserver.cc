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
#include "src/core/nvtx.h"
#include "src/core/request_status.pb.h"
#include "src/core/server.h"
#include "src/core/server_status.h"
#include "src/core/status.h"
#include "src/core/tracing.h"

namespace ni = nvidia::inferenceserver;

namespace {

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
// TrtServerSharedMemoryBlock
//
// Implementation for TRTSERVER_SharedMemoryBlock.
//
class TrtServerSharedMemoryBlock {
 public:
  explicit TrtServerSharedMemoryBlock(
      TRTSERVER_Memory_Type type, const char* name, const char* shm_key,
      const size_t offset, const size_t byte_size);
#ifdef TRTIS_ENABLE_GPU
  explicit TrtServerSharedMemoryBlock(
      TRTSERVER_Memory_Type type, const char* name,
      const cudaIpcMemHandle_t* cuda_shm_handle, const int device_id,
      const size_t byte_size);
#endif  // TRTIS_ENABLE_GPU

  TRTSERVER_Memory_Type Type() const { return type_; }
  const std::string& Name() const { return name_; }
  const std::string& ShmKey() const { return shm_key_; }
#ifdef TRTIS_ENABLE_GPU
  const cudaIpcMemHandle_t* CudaHandle() const { return cuda_shm_handle_; }
  size_t DeviceId() const { return device_id_; }
#endif  // TRTIS_ENABLE_GPU
  size_t Offset() const { return offset_; }
  size_t ByteSize() const { return byte_size_; }

 private:
  const TRTSERVER_Memory_Type type_;
  const std::string name_;
  const std::string shm_key_;
#ifdef TRTIS_ENABLE_GPU
  const cudaIpcMemHandle_t* cuda_shm_handle_;
  const int device_id_;
#endif  // TRTIS_ENABLE_GPU
  const size_t offset_;
  const size_t byte_size_;
};

TrtServerSharedMemoryBlock::TrtServerSharedMemoryBlock(
    TRTSERVER_Memory_Type type, const char* name, const char* shm_key,
    const size_t offset, const size_t byte_size)
#ifdef TRTIS_ENABLE_GPU
    : type_(type), name_(name), shm_key_(shm_key), cuda_shm_handle_(nullptr),
      device_id_(0), offset_(offset), byte_size_(byte_size)
#else
    : type_(type), name_(name), shm_key_(shm_key), offset_(offset),
      byte_size_(byte_size)
#endif  // TRTIS_ENABLE_GPU
{
}

#ifdef TRTIS_ENABLE_GPU
TrtServerSharedMemoryBlock::TrtServerSharedMemoryBlock(
    TRTSERVER_Memory_Type type, const char* name,
    const cudaIpcMemHandle_t* cuda_shm_handle, const int device_id,
    const size_t byte_size)
    : type_(type), name_(name), shm_key_(""), cuda_shm_handle_(cuda_shm_handle),
      device_id_(device_id), offset_(0), byte_size_(byte_size)
{
}
#endif  // TRTIS_ENABLE_GPU

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
      : backend_(backend), request_(request)
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

 private:
  std::shared_ptr<ni::InferenceBackend> backend_;
  std::shared_ptr<ni::InferenceRequest> request_;
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

TRTSERVER_Error*
TRTSERVER_SharedMemoryBlockCpuNew(
    TRTSERVER_SharedMemoryBlock** shared_memory_block, const char* name,
    const char* shm_key, const size_t offset, const size_t byte_size)
{
  *shared_memory_block = reinterpret_cast<TRTSERVER_SharedMemoryBlock*>(
      new TrtServerSharedMemoryBlock(
          TRTSERVER_MEMORY_CPU, name, shm_key, offset, byte_size));
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_SharedMemoryBlockGpuNew(
    TRTSERVER_SharedMemoryBlock** shared_memory_block, const char* name,
    const cudaIpcMemHandle_t* cuda_shm_handle, const size_t byte_size,
    const int device_id)
{
#ifdef TRTIS_ENABLE_GPU
  *shared_memory_block = reinterpret_cast<TRTSERVER_SharedMemoryBlock*>(
      new TrtServerSharedMemoryBlock(
          TRTSERVER_MEMORY_GPU, name, cuda_shm_handle, device_id, byte_size));
  return nullptr;  // Success
#else
  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNSUPPORTED,
      "CUDA shared memory not supported when TRTIS_ENABLE_GPU=0");
#endif  // TRTIS_ENABLE_GPU
}

TRTSERVER_Error*
TRTSERVER_SharedMemoryBlockDelete(
    TRTSERVER_SharedMemoryBlock* shared_memory_block)
{
  TrtServerSharedMemoryBlock* lsmb =
      reinterpret_cast<TrtServerSharedMemoryBlock*>(shared_memory_block);
  delete lsmb;
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_SharedMemoryBlockMemoryType(
    TRTSERVER_SharedMemoryBlock* shared_memory_block,
    TRTSERVER_Memory_Type* memory_type)
{
  TrtServerSharedMemoryBlock* lsmb =
      reinterpret_cast<TrtServerSharedMemoryBlock*>(shared_memory_block);
  *memory_type = lsmb->Type();
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_SharedMemoryBlockMemoryTypeId(
    TRTSERVER_SharedMemoryBlock* shared_memory_block, int64_t* memory_type_id)
{
#ifdef TRTIS_ENABLE_GPU
  TrtServerSharedMemoryBlock* lsmb =
      reinterpret_cast<TrtServerSharedMemoryBlock*>(shared_memory_block);
  *memory_type_id = lsmb->DeviceId();
#else
  *memory_type_id = 0;
#endif             // TRTIS_ENABLE_GPU
  return nullptr;  // Success
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
      loptions->ModelName(), loptions->ModelVersion(),
      lserver->ProtocolVersion()));
  request->SetActualModelVersion(backend->Version());
  request->SetId(loptions->InferRequestHeader()->id());
#ifdef TRTIS_ENABLE_GRPC_V2
  request->SetIdStr(loptions->IdStr());
#endif  // TRTIS_ENABLE_GRPC_V2
  request->SetFlags(loptions->InferRequestHeader()->flags());
  request->SetCorrelationId(loptions->InferRequestHeader()->correlation_id());
  request->SetBatchSize(loptions->InferRequestHeader()->batch_size());
  request->SetPriority(loptions->InferRequestHeader()->priority());
  request->SetTimeoutMicroseconds(loptions->InferRequestHeader()->timeout_microseconds());
  for (const auto& io : loptions->InferRequestHeader()->input()) {
    if (io.has_shared_memory()) {
      RETURN_IF_STATUS_ERROR(request->AddInput(
          io.name(), io.dims(), io.batch_byte_size(), io.shared_memory()));
    } else {
      RETURN_IF_STATUS_ERROR(
          request->AddInput(io.name(), io.dims(), io.batch_byte_size()));
    }
  }
  for (const auto& io : loptions->InferRequestHeader()->output()) {
    uint32_t cls_cnt = io.has_cls() ? io.cls().count() : 0;
    if (io.has_shared_memory()) {
      RETURN_IF_STATUS_ERROR(
          request->RequestOutput(io.name(), cls_cnt, io.shared_memory()));
    } else {
      RETURN_IF_STATUS_ERROR(request->RequestOutput(io.name(), cls_cnt));
    }
  }

  RETURN_IF_STATUS_ERROR(request->Normalize(*backend));

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

  for (const auto& pr : lrequest->Inputs()) {
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

  auto* inputs = lrequest->MutableInputs();
  auto it = inputs->find(input_name);
  if (it == inputs->end()) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INVALID_ARG,
        std::string("input '" + std::string(input_name) + "' does not exist")
            .c_str());
  }

  it->second.AppendData(base, byte_size, memory_type, memory_type_id);

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
TRTSERVER_ServerRegisterSharedMemory(
    TRTSERVER_Server* server, TRTSERVER_SharedMemoryBlock* shared_memory_block)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  TrtServerSharedMemoryBlock* lsmb =
      reinterpret_cast<TrtServerSharedMemoryBlock*>(shared_memory_block);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(),
      ni::ServerStatTimerScoped::Kind::SHARED_MEMORY_CONTROL);
#endif  // TRTIS_ENABLE_STATS

  if (lsmb->Type() == TRTSERVER_MEMORY_CPU) {
    RETURN_IF_STATUS_ERROR(lserver->RegisterSharedMemory(
        lsmb->Name(), lsmb->ShmKey(), lsmb->Offset(), lsmb->ByteSize()));
  } else {
#ifdef TRTIS_ENABLE_GPU
    RETURN_IF_STATUS_ERROR(lserver->RegisterCudaSharedMemory(
        lsmb->Name(), lsmb->CudaHandle(), lsmb->ByteSize(), lsmb->DeviceId()));
#endif  // TRTIS_ENABLE_GPU
  }

  return nullptr;  // success
}

TRTSERVER_Error*
TRTSERVER_ServerUnregisterSharedMemory(
    TRTSERVER_Server* server, TRTSERVER_SharedMemoryBlock* shared_memory_block)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  TrtServerSharedMemoryBlock* lsmb =
      reinterpret_cast<TrtServerSharedMemoryBlock*>(shared_memory_block);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(),
      ni::ServerStatTimerScoped::Kind::SHARED_MEMORY_CONTROL);
#endif  // TRTIS_ENABLE_STATS

  RETURN_IF_STATUS_ERROR(lserver->UnregisterSharedMemory(lsmb->Name()));

  return nullptr;  // success
}

TRTSERVER_Error*
TRTSERVER_ServerUnregisterAllSharedMemory(TRTSERVER_Server* server)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(),
      ni::ServerStatTimerScoped::Kind::SHARED_MEMORY_CONTROL);
#endif  // TRTIS_ENABLE_STATS

  RETURN_IF_STATUS_ERROR(lserver->UnregisterAllSharedMemory());

  return nullptr;  // success
}

TRTSERVER_Error*
TRTSERVER_ServerSharedMemoryAddress(
    TRTSERVER_Server* server, TRTSERVER_SharedMemoryBlock* shared_memory_block,
    size_t offset, size_t byte_size, void** base)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  TrtServerSharedMemoryBlock* lsmb =
      reinterpret_cast<TrtServerSharedMemoryBlock*>(shared_memory_block);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(),
      ni::ServerStatTimerScoped::Kind::SHARED_MEMORY_CONTROL);
#endif  // TRTIS_ENABLE_STATS

  RETURN_IF_STATUS_ERROR(
      lserver->SharedMemoryAddress(lsmb->Name(), offset, byte_size, base));

  return nullptr;  // success
}

TRTSERVER_Error*
TRTSERVER_ServerSharedMemoryStatus(
    TRTSERVER_Server* server, TRTSERVER_Protobuf** status)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

#ifdef TRTIS_ENABLE_STATS
  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(),
      ni::ServerStatTimerScoped::Kind::SHARED_MEMORY_CONTROL);
#endif  // TRTIS_ENABLE_STATS

  ni::SharedMemoryStatus shm_status;
  RETURN_IF_STATUS_ERROR(lserver->GetSharedMemoryStatus(&shm_status));

  TrtServerProtobuf* protobuf = new TrtServerProtobuf(shm_status);
  *status = reinterpret_cast<TRTSERVER_Protobuf*>(protobuf);

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

#ifdef TRTIS_ENABLE_STATS
  auto infer_stats = std::make_shared<ni::ModelInferStats>(
      lserver->StatusManager(), lrequest->ModelName());
  infer_stats->CaptureTimestamp(
      ni::ModelInferStats::TimestampKind::kRequestStart);
  infer_stats->SetRequestedVersion(lrequest->RequestedModelVersion());
  infer_stats->SetMetricReporter(lbackend->MetricReporter());
  infer_stats->SetBatchSize(lrequest->BatchSize());
  infer_stats->SetFailed(true);
  infer_stats->SetTraceManager(trace_manager);
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
    infer_response_provider = del_response_provider;
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

#ifdef __cplusplus
}
#endif
