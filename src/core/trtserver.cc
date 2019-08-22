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

#include "src/core/trtserver.h"

#include <string>
#include <vector>
#include "src/core/backend.h"
#include "src/core/logging.h"
#include "src/core/metrics.h"
#include "src/core/provider_utils.h"
#include "src/core/request_status.pb.h"
#include "src/core/server.h"
#include "src/core/server_status.h"
#include "src/core/status.h"

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

  TRTSERVER_Memory_Type Type() const { return type_; }
  const std::string& Name() const { return name_; }
  const std::string& ShmKey() const { return shm_key_; }
  size_t Offset() const { return offset_; }
  size_t ByteSize() const { return byte_size_; }

 private:
  const TRTSERVER_Memory_Type type_;
  const std::string name_;
  const std::string shm_key_;
  const size_t offset_;
  const size_t byte_size_;
};

TrtServerSharedMemoryBlock::TrtServerSharedMemoryBlock(
    TRTSERVER_Memory_Type type, const char* name, const char* shm_key,
    const size_t offset, const size_t byte_size)
    : type_(type), name_(name), shm_key_(shm_key), offset_(offset),
      byte_size_(byte_size)
{
}

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

  const std::string& ModelRepositoryPath() const { return repo_path_; }
  void SetModelRepositoryPath(const char* p) { repo_path_ = p; }

  ni::ModelControlMode ModelControlMode() const { return model_control_mode_; }
  void SetModelControlMode(ni::ModelControlMode m) { model_control_mode_ = m; }

  bool ExitOnError() const { return exit_on_error_; }
  void SetExitOnError(bool b) { exit_on_error_ = b; }

  bool StrictModelConfig() const { return strict_model_config_; }
  void SetStrictModelConfig(bool b) { strict_model_config_ = b; }

  bool StrictReadiness() const { return strict_readiness_; }
  void SetStrictReadiness(bool b) { strict_readiness_ = b; }

  bool Profiling() const { return profiling_; }
  void SetProfiling(bool b) { profiling_ = b; }

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
  std::string repo_path_;
  ni::ModelControlMode model_control_mode_;
  bool exit_on_error_;
  bool strict_model_config_;
  bool strict_readiness_;
  bool profiling_;
  bool metrics_;
  bool gpu_metrics_;
  unsigned int exit_timeout_;

  bool tf_soft_placement_;
  float tf_gpu_mem_fraction_;
  std::map<int, std::pair<int, uint64_t>> tf_vgpu_memory_limits_;
};

TrtServerOptions::TrtServerOptions()
    : server_id_("inference:0"), model_control_mode_(ni::MODE_POLL),
      exit_on_error_(true), strict_model_config_(true), strict_readiness_(true),
      profiling_(false), metrics_(true), gpu_metrics_(true), exit_timeout_(30),
      tf_soft_placement_(true), tf_gpu_mem_fraction_(0)
{
#ifndef TRTIS_ENABLE_METRICS
  metrics_ = false;
  gpu_metrics_ = false;
#endif  // TRTIS_ENABLE_METRICS
}

//
// TrtServerRequestProvider
//
// Implementation for TRTSERVER_InferenceRequestProvider.
//
class TrtServerRequestProvider {
 public:
  TrtServerRequestProvider(
      const char* model_name, int64_t model_version,
      const std::shared_ptr<ni::InferRequestHeader>& request_header);
  TRTSERVER_Error* Init(ni::InferenceServer* server);

  const std::string& ModelName() const { return model_name_; }
  int64_t ModelVersion() const { return model_version_; }
  ni::InferRequestHeader* InferRequestHeader() const;
  const std::shared_ptr<ni::InferenceBackend>& Backend() const;
  const std::unordered_map<std::string, std::shared_ptr<ni::SystemMemory>>&
  InputMap() const;

  void SetInputData(
      const char* input_name, const void* base, size_t byte_size,
      TRTSERVER_Memory_Type memory_type);

 private:
  const std::string model_name_;
  const int64_t model_version_;
  std::shared_ptr<ni::InferRequestHeader> request_header_;
  std::shared_ptr<ni::InferenceBackend> backend_;
  std::unordered_map<std::string, std::shared_ptr<ni::SystemMemory>> input_map_;
};

TrtServerRequestProvider::TrtServerRequestProvider(
    const char* model_name, int64_t model_version,
    const std::shared_ptr<ni::InferRequestHeader>& request_header)
    : model_name_(model_name), model_version_(model_version),
      request_header_(request_header)
{
}

TRTSERVER_Error*
TrtServerRequestProvider::Init(ni::InferenceServer* server)
{
  // Grab a handle to the backend that this request requires so that
  // the backend doesn't get unloaded (also need the backend to
  // normalize the request).
  RETURN_IF_STATUS_ERROR(
      server->GetInferenceBackend(model_name_, model_version_, &backend_));
  RETURN_IF_STATUS_ERROR(
      ni::NormalizeRequestHeader(*(backend_.get()), *(request_header_.get())));

  return nullptr;  // Success
}

ni::InferRequestHeader*
TrtServerRequestProvider::InferRequestHeader() const
{
  return request_header_.get();
}

const std::shared_ptr<ni::InferenceBackend>&
TrtServerRequestProvider::Backend() const
{
  return backend_;
}

const std::unordered_map<std::string, std::shared_ptr<ni::SystemMemory>>&
TrtServerRequestProvider::InputMap() const
{
  return input_map_;
}

void
TrtServerRequestProvider::SetInputData(
    const char* input_name, const void* base, size_t byte_size,
    TRTSERVER_Memory_Type memory_type)
{
  auto pr = input_map_.emplace(input_name, nullptr);
  std::shared_ptr<ni::SystemMemory>& smem = pr.first->second;
  if (pr.second) {
    smem.reset(new ni::SystemMemoryReference());
  }

  if (byte_size > 0) {
    std::static_pointer_cast<ni::SystemMemoryReference>(smem)->AddBuffer(
        static_cast<const char*>(base), byte_size, memory_type);
  }
}

//
// TrtServerResponse
//
// Implementation for TRTSERVER_InferenceResponse.
//
class TrtServerResponse {
 public:
  TrtServerResponse(
      const ni::Status& infer_status,
      const std::shared_ptr<ni::InferResponseProvider>& provider);
  TRTSERVER_Error* Status() const;
  const ni::InferResponseHeader& Header() const;
  TRTSERVER_Error* OutputData(
      const char* name, const void** base, size_t* byte_size,
      TRTSERVER_Memory_Type* memory_type) const;

 private:
  const ni::Status infer_status_;
  std::shared_ptr<ni::InferResponseProvider> response_provider_;
};

TrtServerResponse::TrtServerResponse(
    const ni::Status& infer_status,
    const std::shared_ptr<ni::InferResponseProvider>& provider)
    : infer_status_(infer_status), response_provider_(provider)
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
    TRTSERVER_Memory_Type* memory_type) const
{
  RETURN_IF_STATUS_ERROR(response_provider_->OutputBufferContents(
      name, base, byte_size, memory_type));
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
TRTSERVER_SharedMemoryBlockDelete(
    TRTSERVER_SharedMemoryBlock* shared_memory_block)
{
  TrtServerSharedMemoryBlock* lsmb =
      reinterpret_cast<TrtServerSharedMemoryBlock*>(shared_memory_block);
  delete lsmb;
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
// TRTSERVER_InferenceRequestProvider
//
TRTSERVER_Error*
TRTSERVER_InferenceRequestProviderNew(
    TRTSERVER_InferenceRequestProvider** request_provider,
    TRTSERVER_Server* server, const char* model_name, int64_t model_version,
    const char* request_header_base, size_t request_header_byte_size)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  std::shared_ptr<ni::InferRequestHeader> request_header =
      std::make_shared<ni::InferRequestHeader>();
  if (!request_header->ParseFromArray(
          request_header_base, request_header_byte_size)) {
    return TrtServerError::Create(
        ni::RequestStatusCode::INVALID_ARG,
        "failed to parse InferRequestHeader");
  }

  TrtServerRequestProvider* provider =
      new TrtServerRequestProvider(model_name, model_version, request_header);
  TRTSERVER_Error* err = provider->Init(lserver);
  if (err == nullptr) {
    *request_provider =
        reinterpret_cast<TRTSERVER_InferenceRequestProvider*>(provider);
  } else {
    delete provider;
  }

  return err;
}

TRTSERVER_Error*
TRTSERVER_InferenceRequestProviderDelete(
    TRTSERVER_InferenceRequestProvider* request_provider)
{
  TrtServerRequestProvider* lprovider =
      reinterpret_cast<TrtServerRequestProvider*>(request_provider);
  delete lprovider;
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_InferenceRequestProviderInputBatchByteSize(
    TRTSERVER_InferenceRequestProvider* request_provider, const char* name,
    uint64_t* byte_size)
{
  TrtServerRequestProvider* lprovider =
      reinterpret_cast<TrtServerRequestProvider*>(request_provider);

  ni::InferRequestHeader* request_header = lprovider->InferRequestHeader();
  for (const auto& io : request_header->input()) {
    if (io.name() == std::string(name)) {
      *byte_size = io.batch_byte_size();
      return nullptr;  // Success
    }
  }

  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_INVALID_ARG,
      std::string(
          "batch byte-size requested for unknown input tensor '" +
          std::string(name) + "', in model '" + lprovider->ModelName() + "'")
          .c_str());
}

TRTSERVER_Error*
TRTSERVER_InferenceRequestProviderSetInputData(
    TRTSERVER_InferenceRequestProvider* request_provider,
    const char* input_name, const void* base, size_t byte_size,
    TRTSERVER_Memory_Type memory_type)
{
  TrtServerRequestProvider* lprovider =
      reinterpret_cast<TrtServerRequestProvider*>(request_provider);
  lprovider->SetInputData(input_name, base, byte_size, memory_type);
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
    size_t* byte_size, TRTSERVER_Memory_Type* memory_type)
{
  TrtServerResponse* lresponse = reinterpret_cast<TrtServerResponse*>(response);
  return lresponse->OutputData(name, base, byte_size, memory_type);
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
TRTSERVER_ServerOptionsSetStrictReadiness(
    TRTSERVER_ServerOptions* options, bool strict)
{
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);
  loptions->SetStrictReadiness(strict);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerOptionsSetProfiling(
    TRTSERVER_ServerOptions* options, bool profiling)
{
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);
  loptions->SetProfiling(profiling);
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
  // Logging is global for now...
  LOG_ENABLE_INFO(log);
  return nullptr;  // Success
}

// Enable or disable warning level logging.
TRTSERVER_Error*
TRTSERVER_ServerOptionsSetLogWarn(TRTSERVER_ServerOptions* options, bool log)
{
  // Logging is global for now...
  LOG_ENABLE_WARNING(log);
  return nullptr;  // Success
}

// Enable or disable error level logging.
TRTSERVER_Error*
TRTSERVER_ServerOptionsSetLogError(TRTSERVER_ServerOptions* options, bool log)
{
  // Logging is global for now...
  LOG_ENABLE_ERROR(log);
  return nullptr;  // Success
}

// Set verbose logging level. Level zero disables verbose logging.
TRTSERVER_Error*
TRTSERVER_ServerOptionsSetLogVerbose(
    TRTSERVER_ServerOptions* options, int level)
{
  // Logging is global for now...
  LOG_SET_VERBOSE(level);
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

#ifdef TRTIS_ENABLE_METRICS
  if (loptions->Metrics() && loptions->GpuMetrics()) {
    ni::Metrics::EnableGPUMetrics();
  }
#endif  // TRTIS_ENABLE_METRICS

  lserver->SetId(loptions->ServerId());
  lserver->SetModelRepositoryPath(loptions->ModelRepositoryPath());
  lserver->SetModelControlMode(loptions->ModelControlMode());
  lserver->SetStrictModelConfigEnabled(loptions->StrictModelConfig());
  lserver->SetStrictReadinessEnabled(loptions->StrictReadiness());
  lserver->SetProfilingEnabled(loptions->Profiling());
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

  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::HEALTH);

  RETURN_IF_STATUS_ERROR(lserver->IsLive(live));
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerIsReady(TRTSERVER_Server* server, bool* ready)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::HEALTH);

  RETURN_IF_STATUS_ERROR(lserver->IsReady(ready));
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerStatus(TRTSERVER_Server* server, TRTSERVER_Protobuf** status)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::STATUS);

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

  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::STATUS);

  ni::ServerStatus server_status;
  RETURN_IF_STATUS_ERROR(
      lserver->GetStatus(&server_status, std::string(model_name)));

  TrtServerProtobuf* protobuf = new TrtServerProtobuf(server_status);
  *status = reinterpret_cast<TRTSERVER_Protobuf*>(protobuf);

  return nullptr;  // success
}

TRTSERVER_Error*
TRTSERVER_ServerLoadModel(TRTSERVER_Server* server, const char* model_name)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::MODEL_CONTROL);

  RETURN_IF_STATUS_ERROR(lserver->LoadModel(std::string(model_name)));

  return nullptr;  // success
}

TRTSERVER_Error*
TRTSERVER_ServerUnloadModel(TRTSERVER_Server* server, const char* model_name)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::MODEL_CONTROL);

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

  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(),
      ni::ServerStatTimerScoped::Kind::SHARED_MEMORY_CONTROL);

  RETURN_IF_STATUS_ERROR(lserver->RegisterSharedMemory(
      lsmb->Name(), lsmb->ShmKey(), lsmb->Offset(), lsmb->ByteSize()));

  return nullptr;  // success
}

TRTSERVER_Error*
TRTSERVER_ServerUnregisterSharedMemory(
    TRTSERVER_Server* server, TRTSERVER_SharedMemoryBlock* shared_memory_block)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  TrtServerSharedMemoryBlock* lsmb =
      reinterpret_cast<TrtServerSharedMemoryBlock*>(shared_memory_block);

  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(),
      ni::ServerStatTimerScoped::Kind::SHARED_MEMORY_CONTROL);

  RETURN_IF_STATUS_ERROR(lserver->UnregisterSharedMemory(lsmb->Name()));

  return nullptr;  // success
}

TRTSERVER_Error*
TRTSERVER_ServerUnregisterAllSharedMemory(TRTSERVER_Server* server)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(),
      ni::ServerStatTimerScoped::Kind::SHARED_MEMORY_CONTROL);

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

  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(),
      ni::ServerStatTimerScoped::Kind::SHARED_MEMORY_CONTROL);

  RETURN_IF_STATUS_ERROR(
      lserver->SharedMemoryAddress(lsmb->Name(), offset, byte_size, base));

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
    TRTSERVER_Server* server,
    TRTSERVER_InferenceRequestProvider* request_provider,
    TRTSERVER_ResponseAllocator* response_allocator,
    void* response_allocator_userp, TRTSERVER_InferenceCompleteFn_t complete_fn,
    void* complete_userp)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  TrtServerRequestProvider* lprovider =
      reinterpret_cast<TrtServerRequestProvider*>(request_provider);
  TrtServerResponseAllocator* lresponsealloc =
      reinterpret_cast<TrtServerResponseAllocator*>(response_allocator);

  ni::InferRequestHeader* request_header = lprovider->InferRequestHeader();

  auto infer_stats = std::make_shared<ni::ModelInferStats>(
      lserver->StatusManager(), lprovider->ModelName());
  auto timer = std::make_shared<ni::ModelInferStats::ScopedTimer>();
  infer_stats->StartRequestTimer(timer.get());
  infer_stats->SetRequestedVersion(lprovider->ModelVersion());
  infer_stats->SetMetricReporter(lprovider->Backend()->MetricReporter());
  infer_stats->SetBatchSize(request_header->batch_size());
  infer_stats->SetFailed(true);

  std::shared_ptr<ni::InferRequestProvider> infer_request_provider;
  RETURN_IF_STATUS_ERROR(ni::InferRequestProvider::Create(
      lprovider->ModelName(), lprovider->ModelVersion(), *request_header,
      lprovider->InputMap(), &infer_request_provider));

  std::shared_ptr<ni::InferResponseProvider> infer_response_provider;
  {
    std::shared_ptr<ni::InferResponseProvider> del_response_provider;
    RETURN_IF_STATUS_ERROR(ni::InferResponseProvider::Create(
        *request_header, lprovider->Backend()->GetLabelProvider(),
        response_allocator, lresponsealloc->AllocFn(), response_allocator_userp,
        lresponsealloc->ReleaseFn(), &del_response_provider));
    infer_response_provider = del_response_provider;
  }

  lserver->Infer(
      lprovider->Backend(), infer_request_provider, infer_response_provider,
      infer_stats,
      [infer_stats, timer, infer_response_provider, server, complete_fn,
       complete_userp](const ni::Status& status) mutable {
        infer_stats->SetFailed(!status.IsOk());
        if (!status.IsOk()) {
          LOG_VERBOSE(1) << "Infer failed: " << status.Message();
        }

        timer.reset();

        TrtServerResponse* response =
            new TrtServerResponse(status, infer_response_provider);
        complete_fn(
            server, reinterpret_cast<TRTSERVER_InferenceResponse*>(response),
            complete_userp);
      });

  return nullptr;  // Success
}

#ifdef __cplusplus
}
#endif
