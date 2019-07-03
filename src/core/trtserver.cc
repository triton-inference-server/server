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

#include "src/core/backend.h"
#include "src/core/logging.h"
#include "src/core/provider_utils.h"
#include "src/core/request_status.pb.h"
#include "src/core/server.h"
#include "src/core/server_status.h"

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
  static TRTSERVER_Error* Create(const ni::RequestStatus& status);

  static ni::RequestStatusCode CodeToStatus(TRTSERVER_Error_Code code);
  static TRTSERVER_Error_Code StatusToCode(ni::RequestStatusCode status);

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
      new TrtServerError(TrtServerError::CodeToStatus(code), msg));
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
TrtServerError::Create(const ni::RequestStatus& status)
{
  return Create(status.code(), status.msg());
}

ni::RequestStatusCode
TrtServerError::CodeToStatus(TRTSERVER_Error_Code code)
{
  switch (code) {
    case TRTSERVER_ERROR_UNKNOWN:
      return ni::RequestStatusCode::UNKNOWN;
    case TRTSERVER_ERROR_INTERNAL:
      return ni::RequestStatusCode::INTERNAL;
    case TRTSERVER_ERROR_NOT_FOUND:
      return ni::RequestStatusCode::NOT_FOUND;
    case TRTSERVER_ERROR_INVALID_ARG:
      return ni::RequestStatusCode::INVALID_ARG;
    case TRTSERVER_ERROR_UNAVAILABLE:
      return ni::RequestStatusCode::UNAVAILABLE;
    case TRTSERVER_ERROR_UNSUPPORTED:
      return ni::RequestStatusCode::UNSUPPORTED;
    case TRTSERVER_ERROR_ALREADY_EXISTS:
      return ni::RequestStatusCode::ALREADY_EXISTS;

    default:
      break;
  }

  return ni::RequestStatusCode::UNKNOWN;
}

TRTSERVER_Error_Code
TrtServerError::StatusToCode(ni::RequestStatusCode status_code)
{
  switch (status_code) {
    case ni::RequestStatusCode::UNKNOWN:
      return TRTSERVER_ERROR_UNKNOWN;
    case ni::RequestStatusCode::INTERNAL:
      return TRTSERVER_ERROR_INTERNAL;
    case ni::RequestStatusCode::NOT_FOUND:
      return TRTSERVER_ERROR_NOT_FOUND;
    case ni::RequestStatusCode::INVALID_ARG:
      return TRTSERVER_ERROR_INVALID_ARG;
    case ni::RequestStatusCode::UNAVAILABLE:
      return TRTSERVER_ERROR_UNAVAILABLE;
    case ni::RequestStatusCode::UNSUPPORTED:
      return TRTSERVER_ERROR_UNSUPPORTED;
    case ni::RequestStatusCode::ALREADY_EXISTS:
      return TRTSERVER_ERROR_ALREADY_EXISTS;

    default:
      break;
  }

  return TRTSERVER_ERROR_UNKNOWN;
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

#define RETURN_IF_STATUS_ERROR(S)                                         \
  do {                                                                    \
    const ni::Status& status__ = (S);                                     \
    if (status__.Code() != ni::RequestStatusCode::SUCCESS) {              \
      return TrtServerError::Create(status__.Code(), status__.Message()); \
    }                                                                     \
  } while (false)

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
// TrtServerOptions
//
// Implementation for TRTSERVER_ServerOptions.
//
class TrtServerOptions {
 public:
  const std::string& ServerId() const { return server_id_; }
  void SetServerId(const char* id) { server_id_ = id; }

  const std::string& ModelRepositoryPath() const { return repo_path_; }
  void SetModelRepositoryPath(const char* p) { repo_path_ = p; }

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

  bool TensorFlowSoftPlacement() const { return tf_soft_placement_; }
  void SetTensorFlowSoftPlacement(bool b) { tf_soft_placement_ = b; }

  float TensorFlowGpuMemoryFraction() const { return tf_gpu_mem_fraction_; }
  void SetTensorFlowGpuMemoryFraction(float f) { tf_gpu_mem_fraction_ = f; }

 private:
  std::string server_id_;
  std::string repo_path_;
  bool exit_on_error_;
  bool strict_model_config_;
  bool strict_readiness_;
  bool profiling_;
  unsigned int exit_timeout_;

  bool tf_soft_placement_;
  float tf_gpu_mem_fraction_;
};

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

  void SetInputData(const char* input_name, const void* base, size_t byte_size);

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
    const char* input_name, const void* base, size_t byte_size)
{
  auto pr = input_map_.emplace(input_name, nullptr);
  std::shared_ptr<ni::SystemMemory>& smem = pr.first->second;
  if (pr.second) {
    smem.reset(new ni::SystemMemoryReference());
  }

  std::static_pointer_cast<ni::SystemMemoryReference>(smem)->AddBuffer(
      static_cast<const char*>(base), byte_size);
}

//
// TrtServerResponse
//
// Implementation for TRTSERVER_InferenceResponse.
//
class TrtServerResponse {
 public:
  TrtServerResponse(
      const std::shared_ptr<ni::RequestStatus>& status,
      const std::shared_ptr<ni::InferResponseProvider>& provider);
  TRTSERVER_Error* Status() const;
  const ni::InferResponseHeader& Header() const;
  TRTSERVER_Error* OutputData(
      const char* name, const void** base, size_t* byte_size) const;

 private:
  std::shared_ptr<ni::RequestStatus> request_status_;
  std::shared_ptr<ni::InferResponseProvider> response_provider_;
};

TrtServerResponse::TrtServerResponse(
    const std::shared_ptr<ni::RequestStatus>& status,
    const std::shared_ptr<ni::InferResponseProvider>& provider)
    : request_status_(status), response_provider_(provider)
{
}

TRTSERVER_Error*
TrtServerResponse::Status() const
{
  return TrtServerError::Create(*request_status_);
}

const ni::InferResponseHeader&
TrtServerResponse::Header() const
{
  return response_provider_->ResponseHeader();
}

TRTSERVER_Error*
TrtServerResponse::OutputData(
    const char* name, const void** base, size_t* byte_size) const
{
  RETURN_IF_STATUS_ERROR(
      response_provider_->OutputBufferContents(name, base, byte_size));
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
  return TrtServerError::StatusToCode(lerror->StatusCode());
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
// TRTSERVER_MemoryAllocator
//
TRTSERVER_Error*
TRTSERVER_MemoryAllocatorNew(
    TRTSERVER_MemoryAllocator** allocator, TRTSERVER_MemoryAllocFn_t alloc_fn,
    TRTSERVER_MemoryDeleteFn_t delete_fn)
{
  // FIXME: allocator requires provider changes that are
  // not-yet-implemented so for now do nothing
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_MemoryAllocatorDelete(TRTSERVER_MemoryAllocator* allocator)
{
  // FIXME: allocator requires provider changes that are
  // not-yet-implemented so for now do nothing
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
    const char* input_name, const void* base, size_t byte_size)
{
  TrtServerRequestProvider* lprovider =
      reinterpret_cast<TrtServerRequestProvider*>(request_provider);
  lprovider->SetInputData(input_name, base, byte_size);
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
    size_t* byte_size)
{
  TrtServerResponse* lresponse = reinterpret_cast<TrtServerResponse*>(response);
  return lresponse->OutputData(name, base, byte_size);
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

//
// TRTSERVER_Server
//
TRTSERVER_Error*
TRTSERVER_ServerNew(TRTSERVER_Server** server, TRTSERVER_ServerOptions* options)
{
  ni::InferenceServer* lserver = new ni::InferenceServer();
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);

  lserver->SetId(loptions->ServerId());
  lserver->SetModelStorePath(loptions->ModelRepositoryPath());
  lserver->SetStrictModelConfigEnabled(loptions->StrictModelConfig());
  lserver->SetStrictReadinessEnabled(loptions->StrictReadiness());
  lserver->SetProfilingEnabled(loptions->Profiling());
  lserver->SetExitTimeoutSeconds(loptions->ExitTimeout());
  lserver->SetTensorFlowSoftPlacementEnabled(
      loptions->TensorFlowSoftPlacement());
  lserver->SetTensorFlowGPUMemoryFraction(
      loptions->TensorFlowGpuMemoryFraction());

  if (!lserver->Init() && loptions->ExitOnError()) {
    delete lserver;
    return TrtServerError::Create(
        ni::RequestStatusCode::INTERNAL,
        "failed to initialize inference server");
  }

  *server = reinterpret_cast<TRTSERVER_Server*>(lserver);
  return nullptr;  // Success
}

TRTSERVER_Error*
TRTSERVER_ServerDelete(TRTSERVER_Server* server)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  if (lserver != nullptr) {
    lserver->Stop();
  }
  delete lserver;
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

  ni::RequestStatus request_status;
  lserver->HandleHealth(&request_status, live, "live");
  return TrtServerError::Create(request_status.code(), request_status.msg());
}

TRTSERVER_Error*
TRTSERVER_ServerIsReady(TRTSERVER_Server* server, bool* ready)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::HEALTH);

  ni::RequestStatus request_status;
  lserver->HandleHealth(&request_status, ready, "ready");
  return TrtServerError::Create(request_status.code(), request_status.msg());
}

TRTSERVER_Error*
TRTSERVER_ServerStatus(TRTSERVER_Server* server, TRTSERVER_Protobuf** status)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  ni::ServerStatTimerScoped timer(
      lserver->StatusManager(), ni::ServerStatTimerScoped::Kind::STATUS);

  ni::RequestStatus request_status;
  ni::ServerStatus server_status;
  lserver->HandleStatus(&request_status, &server_status, std::string());
  if (request_status.code() == ni::RequestStatusCode::SUCCESS) {
    TrtServerProtobuf* protobuf = new TrtServerProtobuf(server_status);
    *status = reinterpret_cast<TRTSERVER_Protobuf*>(protobuf);
  }

  return TrtServerError::Create(request_status.code(), request_status.msg());
}

TRTSERVER_Error*
TRTSERVER_ServerModelStatus(
    TRTSERVER_Server* server, TRTSERVER_Protobuf** status,
    const char* model_name)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  ni::RequestStatus request_status;
  ni::ServerStatus server_status;
  lserver->HandleStatus(
      &request_status, &server_status, std::string(model_name));
  if (request_status.code() == ni::RequestStatusCode::SUCCESS) {
    TrtServerProtobuf* protobuf = new TrtServerProtobuf(server_status);
    *status = reinterpret_cast<TRTSERVER_Protobuf*>(protobuf);
  }

  return TrtServerError::Create(request_status.code(), request_status.msg());
}

TRTSERVER_Error*
TRTSERVER_ServerInferAsync(
    TRTSERVER_Server* server,
    TRTSERVER_InferenceRequestProvider* request_provider,
    void* http_response_provider_hack,
    TRTSERVER_InferenceCompleteFn_t complete_fn, void* userp)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  TrtServerRequestProvider* lprovider =
      reinterpret_cast<TrtServerRequestProvider*>(request_provider);
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
  if (http_response_provider_hack != nullptr) {
    std::shared_ptr<ni::HTTPInferResponseProvider> http_response_provider;
    RETURN_IF_STATUS_ERROR(ni::HTTPInferResponseProvider::Create(
        reinterpret_cast<evbuffer*>(http_response_provider_hack),
        *(lprovider->Backend()), *request_header,
        lprovider->Backend()->GetLabelProvider(), &http_response_provider));
    infer_response_provider = http_response_provider;
  } else {
    std::shared_ptr<ni::DelegatingInferResponseProvider> del_response_provider;
    RETURN_IF_STATUS_ERROR(ni::DelegatingInferResponseProvider::Create(
        *request_header, lprovider->Backend()->GetLabelProvider(),
        &del_response_provider));
    infer_response_provider = del_response_provider;
  }

  auto request_status = std::make_shared<ni::RequestStatus>();
  lserver->HandleInfer(
      request_status.get(), lprovider->Backend(), infer_request_provider,
      infer_response_provider, infer_stats,
      [infer_stats, timer, request_status, infer_response_provider, server,
       complete_fn, userp]() mutable {
        infer_stats->SetFailed(false);
        timer.reset();

        TrtServerResponse* response =
            new TrtServerResponse(request_status, infer_response_provider);
        complete_fn(
            server, reinterpret_cast<TRTSERVER_InferenceResponse*>(response),
            userp);
      });

  return nullptr;  // Success
}

#ifdef __cplusplus
}
#endif
