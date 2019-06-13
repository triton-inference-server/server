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

#include "src/core/request_status.pb.h"
#include "src/core/server.h"

namespace ni = nvidia::inferenceserver;

namespace {

//
// TrtServerError
//
// Implementation for TRTSERVER_Error.
//
class TrtServerError {
 public:
  static TRTSERVER_Error* Create(
      ni::RequestStatusCode code, const std::string& msg);
  ni::RequestStatusCode Code() const { return code_; }
  const std::string& Message() const { return msg_; }

 private:
  TrtServerError(ni::RequestStatusCode code, const std::string& msg);

  ni::RequestStatusCode code_;
  const std::string msg_;
};

TRTSERVER_Error*
TrtServerError::Create(ni::RequestStatusCode code, const std::string& msg)
{
  // If 'code' is success then return nullptr as that indicates
  // success
  if (code == ni::RequestStatusCode::SUCCESS) {
    return nullptr;
  }

  return reinterpret_cast<TRTSERVER_Error*>(new TrtServerError(code, msg));
}

TrtServerError::TrtServerError(
    ni::RequestStatusCode code, const std::string& msg)
    : code_(code), msg_(msg)
{
}

#define RETURN_IF_STATUS_ERROR(S)                                         \
  do {                                                                    \
    const ni::Status& status__ = (S);                                     \
    if (status__.Code() != RequestStatusCode::SUCCESS) {                  \
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
  TrtServerProtobuf(google::protobuf::MessageLite* msg) : msg_(msg) {}

  void Serialize(const char** base, size_t* byte_size);

 private:
  std::unique_ptr<google::protobuf::MessageLite> msg_;
  std::string serialized_;
};

void
TrtServerProtobuf::Serialize(const char** base, size_t* byte_size)
{
  if (serialized_.empty()) {
    msg_->SerializeToString(&serialized_);
  }

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
  const std::string& ModelRepositoryPath() const { return repo_path_; }
  void SetModelRepositoryPath(const char* path) { repo_path_ = path; }

 private:
  std::string repo_path_;
};

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

//
// TRTSERVER_Error
//
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
  switch (lerror->Code()) {
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

const char*
TRTSERVER_ErrorCodeString(TRTSERVER_Error* error)
{
  TrtServerError* lerror = reinterpret_cast<TrtServerError*>(error);
  return ni::RequestStatusCode_Name(lerror->Code()).c_str();
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
TRTSERVER_ServerOptionsSetModelRepositoryPath(
    TRTSERVER_ServerOptions* options, const char* model_repository_path)
{
  TrtServerOptions* loptions = reinterpret_cast<TrtServerOptions*>(options);
  loptions->SetModelRepositoryPath(model_repository_path);
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

  lserver->SetModelStorePath(loptions->ModelRepositoryPath());

  if (!lserver->Init()) {
    delete lserver;
    return TrtServerError::Create(
        ni::RequestStatusCode::INVALID_ARG,
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
TRTSERVER_ServerIsLive(TRTSERVER_Server* server, bool* live)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  ni::RequestStatus request_status;
  lserver->HandleHealth(&request_status, live, "live");
  return TrtServerError::Create(request_status.code(), request_status.msg());
}

TRTSERVER_Error*
TRTSERVER_ServerIsReady(TRTSERVER_Server* server, bool* ready)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  ni::RequestStatus request_status;
  lserver->HandleHealth(&request_status, ready, "ready");
  return TrtServerError::Create(request_status.code(), request_status.msg());
}

TRTSERVER_Error*
TRTSERVER_ServerStatus(TRTSERVER_Server* server, TRTSERVER_Protobuf** status)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  ni::RequestStatus request_status;
  ni::ServerStatus* server_status = new ni::ServerStatus();
  lserver->HandleStatus(&request_status, server_status, std::string());
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
  ni::ServerStatus* server_status = new ni::ServerStatus();
  lserver->HandleStatus(
      &request_status, server_status, std::string(model_name));
  if (request_status.code() == ni::RequestStatusCode::SUCCESS) {
    TrtServerProtobuf* protobuf = new TrtServerProtobuf(server_status);
    *status = reinterpret_cast<TRTSERVER_Protobuf*>(protobuf);
  }

  return TrtServerError::Create(request_status.code(), request_status.msg());
}

#ifdef __cplusplus
}
#endif
