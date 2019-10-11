// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "src/backends/onnx/loader.h"

#include <future>
#include <thread>
#include "src/backends/onnx/onnx_utils.h"

namespace nvidia { namespace inferenceserver {

OnnxLoader* OnnxLoader::loader = nullptr;

OnnxLoader::~OnnxLoader()
{
  if (env_ != nullptr) {
    ort_api->ReleaseEnv(env_);
  }
}

Status
OnnxLoader::Init()
{
  if (loader == nullptr) {
    OrtEnv* env;
    // If needed, provide custom logger with
    // ort_api->CreateEnvWithCustomLogger()
    OrtStatus* status =
        ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "log", &env);
    loader = new OnnxLoader(env);
    RETURN_IF_ORT_ERROR(status);
  } else {
    return Status(
        RequestStatusCode::ALREADY_EXISTS,
        "OnnxLoader singleton already initialized");
  }
  return Status::Success;
}

void
OnnxLoader::TryRelease(bool decrement_session_cnt)
{
  std::lock_guard<std::mutex> lk(loader->mu_);
  if (decrement_session_cnt) {
    loader->live_session_cnt_--;
  }

  if (loader->closing_ && (loader->live_session_cnt_ == 0)) {
    delete loader;
    loader = nullptr;
  }
}

Status
OnnxLoader::Stop()
{
  if (loader != nullptr) {
    loader->closing_ = true;
    TryRelease(false);
  } else {
    return Status(
        RequestStatusCode::UNAVAILABLE,
        "OnnxLoader singleton has not been initialized");
  }
  return Status::Success;
}

Status
OnnxLoader::LoadSession(
    const std::string& model_data, const OrtSessionOptions* session_options,
    OrtSession** session)
{
  if (loader != nullptr) {
    {
      std::lock_guard<std::mutex> lk(loader->mu_);
      if (loader->closing_) {
        return Status(
            RequestStatusCode::UNAVAILABLE, "OnnxLoader has been stopped");
      } else {
        loader->live_session_cnt_++;
      }
    }

    OrtStatus* status = ort_api->CreateSessionFromArray(
        loader->env_, model_data.c_str(), model_data.size(), session_options,
        session);

    if (status != nullptr) {
      TryRelease(true);
    }
    RETURN_IF_ORT_ERROR(status);
  } else {
    return Status(
        RequestStatusCode::UNAVAILABLE,
        "OnnxLoader singleton has not been initialized");
  }
  return Status::Success;
}

Status
OnnxLoader::UnloadSession(OrtSession* session)
{
  if (loader != nullptr) {
    ort_api->ReleaseSession(session);
    TryRelease(true);
  } else {
    return Status(
        RequestStatusCode::UNAVAILABLE,
        "OnnxLoader singleton has not been initialized");
  }
  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
