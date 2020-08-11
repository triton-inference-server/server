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

#include "src/backends/backend/examples/backend_model.h"

#include "src/backends/backend/examples/backend_utils.h"

namespace nvidia { namespace inferenceserver { namespace backend {

//
// BackendModel
//
TRITONSERVER_Error*
BackendModel::Create(
    TRITONBACKEND_Model* triton_model, BackendModel** backend_model)
{
  TRITONSERVER_Message* config_message;
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(
      triton_model, 1 /* config_version */, &config_message));

  // Get the model configuration as a json string from
  // config_message. We use TritonJson, which is a wrapper that
  // returns nice errors (currently the underlying implementation is
  // rapidjson... but others could be added).
  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

  TritonJson::Value model_config;
  TRITONSERVER_Error* err = model_config.Parse(buffer, byte_size);
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
  RETURN_IF_ERROR(err);

  const char* model_name;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(triton_model, &model_name));

  uint64_t model_version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(triton_model, &model_version));

  const char* repository_path = nullptr;
  TRITONBACKEND_ModelArtifactType repository_artifact_type;
  RETURN_IF_ERROR(TRITONBACKEND_ModelRepository(
      triton_model, &repository_artifact_type, &repository_path));
  if (repository_artifact_type != TRITONBACKEND_ARTIFACT_FILESYSTEM) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("unsupported repository artifact type for model '") +
         model_name + "'")
            .c_str());
  }

  TRITONSERVER_Server* triton_server;
  RETURN_IF_ERROR(TRITONBACKEND_ModelServer(triton_model, &triton_server));

  bool pinned_input = false;
  bool pinned_output = false;
  {
    TritonJson::Value optimization;
    if (model_config.Find("optimization", &optimization)) {
      TritonJson::Value pinned_memory;
      if (model_config.Find("input_pinned_memory", &pinned_memory)) {
        RETURN_IF_ERROR(pinned_memory.MemberAsBool("enable", &pinned_input));
      }
      if (model_config.Find("output_pinned_memory", &pinned_memory)) {
        RETURN_IF_ERROR(pinned_memory.MemberAsBool("enable", &pinned_output));
      }
    }
  }

  *backend_model = new BackendModel(
      triton_server, triton_model, model_name, model_version, repository_path,
      pinned_input, pinned_output, std::move(model_config));
  return nullptr;  // success
}

BackendModel::BackendModel(
    TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
    const char* name, const uint64_t version,
    const std::string& repository_path, const bool enable_pinned_input,
    const bool enable_pinned_output, TritonJson::Value&& model_config)
    : triton_server_(triton_server), triton_model_(triton_model), name_(name),
      version_(version), repository_path_(repository_path),
      model_config_(std::move(model_config)),
      enable_pinned_input_(enable_pinned_input),
      enable_pinned_output_(enable_pinned_output),
      supports_batching_initialized_(false), supports_batching_(false)
{
}

TRITONSERVER_Error*
BackendModel::SupportsFirstDimBatching(bool* supports)
{
  // We can't determine this during model initialization because
  // TRITONSERVER_ServerModelBatchProperties can't be called until the
  // model is loaded. So we just cache it here.
  if (!supports_batching_initialized_) {
    uint32_t flags = 0;
    RETURN_IF_ERROR(TRITONSERVER_ServerModelBatchProperties(
        triton_server_, name_.c_str(), version_, &flags, nullptr /* voidp */));
    supports_batching_ = ((flags & TRITONSERVER_BATCH_FIRST_DIM) != 0);
    supports_batching_initialized_ = true;
  }

  *supports = supports_batching_;
  return nullptr;  // success
}

}}}  // namespace nvidia::inferenceserver::backend
