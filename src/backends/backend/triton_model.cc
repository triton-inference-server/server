// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "src/backends/backend/triton_model.h"

#include <vector>
#include "src/backends/backend/tritonbackend.h"
#include "src/core/filesystem.h"
#include "src/core/logging.h"
#include "src/core/model_config_utils.h"
#include "src/core/server_message.h"

namespace nvidia { namespace inferenceserver {

Status
TritonModel::Create(
    const std::string& model_repository_path, const std::string& model_name,
    const int64_t version, const ModelConfig& model_config,
    const double min_compute_capability, std::unique_ptr<TritonModel>* model)
{
  model->reset();

  // The model configuration must specify a backend. The name of the
  // corresponding shared library must be libtriton_<backend>.so.
  if (model_config.backend().empty()) {
    return Status(
        Status::Code::INVALID_ARG,
        "must specify 'backend' for '" + model_config.name() + "'");
  }

  const std::string backend_libname =
      "libtriton_" + model_config.backend() + ".so";

  // Get the path to the backend shared library. Search path is
  // version directory, model directory, global backend directory.
  const auto version_path =
      JoinPath({model_repository_path, model_name, std::to_string(version)});
  const auto model_path = JoinPath({model_repository_path, model_name});
  const std::string global_path =
      "/opt/tritonserver/backends";  // FIXME need cmdline flag
  const std::vector<std::string> search_paths = {version_path, model_path,
                                                 global_path};

  std::string backend_libpath;
  for (const auto& path : search_paths) {
    const auto full_path = JoinPath({path, backend_libname});
    bool exists = false;
    RETURN_IF_ERROR(FileExists(full_path, &exists));
    if (exists) {
      backend_libpath = full_path;
      break;
    }
  }

  if (backend_libpath.empty()) {
    return Status(
        Status::Code::INVALID_ARG, "unable to find '" + backend_libname +
                                       "' for model '" + model_config.name() +
                                       "', searched: " + version_path + ", " +
                                       model_path + ", " + global_path);
  }

  // Find the backend
  std::shared_ptr<TritonBackend> backend;
  RETURN_IF_ERROR(TritonBackendManager::CreateBackend(
      model_config.backend(), backend_libpath, &backend));

  // Create and initialize the model.
  std::unique_ptr<TritonModel> local_model(
      new TritonModel(model_path, backend, min_compute_capability));
  RETURN_IF_ERROR(
      local_model->Init(version_path, model_config, "" /* platform */));

  TRITONBACKEND_Model* triton_model =
      reinterpret_cast<TRITONBACKEND_Model*>(local_model.get());
  TritonBackend::TritonModelExecFn_t model_exec_fn = backend->ModelExecFn();

  // Model initialization is optional... The TRITONBACKEND_Model
  // object is this TritonModel object.
  if (backend->ModelInitFn() != nullptr) {
    RETURN_IF_TRITONSERVER_ERROR(backend->ModelInitFn()(triton_model));
  }

  // Create a scheduler with 1 thread. The backend is already
  // initialized so there is no need to have the scheduler thread call
  // any initialization.
  RETURN_IF_ERROR(local_model->SetConfiguredScheduler(
      1 /* runner_cnt */,
      [](uint32_t runner_idx) -> Status { return Status::Success; },
      [model_exec_fn, triton_model](
          uint32_t runner_idx,
          std::vector<std::unique_ptr<InferenceRequest>>&& requests) {
        // There is only a single thread calling this function so can
        // use a static vector to avoid needing to malloc each time.
        static std::vector<TRITONBACKEND_Request*> triton_requests(1024);
        triton_requests.clear();
        for (auto& r : requests) {
          triton_requests.push_back(
              reinterpret_cast<TRITONBACKEND_Request*>(r.release()));
        }

        // If there is an error then we retain ownership of 'requests'
        // and must send error responses.
        TRITONSERVER_Error* err = model_exec_fn(
            triton_model, &triton_requests[0], triton_requests.size());
        if (err != nullptr) {
          Status status = Status(
              TritonCodeToStatusCode(TRITONSERVER_ErrorCode(err)),
              TRITONSERVER_ErrorMessage(err));
          for (TRITONBACKEND_Request* tr : triton_requests) {
            std::unique_ptr<InferenceRequest> ur(
                reinterpret_cast<InferenceRequest*>(tr));
            InferenceRequest::RespondIfError(
                ur, status, true /* release_requests */);
          }

          TRITONSERVER_ErrorDelete(err);
        }

        return Status::Success;
      }));

  *model = std::move(local_model);
  return Status::Success;
}

TritonModel::TritonModel(
    const std::string& model_path,
    const std::shared_ptr<TritonBackend>& backend,
    const double min_compute_capability)
    : InferenceBackend(min_compute_capability), model_path_(model_path),
      backend_(backend), state_(nullptr)
{
}

TritonModel::~TritonModel()
{
  // Model finalization is optional... The TRITONBACKEND_Model
  // object is this TritonModel object.
  if (backend_->ModelFiniFn() != nullptr) {
    LOG_TRITONSERVER_ERROR(
        backend_->ModelFiniFn()(reinterpret_cast<TRITONBACKEND_Model*>(this)),
        "failed finalizing model");
  }
}

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_ModelName(TRITONBACKEND_Model* model, const char** name)
{
  TritonModel* tm = reinterpret_cast<TritonModel*>(model);
  *name = tm->Name().c_str();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelVersion(TRITONBACKEND_Model* model, uint64_t* version)
{
  TritonModel* tm = reinterpret_cast<TritonModel*>(model);
  *version = tm->Version();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelRepositoryPath(TRITONBACKEND_Model* model, const char** path)
{
  TritonModel* tm = reinterpret_cast<TritonModel*>(model);
  *path = tm->ModelPath().c_str();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelConfig(
    TRITONBACKEND_Model* model, TRITONSERVER_Message** model_config)
{
  TritonModel* tm = reinterpret_cast<TritonModel*>(model);

  std::string model_config_json;
  Status status = ModelConfigToJson(tm->Config(), &model_config_json);
  if (!status.IsOk()) {
    return TRITONSERVER_ErrorNew(
        StatusCodeToTritonCode(status.StatusCode()), status.Message().c_str());
  }

  *model_config = reinterpret_cast<TRITONSERVER_Message*>(
      new TritonServerMessage(std::move(model_config_json)));

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelBackend(
    TRITONBACKEND_Model* model, TRITONBACKEND_Backend** backend)
{
  TritonModel* tm = reinterpret_cast<TritonModel*>(model);
  *backend = reinterpret_cast<TRITONBACKEND_Backend*>(tm->Backend().get());
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelState(TRITONBACKEND_Model* model, void** state)
{
  TritonModel* tm = reinterpret_cast<TritonModel*>(model);
  *state = tm->State();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelSetState(TRITONBACKEND_Model* model, void* state)
{
  TritonModel* tm = reinterpret_cast<TritonModel*>(model);
  tm->SetState(state);
  return nullptr;  // success
}

}  // extern C

}}  // namespace nvidia::inferenceserver
