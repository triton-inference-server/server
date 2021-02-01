// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "src/core/triton_repo_agent.h"

#include <string>
#include "src/core/logging.h"
#include "src/core/shared_library.h"

namespace nvidia { namespace inferenceserver {

namespace {

// If status is non-OK, return the corresponding TRITONSERVER_Error.
#define RETURN_TRITONSERVER_ERROR_IF_ERROR(S)            \
  do {                                                   \
    const Status& status__ = (S);                        \
    if (!status__.IsOk()) {                              \
      return TRITONSERVER_ErrorNew(                      \
          StatusCodeToTritonCode(status__.StatusCode()), \
          status__.Message().c_str());                   \
    }                                                    \
  } while (false)


std::string
TritonRepoAgentLibraryName(const std::string& agent_name)
{
#ifdef _WIN32
  return std::string("tritonrepoagent_") + agent_name + ".dll";
#else
  return std::string("libtritonrepoagent_") + agent_name + ".so";
#endif
}

Status
FileSystemTypeToTritonArtifactType(
    const FileSystemType type, TRITONREPOAGENT_ArtifactType* converted_type)
{
  switch (type) {
    case FileSystemType::LOCAL:
      *converted_type = TRITONREPOAGENT_ARTIFACT_FILESYSTEM;
      break;
    case FileSystemType::AS:
    case FileSystemType::GCS:
    case FileSystemType::S3:
      *converted_type = TRITONREPOAGENT_ARTIFACT_REMOTE_FILESYSTEM;
      break;
    default:
      return Status(
          Status::Code::INTERNAL,
          "Can not convert to matching artifact type for filesystem type " +
              FileSystemTypeString(type));
  }
  return Status::Success;
}

Status
TritonArtifactTypeToFileSystemType(
    const TRITONREPOAGENT_ArtifactType type, const std::string& path,
    FileSystemType* converted_type)
{
  switch (type) {
    case TRITONREPOAGENT_ARTIFACT_FILESYSTEM:
      *converted_type = FileSystemType::LOCAL;
      return Status::Success;
    case TRITONREPOAGENT_ARTIFACT_REMOTE_FILESYSTEM: {
      RETURN_IF_ERROR(GetFileSystemType(path, converted_type));
      if (*converted_type == FileSystemType::LOCAL) {
        return Status(
          Status::Code::INTERNAL,
          "Can not convert to matching filesystem type from artifact type 'TRITONREPOAGENT_ARTIFACT_REMOTE_FILESYSTEM' with location " +
              path);
      }
      return Status::Success;
      }
    default:
      return Status(
          Status::Code::INTERNAL,
          "Can not convert to matching filesystem type from artifact type " +
              std::to_string(type));
  }
}

}  // namespace

std::string TRITONREPOAGENT_ActionTypeString(const TRITONREPOAGENT_ActionType type)
{
  switch (type)
  {
  case TRITONREPOAGENT_ACTION_LOAD:
      return "TRITONREPOAGENT_ACTION_LOAD";
    case TRITONREPOAGENT_ACTION_LOAD_COMPLETE:
      return "TRITONREPOAGENT_ACTION_LOAD_COMPLETE";
    case TRITONREPOAGENT_ACTION_LOAD_FAIL:
      return "TRITONREPOAGENT_ACTION_LOAD_FAIL";
    case TRITONREPOAGENT_ACTION_UNLOAD:
      return "TRITONREPOAGENT_ACTION_UNLOAD";
    case TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE:
      return "TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE";
  }
  return "Unknown TRITONREPOAGENT_ActionType";
}

//
// TritonRepoAgent
//
Status
TritonRepoAgent::Create(
    const std::string& name, const std::string& libpath,
    std::shared_ptr<TritonRepoAgent>* agent)
{
  std::shared_ptr<TritonRepoAgent> lagent(new TritonRepoAgent(name));
  RETURN_IF_ERROR(OpenLibraryHandle(libpath, &lagent->dlhandle_));
  RETURN_IF_ERROR(GetEntrypoint(
      lagent->dlhandle_, "TRITONREPOAGENT_Initialize", true /* optional */,
      reinterpret_cast<void**>(&lagent->init_fn_)));
  RETURN_IF_ERROR(GetEntrypoint(
      lagent->dlhandle_, "TRITONREPOAGENT_Finalize", true /* optional */,
      reinterpret_cast<void**>(&lagent->fini_fn_)));
  RETURN_IF_ERROR(GetEntrypoint(
      lagent->dlhandle_, "TRITONREPOAGENT_ModelInitialize", true /* optional */,
      reinterpret_cast<void**>(&lagent->model_init_fn_)));
  RETURN_IF_ERROR(GetEntrypoint(
      lagent->dlhandle_, "TRITONREPOAGENT_ModelFinalize", true /* optional */,
      reinterpret_cast<void**>(&lagent->model_fini_fn_)));
  RETURN_IF_ERROR(GetEntrypoint(
      lagent->dlhandle_, "TRITONREPOAGENT_ModelAction", false /* optional */,
      reinterpret_cast<void**>(&lagent->model_action_fn_)));
  // Initialize if needed
  if (lagent->init_fn_ != nullptr) {
    RETURN_IF_TRITONSERVER_ERROR(lagent->init_fn_(
        reinterpret_cast<TRITONREPOAGENT_Agent*>(lagent.get())));
  }
  *agent = std::move(lagent);
  return Status::Success;
}

TritonRepoAgent::~TritonRepoAgent()
{
  // Finalize if needed
  if (fini_fn_ != nullptr) {
    auto err = fini_fn_(reinterpret_cast<TRITONREPOAGENT_Agent*>(this));
    if (err != nullptr) {
      LOG_ERROR << "~TritonRepoAgent: "
                << Status(
                       TritonCodeToStatusCode(TRITONSERVER_ErrorCode(err)),
                       TRITONSERVER_ErrorMessage(err))
                       .AsString();
      TRITONSERVER_ErrorDelete(err);
    };
  }
  auto status = CloseLibraryHandle(dlhandle_);
  if (!status.IsOk()) {
    LOG_ERROR << "~TritonRepoAgent: " << status.AsString();
  }
}

//
// TritonRepoAgentModel
//
Status
TritonRepoAgentModel::Create(
      const inference::ModelConfig& config,
      const std::shared_ptr<TritonRepoAgent> agent,
      TritonRepoAgent::Parameters&& agent_parameters,
      std::unique_ptr<TritonRepoAgentModel>* agent_model)
{
  std::unique_ptr<TritonRepoAgentModel> lagent_model(new TritonRepoAgentModel(config, agent, std::move(agent_parameters)));
  if (agent->AgentModelInitFn() != nullptr) {
    RETURN_IF_TRITONSERVER_ERROR(agent->AgentModelInitFn()(
      reinterpret_cast<TRITONREPOAGENT_Agent*>(agent.get()),
      reinterpret_cast<TRITONREPOAGENT_AgentModel*>(lagent_model.get())));
  }
  *agent_model = std::move(lagent_model);
  return Status::Success;
}

TritonRepoAgentModel::~TritonRepoAgentModel()
{
  if (agent_->AgentModelFiniFn() != nullptr) {
    LOG_TRITONSERVER_ERROR(agent_->AgentModelFiniFn()(
      reinterpret_cast<TRITONREPOAGENT_Agent*>(agent_.get()),
      reinterpret_cast<TRITONREPOAGENT_AgentModel*>(this)), "~TritonRepoAgentModel");
  }
  if (!acquired_location_.empty()) {
    DeleteMutableLocation();
  }
}

Status
TritonRepoAgentModel::InvokeAgent(const TRITONREPOAGENT_ActionType action_type)
{
  RETURN_IF_TRITONSERVER_ERROR(agent_->AgentModelActionFn()(
      reinterpret_cast<TRITONREPOAGENT_Agent*>(agent_.get()),
      reinterpret_cast<TRITONREPOAGENT_AgentModel*>(this), action_type));
  return Status::Success;
}

Status
TritonRepoAgentModel::SetLocation(const FileSystemType type, const std::string& location)
{
  type_ = type;
  location_ = location;
  return Status::Success;
}

Status
TritonRepoAgentModel::Location(FileSystemType* type, const char** location)
{
  if (location_.empty()) {
    return Status(Status::Code::INTERNAL, "Model repository location is not set");
  }
  *type = type_;
  *location = location_.c_str();
  return Status::Success;
}

Status
TritonRepoAgentModel::AcquireMutableLocation(
    const FileSystemType type, const char** location)
{
  if (acquired_location_.empty()) {
    std::string lacquired_location;
    RETURN_IF_ERROR(MakeTemporaryDirectory(type, &lacquired_location));
    acquired_location_.swap(lacquired_location);
  }

  if (type != acquired_type_) {
    return Status(Status::Code::INVALID_ARG, "The requested filesystem type is different from existing acquired location");
  }
  *location = acquired_location_.c_str();
  return Status::Success;
}

Status
TritonRepoAgentModel::DeleteMutableLocation()
{
  if (acquired_location_.empty()) {
    return Status(
        Status::Code::UNAVAILABLE, "No mutable location to be deleted");
  }

  auto status = DeleteDirectory(acquired_location_);
  if (!status.IsOk()) {
    LOG_ERROR << "Failed to delete previously acquired location '"
              << acquired_location_ << "': " << status.AsString();
  }
  acquired_location_.clear();
  return Status::Success;
}

//
// TritonRepoAgentManager
//
TritonRepoAgentManager&
TritonRepoAgentManager::Singleton()
{
  static TritonRepoAgentManager triton_repo_agent_manager;
  return triton_repo_agent_manager;
}

Status
TritonRepoAgentManager::CreateAgent(
    const std::string& model_dir, const std::string& agent_name,
    std::shared_ptr<TritonRepoAgent>* agent)
{
  auto& singleton_manager = Singleton();
  std::lock_guard<std::mutex> lock(singleton_manager.mu_);

  // Get the path to the backend shared library. Search path is
  // model directory, global agent directory.
  // FIXME expose global path as Triton option
  static const std::string global_path = "/opt/tritonserver/agents";
  const std::vector<std::string> search_paths = {model_dir, global_path};

  std::string agent_libname =
      TritonRepoAgentLibraryName(agent_name);
  std::string libpath;
  for (const auto& path : search_paths) {
    const auto full_path = JoinPath({path, agent_libname});
    bool exists = false;
    RETURN_IF_ERROR(FileExists(full_path, &exists));
    if (exists) {
      libpath = full_path;
      break;
    }
  }

  if (libpath.empty()) {
    return Status(
        Status::Code::INVALID_ARG,
        "unable to find '" + agent_libname + "' for repo agent '" +
            agent_name + "', searched: " + model_dir + ", " +
            global_path);
  }

  const auto& itr = singleton_manager.agent_map_.find(libpath);
  if (itr != singleton_manager.agent_map_.end()) {
    // Found in map. If the weak_ptr is still valid that means that
    // there are other models using the backend and we just reuse that
    // same backend. If the weak_ptr is not valid then backend has
    // been unloaded so we need to remove the weak_ptr from the map
    // and create the backend again.
    *agent = itr->second.lock();
    if (*agent != nullptr) {
      return Status::Success;
    }

    singleton_manager.agent_map_.erase(itr);
  } else {
    RETURN_IF_ERROR(
        TritonRepoAgent::Create(agent_name, libpath, agent));
    singleton_manager.agent_map_.insert({libpath, *agent});
  }

  return Status::Success;
}

extern "C" {

TRITONSERVER_Error*
TRITONREPOAGENT_ApiVersion(uint32_t* major, uint32_t* minor)
{
  *major = TRITONREPOAGENT_API_VERSION_MAJOR;
  *minor = TRITONREPOAGENT_API_VERSION_MINOR;
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONREPOAGENT_ModelRepositoryLocation(
    TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
    TRITONREPOAGENT_ArtifactType* artifact_type, const char** location)
{
  TritonRepoAgentModel* tam = reinterpret_cast<TritonRepoAgentModel*>(model);
  FileSystemType type = FileSystemType::LOCAL;
  RETURN_TRITONSERVER_ERROR_IF_ERROR(tam->Location(&type, location));
  RETURN_TRITONSERVER_ERROR_IF_ERROR(
      FileSystemTypeToTritonArtifactType(type, artifact_type));
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONREPOAGENT_ModelRepositoryLocationAcquire(
    TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
    const TRITONREPOAGENT_ArtifactType artifact_type, const char** location)
{
  if (artifact_type != TRITONREPOAGENT_ARTIFACT_FILESYSTEM) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Unexpected artifact type, expects 'TRITONREPOAGENT_ARTIFACT_FILESYSTEM'");
  }
  TritonRepoAgentModel* tam = reinterpret_cast<TritonRepoAgentModel*>(model);
  RETURN_TRITONSERVER_ERROR_IF_ERROR(
      tam->AcquireMutableLocation(FileSystemType::LOCAL, location));
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONREPOAGENT_ModelRepositoryLocationDelete(
    TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
    const char* location)
{
  TritonRepoAgentModel* tam = reinterpret_cast<TritonRepoAgentModel*>(model);
  RETURN_TRITONSERVER_ERROR_IF_ERROR(tam->DeleteMutableLocation());
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONREPOAGENT_ModelRepositoryLocationUpdate(
    TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
    const TRITONREPOAGENT_ArtifactType artifact_type, const char* location)
{
  TritonRepoAgentModel* tam = reinterpret_cast<TritonRepoAgentModel*>(model);
  FileSystemType type;
  RETURN_TRITONSERVER_ERROR_IF_ERROR(TritonArtifactTypeToFileSystemType(artifact_type, location, &type));
  RETURN_TRITONSERVER_ERROR_IF_ERROR(tam->SetLocation(type, location));
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONREPOAGENT_ModelParameterCount(
    TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
    uint32_t* count)
{
  TritonRepoAgentModel* tam = reinterpret_cast<TritonRepoAgentModel*>(model);
  *count = tam->AgentParameters().size();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONREPOAGENT_ModelParameter(
    TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
    const uint32_t index, const char** parameter_name,
    const char** parameter_value)
{
  TritonRepoAgentModel* tam = reinterpret_cast<TritonRepoAgentModel*>(model);
  const auto& params = tam->AgentParameters();
  if (index >= params.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "index out of range for model parameters");
  }
  *parameter_name = params[index].first.c_str();
  *parameter_value = params[index].second.c_str();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONREPOAGENT_ModelConfig(
    TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
    const uint32_t config_version, TRITONSERVER_Message** model_config)
{
  TritonRepoAgentModel* tam = reinterpret_cast<TritonRepoAgentModel*>(model);
  std::string model_config_json;
  RETURN_TRITONSERVER_ERROR_IF_ERROR(
      ModelConfigToJson(tam->Config(), config_version, &model_config_json));
  return TRITONSERVER_MessageNewFromSerializedJson(
      model_config, model_config_json.c_str(), model_config_json.length());
}

TRITONSERVER_Error* TRITONREPOAGENT_ModelState(
    TRITONREPOAGENT_AgentModel* model, void** state)
{
  TritonRepoAgentModel* tam = reinterpret_cast<TritonRepoAgentModel*>(model);
  *state = tam->State();
  return nullptr;  // success
}

TRITONSERVER_Error* TRITONREPOAGENT_ModelSetState(
    TRITONREPOAGENT_AgentModel* model, void* state)
{
  TritonRepoAgentModel* tam = reinterpret_cast<TritonRepoAgentModel*>(model);
  tam->SetState(state);
  return nullptr;  // success
}

TRITONSERVER_Error* TRITONREPOAGENT_AgentState(
    TRITONREPOAGENT_Agent* agent, void** state)
{
  TritonRepoAgent* ta = reinterpret_cast<TritonRepoAgent*>(agent);
  *state = ta->State();
  return nullptr;  // success
}

TRITONSERVER_Error* TRITONREPOAGENT_AgentSetState(
    TRITONREPOAGENT_Agent* agent, void* state)
{
  TritonRepoAgent* ta = reinterpret_cast<TritonRepoAgent*>(agent);
  ta->SetState(state);
  return nullptr;  // success
}

}  // extern C

}}  // namespace nvidia::inferenceserver