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
}  // namespace

Status
FileSystemTypeToTritonArtifactType(
    const FileSystemType type, TRITONREPOAGENT_ArtifactType* converted_type)
{
  switch (type) {
    case FileSystemType::LOCAL:
      *converted_type = TRITONREPOAGENT_ARTIFACT_FILESYSTEM;
      return Status::Success;
    default:
      return Status(
          Status::Code::INTERNAL,
          "Can not convert to matching artifact type for filesystem type " +
              FileSystemTypeString(type));
  }
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
      LOG_ERROR << "~TritonRepoAgent: " << Status(TritonCodeToStatusCode(TRITONSERVER_ErrorCode(err)), TRITONSERVER_ErrorMessage(err)).AsString();
      TRITONSERVER_ErrorDelete(err);
    }
    ;  
  }
  auto status = CloseLibraryHandle(dlhandle_);
  if (!status.IsOk()) {
    LOG_ERROR << "~TritonRepoAgent: " << status.AsString();
  }
}

//
// TritonRepoAgentModel
//
TritonRepoAgentModel::~TritonRepoAgentModel()
{
  if (!acquired_location_.empty()) {
    auto status = DeleteDirectory(acquired_location_);
    if (!status.IsOk()) {
      LOG_ERROR << "Failed to delete previously committed location '"
                << acquired_location_ << "': " << status.AsString();
    }
  }
  if (has_committed_ && !committed_location_.empty()) {
    auto status = DeleteDirectory(committed_location_);
    if (!status.IsOk()) {
      LOG_ERROR << "Failed to delete previously committed location '"
                << committed_location_ << "': " << status.AsString();
    }
  }
}

Status
TritonRepoAgentModel::InvokeAgents(const TRITONREPOAGENT_ActionType action_type)
{
  while (current_idx_ != agents_.size()) {
    auto agent_ptr = agents_[current_idx_].first.get();
    RETURN_IF_TRITONSERVER_ERROR(agent_ptr->AgentModelActionFn()(
        reinterpret_cast<TRITONREPOAGENT_Agent*>(agent_ptr),
        reinterpret_cast<TRITONREPOAGENT_AgentModel*>(this), action_type));
    current_idx_++;
  }
  return Status::Success;
}

Status
TritonRepoAgentModel::Location(FileSystemType* type, const char** location)
{
  *type = committed_type_;
  *location = committed_location_.c_str();
  return Status::Success;
}

Status
TritonRepoAgentModel::AcquireMutableLocation(
    const FileSystemType preferred_type, FileSystemType* actual_type,
    const char** location)
{
  if (acquired_location_.empty()) {
    std::string lacquired_location;
    acquired_type_ = preferred_type;
    auto status = MakeTemporaryDirectory(acquired_type_, &lacquired_location);
    if (!status.IsOk()) {
      if (acquired_type_ != FileSystemType::LOCAL) {
        acquired_type_ = FileSystemType::LOCAL;
        RETURN_IF_ERROR(
            MakeTemporaryDirectory(acquired_type_, &lacquired_location));
      } else {
        return status;
      }
    }
    acquired_location_.swap(lacquired_location);
  }

  *actual_type = acquired_type_;
  *location = acquired_location_.c_str();
  return Status::Success;
}

Status
TritonRepoAgentModel::CommitMutableLocation()
{
  if (acquired_location_.empty()) {
    return Status(
        Status::Code::UNAVAILABLE, "No mutable location to be commited");
  }
  if (has_committed_) {
    auto status = DeleteDirectory(committed_location_);
    if (!status.IsOk()) {
      LOG_ERROR << "Failed to delete previously committed location '"
                << committed_location_ << "': " << status.AsString();
    }
  }
  committed_location_ = std::move(acquired_location_);
  has_committed_ = true;
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
TritonRepoAgentManager::CreateAgentModel(
    const std::string& model_dir, const inference::ModelConfig& config,
    std::shared_ptr<TritonRepoAgentModel>* agent_model)
{
  auto& singleton_manager = Singleton();
  std::lock_guard<std::mutex> lock(singleton_manager.mu_);

  // Get the path to the backend shared library. Search path is
  // model directory, global agent directory.
  // FIXME expose global path as Triton option
  static const std::string global_path = "/opt/tritonserver/agents";
  const std::vector<std::string> search_paths = {model_dir, global_path};

  std::vector<
      std::pair<std::shared_ptr<TritonRepoAgent>, TritonRepoAgent::Parameters>>
      agents;
  if (config.has_model_repository_agents()) {
    for (const auto& agent_config : config.model_repository_agents().agents()) {
      std::string agent_libname =
          TritonRepoAgentLibraryName(agent_config.name());
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
                agent_config.name() + "', searched: " + model_dir + ", " +
                global_path);
      }

      std::shared_ptr<TritonRepoAgent> agent;
      const auto& itr = singleton_manager.agent_map_.find(libpath);
      if (itr != singleton_manager.agent_map_.end()) {
        // Found in map. If the weak_ptr is still valid that means that
        // there are other models using the backend and we just reuse that
        // same backend. If the weak_ptr is not valid then backend has
        // been unloaded so we need to remove the weak_ptr from the map
        // and create the backend again.
        agent = itr->second.lock();
        if (agent != nullptr) {
          return Status::Success;
        }

        singleton_manager.agent_map_.erase(itr);
      } else {
        RETURN_IF_ERROR(
            TritonRepoAgent::Create(agent_config.name(), libpath, &agent));
        singleton_manager.agent_map_.insert({libpath, agent});
      }
      TritonRepoAgent::Parameters agent_params;
      for (const auto& parameter : agent_config.parameters()) {
        agent_params.emplace_back(parameter.first, parameter.second);
      }
      agents.emplace_back(std::move(agent), std::move(agent_params));
    }
  }
  FileSystemType type;
  RETURN_IF_ERROR(GetFileSystemType(model_dir, &type));
  agent_model->reset(
      new TritonRepoAgentModel(model_dir, type, config, std::move(agents)));

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
  FileSystemType type;
  RETURN_TRITONSERVER_ERROR_IF_ERROR(tam->Location(&type, location));
  RETURN_TRITONSERVER_ERROR_IF_ERROR(
      FileSystemTypeToTritonArtifactType(type, artifact_type));
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONREPOAGENT_ModelRepositoryLocationAcquire(
    TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
    TRITONREPOAGENT_ArtifactType* artifact_type, const char** location)
{
  TritonRepoAgentModel* tam = reinterpret_cast<TritonRepoAgentModel*>(model);
  FileSystemType type = FileSystemType::LOCAL;
  RETURN_TRITONSERVER_ERROR_IF_ERROR(
      tam->AcquireMutableLocation(type, &type, location));
  RETURN_TRITONSERVER_ERROR_IF_ERROR(
      FileSystemTypeToTritonArtifactType(type, artifact_type));
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONREPOAGENT_ModelRepositoryLocationCommit(
    TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
    const char* location)
{
  TritonRepoAgentModel* tam = reinterpret_cast<TritonRepoAgentModel*>(model);
  RETURN_TRITONSERVER_ERROR_IF_ERROR(tam->CommitMutableLocation());
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
TRITONREPOAGENT_ModelParameterCount(
    TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
    uint32_t* count)
{
  TritonRepoAgentModel* tam = reinterpret_cast<TritonRepoAgentModel*>(model);
  *count = tam->CurrentAgentParameters().size();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONREPOAGENT_ModelParameter(
    TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
    const uint32_t index, const char** parameter_name,
    const char** parameter_value)
{
  TritonRepoAgentModel* tam = reinterpret_cast<TritonRepoAgentModel*>(model);
  const auto& params = tam->CurrentAgentParameters();
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

}  // extern C

}}  // namespace nvidia::inferenceserver