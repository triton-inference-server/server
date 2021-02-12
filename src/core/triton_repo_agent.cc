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
#include "src/core/filesystem.h"
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

}  // namespace

std::string
TritonRepoAgentLibraryName(const std::string& agent_name)
{
#ifdef _WIN32
  return std::string("tritonrepoagent_") + agent_name + ".dll";
#else
  return std::string("libtritonrepoagent_") + agent_name + ".so";
#endif
}

std::string
TRITONREPOAGENT_ActionTypeString(const TRITONREPOAGENT_ActionType type)
{
  switch (type) {
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

std::string
TRITONREPOAGENT_ArtifactTypeString(const TRITONREPOAGENT_ArtifactType type)
{
  switch (type) {
    case TRITONREPOAGENT_ARTIFACT_FILESYSTEM:
      return "TRITONREPOAGENT_ARTIFACT_FILESYSTEM";
    case TRITONREPOAGENT_ARTIFACT_REMOTE_FILESYSTEM:
      return "TRITONREPOAGENT_ARTIFACT_REMOTE_FILESYSTEM";
  }
  return "Unknown TRITONREPOAGENT_ArtifactType";
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
    const TRITONREPOAGENT_ArtifactType type, const std::string& location,
    const inference::ModelConfig& config,
    const std::shared_ptr<TritonRepoAgent> agent,
    const TritonRepoAgent::Parameters& agent_parameters,
    std::unique_ptr<TritonRepoAgentModel>* agent_model)
{
  std::unique_ptr<TritonRepoAgentModel> lagent_model(new TritonRepoAgentModel(
      type, location, config, agent, agent_parameters));
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
  // Need to ensure the proper lifecycle is informed
  if (action_type_set_) {
    switch (current_action_type_) {
      case TRITONREPOAGENT_ACTION_LOAD:
        LOG_TRITONSERVER_ERROR(
            agent_->AgentModelActionFn()(
                reinterpret_cast<TRITONREPOAGENT_Agent*>(agent_.get()),
                reinterpret_cast<TRITONREPOAGENT_AgentModel*>(this),
                TRITONREPOAGENT_ACTION_LOAD_FAIL),
            "Inform TRITONREPOAGENT_ACTION_LOAD_FAIL");
        break;
      case TRITONREPOAGENT_ACTION_LOAD_COMPLETE:
        LOG_TRITONSERVER_ERROR(
            agent_->AgentModelActionFn()(
                reinterpret_cast<TRITONREPOAGENT_Agent*>(agent_.get()),
                reinterpret_cast<TRITONREPOAGENT_AgentModel*>(this),
                TRITONREPOAGENT_ACTION_UNLOAD),
            "Inform TRITONREPOAGENT_ACTION_UNLOAD");
        // Fallthough is not yet an language feature until C++17
        LOG_TRITONSERVER_ERROR(
            agent_->AgentModelActionFn()(
                reinterpret_cast<TRITONREPOAGENT_Agent*>(agent_.get()),
                reinterpret_cast<TRITONREPOAGENT_AgentModel*>(this),
                TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE),
            "Inform TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE");
        break;
      case TRITONREPOAGENT_ACTION_UNLOAD:
        LOG_TRITONSERVER_ERROR(
            agent_->AgentModelActionFn()(
                reinterpret_cast<TRITONREPOAGENT_Agent*>(agent_.get()),
                reinterpret_cast<TRITONREPOAGENT_AgentModel*>(this),
                TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE),
            "Inform TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE");
        break;
      case TRITONREPOAGENT_ACTION_LOAD_FAIL:
      case TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE:
        break;
    }
  }
  if (agent_->AgentModelFiniFn() != nullptr) {
    LOG_TRITONSERVER_ERROR(
        agent_->AgentModelFiniFn()(
            reinterpret_cast<TRITONREPOAGENT_Agent*>(agent_.get()),
            reinterpret_cast<TRITONREPOAGENT_AgentModel*>(this)),
        "~TritonRepoAgentModel");
  }
  if (!acquired_location_.empty()) {
    DeleteMutableLocation();
  }
}

Status
TritonRepoAgentModel::InvokeAgent(const TRITONREPOAGENT_ActionType action_type)
{
  if ((!action_type_set_) && (action_type != TRITONREPOAGENT_ACTION_LOAD)) {
    return Status(
        Status::Code::INTERNAL,
        "Unexpected lifecycle start state " +
            TRITONREPOAGENT_ActionTypeString(action_type));
  }
  switch (action_type) {
    case TRITONREPOAGENT_ACTION_LOAD:
      if (action_type_set_) {
        return Status(
            Status::Code::INTERNAL,
            "Unexpected lifecycle state transition from " +
                TRITONREPOAGENT_ActionTypeString(current_action_type_) +
                " to " + TRITONREPOAGENT_ActionTypeString(action_type));
      }
      break;
    case TRITONREPOAGENT_ACTION_LOAD_COMPLETE:
    case TRITONREPOAGENT_ACTION_LOAD_FAIL:
      if (current_action_type_ != TRITONREPOAGENT_ACTION_LOAD) {
        return Status(
            Status::Code::INTERNAL,
            "Unexpected lifecycle state transition from " +
                TRITONREPOAGENT_ActionTypeString(current_action_type_) +
                " to " + TRITONREPOAGENT_ActionTypeString(action_type));
      }
      break;
    case TRITONREPOAGENT_ACTION_UNLOAD:
      if (current_action_type_ != TRITONREPOAGENT_ACTION_LOAD_COMPLETE) {
        return Status(
            Status::Code::INTERNAL,
            "Unexpected lifecycle state transition from " +
                TRITONREPOAGENT_ActionTypeString(current_action_type_) +
                " to " + TRITONREPOAGENT_ActionTypeString(action_type));
      }
      break;
    case TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE:
      if (current_action_type_ != TRITONREPOAGENT_ACTION_UNLOAD) {
        return Status(
            Status::Code::INTERNAL,
            "Unexpected lifecycle state transition from " +
                TRITONREPOAGENT_ActionTypeString(current_action_type_) +
                " to " + TRITONREPOAGENT_ActionTypeString(action_type));
      }
      break;
  }
  current_action_type_ = action_type;
  action_type_set_ = true;
  RETURN_IF_TRITONSERVER_ERROR(agent_->AgentModelActionFn()(
      reinterpret_cast<TRITONREPOAGENT_Agent*>(agent_.get()),
      reinterpret_cast<TRITONREPOAGENT_AgentModel*>(this), action_type));
  return Status::Success;
}

Status
TritonRepoAgentModel::SetLocation(
    const TRITONREPOAGENT_ArtifactType type, const std::string& location)
{
  if (current_action_type_ != TRITONREPOAGENT_ACTION_LOAD) {
    return Status(
        Status::Code::INVALID_ARG,
        "location can only be updated during TRITONREPOAGENT_ACTION_LOAD, "
        "current action type is " +
            (action_type_set_
                 ? TRITONREPOAGENT_ActionTypeString(current_action_type_)
                 : "not set"));
  }
  type_ = type;
  location_ = location;
  return Status::Success;
}

Status
TritonRepoAgentModel::Location(
    TRITONREPOAGENT_ArtifactType* type, const char** location)
{
  if (location_.empty()) {
    return Status(
        Status::Code::INTERNAL, "Model repository location is not set");
  }
  *type = type_;
  *location = location_.c_str();
  return Status::Success;
}

Status
TritonRepoAgentModel::AcquireMutableLocation(
    const TRITONREPOAGENT_ArtifactType type, const char** location)
{
  if (type != TRITONREPOAGENT_ARTIFACT_FILESYSTEM) {
    return Status(
        Status::Code::INVALID_ARG,
        "Unexpected artifact type, expects "
        "'TRITONREPOAGENT_ARTIFACT_FILESYSTEM'");
  }
  if (acquired_location_.empty()) {
    std::string lacquired_location;
    RETURN_IF_ERROR(
        MakeTemporaryDirectory(FileSystemType::LOCAL, &lacquired_location));
    acquired_location_.swap(lacquired_location);
    acquired_type_ = type;
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
TritonRepoAgentManager::SetGlobalSearchPath(const std::string& path)
{
  auto& singleton_manager = Singleton();
  std::lock_guard<std::mutex> lock(singleton_manager.mu_);
  singleton_manager.global_search_path_ = path;
  return Status::Success;
}

Status
TritonRepoAgentManager::CreateAgent(
    const std::string& agent_name, std::shared_ptr<TritonRepoAgent>* agent)
{
  auto& singleton_manager = Singleton();
  std::lock_guard<std::mutex> lock(singleton_manager.mu_);

  // Get the path to the agent shared library. Search path is global
  // agent directory.  FIXME expose global path as Triton option
  const std::vector<std::string> search_paths = {
      JoinPath({singleton_manager.global_search_path_, agent_name})};

  std::string agent_libname = TritonRepoAgentLibraryName(agent_name);
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
        "unable to find '" + agent_libname + "' for repo agent '" + agent_name +
            "', searched: " + singleton_manager.global_search_path_);
  }

  const auto& itr = singleton_manager.agent_map_.find(libpath);
  if (itr != singleton_manager.agent_map_.end()) {
    // Found in map. If the weak_ptr is still valid that means that
    // there are other models using the agent and we just reuse that
    // same agent. If the weak_ptr is not valid then agent has been
    // unloaded so we need to remove the weak_ptr from the map and
    // create the agent again.
    *agent = itr->second.lock();
    if (*agent != nullptr) {
      return Status::Success;
    }

    singleton_manager.agent_map_.erase(itr);
  }
  RETURN_IF_ERROR(TritonRepoAgent::Create(agent_name, libpath, agent));
  singleton_manager.agent_map_.insert({libpath, *agent});

  return Status::Success;
}

Status
TritonRepoAgentManager::AgentState(
    std::unique_ptr<std::unordered_map<std::string, std::string>>* agent_state)
{
  auto& singleton_manager = Singleton();
  std::lock_guard<std::mutex> lock(singleton_manager.mu_);

  std::unique_ptr<std::unordered_map<std::string, std::string>> agent_state_map(
      new std::unordered_map<std::string, std::string>);
  for (const auto& agent_pair : singleton_manager.agent_map_) {
    auto& libpath = agent_pair.first;
    auto agent = agent_pair.second.lock();

    if (agent != nullptr) {
      agent_state_map->insert({agent->Name(), libpath});
    }
  }

  *agent_state = std::move(agent_state_map);

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
  RETURN_TRITONSERVER_ERROR_IF_ERROR(tam->Location(artifact_type, location));
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONREPOAGENT_ModelRepositoryLocationAcquire(
    TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
    const TRITONREPOAGENT_ArtifactType artifact_type, const char** location)
{
  TritonRepoAgentModel* tam = reinterpret_cast<TritonRepoAgentModel*>(model);
  RETURN_TRITONSERVER_ERROR_IF_ERROR(
      tam->AcquireMutableLocation(artifact_type, location));
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONREPOAGENT_ModelRepositoryLocationRelease(
    TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
    const char* location)
{
  TritonRepoAgentModel* tam = reinterpret_cast<TritonRepoAgentModel*>(model);
  RETURN_TRITONSERVER_ERROR_IF_ERROR(tam->DeleteMutableLocation());
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONREPOAGENT_ModelRepositoryUpdate(
    TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
    const TRITONREPOAGENT_ArtifactType artifact_type, const char* location)
{
  TritonRepoAgentModel* tam = reinterpret_cast<TritonRepoAgentModel*>(model);
  RETURN_TRITONSERVER_ERROR_IF_ERROR(tam->SetLocation(artifact_type, location));
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

TRITONSERVER_Error*
TRITONREPOAGENT_ModelState(TRITONREPOAGENT_AgentModel* model, void** state)
{
  TritonRepoAgentModel* tam = reinterpret_cast<TritonRepoAgentModel*>(model);
  *state = tam->State();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONREPOAGENT_ModelSetState(TRITONREPOAGENT_AgentModel* model, void* state)
{
  TritonRepoAgentModel* tam = reinterpret_cast<TritonRepoAgentModel*>(model);
  tam->SetState(state);
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONREPOAGENT_State(TRITONREPOAGENT_Agent* agent, void** state)
{
  TritonRepoAgent* ta = reinterpret_cast<TritonRepoAgent*>(agent);
  *state = ta->State();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONREPOAGENT_SetState(TRITONREPOAGENT_Agent* agent, void* state)
{
  TritonRepoAgent* ta = reinterpret_cast<TritonRepoAgent*>(agent);
  ta->SetState(state);
  return nullptr;  // success
}

}  // extern C

}}  // namespace nvidia::inferenceserver
