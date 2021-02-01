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
#pragma once

#include "src/core/tritonserver_apis.h"

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "src/core/constants.h"
#include "src/core/filesystem.h"
#include "src/core/model_config.pb.h"

namespace nvidia { namespace inferenceserver {

std::string TRITONREPOAGENT_ActionTypeString(const TRITONREPOAGENT_ActionType type);

class TritonRepoAgent {
 public:
  using Parameters = std::vector<std::pair<std::string, std::string>>;
  typedef TRITONSERVER_Error* (*TritonRepoAgentInitFn_t)(
      TRITONREPOAGENT_Agent* agent);
  typedef TRITONSERVER_Error* (*TritonRepoAgentFiniFn_t)(
      TRITONREPOAGENT_Agent* agent);
  typedef TRITONSERVER_Error* (*TritonRepoAgentModelInitFn_t)(
      TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model);
  typedef TRITONSERVER_Error* (*TritonRepoAgentModelFiniFn_t)(
      TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model);
  typedef TRITONSERVER_Error* (*TritonRepoAgentModelActionFn_t)(
      TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
      const TRITONREPOAGENT_ActionType action_type);

  static Status Create(
      const std::string& name, const std::string& libpath,
      std::shared_ptr<TritonRepoAgent>* agent);
  ~TritonRepoAgent();

  const std::string& Name() { return name_; }
  void* State() { return state_; }
  void SetState(void* state) { state_ = state; }

  TritonRepoAgentModelActionFn_t AgentModelActionFn() const
  {
    return model_action_fn_;
  }

  TritonRepoAgentModelInitFn_t AgentModelInitFn() const
  {
    return model_init_fn_;
  }

  TritonRepoAgentModelFiniFn_t AgentModelFiniFn() const
  {
    return model_fini_fn_;
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(TritonRepoAgent);

  TritonRepoAgent(const std::string& name)
      : name_(name), state_(nullptr),
        dlhandle_(nullptr), init_fn_(nullptr), fini_fn_(nullptr),
        model_init_fn_(nullptr), model_fini_fn_(nullptr),
        model_action_fn_(nullptr)
  {
  }
  const std::string name_;
  void* state_;

  // dlopen / dlsym handles
  void* dlhandle_;
  TritonRepoAgentInitFn_t init_fn_;
  TritonRepoAgentFiniFn_t fini_fn_;
  TritonRepoAgentModelInitFn_t model_init_fn_;
  TritonRepoAgentModelFiniFn_t model_fini_fn_;
  TritonRepoAgentModelActionFn_t model_action_fn_;
};

class TritonRepoAgentModel {
 public:
  static Status Create(
      const inference::ModelConfig& config,
      const std::shared_ptr<TritonRepoAgent> agent,
      TritonRepoAgent::Parameters&& agent_parameters,
      std::unique_ptr<TritonRepoAgentModel>* agent_model);
  ~TritonRepoAgentModel();

  void* State() { return state_; }
  void SetState(void* state) { state_ = state; }

  Status InvokeAgent(const TRITONREPOAGENT_ActionType action_type);
  const TritonRepoAgent::Parameters& AgentParameters()
  {
    return agent_parameters_;
  }

  Status SetLocation(const FileSystemType type, const std::string& location);
  Status Location(FileSystemType* type, const char** location);
  Status AcquireMutableLocation(const FileSystemType type, const char** location);
  Status DeleteMutableLocation();
  const inference::ModelConfig Config() { return config_; }

 private:
  DISALLOW_COPY_AND_ASSIGN(TritonRepoAgentModel);

  TritonRepoAgentModel(
      const inference::ModelConfig& config,
      const std::shared_ptr<TritonRepoAgent> agent,
      TritonRepoAgent::Parameters&& agent_parameters)
      : state_(nullptr), config_(config), agent_(agent), agent_parameters_(std::move(agent_parameters))
  {
  }

  void* state_;
  const inference::ModelConfig config_;
  const std::shared_ptr<TritonRepoAgent> agent_;
  const TritonRepoAgent::Parameters agent_parameters_;
  FileSystemType type_;
  std::string location_;
  FileSystemType acquired_type_;
  std::string acquired_location_;
};

class TritonRepoAgentManager {
 public:
  static Status CreateAgent(
      const std::string& model_dir, const std::string& agent_name,
      std::shared_ptr<TritonRepoAgent>* agent);

 private:
  DISALLOW_COPY_AND_ASSIGN(TritonRepoAgentManager);

  TritonRepoAgentManager() = default;
  static TritonRepoAgentManager& Singleton();
  std::mutex mu_;
  std::unordered_map<std::string, std::weak_ptr<TritonRepoAgent>> agent_map_;
};

}}  // namespace nvidia::inferenceserver
