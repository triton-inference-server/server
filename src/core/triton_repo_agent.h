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
#include "src/core/model_config_utils.h"

namespace nvidia { namespace inferenceserver {

class TritonRepoAgent {
 public:
  using Parameters = std::vector<std::pair<std::string, std::string>>;
  typedef TRITONSERVER_Error* (*TritonRepoAgentInitFn_t)(
      TRITONREPOAGENT_Agent* agent);
  typedef TRITONSERVER_Error* (*TritonRepoAgentFiniFn_t)(
      TRITONREPOAGENT_Agent* agent);
  typedef TRITONSERVER_Error* (*TritonRepoAgentModelActionFn_t)(
      TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
      const TRITONREPOAGENT_ActionType action_type);

  static Status Create(
      const std::string& name, const std::string& libpath,
      std::shared_ptr<TritonRepoAgent>* agent);
  ~TritonRepoAgent();

  TritonRepoAgentModelActionFn_t AgentModelActionFn() const
  {
    return model_action_fn_;
  }

 private:
  TritonRepoAgent(const std::string& name)
      : name_(name), dlhandle_(nullptr), init_fn_(nullptr), fini_fn_(nullptr),
        model_action_fn_(nullptr)
  {
  }
  const std::string name_;

  // dlopen / dlsym handles
  void* dlhandle_;
  TritonRepoAgentInitFn_t init_fn_;
  TritonRepoAgentFiniFn_t fini_fn_;
  TritonRepoAgentModelActionFn_t model_action_fn_;
};

class TritonRepoAgentModel {
 public:
  ~TritonRepoAgentModel();
  TritonRepoAgentModel(
      const std::string& model_dir, const FileSystemType model_dir_type,
      const inference::ModelConfig& config,
      std::vector<std::pair<
          std::shared_ptr<TritonRepoAgent>, TritonRepoAgent::Parameters>>&&
          agents)
      : current_idx_(0), config_(config), agents_(std::move(agents)),
        has_committed_(false), committed_type_(model_dir_type),
        committed_location_(model_dir)
  {
  }
  Status InvokeAgents(const TRITONREPOAGENT_ActionType action_type);
  const TritonRepoAgent::Parameters& CurrentAgentParameters()
  {
    return agents_[current_idx_].second;
  }
  Status Location(FileSystemType* type, const char** location);
  Status AcquireMutableLocation(
      const FileSystemType preferred_type, FileSystemType* actual_type,
      const char** location);
  Status CommitMutableLocation();
  Status DeleteMutableLocation();
  const inference::ModelConfig Config() { return config_; }

 private:
  size_t current_idx_;
  const inference::ModelConfig config_;
  const std::vector<
      std::pair<std::shared_ptr<TritonRepoAgent>, TritonRepoAgent::Parameters>>
      agents_;
  bool has_committed_;
  FileSystemType committed_type_;
  std::string committed_location_;
  FileSystemType acquired_type_;
  std::string acquired_location_;
};

class TritonRepoAgentManager {
 public:
  static Status CreateAgentModel(
      const std::string& model_dir, const inference::ModelConfig& config,
      std::shared_ptr<TritonRepoAgentModel>* agent_model);

 private:
  DISALLOW_COPY_AND_ASSIGN(TritonRepoAgentManager);
  
  TritonRepoAgentManager() = default;
  static TritonRepoAgentManager& Singleton();
  std::mutex mu_;
  std::unordered_map<std::string, std::weak_ptr<TritonRepoAgent>> agent_map_;
};

}}  // namespace nvidia::inferenceserver
