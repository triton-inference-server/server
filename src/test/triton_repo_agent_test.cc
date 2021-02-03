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
#include "gtest/gtest.h"

#include <future>
#include <map>
#include <memory>
#include "src/core/filesystem.h"
#include "src/core/server_message.h"
#include "src/core/triton_repo_agent.h"

namespace ni = nvidia::inferenceserver;

namespace {

//
// Duplication of TRITONSERVER_Error implementation
//
class TritonServerError {
 public:
  static TRITONSERVER_Error* Create(
      TRITONSERVER_Error_Code code, const char* msg);
  static TRITONSERVER_Error* Create(const ni::Status& status);

  TRITONSERVER_Error_Code Code() const { return code_; }
  const std::string& Message() const { return msg_; }

 private:
  TritonServerError(TRITONSERVER_Error_Code code, const std::string& msg)
      : code_(code), msg_(msg)
  {
  }
  TritonServerError(TRITONSERVER_Error_Code code, const char* msg)
      : code_(code), msg_(msg)
  {
  }

  TRITONSERVER_Error_Code code_;
  const std::string msg_;
};

TRITONSERVER_Error*
TritonServerError::Create(TRITONSERVER_Error_Code code, const char* msg)
{
  return reinterpret_cast<TRITONSERVER_Error*>(
      new TritonServerError(code, msg));
}

TRITONSERVER_Error*
TritonServerError::Create(const ni::Status& status)
{
  // If 'status' is success then return nullptr as that indicates
  // success
  if (status.IsOk()) {
    return nullptr;
  }

  return Create(
      StatusCodeToTritonCode(status.StatusCode()), status.Message().c_str());
}

class MockSharedLibraryHandle {
 public:
  bool AddEntryPoint(const std::string& name, void* fn)
  {
    return entry_points_.emplace(name, fn).second;
  }

  bool GetEntryPoint(const std::string& name, void** fn)
  {
    auto it = entry_points_.find(name);
    if (it != entry_points_.end()) {
      *fn = it->second;
      return true;
    }
    return false;
  }

 private:
  std::map<std::string, void*> entry_points_;
};

static std::map<std::string, MockSharedLibraryHandle> global_mock_agents;

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

TRITONSERVER_Error*
TRITONSERVER_ErrorNew(TRITONSERVER_Error_Code code, const char* msg)
{
  return reinterpret_cast<TRITONSERVER_Error*>(
      TritonServerError::Create(code, msg));
}

void
TRITONSERVER_ErrorDelete(TRITONSERVER_Error* error)
{
  TritonServerError* lerror = reinterpret_cast<TritonServerError*>(error);
  delete lerror;
}

TRITONSERVER_Error_Code
TRITONSERVER_ErrorCode(TRITONSERVER_Error* error)
{
  TritonServerError* lerror = reinterpret_cast<TritonServerError*>(error);
  return lerror->Code();
}

const char*
TRITONSERVER_ErrorCodeString(TRITONSERVER_Error* error)
{
  TritonServerError* lerror = reinterpret_cast<TritonServerError*>(error);
  return ni::Status::CodeString(ni::TritonCodeToStatusCode(lerror->Code()));
}

const char*
TRITONSERVER_ErrorMessage(TRITONSERVER_Error* error)
{
  TritonServerError* lerror = reinterpret_cast<TritonServerError*>(error);
  return lerror->Message().c_str();
}

//
// TRITONSERVER_Message
//
TRITONSERVER_Error*
TRITONSERVER_MessageNewFromSerializedJson(
    TRITONSERVER_Message** message, const char* base, size_t byte_size)
{
  *message = reinterpret_cast<TRITONSERVER_Message*>(
      new ni::TritonServerMessage({base, byte_size}));
  return nullptr;
}

#ifdef __cplusplus
}
#endif

namespace nvidia::inferenceserver {

Status
OpenLibraryHandle(const std::string& path, void** handle)
{
  auto it = global_mock_agents.find(path);
  if (it != global_mock_agents.end()) {
    *handle = reinterpret_cast<void*>(&it->second);
    return Status::Success;
  }
  return Status(
      Status::Code::NOT_FOUND,
      "unable to load shared library: mock shared library is not set for "
      "path " +
          path);
}

Status
CloseLibraryHandle(void* handle)
{
  for (auto& global_mock_agent : global_mock_agents) {
    if (reinterpret_cast<void*>(&global_mock_agent.second) == handle) {
      return Status::Success;
    }
  }
  return Status(
      Status::Code::NOT_FOUND,
      "unable to unload shared library: handle does not matach any mock shared "
      "library");
}

Status
GetEntrypoint(
    void* handle, const std::string& name, const bool optional, void** fn)
{
  auto mock_agent = reinterpret_cast<MockSharedLibraryHandle*>(handle);
  bool found = mock_agent->GetEntryPoint(name, fn);
  if (!optional && !found) {
    return Status(
        Status::Code::NOT_FOUND,
        "unable to find required entrypoint '" + name + "' in shared library");
  }
  return Status::Success;
}

}  // namespace nvidia::inferenceserver

namespace {

class TritonRepoAgentTest : public ::testing::Test {
 protected:
  void TearDown() override { global_mock_agents.clear(); }
};

TEST_F(TritonRepoAgentTest, Create)
{
  // Set up agent with only action function defined, check agent properties
  ni::TritonRepoAgent::TritonRepoAgentModelActionFn_t CheckNameModelActionFn =
      [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
         const TRITONREPOAGENT_ActionType action_type) -> TRITONSERVER_Error* {
    auto lagent = reinterpret_cast<ni::TritonRepoAgent*>(agent);
    EXPECT_EQ(lagent->Name(), "minimal_agent")
        << "Expect action function is called with minimal agent";
    return nullptr;
  };
  auto agent_handle = MockSharedLibraryHandle();
  agent_handle.AddEntryPoint(
      "TRITONREPOAGENT_ModelAction",
      reinterpret_cast<void*>(CheckNameModelActionFn));
  global_mock_agents.emplace("minimal_agent_path", agent_handle);

  std::shared_ptr<ni::TritonRepoAgent> minimal_agent;
  auto status = ni::TritonRepoAgent::Create(
      "minimal_agent", "minimal_agent_path", &minimal_agent);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation: " << status.AsString();

  ASSERT_TRUE(minimal_agent->AgentModelActionFn() != nullptr)
      << "Expect action function is provided";
  EXPECT_TRUE(minimal_agent->AgentModelInitFn() == nullptr)
      << "Unexpect model init function is provided";
  EXPECT_TRUE(minimal_agent->AgentModelFiniFn() == nullptr)
      << "Unexpect model fini function is provided";

  auto err = minimal_agent->AgentModelActionFn()(
      reinterpret_cast<TRITONREPOAGENT_Agent*>(minimal_agent.get()), nullptr,
      TRITONREPOAGENT_ACTION_LOAD);
  EXPECT_TRUE(err == nullptr) << "Expect successful action function invocation";
}

TEST_F(TritonRepoAgentTest, CreateFailMissingEndpoint)
{
  // Set up agent with nothing defined
  auto agent_handle = MockSharedLibraryHandle();
  global_mock_agents.emplace("invalid_agent_path", agent_handle);

  std::shared_ptr<ni::TritonRepoAgent> invalid_agent;
  auto status = ni::TritonRepoAgent::Create(
      "invalid_agent", "invalid_agent_path", &invalid_agent);
  ASSERT_FALSE(status.IsOk()) << "Unexpect successful agent creation";
  EXPECT_NE(
      status.Message().find("unable to find required entrypoint"),
      std::string::npos)
      << "Unexpect error message: '" << status.Message()
      << "', expect 'unable to find required entrypoint...'";
}

TEST_F(TritonRepoAgentTest, Lifecycle)
{
  // Set up agent with init / fini function defined
  ni::TritonRepoAgent::TritonRepoAgentInitFn_t InitFn =
      [](TRITONREPOAGENT_Agent* agent) -> TRITONSERVER_Error* {
    auto lagent = reinterpret_cast<ni::TritonRepoAgent*>(agent);
    EXPECT_TRUE(lagent->State() == nullptr)
        << "Expect agent state is not set before initialization";
    bool* state = new bool(false);
    lagent->SetState(reinterpret_cast<void*>(state));
    return nullptr;
  };
  ni::TritonRepoAgent::TritonRepoAgentFiniFn_t FiniFn =
      [](TRITONREPOAGENT_Agent* agent) -> TRITONSERVER_Error* {
    auto lagent = reinterpret_cast<ni::TritonRepoAgent*>(agent);
    bool* state = reinterpret_cast<bool*>(lagent->State());
    EXPECT_TRUE(state != nullptr) << "Expect agent state is set";
    EXPECT_TRUE(*state) << "Expect state is set to true";
    delete state;
    return nullptr;
  };
  ni::TritonRepoAgent::TritonRepoAgentModelActionFn_t ActionFn =
      [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
         const TRITONREPOAGENT_ActionType action_type) -> TRITONSERVER_Error* {
    auto lagent = reinterpret_cast<ni::TritonRepoAgent*>(agent);
    bool* state = reinterpret_cast<bool*>(lagent->State());
    EXPECT_TRUE(state != nullptr) << "Expect agent state is set";
    EXPECT_FALSE(*state) << "Expect state is set to false";
    *state = true;
    return nullptr;
  };
  auto agent_handle = MockSharedLibraryHandle();
  agent_handle.AddEntryPoint(
      "TRITONREPOAGENT_Initialize", reinterpret_cast<void*>(InitFn));
  agent_handle.AddEntryPoint(
      "TRITONREPOAGENT_Finalize", reinterpret_cast<void*>(FiniFn));
  agent_handle.AddEntryPoint(
      "TRITONREPOAGENT_ModelAction", reinterpret_cast<void*>(ActionFn));
  global_mock_agents.emplace("agent_path", agent_handle);

  std::shared_ptr<ni::TritonRepoAgent> agent;
  auto status = ni::TritonRepoAgent::Create("agent", "agent_path", &agent);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation: " << status.AsString();

  ASSERT_TRUE(agent->AgentModelActionFn() != nullptr)
      << "Expect action function is provided";
  EXPECT_TRUE(agent->AgentModelInitFn() == nullptr)
      << "Unexpect model init function is provided";
  EXPECT_TRUE(agent->AgentModelFiniFn() == nullptr)
      << "Unexpect model fini function is provided";

  auto err = agent->AgentModelActionFn()(
      reinterpret_cast<TRITONREPOAGENT_Agent*>(agent.get()), nullptr,
      TRITONREPOAGENT_ACTION_LOAD);
  EXPECT_TRUE(err == nullptr) << "Expect successful action function invocation";
  // Cause destructor to be called
  agent.reset();
}

TEST_F(TritonRepoAgentTest, ModelLifecycle)
{
  // Set up agent with model init / fini function defined
  ni::TritonRepoAgent::TritonRepoAgentModelInitFn_t InitFn =
      [](TRITONREPOAGENT_Agent* agent,
         TRITONREPOAGENT_AgentModel* model) -> TRITONSERVER_Error* {
    auto lmodel_state =
        reinterpret_cast<std::pair<std::promise<void>*, std::future<void>*>*>(
            model);
    lmodel_state->first->set_value();
    return nullptr;
  };
  ni::TritonRepoAgent::TritonRepoAgentModelFiniFn_t FiniFn =
      [](TRITONREPOAGENT_Agent* agent,
         TRITONREPOAGENT_AgentModel* model) -> TRITONSERVER_Error* {
    auto lmodel_state =
        reinterpret_cast<std::pair<std::promise<void>*, std::future<void>*>*>(
            model);
    lmodel_state->second->get();
    return nullptr;
  };
  ni::TritonRepoAgent::TritonRepoAgentModelActionFn_t ActionFn =
      [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
         const TRITONREPOAGENT_ActionType action_type) -> TRITONSERVER_Error* {
    auto lmodel_state =
        reinterpret_cast<std::pair<std::promise<void>*, std::future<void>*>*>(
            model);
    EXPECT_TRUE(lmodel_state->second->valid()) << "Expect promise value is set";
    return nullptr;
  };
  auto agent_handle = MockSharedLibraryHandle();
  agent_handle.AddEntryPoint(
      "TRITONREPOAGENT_ModelInitialize", reinterpret_cast<void*>(InitFn));
  agent_handle.AddEntryPoint(
      "TRITONREPOAGENT_ModelFinalize", reinterpret_cast<void*>(FiniFn));
  agent_handle.AddEntryPoint(
      "TRITONREPOAGENT_ModelAction", reinterpret_cast<void*>(ActionFn));
  global_mock_agents.emplace("agent_path", agent_handle);

  std::shared_ptr<ni::TritonRepoAgent> agent;
  auto status = ni::TritonRepoAgent::Create("agent", "agent_path", &agent);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation: " << status.AsString();

  ASSERT_TRUE(agent->AgentModelActionFn() != nullptr)
      << "Expect action function is provided";
  ASSERT_TRUE(agent->AgentModelInitFn() != nullptr)
      << "Expect model init function is provided";
  ASSERT_TRUE(agent->AgentModelFiniFn() != nullptr)
      << "Expect model fini function is provided";

  std::promise<void> p;
  auto f = p.get_future();
  auto model_state = std::make_pair(&p, &f);
  // Simulate the model lifecycle
  auto err = agent->AgentModelInitFn()(
      reinterpret_cast<TRITONREPOAGENT_Agent*>(agent.get()),
      reinterpret_cast<TRITONREPOAGENT_AgentModel*>(&model_state));
  EXPECT_TRUE(err == nullptr)
      << "Expect successful model init function invocation";
  err = agent->AgentModelActionFn()(
      reinterpret_cast<TRITONREPOAGENT_Agent*>(agent.get()),
      reinterpret_cast<TRITONREPOAGENT_AgentModel*>(&model_state),
      TRITONREPOAGENT_ACTION_LOAD);
  EXPECT_TRUE(err == nullptr) << "Expect successful action function invocation";
  err = agent->AgentModelFiniFn()(
      reinterpret_cast<TRITONREPOAGENT_Agent*>(agent.get()),
      reinterpret_cast<TRITONREPOAGENT_AgentModel*>(&model_state));
  EXPECT_TRUE(err == nullptr)
      << "Expect successful model fini function invocation";
  EXPECT_FALSE(f.valid()) << "Expect future value is retrieved";
}

class TritonRepoAgentModelTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    simple_config_.set_name("simple_config");

    // Add a simple agent handle for convinence
    ni::TritonRepoAgent::TritonRepoAgentModelActionFn_t ActionFn =
        [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
           const TRITONREPOAGENT_ActionType action_type)
        -> TRITONSERVER_Error* { return nullptr; };
    auto agent_handle = MockSharedLibraryHandle();
    agent_handle.AddEntryPoint(
        "TRITONREPOAGENT_ModelAction", reinterpret_cast<void*>(ActionFn));
    global_mock_agents.emplace("simple_agent_path", agent_handle);

    // Add a agent handle for logging actions of the model
    ni::TritonRepoAgent::TritonRepoAgentModelInitFn_t LogInitFn =
        [](TRITONREPOAGENT_Agent* agent,
           TRITONREPOAGENT_AgentModel* model) -> TRITONSERVER_Error* {
      auto lagent = reinterpret_cast<ni::TritonRepoAgent*>(agent);
      auto state = reinterpret_cast<std::vector<std::string>*>(lagent->State());
      if (state == nullptr) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, "Agent state is not set");
      }
      state->emplace_back("Model Initialized");
      return nullptr;
    };
    ni::TritonRepoAgent::TritonRepoAgentModelFiniFn_t LogFiniFn =
        [](TRITONREPOAGENT_Agent* agent,
           TRITONREPOAGENT_AgentModel* model) -> TRITONSERVER_Error* {
      auto lagent = reinterpret_cast<ni::TritonRepoAgent*>(agent);
      auto state = reinterpret_cast<std::vector<std::string>*>(lagent->State());
      if (state == nullptr) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, "Agent state is not set");
      }
      state->emplace_back("Model Finalized");
      return nullptr;
    };
    ni::TritonRepoAgent::TritonRepoAgentModelActionFn_t LogActionFn =
        [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
           const TRITONREPOAGENT_ActionType action_type)
        -> TRITONSERVER_Error* {
      auto lagent = reinterpret_cast<ni::TritonRepoAgent*>(agent);
      auto state = reinterpret_cast<std::vector<std::string>*>(lagent->State());
      if (state == nullptr) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, "Agent state is not set");
      }
      state->emplace_back(ni::TRITONREPOAGENT_ActionTypeString(action_type));
      return nullptr;
    };
    auto log_agent_handle = MockSharedLibraryHandle();
    log_agent_handle.AddEntryPoint(
        "TRITONREPOAGENT_ModelInitialize", reinterpret_cast<void*>(LogInitFn));
    log_agent_handle.AddEntryPoint(
        "TRITONREPOAGENT_ModelFinalize", reinterpret_cast<void*>(LogFiniFn));
    log_agent_handle.AddEntryPoint(
        "TRITONREPOAGENT_ModelAction", reinterpret_cast<void*>(LogActionFn));
    global_mock_agents.emplace("log_agent_path", log_agent_handle);
  }
  void TearDown() override { global_mock_agents.clear(); }

  ni::FileSystemType original_type_ = ni::FileSystemType::LOCAL;
  const std::string original_location_ = "/original";
  inference::ModelConfig simple_config_;
};

TEST_F(TritonRepoAgentModelTest, Create)
{
  // Create agent to be associated with the model
  std::shared_ptr<ni::TritonRepoAgent> agent;
  auto status =
      ni::TritonRepoAgent::Create("agent", "simple_agent_path", &agent);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation: " << status.AsString();

  // Create model
  std::unique_ptr<ni::TritonRepoAgentModel> model;
  status = ni::TritonRepoAgentModel::Create(
      original_type_, original_location_, simple_config_, agent,
      ni::TritonRepoAgent::Parameters(), &model);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful model creation: " << status.AsString();
  EXPECT_EQ(model->Config().name(), simple_config_.name())
      << "Expect the model contains the same config as simple config";
}

TEST_F(TritonRepoAgentModelTest, CreateFailure)
{
  // Create agent to be associated with the model, whose model init function
  // always returns error
  ni::TritonRepoAgent::TritonRepoAgentModelInitFn_t InitFn =
      [](TRITONREPOAGENT_Agent* agent,
         TRITONREPOAGENT_AgentModel* model) -> TRITONSERVER_Error* {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "Model initialization error");
  };
  ni::TritonRepoAgent::TritonRepoAgentModelActionFn_t ActionFn =
      [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
         const TRITONREPOAGENT_ActionType action_type) -> TRITONSERVER_Error* {
    return nullptr;
  };
  auto agent_handle = MockSharedLibraryHandle();
  agent_handle.AddEntryPoint(
      "TRITONREPOAGENT_ModelInitialize", reinterpret_cast<void*>(InitFn));
  agent_handle.AddEntryPoint(
      "TRITONREPOAGENT_ModelAction", reinterpret_cast<void*>(ActionFn));
  global_mock_agents.emplace("agent_path", agent_handle);

  std::shared_ptr<ni::TritonRepoAgent> agent;
  auto status = ni::TritonRepoAgent::Create("agent", "agent_path", &agent);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation: " << status.AsString();

  // Create model
  std::unique_ptr<ni::TritonRepoAgentModel> model;
  status = ni::TritonRepoAgentModel::Create(
      original_type_, original_location_, simple_config_, agent,
      ni::TritonRepoAgent::Parameters(), &model);
  ASSERT_FALSE(status.IsOk()) << "Unexpect successful model creation";
  EXPECT_NE(
      status.Message().find("Model initialization error"), std::string::npos)
      << "Unexpect error message: '" << status.Message()
      << "', expect 'Model initialization error...'";
}

TEST_F(TritonRepoAgentModelTest, Location)
{
  // Create agent to be associated with the model
  std::shared_ptr<ni::TritonRepoAgent> agent;
  auto status =
      ni::TritonRepoAgent::Create("agent", "simple_agent_path", &agent);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation: " << status.AsString();

  // Create model
  std::unique_ptr<ni::TritonRepoAgentModel> model;
  status = ni::TritonRepoAgentModel::Create(
      original_type_, original_location_, simple_config_, agent,
      ni::TritonRepoAgent::Parameters(), &model);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful model creation: " << status.AsString();
  ni::FileSystemType type;
  const char* location;
  status = model->Location(&type, &location);
  ASSERT_TRUE(status.IsOk()) << "Expect location is returned from Location()";
  EXPECT_EQ(type, original_type_) << "Expect returned original filesystem type";
  EXPECT_EQ(std::string(location), original_location_)
      << "Expect returned original location";
}

TEST_F(TritonRepoAgentModelTest, SetLocationFailure)
{
  // Create agent to be associated with the model
  std::shared_ptr<ni::TritonRepoAgent> agent;
  auto status =
      ni::TritonRepoAgent::Create("agent", "simple_agent_path", &agent);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation: " << status.AsString();

  // Create model
  std::unique_ptr<ni::TritonRepoAgentModel> model;
  status = ni::TritonRepoAgentModel::Create(
      original_type_, original_location_, simple_config_, agent,
      ni::TritonRepoAgent::Parameters(), &model);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful model creation: " << status.AsString();
  ni::FileSystemType type = ni::FileSystemType::LOCAL;
  const char* location = "/tmp";
  status = model->SetLocation(type, location);
  ASSERT_FALSE(status.IsOk()) << "Expect error returned from SetLocation()";
  EXPECT_NE(
      status.Message().find(
          "location can only be updated during TRITONREPOAGENT_ACTION_LOAD, "
          "current action type is not set"),
      std::string::npos)
      << "Unexpect error message: '" << status.Message()
      << "', expect 'location can only be updated during "
         "TRITONREPOAGENT_ACTION_LOAD, current action type is not set'";
}

TEST_F(TritonRepoAgentModelTest, SetLocation)
{
  static const ni::FileSystemType new_type = ni::FileSystemType::LOCAL;
  static const std::string new_location = "/new_location";

  // Create agent to be associated with the model
  std::shared_ptr<ni::TritonRepoAgent> agent;
  auto status =
      ni::TritonRepoAgent::Create("agent", "simple_agent_path", &agent);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation: " << status.AsString();

  // Create model
  std::unique_ptr<ni::TritonRepoAgentModel> model;
  status = ni::TritonRepoAgentModel::Create(
      original_type_, original_location_, simple_config_, agent,
      ni::TritonRepoAgent::Parameters(), &model);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful model creation: " << status.AsString();
  // Advance the model lifecycle to be able to set location
  status = model->InvokeAgent(TRITONREPOAGENT_ACTION_LOAD);
  EXPECT_TRUE(status.IsOk())
      << "Expect successful agent invocation with TRITONREPOAGENT_ACTION_LOAD";
  status = model->SetLocation(new_type, new_location);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful SetLocation() after invoking agent with "
         "TRITONREPOAGENT_ACTION_LOAD";
  ni::FileSystemType type = original_type_;
  const char* location = original_location_.c_str();
  status = model->Location(&type, &location);
  ASSERT_TRUE(status.IsOk()) << "Expect location is returned from Location()";
  EXPECT_EQ(type, new_type) << "Expect returned filesystem type is "
                            << FileSystemTypeString(new_type);
  EXPECT_EQ(std::string(location), new_location)
      << "Expect returned location is " << new_location;
}

TEST_F(TritonRepoAgentModelTest, SetLocationWrongActionFailure)
{
  static const ni::FileSystemType new_type = ni::FileSystemType::LOCAL;
  static const std::string new_location = "/new_location";

  // Create agent to be associated with the model
  std::shared_ptr<ni::TritonRepoAgent> agent;
  auto status =
      ni::TritonRepoAgent::Create("agent", "simple_agent_path", &agent);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation: " << status.AsString();

  // Create model
  std::unique_ptr<ni::TritonRepoAgentModel> model;
  status = ni::TritonRepoAgentModel::Create(
      original_type_, original_location_, simple_config_, agent,
      ni::TritonRepoAgent::Parameters(), &model);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful model creation: " << status.AsString();
  // Advance the model lifecycle to be able to set location
  status = model->InvokeAgent(TRITONREPOAGENT_ACTION_LOAD);
  EXPECT_TRUE(status.IsOk())
      << "Expect successful agent invocation with TRITONREPOAGENT_ACTION_LOAD";
  status = model->InvokeAgent(TRITONREPOAGENT_ACTION_LOAD_COMPLETE);
  EXPECT_TRUE(status.IsOk()) << "Expect successful agent invocation with "
                                "TRITONREPOAGENT_ACTION_LOAD_COMPLETE";
  status = model->SetLocation(new_type, new_location);
  ASSERT_FALSE(status.IsOk()) << "Expect error returned from SetLocation()";
  EXPECT_NE(
      status.Message().find(
          "location can only be updated during TRITONREPOAGENT_ACTION_LOAD, "
          "current action type is TRITONREPOAGENT_ACTION_LOAD_COMPLETE"),
      std::string::npos)
      << "Unexpect error message: '" << status.Message()
      << "', expect 'location can only be updated during "
         "TRITONREPOAGENT_ACTION_LOAD, current action type is "
         "TRITONREPOAGENT_ACTION_LOAD_COMPLETE'";
}

TEST_F(TritonRepoAgentModelTest, SetLocationViaAgent)
{
  static const ni::FileSystemType new_type = ni::FileSystemType::LOCAL;
  static const std::string new_location = "/new_location";
  // Create agent to be associated with the model
  ni::TritonRepoAgent::TritonRepoAgentModelActionFn_t ActionFn =
      [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
         const TRITONREPOAGENT_ActionType action_type) -> TRITONSERVER_Error* {
    auto lmodel = reinterpret_cast<ni::TritonRepoAgentModel*>(model);
    auto status = lmodel->SetLocation(new_type, new_location);
    return reinterpret_cast<TRITONSERVER_Error*>(
        TritonServerError::Create(status));
  };
  auto agent_handle = MockSharedLibraryHandle();
  agent_handle.AddEntryPoint(
      "TRITONREPOAGENT_ModelAction", reinterpret_cast<void*>(ActionFn));
  global_mock_agents.emplace("set_location_agent_path", agent_handle);
  std::shared_ptr<ni::TritonRepoAgent> agent;
  auto status =
      ni::TritonRepoAgent::Create("agent", "set_location_agent_path", &agent);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation: " << status.AsString();

  // Create model
  std::unique_ptr<ni::TritonRepoAgentModel> model;
  status = ni::TritonRepoAgentModel::Create(
      original_type_, original_location_, simple_config_, agent,
      ni::TritonRepoAgent::Parameters(), &model);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful model creation: " << status.AsString();
  // Advance the model lifecycle to be able to set location
  status = model->InvokeAgent(TRITONREPOAGENT_ACTION_LOAD);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent invocation with TRITONREPOAGENT_ACTION_LOAD";
  ni::FileSystemType type = original_type_;
  const char* location = original_location_.c_str();
  status = model->Location(&type, &location);
  ASSERT_TRUE(status.IsOk()) << "Expect location is returned from Location()";
  EXPECT_EQ(type, new_type) << "Expect returned filesystem type is "
                            << FileSystemTypeString(new_type);
  EXPECT_EQ(std::string(location), new_location)
      << "Expect returned location is " << new_location;
}

TEST_F(TritonRepoAgentModelTest, DeleteLocationBeforeAcquire)
{
  // Create agent to be associated with the model
  std::shared_ptr<ni::TritonRepoAgent> agent;
  auto status =
      ni::TritonRepoAgent::Create("agent", "simple_agent_path", &agent);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation: " << status.AsString();

  // Create model
  std::unique_ptr<ni::TritonRepoAgentModel> model;
  status = ni::TritonRepoAgentModel::Create(
      original_type_, original_location_, simple_config_, agent,
      ni::TritonRepoAgent::Parameters(), &model);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful model creation: " << status.AsString();

  status = model->DeleteMutableLocation();
  ASSERT_FALSE(status.IsOk())
      << "Expect error returned from DeleteMutableLocation()";
  EXPECT_NE(
      status.Message().find("No mutable location to be deleted"),
      std::string::npos)
      << "Unexpect error message: '" << status.Message()
      << "', expect 'No mutable location to be deleted'";
}

TEST_F(TritonRepoAgentModelTest, AcquireLocalLocationAndDelete)
{
  // Create agent to be associated with the model
  std::shared_ptr<ni::TritonRepoAgent> agent;
  auto status =
      ni::TritonRepoAgent::Create("agent", "simple_agent_path", &agent);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation: " << status.AsString();

  // Create model
  std::unique_ptr<ni::TritonRepoAgentModel> model;
  status = ni::TritonRepoAgentModel::Create(
      original_type_, original_location_, simple_config_, agent,
      ni::TritonRepoAgent::Parameters(), &model);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful model creation: " << status.AsString();
  const char* acquired_location;
  status = model->AcquireMutableLocation(
      ni::FileSystemType::LOCAL, &acquired_location);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful location acquisition: " << status.AsString();

  // Check directory
  bool is_dir = false;
  status = ni::IsDirectory(acquired_location, &is_dir);
  ASSERT_TRUE(status.IsOk())
      << "Expect location proprety can be checked: " << status.AsString();
  EXPECT_TRUE(is_dir) << "Expect a directory is returned as mutable location";
  ni::FileSystemType type = ni::FileSystemType::LOCAL;
  status = ni::GetFileSystemType(acquired_location, &type);
  ASSERT_TRUE(status.IsOk())
      << "Expect location filesystem type can be checked: "
      << status.AsString();
  EXPECT_EQ(type, ni::FileSystemType::LOCAL)
      << "Expect a local mutable location is acquired";

  status = model->DeleteMutableLocation();
  ASSERT_TRUE(status.IsOk())
      << "Expect successful location deletion: " << status.AsString();
  // Check directory
  bool exists = true;
  status = ni::FileExists(acquired_location, &exists);
  ASSERT_TRUE(status.IsOk())
      << "Expect location proprety can be checked: " << status.AsString();
  EXPECT_FALSE(exists) << "Expect the mutable location no longer exists";
}

TEST_F(TritonRepoAgentModelTest, AcquireLocalLocationTwice)
{
  // Create agent to be associated with the model
  std::shared_ptr<ni::TritonRepoAgent> agent;
  auto status =
      ni::TritonRepoAgent::Create("agent", "simple_agent_path", &agent);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation: " << status.AsString();

  // Create model
  std::unique_ptr<ni::TritonRepoAgentModel> model;
  status = ni::TritonRepoAgentModel::Create(
      original_type_, original_location_, simple_config_, agent,
      ni::TritonRepoAgent::Parameters(), &model);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful model creation: " << status.AsString();

  const char* acquired_location;
  status = model->AcquireMutableLocation(
      ni::FileSystemType::LOCAL, &acquired_location);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful location acquisition: " << status.AsString();

  // Acquire the same type again
  const char* second_acquired_location;
  status = model->AcquireMutableLocation(
      ni::FileSystemType::LOCAL, &second_acquired_location);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful location acquisition: " << status.AsString();
  EXPECT_EQ(
      std::string(acquired_location), std::string(second_acquired_location))
      << "Expect the same location is returned";

  // Different type
  status = model->AcquireMutableLocation(
      ni::FileSystemType::S3, &second_acquired_location);
  ASSERT_FALSE(status.IsOk())
      << "Expect error returned from AcquireMutableLocation()";
  EXPECT_NE(
      status.Message().find("The requested filesystem type is different from "
                            "existing acquired location"),
      std::string::npos)
      << "Unexpect error message: '" << status.Message()
      << "', expect 'The requested filesystem type is different from existing "
         "acquired location'";
}

TEST_F(TritonRepoAgentModelTest, DeleteTwiceAfterAcquire)
{
  // Create agent to be associated with the model
  std::shared_ptr<ni::TritonRepoAgent> agent;
  auto status =
      ni::TritonRepoAgent::Create("agent", "simple_agent_path", &agent);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation: " << status.AsString();

  // Create model
  std::unique_ptr<ni::TritonRepoAgentModel> model;
  status = ni::TritonRepoAgentModel::Create(
      original_type_, original_location_, simple_config_, agent,
      ni::TritonRepoAgent::Parameters(), &model);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful model creation: " << status.AsString();
  const char* acquired_location;
  status = model->AcquireMutableLocation(
      ni::FileSystemType::LOCAL, &acquired_location);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful location acquisition: " << status.AsString();

  status = model->DeleteMutableLocation();
  ASSERT_TRUE(status.IsOk())
      << "Expect successful location deletion: " << status.AsString();
  status = model->DeleteMutableLocation();
  ASSERT_FALSE(status.IsOk())
      << "Expect error returned from DeleteMutableLocation()";
  EXPECT_NE(
      status.Message().find("No mutable location to be deleted"),
      std::string::npos)
      << "Unexpect error message: '" << status.Message()
      << "', expect 'No mutable location to be deleted'";
}

TEST_F(TritonRepoAgentModelTest, State)
{
  // Create agent to be associated with the model
  std::shared_ptr<ni::TritonRepoAgent> agent;
  auto status =
      ni::TritonRepoAgent::Create("agent", "simple_agent_path", &agent);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation: " << status.AsString();

  // Create model
  std::unique_ptr<ni::TritonRepoAgentModel> model;
  status = ni::TritonRepoAgentModel::Create(
      original_type_, original_location_, simple_config_, agent,
      ni::TritonRepoAgent::Parameters(), &model);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful model creation: " << status.AsString();
  auto state = model->State();
  ASSERT_TRUE(state == nullptr) << "Expect state is not set";
  bool state_value = true;
  model->SetState(reinterpret_cast<void*>(&state_value));
  state = model->State();
  ASSERT_TRUE(state != nullptr) << "Expect state is set";
  EXPECT_EQ(*reinterpret_cast<bool*>(state), state_value)
      << "Expect state value is true";
}

TEST_F(TritonRepoAgentModelTest, EmptyLifeCycle)
{
  // Create agent to be associated with the model
  std::shared_ptr<ni::TritonRepoAgent> agent;
  auto status = ni::TritonRepoAgent::Create("agent", "log_agent_path", &agent);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation: " << status.AsString();
  std::vector<std::string> log;
  agent->SetState(reinterpret_cast<void*>(&log));

  // Create and destroy model
  std::unique_ptr<ni::TritonRepoAgentModel> model;
  status = ni::TritonRepoAgentModel::Create(
      original_type_, original_location_, simple_config_, agent,
      ni::TritonRepoAgent::Parameters(), &model);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful model creation: " << status.AsString();
  model.reset();

  // Check log
  ASSERT_EQ(log.size(), (size_t)2)
      << "Expect 2 state of model lifecycle is logged, got " << log.size();
  EXPECT_EQ(log[0], "Model Initialized");
  EXPECT_EQ(log[1], "Model Finalized");
}

TEST_F(TritonRepoAgentModelTest, HalfLifeCycle)
{
  // Create agent to be associated with the model
  std::shared_ptr<ni::TritonRepoAgent> agent;
  auto status = ni::TritonRepoAgent::Create("agent", "log_agent_path", &agent);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation: " << status.AsString();
  std::vector<std::string> log;
  agent->SetState(reinterpret_cast<void*>(&log));

  std::unique_ptr<ni::TritonRepoAgentModel> model;
  // Create and destroy model in situations that a full lifecycle should run
  std::vector<std::vector<TRITONREPOAGENT_ActionType>> situations{
      {TRITONREPOAGENT_ACTION_LOAD},
      {TRITONREPOAGENT_ACTION_LOAD, TRITONREPOAGENT_ACTION_LOAD_FAIL}};
  std::vector<std::string> expected_log{
      "Model Initialized", "TRITONREPOAGENT_ACTION_LOAD",
      "TRITONREPOAGENT_ACTION_LOAD_FAIL", "Model Finalized"};
  for (const auto& situation : situations) {
    log.clear();
    status = ni::TritonRepoAgentModel::Create(
        original_type_, original_location_, simple_config_, agent,
        ni::TritonRepoAgent::Parameters(), &model);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful model creation: " << status.AsString();
    for (const auto action : situation) {
      status = model->InvokeAgent(action);
      EXPECT_TRUE(status.IsOk()) << "Expect successful agent invocation with "
                                 << ni::TRITONREPOAGENT_ActionTypeString(action)
                                 << ": " << status.AsString();
    }
    model.reset();

    // Check log
    ASSERT_EQ(log.size(), expected_log.size())
        << "Expect " << expected_log.size()
        << " state of model lifecycle is logged, got " << log.size();
    for (size_t i = 0; i < log.size(); i++) {
      EXPECT_EQ(log[i], expected_log[i]);
    }
  }
}

TEST_F(TritonRepoAgentModelTest, FullLifeCycle)
{
  // Create agent to be associated with the model
  std::shared_ptr<ni::TritonRepoAgent> agent;
  auto status = ni::TritonRepoAgent::Create("agent", "log_agent_path", &agent);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation: " << status.AsString();
  std::vector<std::string> log;
  agent->SetState(reinterpret_cast<void*>(&log));

  std::unique_ptr<ni::TritonRepoAgentModel> model;
  // Create and destroy model in situations that a full lifecycle should run
  std::vector<std::vector<TRITONREPOAGENT_ActionType>> situations{
      {TRITONREPOAGENT_ACTION_LOAD, TRITONREPOAGENT_ACTION_LOAD_COMPLETE},
      {TRITONREPOAGENT_ACTION_LOAD, TRITONREPOAGENT_ACTION_LOAD_COMPLETE,
       TRITONREPOAGENT_ACTION_UNLOAD},
      {TRITONREPOAGENT_ACTION_LOAD, TRITONREPOAGENT_ACTION_LOAD_COMPLETE,
       TRITONREPOAGENT_ACTION_UNLOAD, TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE}};
  std::vector<std::string> expected_log{
      "Model Initialized",
      "TRITONREPOAGENT_ACTION_LOAD",
      "TRITONREPOAGENT_ACTION_LOAD_COMPLETE",
      "TRITONREPOAGENT_ACTION_UNLOAD",
      "TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE",
      "Model Finalized"};
  for (const auto& situation : situations) {
    log.clear();
    status = ni::TritonRepoAgentModel::Create(
        original_type_, original_location_, simple_config_, agent,
        ni::TritonRepoAgent::Parameters(), &model);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful model creation: " << status.AsString();
    for (const auto action : situation) {
      status = model->InvokeAgent(action);
      EXPECT_TRUE(status.IsOk()) << "Expect successful agent invocation with "
                                 << ni::TRITONREPOAGENT_ActionTypeString(action)
                                 << ": " << status.AsString();
    }
    model.reset();

    // Check log
    ASSERT_EQ(log.size(), expected_log.size())
        << "Expect " << expected_log.size()
        << " state of model lifecycle is logged, got " << log.size();
    for (size_t i = 0; i < log.size(); i++) {
      EXPECT_EQ(log[i], expected_log[i]);
    }
  }
}

TEST_F(TritonRepoAgentModelTest, WrongLifeCycle)
{
  // Create agent to be associated with the model
  std::shared_ptr<ni::TritonRepoAgent> agent;
  auto status = ni::TritonRepoAgent::Create("agent", "log_agent_path", &agent);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation: " << status.AsString();
  std::vector<std::string> log;
  agent->SetState(reinterpret_cast<void*>(&log));

  // Create model and run all action combinations
  std::vector<std::vector<TRITONREPOAGENT_ActionType>> valid_lifecycles{
      {TRITONREPOAGENT_ACTION_LOAD, TRITONREPOAGENT_ACTION_LOAD_FAIL},
      {TRITONREPOAGENT_ACTION_LOAD, TRITONREPOAGENT_ACTION_LOAD_COMPLETE,
       TRITONREPOAGENT_ACTION_UNLOAD, TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE}};
  std::vector<TRITONREPOAGENT_ActionType> available_actions{
      TRITONREPOAGENT_ACTION_LOAD, TRITONREPOAGENT_ACTION_LOAD_FAIL,
      TRITONREPOAGENT_ACTION_LOAD_COMPLETE, TRITONREPOAGENT_ACTION_UNLOAD,
      TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE};
  std::map<TRITONREPOAGENT_ActionType, std::set<TRITONREPOAGENT_ActionType>>
      valid_actions{{TRITONREPOAGENT_ACTION_LOAD,
                     {TRITONREPOAGENT_ACTION_LOAD_FAIL,
                      TRITONREPOAGENT_ACTION_LOAD_COMPLETE}},
                    {TRITONREPOAGENT_ACTION_LOAD_FAIL, {}},
                    {TRITONREPOAGENT_ACTION_LOAD_COMPLETE,
                     {TRITONREPOAGENT_ACTION_UNLOAD}},
                    {TRITONREPOAGENT_ACTION_UNLOAD,
                     {TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE}},
                    {TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE, {}}};
  for (const auto& valid_lifecycle : valid_lifecycles) {
    log.clear();
    std::unique_ptr<ni::TritonRepoAgentModel> model;
    status = ni::TritonRepoAgentModel::Create(
        original_type_, original_location_, simple_config_, agent,
        ni::TritonRepoAgent::Parameters(), &model);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful model creation: " << status.AsString();
    for (size_t idx = 0; idx < valid_lifecycle.size(); idx++) {
      const auto next_lifecycle_action = valid_lifecycle[idx];
      // Handle the first action specially
      if (idx == 0) {
        for (const auto action : available_actions) {
          if (action == valid_lifecycle[0]) {
            continue;
          }
          status = model->InvokeAgent(action);
          if (status.IsOk()) {
            for (const auto& state_log : log) {
              EXPECT_TRUE(false) << state_log;
            }
          }
          ASSERT_FALSE(status.IsOk())
              << "Unexpect successful agent invocation with "
              << ni::TRITONREPOAGENT_ActionTypeString(action);
        }
        status = model->InvokeAgent(valid_lifecycle[0]);
        if (!status.IsOk()) {
          for (const auto& state_log : log) {
            EXPECT_TRUE(false) << state_log;
          }
        }
        ASSERT_TRUE(status.IsOk())
            << "Expect successful agent invocation with "
            << ni::TRITONREPOAGENT_ActionTypeString(next_lifecycle_action)
            << ": " << status.AsString();
        continue;
      }
      const auto& current_valid_actions =
          valid_actions[valid_lifecycle[idx - 1]];
      for (const auto action : available_actions) {
        if (current_valid_actions.find(action) != current_valid_actions.end()) {
          continue;
        }
        status = model->InvokeAgent(action);
        if (status.IsOk()) {
          for (const auto& state_log : log) {
            EXPECT_TRUE(false) << state_log;
          }
        }
        ASSERT_FALSE(status.IsOk())
            << "Unexpect successful agent invocation with "
            << ni::TRITONREPOAGENT_ActionTypeString(action);
      }
      status = model->InvokeAgent(next_lifecycle_action);
      if (!status.IsOk()) {
        for (const auto& state_log : log) {
          EXPECT_TRUE(false) << state_log;
        }
      }
      ASSERT_TRUE(status.IsOk())
          << "Expect successful agent invocation with "
          << ni::TRITONREPOAGENT_ActionTypeString(next_lifecycle_action) << ": "
          << status.AsString();
    }
  }
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
