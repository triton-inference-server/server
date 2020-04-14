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

#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#include <functional>
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
      ni::StatusCodeToTritonCode(status.StatusCode()),
      status.Message().c_str());
}

class MockSharedLibraryHandle {
 public:
  bool AddEntryPoint(const std::string& name, void* fn)
  {
    auto it = entry_points_.find(name);
    if (it == entry_points_.end()) {
      entry_points_.emplace(name, fn).second;
      return true;
    } else {
      it->second = fn;
      return false;
    }
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

TRITONSERVER_Error*
TRITONSERVER_MessageSerializeToJson(
    TRITONSERVER_Message* message, const char** base, size_t* byte_size)
{
  ni::TritonServerMessage* lmessage =
      reinterpret_cast<ni::TritonServerMessage*>(message);
  lmessage->Serialize(base, byte_size);
  return nullptr;  // Success
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

TEST_F(TritonRepoAgentTest, CreateFailInvalidSharedLibrary)
{
  // Passing a agent path that is not in global_mock_agents to
  // simulate failure on opening shared library handle
  std::shared_ptr<ni::TritonRepoAgent> invalid_agent;
  auto status = ni::TritonRepoAgent::Create(
      "invalid_agent", "invalid_agent_path", &invalid_agent);
  ASSERT_FALSE(status.IsOk()) << "Unexpect successful agent creation";
  EXPECT_NE(
      status.Message().find("unable to load shared library"), std::string::npos)
      << "Unexpect error message: '" << status.Message()
      << "', expect 'unable to load shared library...'";
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

class TritonRepoAgentManagerTest : public ::testing::Test {
 public:
  static size_t agent_init_counter_;
  static size_t agent_fini_counter_;

 protected:
  void SetUp() override
  {
    // Set up agent with init / fini function defined
    ni::TritonRepoAgent::TritonRepoAgentInitFn_t InitFn =
        [](TRITONREPOAGENT_Agent* agent) -> TRITONSERVER_Error* {
      agent_init_counter_++;
      return nullptr;
    };
    ni::TritonRepoAgent::TritonRepoAgentFiniFn_t FiniFn =
        [](TRITONREPOAGENT_Agent* agent) -> TRITONSERVER_Error* {
      agent_fini_counter_++;
      return nullptr;
    };
    ni::TritonRepoAgent::TritonRepoAgentModelActionFn_t ActionFn =
        [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
           const TRITONREPOAGENT_ActionType action_type)
        -> TRITONSERVER_Error* { return nullptr; };
    auto agent_handle = MockSharedLibraryHandle();
    agent_handle.AddEntryPoint(
        "TRITONREPOAGENT_Initialize", reinterpret_cast<void*>(InitFn));
    agent_handle.AddEntryPoint(
        "TRITONREPOAGENT_Finalize", reinterpret_cast<void*>(FiniFn));
    agent_handle.AddEntryPoint(
        "TRITONREPOAGENT_ModelAction", reinterpret_cast<void*>(ActionFn));

    // Reserve valid shared library paths because manager searches the libraries
    // via the FileSystem API
    const ni::FileSystemType type = ni::FileSystemType::LOCAL;
    auto status = ni::MakeTemporaryDirectory(type, &root_agent_path_);
    ASSERT_TRUE(status.IsOk()) << "TritonRepoAgentManagerTest set up failed: "
                                  "create temporary directory: "
                               << status.AsString();
    // FIXME make the following platform independent
    global_agent_path_ = ni::JoinPath({root_agent_path_, "global"});
    int err = mkdir(
        global_agent_path_.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    ASSERT_EQ(err, 0) << "TritonRepoAgentManagerTest set up failed: create "
                         "global agent directory: "
                      << err;
    const std::set<std::string> agent_names{"global_agent"};
    for (const auto& agent_name : agent_names) {
      auto global_path_to_agent =
          ni::JoinPath({global_agent_path_, agent_name});
      auto global_agent = ni::JoinPath(
          {global_path_to_agent, ni::TritonRepoAgentLibraryName(agent_name)});
      err = mkdir(
          global_path_to_agent.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      ASSERT_EQ(err, 0) << "TritonRepoAgentManagerTest set up failed: create "
                           "global agent directory: "
                        << err;
      std::ofstream global_agent_file(global_agent);
      global_mock_agents.emplace(global_agent, agent_handle);
    }
    status =
        ni::TritonRepoAgentManager::SetGlobalSearchPath(global_agent_path_);
    ASSERT_TRUE(status.IsOk()) << "TritonRepoAgentManagerTest set up failed: "
                                  "create temporary directory: "
                               << status.AsString();
  }
  void TearDown() override
  {
    agent_init_counter_ = 0;
    agent_fini_counter_ = 0;
    if (!root_agent_path_.empty()) {
      // ni::DeleteDirectory(root_agent_path_);
    }
    global_mock_agents.clear();
  }

  std::string root_agent_path_;
  std::string global_agent_path_;
  std::string local_agent_path_;
};
size_t TritonRepoAgentManagerTest::agent_init_counter_ = 0;
size_t TritonRepoAgentManagerTest::agent_fini_counter_ = 0;

TEST_F(TritonRepoAgentManagerTest, CreateFailureFileNotExist)
{
  // Passing a agent path that is not in global_mock_agents to
  // simulate failure on opening shared library handle
  std::shared_ptr<ni::TritonRepoAgent> invalid_agent;
  auto status = ni::TritonRepoAgentManager::CreateAgent(
      "invalid_agent_name", &invalid_agent);
  ASSERT_FALSE(status.IsOk()) << "Unexpect successful agent creation";
  EXPECT_NE(status.Message().find("unable to find"), std::string::npos)
      << "Unexpect error message: '" << status.Message()
      << "', expect 'unable to find...'";
}

TEST_F(TritonRepoAgentManagerTest, CreateGlobalAgent)
{
  std::shared_ptr<ni::TritonRepoAgent> agent;
  auto status = ni::TritonRepoAgentManager::CreateAgent("global_agent", &agent);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation" << status.AsString();
  agent.reset();
  EXPECT_EQ(agent_init_counter_, (size_t)1) << "Expect 1 agent initialization";
  EXPECT_EQ(agent_fini_counter_, (size_t)1) << "Expect 1 agent finalization";
}

TEST_F(TritonRepoAgentManagerTest, AgentPersistence)
{
  std::shared_ptr<ni::TritonRepoAgent> agent1;
  std::shared_ptr<ni::TritonRepoAgent> agent2;
  auto status =
      ni::TritonRepoAgentManager::CreateAgent("global_agent", &agent1);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation" << status.AsString();
  EXPECT_EQ(agent_init_counter_, (size_t)1) << "Expect 1 agent initialization";
  EXPECT_EQ(agent_fini_counter_, (size_t)0) << "Expect 0 agent finalization";

  status = ni::TritonRepoAgentManager::CreateAgent("global_agent", &agent2);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation" << status.AsString();
  EXPECT_EQ(agent_init_counter_, (size_t)1) << "Expect 1 agent initialization";
  EXPECT_EQ(agent_fini_counter_, (size_t)0) << "Expect 0 agent finalization";

  agent1.reset();
  EXPECT_EQ(agent_init_counter_, (size_t)1) << "Expect 1 agent initialization";
  EXPECT_EQ(agent_fini_counter_, (size_t)0) << "Expect 0 agent finalization";
  agent2.reset();
  EXPECT_EQ(agent_init_counter_, (size_t)1) << "Expect 1 agent initialization";
  EXPECT_EQ(agent_fini_counter_, (size_t)1) << "Expect 1 agent finalization";

  // Create again after all previous agents are reset
  status = ni::TritonRepoAgentManager::CreateAgent("global_agent", &agent1);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation" << status.AsString();
  EXPECT_EQ(agent_init_counter_, (size_t)2) << "Expect 2 agent initialization";
  EXPECT_EQ(agent_fini_counter_, (size_t)1) << "Expect 1 agent finalization";
  agent1.reset();
  EXPECT_EQ(agent_init_counter_, (size_t)2) << "Expect 2 agent initialization";
  EXPECT_EQ(agent_fini_counter_, (size_t)2) << "Expect 2 agent finalization";
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

  TRITONREPOAGENT_ArtifactType original_type_ =
      TRITONREPOAGENT_ARTIFACT_FILESYSTEM;
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
  TRITONREPOAGENT_ArtifactType type;
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
  TRITONREPOAGENT_ArtifactType type = TRITONREPOAGENT_ARTIFACT_FILESYSTEM;
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
  static const TRITONREPOAGENT_ArtifactType new_type =
      TRITONREPOAGENT_ARTIFACT_FILESYSTEM;
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
  TRITONREPOAGENT_ArtifactType type = original_type_;
  const char* location = original_location_.c_str();
  status = model->Location(&type, &location);
  ASSERT_TRUE(status.IsOk()) << "Expect location is returned from Location()";
  EXPECT_EQ(type, new_type) << "Expect returned filesystem type is "
                            << ni::TRITONREPOAGENT_ArtifactTypeString(new_type);
  EXPECT_EQ(std::string(location), new_location)
      << "Expect returned location is " << new_location;
}

TEST_F(TritonRepoAgentModelTest, SetLocationWrongActionFailure)
{
  static const TRITONREPOAGENT_ArtifactType new_type =
      TRITONREPOAGENT_ARTIFACT_FILESYSTEM;
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
  static const TRITONREPOAGENT_ArtifactType new_type =
      TRITONREPOAGENT_ARTIFACT_FILESYSTEM;
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
  TRITONREPOAGENT_ArtifactType type = original_type_;
  const char* location = original_location_.c_str();
  status = model->Location(&type, &location);
  ASSERT_TRUE(status.IsOk()) << "Expect location is returned from Location()";
  EXPECT_EQ(type, new_type) << "Expect returned filesystem type is "
                            << ni::TRITONREPOAGENT_ArtifactTypeString(new_type);
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
      TRITONREPOAGENT_ARTIFACT_FILESYSTEM, &acquired_location);
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
      TRITONREPOAGENT_ARTIFACT_FILESYSTEM, &acquired_location);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful location acquisition: " << status.AsString();

  // Acquire the same type again
  const char* second_acquired_location;
  status = model->AcquireMutableLocation(
      TRITONREPOAGENT_ARTIFACT_FILESYSTEM, &second_acquired_location);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful location acquisition: " << status.AsString();
  EXPECT_EQ(
      std::string(acquired_location), std::string(second_acquired_location))
      << "Expect the same location is returned";
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
      TRITONREPOAGENT_ARTIFACT_FILESYSTEM, &acquired_location);
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

TEST_F(TritonRepoAgentModelTest, AcquireRemoteLocation)
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
      TRITONREPOAGENT_ARTIFACT_REMOTE_FILESYSTEM, &acquired_location);
  ASSERT_FALSE(status.IsOk())
      << "Expect error returned from AcquireMutableLocation()";
  const std::string search_msg =
      "Unexpected artifact type, expects 'TRITONREPOAGENT_ARTIFACT_FILESYSTEM'";
  EXPECT_NE(status.Message().find(search_msg), std::string::npos)
      << "Unexpect error message: '" << status.Message() << "', expect '"
      << search_msg << "'";
}

TEST_F(TritonRepoAgentModelTest, AgentParameters)
{
  // Create agent to be associated with the model
  std::shared_ptr<ni::TritonRepoAgent> agent;
  auto status =
      ni::TritonRepoAgent::Create("agent", "simple_agent_path", &agent);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful agent creation: " << status.AsString();

  // Create model
  ni::TritonRepoAgent::Parameters expected_params{{"key_a", "value_b"},
                                                  {"key_b", "value_b"}};
  std::unique_ptr<ni::TritonRepoAgentModel> model;
  status = ni::TritonRepoAgentModel::Create(
      original_type_, original_location_, simple_config_, agent,
      expected_params, &model);
  ASSERT_TRUE(status.IsOk())
      << "Expect successful model creation: " << status.AsString();
  auto agent_params = model->AgentParameters();
  ASSERT_EQ(agent_params.size(), expected_params.size());
  for (size_t idx = 0; idx < agent_params.size(); ++idx) {
    EXPECT_EQ(agent_params[idx].first, expected_params[idx].first);
    EXPECT_EQ(agent_params[idx].second, expected_params[idx].second);
  }
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
    for (size_t i = 0; i < log.size(); ++i) {
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
    for (size_t i = 0; i < log.size(); ++i) {
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
    for (size_t idx = 0; idx < valid_lifecycle.size(); ++idx) {
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

class TritonRepoAgentAPITest : public ::testing::Test {
 public:
  static std::function<void(TRITONREPOAGENT_Agent*)> agent_init_fn_;
  static std::function<void(TRITONREPOAGENT_Agent*)> agent_fini_fn_;
  static std::function<void(
      TRITONREPOAGENT_Agent*, TRITONREPOAGENT_AgentModel*)>
      model_init_fn_;
  static std::function<void(
      TRITONREPOAGENT_Agent*, TRITONREPOAGENT_AgentModel*)>
      model_action_fn_;
  static std::function<void(
      TRITONREPOAGENT_Agent*, TRITONREPOAGENT_AgentModel*)>
      model_fini_fn_;

 protected:
  void SetUp() override
  {
    simple_config_.set_name("simple_config");
    // Add a agent handle for flexible testing
    ni::TritonRepoAgent::TritonRepoAgentInitFn_t AgentInitFn =
        [](TRITONREPOAGENT_Agent* agent) -> TRITONSERVER_Error* {
      if (agent_init_fn_ != nullptr) {
        agent_init_fn_(agent);
      }
      return nullptr;
    };
    ni::TritonRepoAgent::TritonRepoAgentFiniFn_t AgentFiniFn =
        [](TRITONREPOAGENT_Agent* agent) -> TRITONSERVER_Error* {
      if (agent_fini_fn_ != nullptr) {
        agent_fini_fn_(agent);
      }
      return nullptr;
    };
    ni::TritonRepoAgent::TritonRepoAgentModelInitFn_t ModelInitFn =
        [](TRITONREPOAGENT_Agent* agent,
           TRITONREPOAGENT_AgentModel* model) -> TRITONSERVER_Error* {
      if (model_init_fn_ != nullptr) {
        model_init_fn_(agent, model);
      }
      return nullptr;
    };
    ni::TritonRepoAgent::TritonRepoAgentModelActionFn_t ModelActionFn =
        [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
           const TRITONREPOAGENT_ActionType action_type)
        -> TRITONSERVER_Error* {
      if (model_action_fn_ != nullptr) {
        model_action_fn_(agent, model);
      }
      return nullptr;
    };
    ni::TritonRepoAgent::TritonRepoAgentModelFiniFn_t ModelFiniFn =
        [](TRITONREPOAGENT_Agent* agent,
           TRITONREPOAGENT_AgentModel* model) -> TRITONSERVER_Error* {
      if (model_fini_fn_ != nullptr) {
        model_fini_fn_(agent, model);
      }
      return nullptr;
    };
    auto agent_handle = MockSharedLibraryHandle();
    agent_handle.AddEntryPoint(
        "TRITONREPOAGENT_Initialize", reinterpret_cast<void*>(AgentInitFn));
    agent_handle.AddEntryPoint(
        "TRITONREPOAGENT_Finalize", reinterpret_cast<void*>(AgentFiniFn));
    agent_handle.AddEntryPoint(
        "TRITONREPOAGENT_ModelInitialize",
        reinterpret_cast<void*>(ModelInitFn));
    agent_handle.AddEntryPoint(
        "TRITONREPOAGENT_ModelAction", reinterpret_cast<void*>(ModelActionFn));
    agent_handle.AddEntryPoint(
        "TRITONREPOAGENT_ModelFinalize", reinterpret_cast<void*>(ModelFiniFn));
    global_mock_agents.emplace("agent_path", agent_handle);
  }
  void TearDown() override
  {
    global_mock_agents.clear();
    agent_init_fn_ = nullptr;
    agent_fini_fn_ = nullptr;
    model_init_fn_ = nullptr;
    model_action_fn_ = nullptr;
    model_fini_fn_ = nullptr;
  }

  TRITONREPOAGENT_ArtifactType original_type_ =
      TRITONREPOAGENT_ARTIFACT_FILESYSTEM;
  const std::string original_location_ = "/original";
  inference::ModelConfig simple_config_;

  std::vector<std::vector<TRITONREPOAGENT_ActionType>> valid_lifecycles_{
      {TRITONREPOAGENT_ACTION_LOAD, TRITONREPOAGENT_ACTION_LOAD_FAIL},
      {TRITONREPOAGENT_ACTION_LOAD, TRITONREPOAGENT_ACTION_LOAD_COMPLETE,
       TRITONREPOAGENT_ACTION_UNLOAD, TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE}};
};

std::function<void(TRITONREPOAGENT_Agent*)>
    TritonRepoAgentAPITest::agent_init_fn_ = nullptr;
std::function<void(TRITONREPOAGENT_Agent*)>
    TritonRepoAgentAPITest::agent_fini_fn_ = nullptr;
std::function<void(TRITONREPOAGENT_Agent*, TRITONREPOAGENT_AgentModel*)>
    TritonRepoAgentAPITest::model_init_fn_ = nullptr;
std::function<void(TRITONREPOAGENT_Agent*, TRITONREPOAGENT_AgentModel*)>
    TritonRepoAgentAPITest::model_action_fn_ = nullptr;
std::function<void(TRITONREPOAGENT_Agent*, TRITONREPOAGENT_AgentModel*)>
    TritonRepoAgentAPITest::model_fini_fn_ = nullptr;

TEST_F(TritonRepoAgentAPITest, TRITONREPOAGENT_ApiVersion)
{
  agent_init_fn_ =
      [](TRITONREPOAGENT_Agent* agent) {
        uint32_t major = 0;
        uint32_t minor = 0;
        auto err = TRITONREPOAGENT_ApiVersion(&major, &minor);
        if (err != nullptr) {
          EXPECT_TRUE(false)
              << "Expect successful TRITONREPOAGENT_ApiVersion() invokation: "
              << TRITONSERVER_ErrorMessage(err);
          TRITONSERVER_ErrorDelete(err);
        } else {
          EXPECT_EQ(major, (uint32_t)TRITONREPOAGENT_API_VERSION_MAJOR)
              << "Unexpected major veresion";
          EXPECT_EQ(minor, (uint32_t)TRITONREPOAGENT_API_VERSION_MINOR)
              << "Unexpected major veresion";
        }
      };
  agent_fini_fn_ = agent_init_fn_;
  model_init_fn_ =
      [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model) {
        uint32_t major = 0;
        uint32_t minor = 0;
        auto err = TRITONREPOAGENT_ApiVersion(&major, &minor);
        if (err != nullptr) {
          EXPECT_TRUE(false)
              << "Expect successful TRITONREPOAGENT_ApiVersion() invokation: "
              << TRITONSERVER_ErrorMessage(err);
          TRITONSERVER_ErrorDelete(err);
        } else {
          EXPECT_EQ(major, (uint32_t)TRITONREPOAGENT_API_VERSION_MAJOR)
              << "Unexpected major veresion";
          EXPECT_EQ(minor, (uint32_t)TRITONREPOAGENT_API_VERSION_MINOR)
              << "Unexpected major veresion";
        }
      };
  model_action_fn_ = model_init_fn_;
  model_fini_fn_ = model_init_fn_;

  const auto lifecycles = valid_lifecycles_;
  for (const auto& lifecycle : lifecycles) {
    // Create agent to be associated with the model
    std::shared_ptr<ni::TritonRepoAgent> agent;
    auto status = ni::TritonRepoAgent::Create("agent", "agent_path", &agent);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful agent creation: " << status.AsString();
    std::unique_ptr<ni::TritonRepoAgentModel> model;
    status = ni::TritonRepoAgentModel::Create(
        original_type_, original_location_, simple_config_, agent,
        ni::TritonRepoAgent::Parameters(), &model);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful model creation: " << status.AsString();
    for (const auto action : lifecycle) {
      status = model->InvokeAgent(action);
      ASSERT_TRUE(status.IsOk()) << "Expect successful agent invocation with "
                                 << ni::TRITONREPOAGENT_ActionTypeString(action)
                                 << ": " << status.AsString();
    }
  }
}

TEST_F(TritonRepoAgentAPITest, TRITONREPOAGENT_ModelRepositoryLocation)
{
  model_init_fn_ =
      [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model) {
        TRITONREPOAGENT_ArtifactType artifact_type =
            TRITONREPOAGENT_ARTIFACT_REMOTE_FILESYSTEM;
        const char* location = nullptr;
        auto err = TRITONREPOAGENT_ModelRepositoryLocation(
            agent, model, &artifact_type, &location);
        if (err != nullptr) {
          EXPECT_TRUE(false)
              << "Expect successful TRITONREPOAGENT_ModelRepositoryLocation(): "
              << TRITONSERVER_ErrorMessage(err);
          TRITONSERVER_ErrorDelete(err);
        } else {
          EXPECT_EQ(artifact_type, TRITONREPOAGENT_ARTIFACT_FILESYSTEM)
              << "Unexpected artifact type";
          EXPECT_EQ(std::string(location), "/original")
              << "Unexpected location";
        }
      };
  model_action_fn_ = model_init_fn_;
  model_fini_fn_ = model_init_fn_;

  const auto lifecycles = valid_lifecycles_;
  for (const auto& lifecycle : lifecycles) {
    // Create agent to be associated with the model
    std::shared_ptr<ni::TritonRepoAgent> agent;
    auto status = ni::TritonRepoAgent::Create("agent", "agent_path", &agent);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful agent creation: " << status.AsString();
    std::unique_ptr<ni::TritonRepoAgentModel> model;
    status = ni::TritonRepoAgentModel::Create(
        original_type_, original_location_, simple_config_, agent,
        ni::TritonRepoAgent::Parameters(), &model);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful model creation: " << status.AsString();
    for (const auto action : lifecycle) {
      status = model->InvokeAgent(action);
      ASSERT_TRUE(status.IsOk()) << "Expect successful agent invocation with "
                                 << ni::TRITONREPOAGENT_ActionTypeString(action)
                                 << ": " << status.AsString();
    }
  }
}

TEST_F(
    TritonRepoAgentAPITest,
    TRITONREPOAGENT_ModelRepositoryLocationAcquireRemote)
{
  model_init_fn_ = [](TRITONREPOAGENT_Agent* agent,
                      TRITONREPOAGENT_AgentModel* model) {
    TRITONREPOAGENT_ArtifactType artifact_type =
        TRITONREPOAGENT_ARTIFACT_REMOTE_FILESYSTEM;
    const char* location = nullptr;
    auto err = TRITONREPOAGENT_ModelRepositoryLocationAcquire(
        agent, model, artifact_type, &location);
    if (err != nullptr) {
      const std::string err_msg = TRITONSERVER_ErrorMessage(err);
      const std::string search_msg =
          "Unexpected artifact type, expects "
          "'TRITONREPOAGENT_ARTIFACT_FILESYSTEM'";
      EXPECT_NE(err_msg.find(search_msg), std::string::npos)
          << "Unexpect error message: '" << err_msg << "', expect '"
          << search_msg << "'";
      TRITONSERVER_ErrorDelete(err);
    } else {
      EXPECT_TRUE(false) << "Expect error returned from "
                            "TRITONREPOAGENT_ModelRepositoryLocationAcquire()";
    }
  };
  model_action_fn_ = model_init_fn_;
  model_fini_fn_ = model_init_fn_;

  const auto lifecycles = valid_lifecycles_;
  for (const auto& lifecycle : lifecycles) {
    // Create agent to be associated with the model
    std::shared_ptr<ni::TritonRepoAgent> agent;
    auto status = ni::TritonRepoAgent::Create("agent", "agent_path", &agent);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful agent creation: " << status.AsString();
    std::unique_ptr<ni::TritonRepoAgentModel> model;
    status = ni::TritonRepoAgentModel::Create(
        original_type_, original_location_, simple_config_, agent,
        ni::TritonRepoAgent::Parameters(), &model);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful model creation: " << status.AsString();
    for (const auto action : lifecycle) {
      status = model->InvokeAgent(action);
      ASSERT_TRUE(status.IsOk()) << "Expect successful agent invocation with "
                                 << ni::TRITONREPOAGENT_ActionTypeString(action)
                                 << ": " << status.AsString();
    }
  }
}

TEST_F(TritonRepoAgentAPITest, TRITONREPOAGENT_ModelRepositoryLocationAcquire)
{
  model_init_fn_ = [](TRITONREPOAGENT_Agent* agent,
                      TRITONREPOAGENT_AgentModel* model) {
    // Acquire, acquire (same), release
    TRITONREPOAGENT_ArtifactType artifact_type =
        TRITONREPOAGENT_ARTIFACT_FILESYSTEM;
    const char* location = nullptr;
    auto err = TRITONREPOAGENT_ModelRepositoryLocationAcquire(
        agent, model, artifact_type, &location);
    if (err != nullptr) {
      EXPECT_TRUE(false) << "Expect successful "
                            "TRITONREPOAGENT_ModelRepositoryLocationAcquire(): "
                         << TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
    }

    std::string acquired_location = location;
    err = TRITONREPOAGENT_ModelRepositoryLocationAcquire(
        agent, model, artifact_type, &location);
    if (err != nullptr) {
      EXPECT_TRUE(false) << "Expect successful "
                            "TRITONREPOAGENT_ModelRepositoryLocationAcquire(): "
                         << TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
    } else {
      EXPECT_EQ(acquired_location, std::string(location))
          << "Expect the same location is acquired";
    }
  };
  model_action_fn_ = model_init_fn_;
  model_fini_fn_ = model_init_fn_;

  const auto lifecycles = valid_lifecycles_;
  for (const auto& lifecycle : lifecycles) {
    // Create agent to be associated with the model
    std::shared_ptr<ni::TritonRepoAgent> agent;
    auto status = ni::TritonRepoAgent::Create("agent", "agent_path", &agent);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful agent creation: " << status.AsString();
    std::unique_ptr<ni::TritonRepoAgentModel> model;
    status = ni::TritonRepoAgentModel::Create(
        original_type_, original_location_, simple_config_, agent,
        ni::TritonRepoAgent::Parameters(), &model);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful model creation: " << status.AsString();
    for (const auto action : lifecycle) {
      status = model->InvokeAgent(action);
      ASSERT_TRUE(status.IsOk()) << "Expect successful agent invocation with "
                                 << ni::TRITONREPOAGENT_ActionTypeString(action)
                                 << ": " << status.AsString();
    }
  }
}

TEST_F(TritonRepoAgentAPITest, TRITONREPOAGENT_ModelRepositoryLocationRelease)
{
  model_init_fn_ =
      [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model) {
        // relase (fail), acquire, release
        const char* location = "nonexisting_location";
        auto err = TRITONREPOAGENT_ModelRepositoryLocationRelease(
            agent, model, location);
        if (err != nullptr) {
          const std::string search_msg = "No mutable location to be deleted";
          const std::string err_msg = TRITONSERVER_ErrorMessage(err);
          EXPECT_NE(err_msg.find(search_msg), std::string::npos)
              << "Unexpect error message: '" << err_msg << "', expect '"
              << search_msg << "'";
          TRITONSERVER_ErrorDelete(err);
        } else {
          EXPECT_TRUE(false)
              << "Expect error returned from "
                 "TRITONREPOAGENT_ModelRepositoryLocationRelease()";
        }

        TRITONREPOAGENT_ArtifactType artifact_type =
            TRITONREPOAGENT_ARTIFACT_FILESYSTEM;
        err = TRITONREPOAGENT_ModelRepositoryLocationAcquire(
            agent, model, artifact_type, &location);
        if (err != nullptr) {
          EXPECT_TRUE(false)
              << "Expect successful TRITONREPOAGENT_ModelRepositoryLocation(): "
              << TRITONSERVER_ErrorMessage(err);
          TRITONSERVER_ErrorDelete(err);
        }

        err = TRITONREPOAGENT_ModelRepositoryLocationRelease(
            agent, model, location);
        if (err != nullptr) {
          EXPECT_TRUE(false)
              << "Expect successful TRITONREPOAGENT_ModelRepositoryLocation(): "
              << TRITONSERVER_ErrorMessage(err);
          TRITONSERVER_ErrorDelete(err);
        }
      };
  model_action_fn_ = model_init_fn_;
  model_fini_fn_ = model_init_fn_;

  const auto lifecycles = valid_lifecycles_;
  for (const auto& lifecycle : lifecycles) {
    // Create agent to be associated with the model
    std::shared_ptr<ni::TritonRepoAgent> agent;
    auto status = ni::TritonRepoAgent::Create("agent", "agent_path", &agent);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful agent creation: " << status.AsString();
    std::unique_ptr<ni::TritonRepoAgentModel> model;
    status = ni::TritonRepoAgentModel::Create(
        original_type_, original_location_, simple_config_, agent,
        ni::TritonRepoAgent::Parameters(), &model);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful model creation: " << status.AsString();
    for (const auto action : lifecycle) {
      status = model->InvokeAgent(action);
      ASSERT_TRUE(status.IsOk()) << "Expect successful agent invocation with "
                                 << ni::TRITONREPOAGENT_ActionTypeString(action)
                                 << ": " << status.AsString();
    }
  }
}

TEST_F(TritonRepoAgentAPITest, TRITONREPOAGENT_ModelRepositoryUpdate)
{
  static std::string current_location = original_location_;
  static TRITONREPOAGENT_ArtifactType current_type =
      TRITONREPOAGENT_ARTIFACT_FILESYSTEM;
  model_init_fn_ =
      [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model) {
        std::string new_location = current_location + "_new";
        TRITONREPOAGENT_ArtifactType artifact_type =
            TRITONREPOAGENT_ARTIFACT_FILESYSTEM;
        const char* location = new_location.c_str();
        auto err = TRITONREPOAGENT_ModelRepositoryUpdate(
            agent, model, artifact_type, location);
        if (err != nullptr) {
          const std::string search_msg =
              "location can only be updated during TRITONREPOAGENT_ACTION_LOAD";
          const std::string err_msg = TRITONSERVER_ErrorMessage(err);
          EXPECT_NE(err_msg.find(search_msg), std::string::npos)
              << "Unexpect error message: '" << err_msg << "', expect '"
              << search_msg << "...'";
          TRITONSERVER_ErrorDelete(err);
        } else {
          EXPECT_TRUE(false) << "Expect error returned from "
                                "TRITONREPOAGENT_ModelRepositoryUpdate()";
        }

        // Check location shouldn't be changed
        err = TRITONREPOAGENT_ModelRepositoryLocation(
            agent, model, &artifact_type, &location);
        if (err != nullptr) {
          EXPECT_TRUE(false)
              << "Expect successful TRITONREPOAGENT_ModelRepositoryLocation(): "
              << TRITONSERVER_ErrorMessage(err);
          TRITONSERVER_ErrorDelete(err);
        } else {
          EXPECT_EQ(artifact_type, current_type) << "Unexpected artifact type";
          EXPECT_EQ(std::string(location), current_location)
              << "Unexpected location";
        }
      };
  model_action_fn_ = model_init_fn_;
  model_fini_fn_ = model_init_fn_;

  // Overriding the model action function in agent handle because the action
  // type needs to be checked here
  ni::TritonRepoAgent::TritonRepoAgentModelActionFn_t ModelActionFn =
      [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
         const TRITONREPOAGENT_ActionType action_type) -> TRITONSERVER_Error* {
    std::string new_location = current_location + "_new";
    TRITONREPOAGENT_ArtifactType artifact_type =
        TRITONREPOAGENT_ARTIFACT_REMOTE_FILESYSTEM;
    const char* location = new_location.c_str();
    auto err = TRITONREPOAGENT_ModelRepositoryUpdate(
        agent, model, artifact_type, location);
    if (action_type == TRITONREPOAGENT_ACTION_LOAD) {
      if (err != nullptr) {
        EXPECT_TRUE(false)
            << "Expect successful TRITONREPOAGENT_ModelRepositoryUpdate(): "
            << TRITONSERVER_ErrorMessage(err);
        TRITONSERVER_ErrorDelete(err);
      } else {
        current_location = new_location;
        current_type = artifact_type;
      }
    } else {
      if (err != nullptr) {
        const std::string search_msg =
            "location can only be updated during TRITONREPOAGENT_ACTION_LOAD";
        const std::string err_msg = TRITONSERVER_ErrorMessage(err);
        EXPECT_NE(err_msg.find(search_msg), std::string::npos)
            << "Unexpect error message: '" << err_msg << "', expect '"
            << search_msg << "...'";
        TRITONSERVER_ErrorDelete(err);
      } else {
        EXPECT_TRUE(false) << "Expect error returned from "
                              "TRITONREPOAGENT_ModelRepositoryUpdate()";
      }
    }

    // Check location
    err = TRITONREPOAGENT_ModelRepositoryLocation(
        agent, model, &artifact_type, &location);
    if (err != nullptr) {
      EXPECT_TRUE(false)
          << "Expect successful TRITONREPOAGENT_ModelRepositoryLocation(): "
          << TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
    } else {
      EXPECT_EQ(artifact_type, current_type) << "Unexpected artifact type";
      EXPECT_EQ(std::string(location), current_location)
          << "Unexpected location";
    }
    return nullptr;
  };
  global_mock_agents["agent_path"].AddEntryPoint(
      "TRITONREPOAGENT_ModelAction", reinterpret_cast<void*>(ModelActionFn));

  const auto lifecycles = valid_lifecycles_;
  for (const auto& lifecycle : lifecycles) {
    // Reset location and type
    current_location = original_location_;
    current_type = TRITONREPOAGENT_ARTIFACT_FILESYSTEM;
    // Create agent to be associated with the model
    std::shared_ptr<ni::TritonRepoAgent> agent;
    auto status = ni::TritonRepoAgent::Create("agent", "agent_path", &agent);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful agent creation: " << status.AsString();
    std::unique_ptr<ni::TritonRepoAgentModel> model;
    status = ni::TritonRepoAgentModel::Create(
        original_type_, current_location, simple_config_, agent,
        ni::TritonRepoAgent::Parameters(), &model);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful model creation: " << status.AsString();
    for (const auto action : lifecycle) {
      status = model->InvokeAgent(action);
      ASSERT_TRUE(status.IsOk()) << "Expect successful agent invocation with "
                                 << ni::TRITONREPOAGENT_ActionTypeString(action)
                                 << ": " << status.AsString();
    }
  }
}

TEST_F(TritonRepoAgentAPITest, TRITONREPOAGENT_ModelParameter)
{
  static ni::TritonRepoAgent::Parameters expected_params{{"key_a", "value_a"},
                                                         {"key_b", "value_b"}};
  model_init_fn_ =
      [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model) {
        uint32_t count;
        auto err = TRITONREPOAGENT_ModelParameterCount(agent, model, &count);
        if (err != nullptr) {
          EXPECT_TRUE(false)
              << "Expect successful TRITONREPOAGENT_ModelParameterCount(): "
              << TRITONSERVER_ErrorMessage(err);
          TRITONSERVER_ErrorDelete(err);
        } else {
          EXPECT_EQ(count, expected_params.size());
        }

        const char* parameter_name = nullptr;
        const char* parameter_value = nullptr;
        for (size_t idx = 0; idx < count; ++idx) {
          err = TRITONREPOAGENT_ModelParameter(
              agent, model, idx, &parameter_name, &parameter_value);
          if (err != nullptr) {
            EXPECT_TRUE(false)
                << "Expect successful TRITONREPOAGENT_ModelParameter(): "
                << TRITONSERVER_ErrorMessage(err);
            TRITONSERVER_ErrorDelete(err);
          } else {
            EXPECT_EQ(std::string(parameter_name), expected_params[idx].first);
            EXPECT_EQ(
                std::string(parameter_value), expected_params[idx].second);
          }
        }
        // out of range
        err = TRITONREPOAGENT_ModelParameter(
            agent, model, count, &parameter_name, &parameter_value);
        if (err != nullptr) {
          const std::string search_msg =
              "index out of range for model parameters";
          const std::string err_msg = TRITONSERVER_ErrorMessage(err);
          EXPECT_NE(err_msg.find(search_msg), std::string::npos)
              << "Unexpect error message: '" << err_msg << "', expect '"
              << search_msg << "...'";
          TRITONSERVER_ErrorDelete(err);
        } else {
          EXPECT_TRUE(false)
              << "Expect error returned from TRITONREPOAGENT_ModelParameter()";
        }
      };
  model_action_fn_ = model_init_fn_;
  model_fini_fn_ = model_init_fn_;

  const auto lifecycles = valid_lifecycles_;
  for (const auto& lifecycle : lifecycles) {
    // Create agent to be associated with the model
    std::shared_ptr<ni::TritonRepoAgent> agent;
    auto status = ni::TritonRepoAgent::Create("agent", "agent_path", &agent);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful agent creation: " << status.AsString();
    std::unique_ptr<ni::TritonRepoAgentModel> model;
    status = ni::TritonRepoAgentModel::Create(
        original_type_, original_location_, simple_config_, agent,
        expected_params, &model);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful model creation: " << status.AsString();
    for (const auto action : lifecycle) {
      status = model->InvokeAgent(action);
      ASSERT_TRUE(status.IsOk()) << "Expect successful agent invocation with "
                                 << ni::TRITONREPOAGENT_ActionTypeString(action)
                                 << ": " << status.AsString();
    }
  }
}

TEST_F(TritonRepoAgentAPITest, TRITONREPOAGENT_ModelConfig)
{
  model_init_fn_ =
      [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model) {
        TRITONSERVER_Message* config = nullptr;
        auto err = TRITONREPOAGENT_ModelConfig(agent, model, 1, &config);
        if (err != nullptr) {
          EXPECT_TRUE(false)
              << "Expect successful TRITONREPOAGENT_ModelConfig(): "
              << TRITONSERVER_ErrorMessage(err);
          TRITONSERVER_ErrorDelete(err);
        }
        const char* base = nullptr;
        size_t byte_size = 0;
        err = TRITONSERVER_MessageSerializeToJson(config, &base, &byte_size);
        if (err != nullptr) {
          EXPECT_TRUE(false)
              << "Expect successful TRITONSERVER_MessageSerializeToJson(): "
              << TRITONSERVER_ErrorMessage(err);
          TRITONSERVER_ErrorDelete(err);
        } else {
          const std::string search_msg = "simple_config";
          const std::string serialized_config(base, byte_size);
          EXPECT_NE(serialized_config.find(search_msg), std::string::npos)
              << "Expect finding '" << search_msg
              << "' in returned config: " << serialized_config;
        }

        // unsupport version
        err = TRITONREPOAGENT_ModelConfig(agent, model, 2, &config);
        if (err != nullptr) {
          const std::string search_msg =
              "model configuration version 2 not supported, supported versions "
              "are: 1";
          const std::string err_msg = TRITONSERVER_ErrorMessage(err);
          EXPECT_NE(err_msg.find(search_msg), std::string::npos)
              << "Unexpect error message: '" << err_msg << "', expect '"
              << search_msg << "...'";
          TRITONSERVER_ErrorDelete(err);
        } else {
          EXPECT_TRUE(false)
              << "Expect error returned from TRITONREPOAGENT_ModelConfig()";
        }
      };
  model_action_fn_ = model_init_fn_;
  model_fini_fn_ = model_init_fn_;

  const auto lifecycles = valid_lifecycles_;
  for (const auto& lifecycle : lifecycles) {
    // Create agent to be associated with the model
    std::shared_ptr<ni::TritonRepoAgent> agent;
    auto status = ni::TritonRepoAgent::Create("agent", "agent_path", &agent);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful agent creation: " << status.AsString();
    std::unique_ptr<ni::TritonRepoAgentModel> model;
    status = ni::TritonRepoAgentModel::Create(
        original_type_, original_location_, simple_config_, agent,
        ni::TritonRepoAgent::Parameters(), &model);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful model creation: " << status.AsString();
    for (const auto action : lifecycle) {
      status = model->InvokeAgent(action);
      ASSERT_TRUE(status.IsOk()) << "Expect successful agent invocation with "
                                 << ni::TRITONREPOAGENT_ActionTypeString(action)
                                 << ": " << status.AsString();
    }
  }
}

TEST_F(TritonRepoAgentAPITest, TRITONREPOAGENT_ModelState)
{
  model_init_fn_ =
      [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model) {
        size_t* state = nullptr;
        auto err =
            TRITONREPOAGENT_ModelState(model, reinterpret_cast<void**>(&state));
        if (err != nullptr) {
          EXPECT_TRUE(false)
              << "Expect successful TRITONREPOAGENT_ModelState(): "
              << TRITONSERVER_ErrorMessage(err);
          TRITONSERVER_ErrorDelete(err);
        } else {
          EXPECT_TRUE(state == nullptr) << "Expect state is not set";
        }
        state = new size_t(0);
        err = TRITONREPOAGENT_ModelSetState(
            model, reinterpret_cast<void*>(state));
        if (err != nullptr) {
          EXPECT_TRUE(false)
              << "Expect successful TRITONREPOAGENT_ModelSetState(): "
              << TRITONSERVER_ErrorMessage(err);
          TRITONSERVER_ErrorDelete(err);
          delete state;
        }
      };
  model_fini_fn_ =
      [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model) {
        size_t* state = nullptr;
        auto err =
            TRITONREPOAGENT_ModelState(model, reinterpret_cast<void**>(&state));
        if (err != nullptr) {
          EXPECT_TRUE(false)
              << "Expect successful TRITONREPOAGENT_ModelState(): "
              << TRITONSERVER_ErrorMessage(err);
          TRITONSERVER_ErrorDelete(err);
        } else {
          EXPECT_TRUE(state != nullptr) << "Expect state is set";
          EXPECT_EQ(*state, size_t(0));
        }

        // Sanity check that set state works elsewhere
        size_t* new_state = new size_t(*state);
        delete state;
        err = TRITONREPOAGENT_ModelSetState(
            model, reinterpret_cast<void*>(new_state));
        if (err != nullptr) {
          EXPECT_TRUE(false)
              << "Expect successful TRITONREPOAGENT_ModelSetState(): "
              << TRITONSERVER_ErrorMessage(err);
          TRITONSERVER_ErrorDelete(err);
        }

        // Delete state before end of model lifecycle
        delete new_state;
      };
  // Overriding the model action function in agent handle because the action
  // type needs to be checked here
  ni::TritonRepoAgent::TritonRepoAgentModelActionFn_t ModelActionFn =
      [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
         const TRITONREPOAGENT_ActionType action_type) -> TRITONSERVER_Error* {
    size_t* state = nullptr;
    auto err =
        TRITONREPOAGENT_ModelState(model, reinterpret_cast<void**>(&state));
    if (err != nullptr) {
      EXPECT_TRUE(false) << "Expect successful TRITONREPOAGENT_ModelState(): "
                         << TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
    }
    EXPECT_TRUE(state != nullptr) << "Expect state is set";
    switch (action_type) {
      case TRITONREPOAGENT_ACTION_LOAD: {
        EXPECT_EQ(*state, size_t(0));
        ++*state;
        break;
      }
      case TRITONREPOAGENT_ACTION_LOAD_COMPLETE: {
        EXPECT_EQ(*state, size_t(1));
        ++*state;
        break;
      }
      case TRITONREPOAGENT_ACTION_LOAD_FAIL: {
        EXPECT_EQ(*state, size_t(1));
        --*state;
        break;
      }
      case TRITONREPOAGENT_ACTION_UNLOAD: {
        EXPECT_EQ(*state, size_t(2));
        --*state;
        break;
      }
      case TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE: {
        EXPECT_EQ(*state, size_t(1));
        --*state;
        break;
      }
    }

    // Sanity check that set state works elsewhere
    size_t* new_state = new size_t(*state);
    delete state;
    err = TRITONREPOAGENT_ModelSetState(
        model, reinterpret_cast<void*>(new_state));
    if (err != nullptr) {
      EXPECT_TRUE(false)
          << "Expect successful TRITONREPOAGENT_ModelSetState(): "
          << TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
      delete new_state;
    }
    return nullptr;
  };
  global_mock_agents["agent_path"].AddEntryPoint(
      "TRITONREPOAGENT_ModelAction", reinterpret_cast<void*>(ModelActionFn));


  const auto lifecycles = valid_lifecycles_;
  for (const auto& lifecycle : lifecycles) {
    // Create agent to be associated with the model
    std::shared_ptr<ni::TritonRepoAgent> agent;
    auto status = ni::TritonRepoAgent::Create("agent", "agent_path", &agent);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful agent creation: " << status.AsString();
    std::unique_ptr<ni::TritonRepoAgentModel> model;
    status = ni::TritonRepoAgentModel::Create(
        original_type_, original_location_, simple_config_, agent,
        ni::TritonRepoAgent::Parameters(), &model);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful model creation: " << status.AsString();
    for (const auto action : lifecycle) {
      status = model->InvokeAgent(action);
      ASSERT_TRUE(status.IsOk()) << "Expect successful agent invocation with "
                                 << ni::TRITONREPOAGENT_ActionTypeString(action)
                                 << ": " << status.AsString();
    }
  }
}

TEST_F(TritonRepoAgentAPITest, TRITONREPOAGENT_AgentState)
{
  // Two models share one agent, check if agent state is properly shared
  agent_init_fn_ = [](TRITONREPOAGENT_Agent* agent) {
    size_t* state = nullptr;
    auto err = TRITONREPOAGENT_State(agent, reinterpret_cast<void**>(&state));
    if (err != nullptr) {
      EXPECT_TRUE(false) << "Expect successful TRITONREPOAGENT_State(): "
                         << TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
    } else {
      EXPECT_TRUE(state == nullptr) << "Expect state is not set";
    }
    state = new size_t(0);
    err = TRITONREPOAGENT_SetState(agent, reinterpret_cast<void*>(state));
    if (err != nullptr) {
      EXPECT_TRUE(false) << "Expect successful TRITONREPOAGENT_SetState(): "
                         << TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
      delete state;
    }
  };
  agent_fini_fn_ = [](TRITONREPOAGENT_Agent* agent) {
    size_t* state = nullptr;
    auto err = TRITONREPOAGENT_State(agent, reinterpret_cast<void**>(&state));
    if (err != nullptr) {
      EXPECT_TRUE(false) << "Expect successful TRITONREPOAGENT_State(): "
                         << TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
    } else {
      EXPECT_TRUE(state != nullptr) << "Expect state is set";
      EXPECT_EQ(*state, size_t(0));
    }

    // Sanity check that set state works elsewhere
    size_t* new_state = new size_t(*state);
    delete state;
    err = TRITONREPOAGENT_SetState(agent, reinterpret_cast<void*>(new_state));
    if (err != nullptr) {
      EXPECT_TRUE(false) << "Expect successful TRITONREPOAGENT_SetState(): "
                         << TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
    }

    // Delete state before end of agent lifecycle
    delete new_state;
  };
  model_init_fn_ =
      [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model) {
        size_t* state = nullptr;
        auto err =
            TRITONREPOAGENT_State(agent, reinterpret_cast<void**>(&state));
        if (err != nullptr) {
          EXPECT_TRUE(false) << "Expect successful TRITONREPOAGENT_State(): "
                             << TRITONSERVER_ErrorMessage(err);
          TRITONSERVER_ErrorDelete(err);
        } else {
          EXPECT_TRUE(state != nullptr) << "Expect state is set";
        }

        // Agent state maybe 0 or 1 depending on the order of model lifecycle,
        // record that in model state to keep track of the order
        if ((*state == 0) || (*state == 1)) {
          size_t* model_state = new size_t(*state);
          err = TRITONREPOAGENT_ModelSetState(
              model, reinterpret_cast<void*>(model_state));
          if (err != nullptr) {
            EXPECT_TRUE(false)
                << "Expect successful TRITONREPOAGENT_ModelSetState(): "
                << TRITONSERVER_ErrorMessage(err);
            TRITONSERVER_ErrorDelete(err);
          }
        } else {
          EXPECT_TRUE(false) << "Expect agent state is either 0 or 1";
        }

        // Sanity check that set state works elsewhere
        ++*state;
        size_t* new_state = new size_t(*state);
        delete state;
        err =
            TRITONREPOAGENT_SetState(agent, reinterpret_cast<void*>(new_state));
        if (err != nullptr) {
          EXPECT_TRUE(false) << "Expect successful TRITONREPOAGENT_SetState(): "
                             << TRITONSERVER_ErrorMessage(err);
          TRITONSERVER_ErrorDelete(err);
          delete new_state;
        }
      };
  model_fini_fn_ = [](TRITONREPOAGENT_Agent* agent,
                      TRITONREPOAGENT_AgentModel* model) {
    size_t* model_state = nullptr;
    auto err = TRITONREPOAGENT_ModelState(
        model, reinterpret_cast<void**>(&model_state));
    if (err != nullptr) {
      EXPECT_TRUE(false) << "Expect successful TRITONREPOAGENT_ModelState(): "
                         << TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
    } else {
      EXPECT_TRUE(model_state != nullptr) << "Expect state is set";
    }

    size_t* state = nullptr;
    err = TRITONREPOAGENT_State(agent, reinterpret_cast<void**>(&state));
    if (err != nullptr) {
      EXPECT_TRUE(false) << "Expect successful TRITONREPOAGENT_State(): "
                         << TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
    } else {
      EXPECT_TRUE(state != nullptr) << "Expect state is set";
      EXPECT_EQ(*state, size_t(2) - *model_state);
    }

    // Sanity check that set state works elsewhere
    --*state;
    size_t* new_state = new size_t(*state);
    delete state;
    err = TRITONREPOAGENT_SetState(agent, reinterpret_cast<void*>(new_state));
    if (err != nullptr) {
      EXPECT_TRUE(false) << "Expect successful TRITONREPOAGENT_SetState(): "
                         << TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
      delete new_state;
    }

    // Delete state before end of model lifecycle
    delete model_state;
  };
  // Overriding the model action function in agent handle because the action
  // type needs to be checked here
  ni::TritonRepoAgent::TritonRepoAgentModelActionFn_t ModelActionFn =
      [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
         const TRITONREPOAGENT_ActionType action_type) -> TRITONSERVER_Error* {
    size_t* model_state = nullptr;
    auto err = TRITONREPOAGENT_ModelState(
        model, reinterpret_cast<void**>(&model_state));
    if (err != nullptr) {
      EXPECT_TRUE(false) << "Expect successful TRITONREPOAGENT_ModelState(): "
                         << TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
    } else {
      EXPECT_TRUE(model_state != nullptr) << "Expect state is set";
    }

    size_t* state = nullptr;
    err = TRITONREPOAGENT_State(agent, reinterpret_cast<void**>(&state));
    if (err != nullptr) {
      EXPECT_TRUE(false) << "Expect successful TRITONREPOAGENT_State(): "
                         << TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
    }
    EXPECT_TRUE(state != nullptr) << "Expect state is set";
    switch (action_type) {
      case TRITONREPOAGENT_ACTION_LOAD: {
        EXPECT_EQ(*state, size_t(2) + *model_state);
        ++*state;
        break;
      }
      case TRITONREPOAGENT_ACTION_LOAD_COMPLETE: {
        EXPECT_EQ(*state, size_t(4) + *model_state);
        ++*state;
        break;
      }
      case TRITONREPOAGENT_ACTION_LOAD_FAIL: {
        EXPECT_EQ(*state, size_t(4) - *model_state);
        --*state;
        break;
      }
      case TRITONREPOAGENT_ACTION_UNLOAD: {
        EXPECT_EQ(*state, size_t(6) - *model_state);
        --*state;
        break;
      }
      case TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE: {
        EXPECT_EQ(*state, size_t(4) - *model_state);
        --*state;
        break;
      }
    }

    // Sanity check that set state works elsewhere
    size_t* new_state = new size_t(*state);
    delete state;
    err = TRITONREPOAGENT_SetState(agent, reinterpret_cast<void*>(new_state));
    if (err != nullptr) {
      EXPECT_TRUE(false) << "Expect successful TRITONREPOAGENT_SetState(): "
                         << TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
      delete new_state;
    }
    return nullptr;
  };
  global_mock_agents["agent_path"].AddEntryPoint(
      "TRITONREPOAGENT_ModelAction", reinterpret_cast<void*>(ModelActionFn));


  const auto lifecycles = valid_lifecycles_;
  for (const auto& lifecycle : lifecycles) {
    // Create agent to be associated with the model
    std::shared_ptr<ni::TritonRepoAgent> agent;
    auto status = ni::TritonRepoAgent::Create("agent", "agent_path", &agent);
    ASSERT_TRUE(status.IsOk())
        << "Expect successful agent creation: " << status.AsString();
    std::vector<std::unique_ptr<ni::TritonRepoAgentModel>> models(2);
    for (auto& model : models) {
      status = ni::TritonRepoAgentModel::Create(
          original_type_, original_location_, simple_config_, agent,
          ni::TritonRepoAgent::Parameters(), &model);
      ASSERT_TRUE(status.IsOk())
          << "Expect successful model creation: " << status.AsString();
    }
    for (const auto action : lifecycle) {
      for (auto& model : models) {
        status = model->InvokeAgent(action);
        ASSERT_TRUE(status.IsOk())
            << "Expect successful agent invocation with "
            << ni::TRITONREPOAGENT_ActionTypeString(action) << ": "
            << status.AsString();
      }
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
