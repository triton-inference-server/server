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

#include <map>
#include <memory>
#include "src/core/triton_repo_agent.h"

namespace ni = nvidia::inferenceserver;

namespace {

class MockSharedLibraryHandle {
 public:
  bool AddEntryPoint(const std::string& name, void* fn) {
    return entry_points_.emplace(name, fn).second;
  }

  bool GetEntryPoint(const std::string& name, void** fn) {
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

}

namespace nvidia::inferenceserver {

Status OpenLibraryHandle(const std::string& path, void** handle)
{
  auto it = global_mock_agents.find(path);
  if (it != global_mock_agents.end()) {
    *handle = reinterpret_cast<void*>(&it->second);
    return Status::Success;
  }
  return Status(
      Status::Code::NOT_FOUND,
      "unable to load shared library: mock shared library is not set for path " + path);
}

Status CloseLibraryHandle(void* handle)
{
  for (const auto& global_mock_agent : global_mock_agents) {
    if (reinterpret_cast<void*>(&global_mock_agent.second) == handle) {
      return Status::Success;
    }
  }
  return Status(
      Status::Code::NOT_FOUND,
      "unable to unload shared library: handle does not matach any mock shared library");
}

Status GetEntrypoint(
    void* handle, const std::string& name, const bool optional, void** fn)
{
  auto mock_agent = reinterpret_cast<MockSharedLibraryHandle*>(handle);
  bool found = mock_agent->GetEntryPoint(name, fn);
  if (!optional && !found) {
    return Status(
        Status::Code::NOT_FOUND, "unable to find required entrypoint '" + name +
                                     "' in shared library");
  }
  return Status::Success;
}

}

namespace {

class TritonRepoAgentTest : public ::testing::Test {
 protected:
  void TearDown() override { global_mock_agents.clear() }
};

TEST_F(TritonRepoAgentTest, MinimalAgent)
{
  auto CheckNameModelActionFn = [](TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
      const TRITONREPOAGENT_ActionType action_type) {
        auto lagent = reinterpret_cast<TritonRepoAgent*>(agent);
        EXPECT_EQ(lagent->Name(), "minimal_agent");
        return nullptr;
  };
  auto minimal_agent_handle = MockSharedLibraryHandle();
  minimal_agent.AddEntryPoint("TRITONREPOAGENT_ModelAction", CheckNameModelActionFn)
  global_mock_agents.emplace("minimal_agent_path", minimal_agent_handle);
  std::shared_ptr<TritonRepoAgent> minimal_agent;
  ASSERT_TRUE(TritonRepoAgent::Create("minimal_agent", "minimal_agent_path", &minimal_agent).IsOk());
  ASSERT_TRUE(minimal_agent->AgentModelActionFn()(
    reinterpret_cast<TRITONREPOAGENT_Agent*>(minimal_agent.get()),
    nullptr, TRITONREPOAGENT_ACTION_LOAD));
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
