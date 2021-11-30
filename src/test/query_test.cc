// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <unistd.h>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "triton/core/tritonserver.h"

namespace {

#define FAIL_TEST_IF_ERR(X, MSG)                                              \
  do {                                                                        \
    std::shared_ptr<TRITONSERVER_Error> err__((X), TRITONSERVER_ErrorDelete); \
    ASSERT_TRUE((err__ == nullptr))                                           \
        << "error: " << (MSG) << ": "                                         \
        << TRITONSERVER_ErrorCodeString(err__.get()) << " - "                 \
        << TRITONSERVER_ErrorMessage(err__.get());                            \
  } while (false)

using NameMap =
    std::map<std::string, std::tuple<TRITONSERVER_MemoryType, int64_t, size_t>>;
struct QueryTracker {
  QueryTracker(
      const char* tensor_name, size_t* byte_size,
      TRITONSERVER_MemoryType memory_type, int64_t memory_type_id)
      : has_name_(tensor_name != nullptr), has_byte_size_(byte_size != nullptr),
        caller_preferred_type_(memory_type),
        caller_preferred_id_(memory_type_id)
  {
    if (has_name_) {
      name_ = tensor_name;
    }
    if (has_byte_size_) {
      byte_size_ = *byte_size;
    }
  }
  bool has_name_;
  bool has_byte_size_;
  std::string name_;
  size_t byte_size_;
  TRITONSERVER_MemoryType caller_preferred_type_;
  int64_t caller_preferred_id_;
};

TRITONSERVER_Error*
ResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  auto& output_tracker =
      (reinterpret_cast<std::pair<std::vector<QueryTracker>, NameMap>*>(userp)
           ->second);
  output_tracker.emplace(
      tensor_name,
      std::make_tuple(
          preferred_memory_type, preferred_memory_type_id, byte_size));
  return nullptr;  // Success
}

TRITONSERVER_Error*
ResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  return nullptr;  // Success
}

void
InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  TRITONSERVER_InferenceRequestDelete(request);
}

void
InferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  if (response != nullptr) {
    // Notify that the completion.
    std::promise<TRITONSERVER_Error*>* p =
        reinterpret_cast<std::promise<TRITONSERVER_Error*>*>(userp);
    p->set_value(TRITONSERVER_InferenceResponseError(response));
  }
  TRITONSERVER_InferenceResponseDelete(response);
}

class QueryTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite()
  {
    // Create the server...
    TRITONSERVER_ServerOptions* server_options = nullptr;
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsNew(&server_options),
        "creating server options");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsSetModelRepositoryPath(
            server_options, "./models"),
        "setting model repository path");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsSetBackendDirectory(
            server_options, "/opt/tritonserver/backends"),
        "setting backend directory");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
            server_options, "/opt/tritonserver/repoagents"),
        "setting repository agent directory");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsSetStrictModelConfig(server_options, true),
        "setting strict model configuration");

    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerNew(&server_, server_options), "creating server");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsDelete(server_options),
        "deleting server options");
  }

  static void TearDownTestSuite()
  {
    FAIL_TEST_IF_ERR(TRITONSERVER_ServerDelete(server_), "deleting server");
  }

  void SetUp() override
  {
    ASSERT_TRUE(server_ != nullptr) << "Server has not created";
    // Wait until the server is both live and ready.
    size_t health_iters = 0;
    while (true) {
      bool live, ready;
      FAIL_TEST_IF_ERR(
          TRITONSERVER_ServerIsLive(server_, &live),
          "unable to get server liveness");
      FAIL_TEST_IF_ERR(
          TRITONSERVER_ServerIsReady(server_, &ready),
          "unable to get server readiness");
      if (live && ready) {
        break;
      }

      if (++health_iters >= 10) {
        FAIL() << "failed to find healthy inference server";
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Create allocator with common callback
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ResponseAllocatorNew(
            &allocator_, ResponseAlloc, ResponseRelease,
            nullptr /* start_fn */),
        "creating response allocator");

    FAIL_TEST_IF_ERR(
        TRITONSERVER_InferenceRequestNew(
            &irequest_, server_, "query", -1 /* model_version */),
        "creating inference request");

    FAIL_TEST_IF_ERR(
        TRITONSERVER_InferenceRequestSetReleaseCallback(
            irequest_, InferRequestComplete,
            nullptr /* request_release_userp */),
        "setting request release callback");

    std::vector<int64_t> shape{1};
    FAIL_TEST_IF_ERR(
        TRITONSERVER_InferenceRequestAddInput(
            irequest_, "INPUT", TRITONSERVER_TYPE_UINT8, shape.data(),
            shape.size()),
        "setting input for the request");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_InferenceRequestAppendInputData(
            irequest_, "INPUT", input_data_.data(), input_data_.size(),
            TRITONSERVER_MEMORY_CPU, 0),
        "assigning INPUT data");

    FAIL_TEST_IF_ERR(
        TRITONSERVER_InferenceRequestSetResponseCallback(
            irequest_, allocator_, reinterpret_cast<void*>(&output_info_),
            InferResponseComplete, reinterpret_cast<void*>(&completed_)),
        "setting response callback");
  }

  void TearDown() override
  {
    unsetenv("TEST_ANONYMOUS");
    unsetenv("TEST_BYTE_SIZE");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ResponseAllocatorDelete(allocator_),
        "deleting response allocator");
  }

  static TRITONSERVER_Server* server_;
  TRITONSERVER_ResponseAllocator* allocator_ = nullptr;
  static std::vector<uint8_t> input_data_;
  TRITONSERVER_InferenceRequest* irequest_ = nullptr;
  std::promise<TRITONSERVER_Error*> completed_;
  std::pair<std::vector<QueryTracker>, NameMap> output_info_;
};

TRITONSERVER_Server* QueryTest::server_ = nullptr;
std::vector<uint8_t> QueryTest::input_data_{1};

TEST_F(QueryTest, DefaultQuery)
{
  TRITONSERVER_ResponseAllocatorQueryFn_t query_fn =
      [](TRITONSERVER_ResponseAllocator* allocator, void* userp,
         const char* tensor_name, size_t* byte_size,
         TRITONSERVER_MemoryType* memory_type,
         int64_t* memory_type_id) -> TRITONSERVER_Error* {
    auto& query_tracker =
        (reinterpret_cast<std::pair<std::vector<QueryTracker>, NameMap>*>(userp)
             ->first);
    query_tracker.emplace_back(
        tensor_name, byte_size, *memory_type, *memory_type_id);
    *memory_type = TRITONSERVER_MEMORY_CPU;
    *memory_type_id = 0;
    return nullptr;
  };
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ResponseAllocatorSetQueryFunction(allocator_, query_fn),
      "setting response callback");

  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerInferAsync(server_, irequest_, nullptr /* trace */),
      "running inference");

  auto err = completed_.get_future().get();
  ASSERT_TRUE(err == nullptr) << "Expect successful inference";

  // Check query tracker to see if the query function is connected properly
  ASSERT_EQ(output_info_.first.size(), size_t(2));
  for (size_t i = 0; i < output_info_.first.size(); ++i) {
    const auto& query_info = output_info_.first[i];
    EXPECT_EQ(query_info.has_name_, true);
    EXPECT_EQ(query_info.name_, (std::string("OUTPUT") + std::to_string(i)));
    EXPECT_EQ(query_info.has_byte_size_, false);
    EXPECT_EQ(
        query_info.caller_preferred_type_, TRITONSERVER_MEMORY_CPU_PINNED);
    EXPECT_EQ(query_info.caller_preferred_id_, 1);
  }

  const auto& output_0 = output_info_.second["OUTPUT0"];
  EXPECT_EQ(std::get<0>(output_0), TRITONSERVER_MEMORY_CPU);
  EXPECT_EQ(std::get<1>(output_0), int64_t(0));
  EXPECT_EQ(std::get<2>(output_0), size_t(2));

  const auto& output_1 = output_info_.second["OUTPUT1"];
  EXPECT_EQ(std::get<0>(output_1), TRITONSERVER_MEMORY_CPU);
  EXPECT_EQ(std::get<1>(output_1), int64_t(0));
  EXPECT_EQ(std::get<2>(output_1), size_t(2));
}

TEST_F(QueryTest, NoQueryFn)
{
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerInferAsync(server_, irequest_, nullptr /* trace */),
      "running inference");

  auto err = completed_.get_future().get();
  ASSERT_TRUE(err != nullptr) << "Expect error";
  EXPECT_EQ(TRITONSERVER_ErrorCode(err), TRITONSERVER_ERROR_UNAVAILABLE);
  EXPECT_STREQ(
      TRITONSERVER_ErrorMessage(err), "Output properties are not available");
}

TEST_F(QueryTest, UnnamedQuery)
{
  setenv("TEST_ANONYMOUS", "", 1);
  setenv("TEST_BYTE_SIZE", "32", 1);
  TRITONSERVER_ResponseAllocatorQueryFn_t query_fn =
      [](TRITONSERVER_ResponseAllocator* allocator, void* userp,
         const char* tensor_name, size_t* byte_size,
         TRITONSERVER_MemoryType* memory_type,
         int64_t* memory_type_id) -> TRITONSERVER_Error* {
    auto& query_tracker =
        (reinterpret_cast<std::pair<std::vector<QueryTracker>, NameMap>*>(userp)
             ->first);
    query_tracker.emplace_back(
        tensor_name, byte_size, *memory_type, *memory_type_id);
    // Slightly different setting
    *memory_type = TRITONSERVER_MEMORY_GPU;
    *memory_type_id = 2;
    return nullptr;
  };
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ResponseAllocatorSetQueryFunction(allocator_, query_fn),
      "setting response callback");

  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerInferAsync(server_, irequest_, nullptr /* trace */),
      "running inference");

  auto err = completed_.get_future().get();
  ASSERT_TRUE(err == nullptr) << "Expect successful inference";

  // Check query tracker to see if the query function is connected properly
  ASSERT_EQ(output_info_.first.size(), size_t(1));
  for (size_t i = 0; i < output_info_.first.size(); ++i) {
    const auto& query_info = output_info_.first[i];
    EXPECT_EQ(query_info.has_name_, false);
    EXPECT_EQ(query_info.has_byte_size_, true);
    EXPECT_EQ(query_info.byte_size_, size_t(32));
    EXPECT_EQ(
        query_info.caller_preferred_type_, TRITONSERVER_MEMORY_CPU_PINNED);
    EXPECT_EQ(query_info.caller_preferred_id_, 1);
  }

  const auto& output_0 = output_info_.second["OUTPUT0"];
  EXPECT_EQ(std::get<0>(output_0), TRITONSERVER_MEMORY_GPU);
  EXPECT_EQ(std::get<1>(output_0), int64_t(2));
  EXPECT_EQ(std::get<2>(output_0), size_t(16));

  const auto& output_1 = output_info_.second["OUTPUT1"];
  EXPECT_EQ(std::get<0>(output_1), TRITONSERVER_MEMORY_GPU);
  EXPECT_EQ(std::get<1>(output_1), int64_t(2));
  EXPECT_EQ(std::get<2>(output_1), size_t(16));
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
