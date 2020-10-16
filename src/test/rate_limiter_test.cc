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
#include "gtest/gtest.h"

#include "src/core/model_config.pb.h"
#include "src/core/rate_limiter.h"

namespace ni = nvidia::inferenceserver;

namespace {
class RateLimiterTest : public ::testing::Test {
 protected:
  using Resources = std::map<std::string, uint32_t>;

  void SetUp() override
  {
    ni::RateLimiter::Create(false /* ignore_resources_and_priority */, &rate_limiter_);
  }

  void AddInstanceGroup(
      inference::ModelConfig* config,
      const std::vector<int>& gpu_ids = std::vector<int>(),
      const int instance_count = 1, const uint32_t priority = 1,
      const Resources& global_resources = Resources(),
      const Resources& local_resources = Resources())
  {
    auto group = config->add_instance_group();
    group->set_count(instance_count);
    if (gpu_ids.empty()) {
      group->set_kind(inference::ModelInstanceGroup::KIND_CPU);
    } else {
      group->set_kind(inference::ModelInstanceGroup::KIND_GPU);
      for (auto gpu_id : gpu_ids) {
        group->add_gpus(gpu_id);
      }
    }
    group->mutable_rate_limiter()->set_priority(priority);

    for (const auto& resource : global_resources) {
      auto rate_limiter_config = group->mutable_rate_limiter()->add_resources();
      rate_limiter_config->set_name(resource.first);
      rate_limiter_config->set_global(true);
      rate_limiter_config->set_count(resource.second);
    }

    for (const auto& resource : local_resources) {
      auto rate_limiter_config = group->mutable_rate_limiter()->add_resources();
      rate_limiter_config->set_name(resource.first);
      rate_limiter_config->set_count(resource.second);
    }
  }

  void TearDown() override {}

  std::unique_ptr<ni::RateLimiter> rate_limiter_;
};

TEST_F(RateLimiterTest, SingleInstanceSingleInfer)
{
  // A simple test with a single instance
  std::string model_name("test_model");
  int64_t version(1);

  inference::ModelConfig test_config;
  // Create a model configuration a single instance
  AddInstanceGroup(&test_config);
  rate_limiter_->LoadModel(model_name, version, test_config);

  std::atomic<int> callback_count(0);

  auto callback_fn =
      [&callback_count](ni::RateLimiter::ModelInstance* instance) {
        callback_count++;
        // Releasing the instance right away
        instance->Release();
      };

  rate_limiter_->EnqueueModelRequest(callback_fn, model_name, version);

  EXPECT_EQ(1, callback_count)
      << "Expect callback_count: " << 1 << ", got: " << callback_count;
}

TEST_F(RateLimiterTest, SingleInstanceMultiInfer)
{
  // A simple test with a single instance
  std::string model_name("test_model");
  int64_t version(1);

  inference::ModelConfig test_config;
  // Create a model configuration a single instance
  AddInstanceGroup(&test_config);
  rate_limiter_->LoadModel(model_name, version, test_config);

  std::queue<ni::RateLimiter::ModelInstance*> instance_queue;
  std::mutex mtx;
  std::atomic<int32_t> callback_count(0);

  auto callback_fn = [&instance_queue, &mtx, &callback_count](
                         ni::RateLimiter::ModelInstance* instance) {
    callback_count++;
    {
      std::lock_guard<std::mutex> lk(mtx);
      instance_queue.push(instance);
    }
  };

  int request_count = 10;
  // Enqueue all the requests
  for (int i = 0; i < request_count; i++) {
    rate_limiter_->EnqueueModelRequest(callback_fn, model_name, version);
  }

  // As there is only a single instance only one callback should be invoked.
  // Other queued callbacks will be executed as and when they are released.
  for (int i = 0; i < request_count; i++) {
    EXPECT_EQ(i + 1, callback_count)
        << "Expect callback_count: " << i + 1 << ", got: " << callback_count;
    auto instance = instance_queue.front();
    instance_queue.pop();
    instance->Release();
  }
}

TEST_F(RateLimiterTest, MultiInstanceMultiInfer)
{
  // A simple test with a single instance
  std::string model_name("test_model");
  int64_t version(1);

  inference::ModelConfig test_config;
  // Create a model configuration with two instances on different
  // gpu devices.
  AddInstanceGroup(&test_config, std::vector<int>{1, 2});
  rate_limiter_->LoadModel(model_name, version, test_config);

  std::queue<ni::RateLimiter::ModelInstance*> instance_queue;
  std::mutex mtx;
  std::atomic<int32_t> callback_count(0);

  auto callback_fn = [&instance_queue, &mtx, &callback_count](
                         ni::RateLimiter::ModelInstance* instance) {
    callback_count++;
    {
      std::lock_guard<std::mutex> lk(mtx);
      instance_queue.push(instance);
    }
  };

  int request_count = 10;
  // Enqueue all the requests
  for (int i = 0; i < request_count; i++) {
    rate_limiter_->EnqueueModelRequest(callback_fn, model_name, version);
  }

  // As there are two instances offset will be 2
  int offset = 2;
  for (int i = 0; i < (request_count - offset + 1); i++) {
    EXPECT_EQ(i + offset, callback_count)
        << "Expect callback_count: " << i + offset
        << ", got: " << callback_count;
    auto instance = instance_queue.front();
    instance_queue.pop();
    instance->Release();
  }

  // Release any other instances that might be remaining
  while (!instance_queue.empty()) {
    auto instance = instance_queue.front();
    instance_queue.pop();
    instance->Release();
  }
}

TEST_F(RateLimiterTest, SpecificInstanceMultiInfer)
{
  // A simple test with a single instance
  std::string model_name("test_model");
  int64_t version(1);

  inference::ModelConfig test_config;
  // Create a model configuration with two instances on different
  // gpu devices.
  AddInstanceGroup(&test_config, std::vector<int>{1, 2});
  rate_limiter_->LoadModel(model_name, version, test_config);

  std::queue<ni::RateLimiter::ModelInstance*> instance_queue;
  std::mutex mtx;
  std::atomic<int32_t> callback_count(0);

  auto callback_fn = [&instance_queue, &mtx, &callback_count](
                         ni::RateLimiter::ModelInstance* instance) {
    callback_count++;
    {
      std::lock_guard<std::mutex> lk(mtx);
      instance_queue.push(instance);
    }
  };

  int request_count = 10;
  // Enqueue all the requests
  for (int i = 0; i < request_count; i++) {
    rate_limiter_->EnqueueModelRequest(
        callback_fn, model_name, version, 0 /* instance index */);
  }

  // As there are two instances but requests are generated for a single instance
  int offset = 1;
  for (int i = 0; i < (request_count - offset + 1); i++) {
    EXPECT_EQ(i + offset, callback_count)
        << "Expect callback_count: " << i + offset
        << ", got: " << callback_count;
    auto instance = instance_queue.front();
    instance_queue.pop();
    instance->Release();
  }

  // Release any other instances that might be remaining
  while (!instance_queue.empty()) {
    auto instance = instance_queue.front();
    instance_queue.pop();
    instance->Release();
  }
}

TEST_F(RateLimiterTest, SimplePriority)
{
  // A simple test with a single instance
  std::string model_name("test_model");
  int64_t version(1);

  inference::ModelConfig test_config;
  // Create a model configuration with two instances with different priority
  AddInstanceGroup(&test_config, std::vector<int>(), 1, 1);
  AddInstanceGroup(&test_config, std::vector<int>(), 1, 2);
  rate_limiter_->LoadModel(model_name, version, test_config);

  std::vector<int32_t> callback_counts{0, 0};

  auto callback_fn =
      [&callback_counts](ni::RateLimiter::ModelInstance* instance) {
        callback_counts[instance->Index()]++;
        // Releasing the instance right away
        instance->Release();
      };

  int request_count = 12;
  // Enqueue all the requests
  for (int i = 0; i < request_count; i++) {
    rate_limiter_->EnqueueModelRequest(callback_fn, model_name, version);
  }

  EXPECT_EQ(callback_counts[0], 8)
      << "Expect callback_count: " << 8 << ", got: " << callback_counts[0];

  EXPECT_EQ(callback_counts[1], 4)
      << "Expect callback_count: " << 4 << ", got: " << callback_counts[1];
}

TEST_F(RateLimiterTest, SimpleResource)
{
  // A simple test with a single instance
  std::string model_name("test_model");
  int64_t version(1);

  inference::ModelConfig test_config;

  Resources global_resources;
  global_resources["dummy_resource"] = 10;

  AddInstanceGroup(
      &test_config, std::vector<int>{1, 2}, 1, 1, global_resources);
  rate_limiter_->LoadModel(model_name, version, test_config);

  std::queue<ni::RateLimiter::ModelInstance*> instance_queue;
  std::mutex mtx;
  std::atomic<int32_t> callback_count(0);

  auto callback_fn = [&instance_queue, &mtx, &callback_count](
                         ni::RateLimiter::ModelInstance* instance) {
    callback_count++;
    {
      std::lock_guard<std::mutex> lk(mtx);
      instance_queue.push(instance);
    }
  };

  int request_count = 10;
  // Enqueue all the requests
  for (int i = 0; i < request_count; i++) {
    rate_limiter_->EnqueueModelRequest(callback_fn, model_name, version);
  }

  // Although there are two instances, but because of the resource
  // constraint only one instance can run at a time.
  int offset = 1;
  for (int i = 0; i < (request_count - offset + 1); i++) {
    EXPECT_EQ(i + offset, callback_count)
        << "Expect callback_count: " << i + offset
        << ", got: " << callback_count;
    auto instance = instance_queue.front();
    instance_queue.pop();
    instance->Release();
  }

  // Release any other instances that might be remaining
  while (!instance_queue.empty()) {
    auto instance = instance_queue.front();
    instance_queue.pop();
    instance->Release();
  }
}

TEST_F(RateLimiterTest, NoLimiting)
{
  // A simple test with a single instance
  std::string model_name("test_model");
  int64_t version(1);

  inference::ModelConfig test_config;

  Resources global_resources;
  global_resources["dummy_resource"] = 10;

  AddInstanceGroup(
      &test_config, std::vector<int>{1, 2}, 1, 1, global_resources);

  std::unique_ptr<ni::RateLimiter> rate_limiter;
  ni::RateLimiter::Create(true /* ignore_resources_and_priority */, &rate_limiter);

  rate_limiter->LoadModel(model_name, version, test_config);

  std::queue<ni::RateLimiter::ModelInstance*> instance_queue;
  std::mutex mtx;
  std::atomic<int32_t> callback_count(0);

  auto callback_fn = [&instance_queue, &mtx, &callback_count](
                         ni::RateLimiter::ModelInstance* instance) {
    callback_count++;
    {
      std::lock_guard<std::mutex> lk(mtx);
      instance_queue.push(instance);
    }
  };

  int request_count = 10;
  // Enqueue all the requests
  for (int i = 0; i < request_count; i++) {
    rate_limiter->EnqueueModelRequest(callback_fn, model_name, version);
  }

  // Even though there is a resource constraint, as the rate limiting is
  // disabled we must see execution across all the instances.
  int offset = 2;
  for (int i = 0; i < (request_count - offset + 1); i++) {
    EXPECT_EQ(i + offset, callback_count)
        << "Expect callback_count: " << i + offset
        << ", got: " << callback_count;
    auto instance = instance_queue.front();
    instance_queue.pop();
    instance->Release();
  }

  // Release any other instances that might be remaining
  while (!instance_queue.empty()) {
    auto instance = instance_queue.front();
    instance_queue.pop();
    instance->Release();
  }
}


}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
