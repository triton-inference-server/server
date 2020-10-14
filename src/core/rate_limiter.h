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
#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <vector>

#include "src/core/model_config.pb.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

// Limits the rate at which requests are dispatched from the scheduler
class RateLimiter {
 public:
  class ModelInstance;

 private:
  class ModelContext;

  using StandardReleaseFunc = std::function<void(ModelInstance*)>;
  using StandardScheduleFunc = std::function<void(ModelInstance*)>;
  using StandardStageFunc = std::function<void(ModelInstance*)>;
  using Specification = inference::ModelInstanceGroup::RateLimiterSpec;

 public:
  /// Creates a rate limiter object which will funnel the inference requests to
  /// the model instances.
  /// \param enable_rate_limiting Whether or not to enable rate limiting.
  /// \return Status object indicating success or failure.
  static Status Create(
      const bool enable_rate_limiting,
      std::unique_ptr<RateLimiter>* rate_limiter);

  /// Loads the specified model to the RateLimiter
  /// \param enable_rate_limiting Whether or not to enable rate limiting.
  /// \return Status object indicating success or failure.
  Status LoadModel(
      const std::string& model_name, const int64_t version,
      const inference::ModelConfig& model_config);

  /// Unloads the specified model from the RateLimiter
  /// \param enable_rate_limiting Whether or not to enable rate limiting.
  /// \return Status object indicating success or failure.
  Status UnloadModel(const std::string& model_name, const int64_t version);

  /// Enqueues the callback to the specified model. In future, when the
  /// conditions are met, the callback will be invoked and a pointer to
  /// allocated RateLimiter::ModelInstance object will be exposed as a
  /// parameter. The user must ensure RateLimiter::ModelInstance::Release
  /// gets called on the instance once the inference request is complete
  /// so that the instance and its resources are returned to the available
  /// pool. Also, note the callback should be a light-weight call and
  /// must not itself invoke the inference execution.
  /// \param OnSchedule The callback function to be called when scheduling.
  /// \param model_name The name of the model.
  /// \param version The version of the model.
  /// \param instance_index The index to a specific instance of the model.
  /// The default value is -1 which means that an instance with highest
  /// priority will be selected for the execution.
  /// \return Status object indicating success or failure.
  Status EnqueueModelRequest(
      const StandardScheduleFunc& OnSchedule, const std::string& model_name,
      const int64_t version, const int instance_index = -1);

  // Holds the state of the model instance.
  class ModelInstance {
   public:
    friend class RateLimiter;
    friend class ResourceManager;
    enum State { AVAILABLE, STAGED, ALLOCATED, UNLOADED };

    /// Should be called when the request on the model instance is
    /// complete. This function releases the resources allocated to
    /// the model instance and sends the instance into the available
    /// pool so that it can serve other requests.
    void Release();

    /// Returns the index of the instance
    int32_t Index() { return index_; }

   private:
    ModelInstance(
        const std::string& model_name, const int64_t version,
        ModelContext* model_context, const uint32_t index, const int gpu_device,
        const Specification& spec, StandardStageFunc OnStage,
        StandardReleaseFunc OnRelease);

    std::pair<std::string, int64_t> ModelIdentifier();
    int32_t DeviceId() const { return gpu_device_; }
    const Specification* GetSpecification() const { return &spec_; }
    void MarkAvailable();
    double ScaledPriority();
    Status Stage(StandardScheduleFunc OnSchedule);
    Status Allocate();
    Status DirectAllocate(StandardScheduleFunc OnSchedule);
    void Unload();
    void WaitForUnload();

    std::string model_name_;
    int64_t version_;
    ModelContext* model_context_;
    int32_t index_;
    int gpu_device_;
    Specification spec_;
    StandardStageFunc OnStage_;
    StandardReleaseFunc OnRelease_;
    std::atomic<uint64_t> exec_count_;

    State state_;
    bool unloading_;
    std::mutex state_mtx_;

    StandardScheduleFunc OnSchedule_;

    std::condition_variable cv_;
  };

 private:
  RateLimiter(const bool enable_rate_limiting);

  void LoadModelHelper(
      const std::string& model_name, const int64_t version, const int device_id,
      const Specification& rate_limit_spec, ModelContext* model_context,
      std::vector<std::shared_ptr<ModelInstance>>* model_instances);

  void OnStage(ModelInstance* instance_ptr);
  void OnRelease(ModelInstance* instance_ptr);
  void AttemptAllocation();

  class Comparator {
   public:
    bool operator()(ModelInstance* a, ModelInstance* b)
    {
      return a->ScaledPriority() > b->ScaledPriority();
    }
  };

  using PriorityQueue = std::priority_queue<
      ModelInstance*, std::vector<ModelInstance*>, Comparator>;

  // Holds the active context to a loaded model
  class ModelContext {
   public:
    ModelContext();

    Status EnqueueModelRequest(
        const StandardScheduleFunc& OnSchedule, const int instance_index);
    void AddAvailableInstance(ModelInstance* instance);
    void StageInstanceIfAvailable();
    void AllocateInstanceIfAvailable();
    void SetSpecificQueueCount(int queue_count);
    bool ContainsPendingRequests(int32_t index);
    void Unload();
    bool isUnloading() { return unloading_; }

   private:
    bool unloading_;

    // Queue holding pending scheduling request
    std::queue<StandardScheduleFunc> generic_request_queue_;
    std::vector<std::queue<StandardScheduleFunc>> specific_request_queues_;
    std::recursive_mutex request_queue_mtx_;

    // The set of instances that are available at the moment
    PriorityQueue avbl_instances_;
    std::recursive_mutex avbl_instances_mtx_;
  };

  // Manages and keep track of resource allocation to the model instances.
  class ResourceManager {
   public:
    // GPU device number that indicates that no gpu is available for a
    // context
    static constexpr int NO_GPU_DEVICE = -1;
    // Key for holding global resources
    static constexpr int GLOBAL_RESOURCE_KEY = -2;

    static Status Create(std::unique_ptr<ResourceManager>* resource_manager);
    void LoadModelInstance(const RateLimiter::ModelInstance* instance);
    Status UnloadModelInstance(const RateLimiter::ModelInstance* instance);
    void UpdateResourceLimits();
    bool AllocateResources(const RateLimiter::ModelInstance* instance);
    Status ReleaseResources(const RateLimiter::ModelInstance* instance);

   private:
    ResourceManager();

    using ResourceMap = std::map<int, std::map<std::string, uint32_t>>;

    std::map<const RateLimiter::ModelInstance*, ResourceMap> model_resources_;
    std::mutex model_resources_mtx_;

    ResourceMap max_resources_;
    std::mutex max_resources_mtx_;

    ResourceMap allocated_resources_;
    std::mutex allocated_resources_mtx_;
  };

  // Instances for the loaded models
  std::map<
      std::pair<std::string, int64_t>,
      std::vector<std::shared_ptr<ModelInstance>>>
      model_instances_;
  std::mutex model_instances_mtx_;

  // Running context of the loaded models
  std::map<std::pair<std::string, int64_t>, ModelContext> model_contexts_;
  std::mutex model_contexts_mtx_;

  // Holds the model instances that have been staged
  PriorityQueue staged_instances_;
  std::recursive_mutex staged_instances_mtx_;

  // Manager to keep track of the resource allocations
  std::unique_ptr<ResourceManager> resource_manager_;

  bool enable_rate_limiting_;
};

}}  // namespace nvidia::inferenceserver