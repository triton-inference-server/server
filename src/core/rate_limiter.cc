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

#include "src/core/rate_limiter.h"

#include "src/backends/backend/triton_model.h"
#include "src/backends/backend/triton_model_instance.h"

namespace nvidia { namespace inferenceserver {

//=========================================================================
//  Core Implementation
//=========================================================================

Status
RateLimiter::Create(
    const bool ignore_resources_and_priority,
    const RateLimiter::ResourceMap& resource_map,
    std::unique_ptr<RateLimiter>* rate_limiter)
{
  std::unique_ptr<RateLimiter> local_rate_limiter(
      new RateLimiter(ignore_resources_and_priority, resource_map));
  *rate_limiter = std::move(local_rate_limiter);

  return Status::Success;
}

Status
RateLimiter::RegisterModelInstance(
    const TritonModelInstance* triton_model_instance,
    const RateLimiterConfig& rate_limiter_config)
{
  {
    std::lock_guard<std::mutex> lk1(model_contexts_mtx_);
    std::lock_guard<std::mutex> lk2(model_instances_mtx_);


    auto& model_context = model_contexts_[triton_model_instance->Model()];
    auto& model_instances = model_instances_[triton_model_instance->Model()];

    model_instances.push_back(std::shared_ptr<ModelInstance>(new ModelInstance(
        triton_model_instance, &model_context, rate_limiter_config,
        [this](ModelInstance* instance) { OnStage(instance); },
        [this](ModelInstance* instance) { OnRelease(instance); })));
    model_context.AddAvailableInstance(model_instances.back().get());
    resource_manager_->AddModelInstance(model_instances.back().get());
    model_context.AddSpecificRequestQueue();


    resource_manager_->UpdateResourceLimits();
  }

  return Status::Success;
}


Status
RateLimiter::UnregisterModel(const TritonModel* model)
{
  {
    std::lock_guard<std::mutex> lk1(model_contexts_mtx_);
    std::lock_guard<std::mutex> lk2(model_instances_mtx_);

    auto& model_context = model_contexts_[model];

    model_context.RequestRemoval();
    for (const auto& instance : model_instances_[model]) {
      instance->WaitForRemoval();
      resource_manager_->RemoveModelInstance(instance.get());
    }

    model_instances_.erase(model);
    model_contexts_.erase(model);
  }

  resource_manager_->UpdateResourceLimits();

  return Status::Success;
}

Status
RateLimiter::RequestModelInstance(
    const StandardScheduleFunc& OnSchedule, const TritonModel* model,
    const TritonModelInstance* triton_model_instance)
{
  std::lock_guard<std::mutex> lk(model_contexts_mtx_);

  auto itr = model_contexts_.find(model);
  if (itr == model_contexts_.end()) {
    return Status(
        Status::Code::INTERNAL, "Requested model is not yet registered");
  }

  if (itr->second.isRemovalInProgress()) {
    return Status(
        Status::Code::INTERNAL,
        "New model requests can not be made to a model that is being "
        "removed");
  }

  itr->second.EnqueueModelInstanceRequest(OnSchedule, triton_model_instance);
  if (ignore_resources_and_priority_) {
    // Directly allocate an available model instance if not using rate
    // limiter.
    itr->second.AllocateInstanceIfAvailable();
  } else {
    itr->second.StageInstanceIfAvailable();
  }

  return Status::Success;
}

RateLimiter::RateLimiter(
    const bool ignore_resources_and_priority, const ResourceMap& resource_map)
    : ignore_resources_and_priority_(ignore_resources_and_priority)
{
  ResourceManager::Create(resource_map, &resource_manager_);
}

void
RateLimiter::OnStage(ModelInstance* instance)
{
  {
    std::lock_guard<std::recursive_mutex> lk(staged_instances_mtx_);
    staged_instances_.push(instance);
  }
  AttemptAllocation();
}

void
RateLimiter::OnRelease(ModelInstance* instance)
{
  auto& model_context = model_contexts_[instance->RawInstance()->Model()];
  model_context.AddAvailableInstance(instance);
  resource_manager_->ReleaseResources(instance);
  if (model_context.ContainsPendingRequests(instance->RawInstance()->Index())) {
    if (ignore_resources_and_priority_) {
      // Directly allocate an available model instance if not using rate
      // limiter.
      model_context.AllocateInstanceIfAvailable();
    } else {
      model_context.StageInstanceIfAvailable();
    }
  }
  AttemptAllocation();
}

void
RateLimiter::AttemptAllocation()
{
  std::lock_guard<std::recursive_mutex> lk(staged_instances_mtx_);
  if (!staged_instances_.empty()) {
    ModelInstance* instance = staged_instances_.top();
    if (resource_manager_->AllocateResources(instance)) {
      staged_instances_.pop();
      instance->Allocate();
    }
  }
}

//=========================================================================
//  ModelContext Implementation
//=========================================================================

RateLimiter::ModelContext::ModelContext() : removal_in_progress_(false) {}

Status
RateLimiter::ModelContext::EnqueueModelInstanceRequest(
    const StandardScheduleFunc& OnSchedule,
    const TritonModelInstance* triton_model_instance)
{
  std::lock_guard<std::recursive_mutex> lk(request_queue_mtx_);

  if (triton_model_instance == nullptr) {
    generic_request_queue_.push(OnSchedule);
  } else if (
      (uint32_t)triton_model_instance->Index() <
      specific_request_queues_.size()) {
    specific_request_queues_[triton_model_instance->Index()].push(OnSchedule);
  } else {
    return Status(
        Status::Code::INTERNAL,
        "expected instance index between 0 and " +
            std::to_string(specific_request_queues_.size()) + ", got " +
            std::to_string(triton_model_instance->Index()));
  }

  return Status::Success;
}

void
RateLimiter::ModelContext::AddAvailableInstance(ModelInstance* instance)
{
  std::lock_guard<std::recursive_mutex> lk(avbl_instances_mtx_);
  avbl_instances_.push(instance);
  instance->MarkAvailable();
}

void
RateLimiter::ModelContext::StageInstanceIfAvailable()
{
  std::lock_guard<std::recursive_mutex> lk1(request_queue_mtx_);
  std::lock_guard<std::recursive_mutex> lk2(avbl_instances_mtx_);
  PriorityQueue backup_queue;
  while (!avbl_instances_.empty()) {
    ModelInstance* instance = avbl_instances_.top();
    if (!specific_request_queues_[instance->RawInstance()->Index()].empty()) {
      // Prioritize the specific requests for the available model
      // instance highest priority.
      const StandardScheduleFunc func =
          specific_request_queues_[instance->RawInstance()->Index()].front();
      specific_request_queues_[instance->RawInstance()->Index()].pop();
      instance->Stage(func);
    } else if (!generic_request_queue_.empty()) {
      // If request is for generic model instance then use the
      // instance with the highest priority.
      const StandardScheduleFunc func = generic_request_queue_.front();
      generic_request_queue_.pop();
      instance->Stage(func);
    } else {
      // If there are requests for a specific model instance then backup
      // the model instance and keep searching through the available
      // model instances. The prioritization will be taken care of in the
      // staging priority queue.
      backup_queue.push(instance);
    }
    avbl_instances_.pop();
  }
  // Restore the backup queue
  if (!backup_queue.empty()) {
    avbl_instances_.swap(backup_queue);
  }
}

void
RateLimiter::ModelContext::AllocateInstanceIfAvailable()
{
  std::lock_guard<std::recursive_mutex> lk1(request_queue_mtx_);
  std::lock_guard<std::recursive_mutex> lk2(avbl_instances_mtx_);
  PriorityQueue backup_queue;
  while (!avbl_instances_.empty()) {
    ModelInstance* instance = avbl_instances_.top();
    if (!specific_request_queues_[instance->RawInstance()->Index()].empty()) {
      // Prioritize the specific requests for the available model
      // instance highest priority.
      const StandardScheduleFunc func =
          specific_request_queues_[instance->RawInstance()->Index()].front();
      specific_request_queues_[instance->RawInstance()->Index()].pop();
      instance->DirectAllocate(func);
    } else if (!generic_request_queue_.empty()) {
      // If request is for generic model instance then use the
      // instance with the highest priority.
      const StandardScheduleFunc func = generic_request_queue_.front();
      generic_request_queue_.pop();
      instance->DirectAllocate(func);
    } else {
      // If there are requests for a specific model instance then backup
      // the model instance and keep searching through the available
      // model instances. The prioritization will be taken care of in the
      // staging priority queue.
      backup_queue.push(instance);
    }
    avbl_instances_.pop();
  }
  // Restore the backup queue
  if (!backup_queue.empty()) {
    avbl_instances_.swap(backup_queue);
  }
}

void
RateLimiter::ModelContext::AddSpecificRequestQueue()
{
  std::lock_guard<std::recursive_mutex> lk(request_queue_mtx_);
  specific_request_queues_.emplace_back();
}

bool
RateLimiter::ModelContext::ContainsPendingRequests(int index)
{
  std::lock_guard<std::recursive_mutex> lk(request_queue_mtx_);
  return (generic_request_queue_.size() != 0) ||
         (specific_request_queues_[index].size() != 0);
}

void
RateLimiter::ModelContext::RequestRemoval()
{
  removal_in_progress_ = true;
}


//=========================================================================
//  ModelInstance Implementation
//=========================================================================

RateLimiter::ModelInstance::ModelInstance(
    const TritonModelInstance* triton_model_instance,
    RateLimiter::ModelContext* model_context,
    const RateLimiter::RateLimiterConfig& rate_limiter_config,
    RateLimiter::StandardStageFunc OnStage,
    RateLimiter::StandardReleaseFunc OnRelease)
    : triton_model_instance_(triton_model_instance),
      model_context_(model_context), rate_limiter_config_(rate_limiter_config),
      OnStage_(OnStage), OnRelease_(OnRelease), exec_count_(0),
      state_(AVAILABLE)
{
}

void
RateLimiter::ModelInstance::MarkAvailable()
{
  std::lock_guard<std::mutex> lk(state_mtx_);
  state_ = AVAILABLE;
}

Status
RateLimiter::ModelInstance::Stage(StandardScheduleFunc OnSchedule)
{
  {
    std::lock_guard<std::mutex> lk(state_mtx_);

    if (state_ != AVAILABLE) {
      return Status(
          Status::Code::INTERNAL,
          "Can not stage a model instance that is not yet available");
    }

    state_ = STAGED;
    OnSchedule_ = OnSchedule;
  }

  OnStage_(this);

  return Status::Success;
}

Status
RateLimiter::ModelInstance::Allocate()
{
  {
    std::lock_guard<std::mutex> lk(state_mtx_);

    if (state_ != STAGED) {
      return Status(
          Status::Code::INTERNAL,
          "Can not allocate a model instance that is not yet staged");
    }

    state_ = ALLOCATED;
  }

  OnSchedule_(this);

  return Status::Success;
}

Status
RateLimiter::ModelInstance::DirectAllocate(StandardScheduleFunc OnSchedule)
{
  {
    std::lock_guard<std::mutex> lk(state_mtx_);

    if (state_ != AVAILABLE) {
      return Status(
          Status::Code::INTERNAL,
          "Can not allocate a model instance that is not yet available");
    }

    state_ = ALLOCATED;
  }

  OnSchedule(this);

  return Status::Success;
}

void
RateLimiter::ModelInstance::Release(bool executed)
{
  if (executed) {
    exec_count_++;
  }

  OnRelease_(this);

  {
    std::lock_guard<std::mutex> lk(state_mtx_);
    if ((model_context_->isRemovalInProgress()) && (state_ == AVAILABLE) &&
        (!model_context_->ContainsPendingRequests(
            triton_model_instance_->Index()))) {
      state_ = REMOVED;
    }
  }

  cv_.notify_all();
}

void
RateLimiter::ModelInstance::RequestRemoval()
{
  std::lock_guard<std::mutex> lk(state_mtx_);

  if ((state_ == AVAILABLE) && (!model_context_->ContainsPendingRequests(
                                   triton_model_instance_->Index()))) {
    state_ = REMOVED;
  }
}

void
RateLimiter::ModelInstance::WaitForRemoval()
{
  if (!model_context_->isRemovalInProgress()) {
    model_context_->RequestRemoval();
  }

  RequestRemoval();

  // Wait for the instance to be removed
  {
    std::unique_lock<std::mutex> lk(state_mtx_);
    cv_.wait(lk, [this] { return state_ == REMOVED; });
  }
}

double
RateLimiter::ModelInstance::ScaledPriority()
{
  // TODO: Different schemes for the prioritization of
  // model instance can be added here.
  return (exec_count_ * rate_limiter_config_.priority());
}


//=========================================================================
//  ResourceManager Implementation
//=========================================================================

Status
RateLimiter::ResourceManager::Create(
    const ResourceMap& resource_map,
    std::unique_ptr<ResourceManager>* resource_manager)
{
  std::unique_ptr<ResourceManager> local_resource_manager(
      new ResourceManager(resource_map));
  *resource_manager = std::move(local_resource_manager);
  return Status::Success;
}

void
RateLimiter::ResourceManager::AddModelInstance(
    const RateLimiter::ModelInstance* instance)
{
  std::lock_guard<std::mutex> lk(model_resources_mtx_);
  auto pr = model_resources_.emplace(std::make_pair(instance, ResourceMap()));
  for (const auto& resource : instance->GetRateLimiterConfig()->resources()) {
    if (resource.global()) {
      (pr.first->second[GLOBAL_RESOURCE_KEY])[resource.name()] =
          resource.count();
    } else {
      (pr.first->second[instance->RawInstance()->DeviceId()])[resource.name()] =
          resource.count();
    }
  }
}

Status
RateLimiter::ResourceManager::RemoveModelInstance(
    const RateLimiter::ModelInstance* instance)
{
  std::lock_guard<std::mutex> lk(model_resources_mtx_);
  const auto& itr = model_resources_.find(instance);
  if (itr == model_resources_.end()) {
    return Status(
        Status::Code::INTERNAL, "Can not find the instance to remove");
  }
  model_resources_.erase(instance);
  return Status::Success;
}

void
RateLimiter::ResourceManager::UpdateResourceLimits()
{
  std::lock_guard<std::mutex> lk1(max_resources_mtx_);
  std::lock_guard<std::mutex> lk2(model_resources_mtx_);
  max_resources_.clear();
  // Obtain the maximum resource across all the instances
  // and use it as the default available.
  // TODO: Add the enhancement to provide resources via CLI
  // Will save some cycles in obtaineing the resource limits.
  for (const auto& instance_resources : model_resources_) {
    for (const auto& resource_device_map : instance_resources.second) {
      auto ditr = max_resources_.find(resource_device_map.first);
      if (ditr == max_resources_.end()) {
        ditr =
            max_resources_
                .emplace(resource_device_map.first, resource_device_map.second)
                .first;
      } else {
        for (const auto resource : resource_device_map.second) {
          auto ritr = ditr->second.find(resource.first);
          if (ritr == ditr->second.end()) {
            ritr = ditr->second.emplace(resource.first, resource.second).first;
          } else {
            if (ritr->second < resource.second) {
              ritr->second = resource.second;
            }
          }
        }
      }
    }
  }
}

bool
RateLimiter::ResourceManager::AllocateResources(
    const RateLimiter::ModelInstance* instance)
{
  std::lock_guard<std::mutex> lk1(model_resources_mtx_);
  std::lock_guard<std::mutex> lk2(allocated_resources_mtx_);
  const auto& itr = model_resources_.find(instance);
  if (itr == model_resources_.end()) {
    return false;
  } else {
    // First pass to verify if resources are available
    {
      std::lock_guard<std::mutex> lk3(max_resources_mtx_);
      for (const auto& ditr : itr->second) {
        auto allocated_ditr = allocated_resources_.find(ditr.first);
        if (allocated_ditr == allocated_resources_.end()) {
          allocated_ditr =
              allocated_resources_
                  .emplace(ditr.first, std::map<std::string, size_t>())
                  .first;
        }
        for (const auto& ritr : ditr.second) {
          auto allocated_ritr = allocated_ditr->second.find(ritr.first);
          if (allocated_ritr == allocated_ditr->second.end()) {
            allocated_ritr =
                allocated_ditr->second.emplace(ritr.first, 0).first;
          }
          if ((allocated_ritr->second + ritr.second) >
              (max_resources_[ditr.first])[ritr.first]) {
            return false;
          }
        }
      }
    }

    // Second pass to actually allocate the resources
    for (const auto& ditr : itr->second) {
      for (const auto& ritr : ditr.second) {
        (allocated_resources_[ditr.first])[ritr.first] += ritr.second;
      }
    }
  }

  return true;
}

Status
RateLimiter::ResourceManager::ReleaseResources(
    const RateLimiter::ModelInstance* instance)
{
  std::lock_guard<std::mutex> lk1(model_resources_mtx_);
  std::lock_guard<std::mutex> lk2(allocated_resources_mtx_);
  const auto& itr = model_resources_.find(instance);
  if (itr == model_resources_.end()) {
    return Status(
        Status::Code::INTERNAL,
        "Unable find the instance resources to release");
  } else {
    for (const auto& ditr : itr->second) {
      for (const auto& ritr : ditr.second) {
        (allocated_resources_[ditr.first])[ritr.first] -= ritr.second;
      }
    }
  }

  return Status::Success;
}

RateLimiter::ResourceManager::ResourceManager(const ResourceMap& resource_map)
    : explicit_max_resources_(resource_map)
{
}
}}  // namespace nvidia::inferenceserver
