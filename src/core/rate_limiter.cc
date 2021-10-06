// Copyright 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "src/core/logging.h"

namespace nvidia { namespace inferenceserver {

constexpr size_t MAX_PAYLOAD_BUCKET_COUNT = 1000;

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
    TritonModelInstance* triton_model_instance,
    const RateLimiterConfig& rate_limiter_config)
{
  {
    std::lock_guard<std::mutex> lk1(model_ctx_mtx_);
    std::lock_guard<std::mutex> lk2(model_instance_ctx_mtx_);

    auto& model_context = model_contexts_[triton_model_instance->Model()];
    auto& model_instances =
        model_instance_ctxs_[triton_model_instance->Model()];

    model_instances.push_back(
        std::shared_ptr<ModelInstanceContext>(new ModelInstanceContext(
            triton_model_instance, &model_context, rate_limiter_config,
            [this](ModelInstanceContext* instance) { OnStage(instance); },
            [this](ModelInstanceContext* instance) { OnRelease(instance); })));
    model_context.AddAvailableInstance(model_instances.back().get());
    model_context.AddSpecificRequestQueue();

    if (!ignore_resources_and_priority_) {
      resource_manager_->AddModelInstance(model_instances.back().get());
      RETURN_IF_ERROR(resource_manager_->UpdateResourceLimits());
    }
  }

  InitializePayloadQueues(triton_model_instance);

  return Status::Success;
}

Status
RateLimiter::UnregisterModel(const TritonModel* model)
{
  {
    std::lock_guard<std::mutex> lk1(model_ctx_mtx_);
    std::lock_guard<std::mutex> lk2(model_instance_ctx_mtx_);

    auto& model_context = model_contexts_[model];

    model_context.RequestRemoval();
    for (const auto& instance : model_instance_ctxs_[model]) {
      instance->WaitForRemoval();
      if (!ignore_resources_and_priority_) {
        resource_manager_->RemoveModelInstance(instance.get());
      }
    }

    model_instance_ctxs_.erase(model);
    model_contexts_.erase(model);
  }

  if (!ignore_resources_and_priority_) {
    RETURN_IF_ERROR(resource_manager_->UpdateResourceLimits());
  }

  if (payload_queues_.find(model) != payload_queues_.end()) {
    payload_queues_.erase(model);
  }

  return Status::Success;
}

bool
RateLimiter::PayloadSlotAvailable(const TritonModel* model)
{
  bool result;
  PayloadQueue* payload_queue = payload_queues_[model].get();
  {
    std::lock_guard<std::mutex> lk(payload_queue->mu_);
    result = payload_queue->queue_->Size() <
             2 * payload_queue->specific_queues_.size();
  }
  return result;
}

Status
RateLimiter::EnqueuePayload(
    const TritonModel* model, std::shared_ptr<Payload> payload)
{
  auto pinstance = payload->GetInstance();
  if (payload_queues_.find(model) == payload_queues_.end()) {
    LOG_INFO << "Should not print this ";
  }
  PayloadQueue* payload_queue = payload_queues_[model].get();
  {
    std::lock_guard<std::mutex> lk(payload_queue->mu_);
    payload->SetState(Payload::State::REQUESTED);
    if (ignore_resources_and_priority_) {
      SchedulePayload(pinstance, payload_queue, payload);
    }
  }
  if (ignore_resources_and_priority_) {
    if (pinstance == nullptr) {
      payload_queue->cv_.notify_one();
    } else {
      payload_queue->cv_.notify_all();
    }
  } else {
    StandardScheduleFunc sched_func = [this, payload_queue,
                                       payload](ModelInstanceContext* mi) {
      {
        std::lock_guard<std::mutex> lk(payload_queue->mu_);
        this->SchedulePayload(mi->RawInstance(), payload_queue, payload);
      }
      auto cb = [mi]() { mi->Release(); };
      payload->SetSecondaryCallback(cb);
      if (mi->RawInstance() == nullptr) {
        payload_queue->cv_.notify_one();
      } else {
        payload_queue->cv_.notify_all();
      }
    };
    DeferPayloadSchedule(sched_func, model, payload->GetInstance());
  }
  return Status::Success;
}

void
RateLimiter::DequeuePayload(
    std::deque<TritonModelInstance*>& instances,
    std::shared_ptr<Payload>* payload)
{
  payload->reset();
  if (payload_queues_.find(instances[0]->Model()) == payload_queues_.end()) {
    LOG_INFO << "Should not print this ";
  }
  PayloadQueue* payload_queue = payload_queues_[instances[0]->Model()].get();
  std::vector<std::shared_ptr<Payload>> merged_payloads;
  size_t instance_index;
  {
    std::unique_lock<std::mutex> lk(payload_queue->mu_);
    payload_queue->cv_.wait(lk, [&instances, &instance_index, payload_queue]() {
      bool empty = payload_queue->queue_->Empty();
      if (empty) {
        instance_index = 0;
        for (const auto instance : instances) {
          empty = payload_queue->specific_queues_[instance]->Empty();
          if (empty) {
            instance_index++;
          } else {
            break;
          }
        }
      }
      return !empty;
    });
    if (instance_index < instances.size()) {
      TritonModelInstance* instance = instances[instance_index];
      if (!payload_queue->specific_queues_[instance]->Empty()) {
        payload_queue->specific_queues_[instance]->Dequeue(
            payload, &merged_payloads);
      }
    } else {
      payload_queue->queue_->Dequeue(payload, &merged_payloads);
    }
  }
  for (auto& merge_payload : merged_payloads) {
    PayloadRelease(merge_payload);
  }
  (*payload)->Callback();
  if ((*payload)->GetInstance() == nullptr) {
    (*payload)->SetInstance(instances.front());
    instances.pop_front();
  } else {
    instances.erase(instances.begin() + instance_index);
  }
}

std::shared_ptr<Payload>
RateLimiter::GetPayload(
    const Payload::Operation op_type, TritonModelInstance* instance)
{
  std::shared_ptr<Payload> payload;

  if (max_payload_bucket_count_ > 0) {
    std::lock_guard<std::mutex> lock(alloc_mu_);

    if (!payload_bucket_.empty()) {
      payload = payload_bucket_.back();
      payload_bucket_.pop_back();
    }
    if (payload.get() == nullptr && (!payloads_in_use_.empty())) {
      // Just checking the front of the queue instead the entire queue for
      // an available payload to save time.
      if (payloads_in_use_.front().use_count() == 1) {
        payload = payloads_in_use_.front();
        payloads_in_use_.pop_front();
      }
    }
  }

  if (payload.get() == nullptr) {
    payload.reset(new Payload());
  }

  payload->Reset(op_type, instance);
  return payload;
}

void
RateLimiter::PayloadRelease(std::shared_ptr<Payload>& payload)
{
  payload->SecondaryCallback();
  if (max_payload_bucket_count_ > 0) {
    std::lock_guard<std::mutex> lock(alloc_mu_);

    if (payloads_in_use_.size() + payload_bucket_.size() <
        max_payload_bucket_count_) {
      // Release iff the payload shared_ptr is uniquely held.
      if (payload.use_count() == 1) {
        payload->Release();
        payload_bucket_.push_back(std::move(payload));
        return;
      } else {
        payloads_in_use_.push_back(std::move(payload));
      }
    }
  }
}

RateLimiter::RateLimiter(
    const bool ignore_resources_and_priority, const ResourceMap& resource_map)
    : ignore_resources_and_priority_(ignore_resources_and_priority),
      max_payload_bucket_count_(MAX_PAYLOAD_BUCKET_COUNT)
{
  ResourceManager::Create(resource_map, &resource_manager_);
}

void
RateLimiter::InitializePayloadQueues(const TritonModelInstance* instance)
{
  auto& config = instance->Model()->Config();
  uint64_t max_queue_delay_microseconds;
  if (config.has_sequence_batching()) {
    const auto& batcher_config = config.sequence_batching();
    if (batcher_config.has_oldest()) {
      max_queue_delay_microseconds =
          batcher_config.oldest().max_queue_delay_microseconds();
    } else {
      max_queue_delay_microseconds = 0;
    }
  } else if (config.has_dynamic_batching()) {
    max_queue_delay_microseconds =
        config.dynamic_batching().max_queue_delay_microseconds();
  } else {
    max_queue_delay_microseconds = 0;
  }
  if (payload_queues_.find(instance->Model()) == payload_queues_.end()) {
    payload_queues_.emplace(
        instance->Model(),
        new PayloadQueue(
            config.max_batch_size(), max_queue_delay_microseconds * 1000));
  }
  PayloadQueue* payload_queue = payload_queues_[instance->Model()].get();
  if (payload_queue->specific_queues_.find(instance) ==
      payload_queue->specific_queues_.end()) {
    payload_queue->specific_queues_.emplace(
        instance,
        new InstanceQueue(
            config.max_batch_size(), max_queue_delay_microseconds * 1000));
  }
}

Status
RateLimiter::DeferPayloadSchedule(
    const StandardScheduleFunc& OnSchedule, const TritonModel* model,
    TritonModelInstance* triton_model_instance)
{
  std::lock_guard<std::mutex> lk(model_ctx_mtx_);

  auto itr = model_contexts_.find(model);
  if (itr == model_contexts_.end()) {
    return Status(
        Status::Code::INTERNAL,
        "Requested model is not yet registered with rate limiter");
  }

  if (itr->second.isRemovalInProgress()) {
    return Status(
        Status::Code::INTERNAL,
        "New model requests can not be made to a model that is being "
        "removed");
  }

  itr->second.EnqueueModelInstanceRequest(OnSchedule, triton_model_instance);
  itr->second.StageInstanceIfAvailable(triton_model_instance);

  return Status::Success;
}

void
RateLimiter::SchedulePayload(
    TritonModelInstance* tmi, PayloadQueue* payload_queue,
    const std::shared_ptr<Payload>& payload)
{
  if (tmi == nullptr) {
    payload_queue->queue_->Enqueue(payload);
  } else {
    payload_queue->specific_queues_[tmi]->Enqueue(payload);
  }
  payload->SetState(Payload::State::SCHEDULED);
}

void
RateLimiter::OnStage(ModelInstanceContext* instance)
{
  {
    std::lock_guard<std::recursive_mutex> lk(staged_instances_mtx_);
    staged_instances_.push(instance);
  }
  AttemptAllocation();
}

void
RateLimiter::OnRelease(ModelInstanceContext* instance)
{
  auto& model_context = model_contexts_[instance->RawInstance()->Model()];
  model_context.AddAvailableInstance(instance);
  resource_manager_->ReleaseResources(instance);
  if (model_context.ContainsPendingRequests(instance->RawInstance()->Index())) {
    model_context.StageInstanceIfAvailable(instance->RawInstance());
  }
  AttemptAllocation();
}

void
RateLimiter::AttemptAllocation()
{
  std::lock_guard<std::recursive_mutex> lk(staged_instances_mtx_);
  if (!staged_instances_.empty()) {
    ModelInstanceContext* instance = staged_instances_.top();
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
    TritonModelInstance* triton_model_instance)
{
  std::lock_guard<std::recursive_mutex> lk(sched_request_queue_mtx_);

  if (triton_model_instance == nullptr) {
    generic_sched_request_queue_.push(OnSchedule);
  } else if (
      (uint32_t)triton_model_instance->Index() <
      specific_sched_request_queues_.size()) {
    specific_sched_request_queues_[triton_model_instance->Index()].push(
        OnSchedule);
  } else {
    return Status(
        Status::Code::INTERNAL,
        "expected instance index between 0 and " +
            std::to_string(specific_sched_request_queues_.size()) + ", got " +
            std::to_string(triton_model_instance->Index()));
  }

  return Status::Success;
}

void
RateLimiter::ModelContext::AddAvailableInstance(ModelInstanceContext* instance)
{
  std::lock_guard<std::recursive_mutex> lk(avbl_instances_mtx_);
  avbl_instances_.push(instance);
  instance->MarkAvailable();
}


void
RateLimiter::ModelContext::StageInstanceIfAvailable(
    TritonModelInstance* req_instance)
{
  std::lock_guard<std::recursive_mutex> lk1(sched_request_queue_mtx_);
  std::lock_guard<std::recursive_mutex> lk2(avbl_instances_mtx_);
  PriorityQueue backup_queue;

  while (!avbl_instances_.empty()) {
    ModelInstanceContext* instance = avbl_instances_.top();
    if ((req_instance != nullptr) &&
        (instance->RawInstance() != req_instance)) {
      backup_queue.push(instance);
      avbl_instances_.pop();
      continue;
    }
    if (!specific_sched_request_queues_[instance->RawInstance()->Index()]
             .empty()) {
      // Prioritize the specific requests for the available model
      // instance highest priority.
      const StandardScheduleFunc func =
          specific_sched_request_queues_[instance->RawInstance()->Index()]
              .front();
      specific_sched_request_queues_[instance->RawInstance()->Index()].pop();
      instance->Stage(func);
    } else if (!generic_sched_request_queue_.empty()) {
      // If request is for generic model instance then use the
      // instance with the highest priority.
      const StandardScheduleFunc func = generic_sched_request_queue_.front();
      generic_sched_request_queue_.pop();
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
  std::lock_guard<std::recursive_mutex> lk1(sched_request_queue_mtx_);
  std::lock_guard<std::recursive_mutex> lk2(avbl_instances_mtx_);
  PriorityQueue backup_queue;
  while (!avbl_instances_.empty()) {
    ModelInstanceContext* instance = avbl_instances_.top();
    if (!specific_sched_request_queues_[instance->RawInstance()->Index()]
             .empty()) {
      // Prioritize the specific requests for the available model
      // instance highest priority.
      const StandardScheduleFunc func =
          specific_sched_request_queues_[instance->RawInstance()->Index()]
              .front();
      specific_sched_request_queues_[instance->RawInstance()->Index()].pop();
      instance->DirectAllocate(func);
    } else if (!generic_sched_request_queue_.empty()) {
      // If request is for generic model instance then use the
      // instance with the highest priority.
      const StandardScheduleFunc func = generic_sched_request_queue_.front();
      generic_sched_request_queue_.pop();
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
  std::lock_guard<std::recursive_mutex> lk(sched_request_queue_mtx_);
  specific_sched_request_queues_.emplace_back();
}

bool
RateLimiter::ModelContext::ContainsPendingRequests(int index)
{
  std::lock_guard<std::recursive_mutex> lk(sched_request_queue_mtx_);
  return (generic_sched_request_queue_.size() != 0) ||
         (specific_sched_request_queues_[index].size() != 0);
}

void
RateLimiter::ModelContext::RequestRemoval()
{
  removal_in_progress_ = true;
}


//=========================================================================
//  ModelInstanceContext Implementation
//=========================================================================

RateLimiter::ModelInstanceContext::ModelInstanceContext(
    TritonModelInstance* triton_model_instance,
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
RateLimiter::ModelInstanceContext::MarkAvailable()
{
  std::lock_guard<std::mutex> lk(state_mtx_);
  state_ = AVAILABLE;
}

Status
RateLimiter::ModelInstanceContext::Stage(StandardScheduleFunc OnSchedule)
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
RateLimiter::ModelInstanceContext::Allocate()
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
RateLimiter::ModelInstanceContext::DirectAllocate(
    StandardScheduleFunc OnSchedule)
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
RateLimiter::ModelInstanceContext::Release()
{
  exec_count_++;

  OnRelease_(this);

  {
    std::lock_guard<std::mutex> lk(state_mtx_);
    if ((model_context_->isRemovalInProgress()) && (state_ == AVAILABLE) &&
        (!model_context_->ContainsPendingRequests(
            triton_model_instance_->Index()))) {
      state_ = REMOVED;
    }
  }

  if (state_ == REMOVED) {
    cv_.notify_all();
  }
}

void
RateLimiter::ModelInstanceContext::RequestRemoval()
{
  std::lock_guard<std::mutex> lk(state_mtx_);

  if ((state_ == AVAILABLE) && (!model_context_->ContainsPendingRequests(
                                   triton_model_instance_->Index()))) {
    state_ = REMOVED;
  }
}

void
RateLimiter::ModelInstanceContext::WaitForRemoval()
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
RateLimiter::ModelInstanceContext::ScaledPriority()
{
  // TODO: Different schemes for the prioritization of
  // model instance can be added here.
  // The priority of instance is 1 by default. If specified
  // as 0, the priority is still treated as 1.
  auto priority = std::max(rate_limiter_config_.priority(), 1u);
  return (exec_count_ * priority);
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
    const ModelInstanceContext* instance)
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
    const ModelInstanceContext* instance)
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

Status
RateLimiter::ResourceManager::UpdateResourceLimits()
{
  std::lock_guard<std::mutex> lk1(max_resources_mtx_);
  std::lock_guard<std::mutex> lk2(model_resources_mtx_);
  max_resources_.clear();
  // Obtain the maximum resource across all the instances
  // and use it as the default available.
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
  if (!explicit_max_resources_.empty()) {
    RETURN_IF_ERROR(ParseAndValidateExplicitResources());
  }
  RETURN_IF_ERROR(ValidateMaxResources());

  if (LOG_VERBOSE_IS_ON(1)) {
    std::string resource_map_str{"\nMax Resource Map===>\n"};
    for (const auto& ditr : max_resources_) {
      if (!ditr.second.empty()) {
        std::string device_str{(ditr.first == GLOBAL_RESOURCE_KEY)
                                   ? "GLOBAL"
                                   : std::to_string(ditr.first)};
        resource_map_str += "\tDevice: " + device_str + "\n";
        for (const auto& ritr : ditr.second) {
          resource_map_str += "\t\tResource: " + ritr.first +
                              "\t Count: " + std::to_string(ritr.second) + "\n";
        }
      }
    }
    LOG_VERBOSE(1) << resource_map_str;
  }

  return Status::Success;
}

Status
RateLimiter::ResourceManager::ValidateMaxResources()
{
  for (const auto& global_resource : max_resources_[GLOBAL_RESOURCE_KEY]) {
    for (const auto& ditr : max_resources_) {
      if (ditr.first != GLOBAL_RESOURCE_KEY) {
        for (const auto& ritr : ditr.second) {
          if (global_resource.first.compare(ritr.first) == 0) {
            return Status(
                Status::Code::INVALID_ARG,
                (std::string("Resource \"") + ritr.first +
                 "\" is present as both global and device-specific resource in "
                 "the model configuration.")
                    .c_str());
          }
        }
      }
    }
  }
  return Status::Success;
}

Status
RateLimiter::ResourceManager::ParseAndValidateExplicitResources()
{
  for (auto& ditr : max_resources_) {
    for (auto& ritr : ditr.second) {
      // If not specified explicitly, consider the resource to be unavailable.
      size_t resource_count = 0;
      if (ditr.first == GLOBAL_RESOURCE_KEY) {
        // Ignore the device specification... will search for all resources in
        // the map...
        for (const auto& exp_ditr : explicit_max_resources_) {
          for (const auto& exp_ritr : exp_ditr.second) {
            if (ritr.first.compare(exp_ritr.first) == 0) {
              if (resource_count < exp_ritr.second) {
                resource_count = exp_ritr.second;
              }
            }
          }
        }
      } else {
        // Search only for the device specific or per-device resources...
        // device-specific
        for (const auto& exp_ritr : explicit_max_resources_[ditr.first]) {
          if (ritr.first.compare(exp_ritr.first) == 0) {
            if (resource_count < exp_ritr.second) {
              resource_count = exp_ritr.second;
            }
          }
        }
        // per-device
        for (const auto& exp_ritr :
             explicit_max_resources_[PER_DEVICE_RESOURCE_KEY]) {
          if (ritr.first.compare(exp_ritr.first) == 0) {
            if (resource_count < exp_ritr.second) {
              resource_count = exp_ritr.second;
            }
          }
        }
      }
      if (resource_count < ritr.second) {
        return Status(
            Status::Code::INVALID_ARG,
            (std::string("Resource count for \"") + ritr.first +
             "\" is limited to " + std::to_string(resource_count) +
             " which will prevent scheduling of one or more model "
             "instances, the minimum required count is " +
             std::to_string(ritr.second))
                .c_str());
      } else {
        ritr.second = resource_count;
      }
    }
  }

  return Status::Success;
}

bool
RateLimiter::ResourceManager::AllocateResources(
    const ModelInstanceContext* instance)
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
    const ModelInstanceContext* instance)
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
