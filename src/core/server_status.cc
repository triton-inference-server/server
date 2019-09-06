// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "src/core/server_status.h"

#include <time.h>
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/metric_model_reporter.h"
#include "src/core/metrics.h"
#include "src/core/provider.h"

namespace nvidia { namespace inferenceserver {

namespace {

void
SetModelVersionReadyState(
    ModelStatus& ms, ModelRepositoryManager* model_repository_manager)
{
  const std::string& model_name = ms.config().name();

  // Set all model versions for which we have status to
  // unavailable... and then override that with actual status for the
  // versions that are currently being served.
  auto& mvs = *ms.mutable_version_status();
  for (auto& itr : mvs) {
    itr.second.set_ready_state(ModelReadyState::MODEL_UNAVAILABLE);
  }

  // [TODO] Once ModelRepositoryManager (MRM) is improved, instead of polling
  // version states from MRM, MRM will notify ServerStatusManager if there are
  // version state changes. In this way, there is no cross referencing between
  // these two classes.
  const auto versions_and_states =
      model_repository_manager->GetVersionStates(model_name);
  for (const auto& version_and_state : versions_and_states) {
    mvs[version_and_state.first].set_ready_state(version_and_state.second);
  }
}

}  // namespace

ServerStatusManager::ServerStatusManager(const std::string& server_version)
{
  const auto& version = server_version;
  if (!version.empty()) {
    server_status_.set_version(version);
  }
}

Status
ServerStatusManager::InitForModel(
    const std::string& model_name, const ModelConfig& model_config)
{
  std::lock_guard<std::mutex> lock(mu_);

  auto& ms = *server_status_.mutable_model_status();
  if (ms.find(model_name) == ms.end()) {
    LOG_INFO << "New status tracking for model '" << model_name << "'";
  } else {
    LOG_INFO << "New status tracking for re-added model '" << model_name << "'";
    ms[model_name].Clear();
  }

  ms[model_name].mutable_config()->CopyFrom(model_config);

  return Status::Success;
}

Status
ServerStatusManager::UpdateConfigForModel(
    const std::string& model_name, const ModelConfig& model_config)
{
  std::lock_guard<std::mutex> lock(mu_);

  auto& ms = *server_status_.mutable_model_status();
  if (ms.find(model_name) == ms.end()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "try to update config for non-existing model '" + model_name + "'");
  } else {
    LOG_INFO << "Updating config for model '" << model_name << "'";
  }

  ms[model_name].mutable_config()->CopyFrom(model_config);

  return Status::Success;
}

Status
ServerStatusManager::Get(
    ServerStatus* server_status, const std::string& server_id,
    ServerReadyState server_ready_state, uint64_t server_uptime_ns,
    ModelRepositoryManager* model_repository_manager) const
{
  std::lock_guard<std::mutex> lock(mu_);
  server_status->CopyFrom(server_status_);
  server_status->set_id(server_id);
  server_status->set_ready_state(server_ready_state);
  server_status->set_uptime_ns(server_uptime_ns);

  for (auto& msitr : *server_status->mutable_model_status()) {
    SetModelVersionReadyState(msitr.second, model_repository_manager);
  }

  return Status::Success;
}

Status
ServerStatusManager::Get(
    ServerStatus* server_status, const std::string& server_id,
    ServerReadyState server_ready_state, uint64_t server_uptime_ns,
    const std::string& model_name,
    ModelRepositoryManager* model_repository_manager) const
{
  std::lock_guard<std::mutex> lock(mu_);

  server_status->Clear();
  server_status->set_version(server_status_.version());
  server_status->set_id(server_id);
  server_status->set_ready_state(server_ready_state);
  server_status->set_uptime_ns(server_uptime_ns);

  const auto& itr = server_status_.model_status().find(model_name);
  if (itr == server_status_.model_status().end()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "no status available for unknown model '" + model_name + "'");
  }

  auto& ms = *server_status->mutable_model_status();
  ms[model_name].CopyFrom(itr->second);
  SetModelVersionReadyState(ms[model_name], model_repository_manager);

  return Status::Success;
}

void
ServerStatusManager::UpdateServerStat(
    uint64_t duration, ServerStatTimerScoped::Kind kind)
{
  std::lock_guard<std::mutex> lock(mu_);

  switch (kind) {
    case ServerStatTimerScoped::Kind::STATUS: {
      StatDuration* d =
          server_status_.mutable_status_stats()->mutable_success();
      d->set_count(d->count() + 1);
      d->set_total_time_ns(d->total_time_ns() + duration);
      break;
    }

    case ServerStatTimerScoped::Kind::HEALTH: {
      StatDuration* d =
          server_status_.mutable_health_stats()->mutable_success();
      d->set_count(d->count() + 1);
      d->set_total_time_ns(d->total_time_ns() + duration);
      break;
    }

    case ServerStatTimerScoped::Kind::MODEL_CONTROL: {
      StatDuration* d =
          server_status_.mutable_model_control_stats()->mutable_success();
      d->set_count(d->count() + 1);
      d->set_total_time_ns(d->total_time_ns() + duration);
      break;
    }

    case ServerStatTimerScoped::Kind::SHARED_MEMORY_CONTROL: {
      StatDuration* d =
          server_status_.mutable_shm_control_stats()->mutable_success();
      d->set_count(d->count() + 1);
      d->set_total_time_ns(d->total_time_ns() + duration);
      break;
    }

    case ServerStatTimerScoped::Kind::TRACE_CONTROL: {
      StatDuration* d =
          server_status_.mutable_trace_control_stats()->mutable_success();
      d->set_count(d->count() + 1);
      d->set_total_time_ns(d->total_time_ns() + duration);
      break;
    }
  }
}

void
ServerStatusManager::UpdateFailedInferStats(
    const std::string& model_name, const int64_t model_version,
    size_t batch_size, uint64_t request_duration_ns)
{
  std::lock_guard<std::mutex> lock(mu_);

  // Model must exist...
  auto itr = server_status_.mutable_model_status()->find(model_name);
  if (itr == server_status_.model_status().end()) {
    LOG_ERROR << "can't update INFER duration stat for " << model_name;
  } else {
    // batch_size may be zero if the failure occurred before it could
    // be determined... but we still record the failure.

    // model version
    auto& mvs = *itr->second.mutable_version_status();
    auto mvs_itr = mvs.find(model_version);
    if (mvs_itr == mvs.end()) {
      ModelVersionStatus& version_status = mvs[model_version];
      InferRequestStats& stats =
          (*version_status.mutable_infer_stats())[batch_size];
      stats.mutable_failed()->set_count(1);
      stats.mutable_failed()->set_total_time_ns(request_duration_ns);
    } else {
      ModelVersionStatus& version_status = mvs_itr->second;
      auto& is = *version_status.mutable_infer_stats();
      auto is_itr = is.find(batch_size);
      if (is_itr == is.end()) {
        InferRequestStats& stats = is[batch_size];
        stats.mutable_failed()->set_count(1);
        stats.mutable_failed()->set_total_time_ns(request_duration_ns);
      } else {
        InferRequestStats& stats = is_itr->second;
        stats.mutable_failed()->set_count(stats.failed().count() + 1);
        stats.mutable_failed()->set_total_time_ns(
            stats.failed().total_time_ns() + request_duration_ns);
      }
    }
  }
}

void
ServerStatusManager::UpdateSuccessInferStats(
    const std::string& model_name, const int64_t model_version,
    size_t batch_size, uint32_t execution_cnt, uint64_t request_duration_ns,
    uint64_t queue_duration_ns, uint64_t compute_duration_ns)
{
  std::lock_guard<std::mutex> lock(mu_);

  // Model must exist...
  auto itr = server_status_.mutable_model_status()->find(model_name);
  if (itr == server_status_.model_status().end()) {
    LOG_ERROR << "can't update duration stat for " << model_name;
  } else if (batch_size == 0) {
    LOG_ERROR << "can't update INFER durations without batch size for "
              << model_name;
  } else {
    // model version
    auto& mvs = *itr->second.mutable_version_status();
    auto mvs_itr = mvs.find(model_version);
    InferRequestStats* new_stats = nullptr;
    InferRequestStats* existing_stats = nullptr;
    if (mvs_itr == mvs.end()) {
      ModelVersionStatus& version_status = mvs[model_version];
      version_status.set_model_inference_count(batch_size);
      version_status.set_model_execution_count(execution_cnt);
      new_stats = &((*version_status.mutable_infer_stats())[batch_size]);
    } else {
      ModelVersionStatus& version_status = mvs_itr->second;
      version_status.set_model_inference_count(
          version_status.model_inference_count() + batch_size);
      version_status.set_model_execution_count(
          version_status.model_execution_count() + execution_cnt);

      auto& is = *version_status.mutable_infer_stats();
      auto is_itr = is.find(batch_size);
      if (is_itr == is.end()) {
        new_stats = &is[batch_size];
      } else {
        existing_stats = &is_itr->second;
      }
    }

    if (new_stats != nullptr) {
      new_stats->mutable_success()->set_count(1);
      new_stats->mutable_success()->set_total_time_ns(request_duration_ns);
      new_stats->mutable_compute()->set_count(1);
      new_stats->mutable_compute()->set_total_time_ns(compute_duration_ns);
      new_stats->mutable_queue()->set_count(1);
      new_stats->mutable_queue()->set_total_time_ns(queue_duration_ns);
    } else if (existing_stats != nullptr) {
      InferRequestStats& stats = *existing_stats;
      stats.mutable_success()->set_count(stats.success().count() + 1);
      stats.mutable_success()->set_total_time_ns(
          stats.success().total_time_ns() + request_duration_ns);
      stats.mutable_compute()->set_count(stats.compute().count() + 1);
      stats.mutable_compute()->set_total_time_ns(
          stats.compute().total_time_ns() + compute_duration_ns);
      stats.mutable_queue()->set_count(stats.queue().count() + 1);
      stats.mutable_queue()->set_total_time_ns(
          stats.queue().total_time_ns() + queue_duration_ns);
    } else {
      LOG_ERROR << "Internal error logging INFER stats for " << model_name;
    }
  }
}

ServerStatTimerScoped::~ServerStatTimerScoped()
{
  // Do nothing reporting is disabled...
  if (enabled_) {
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);

    uint64_t start_ns = start_.tv_sec * NANOS_PER_SECOND + start_.tv_nsec;
    uint64_t end_ns = end.tv_sec * NANOS_PER_SECOND + end.tv_nsec;
    uint64_t duration = (start_ns > end_ns) ? 0 : end_ns - start_ns;

    status_manager_->UpdateServerStat(duration, kind_);
  }
}

ModelInferStats::ScopedTimer::ScopedTimer()
    : cummulative_duration_ns_(0), duration_ptr_(nullptr)
{
  start_.tv_sec = 0;
  start_.tv_nsec = 0;
}

ModelInferStats::ScopedTimer::~ScopedTimer()
{
  if (duration_ptr_ != nullptr) {
    Stop();
    *duration_ptr_ = cummulative_duration_ns_;
  }
}

struct timespec
ModelInferStats::ScopedTimer::Start()
{
  clock_gettime(CLOCK_MONOTONIC, &start_);
  return start_;
}

void
ModelInferStats::ScopedTimer::Stop()
{
  // Ignore the stop if the timer hasn't been started
  if (start_.tv_sec != 0) {
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);

    uint64_t start_ns = start_.tv_sec * NANOS_PER_SECOND + start_.tv_nsec;
    uint64_t end_ns = end.tv_sec * NANOS_PER_SECOND + end.tv_nsec;
    cummulative_duration_ns_ += (start_ns > end_ns) ? 0 : end_ns - start_ns;

    start_.tv_sec = 0;
    start_.tv_nsec = 0;
  }
}

ModelInferStats::~ModelInferStats()
{
  // If the inference request failed before a backend could be
  // determined, there will be no metrics reporter.. so just use the
  // version directly from the inference request.
  const int64_t model_version = (metric_reporter_ != nullptr)
                                    ? metric_reporter_->ModelVersion()
                                    : requested_model_version_;

  if (failed_) {
    status_manager_->UpdateFailedInferStats(
        model_name_, model_version, batch_size_, request_duration_ns_);
#ifdef TRTIS_ENABLE_METRICS
    if (metric_reporter_ != nullptr) {
      metric_reporter_->MetricInferenceFailure(gpu_device_).Increment();
    }
#endif  // TRTIS_ENABLE_METRICS
  } else {
    status_manager_->UpdateSuccessInferStats(
        model_name_, model_version, batch_size_, execution_count_,
        request_duration_ns_, queue_duration_ns_, compute_duration_ns_);

#ifdef TRTIS_ENABLE_METRICS
    if (metric_reporter_ != nullptr) {
      metric_reporter_->MetricInferenceSuccess(gpu_device_).Increment();
      metric_reporter_->MetricInferenceCount(gpu_device_)
          .Increment(batch_size_);
      if (execution_count_ > 0) {
        metric_reporter_->MetricInferenceExecutionCount(gpu_device_)
            .Increment(execution_count_);
      }

      metric_reporter_->MetricInferenceRequestDuration(gpu_device_)
          .Increment(request_duration_ns_ / 1000);
      metric_reporter_->MetricInferenceComputeDuration(gpu_device_)
          .Increment(compute_duration_ns_ / 1000);
      metric_reporter_->MetricInferenceQueueDuration(gpu_device_)
          .Increment(queue_duration_ns_ / 1000);

      metric_reporter_->MetricInferenceLoadRatio(gpu_device_)
          .Observe(
              (double)request_duration_ns_ /
              std::max(1.0, (double)compute_duration_ns_));
    }
#endif  // TRTIS_ENABLE_METRICS
  }
}


void
ModelInferStats::IncrementQueueDuration(const uint64_t increment_value)
{
  queue_duration_ns_ += increment_value;
}

void
ModelInferStats::IncrementComputeDuration(const uint64_t increment_value)
{
  compute_duration_ns_ += increment_value;
}

struct timespec
ModelInferStats::StartRequestTimer(ScopedTimer* timer) const
{
  timer->duration_ptr_ = &request_duration_ns_;
  return timer->Start();
}

struct timespec
ModelInferStats::StartQueueTimer(ScopedTimer* timer) const
{
  timer->duration_ptr_ = &queue_duration_ns_;
  return timer->Start();
}

struct timespec
ModelInferStats::StartComputeTimer(ScopedTimer* timer) const
{
  timer->duration_ptr_ = &compute_duration_ns_;
  return timer->Start();
}

}}  // namespace nvidia::inferenceserver
