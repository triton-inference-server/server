// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "src/backends/backend/triton_model_instance.h"

#ifndef _WIN32
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include "model_config.pb.h"
#include "src/backends/backend/triton_model.h"
#include "src/core/logging.h"
#include "src/core/metrics.h"
#include "src/core/nvtx.h"

namespace nvidia { namespace inferenceserver {

TritonModelInstance::TritonModelInstance(
    TritonModel* model, const std::string& name, const size_t index,
    const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
    const std::vector<std::string>& profile_names, const bool passive)
    : model_(model), name_(name), index_(index), kind_(kind),
      device_id_(device_id), profile_names_(profile_names), passive_(passive),
      state_(nullptr)
{
#ifdef TRITON_ENABLE_METRICS
  if (Metrics::Enabled()) {
    // Use an ID in the metric only for GPU instances. Otherwise use
    // -1 to indicate no device should be reported in the metric.
    const int id =
        (kind_ == TRITONSERVER_INSTANCEGROUPKIND_GPU) ? device_id_ : -1;
    MetricModelReporter::Create(
        model_->Name(), model_->Version(), id, model_->Config().metric_tags(),
        &reporter_);
  }
#endif  // TRITON_ENABLE_METRICS
}

TritonModelInstance::~TritonModelInstance()
{
  // Model finalization is optional...
  if (model_->Backend()->ModelInstanceFiniFn() != nullptr) {
    LOG_TRITONSERVER_ERROR(
        model_->Backend()->ModelInstanceFiniFn()(
            reinterpret_cast<TRITONBACKEND_ModelInstance*>(this)),
        "failed finalizing model instance");
  }
}

Status
TritonModelInstance::CreateInstances(
    TritonModel* model, const inference::ModelConfig& model_config)
{
  bool use_backend_threads = false;
  size_t count = 0;
  for (const auto& group : model_config.instance_group()) {
    if (!group.passive()) {
      count += group.count();
      if (count > 1) {
        use_backend_threads = true;
        break;
      }
    }
  }

  for (const auto& group : model_config.instance_group()) {
    std::vector<std::string> profile_names;
    for (const auto& profile_name : group.profile()) {
      profile_names.push_back(profile_name);
    }
    for (int32_t c = 0; c < group.count(); ++c) {
      std::string instance_name{group.count() > 1
                                    ? group.name() + "_" + std::to_string(c)
                                    : group.name()};
      const bool passive = group.passive();
      if (group.kind() == inference::ModelInstanceGroup::KIND_CPU) {
        RETURN_IF_ERROR(CreateInstance(
            model, instance_name, c, TRITONSERVER_INSTANCEGROUPKIND_CPU,
            0 /* device_id */, profile_names, passive, group.rate_limiter(),
            use_backend_threads));
      } else if (group.kind() == inference::ModelInstanceGroup::KIND_GPU) {
        for (const int32_t device_id : group.gpus()) {
          RETURN_IF_ERROR(CreateInstance(
              model, instance_name, c, TRITONSERVER_INSTANCEGROUPKIND_GPU,
              device_id, profile_names, passive, group.rate_limiter(),
              use_backend_threads));
        }
      } else if (group.kind() == inference::ModelInstanceGroup::KIND_MODEL) {
        RETURN_IF_ERROR(CreateInstance(
            model, instance_name, c, TRITONSERVER_INSTANCEGROUPKIND_MODEL,
            0 /* device_id */, profile_names, passive, group.rate_limiter(),
            use_backend_threads));
      } else {
        return Status(
            Status::Code::INVALID_ARG,
            std::string("instance_group kind ") +
                ModelInstanceGroup_Kind_Name(group.kind()) + " not supported");
      }
    }
  }

  return Status::Success;
}

Status
TritonModelInstance::CreateInstance(
    TritonModel* model, const std::string& name, const size_t index,
    const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
    const std::vector<std::string>& profile_names, const bool passive,
    const inference::ModelRateLimiter& rate_limiter_config,
    const bool use_backend_threads)
{
  std::unique_ptr<TritonModelInstance> local_instance(new TritonModelInstance(
      model, name, index, kind, device_id, profile_names, passive));

  TRITONBACKEND_ModelInstance* triton_instance =
      reinterpret_cast<TRITONBACKEND_ModelInstance*>(local_instance.get());

  if (use_backend_threads) {
    local_instance->SetBackendThread();
  }

  // Instance initialization is optional...
  if (model->Backend()->ModelInstanceInitFn() != nullptr) {
    RETURN_IF_TRITONSERVER_ERROR(
        model->Backend()->ModelInstanceInitFn()(triton_instance));
  }

  RETURN_IF_ERROR(model->AddInstance(
      std::move(local_instance), passive, rate_limiter_config));

  return Status::Success;
}

Status
TritonModelInstance::SetBackendThread()
{
  // TODO: Currently each instance has a dedicated TritonBackendThread.
  // For device blocking execution policy we must share the
  // TritonBackendThread object with the instances on same device.
  // FIXME: Get the nice value for the thread from config
  std::unique_ptr<TritonBackendThread> local_backend_thread;
  RETURN_IF_ERROR(TritonBackendThread::CreateBackendThread(
      Name(), 0 /* nice */, &local_backend_thread));
  triton_backend_thread_ = std::move(local_backend_thread);

  return Status::Success;
}

Status
TritonModelInstance::Initialize()
{
  Status status;
  if (triton_backend_thread_.get() != nullptr) {
    auto payload = std::make_shared<TritonBackendThread::Payload>(
        TritonBackendThread::Operation::INIT, this);
    triton_backend_thread_->Enqueue(payload);
    status = payload->Wait();
  } else {
    status = InitializeFunc();
  }
  return status;
}

Status
TritonModelInstance::WarmUp()
{
  Status status;
  if (triton_backend_thread_.get() != nullptr) {
    auto payload = std::make_shared<TritonBackendThread::Payload>(
        TritonBackendThread::Operation::WARM_UP, this);
    triton_backend_thread_->Enqueue(payload);
    status = payload->Wait();
  } else {
    status = WarmUpFunc();
  }
  return status;
}

void
TritonModelInstance::Schedule(
    std::vector<std::unique_ptr<InferenceRequest>>&& requests,
    const std::function<void()>& OnCompletion)
{
  if (triton_backend_thread_.get() != nullptr) {
    auto payload = std::make_shared<TritonBackendThread::Payload>(
        TritonBackendThread::Operation::INFER_RUN, this, std::move(requests),
        OnCompletion);
    triton_backend_thread_->Enqueue(payload);
  } else {
    ScheduleFunc(std::move(requests), OnCompletion);
  }
}

void
TritonModelInstance::ScheduleFunc(
    std::vector<std::unique_ptr<InferenceRequest>>&& requests,
    const std::function<void()>& OnCompletion)
{
  // Use a thread local vector to avoid needing to malloc each
  // time an inference is run.
  thread_local std::vector<TRITONBACKEND_Request*> triton_requests(1024);
  triton_requests.clear();
  for (auto& r : requests) {
    triton_requests.push_back(
        reinterpret_cast<TRITONBACKEND_Request*>(r.release()));
  }

  TRITONBACKEND_ModelInstance* triton_model_instance =
      reinterpret_cast<TRITONBACKEND_ModelInstance*>(this);
  TritonBackend::TritonModelInstanceExecFn_t inst_exec_fn =
      model_->Backend()->ModelInstanceExecFn();

  // If there is an error then we retain ownership of 'requests'
  // and must send error responses.
  TRITONSERVER_Error* err = inst_exec_fn(
      triton_model_instance, &triton_requests[0], triton_requests.size());
  if (err != nullptr) {
    Status status = Status(
        TritonCodeToStatusCode(TRITONSERVER_ErrorCode(err)),
        TRITONSERVER_ErrorMessage(err));
    for (TRITONBACKEND_Request* tr : triton_requests) {
      std::unique_ptr<InferenceRequest> ur(
          reinterpret_cast<InferenceRequest*>(tr));
      InferenceRequest::RespondIfError(ur, status, true /* release_requests */);
    }

    TRITONSERVER_ErrorDelete(err);
  }

  OnCompletion();
}

Status
TritonModelInstance::WarmUpFunc()
{
  LOG_ERROR << "WarmUp is not yet supported";
  return Status::Success;
}

Status
TritonModelInstance::TritonBackendThread::CreateBackendThread(
    const std::string name, const int nice,
    std::unique_ptr<TritonBackendThread>* triton_backend_thread)
{
  TritonBackendThread* raw_triton_backend_thread =
      new TritonBackendThread(name);
  std::unique_ptr<TritonBackendThread> runner(raw_triton_backend_thread);

  runner->backend_thread_ = std::thread([raw_triton_backend_thread, nice]() {
    raw_triton_backend_thread->BackendThread(nice);
  });

  triton_backend_thread->reset(runner.release());

  return Status::Success;
}

TritonModelInstance::TritonBackendThread::TritonBackendThread(
    const std::string& name)
    : name_(name)
{
}

TritonModelInstance::TritonBackendThread::~TritonBackendThread()
{
  // Signal the backend thread to exit and then wait for it..
  auto exit_payload = std::make_shared<Payload>(Operation::EXIT, nullptr);
  queue_.Put(exit_payload);
  if (backend_thread_.joinable()) {
    backend_thread_.join();
  }
}

TritonModelInstance::TritonBackendThread::Payload::Payload(
    const Operation op_type, TritonModelInstance* instance)
    : op_type_(op_type), instance_(instance),
      requests_(std::vector<std::unique_ptr<InferenceRequest>>()),
      OnCompletion_([]() {})
{
}

TritonModelInstance::TritonBackendThread::Payload::Payload(
    const Operation op_type, TritonModelInstance* instance,
    std::vector<std::unique_ptr<InferenceRequest>>&& requests,
    std::function<void()> OnCompletion)
    : op_type_(op_type), instance_(instance), requests_(std::move(requests)),
      OnCompletion_(OnCompletion)
{
}

Status
TritonModelInstance::TritonBackendThread::Payload::Wait()
{
  return status_.get_future().get();
}

void
TritonModelInstance::TritonBackendThread::Payload::Execute(bool* should_exit)
{
  *should_exit = false;

  Status status;
  switch (op_type_) {
    case Operation::INFER_RUN:
      instance_->ScheduleFunc(std::move(requests_), OnCompletion_);

      break;
    case Operation::INIT:
      status = instance_->InitializeFunc();
      break;
    case Operation::WARM_UP:
      status = instance_->WarmUpFunc();
      break;
    case Operation::EXIT:
      *should_exit = true;
  }

  status_.set_value(status);
}

void
TritonModelInstance::TritonBackendThread::Enqueue(
    std::shared_ptr<Payload> payload)
{
  queue_.Put(payload);
}

void
TritonModelInstance::TritonBackendThread::BackendThread(const int nice)
{
#ifndef _WIN32
  if (setpriority(PRIO_PROCESS, syscall(SYS_gettid), nice) == 0) {
    LOG_VERBOSE(1) << "Starting backend thread for " << name_ << " at nice "
                   << nice << "...";
  } else {
    LOG_VERBOSE(1) << "Starting backend thread for " << name_
                   << " at default nice (requested nice " << nice
                   << " failed)...";
  }
#else
  LOG_VERBOSE(1) << "Starting backend thread for " << name_
                 << " at default nice...";
#endif

  bool should_exit = false;
  while (!should_exit) {
    NVTX_RANGE(nvtx_, "BackendThread " + name_);
    auto payload = queue_.Get();
    payload->Execute(&should_exit);
  }

  LOG_VERBOSE(1) << "Stopping backend thread for " << name_ << "...";
}

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceName(
    TRITONBACKEND_ModelInstance* instance, const char** name)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *name = ti->Name().c_str();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceKind(
    TRITONBACKEND_ModelInstance* instance, TRITONSERVER_InstanceGroupKind* kind)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *kind = ti->Kind();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceDeviceId(
    TRITONBACKEND_ModelInstance* instance, int32_t* device_id)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *device_id = ti->DeviceId();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceProfileCount(
    TRITONBACKEND_ModelInstance* instance, uint32_t* count)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *count = ti->Profiles().size();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceProfileName(
    TRITONBACKEND_ModelInstance* instance, const uint32_t index,
    const char** profile_name)
{
  *profile_name = nullptr;

  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  const auto& rprofiles = ti->Profiles();
  if (index >= rprofiles.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("out of bounds index ") + std::to_string(index) +
         ": instance is configured with " + std::to_string(rprofiles.size()) +
         " profiles")
            .c_str());
  }

  *profile_name = rprofiles[index].c_str();

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceIsPassive(
    TRITONBACKEND_ModelInstance* instance, bool* is_passive)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *is_passive = ti->IsPassive();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceModel(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Model** model)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *model = reinterpret_cast<TRITONBACKEND_Model*>(ti->Model());
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceState(
    TRITONBACKEND_ModelInstance* instance, void** state)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *state = ti->State();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceSetState(
    TRITONBACKEND_ModelInstance* instance, void* state)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  ti->SetState(state);
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceReportStatistics(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request* request,
    const bool success, const uint64_t exec_start_ns,
    const uint64_t compute_start_ns, const uint64_t compute_end_ns,
    const uint64_t exec_end_ns)
{
#ifdef TRITON_ENABLE_STATS
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  tr->ReportStatistics(
      ti->MetricReporter(), success, exec_start_ns, compute_start_ns,
      compute_end_ns, exec_end_ns);
#endif  // TRITON_ENABLE_STATS

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceReportBatchStatistics(
    TRITONBACKEND_ModelInstance* instance, const uint64_t batch_size,
    const uint64_t exec_start_ns, const uint64_t compute_start_ns,
    const uint64_t compute_end_ns, const uint64_t exec_end_ns)
{
#ifdef TRITON_ENABLE_STATS
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  ti->Model()->MutableStatsAggregator()->UpdateInferBatchStats(
      ti->MetricReporter(), batch_size, exec_start_ns, compute_start_ns,
      compute_end_ns, exec_end_ns);
#endif  // TRITON_ENABLE_STATS

  return nullptr;  // success
}

}  // extern C
}}  // namespace nvidia::inferenceserver
