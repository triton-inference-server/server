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

#include "src/backends/backend/triton_backend_threadpool.h"

#include "src/backends/backend/triton_model_instance.h"

#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include "src/core/logging.h"

namespace nvidia { namespace inferenceserver {

Status
TritonBackendThreadPool::CreateThreadPool(
    const TritonModel* triton_model, const StandardInitFuncV2 OnInit,
    const StandardRunFuncV2 OnRun,
    std::unique_ptr<TritonBackendThreadPool>* threadpool)
{
  // int nice = GetCpuNiceLevel(triton_model->Backend()->BackendConfig());
  int nice = 0;
  std::unique_ptr<TritonBackendThreadPool> local_threadpool(
      new TritonBackendThreadPool(nice, OnInit, OnRun));

  bool initialization_failed = false;
  for (const auto& instance : triton_model->Instances()) {
    std::promise<bool> init_state;
    TritonModelInstance* raw_instance = instance.get();
    auto& this_context =
        local_threadpool->backend_thread_contexts_[raw_instance];
    this_context.reset(new BackendThreadContext());
    this_context->thread_.reset(new std::thread([&] {
      local_threadpool->BackendThread(
          raw_instance, this_context.get(), &init_state);
    }));
    if (!init_state.get_future().get()) {
      initialization_failed = true;
      if (this_context->thread_->joinable()) {
        this_context->thread_->join();
      }
      local_threadpool->backend_thread_contexts_.erase(raw_instance);
    }
  }

  if (initialization_failed) {
    return Status(
        Status::Code::INTERNAL, "Initialization failed for backend threads");
  }

  threadpool->reset(local_threadpool.release());

  return Status::Success;
}

TritonBackendThreadPool::TritonBackendThreadPool(
    const int nice, const StandardInitFuncV2& OnInit,
    const StandardRunFuncV2& OnRun)
    : nice_(nice), OnInit_(OnInit), OnRun_(OnRun), signal_exit_(false)
{
}

TritonBackendThreadPool::~TritonBackendThreadPool()
{
  signal_exit_ = true;
  for (const auto& thread_context_ : backend_thread_contexts_) {
    if (thread_context_.second->thread_->joinable()) {
      thread_context_.second->thread_->join();
    }
  }
}

Status
TritonBackendThreadPool::SubmitRequest(
    TritonModelInstance* instance,
    std::vector<std::unique_ptr<InferenceRequest>>&& requests,
    std::function<void()> callback_fn)
{
  // Pass this to appropriate backend thread
  LOG_INFO << "Inside Submit Request";

  // Single instance case... Execute on this thread itself...
  OnRun_(instance, std::move(requests));

  // TODO: For multiple instance push the requests to the backend thread
  // associated with the instance

  callback_fn();

  return Status::Success;
}

void
TritonBackendThreadPool::BackendThread(
    TritonModelInstance* instance, BackendThreadContext* rthread_context,
    std::promise<bool>* is_initialized)
{
#ifndef _WIN32
  if (setpriority(PRIO_PROCESS, syscall(SYS_gettid), nice_) == 0) {
    LOG_VERBOSE(1) << "Starting backend thread for instance \""
                   << instance->Name() << "\" at nice " << nice_ << "...";
  } else {
    LOG_VERBOSE(1) << "Starting backend thread for instance \""
                   << instance->Name() << "\" at default nice (requested nice "
                   << nice_ << " failed)...";
  }
#else
  LOG_VERBOSE(1) << "Starting backend thread for instance \""
                 << instance->Name() << "\" at default nice...";
#endif

  // Initialize using the thread. If error then just exit this thread
  // now... that means the corresponding model instance will not have
  // any runner and so will not get used for execution.
  Status startup_status = OnInit_(instance);

  // Run warmup function if initialization succeed.
  // if (startup_status.IsOk()) {
  //  startup_status = OnWarmup_(instance);
  // }

  if (!startup_status.IsOk()) {
    LOG_ERROR << "Initialization failed for backend thread for instance \""
              << instance->Name() << "\": " << startup_status.Message();
    is_initialized->set_value(false);
    return;
  } else {
    is_initialized->set_value(true);
  }

  while (!signal_exit_) {
    LOG_ERROR << "Inside backend thread for instance \"" << instance->Name()
              << "\": " << startup_status.Message();
    std::this_thread::sleep_for(std::chrono::seconds(5));
  }

  LOG_VERBOSE(1) << "Stopping backend thread for instance \""
                 << instance->Name() << "\"...";
}

}}  // namespace nvidia::inferenceserver