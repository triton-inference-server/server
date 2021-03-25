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
  std::unique_ptr<TritonBackendThreadPool> local_threadpool(
      new TritonBackendThreadPool(OnInit, OnRun));

  bool initialization_failed = false;
  if (triton_model->Instances().size() != 1) {
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
  }

  if (initialization_failed) {
    return Status(
        Status::Code::INTERNAL, "Initialization failed for backend threads");
  }

  threadpool->reset(local_threadpool.release());

  return Status::Success;
}

TritonBackendThreadPool::TritonBackendThreadPool(
    const StandardInitFuncV2& OnInit, const StandardRunFuncV2& OnRun)
    : OnInit_(OnInit), OnRun_(OnRun), signal_exit_(false)
{
}

TritonBackendThreadPool::~TritonBackendThreadPool()
{
  signal_exit_ = true;
  for (const auto& thread_context_ : backend_thread_contexts_) {
    thread_context_.second->ready_.store(true);
    thread_context_.second->cv_.notify_one();
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
  if (backend_thread_contexts_.empty()) {
    // Single instance case... Execute on this thread itself...
    LOG_VERBOSE(1) << "Executing on scheduler thread for instance \""
                   << instance->Name() << "\"...";
    OnRun_(instance, std::move(requests));
    callback_fn();
  } else {
    {
      std::lock_guard<std::mutex> lk(backend_thread_contexts_[instance]->mtx_);
      backend_thread_contexts_[instance]->requests_ = std::move(requests);
      backend_thread_contexts_[instance]->completion_cb_ = callback_fn;
      backend_thread_contexts_[instance]->ready_.store(true);
    }
    backend_thread_contexts_[instance]->cv_.notify_one();
  }

  return Status::Success;
}

void
TritonBackendThreadPool::BackendThread(
    TritonModelInstance* instance, BackendThreadContext* rthread_context,
    std::promise<bool>* is_initialized)
{
  LOG_VERBOSE(1) << "Starting backend thread for instance \""
                 << instance->Name() << "\"...";


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
    std::unique_lock<std::mutex> lk(rthread_context->mtx_);
    rthread_context->cv_.wait(
        lk, [rthread_context] { return rthread_context->ready_.load(); });
    if (!rthread_context->requests_.empty()) {
      LOG_VERBOSE(1) << "Executing backend thread for instance \""
                     << instance->Name() << "\"...";
      OnRun_(instance, std::move(rthread_context->requests_));
      rthread_context->requests_.clear();
      rthread_context->completion_cb_();
      rthread_context->count_++;
    } else {
      rthread_context->ready_.store(false);
    }
  }

  LOG_VERBOSE(1) << "Stopping backend thread for instance \""
                 << instance->Name() << "\"..."
                 << " with count " << rthread_context->count_;
}

}}  // namespace nvidia::inferenceserver