// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include "src/core/ensemble_scheduler.h"

#include "src/core/api.pb.h"
#include "src/core/backend.h"
#include "src/core/logging.h"
#include "src/core/server.h"
#include "src/core/server_status.h"

namespace nvidia { namespace inferenceserver {

namespace {

// EnsembleStep specifies the backend, providers and status objects used for
// the internal infer request
struct EnsembleStep {
  EnsembleStep(
      const std::shared_ptr<InferenceServer::InferBackendHandle>& backend,
      const std::shared_ptr<InferRequestProvider>& request_provider,
      const std::shared_ptr<InferResponseProvider>& response_provider)
      : backend_(backend), request_provider_(request_provider),
        response_provider_(response_provider)
  {
  }

  std::shared_ptr<InferenceServer::InferBackendHandle> backend_;
  std::shared_ptr<InferRequestProvider> request_provider_;
  std::shared_ptr<InferResponseProvider> response_provider_;
  RequestStatus request_status_;
};

// EnsembleContext maintains the state of the ensemble request
//
// Using static functions to take advantage of shared_ptr, a copy of the
// shared_ptr will be made when a step is scheduled and it will go out of
// scope after the step's callback is finished. The step's callback will
// schedule new steps if available and the last step will finish the ensemble
// request.
// So we don't have to maintian the context in scheduler as the shared_ptr
// will destroy the context for us if there are no "in-flight" steps.
class EnsembleContext {
 public:
  EnsembleContext(
      InferenceServer* is, const ModelConfig& config,
      const std::unordered_map<std::string, EnsembleTensor>& ensemble_graph,
      const std::shared_ptr<ModelInferStats>& stats,
      const std::shared_ptr<InferRequestProvider>& request_provider,
      const std::shared_ptr<InferResponseProvider>& response_provider,
      std::function<void(Status)> OnComplete);

  // Initiate transition on 'context' state
  static void Proceed(const std::shared_ptr<EnsembleContext>& context);

 private:
  using StepList = std::vector<std::shared_ptr<EnsembleStep>>;

  // Perform transition on 'context' state given the information of
  // 'completed_step'
  static void Proceed(
      const std::shared_ptr<EnsembleContext>& context,
      const std::shared_ptr<EnsembleStep>& completed_step);

  // Helper function that updates ensemble state given 'completed_step'
  void UpdateEnsembleState(const std::shared_ptr<EnsembleStep>& completed_step);

  // Helper function that returns a list of step that should be run
  // under current state.
  // 'ok' will be set to false if the ensemble context will not make any
  // progress from current state. 'ok' will be set to true otherwise
  StepList Next(bool& ok);

  // Helper function that completes the response of the ensemble request
  void FinishEnsemble();

  // Helper function that prepares and calls the inference server's function
  // to process infer requests specified in 'steps'
  static void ScheduleSteps(
      const std::shared_ptr<EnsembleContext>& context, const StepList& steps);

  InferenceServer* is_;
  // [TODO] we don't really need model config, we can construct data structure
  //   for each step based on the model config in constructor
  ModelConfig config_;
  std::unordered_map<std::string, EnsembleTensor> ensemble_graph_;
  std::shared_ptr<ModelInferStats> stats_;
  std::shared_ptr<InferRequestProvider> request_provider_;
  std::shared_ptr<InferResponseProvider> response_provider_;
  std::function<void(Status)> OnComplete_;
};

EnsembleContext::EnsembleContext(
    InferenceServer* is, const ModelConfig& config,
    const std::unordered_map<std::string, EnsembleTensor>& ensemble_graph,
    const std::shared_ptr<ModelInferStats>& stats,
    const std::shared_ptr<InferRequestProvider>& request_provider,
    const std::shared_ptr<InferResponseProvider>& response_provider,
    std::function<void(Status)> OnComplete)
    : is_(is), config_(config), ensemble_graph_(ensemble_graph), stats_(stats),
      request_provider_(request_provider),
      response_provider_(response_provider), OnComplete_(OnComplete)
{
}

void
EnsembleContext::Proceed(const std::shared_ptr<EnsembleContext>& context)
{
  Proceed(context, nullptr);
}

void
EnsembleContext::Proceed(
    const std::shared_ptr<EnsembleContext>& context,
    const std::shared_ptr<EnsembleStep>& completed_step)
{
  bool ok = false;
  context->UpdateEnsembleState(completed_step);
  StepList ready_steps = context->Next(ok);
  if (!ok) {
    context->FinishEnsemble();
  } else {
    ScheduleSteps(context, ready_steps);
  }
}

void
EnsembleContext::UpdateEnsembleState(
    const std::shared_ptr<EnsembleStep>& completed_step)
{
  // [TODO]
  //   If 'completed_step' is nullptr, update all ensemble tensor states.
  //   Otherwise,
  //     - Check request status, set ensemble fail if not ok.
  //     - If ok, find the corresponding ensemble tensor
  //       and update its state (ready, not write, read--).
}

EnsembleContext::StepList
EnsembleContext::Next(bool& ok)
{
  // [TODO]
  //   Find ready steps in model config, find corresponding ensemble tensor,
  //   create InferRequestHeader and create providers.
  //   Push those steps into StepList, also mark those steps as "processing".
  //   Set 'ok' to false if no progress can be made from this point:
  //     - StepList is empty and no steps are "processing"
  //     - Or ensemble fail due to 'not ok' request status from steps
  //   Otherwise, true.
  StepList res;
  return res;
}

void
EnsembleContext::FinishEnsemble()
{
  // [TODO]
  //   If this function has been called, ignore. Otherwise,
  //     - Update stats_ on request failure or not.
  //         ([TODO] how to time queue/compute time of ensemble request?)
  //     - Set InferResponseProvider that are passed at the start of ensemble
  //       (actual provider that sends response back).
  //     - Call OnComplete callback.
}

void
EnsembleContext::ScheduleSteps(
    const std::shared_ptr<EnsembleContext>& context, const StepList& steps)
{
  for (const auto& step : steps) {
    InferenceBackend* backend = (*step->backend_)();

    auto infer_stats = std::make_shared<ModelInferStats>(
        context->is_->StatusManager(), backend->Name());
    auto timer = std::make_shared<ModelInferStats::ScopedTimer>();
    infer_stats->StartRequestTimer(timer.get());
    infer_stats->SetRequestedVersion(backend->Version());
    infer_stats->SetModelBackend(backend);
    infer_stats->SetBatchSize(
        step->request_provider_->RequestHeader().batch_size());

    context->is_->HandleInfer(
        &(step->request_status_), step->backend_, step->request_provider_,
        step->response_provider_, infer_stats,
        [context, step, infer_stats, timer]() mutable {
          timer.reset();
          infer_stats.reset();
          Proceed(context, step);
        });
  }
}

}  // namespace

Status
EnsembleScheduler::Create(
    const ModelConfig& config, std::unique_ptr<Scheduler>* scheduler)
{
  EnsembleScheduler* ensemble_sched = new EnsembleScheduler(config);
  std::unique_ptr<EnsembleScheduler> sched(ensemble_sched);

  RETURN_IF_ERROR(BuildEnsembleGraph(config, sched->ensemble_graph_));

  scheduler->reset(sched.release());

  return Status::Success;
}

void
EnsembleScheduler::Enqueue(
    const std::shared_ptr<ModelInferStats>& stats,
    const std::shared_ptr<InferRequestProvider>& request_provider,
    const std::shared_ptr<InferResponseProvider>& response_provider,
    std::function<void(Status)> OnComplete)
{
  std::shared_ptr<EnsembleContext> context(new EnsembleContext(
      is_, config_, ensemble_graph_, stats, request_provider, response_provider,
      OnComplete));
  EnsembleContext::Proceed(context);
}

}}  // namespace nvidia::inferenceserver
