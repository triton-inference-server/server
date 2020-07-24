// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <memory>
#include "src/backends/caffe2/netdef_backend_c2.h"
#include "src/core/backend.h"
#include "src/core/backend_context.h"
#include "src/core/metric_model_reporter.h"
#include "src/core/model_config.pb.h"
#include "src/core/scheduler.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class NetDefBackend : public InferenceBackend {
 public:
  explicit NetDefBackend(const double min_compute_capability)
      : InferenceBackend(min_compute_capability)
  {
  }
  NetDefBackend(NetDefBackend&&) = default;

  // Create a context for execution for each instance for the
  // serialized netdefs specified in 'models'.
  Status CreateExecutionContexts(
      const std::unordered_map<std::string, std::vector<char>>& models);
  Status CreateExecutionContext(
      const std::string& instance_name, const int gpu_device,
      const std::unordered_map<std::string, std::vector<char>>& models);

 private:
  Status ValidateBooleanSequenceControl(
      const inference::ModelSequenceBatching::Control::Kind control_kind,
      std::vector<std::string>* input_names, bool required);
  Status ValidateTypedSequenceControl(
      const inference::ModelSequenceBatching::Control::Kind control_kind,
      std::vector<std::string>* input_names, bool required);

 private:
  DISALLOW_COPY_AND_ASSIGN(NetDefBackend);
  friend std::ostream& operator<<(std::ostream&, const NetDefBackend&);

  // For each model instance there is a context.
  struct Context : BackendContext {
    Context(
        const std::string& name, const int gpu_device, const int max_batch_size,
        const bool enable_pinned_input, const bool enable_pinned_output,
        std::unique_ptr<MetricModelReporter>&& metric_reporter);
    ~Context();

    DISALLOW_MOVE(Context);
    DISALLOW_COPY_AND_ASSIGN(Context);

    Status ValidateInputs(
        const ::google::protobuf::RepeatedPtrField<inference::ModelInput>& ios);
    Status ValidateOutputs(
        const ::google::protobuf::RepeatedPtrField<inference::ModelOutput>& ios);

    // Set input tensors from one or more requests.
    Status SetInputTensors(
        size_t total_batch_size,
        const std::vector<std::unique_ptr<InferenceRequest>>& requests,
        std::vector<std::unique_ptr<InferenceResponse>>* responses,
        BackendInputCollector* collector,
        std::vector<std::unique_ptr<AllocatedMemory>>* input_buffers,
        bool* cuda_copy);

    // See BackendContext::Run()
    void Run(
        InferenceBackend* base,
        std::vector<std::unique_ptr<InferenceRequest>>&& requests) override;

    // Read output tensors to one or more requests.
    Status ReadOutputTensors(
        const InferenceBackend* base, size_t total_batch_size,
        const std::vector<std::unique_ptr<InferenceRequest>>& requests,
        std::vector<std::unique_ptr<InferenceResponse>>* responses);

    // Caffe2 workspace.
    std::unique_ptr<Caffe2Workspace> workspace_;
  };
};

std::ostream& operator<<(std::ostream& out, const NetDefBackend& pb);

}}  // namespace nvidia::inferenceserver
