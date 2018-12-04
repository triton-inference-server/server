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
#pragma once

#include "src/core/infer.h"
#include "src/core/label_provider.h"
#include "src/core/model_config.pb.h"
#include "src/core/scheduler.h"
#include "src/servables/caffe2/netdef_bundle_c2.h"
#include "tensorflow/core/lib/core/errors.h"

namespace nvidia { namespace inferenceserver {

class NetDefBundle : public InferenceServable {
 public:
  NetDefBundle() = default;
  NetDefBundle(NetDefBundle&&) = default;

  tensorflow::Status Init(
    const tensorflow::StringPiece& path, const ModelConfig& config);

  // Create a context for execution for each instance for the
  // serialized netdefs specified in 'models'.
  tensorflow::Status CreateExecutionContexts(
    const std::unordered_map<std::string, std::vector<char>>& models);
  tensorflow::Status CreateExecutionContext(
    const std::string& instance_name, const int gpu_device,
    const std::unordered_map<std::string, std::vector<char>>& models);

  tensorflow::Status GetOutputDataType(
    const std::string& name, DataType* dtype) const override;
  const LabelProvider& GetLabelProvider() const override
  {
    return label_provider_;
  }

 private:
  // Run model on the context associated with 'runner_idx' to
  // execute for one or more requests.
  void Run(
    uint32_t runner_idx, std::vector<Scheduler::Payload>* payloads,
    std::function<void(tensorflow::Status)> OnCompleteQueuedPayloads);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(NetDefBundle);
  friend std::ostream& operator<<(std::ostream&, const NetDefBundle&);

  // Label provider for this bundle.
  LabelProvider label_provider_;

  // Map from an output name to the datatype of that output.
  std::unordered_map<std::string, DataType> output_dtype_map_;

  // For each model instance there is a context.
  struct Context {
    // GPU device number that indicates that no gpu is available for a
    // context
    static constexpr int NO_GPU_DEVICE = -1;

    // Max batch size value that indicates batching is not supported.
    static constexpr int NO_BATCHING = 0;

    Context(
      const std::string& name, const int gpu_device, const int max_batch_size);
    Context(Context&& o);
    ~Context();

    TF_DISALLOW_COPY_AND_ASSIGN(Context);

    tensorflow::Status InitializeInputs(
      const ::google::protobuf::RepeatedPtrField<ModelInput>& ios);
    tensorflow::Status InitializeOutputs(
      const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios);

    // Run model to execute for one or more requests. This function
    // assumes that it is only called by the single runner thread that
    // is assigned to this context. A non-OK return status indicates
    // an internal error that prevents any of the of requests from
    // completing. If an error is isolate to a single request payload
    // it will be reported in that payload.
    tensorflow::Status Run(std::vector<Scheduler::Payload>* payloads);

    // Name of the model instance
    std::string name_;

    // The GPU index active when this context was created.
    int gpu_device_;

    // Maximum batch size to allow. NO_BATCHING indicates that
    // batching is not supported.
    int max_batch_size_;

    // Caffe2 workspace.
    std::unique_ptr<Caffe2Workspace> workspace_;
  };

  std::vector<Context> contexts_;
};

std::ostream& operator<<(std::ostream& out, const NetDefBundle& pb);

}}  // namespace nvidia::inferenceserver
