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
#include "src/core/model_config.pb.h"
#include "src/core/scheduler.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/public/session.h"

namespace nvidia { namespace inferenceserver {

// Base for both GraphDef and SavedModel servables
class BaseBundle : public InferenceServable {
 public:
  BaseBundle() = default;
  BaseBundle(BaseBundle&&) = default;

  tensorflow::Status Init(
      const tensorflow::StringPiece& path, const ModelConfig& config);

  // Create a context for execution for each instance of the
  // tensorflow model specified in 'paths'. The model can be either a
  // graphdef or savedmodel
  tensorflow::Status CreateExecutionContexts(
      const tensorflow::ConfigProto& session_config,
      const std::unordered_map<std::string, std::string>& paths);
  tensorflow::Status CreateExecutionContext(
      const std::string& instance_name, const int gpu_device,
      const tensorflow::ConfigProto& session_config,
      const std::unordered_map<std::string, std::string>& paths);

 protected:
  using IONameMap = std::unordered_map<std::string, std::string>;

  // Load model and create a corresponding session object.
  virtual tensorflow::Status CreateSession(
      const tensorflow::SessionOptions& options, const int gpu_device,
      const std::string& model_path, tensorflow::Session** session,
      IONameMap* input_name_map, IONameMap* output_name_map) = 0;

  // For each model instance there is a context.
  struct Context {
    using TensorMap = std::unordered_map<std::string, tensorflow::Tensor>;

    // GPU device number that indicates that no gpu is available for a
    // context.
    static constexpr int NO_GPU_DEVICE = -1;

    // Max batch size value that indicates batching is not supported.
    static constexpr int NO_BATCHING = 0;

    Context(
        const std::string& name, const int gpu_device,
        const int max_batch_size);
    Context(Context&& o);
    ~Context();

    TF_DISALLOW_COPY_AND_ASSIGN(Context);

    tensorflow::Status InitializeOutputs(
        const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios);

    // Run model to execute for one or more requests. This function
    // assumes that it is only called by the single runner thread that
    // is assigned to this context. A non-OK return status indicates
    // an internal error that prevents any of the of requests from
    // completing. If an error is isolate to a single request payload
    // it will be reported in that payload.
    tensorflow::Status Run(
        const BaseBundle* base, std::vector<Scheduler::Payload>* payloads);

    // Name of the model instance
    std::string name_;

    // The GPU index active when this context was created.
    int gpu_device_;

    // Maximum batch size to allow. NO_BATCHING indicates that
    // batching is not supported.
    int max_batch_size_;

    // Map from configuration name for an input to tensor name for
    // that input in the model.
    IONameMap input_name_map_;

    // Map from configuration name for an output to tensor name for
    // that output in the model.
    IONameMap output_name_map_;

    // The output tensors. These are not used as actual outputs to the
    // model (those are created dynamically for each run) but simply
    // to hold information about each tensor (data-type, shape).
    TensorMap outputs_;

    // Tensorflow session for this context.
    tensorflow::Session* session_;
  };

 private:
  // Run model on the context associated with 'runner_idx' to
  // execute for one or more requests.
  void Run(
      uint32_t runner_idx, std::vector<Scheduler::Payload>* payloads,
      std::function<void(tensorflow::Status)> OnCompleteQueuedPayloads);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(BaseBundle);
  friend std::ostream& operator<<(std::ostream&, const BaseBundle&);

  // The contexts for this servable.
  std::vector<Context> contexts_;
};

std::ostream& operator<<(std::ostream& out, const BaseBundle& pb);

}}  // namespace nvidia::inferenceserver
