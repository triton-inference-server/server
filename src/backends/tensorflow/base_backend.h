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
#pragma once

#include "src/backends/tensorflow/graphdef_backend_factory.h"
#include "src/backends/tensorflow/tensorflow_backend_tf.h"
#include "src/core/backend.h"
#include "src/core/backend_context.h"
#include "src/core/model_config.pb.h"
#include "src/core/scheduler.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

// Base for both GraphDef and SavedModel backends
class BaseBackend : public InferenceBackend {
 public:
  BaseBackend() = default;
  BaseBackend(BaseBackend&&) = default;

  Status Init(const std::string& path, const ModelConfig& config);

  // Create a context for execution for each instance of the
  // tensorflow model specified in 'paths'. The model can be either a
  // graphdef or savedmodel
  Status CreateExecutionContexts(
      const std::shared_ptr<GraphDefBackendFactory::Config>& backend_config,
      const std::unordered_map<std::string, std::string>& paths);
  Status CreateExecutionContext(
      const std::string& instance_name, const int gpu_device,
      const std::shared_ptr<GraphDefBackendFactory::Config>& backend_config,
      const std::unordered_map<std::string, std::string>& paths);

 protected:
  using TRTISTFModelHandle =
      std::unique_ptr<TRTISTF_Model, decltype(&TRTISTF_ModelDelete)>;
  using IONameMap = std::unordered_map<std::string, std::string>;

  // Load model and create a corresponding TRTISTF model object.
  virtual Status CreateTRTISTFModel(
      const std::shared_ptr<GraphDefBackendFactory::Config>& backend_config,
      const int gpu_device, const bool has_graph_level, const int graph_level,
      const std::string& model_path, TRTISTFModelHandle* trtistf_model,
      IONameMap* input_name_map, IONameMap* output_name_map) = 0;

  // For each model instance there is a context.
  struct Context : BackendContext {
    // GPU device number that indicates model will be loaded on GPUs
    // as specified in model graph
    static constexpr int MODEL_DEVICE = -2;

    Context(
        const std::string& name, const int gpu_device,
        const int max_batch_size);
    ~Context();

    DISALLOW_MOVE(Context);
    DISALLOW_COPY_AND_ASSIGN(Context);

    Status ValidateInputs(
        const ::google::protobuf::RepeatedPtrField<ModelInput>& ios);
    Status ValidateOutputs(
        const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios);

    // Set an input tensor data from payloads.
    Status SetInput(
        const std::string& name, const DataType datatype, const DimsList& dims,
        const size_t total_batch_size,
        std::vector<Scheduler::Payload>* payloads,
        TRTISTF_TensorList** input_tensors);

    // Helper function to set the input for fixed-sized data type
    void SetFixedSizedInputTensor(
        TRTISTF_Tensor* tensor, const std::string& input_name,
        const size_t batch1_byte_size,
        std::vector<Scheduler::Payload>* payloads);

    // Helper function to set the output with fixed-sized data type in payload
    void ReadFixedSizedOutputTensor(
        TRTISTF_Tensor* tensor, const std::string& output_name,
        const std::vector<int64_t>& shape, const size_t batch1_byte_size,
        std::vector<Scheduler::Payload>* payloads);

    // Run model to execute for one or more requests. This function
    // assumes that it is only called by the single runner thread that
    // is assigned to this context. A non-OK return status indicates
    // an internal error that prevents any of the of requests from
    // completing. If an error is isolate to a single request payload
    // it will be reported in that payload.
    Status Run(
        const BaseBackend* base, std::vector<Scheduler::Payload>* payloads);

    // Map from configuration name for an input to tensor name for
    // that input in the model.
    IONameMap input_name_map_;

    // Map from configuration name for an output to tensor name for
    // that output in the model.
    IONameMap output_name_map_;

    // TRTISTFModel for this context.
    TRTISTFModelHandle trtistf_model_;
  };

 private:
  // Run model on the context associated with 'runner_idx' to
  // execute for one or more requests.
  void Run(
      uint32_t runner_idx, std::vector<Scheduler::Payload>* payloads,
      std::function<void(Status)> OnCompleteQueuedPayloads);

 private:
  DISALLOW_COPY_AND_ASSIGN(BaseBackend);
  friend std::ostream& operator<<(std::ostream&, const BaseBackend&);

  // The contexts for this backend.
  std::vector<std::unique_ptr<Context>> contexts_;
};

std::ostream& operator<<(std::ostream& out, const BaseBackend& pb);

}}  // namespace nvidia::inferenceserver
