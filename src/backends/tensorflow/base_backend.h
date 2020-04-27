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
  explicit BaseBackend(const double min_compute_capability)
      : InferenceBackend(min_compute_capability)
  {
  }
  BaseBackend(BaseBackend&&) = default;

  Status Init(
      const std::string& path, const ModelConfig& model_config,
      const GraphDefBackendFactory::Config* backend_config,
      const std::string& platform);

  // Create a context for execution for each instance of the
  // tensorflow model specified in 'paths'. The model can be either a
  // graphdef or savedmodel
  Status CreateExecutionContexts(
      const std::unordered_map<std::string, std::string>& paths);
  Status CreateExecutionContext(
      const std::string& instance_name, const int gpu_device,
      const std::unordered_map<std::string, std::string>& paths);

  using IONameMap = std::unordered_map<std::string, std::string>;
  using TRTISTFModelHandle =
      std::unique_ptr<TRTISTF_Model, decltype(&TRTISTF_ModelDelete)>;

  // For each model instance there is a context.
  struct Context : BackendContext {
    // GPU device number that indicates model will be loaded on GPUs
    // as specified in model graph
    static constexpr int MODEL_DEVICE = -2;

    Context(
        const std::string& name, const int gpu_device, const int max_batch_size,
        const bool enable_pinned_input, const bool enable_pinned_output);
    ~Context();

    DISALLOW_MOVE(Context);
    DISALLOW_COPY_AND_ASSIGN(Context);

    Status ValidateInputs(
        const ::google::protobuf::RepeatedPtrField<ModelInput>& ios);
    Status ValidateOutputs(
        const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios);

    // Set an input tensor data from requests.
    Status SetInput(
        const std::string& name, const DataType datatype,
        const std::vector<int64_t>& dims, const size_t total_batch_size,
        std::vector<std::unique_ptr<InferenceRequest>>* requests,
        std::vector<InputInfo>* inputs, TRTISTF_TensorList** input_tensors,
        bool* cuda_copy);

    // Helper function to set the input for fixed-sized data type
    void SetFixedSizedInputTensor(
        TRTISTF_Tensor* tensor, const std::string& input_name,
        const size_t batch1_byte_size,
        std::vector<std::unique_ptr<InferenceRequest>>* requests,
        InputInfo* input, bool* cuda_copy);

    // Helper function to set the input for String data type
    void SetStringInputTensor(
        TRTISTF_Tensor* tensor, const std::string& input_name,
        const size_t batch1_element_cnt,
        std::vector<std::unique_ptr<InferenceRequest>>* requests);

    // Helper function to set an output with a BYTES data type
    void ReadStringOutputTensor(
        TRTISTF_Tensor* tensor, const std::string& output_name,
        const std::vector<int64_t>& shape, const size_t batch1_element_cnt,
        std::vector<std::unique_ptr<InferenceRequest>>* requests,
        bool* cuda_copy);

    // See BackendContext::Run()
    void Run(
        const InferenceBackend* base,
        std::vector<std::unique_ptr<InferenceRequest>>&& requests) override;

    // Map from configuration name for an input to tensor name for
    // that input in the model.
    IONameMap input_name_map_;

    // Map from configuration name for an output to tensor name for
    // that output in the model.
    IONameMap output_name_map_;

    // TRTISTFModel for this context.
    TRTISTFModelHandle trtistf_model_;

    // use for GPU allocator
    int input_device_id_;
  };

 protected:
  // Load model and create a corresponding TRTISTF model object.
  virtual Status CreateTRTISTFModel(
      const GraphDefBackendFactory::Config* backend_config,
      const int gpu_device, const bool has_graph_level, const int graph_level,
      const std::string& model_name, const std::string& model_path,
      TRTISTFModelHandle* trtistf_model, IONameMap* input_name_map,
      IONameMap* output_name_map, const TRTISTF_TFTRTConfig* tftrt_config) = 0;

 private:
  DISALLOW_COPY_AND_ASSIGN(BaseBackend);
  friend std::ostream& operator<<(std::ostream&, const BaseBackend&);

  const GraphDefBackendFactory::Config* backend_config_;
};

std::ostream& operator<<(std::ostream& out, const BaseBackend& pb);

}}  // namespace nvidia::inferenceserver
