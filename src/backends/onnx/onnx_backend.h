// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <onnxruntime_c_api.h>
#include "src/core/backend.h"
#include "src/core/backend_context.h"
#include "src/core/model_config.pb.h"
#include "src/core/scheduler.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class OnnxBackend : public InferenceBackend {
 public:
  OnnxBackend() = default;
  OnnxBackend(OnnxBackend&&) = default;

  // Create a context for execution for each instance for the
  // serialized plans specified in 'models'.
  Status CreateExecutionContexts(
      const std::unordered_map<std::string, std::string>& paths);
  Status CreateExecutionContext(
      const std::string& instance_name, const int gpu_device,
      OrtSessionOptions* base_session_options,
      const std::unordered_map<std::string, std::string>& paths);

 private:
  // Helper function for CreateExecutionContexts() so that session_options
  // will be released properly regardless of possible errors
  Status CreateExecutionContextsHelper(
      OrtSessionOptions* session_options,
      const std::unordered_map<std::string, std::string>& paths);

 private:
  DISALLOW_COPY_AND_ASSIGN(OnnxBackend);
  friend std::ostream& operator<<(std::ostream&, const OnnxBackend&);

  // For each model instance there is a context.
  struct Context : BackendContext {
    Context(
        const std::string& name, const int gpu_device,
        const int max_batch_size);
    ~Context();

    DISALLOW_MOVE(Context);
    DISALLOW_COPY_AND_ASSIGN(Context);

    Status ValidateInputs(
        const std::string& model_name,
        const ::google::protobuf::RepeatedPtrField<ModelInput>& ios,
        const size_t expected_input_cnt);
    Status ValidateOutputs(
        const std::string& model_name,
        const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios);
    Status ValidateBooleanSequenceControl(
        const std::string& model_name, const ModelSequenceBatching& batcher,
        const ModelSequenceBatching::Control::Kind control_kind, bool required,
        bool* have_control);
    Status ValidateTypedSequenceControl(
        const std::string& model_name, const ModelSequenceBatching& batcher,
        const ModelSequenceBatching::Control::Kind control_kind, bool required,
        bool* have_control);

    // See BackendContext::Run()
    Status Run(
        const InferenceBackend* base,
        std::vector<Scheduler::Payload>* payloads);

    // Set an input tensor from one or more payloads.
    Status SetInputTensor(
        const std::string& name, const DataType data_type, const DimsList& dims,
        size_t total_batch_size, std::vector<Scheduler::Payload>* payloads,
        std::vector<std::unique_ptr<char[]>>* input_buffers,
        std::vector<const char*>* input_names);

    // Helper function to modify 'input_buffer' into format needed for creating
    // Onnx String tensor and to set meta data 'string_data'
    void SetStringInputBuffer(
        const std::string& name, const std::vector<size_t>& expected_byte_sizes,
        const std::vector<size_t>& expected_element_cnts,
        std::vector<Scheduler::Payload>* payloads, char* input_buffer,
        std::vector<const char*>* string_data);

    // Helper function to fill 'string_data' with 'cnt' number of empty string
    void FillStringData(std::vector<const char*>* string_data, size_t cnt);

    // Read output tensors into one or more payloads accordingly.
    Status ReadOutputTensors(
        const InferenceBackend* base, const size_t total_batch_size,
        const std::vector<const char*>& output_names,
        std::vector<Scheduler::Payload>* payloads);

    // Helper function to set output buffer of string data type to payloads.
    // Return true if cudaMemcpyAsync is called, and the caller should call
    // cudaStreamSynchronize before using the data. Otherwise, return false.
    bool SetStringOutputBuffer(
        const std::string& name, const size_t batch1_element_cnt,
        const char* content, const std::vector<int64_t>& content_shape,
        const size_t* offsets, std::vector<Scheduler::Payload>* payloads);

    // Release the Onnx Runtime resources allocated for the run, if any.
    void ReleaseOrtRunResources();

    // Onnx Runtime variables that are used across runs
    OrtSession* session_;
    OrtAllocator* allocator_;

    // Onnx Runtime variables that will be reset and used for every run
    std::vector<OrtValue*> input_tensors_;
    std::vector<OrtValue*> output_tensors_;
  };
};

std::ostream& operator<<(std::ostream& out, const OnnxBackend& pb);

}}  // namespace nvidia::inferenceserver
