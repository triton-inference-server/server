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

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace nvidia { namespace inferenceserver {

// To avoid namespace and protobuf collision between TensorFlow and
// Caffe2, we keep Caffe2 interface isolated to netdef_bundle_c2. The
// interface to those isolated functions is provided by
// Caffe2Workspace.
class Caffe2Workspace {
 public:
  // GPU device number that indicates that no gpu is available for a
  // context
  static constexpr int NO_GPU_DEVICE = -1;

  // Max batch size value that indicates batching is not supported.
  static constexpr int NO_BATCHING = 0;

  // Error reporting
  class Error {
   public:
    Error() : success_(true) {}
    Error(const std::string& msg) : success_(false), msg_(msg) {}

    bool IsOk() const { return success_; }
    const std::string& Message() const { return msg_; }

   private:
    const bool success_;
    const std::string msg_;
  };

  // Input or output datatype. Prototypes can't cross the
  // Caffe2Workspace boundary so need to have this non-protobuf
  // definition.
  enum DataType {
    TYPE_INVALID,
    TYPE_BOOL,
    TYPE_UINT8,
    TYPE_UINT16,
    TYPE_UINT32,
    TYPE_UINT64,
    TYPE_INT8,
    TYPE_INT16,
    TYPE_INT32,
    TYPE_INT64,
    TYPE_FP16,
    TYPE_FP32,
    TYPE_FP64,
    TYPE_STRING
  };

  virtual ~Caffe2Workspace() = default;

  // Return names of all possible inputs and outputs for the
  // model. These are names reported by the model netdef itself as
  // external inputs and outputs.
  virtual const std::set<std::string>& PotentialInputNames() const = 0;
  virtual const std::set<std::string>& PotentialOutputNames() const = 0;

  // The outputs of the model, as a map from the name to the size of
  // the tensor.
  virtual const std::unordered_map<std::string, size_t>& Outputs() const = 0;

  // Add an output tensor to the model.
  virtual Error AddOutputTensor(
      const std::string& name, const DataType datatype,
      const std::vector<int>& dims) = 0;

  // Set the value for an input tensor in preparation for inferencing.
  virtual Error SetInputTensor(
      const std::string& name, const std::vector<int64_t>& shape,
      const DataType dtype, const char* content, size_t byte_size) = 0;

  // Get the value for an output tensor after inferencing.
  virtual Error GetOutputTensor(
      const std::string& name, size_t batch_size, const char** content,
      size_t byte_size) = 0;

  // Run the model.
  virtual Error Run() = 0;
};

extern "C" {

#if defined(_MSC_VER)
#define CAFFE2WS_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define CAFFE2WS_EXPORT __attribute__((__visibility__("default")))
#else
#define CAFFE2WS_EXPORT
#endif

// Create a Caffe2Workspace that interfaces with the Caffe2 library
// for a model specified by an init and network blob.
CAFFE2WS_EXPORT Caffe2Workspace::Error Caffe2WorkspaceCreate(
    Caffe2Workspace** c2ws, const std::string& model_name,
    const int max_batch_size, const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names, const int gpu_device,
    const std::vector<char>& init_blob, const std::vector<char>& model_blob);

}  // extern "C"

}}  // namespace nvidia::inferenceserver
