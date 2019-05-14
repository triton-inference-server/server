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

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace nvidia { namespace inferenceserver {

// To avoid namespace and protobuf collision between TRTIS and
// TensorFlow, we keep TensorFlow interface isolated to
// tensorflow_backend_tf. The interface to those isolated functions is
// provided by TFWorkspace.
class TFWorkspace {
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

  // Input or output datatype. Protobufs can't cross the
  // TFWorkspace boundary so need to have this non-protobuf
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

  // An input or output
  struct IO {
    IO() = default;
    std::string name_;
    std::string inmodel_name_;
    DataType data_type_;
    std::vector<int64_t> shape_;
  };
  using IOList = std::vector<IO>;

  // A tensor
  struct Tensor {
    static Error Create(
        const DataType data_type, const std::vector<int64_t>& shape,
        std::unique_ptr<Tensor>* tensor);

    virtual ~Tensor() = default;

    // Return the tensor datatype.
    virtual TFWorkspace::DataType DataType() const = 0;

    // Return the size of the tensor datatype, in bytes.
    virtual int64_t DataTypeByteSize() const = 0;

    // Return the shape of the tensor, including the batch dimension.
    virtual void Shape(std::vector<int64_t>* shape) const = 0;

    // Get the base of the tensor data. Defined only for non-string
    // types.. bad things might happen if called for string type
    // tensor.
    virtual char* Base() const = 0;

    // Get the size, in bytes, of the tensor data. Defined only for
    // non-string types.. bad things might happen if called for string
    // type tensor.
    virtual size_t ByteSize() const = 0;

    // Get a string at a specified index within a tensor. Defined only
    // for string type.. bad things might happen if called for
    // non-string type tensor.
    virtual const std::string& String(size_t idx) const = 0;

    // Set a string at a specified index within a tensor. Defined only
    // for string type.. bad things might happen if called for
    // non-string type tensor.
    virtual void SetString(size_t idx, const std::string& str) = 0;
  };

  using TensorVec =
      std::vector<std::pair<std::string, std::unique_ptr<TFWorkspace::Tensor>>>;

  virtual ~TFWorkspace() = default;

  // Get the model inputs
  virtual const IOList& Inputs() const = 0;

  // Get the model outputs
  virtual const IOList& Outputs() const = 0;

  // Run the model using the provides input tensors to produce the
  // named outputs. The input tensors are destroyed as a side-effect.
  virtual TFWorkspace::Error Run(
      TFWorkspace::TensorVec* input_tensors,
      const std::vector<std::string>& output_names,
      std::vector<std::unique_ptr<TFWorkspace::Tensor>>* outputs) = 0;
};

extern "C" {

#if defined(_MSC_VER)
#define TFWS_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define TFWS_EXPORT __attribute__((__visibility__("default")))
#else
#define TFWS_EXPORT
#endif

// Create a TFWorkspace that interfaces with the Tensorflow library
// for a graphdef model.
TFWS_EXPORT TFWorkspace::Error TFWorkspaceCreateFromGraphDef(
    TFWorkspace** tfws, const std::string& model_name,
    const std::string& model_path, const int gpu_device,
    const bool has_graph_level, const int graph_level,
    const bool allow_gpu_memory_growth,
    const float per_process_gpu_memory_fraction,
    const bool allow_soft_placement);

// Create a TFWorkspace that interfaces with the Tensorflow library
// for a savedmodel model.
TFWS_EXPORT TFWorkspace::Error TFWorkspaceCreateFromSavedModel(
    TFWorkspace** tfws, const std::string& model_name,
    const std::string& model_path, const int gpu_device,
    const bool has_graph_level, const int graph_level,
    const bool allow_gpu_memory_growth,
    const float per_process_gpu_memory_fraction,
    const bool allow_soft_placement);

}  // extern "C"

}}  // namespace nvidia::inferenceserver
