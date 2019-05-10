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

#include <ATen/Tensor.h>
#include <ATen/ATen.h>
#include <ATen/dlpack.h>
#include <ATen/Functions.h>

namespace nvidia { namespace inferenceserver {

// To avoid namespace and protobuf collision between TensorFlow and
// Torch, we keep Torch interface isolated to torch_backend_c2. The
// interface to those isolated functions is provided by
// LibTorchWorkspace.
class LibTorchWorkspace {
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
  // LibTorchWorkspace boundary so need to have this non-protobuf
  // definition.
  typedef enum {
    kDLCPU = -1,
    kDLGPU = 0,
    kDLCPUPinned = 3,
    kDLOpenCL = 4,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
  } DeviceType;

  typedef enum {
    kDLInt = 0U,
    kDLUInt = 1U,
    kDLFloat = 2U,
    Invalid = 3U,
  } DataTypeCode;

  typedef struct {
    // The value should be one of DLDataTypeCode enum values
    uint8_t code;
    uint8_t bits;
    // Number of lanes in the type, used for vector types
    uint16_t lanes;
  } DataType;

  typedef struct {
    DeviceType device_type;
    /* The device index */
    int device_id;
  } DeviceContext;

  virtual ~LibTorchWorkspace() = default;

  // Return names of all possible inputs and outputs for the
  // model. These are names reported by the model netdef itself as
  // external inputs and outputs.
  virtual const std::set<std::string>& PotentialInputNames() const = 0;
  virtual const std::set<std::string>& PotentialOutputNames() const = 0;

  // Set the value for an input tensor in preparation for inferencing.
  virtual Error SetInputTensor(
      const std::string& name, const std::vector<int64_t>& shape,
      const DataType dtype, const char* content, size_t byte_size) = 0;

  // Get the value for an output tensor after inferencing.
  virtual Error GetOutputTensor(
      const std::string& name, const LibTorchWorkspace::DataType dtype,
      const char** content, size_t* byte_size,
      std::vector<int64_t>* content_shape) = 0;

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

// Create a LibTorchWorkspace that interfaces with the LibTorch library
// for a model specified by its model path.
CAFFE2WS_EXPORT LibTorchWorkspace::Error LibTorchWorkspaceCreate(
    LibTorchWorkspace** c2ws, const std::string& model_name,
    const int max_batch_size, const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names, const int gpu_device,
    const std::string torch_model_path);

}  // extern "C"

}}  // namespace nvidia::inferenceserver
