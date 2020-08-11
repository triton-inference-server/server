// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <string>
#include "src/backends/backend/tritonbackend.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace nvidia { namespace inferenceserver { namespace backend {

#ifndef TRITON_ENABLE_GPU
using cudaStream_t = void*;
#endif  // !TRITON_ENABLE_GPU

class BackendModel;

//
// BackendModelInstance
//
// Common functionality for a backend model instance. This class is
// provided as a convenience; backends are not required to use this
// class.
//
class BackendModelInstance {
 public:
  BackendModelInstance(
      BackendModel* backend_model,
      TRITONBACKEND_ModelInstance* triton_model_instance);
  virtual ~BackendModelInstance();

  // Get the name, kind and device ID of the instance.
  const std::string& Name() const { return name_; }
  TRITONSERVER_InstanceGroupKind Kind() const { return kind_; }
  int32_t DeviceId() const { return device_id_; }

  // Get the handle to the TRITONBACKEND model instance.
  TRITONBACKEND_ModelInstance* TritonModelInstance()
  {
    return triton_model_instance_;
  }

  // Get the BackendModel representing the model that corresponds to
  // this instance.
  BackendModel* Model() const { return backend_model_; }

  // The model configuration 'default_model_filename' value, or the
  // value in model configuration 'cc_model_filenames' for the GPU
  // targeted by this instance. If neither are specified in the model
  // configuration, the return empty string.
  const std::string& ArtifactFilename() const { return artifact_filename_; }

  // Returns the stream associated with this instance that can be used
  // for GPU<->CPU memory transfers. Returns nullptr if GPU support is
  // disabled or if this instance is not executing on a GPU.
  cudaStream_t CudaStream() { return stream_; }

 protected:
  BackendModel* backend_model_;
  TRITONBACKEND_ModelInstance* triton_model_instance_;

  std::string name_;
  TRITONSERVER_InstanceGroupKind kind_;
  int32_t device_id_;

  std::string artifact_filename_;
  cudaStream_t stream_;
};

//
// BackendModelInstanceException
//
// Exception thrown if error occurs while constructing an
// BackendModelInstance.
//
struct BackendModelInstanceException {
  BackendModelInstanceException(TRITONSERVER_Error* err) : err_(err) {}
  TRITONSERVER_Error* err_;
};

#define THROW_IF_BACKEND_INSTANCE_ERROR(X)                                   \
  do {                                                                       \
    TRITONSERVER_Error* tie_err__ = (X);                                     \
    if (tie_err__ != nullptr) {                                              \
      throw nvidia::inferenceserver::backend::BackendModelInstanceException( \
          tie_err__);                                                        \
    }                                                                        \
  } while (false)

}}}  // namespace nvidia::inferenceserver::backend
