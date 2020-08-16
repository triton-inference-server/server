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

#include "src/backends/backend/examples/backend_model_instance.h"

#include <vector>
#include "src/backends/backend/examples/backend_model.h"
#include "src/backends/backend/examples/backend_utils.h"

namespace nvidia { namespace inferenceserver { namespace backend {

#define THROW_IF_ERROR(X)                             \
  do {                                                \
    TRITONSERVER_Error* tie_err__ = (X);              \
    if (tie_err__ != nullptr) {                       \
      throw BackendModelInstanceException(tie_err__); \
    }                                                 \
  } while (false)

//
// BackendModelInstance
//
BackendModelInstance::BackendModelInstance(
    BackendModel* backend_model,
    TRITONBACKEND_ModelInstance* triton_model_instance)
    : backend_model_(backend_model),
      triton_model_instance_(triton_model_instance)
{
  const char* instance_name;
  THROW_IF_ERROR(
      TRITONBACKEND_ModelInstanceName(triton_model_instance, &instance_name));
  name_ = instance_name;

  THROW_IF_ERROR(
      TRITONBACKEND_ModelInstanceKind(triton_model_instance, &kind_));

  THROW_IF_ERROR(
      TRITONBACKEND_ModelInstanceDeviceId(triton_model_instance, &device_id_));

  TritonJson::Value& model_config = backend_model->ModelConfig();

  // If the model configuration specifies a 'default_model_filename'
  // and/or specifies 'cc_model_filenames' then determine the
  // appropriate 'artifact_filename' value. If model configuration
  // does not specify then just leave 'artifact_filename' empty and
  // the backend can then provide its own logic for determine the
  // filename if that is appropriate.
  THROW_IF_ERROR(model_config.MemberAsString(
      "default_model_filename", &artifact_filename_));

  switch (kind_) {
    case TRITONSERVER_INSTANCEGROUPKIND_CPU: {
      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string("Creating instance ") + name_ +
           " on CPU using artifact '" + artifact_filename_ + "'")
              .c_str());
      break;
    }
    case TRITONSERVER_INSTANCEGROUPKIND_MODEL: {
      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string("Creating instance ") + name_ +
           " on model-specified devices using artifact '" + artifact_filename_ +
           "'")
              .c_str());
      break;
    }
    case TRITONSERVER_INSTANCEGROUPKIND_GPU: {
#ifdef TRITON_ENABLE_GPU
      cudaDeviceProp cuprops;
      cudaError_t cuerr = cudaGetDeviceProperties(&cuprops, device_id_);
      if (cuerr != cudaSuccess) {
        throw BackendModelInstanceException(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("unable to get CUDA device properties for ") + name_ +
             ": " + cudaGetErrorString(cuerr))
                .c_str()));
      }

      const std::string cc =
          std::to_string(cuprops.major) + "." + std::to_string(cuprops.minor);
      TritonJson::Value cc_names;
      TritonJson::Value cc_name;
      if ((model_config.Find("cc_model_filenames", &cc_names)) &&
          (cc_names.Find(cc.c_str(), &cc_name))) {
        cc_name.AsString(&artifact_filename_);
      }

      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string("Creating instance ") + name_ + " on GPU " +
           std::to_string(device_id_) + " (" + cc + ") using artifact '" +
           artifact_filename_ + "'")
              .c_str());
#else
      throw BackendModelInstanceException(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "GPU instances not supported"));
#endif  // TRITON_ENABLE_GPU
      break;
    }
    default: {
      throw BackendModelInstanceException(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unexpected instance kind for ") + name_).c_str()));
    }
  }

  stream_ = nullptr;
  if (kind_ == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    THROW_IF_ERROR(
        CreateCudaStream(device_id_, 0 /* cuda_stream_priority */, &stream_));
  }
}


BackendModelInstance::~BackendModelInstance()
{
#ifdef TRITON_ENABLE_GPU
  if (stream_ != nullptr) {
    cudaError_t err = cudaStreamDestroy(stream_);
    if (err != cudaSuccess) {
      TRITONSERVER_LogMessage(
          TRITONSERVER_LOG_ERROR, __FILE__, __LINE__,
          (std::string("~BackendModelInstance: ") + name_ +
           " failed to destroy cuda stream: " + cudaGetErrorString(err))
              .c_str());
    }
    stream_ = nullptr;
  }
#endif  // TRITON_ENABLE_GPU
}

}}}  // namespace nvidia::inferenceserver::backend
