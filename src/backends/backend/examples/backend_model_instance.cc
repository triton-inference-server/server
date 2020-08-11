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

//
// BackendModelInstance
//
TRITONSERVER_Error*
BackendModelInstance::Create(
    BackendModel* backend_model,
    TRITONBACKEND_ModelInstance* triton_model_instance,
    BackendModelInstance** backend_model_instance)
{
  const char* instance_name;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceName(triton_model_instance, &instance_name));

  TRITONSERVER_InstanceGroupKind instance_kind;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceKind(triton_model_instance, &instance_kind));

  int32_t instance_id;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceDeviceId(triton_model_instance, &instance_id));

  TritonJson::Value& model_config = backend_model->ModelConfig();

  // If the model configuration specifies a 'default_model_filename'
  // and/or specifies 'cc_model_filenames' then determine the
  // appropriate 'artifact_filename' value. If model configuration
  // does not specify then just leave 'artifact_filename' empty and
  // the backend can then provide its own logic for determine the
  // filename if that is appropriate.
  std::string cc_model_filename;
  RETURN_IF_ERROR(model_config.MemberAsString(
      "default_model_filename", &cc_model_filename));

  switch (instance_kind) {
    case TRITONSERVER_INSTANCEGROUPKIND_CPU: {
      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string("Creating instance ") + instance_name +
           " on CPU using artifact '" + cc_model_filename + "'")
              .c_str());
      break;
    }
    case TRITONSERVER_INSTANCEGROUPKIND_MODEL: {
      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string("Creating instance ") + instance_name +
           " on model-specified devices using artifact '" + cc_model_filename +
           "'")
              .c_str());
      break;
    }
    case TRITONSERVER_INSTANCEGROUPKIND_GPU: {
#ifdef TRITON_ENABLE_GPU
      cudaDeviceProp cuprops;
      cudaError_t cuerr = cudaGetDeviceProperties(&cuprops, instance_id);
      if (cuerr != cudaSuccess) {
        RETURN_ERROR_IF_FALSE(
            false, TRITONSERVER_ERROR_INTERNAL,
            std::string("unable to get CUDA device properties for ") +
                instance_name + ": " + cudaGetErrorString(cuerr));
      }

      const std::string cc =
          std::to_string(cuprops.major) + "." + std::to_string(cuprops.minor);
      TritonJson::Value cc_names;
      TritonJson::Value cc_name;
      if ((model_config.Find("cc_model_filenames", &cc_names)) &&
          (cc_names.Find(cc.c_str(), &cc_name))) {
        cc_name.AsString(&cc_model_filename);
      }

      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string("Creating instance ") + instance_name + " on GPU " +
           std::to_string(instance_id) + " (" + cc + ") using artifact '" +
           cc_model_filename + "'")
              .c_str());
#else
      RETURN_ERROR_IF_FALSE(
          false, TRITONSERVER_ERROR_INTERNAL,
          std::string("GPU instances not supported"));
#endif  // TRITON_ENABLE_GPU
      break;
    }
    default: {
      RETURN_ERROR_IF_FALSE(
          false, TRITONSERVER_ERROR_INTERNAL,
          std::string("unexpected instance kind for ") + instance_name);
    }
  }

  cudaStream_t stream = nullptr;
  if (instance_kind == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    RETURN_IF_ERROR(
        CreateCudaStream(instance_id, 0 /* cuda_stream_priority */, &stream));
  }

  *backend_model_instance = new BackendModelInstance(
      backend_model, triton_model_instance, instance_name, instance_kind,
      instance_id, cc_model_filename, stream);
  return nullptr;  // success
}


BackendModelInstance::BackendModelInstance(
    BackendModel* backend_model,
    TRITONBACKEND_ModelInstance* triton_model_instance, const char* name,
    const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
    const std::string& artifact_filename, cudaStream_t stream)
    : backend_model_(backend_model),
      triton_model_instance_(triton_model_instance), name_(name), kind_(kind),
      device_id_(device_id), artifact_filename_(artifact_filename),
      stream_(stream)
{
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
