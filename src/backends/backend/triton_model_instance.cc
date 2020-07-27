// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "src/backends/backend/triton_model_instance.h"

#include "src/backends/backend/triton_model.h"
#include "src/core/logging.h"
#include "src/core/model_config.pb.h"

namespace nvidia { namespace inferenceserver {

TritonModelInstance::TritonModelInstance(
    TritonModel* model, const std::string& name, const size_t index,
    const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id)
    : model_(model), name_(name), index_(index), kind_(kind),
      device_id_(device_id), state_(nullptr)
{
}

TritonModelInstance::~TritonModelInstance()
{
  // Model finalization is optional...
  if (model_->Backend()->ModelInstanceFiniFn() != nullptr) {
    LOG_TRITONSERVER_ERROR(
        model_->Backend()->ModelInstanceFiniFn()(
            reinterpret_cast<TRITONBACKEND_ModelInstance*>(this)),
        "failed finalizing model instance");
  }
}

Status
TritonModelInstance::CreateInstances(
    TritonModel* model, const ModelConfig& model_config,
    std::vector<std::unique_ptr<TritonModelInstance>>* instances)
{
  for (const auto& group : model_config.instance_group()) {
    for (int32_t c = 0; c < group.count(); ++c) {
      if (group.kind() == ModelInstanceGroup::KIND_CPU) {
        RETURN_IF_ERROR(CreateInstance(
            model, group.name(), c, TRITONSERVER_INSTANCEGROUPKIND_CPU,
            0 /* device_id */, instances));
      } else if (group.kind() == ModelInstanceGroup::KIND_GPU) {
        for (const int32_t device_id : group.gpus()) {
          RETURN_IF_ERROR(CreateInstance(
              model, group.name(), c, TRITONSERVER_INSTANCEGROUPKIND_GPU,
              device_id, instances));
        }
      } else {
        return Status(
            Status::Code::INVALID_ARG,
            std::string("instance_group ") + group.name() + " not supported");
      }
    }
  }

  return Status::Success;
}

Status
TritonModelInstance::CreateInstance(
    TritonModel* model, const std::string& name, const size_t index,
    const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
    std::vector<std::unique_ptr<TritonModelInstance>>* instances)
{
  std::unique_ptr<TritonModelInstance> local_instance(
      new TritonModelInstance(model, name, index, kind, device_id));

  TRITONBACKEND_ModelInstance* triton_instance =
      reinterpret_cast<TRITONBACKEND_ModelInstance*>(local_instance.get());

  // Instance initialization is optional...
  if (model->Backend()->ModelInstanceInitFn() != nullptr) {
    RETURN_IF_TRITONSERVER_ERROR(
        model->Backend()->ModelInstanceInitFn()(triton_instance));
  }

  instances->push_back(std::move(local_instance));

  return Status::Success;
}

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceName(
    TRITONBACKEND_ModelInstance* instance, const char** name)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *name = ti->Name().c_str();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceKind(
    TRITONBACKEND_ModelInstance* instance, TRITONSERVER_InstanceGroupKind* kind)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *kind = ti->Kind();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceDeviceId(
    TRITONBACKEND_ModelInstance* instance, int32_t* device_id)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *device_id = ti->DeviceId();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceModel(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Model** model)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *model = reinterpret_cast<TRITONBACKEND_Model*>(ti->Model());
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceState(
    TRITONBACKEND_ModelInstance* instance, void** state)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *state = ti->State();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceSetState(
    TRITONBACKEND_ModelInstance* instance, void* state)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  ti->SetState(state);
  return nullptr;  // success
}

}  // extern C

}}  // namespace nvidia::inferenceserver
