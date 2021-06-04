// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "model_config.pb.h"
#include "src/backends/backend/triton_model.h"
#include "src/core/logging.h"
#include "src/core/metrics.h"
#include "src/core/numa_utils.h"
#include "src/core/shared_library.h"

namespace nvidia { namespace inferenceserver {

TritonModelInstance::TritonModelInstance(
    TritonModel* model, const std::string& name, const size_t index,
    const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
    const std::vector<std::string>& profile_names, const bool passive,
    const HostPolicyCmdlineConfig& host_policy,
    const TritonServerMessage& host_policy_message)
    : model_(model), name_(name), index_(index), kind_(kind),
      device_id_(device_id), host_policy_(host_policy),
      host_policy_message_(host_policy_message), profile_names_(profile_names),
      passive_(passive), state_(nullptr)
{
#ifdef TRITON_ENABLE_METRICS
  if (Metrics::Enabled()) {
    // Use an ID in the metric only for GPU instances. Otherwise use
    // -1 to indicate no device should be reported in the metric.
    const int id =
        (kind_ == TRITONSERVER_INSTANCEGROUPKIND_GPU) ? device_id_ : -1;
    MetricModelReporter::Create(
        model_->Name(), model_->Version(), id, model_->Config().metric_tags(),
        &reporter_);
  }
#endif  // TRITON_ENABLE_METRICS
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
    TritonModel* model, const HostPolicyCmdlineConfigMap& host_policy_map,
    const inference::ModelConfig& model_config)
{
  static HostPolicyCmdlineConfig empty_host_policy;
  for (const auto& group : model_config.instance_group()) {
    std::vector<std::string> profile_names;
    for (const auto& profile_name : group.profile()) {
      profile_names.push_back(profile_name);
    }
    for (int32_t c = 0; c < group.count(); ++c) {
      std::string instance_name{group.count() > 1
                                    ? group.name() + "_" + std::to_string(c)
                                    : group.name()};
      const bool passive = group.passive();
      std::vector<
          std::tuple<std::string, TRITONSERVER_InstanceGroupKind, int32_t>>
          instance_setting;
      if (group.kind() == inference::ModelInstanceGroup::KIND_CPU) {
        instance_setting.emplace_back(
            group.host_policy().empty() ? "cpu" : group.host_policy(),
            TRITONSERVER_INSTANCEGROUPKIND_CPU, 0 /* device_id */);
      } else if (group.kind() == inference::ModelInstanceGroup::KIND_GPU) {
        for (const int32_t device_id : group.gpus()) {
          instance_setting.emplace_back(
              group.host_policy().empty() ? ("gpu_" + std::to_string(device_id))
                                          : group.host_policy(),
              TRITONSERVER_INSTANCEGROUPKIND_GPU, device_id);
        }
      } else if (group.kind() == inference::ModelInstanceGroup::KIND_MODEL) {
        instance_setting.emplace_back(
            group.host_policy().empty() ? "model" : group.host_policy(),
            TRITONSERVER_INSTANCEGROUPKIND_MODEL, 0 /* device_id */);
      } else {
        return Status(
            Status::Code::INVALID_ARG,
            std::string("instance_group kind ") +
                ModelInstanceGroup_Kind_Name(group.kind()) + " not supported");
      }
      for (const auto is : instance_setting) {
        const std::string& policy_name = std::get<0>(is);
        const HostPolicyCmdlineConfig* host_policy;
        const auto policy_it = host_policy_map.find(policy_name);
        if (policy_it != host_policy_map.end()) {
          host_policy = &policy_it->second;
        } else {
          host_policy = &empty_host_policy;
        }
        RETURN_IF_ERROR(SetNumaConfigOnThread(*host_policy));
        auto err = CreateInstance(
            model, instance_name, c, std::get<1>(is), std::get<2>(is),
            profile_names, passive, policy_name, *host_policy);
        RETURN_IF_ERROR(ResetNumaMemoryPolicy());
        RETURN_IF_ERROR(err);
      }
    }
  }

  return Status::Success;
}

Status
TritonModelInstance::CreateInstance(
    TritonModel* model, const std::string& name, const size_t index,
    const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
    const std::vector<std::string>& profile_names, const bool passive,
    const std::string& host_policy_name,
    const HostPolicyCmdlineConfig& host_policy)
{
  // Create the JSON representation of the backend configuration.
  triton::common::TritonJson::Value host_policy_json(
      triton::common::TritonJson::ValueType::OBJECT);
  if (!host_policy.empty()) {
    triton::common::TritonJson::Value policy_setting_json(
        host_policy_json, triton::common::TritonJson::ValueType::OBJECT);
    for (const auto& pr : host_policy) {
      RETURN_IF_ERROR(
          policy_setting_json.AddString(pr.first.c_str(), pr.second));
    }

    RETURN_IF_ERROR(host_policy_json.Add(
        host_policy_name.c_str(), std::move(policy_setting_json)));
  }
  TritonServerMessage host_policy_message(host_policy_json);

  std::unique_ptr<TritonModelInstance> local_instance(new TritonModelInstance(
      model, name, index, kind, device_id, profile_names, passive, host_policy,
      host_policy_message));

  TRITONBACKEND_ModelInstance* triton_instance =
      reinterpret_cast<TRITONBACKEND_ModelInstance*>(local_instance.get());

  // Instance initialization is optional... We must set set shared
  // library path to point to the backend directory in case the
  // backend library attempts to load additional shared libaries.
  if (model->Backend()->ModelInstanceInitFn() != nullptr) {
    std::unique_ptr<SharedLibrary> slib;
    RETURN_IF_ERROR(SharedLibrary::Acquire(&slib));
    RETURN_IF_ERROR(slib->SetLibraryDirectory(model->Backend()->Directory()));

    TRITONSERVER_Error* err =
        model->Backend()->ModelInstanceInitFn()(triton_instance);

    RETURN_IF_ERROR(slib->ResetLibraryDirectory());
    RETURN_IF_TRITONSERVER_ERROR(err);
  }

  model->AddInstance(std::move(local_instance), passive);

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
TRITONBACKEND_ModelInstanceHostPolicy(
    TRITONBACKEND_ModelInstance* instance, TRITONSERVER_Message** host_policy)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *host_policy = const_cast<TRITONSERVER_Message*>(
      reinterpret_cast<const TRITONSERVER_Message*>(&ti->HostPolicyMessage()));
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceProfileCount(
    TRITONBACKEND_ModelInstance* instance, uint32_t* count)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *count = ti->Profiles().size();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceProfileName(
    TRITONBACKEND_ModelInstance* instance, const uint32_t index,
    const char** profile_name)
{
  *profile_name = nullptr;

  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  const auto& rprofiles = ti->Profiles();
  if (index >= rprofiles.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("out of bounds index ") + std::to_string(index) +
         ": instance is configured with " + std::to_string(rprofiles.size()) +
         " profiles")
            .c_str());
  }

  *profile_name = rprofiles[index].c_str();

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceIsPassive(
    TRITONBACKEND_ModelInstance* instance, bool* is_passive)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *is_passive = ti->IsPassive();
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

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceReportStatistics(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request* request,
    const bool success, const uint64_t exec_start_ns,
    const uint64_t compute_start_ns, const uint64_t compute_end_ns,
    const uint64_t exec_end_ns)
{
#ifdef TRITON_ENABLE_STATS
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  tr->ReportStatistics(
      ti->MetricReporter(), success, exec_start_ns, compute_start_ns,
      compute_end_ns, exec_end_ns);
#endif  // TRITON_ENABLE_STATS

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceReportBatchStatistics(
    TRITONBACKEND_ModelInstance* instance, const uint64_t batch_size,
    const uint64_t exec_start_ns, const uint64_t compute_start_ns,
    const uint64_t compute_end_ns, const uint64_t exec_end_ns)
{
#ifdef TRITON_ENABLE_STATS
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  ti->Model()->MutableStatsAggregator()->UpdateInferBatchStats(
      ti->MetricReporter(), batch_size, exec_start_ns, compute_start_ns,
      compute_end_ns, exec_end_ns);
#endif  // TRITON_ENABLE_STATS

  return nullptr;  // success
}

}  // extern C

}}  // namespace nvidia::inferenceserver
