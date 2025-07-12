// Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "metric_family.h"

#ifdef TRITON_PB_STUB
#include "pb_stub.h"
#endif

namespace triton { namespace backend { namespace python {

MetricFamily::MetricFamily(
    const std::string& name, const std::string& description,
    const MetricKind& kind)
    : name_(name), description_(description), kind_(kind),
      metric_family_address_(nullptr)
{
#ifdef TRITON_PB_STUB
  SendCreateMetricFamilyRequest();
#endif
}

MetricFamily::~MetricFamily()
{
#ifdef TRITON_PB_STUB
  // Clear all the metrics first
  {
    std::lock_guard<std::mutex> lock(metric_map_mu_);
    for (auto& m : metric_map_) {
      m.second->Clear();
    }
  }

  // Send the request to delete the MetricFamily to the parent process
  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
  SaveToSharedMemory(stub->ShmPool());
  AllocatedSharedMemory<CustomMetricsMessage> custom_metrics_shm;
  try {
    stub->SendMessage<CustomMetricsMessage>(
        custom_metrics_shm, PYTHONSTUB_MetricFamilyRequestDelete, shm_handle_);
  }
  catch (const PythonBackendException& pb_exception) {
    std::cerr << "Error when deleting MetricFamily: " << pb_exception.what()
              << "\n";
  }
#endif
};

void
MetricFamily::SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool)
{
  AllocatedSharedMemory<MetricFamilyShm> custom_metric_family_shm =
      shm_pool->Construct<MetricFamilyShm>();

  custom_metric_family_shm_ptr_ = custom_metric_family_shm.data_.get();
  std::unique_ptr<PbString> name_shm = PbString::Create(shm_pool, name_);
  std::unique_ptr<PbString> description_shm =
      PbString::Create(shm_pool, description_);

  custom_metric_family_shm_ptr_->kind = kind_;
  custom_metric_family_shm_ptr_->name_shm_handle = name_shm->ShmHandle();
  custom_metric_family_shm_ptr_->description_shm_handle =
      description_shm->ShmHandle();
  custom_metric_family_shm_ptr_->metric_family_address = metric_family_address_;

  // Save the references to shared memory.
  custom_metric_family_shm_ = std::move(custom_metric_family_shm);
  name_shm_ = std::move(name_shm);
  description_shm_ = std::move(description_shm);
  shm_handle_ = custom_metric_family_shm_.handle_;
}

std::unique_ptr<MetricFamily>
MetricFamily::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t handle)
{
  AllocatedSharedMemory<MetricFamilyShm> custom_metric_family_shm =
      shm_pool->Load<MetricFamilyShm>(handle);
  MetricFamilyShm* custom_metric_family_shm_ptr =
      custom_metric_family_shm.data_.get();
  std::unique_ptr<PbString> name_shm = PbString::LoadFromSharedMemory(
      shm_pool, custom_metric_family_shm_ptr->name_shm_handle);
  std::unique_ptr<PbString> description_shm = PbString::LoadFromSharedMemory(
      shm_pool, custom_metric_family_shm_ptr->description_shm_handle);

  return std::unique_ptr<MetricFamily>(
      new MetricFamily(custom_metric_family_shm, name_shm, description_shm));
}

MetricFamily::MetricFamily(
    AllocatedSharedMemory<MetricFamilyShm>& custom_metric_family_shm,
    std::unique_ptr<PbString>& name_shm,
    std::unique_ptr<PbString>& description_shm)
    : custom_metric_family_shm_(std::move(custom_metric_family_shm)),
      name_shm_(std::move(name_shm)),
      description_shm_(std::move(description_shm))
{
  custom_metric_family_shm_ptr_ = custom_metric_family_shm_.data_.get();
  name_ = name_shm_->String();
  description_ = description_shm_->String();
  kind_ = custom_metric_family_shm_ptr_->kind;
  metric_family_address_ = custom_metric_family_shm_ptr_->metric_family_address;
}

void*
MetricFamily::MetricFamilyAddress()
{
  return metric_family_address_;
}

#ifdef TRITON_PB_STUB
std::shared_ptr<MetricFamily>
MetricFamily::CreateMetricFamily(
    const std::string& name, const std::string& description,
    const MetricKind& kind)
{
  std::shared_ptr<MetricFamily> metric_family =
      std::make_shared<MetricFamily>(name, description, kind);
  metric_family->SendCreateMetricFamilyRequest();
  return metric_family;
}

void
MetricFamily::SendCreateMetricFamilyRequest()
{
  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
  SaveToSharedMemory(stub->ShmPool());
  CustomMetricsMessage* custom_metrics_msg = nullptr;
  AllocatedSharedMemory<CustomMetricsMessage> custom_metrics_shm;
  try {
    stub->SendMessage<CustomMetricsMessage>(
        custom_metrics_shm, PYTHONSTUB_MetricFamilyRequestNew, shm_handle_);
  }
  catch (const PythonBackendException& pb_exception) {
    throw PythonBackendException(
        "Error when creating MetricFamily: " +
        std::string(pb_exception.what()));
  }

  custom_metrics_msg = custom_metrics_shm.data_.get();
  metric_family_address_ = custom_metrics_msg->address;
}

std::shared_ptr<Metric>
MetricFamily::CreateMetric(const py::object& labels, const py::object& buckets)
{
  if (!labels.is_none()) {
    if (!py::isinstance<py::dict>(labels)) {
      throw PythonBackendException(
          "Failed to create metric. Labels must be a dictionary.");
    }
  }

  py::module json = py::module_::import("json");
  std::string labels_str = std::string(py::str(json.attr("dumps")(labels)));

  std::optional<std::vector<double>> buckets_vec;
  if (!buckets.is_none()) {
    if (!py::isinstance<py::list>(buckets)) {
      throw PythonBackendException(
          "Failed to create metric. Buckets must be a list.");
    }
    if (kind_ == kCounter || kind_ == kGauge) {
      throw PythonBackendException(
          "Failed to create metric. Unexpected buckets found.");
    }
    buckets_vec = buckets.cast<std::vector<double>>();
  } else {
    if (kind_ == kHistogram) {
      throw PythonBackendException(
          "Failed to create metric. Missing required buckets.");
    }
    buckets_vec = std::nullopt;
  }

  auto metric =
      std::make_shared<Metric>(labels_str, buckets_vec, metric_family_address_);
  {
    std::lock_guard<std::mutex> lock(metric_map_mu_);
    metric_map_.insert({metric->MetricAddress(), metric});
  }

  return metric;
}
#else
void*
MetricFamily::InitializeTritonMetricFamily()
{
  TRITONSERVER_MetricKind triton_kind = ToTritonServerMetricKind(kind_);
  TRITONSERVER_MetricFamily* triton_metric_family = nullptr;
  THROW_IF_TRITON_ERROR(TRITONSERVER_MetricFamilyNew(
      &triton_metric_family, triton_kind, name_.c_str(), description_.c_str()));
  return reinterpret_cast<void*>(triton_metric_family);
}

TRITONSERVER_MetricKind
MetricFamily::ToTritonServerMetricKind(const MetricKind& kind)
{
  switch (kind) {
    case kCounter:
      return TRITONSERVER_METRIC_KIND_COUNTER;
    case kGauge:
      return TRITONSERVER_METRIC_KIND_GAUGE;
    case kHistogram:
      return TRITONSERVER_METRIC_KIND_HISTOGRAM;
    default:
      throw PythonBackendException("Unknown metric kind");
  }
}

void
MetricFamily::ClearTritonMetricFamily()
{
  auto metric_family =
      reinterpret_cast<TRITONSERVER_MetricFamily*>(metric_family_address_);
  if (metric_family != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_MetricFamilyDelete(metric_family),
        "deleting metric family");
  }
}
#endif

}}}  // namespace triton::backend::python
