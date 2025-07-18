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

#include "metric.h"

#ifdef TRITON_PB_STUB
#include "pb_stub.h"
#endif

namespace triton { namespace backend { namespace python {

Metric::Metric(
    const std::string& labels, std::optional<const std::vector<double>> buckets,
    void* metric_family_address)
    : labels_(labels), buckets_(buckets), operation_value_(0),
      metric_address_(nullptr), metric_family_address_(metric_family_address),
      is_cleared_(false)
{
#ifdef TRITON_PB_STUB
  SendCreateMetricRequest();
#endif
}

Metric::~Metric()
{
#ifdef TRITON_PB_STUB
  Clear();
#endif
}

void
Metric::SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool)
{
  AllocatedSharedMemory<MetricShm> custom_metric_shm =
      shm_pool->Construct<MetricShm>();
  custom_metric_shm_ptr_ = custom_metric_shm.data_.get();

  std::unique_ptr<PbString> labels_shm = PbString::Create(shm_pool, labels_);

  custom_metric_shm_ptr_->operation_value = operation_value_;
  custom_metric_shm_ptr_->labels_shm_handle = labels_shm->ShmHandle();
  custom_metric_shm_ptr_->metric_family_address = metric_family_address_;
  custom_metric_shm_ptr_->metric_address = metric_address_;

  // Histogram specific case
  if (buckets_.has_value()) {
    auto buckets_size = buckets_.value().size() * sizeof(double);
    std::unique_ptr<PbMemory> buckets_shm = PbMemory::Create(
        shm_pool, TRITONSERVER_MemoryType::TRITONSERVER_MEMORY_CPU, 0,
        buckets_size, reinterpret_cast<char*>(buckets_.value().data()),
        false /* copy_gpu */);
    custom_metric_shm_ptr_->buckets_shm_handle = buckets_shm->ShmHandle();
    buckets_shm_ = std::move(buckets_shm);
  } else {
    custom_metric_shm_ptr_->buckets_shm_handle = 0;
    buckets_shm_ = nullptr;
  }

  // Save the references to shared memory.
  custom_metric_shm_ = std::move(custom_metric_shm);
  labels_shm_ = std::move(labels_shm);
  shm_handle_ = custom_metric_shm_.handle_;
}

std::unique_ptr<Metric>
Metric::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t handle)
{
  AllocatedSharedMemory<MetricShm> custom_metric_shm =
      shm_pool->Load<MetricShm>(handle);
  MetricShm* custom_metric_shm_ptr = custom_metric_shm.data_.get();

  std::unique_ptr<PbString> labels_shm = PbString::LoadFromSharedMemory(
      shm_pool, custom_metric_shm_ptr->labels_shm_handle);

  std::unique_ptr<PbMemory> buckets_shm = nullptr;
  if (custom_metric_shm_ptr->buckets_shm_handle != 0) {
    buckets_shm = PbMemory::LoadFromSharedMemory(
        shm_pool, custom_metric_shm_ptr->buckets_shm_handle,
        false /* open_cuda_handle */);
  }

  return std::unique_ptr<Metric>(
      new Metric(custom_metric_shm, labels_shm, buckets_shm));
}

Metric::Metric(
    AllocatedSharedMemory<MetricShm>& custom_metric_shm,
    std::unique_ptr<PbString>& labels_shm,
    std::unique_ptr<PbMemory>& buckets_shm)
    : custom_metric_shm_(std::move(custom_metric_shm)),
      labels_shm_(std::move(labels_shm)), buckets_shm_(std::move(buckets_shm))
{
  custom_metric_shm_ptr_ = custom_metric_shm_.data_.get();

  // FIXME: This constructor is called during each
  // set/increment/observe/get_value call. It only needs the pointers.
  labels_ = labels_shm_->String();
  if (buckets_shm_ != nullptr) {  // Histogram
    size_t bucket_size = buckets_shm_->ByteSize() / sizeof(double);
    std::vector<double> buckets;
    buckets.reserve(bucket_size);
    for (size_t i = 0; i < bucket_size; ++i) {
      buckets.emplace_back(
          reinterpret_cast<double*>(buckets_shm_->DataPtr())[i]);
    }
    buckets_ = std::move(buckets);
  }

  operation_value_ = custom_metric_shm_ptr_->operation_value;
  metric_family_address_ = custom_metric_shm_ptr_->metric_family_address;
  metric_address_ = custom_metric_shm_ptr_->metric_address;
}

void*
Metric::MetricAddress()
{
  return metric_address_;
}

#ifdef TRITON_PB_STUB
void
Metric::SendCreateMetricRequest()
{
  // Send the request to create the Metric to the parent process
  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
  SaveToSharedMemory(stub->ShmPool());
  CustomMetricsMessage* custom_metrics_msg = nullptr;
  AllocatedSharedMemory<CustomMetricsMessage> custom_metrics_shm;
  try {
    stub->SendMessage<CustomMetricsMessage>(
        custom_metrics_shm, PYTHONSTUB_MetricRequestNew, shm_handle_);
  }
  catch (const PythonBackendException& pb_exception) {
    throw PythonBackendException(
        "Error when creating Metric: " + std::string(pb_exception.what()));
  }

  custom_metrics_msg = custom_metrics_shm.data_.get();
  metric_address_ = custom_metrics_msg->address;
}

void
Metric::SendIncrementRequest(const double& value)
{
  py::gil_scoped_release release;
  try {
    CheckIfCleared();
    std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
    operation_value_ = value;
    SaveToSharedMemory(stub->ShmPool());
    AllocatedSharedMemory<CustomMetricsMessage> custom_metrics_shm;
    stub->SendMessage<CustomMetricsMessage>(
        custom_metrics_shm, PYTHONSTUB_MetricRequestIncrement, shm_handle_);
  }
  catch (const PythonBackendException& pb_exception) {
    throw PythonBackendException(
        "Failed to increment metric value: " +
        std::string(pb_exception.what()));
  }
}

void
Metric::SendSetValueRequest(const double& value)
{
  try {
    CheckIfCleared();
    std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
    operation_value_ = value;
    SaveToSharedMemory(stub->ShmPool());
    AllocatedSharedMemory<CustomMetricsMessage> custom_metrics_shm;
    stub->SendMessage<CustomMetricsMessage>(
        custom_metrics_shm, PYTHONSTUB_MetricRequestSet, shm_handle_);
  }
  catch (const PythonBackendException& pb_exception) {
    throw PythonBackendException(
        "Failed to set metric value: " + std::string(pb_exception.what()));
  }
}

void
Metric::SendObserveRequest(const double& value)
{
  py::gil_scoped_release release;
  try {
    CheckIfCleared();
    std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
    operation_value_ = value;
    SaveToSharedMemory(stub->ShmPool());
    AllocatedSharedMemory<CustomMetricsMessage> custom_metrics_shm;
    stub->SendMessage<CustomMetricsMessage>(
        custom_metrics_shm, PYTHONSTUB_MetricRequestObserve, shm_handle_);
  }
  catch (const PythonBackendException& pb_exception) {
    throw PythonBackendException(
        "Failed to observe metric value: " + std::string(pb_exception.what()));
  }
}

double
Metric::SendGetValueRequest()
{
  CustomMetricsMessage* custom_metrics_msg = nullptr;
  AllocatedSharedMemory<CustomMetricsMessage> custom_metrics_shm;
  try {
    CheckIfCleared();
    std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
    SaveToSharedMemory(stub->ShmPool());
    stub->SendMessage<CustomMetricsMessage>(
        custom_metrics_shm, PYTHONSTUB_MetricRequestValue, shm_handle_);
  }
  catch (const PythonBackendException& pb_exception) {
    throw PythonBackendException(
        "Failed to get metric value: " + std::string(pb_exception.what()));
  }

  custom_metrics_msg = custom_metrics_shm.data_.get();
  return custom_metrics_msg->value;
}

void
Metric::Clear()
{
  // Need to check if the metric has been cleared before as the Clear()'
  // function can be called from two different locations: when the metric family
  // clears the 'metric_map_' and when the 'Metric' object goes out of
  // scope/being deleted.
  if (!is_cleared_) {
    is_cleared_ = true;
    std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
    SaveToSharedMemory(stub->ShmPool());
    AllocatedSharedMemory<CustomMetricsMessage> custom_metrics_shm;
    try {
      stub->SendMessage<CustomMetricsMessage>(
          custom_metrics_shm, PYTHONSTUB_MetricRequestDelete, shm_handle_);
    }
    catch (const PythonBackendException& pb_exception) {
      std::cerr << "Error when deleting Metric: " << pb_exception.what()
                << "\n";
    }
  }
}

void
Metric::CheckIfCleared()
{
  if (is_cleared_) {
    throw PythonBackendException(
        "Invalid metric operation as the corresponding 'MetricFamily' has been "
        "deleted. The 'MetricFamily' object should be deleted AFTER its "
        "corresponding 'Metric' objects have been deleted.");
  }
}

#else
void*
Metric::InitializeTritonMetric()
{
  std::vector<const TRITONSERVER_Parameter*> labels_params;
  ParseLabels(labels_params, labels_);
  TRITONSERVER_MetricKind kind;
  THROW_IF_TRITON_ERROR(TRITONSERVER_GetMetricFamilyKind(
      reinterpret_cast<TRITONSERVER_MetricFamily*>(metric_family_address_),
      &kind));
  TRITONSERVER_MetricArgs* args = nullptr;
  switch (kind) {
    case TRITONSERVER_METRIC_KIND_COUNTER:
    case TRITONSERVER_METRIC_KIND_GAUGE:
      break;
    case TRITONSERVER_METRIC_KIND_HISTOGRAM: {
      const std::vector<double>& buckets = buckets_.value();
      THROW_IF_TRITON_ERROR(TRITONSERVER_MetricArgsNew(&args));
      THROW_IF_TRITON_ERROR(TRITONSERVER_MetricArgsSetHistogram(
          args, buckets.data(), buckets.size()));
      break;
    }
    default:
      break;
  }

  TRITONSERVER_Metric* triton_metric = nullptr;
  THROW_IF_TRITON_ERROR(TRITONSERVER_MetricNewWithArgs(
      &triton_metric,
      reinterpret_cast<TRITONSERVER_MetricFamily*>(metric_family_address_),
      labels_params.data(), labels_params.size(), args));
  for (const auto label : labels_params) {
    TRITONSERVER_ParameterDelete(const_cast<TRITONSERVER_Parameter*>(label));
  }
  THROW_IF_TRITON_ERROR(TRITONSERVER_MetricArgsDelete(args));
  return reinterpret_cast<void*>(triton_metric);
}

void
Metric::ParseLabels(
    std::vector<const TRITONSERVER_Parameter*>& labels_params,
    const std::string& labels)
{
  triton::common::TritonJson::Value labels_json;
  THROW_IF_TRITON_ERROR(labels_json.Parse(labels));

  std::vector<std::string> members;
  labels_json.Members(&members);
  for (const auto& member : members) {
    std::string value;
    THROW_IF_TRITON_ERROR(labels_json.MemberAsString(member.c_str(), &value));
    labels_params.emplace_back(TRITONSERVER_ParameterNew(
        member.c_str(), TRITONSERVER_PARAMETER_STRING, value.c_str()));
  }
}

void
Metric::HandleMetricOperation(
    CustomMetricsMessage* metrics_message_ptr,
    const PYTHONSTUB_CommandType& command_type)
{
  if (command_type == PYTHONSTUB_MetricRequestValue) {
    metrics_message_ptr->value = GetValue();
  } else if (command_type == PYTHONSTUB_MetricRequestIncrement) {
    Increment(operation_value_);
  } else if (command_type == PYTHONSTUB_MetricRequestSet) {
    SetValue(operation_value_);
  } else if (command_type == PYTHONSTUB_MetricRequestObserve) {
    Observe(operation_value_);
  } else {
    throw PythonBackendException("Unknown metric operation");
  }
}

void
Metric::Increment(const double& value)
{
  auto triton_metric = reinterpret_cast<TRITONSERVER_Metric*>(metric_address_);
  THROW_IF_TRITON_ERROR(TRITONSERVER_MetricIncrement(triton_metric, value));
}

void
Metric::SetValue(const double& value)
{
  auto triton_metric = reinterpret_cast<TRITONSERVER_Metric*>(metric_address_);
  THROW_IF_TRITON_ERROR(TRITONSERVER_MetricSet(triton_metric, value));
}

void
Metric::Observe(const double& value)
{
  auto triton_metric = reinterpret_cast<TRITONSERVER_Metric*>(metric_address_);
  THROW_IF_TRITON_ERROR(TRITONSERVER_MetricObserve(triton_metric, value));
}

double
Metric::GetValue()
{
  double value;
  auto triton_metric = reinterpret_cast<TRITONSERVER_Metric*>(metric_address_);
  THROW_IF_TRITON_ERROR(TRITONSERVER_MetricValue(triton_metric, &value));
  return value;
}

void
Metric::ClearTritonMetric()
{
  auto triton_metric = reinterpret_cast<TRITONSERVER_Metric*>(metric_address_);
  if (triton_metric != nullptr) {
    LOG_IF_ERROR(TRITONSERVER_MetricDelete(triton_metric), "deleting metric");
  }
}

#endif

}}}  // namespace triton::backend::python
