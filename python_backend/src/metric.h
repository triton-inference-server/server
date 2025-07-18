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

#pragma once

#include <optional>
#include <string>

#include "ipc_message.h"
#include "pb_memory.h"
#include "pb_string.h"
#include "pb_utils.h"

#ifdef TRITON_PB_STUB
#include <pybind11/embed.h>
namespace py = pybind11;
#else
#include "triton/core/tritonserver.h"
#endif

namespace triton { namespace backend { namespace python {

// The 'MetricShm' struct is utilized by the 'Metric' class for saving the
// essential data to shared memory and for loading the data from shared memory
// in order to reconstruct the 'Metric' object.
struct MetricShm {
  // The shared memory handle of the labels in PbString format.
  bi::managed_external_buffer::handle_t labels_shm_handle;
  // The shared memory handle of the buckets in PbMemory format.
  bi::managed_external_buffer::handle_t buckets_shm_handle;
  // The value used for incrementing or setting the metric.
  double operation_value;
  // The address of the TRITONSERVER_Metric object.
  void* metric_address;
  // The address corresponds to the TRITONSERVER_MetricFamily object that this
  // metric belongs to.
  void* metric_family_address;
};

class Metric {
 public:
  Metric(
      const std::string& labels,
      std::optional<const std::vector<double>> buckets,
      void* metric_family_address);

  ~Metric();

  /// Save Custom Metric object to shared memory.
  /// \param shm_pool Shared memory pool to save the custom metric object.
  void SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool);

  /// Create a Custom Metric object from shared memory.
  /// \param shm_pool Shared memory pool
  /// \param handle Shared memory handle of the custom metric.
  /// \return Returns the custom metrics in the specified request_handle
  /// location.
  static std::unique_ptr<Metric> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t handle);

  /// Get the address of the TRITONSERVER_Metric object.
  /// \return Returns the address of the TRITONSERVER_Metric object.
  void* MetricAddress();

  /// Send the request to the parent process to delete the Metric object.
  void Clear();

#ifdef TRITON_PB_STUB
  /// Send a request to register a new 'TRITONSERVER_Metric' object to the
  /// parent process.
  void SendCreateMetricRequest();

  /// Send the request to the parent process to increment the metric by the
  /// specified value.
  /// \param value The value to increment the metric by.
  void SendIncrementRequest(const double& value);

  /// Send the request to the parent process to set the metric to the specified
  /// value.
  /// \param value The value to set the metric to.
  void SendSetValueRequest(const double& value);

  /// Send the request to the parent process to observe the value to the metric.
  /// \param value The value to set the metric to.
  void SendObserveRequest(const double& value);

  /// Send the request to the parent process to get the value of the metric.
  /// \return Returns the value of the metric.
  double SendGetValueRequest();

  /// Throws an exception if the metric has been cleared. This check is to avoid
  /// the user error where the corresponding metric family has been deleted
  /// before the metric is deleted.
  void CheckIfCleared();
#else
  // Initialize the TRITONSERVER_Metric object.
  /// \return Returns the address of the TRITONSERVER_Metric object.
  void* InitializeTritonMetric();

  /// Parse the labels string into a vector of TRITONSERVER_Parameter.
  /// \param labels_params The vector of TRITONSERVER_Parameter to store the
  /// parsed labels.
  /// \param labels The labels string to parse.
  void ParseLabels(
      std::vector<const TRITONSERVER_Parameter*>& labels_params,
      const std::string& labels);

  /// Handle the metric operation.
  /// \param metrics_message_ptr The pointer to the CustomMetricsMessage object.
  void HandleMetricOperation(
      CustomMetricsMessage* metrics_message_ptr,
      const PYTHONSTUB_CommandType& command_type);

  /// Use Triton C API to increment the value of the metric by the given value.
  /// \param value The value to increment the metric by.
  void Increment(const double& value);

  /// Use Triton C API to set the value of the metric to the given value.
  /// \param value The value to set the metric to.
  void SetValue(const double& value);

  /// Use Triton C API to sample the observation to the metric.
  /// \param value The value to sample observation to the metric.
  void Observe(const double& value);

  /// Use Triton C API to get the value of the metric.
  double GetValue();

  /// Clear the TRITONSERVER_Metric object.
  void ClearTritonMetric();
#endif

  /// Disallow copying the custom metric object.
  DISALLOW_COPY_AND_ASSIGN(Metric);

 private:
  // The private constructor for creating a Metric object from shared memory.
  Metric(
      AllocatedSharedMemory<MetricShm>& custom_metric_shm,
      std::unique_ptr<PbString>& labels_shm,
      std::unique_ptr<PbMemory>& buckets);

  // The labels of the metric, which is the identifier of the metric.
  std::string labels_;
  // Monotonically increasing values representing bucket boundaries for creating
  // histogram metric.
  std::optional<std::vector<double>> buckets_;
  // The value used for incrementing or setting the metric.
  double operation_value_;
  // The address of the TRITONSERVER_Metric object.
  void* metric_address_;
  // The address corresponds to the TRITONSERVER_MetricFamily object that this
  // metric belongs to.
  void* metric_family_address_;
  // Indicates whether the metric has been cleared. It is needed as the Clear()'
  // function can be called from two different locations: when the metric family
  // clears the 'metric_map_' and when the 'Metric' object goes out of
  // scope/being deleted.
  bool is_cleared_;

  // Shared Memory Data Structures
  AllocatedSharedMemory<MetricShm> custom_metric_shm_;
  MetricShm* custom_metric_shm_ptr_;
  bi::managed_external_buffer::handle_t shm_handle_;
  std::unique_ptr<PbString> labels_shm_;
  std::unique_ptr<PbMemory> buckets_shm_;
};

}}};  // namespace triton::backend::python
