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

#include <string>

#include "ipc_message.h"
#include "metric.h"
#include "pb_string.h"
#include "pb_utils.h"

#ifdef TRITON_PB_STUB
#include <pybind11/embed.h>
namespace py = pybind11;
#else
#include "triton/core/tritonserver.h"
#endif

namespace triton { namespace backend { namespace python {

// The 'MetricFamilyShm' struct is utilized by the 'MetricFamily' class for
// saving the essential data to shared memory and for loading the data from
// shared memory in order to reconstruct the 'MetricFamily' object.
struct MetricFamilyShm {
  // The shared memory handle of the name in PbString format.
  bi::managed_external_buffer::handle_t name_shm_handle;
  // The shared memory handle of the description in PbString format.
  bi::managed_external_buffer::handle_t description_shm_handle;
  // The metric kind of the 'MetricFamily'.
  MetricKind kind;
  // The address of the 'TRITONSERVER_MetricFamily' object.
  void* metric_family_address;
};

class MetricFamily {
 public:
  MetricFamily(
      const std::string& name, const std::string& description,
      const MetricKind& kind);

  ~MetricFamily();

  /// Save a custom metric family to shared memory.
  /// \param shm_pool Shared memory pool to save the custom metric family.
  void SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool);

  /// Create a Custom Metric Family object from shared memory.
  /// \param shm_pool Shared memory pool
  /// \param handle Shared memory handle of the custom metric family.
  /// \return Returns the custom metric family in the specified handle
  /// location.
  static std::unique_ptr<MetricFamily> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t handle);

  /// Get the address of the TRITONSERVER_MetricFamily object.
  /// \return Returns the address of the TRITONSERVER_MetricFamily object.
  void* MetricFamilyAddress();

#ifdef TRITON_PB_STUB
  /// Create a metric family object and returned as a shared pointer.
  /// \param name The name of the metric family.
  /// \param description The description of the metric family.
  /// \param kind The metric kind of the metric family.
  /// \return Returns the shared pointer to the created metric family.
  static std::shared_ptr<MetricFamily> CreateMetricFamily(
      const std::string& name, const std::string& description,
      const MetricKind& kind);

  /// Send a request to register a new 'TRITONSERVER_MetricFamily' object to the
  /// parent process.
  void SendCreateMetricFamilyRequest();

  /// Create a metric from the metric family and store it in the metric map.
  /// \param labels The labels of the metric.
  /// \param buckets Monotonically increasing values representing bucket
  /// boundaries for creating histogram metric.
  /// \return Returns the shared pointer to the created metric.
  std::shared_ptr<Metric> CreateMetric(
      const py::object& labels, const py::object& buckets);
#else
  /// Initialize the TRITONSERVER_MetricFamily object.
  /// \return Returns the address of the TRITONSERVER_MetricFamily object.
  void* InitializeTritonMetricFamily();

  /// Helper function to convert the MetricKind enum to TRITONSERVER_MetricKind
  /// \param kind The MetricKind enum to be converted.
  /// \return Returns the TRITONSERVER_MetricKind enum.
  TRITONSERVER_MetricKind ToTritonServerMetricKind(const MetricKind& kind);

  /// Clear the TRITONSERVER_MetricFamily object.
  void ClearTritonMetricFamily();
#endif

  /// Disallow copying the metric family object.
  DISALLOW_COPY_AND_ASSIGN(MetricFamily);

 private:
  // The private constructor for creating a MetricFamily object from shared
  // memory.
  MetricFamily(
      AllocatedSharedMemory<MetricFamilyShm>& custom_metric_family_shm,
      std::unique_ptr<PbString>& name_shm,
      std::unique_ptr<PbString>& description_shm);

  // The name of the metric family.
  std::string name_;
  // The description of the metric family.
  std::string description_;
  // The metric kind of the metric family. Currently only supports GAUGE,
  // COUNTER and HISTOGRAM.
  MetricKind kind_;
  // The address of the TRITONSERVER_MetricFamily object.
  void* metric_family_address_;

  // The mutex to protect the 'metric_map_'.
  std::mutex metric_map_mu_;
  // Need to keep track of the metrics associated with the metric family to make
  // sure the metrics are cleaned up before the metric family is deleted.
  std::unordered_map<void*, std::shared_ptr<Metric>> metric_map_;

  // Shared Memory Data Structures
  AllocatedSharedMemory<MetricFamilyShm> custom_metric_family_shm_;
  MetricFamilyShm* custom_metric_family_shm_ptr_;
  bi::managed_external_buffer::handle_t shm_handle_;
  std::unique_ptr<PbString> name_shm_;
  std::unique_ptr<PbString> description_shm_;
};

}}};  // namespace triton::backend::python
