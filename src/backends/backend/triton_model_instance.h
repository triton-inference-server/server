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
#pragma once

#include <memory>
#include <string>
#include "src/core/constants.h"
#include "src/core/metric_model_reporter.h"
#include "triton/core/model_config.pb.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class TritonModel;

//
// Represents a model instance.
//
class TritonModelInstance {
 public:
  static Status CreateInstances(
      TritonModel* model, const inference::ModelConfig& model_config,
      std::vector<std::unique_ptr<TritonModelInstance>>* instances);
  ~TritonModelInstance();

  const std::string& Name() const { return name_; }
  size_t Index() const { return index_; }
  TRITONSERVER_InstanceGroupKind Kind() const { return kind_; }
  int32_t DeviceId() const { return device_id_; }

  TritonModel* Model() { return model_; }
  void* State() { return state_; }
  void SetState(void* state) { state_ = state; }

  MetricModelReporter* MetricReporter() const { return reporter_.get(); }

 private:
  DISALLOW_COPY_AND_ASSIGN(TritonModelInstance);

  TritonModelInstance(
      TritonModel* model, const std::string& name, const size_t index,
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id);
  static Status CreateInstance(
      TritonModel* model, const std::string& name, const size_t index,
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
      std::vector<std::unique_ptr<TritonModelInstance>>* instances);

  // The TritonModel object that owns this instance. The instance
  // holds this as a raw pointer because the lifetime of the model is
  // guaranteed to be longer than the lifetime of an instance owned by the
  // model.
  TritonModel* model_;

  std::string name_;
  size_t index_;

  // For CPU device_id_ is always 0. For GPU device_id_ indicates the
  // GPU device to be used by the instance.
  TRITONSERVER_InstanceGroupKind kind_;
  int32_t device_id_;

  // Reporter for metrics, or nullptr if no metrics should be reported
  std::unique_ptr<MetricModelReporter> reporter_;

  // Opaque state associated with this model instance.
  void* state_;
};

}}  // namespace nvidia::inferenceserver
