// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>

#include "triton/core/tritonbackend.h"

namespace triton { namespace backend { namespace python {
class PbMetricReporter {
  TRITONBACKEND_ModelInstance* instance_;
  TRITONBACKEND_Request** requests_;
  uint32_t request_count_;
  std::shared_ptr<std::vector<TRITONBACKEND_Response*>> responses_;
  size_t total_batch_size_;
  uint64_t exec_start_ns_;
  uint64_t compute_start_ns_;
  uint64_t compute_end_ns_;
  uint64_t exec_end_ns_;
  bool success_status_;

 public:
  PbMetricReporter(
      TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::shared_ptr<std::vector<TRITONBACKEND_Response*>> responses);
  ~PbMetricReporter();
  void SetBatchStatistics(size_t total_batch_size);
  void SetExecStartNs(const uint64_t exec_start_ns);
  void SetComputeStartNs(const uint64_t compute_start_ns);
  void SetComputeEndNs(const uint64_t compute_end_ns);
  void SetExecEndNs(const uint64_t exec_end_ns);
  void SetSuccessStatus(const bool success_status);
};
}}};  // namespace triton::backend::python
