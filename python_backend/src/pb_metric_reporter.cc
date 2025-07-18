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

#include "pb_metric_reporter.h"

#include "triton/backend/backend_common.h"

namespace triton { namespace backend { namespace python {

PbMetricReporter::PbMetricReporter(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::shared_ptr<std::vector<TRITONBACKEND_Response*>> responses)
    : instance_(instance), requests_(requests), request_count_(request_count),
      responses_(responses), total_batch_size_(0), exec_start_ns_(0),
      compute_start_ns_(0), compute_end_ns_(0), exec_end_ns_(0),
      success_status_(true)
{
}

PbMetricReporter::~PbMetricReporter()
{
  for (uint32_t r = 0; r < request_count_; ++r) {
    TRITONBACKEND_Request* request = requests_[r];

    // Report statistics for the request. Note that there could
    // still be responses that have not yet been sent but those
    // cannot be captured in the statistics as they reflect only the
    // request object. We use the execution start/end time for
    // compute also so that the entire execution time is associated
    // with the inference computation.
    if (responses_) {
      LOG_IF_ERROR(
          TRITONBACKEND_ModelInstanceReportStatistics(
              instance_, request, ((*responses_)[r] != nullptr) /* success */,
              exec_start_ns_, compute_start_ns_, compute_end_ns_, exec_end_ns_),
          "failed reporting request statistics");
    } else {
      LOG_IF_ERROR(
          TRITONBACKEND_ModelInstanceReportStatistics(
              instance_, request, success_status_, exec_start_ns_,
              compute_start_ns_, compute_end_ns_, exec_end_ns_),
          "failed reporting request statistics");
    }
  }

  // Report the entire batch statistics. This backend does not support
  // batching so the total batch size is always 1.
  if (total_batch_size_ != 0) {
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportBatchStatistics(
            instance_, total_batch_size_, exec_start_ns_, compute_start_ns_,
            compute_end_ns_, exec_end_ns_),
        "failed reporting batch request statistics");
  }
}

void
PbMetricReporter::SetBatchStatistics(size_t total_batch_size)
{
  total_batch_size_ = total_batch_size;
}

void
PbMetricReporter::SetExecStartNs(const uint64_t exec_start_ns)
{
  exec_start_ns_ = exec_start_ns;
}

void
PbMetricReporter::SetComputeStartNs(const uint64_t compute_start_ns)
{
  compute_start_ns_ = compute_start_ns;
}

void
PbMetricReporter::SetComputeEndNs(const uint64_t compute_end_ns)
{
  compute_end_ns_ = compute_end_ns;
}

void
PbMetricReporter::SetExecEndNs(const uint64_t exec_end_ns)
{
  exec_end_ns_ = exec_end_ns;
}

void
PbMetricReporter::SetSuccessStatus(const bool success_status)
{
  success_status_ = success_status;
}

}}}  // namespace triton::backend::python
