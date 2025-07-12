// Copyright 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pb_response_iterator.h"

#include <pybind11/embed.h>

#include <chrono>

#include "pb_stub.h"
namespace py = pybind11;

namespace triton { namespace backend { namespace python {

ResponseIterator::ResponseIterator(
    const std::shared_ptr<InferResponse>& response)
    : id_(response->Id()), is_finished_(false), is_cleared_(false), idx_(0)
{
  response_buffer_.push(response);
  pb_bls_cancel_ = std::make_shared<PbBLSCancel>(response->Id());
}

ResponseIterator::~ResponseIterator()
{
  // Fetch all the remaining responses if not finished yet.
  if (!is_finished_) {
    bool done = false;
    while (!done) {
      try {
        Next();
      }
      catch (const py::stop_iteration& exception) {
        done = true;
      }
    }
  }

  if (!is_cleared_) {
    Clear();
  }
  responses_.clear();
}

std::shared_ptr<InferResponse>
ResponseIterator::Next()
{
  if (is_finished_) {
    if (!is_cleared_) {
      Clear();
    }

    if (idx_ < responses_.size()) {
      return responses_[idx_++];
    } else {
      throw py::stop_iteration("Iteration is done for the responses.");
    }
  } else {
    std::shared_ptr<InferResponse> response;
    {
      {
        std::unique_lock<std::mutex> lock{mu_};
        while (response_buffer_.empty()) {
          py::gil_scoped_release release;
          cv_.wait(lock);
        }
        response = response_buffer_.front();
        response_buffer_.pop();
        is_finished_ = response->IsLastResponse();
        responses_.push_back(response);
      }
    }

    if (is_finished_) {
      idx_ = responses_.size();
      Clear();
    }
    return response;
  }
}

void
ResponseIterator::Iter()
{
  if (is_finished_) {
    // If the previous iteration is finished, reset the index so that it will
    // iterator from the beginning of the responses. Otherwise just resume the
    // iteration from the previous index.
    if (idx_ >= responses_.size()) {
      idx_ = 0;
    }
  }
}

void
ResponseIterator::EnqueueResponse(std::shared_ptr<InferResponse> infer_response)
{
  {
    std::lock_guard<std::mutex> lock{mu_};
    response_buffer_.push(infer_response);
  }
  cv_.notify_one();
}

void*
ResponseIterator::Id()
{
  return id_;
}

void
ResponseIterator::Clear()
{
  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
  stub->EnqueueCleanupId(id_, PYTHONSTUB_BLSDecoupledInferPayloadCleanup);
  {
    std::lock_guard<std::mutex> lock{mu_};
    response_buffer_.push(DUMMY_MESSAGE);
  }
  cv_.notify_all();
  std::queue<std::shared_ptr<InferResponse>> empty;
  std::swap(response_buffer_, empty);
  is_cleared_ = true;
}

std::vector<std::shared_ptr<InferResponse>>
ResponseIterator::GetExistingResponses()
{
  std::vector<std::shared_ptr<InferResponse>> responses;
  std::unique_lock<std::mutex> lock{mu_};
  while (!response_buffer_.empty()) {
    responses.push_back(response_buffer_.front());
    response_buffer_.pop();
  }
  is_finished_ = true;
  is_cleared_ = true;

  return responses;
}

void
ResponseIterator::Cancel()
{
  if (!is_finished_) {
    pb_bls_cancel_->Cancel();
  }
}

}}}  // namespace triton::backend::python
