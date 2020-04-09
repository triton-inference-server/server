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

#include <string>
#include <unordered_map>
#include <vector>
#include "src/core/backend.h"
#include "src/core/model_config.h"
#include "src/core/status.h"
#include "src/core/tritonserver.h"

namespace nvidia { namespace inferenceserver {

class InferenceResponse;

//
// An inference response factory.
//
class InferenceResponseFactory {
 public:
  InferenceResponseFactory(
      const std::shared_ptr<InferenceBackend>& backend, const std::string& id,
      TRITONSERVER_ResponseAllocator* allocator,
      TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn,
      TRITONSERVER_ResponseAllocatorReleaseFn_t release_fn, void* alloc_userp)
      : backend_(backend), id_(id), allocator_(allocator), alloc_fn_(alloc_fn),
        release_fn_(release_fn), alloc_userp_(alloc_userp)
  {
  }

  // Create a new response.
  Status CreateResponse(std::unique_ptr<InferenceResponse>* response) const;

 private:
  // The backend associated with this factory. For normal
  // requests/responses this will always be defined and acts to keep
  // the backend loaded as long as this factory is live. It may be
  // nullptr for cases where the backend itself created the request
  // (like running requests for warmup) and so must protect any uses
  // to handle the nullptr case.
  std::shared_ptr<InferenceBackend> backend_;

  std::string id_;

  // The allocation function and allocation object for responses
  // created by this factory. These pointers are not owned by this
  // object and so should not be destroyed when the object is
  // destroyed.
  TRITONSERVER_ResponseAllocator* allocator_;
  TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn_;
  TRITONSERVER_ResponseAllocatorReleaseFn_t release_fn_;
  void* alloc_userp_;
};

//
// An inference response.
//
class InferenceResponse {
 public:
  // Output tensor
  class Output {
   public:
    Output(
        const std::string& name, const DataType datatype,
        const std::vector<int64_t>& shape,
        TRITONSERVER_ResponseAllocator* allocator,
        TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn,
        TRITONSERVER_ResponseAllocatorReleaseFn_t release_fn, void* alloc_userp)
        : name_(name), datatype_(datatype), shape_(shape),
          allocator_(allocator), alloc_fn_(alloc_fn), release_fn_(release_fn),
          alloc_userp_(alloc_userp), allocated_buffer_(nullptr)
    {
    }

    ~Output();

    // The name of the output tensor. There is no mutable operator for
    // the name because it is used in a InferenceResponse map and a
    // mutable method would allow it to get out-of-sync.
    const std::string& Name() const { return name_; }

    // Data type of the output tensor.
    DataType DType() const { return datatype_; }

    // The shape of the output tensor.
    const std::vector<int64_t>& Shape() const { return shape_; }

    // Get information about the buffer allocated for this output
    // tensor's data. If no buffer is allocated 'buffer' will return
    // nullptr and the other returned values will be undefined.
    Status Buffer(
        void** buffer, size_t* buffer_byte_size,
        TRITONSERVER_Memory_Type* memory_type, int64_t* memory_type_id);

    // Allocate the buffer that should be used for this output
    // tensor's data. 'buffer' must return a buffer of size
    // 'buffer_byte_size'.  'memory_type' acts as both input and
    // output. On input gives the buffer memory type preferred by the
    // caller and on return holds the actual memory type of
    // 'buffer'. 'memory_type_id' acts as both input and output. On
    // input gives the buffer memory type id preferred by the caller
    // and returns the actual memory type id of 'buffer'. Only a
    // single buffer may be allocated for the output at any time, so
    // multiple calls to AllocateBuffer without intervening
    // ReleaseBuffer call will result in an error.
    Status AllocateBuffer(
        void** buffer, const size_t buffer_byte_size,
        TRITONSERVER_Memory_Type* memory_type, int64_t* memory_type_id);

    // Release the buffer that was previously allocated by
    // AllocateBuffer(). Do nothing if AllocateBuffer() has not been
    // called.
    Status ReleaseBuffer();

   private:
    friend std::ostream& operator<<(
        std::ostream& out, const InferenceResponse::Output& output);

    std::string name_;
    DataType datatype_;
    std::vector<int64_t> shape_;

    // The allocation function and allocation object for responses
    // created by this factory. These pointers are not owned by this
    // object and so should not be destroyed when the object is
    // destroyed.
    TRITONSERVER_ResponseAllocator* allocator_;
    TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn_;
    TRITONSERVER_ResponseAllocatorReleaseFn_t release_fn_;
    void* alloc_userp_;

    // Information about the buffer allocated by
    // AllocateBuffer(). This information is needed by
    // ReleaseBuffer().
    void* allocated_buffer_;
    size_t allocated_buffer_byte_size_;
    TRITONSERVER_Memory_Type allocated_memory_type_;
    int64_t allocated_memory_type_id_;
    void* allocated_userp_;
  };

  // InferenceResponse
  InferenceResponse(
      const std::shared_ptr<InferenceBackend>& backend, const std::string& id,
      TRITONSERVER_ResponseAllocator* allocator,
      TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn,
      TRITONSERVER_ResponseAllocatorReleaseFn_t release_fn, void* alloc_userp)
      : backend_(backend), id_(id), allocator_(allocator), alloc_fn_(alloc_fn),
        release_fn_(release_fn), alloc_userp_(alloc_userp)
  {
  }

  const std::string& Id() const { return id_; }
  const std::string& ModelName() const
  {
    static const std::string unknown("<unknown>");
    return (backend_ == nullptr) ? unknown : backend_->Name();
  }
  int64_t ActualModelVersion() const
  {
    return (backend_ == nullptr) ? -1 : backend_->Version();
  }

  const std::unordered_map<std::string, Output>& Outputs() const
  {
    return outputs_;
  }

  // Add an output to the response.
  Status AddOutput(
      const std::string& name, const DataType datatype,
      const std::vector<int64_t>& shape);

 private:
  friend std::ostream& operator<<(
      std::ostream& out, const InferenceResponse& response);

  // The backend associated with this factory. For normal
  // requests/responses this will always be defined and acts to keep
  // the backend loaded as long as this factory is live. It may be
  // nullptr for cases where the backend itself created the request
  // (like running requests for warmup) and so must protect any uses
  // to handle the nullptr case.
  std::shared_ptr<InferenceBackend> backend_;

  std::string id_;
  std::unordered_map<std::string, Output> outputs_;

  // The allocation function and allocation object for responses
  // created by this factory. These pointers are not owned by this
  // object and so should not be destroyed when the object is
  // destroyed.
  TRITONSERVER_ResponseAllocator* allocator_;
  TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn_;
  TRITONSERVER_ResponseAllocatorReleaseFn_t release_fn_;
  void* alloc_userp_;
};

std::ostream& operator<<(std::ostream& out, const InferenceResponse& response);
std::ostream& operator<<(
    std::ostream& out, const InferenceResponse::Output& output);

}}  // namespace nvidia::inferenceserver
