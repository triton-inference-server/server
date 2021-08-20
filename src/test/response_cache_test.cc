// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gtest/gtest.h"

#include "src/core/memory.h"
#include "src/core/response_cache.h"

namespace ni = nvidia::inferenceserver;

/* Mock classes for Unit Testing */
namespace nvidia { namespace inferenceserver {

/* InferenceRequest*/

// InferenceRequest Input Constructor
InferenceRequest::Input::Input(
    const std::string& name, const inference::DataType datatype,
    const int64_t* shape, const uint64_t dim_count)
    : name_(name), datatype_(datatype),
      original_shape_(shape, shape + dim_count), is_shape_tensor_(false),
      data_(new MemoryReference), has_host_policy_specific_data_(false)
{}

// Same as defined in infer_request.cc
const std::string&
InferenceRequest::ModelName() const
{
  return backend_raw_->Name();
}

void
InferenceRequest::SetPriority(unsigned int)
{}

Status
InferenceRequest::AddOriginalInput(
    const std::string& name, const inference::DataType datatype,
    const int64_t* shape, const uint64_t dim_count,
    InferenceRequest::Input** input)
{
  const auto& pr = original_inputs_.emplace(
      std::piecewise_construct, std::forward_as_tuple(name),
      std::forward_as_tuple(name, datatype, shape, dim_count));
  if (!pr.second) {
    return Status(
        Status::Code::INVALID_ARG,
        "input '" + name + "' already exists in request");
  }

  if (input != nullptr) {
    *input = std::addressof(pr.first->second);
  }

  needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::AddOriginalInput(
    const std::string& name, const inference::DataType datatype,
    const std::vector<int64_t>& shape, InferenceRequest::Input** input)
{
  return AddOriginalInput(name, datatype, &shape[0], shape.size(), input);
}

/* InferenceResponse */

// Same as defined in infer_response.cc
Status
InferenceResponse::Output::DataBuffer(
    const void** buffer, size_t* buffer_byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id,
    void** userp) const
{
  *buffer = allocated_buffer_;
  *buffer_byte_size = allocated_buffer_byte_size_;
  *memory_type = allocated_memory_type_;
  *memory_type_id = allocated_memory_type_id_;
  *userp = allocated_userp_;
  return Status::Success;
}

// Simplified version of AllocateDataBuffer for CPU memory only
Status
InferenceResponse::Output::AllocateDataBuffer(
    void** buffer, size_t buffer_byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
{
    if (allocated_buffer_ != nullptr) {
        return Status(
            Status::Code::ALREADY_EXISTS,
            "allocated buffer for output '" + name_ + "' already exists");
    }
    
    // Simplifications - CPU memory only for now
    if (*memory_type != TRITONSERVER_MEMORY_CPU || *memory_type_id != 0) {
        return Status(
            Status::Code::INTERNAL,
            "Only standard CPU memory supported for now");
    }

    // Set relevant member variables for DataBuffer() to return
    std::memcpy(&allocated_buffer_, &buffer, buffer_byte_size);
    allocated_buffer_byte_size_ = buffer_byte_size;
    allocated_memory_type_ = *memory_type;
    allocated_memory_type_id_ = *memory_type_id;
    allocated_userp_ = nullptr;
    return Status::Success;
}

}}  // namespace nvidia::inferenceserver


namespace {

// Test Fixture
class RequestResponseCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Simplest test possible just to test flow
TEST_F(RequestResponseCacheTest, TestHelloWorld) {
    std::cout << "Hello, World!" << std::endl;
}

// Test hashing for consistency on same request
TEST_F(RequestResponseCacheTest, TestRequestHashing) {
    // Create cache
    std::cout << "Create cache" << std::endl;
    uint64_t cache_size = 4*1024*1024;
    ni::RequestResponseCache cache(cache_size);
    // Create backend
    std::cout << "Create backend" << std::endl;
    std::shared_ptr<ni::InferenceBackend> backend;
    const uint64_t model_version = 1;
    // Create request
    std::cout << "Create request" << std::endl;
    ni::InferenceRequest request(backend, model_version);
    // Create input
    std::cout << "Create input" << std::endl;
    std::string input_name = "input0";
    inference::DataType dtype = inference::DataType::TYPE_INT32;
    std::vector<int64_t> shape{1, 256};
    //ni::InferenceRequest::Input input0(input_name, dtype, shape); 
    ni::InferenceRequest::Input* input0 = nullptr;
    // Add input to request
    std::cout << "Add input to request" << std::endl;
    request.AddOriginalInput(input_name, dtype, shape, &input0);
    // Compare hashes
    std::cout << "Compare hashes" << std::endl;
    // TODO: Segfault
    uint64_t hash1 = cache.Hash(request);
    uint64_t hash2 = cache.Hash(request);
    std::cout << "Hash1: " << hash1 << std::endl;
    std::cout << "Hash2: " << hash2 << std::endl;
    assert(hash1 == hash2);
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
