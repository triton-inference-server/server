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

#define DLL_EXPORTING

#include "src/clients/c++/experimental_api_v2/library/grpc_utils.h"

namespace nvidia { namespace inferenceserver { namespace client {

//==============================================================================

Error
InferInputGrpc::Create(
    InferInputGrpc** infer_input, const std::string& name,
    const std::vector<int64_t>& dims, const std::string& datatype)
{
  *infer_input = new InferInputGrpc(name, dims, datatype);
  return Error::Success;
}

Error
InferInputGrpc::GetName(std::string* name) const
{
  *name = input_tensor_.name();
  return Error::Success;
}

Error
InferInputGrpc::GetDatatype(std::string* datatype) const
{
  *datatype = input_tensor_.datatype();
  return Error::Success;
}

Error
InferInputGrpc::GetShape(std::vector<int64_t>* dims) const
{
  for (const auto dim : input_tensor_.shape()) {
    dims->push_back(dim);
  }
  return Error::Success;
}

Error
InferInputGrpc::SetShape(const std::vector<int64_t>& dims)
{
  input_tensor_.mutable_shape()->Clear();
  input_tensor_.mutable_shape()->Clear();
  for (const auto dim : dims) {
    input_tensor_.mutable_shape()->Add(dim);
  }
  return Error::Success;
}

Error
InferInputGrpc::SetRaw(const uint8_t* input, size_t input_byte_size)
{
  input_tensor_.mutable_contents()->set_raw_contents(input, input_byte_size);
  return Error::Success;
}

Error
InferInputGrpc::SetRaw(const std::vector<uint8_t>& input)
{
  return SetRaw(&input[0], input.size());
}

Error
InferInputGrpc::SetFromString(const std::vector<std::string>& input)
{
  std::string sbuf;
  for (const auto& str : input) {
    uint32_t len = str.size();
    sbuf.append(reinterpret_cast<const char*>(&len), sizeof(uint32_t));
    sbuf.append(str);
  }
  return SetRaw((uint8_t*)&sbuf[0], sbuf.size());
}

Error
InferInputGrpc::SetSharedMemory(
    const std::string& region_name, const size_t byte_size, const size_t offset)
{
  (*input_tensor_.mutable_parameters())["shared_memory_region"]
      .set_string_param(region_name);
  (*input_tensor_.mutable_parameters())["shared_memory_byte_size"]
      .set_int64_param(byte_size);
  if (offset != 0) {
    (*input_tensor_.mutable_parameters())["shared_memory_offset"]
        .set_int64_param(offset);
  }
  return Error::Success;
}

InferInputGrpc::InferInputGrpc(
    const std::string& name, const std::vector<int64_t>& dims,
    const std::string& datatype)
{
  input_tensor_.set_name(name);
  input_tensor_.mutable_shape()->Clear();
  for (const auto dim : dims) {
    input_tensor_.mutable_shape()->Add(dim);
  }
  input_tensor_.set_datatype(datatype);
}

//==============================================================================

Error
InferOutputGrpc::Create(
    InferOutputGrpc** infer_output, const std::string& name,
    const size_t class_count)
{
  *infer_output = new InferOutputGrpc(name, class_count);
  return Error::Success;
}

Error
InferOutputGrpc::GetName(std::string* name) const
{
  *name = output_tensor_.name();
  return Error::Success;
}

Error
InferOutputGrpc::SetSharedMemory(
    const std::string& region_name, const size_t byte_size, const size_t offset)
{
  (*output_tensor_.mutable_parameters())["shared_memory_region"]
      .set_string_param(region_name);
  (*output_tensor_.mutable_parameters())["shared_memory_byte_size"]
      .set_int64_param(byte_size);
  if (offset != 0) {
    (*output_tensor_.mutable_parameters())["shared_memory_offset"]
        .set_int64_param(offset);
  }
  return Error::Success;
}


InferOutputGrpc::InferOutputGrpc(
    const std::string& name, const size_t class_count)
{
  output_tensor_.set_name(name);
  if (class_count != 0) {
    (*output_tensor_.mutable_parameters())["classification"].set_int64_param(
        class_count);
  }
}

//==============================================================================

Error
InferResultGrpc::Create(
    InferResultGrpc** infer_result,
    std::shared_ptr<ModelInferResponse> response)
{
  *infer_result = new InferResultGrpc(response);
  return Error::Success;
}

Error
InferResultGrpc::GetShape(
    const std::string& output_name, std::vector<int64_t>* shape) const
{
  shape->clear();
  auto it = output_name_to_result_map_.find(output_name);
  if (it != output_name_to_result_map_.end()) {
    for (const auto dim : it->second->shape()) {
      shape->push_back(dim);
    }
  } else {
    return Error(
        "The response does not contain results or output name " + output_name);
  }
  return Error::Success;
}

Error
InferResultGrpc::GetDatatype(
    const std::string& output_name, std::string* datatype) const
{
  auto it = output_name_to_result_map_.find(output_name);
  if (it != output_name_to_result_map_.end()) {
    *datatype = it->second->datatype();
  } else {
    return Error(
        "The response does not contain results or output name " + output_name);
  }
  return Error::Success;
}


Error
InferResultGrpc::GetRaw(
    const std::string& output_name, const uint8_t** buf,
    size_t* byte_size) const
{
  auto it = output_name_to_result_map_.find(output_name);
  if (it != output_name_to_result_map_.end()) {
    *buf = (uint8_t*)&(it->second->contents().raw_contents()[0]);
    *byte_size = it->second->contents().raw_contents().size();
  } else {
    return Error(
        "The response does not contain results or output name " + output_name);
  }

  return Error::Success;
}

InferResultGrpc::InferResultGrpc(std::shared_ptr<ModelInferResponse> response)
{
  response_ = response;
  for (const auto& output : response_->outputs()) {
    output_name_to_result_map_[output.name()] = &output;
  }
}


}}}  // namespace nvidia::inferenceserver::client
