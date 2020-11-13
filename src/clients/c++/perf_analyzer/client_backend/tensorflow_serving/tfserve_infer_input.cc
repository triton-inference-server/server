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

#include "src/clients/c++/perf_analyzer/client_backend/tensorflow_serving/tfserve_infer_input.h"

namespace perfanalyzer { namespace clientbackend {

Error
TFServeInferInput::Create(
    InferInput** infer_input, const std::string& name,
    const std::vector<int64_t>& dims, const std::string& datatype)
{
  TFServeInferInput* local_infer_input =
      new TFServeInferInput(name, dims, datatype);

  *infer_input = local_infer_input;
  return Error::Success;
}

Error
TFServeInferInput::SetShape(const std::vector<int64_t>& shape)
{
  shape_ = shape;
  return Error::Success;
}

Error
TFServeInferInput::Reset()
{
  bufs_.clear();
  buf_byte_sizes_.clear();
  bufs_idx_ = 0;
  byte_size_ = 0;
  return Error::Success;
}

Error
TFServeInferInput::AppendRaw(const uint8_t* input, size_t input_byte_size)
{
  byte_size_ += input_byte_size;

  bufs_.push_back(input);
  buf_byte_sizes_.push_back(input_byte_size);

  return Error::Success;
}

Error
TFServeInferInput::ByteSize(size_t* byte_size) const
{
  *byte_size = byte_size_;
  return Error::Success;
}

Error
TFServeInferInput::PrepareForRequest()
{
  // Reset position so request sends entire input.
  bufs_idx_ = 0;
  buf_pos_ = 0;
  return Error::Success;
}

Error
TFServeInferInput::GetNext(
    const uint8_t** buf, size_t* input_bytes, bool* end_of_input)
{
  if (bufs_idx_ < bufs_.size()) {
    *buf = bufs_[bufs_idx_];
    *input_bytes = buf_byte_sizes_[bufs_idx_];
    bufs_idx_++;
  } else {
    *buf = nullptr;
    *input_bytes = 0;
  }
  *end_of_input = (bufs_idx_ >= bufs_.size());

  return Error::Success;
}

TFServeInferInput::TFServeInferInput(
    const std::string& name, const std::vector<int64_t>& dims,
    const std::string& datatype)
    : InferInput(BackendKind::TENSORFLOW_SERVING, name, datatype), shape_(dims)
{
}

}}  // namespace perfanalyzer::clientbackend
