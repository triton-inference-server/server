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

#include "src/clients/c++/perf_analyzer/client_backend/client_backend.h"
#include "src/clients/c++/perf_analyzer/perf_utils.h"


namespace perfanalyzer { namespace clientbackend {

//==============================================================
/// TorchServeInferInput instance holds the information regarding
/// model input tensor. In this case the content held will be
/// the path to the file holding data.
///
class TorchServeInferInput : public InferInput {
 public:
  static Error Create(
      InferInput** infer_input, const std::string& name,
      const std::vector<int64_t>& dims, const std::string& datatype);
  /// See InferInput::Shape()
  const std::vector<int64_t>& Shape() const override { return shape_; }
  /// See InferInput::SetShape()
  Error SetShape(const std::vector<int64_t>& shape) override;
  /// See InferInput::Reset()
  Error Reset() override;
  /// See InferInput::AppendRaw()
  Error AppendRaw(const uint8_t* input, size_t input_byte_size) override;
  /// Gets the size of data added into this input in bytes.
  /// \param byte_size The size of data added in bytes.
  /// \return Error object indicating success or failure.
  Error ByteSize(size_t* byte_size) const;
  /// Resets the heads to start providing data from the begining.
  Error PrepareForRequest();
  /// Get the next chunk of data if available.
  Error GetNext(const uint8_t** buf, size_t* input_bytes, bool* end_of_input);

 private:
  explicit TorchServeInferInput(
      const std::string& name, const std::vector<int64_t>& dims,
      const std::string& datatype);

  std::vector<int64_t> shape_;
  size_t byte_size_;

  size_t bufs_idx_, buf_pos_;
  std::vector<const uint8_t*> bufs_;
  std::vector<size_t> buf_byte_sizes_;
};

}}  // namespace perfanalyzer::clientbackend
