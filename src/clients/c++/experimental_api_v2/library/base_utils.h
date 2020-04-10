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

/// \file

#include <vector>
#include "src/clients/c++/experimental_api_v2/library/common_utils.h"

namespace nvidia { namespace inferenceserver { namespace client {

//==============================================================================
/// An interface for InferInput object to describe the model input for
/// inference.
///
class InferInput {
 public:
  /// Get the name of input associated with this object
  /// \param name Returns the name of input.
  /// \return Error object indicating success or failure of the
  /// request.
  virtual Error GetName(std::string* name) const = 0;

  /// Get the datatype of input associated with this object
  /// \param datatype Returns the datatype of input.
  /// \return Error object indicating success or failure of the
  /// request.
  virtual Error GetDatatype(std::string* datatype) const = 0;

  /// Get the shape of input associated with this object.
  /// \param dims Returns the vector of dims representing the
  /// shape of input.
  /// \return Error object indicating success or failure of the
  /// request.
  virtual Error GetShape(std::vector<int64_t>* dims) const = 0;

  /// Set the shape of input associated with this object.
  /// \param dims the vector of dims representing the new shape
  /// of input.
  /// \return Error object indicating success or failure of the
  /// request.
  virtual Error SetShape(const std::vector<int64_t>& dims) = 0;

  /// Set the tensor data from the specified buffer described as
  /// starting address and buffer size.
  /// \param input The base pointer of buffer holding the input data.
  /// \param input_byte_size The size of buffer in bytes.
  /// \return Error object indicating success or failure of the
  /// request.
  virtual Error SetRaw(const uint8_t* input, size_t input_byte_size) = 0;

  /// Set the tensor data from the specified buffer represented
  /// as vector.
  /// \param input The vector holding input data.
  /// \return Error object indicating success or failure of the
  /// request.
  virtual Error SetRaw(const std::vector<uint8_t>& input) = 0;

  /// Set the tensor data from the specified vector of strings, where
  /// each string represents an lement of the "BYTES" tensor ordered
  /// in row-major format.
  /// \param input The vector holding strings.
  /// \return Error object indicating success or failure of the
  /// request.
  virtual Error SetFromString(const std::vector<std::string>& input) = 0;

  /// Set the tensor data to be read from a registered shared memory region.
  /// \param region_name The name of the shared memory region.
  /// \param byte_size The size of data in bytes.
  /// \param offset The offset in shared memory region. Default value is 0.
  /// \return Error object indicating success or failure of the
  /// request.
  virtual Error SetSharedMemory(
      const std::string& region_name, const size_t byte_size,
      const size_t offset = 0) = 0;
};

//==============================================================================
/// An interface for InferOutput object to describe the requested model
/// output for inference.
///
class InferOutput {
 public:
  /// Get the name of output associated with this object
  /// \param name Returns the name of output.
  /// \return Error object indicating success or failure of the
  /// request.
  virtual Error GetName(std::string* name) const = 0;

  /// Set the output tensor data to be written to specified shared
  /// memory region.
  /// \param region_name The name of the shared memory region.
  /// \param byte_size The size of data in bytes.
  /// \param offset The offset in shared memory region. Default value is 0.
  /// \return Error object indicating success or failure of the
  /// request.
  virtual Error SetSharedMemory(
      const std::string& region_name, const size_t byte_size,
      const size_t offset = 0) = 0;
};

//==============================================================================
/// An interface for InferResult object which allows to access and
/// interpret the response from inference request.
///
class InferResult {
 public:
  /// Get the shape of output returned in the response.
  /// \param output_name The name of the output to get shape.
  /// \param shape Returns the vector of integers representing the
  /// shape of output.
  /// \return Error object indicating success or failure of the
  /// request.
  virtual Error GetShape(
      const std::string& output_name, std::vector<int64_t>* shape) const = 0;

  /// Get the datatype of output returned in the response.
  /// \param output_name The name of the output to get datatype.
  /// \param datatype Returns the datatype string.
  /// \return Error object indicating success or failure of the
  /// request.
  virtual Error GetDatatype(
      const std::string& output_name, std::string* datatype) const = 0;

  /// Get access to the buffer holding raw results from the inference
  /// execution.
  /// \param output_name The name of the output to get datatype.
  /// \param buf Returns the pointer to the start of the buffer.
  /// \param byte_size Returns the size of buffer in bytes.
  /// \return Error object indicating success or failure of the
  /// request.
  virtual Error GetRaw(
      const std::string& output_name, const uint8_t** buf,
      size_t* byte_size) const = 0;
};

}}}  // namespace nvidia::inferenceserver::client
