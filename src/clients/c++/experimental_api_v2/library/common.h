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

#ifdef _MSC_VER
#ifdef DLL_EXPORTING
#define DECLSPEC __declspec(dllexport)
#else
#define DECLSPEC __declspec(dllimport)
#endif
#else
#define DECLSPEC
#endif


#include <iostream>
#include <string>
#include <vector>


namespace nvidia { namespace inferenceserver { namespace client {

//==============================================================================
/// Error status reported by client API.
///
class DECLSPEC Error {
 public:
  /// Create an error with the specified message.
  /// \param msg The message for the error
  explicit Error(const std::string& msg = "");

  /// Accessor for the message of this error.
  /// \return The messsage for the error. Empty if no error.
  const std::string& Message() const { return msg_; }


  /// Does this error indicate OK status?
  /// \return True if this error indicates "ok"/"success", false if
  /// error indicates a failure.
  bool IsOk() const { return msg_.empty(); }

  /// Convenience "success" value. Can be used as Error::Success to
  /// indicate no error.
  static const Error Success;

 private:
  friend std::ostream& operator<<(std::ostream&, const Error&);
  std::string msg_;
};

//==============================================================================
/// Structure to hold options for Inference Request.
///
struct InferOptions {
  explicit InferOptions(const std::string& model_name)
      : model_name_(model_name), model_version_(""), request_id_(""),
        sequence_id_(0), sequence_start_(false), sequence_end_(false),
        priority_(0), timeout_(0)
  {
  }
  /// The name of the model to run inference.
  std::string model_name_;
  /// The version of the model to use while running inference. The default
  /// value is an empty string which means the server will select the
  /// version of the model based on its internal policy.
  std::string model_version_;
  /// An identifier for the request. If specified will be returned
  /// in the response. Default value is an empty string which means no
  /// request_id will be used.
  std::string request_id_;
  /// The unique identifier for the sequence being represented by the
  /// object. Default value is 0 which means that the request does not
  /// belong to a sequence.
  uint64_t sequence_id_;
  /// Indicates whether the request being added marks the start of the
  /// sequence. Default value is False. This argument is ignored if
  /// 'sequence_id' is 0.
  bool sequence_start_;
  /// Indicates whether the request being added marks the end of the
  /// sequence. Default value is False. This argument is ignored if
  /// 'sequence_id' is 0.
  bool sequence_end_;
  /// Indicates the priority of the request. Priority value zero
  /// indicates that the default priority level should be used
  /// (i.e. same behavior as not specifying the priority parameter).
  /// Lower value priorities indicate higher priority levels. Thus
  /// the highest priority level is indicated by setting the parameter
  /// to 1, the next highest is 2, etc. If not provided, the server
  /// will handle the request using default setting for the model.
  uint64_t priority_;
  /// The timeout value for the request, in microseconds. If the request
  /// cannot be completed within the time the server can take a
  /// model-specific action such as terminating the request. If not
  /// provided, the server will handle the request using default setting
  /// for the model.
  uint64_t timeout_;
};

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
  /// execution. Note the buffer is owned by InferResult instance.
  /// Users can copy out the data if required to extend the lifetime.
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
