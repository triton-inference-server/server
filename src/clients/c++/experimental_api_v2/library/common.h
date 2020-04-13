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

#include "src/core/constants.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <list>
#include <memory>
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
/// Records timestamps for different stages of request handling.
///
class RequestTimers {
 public:
  /// Timestamp kinds.
  enum class Kind {
    /// The start of request handling.
    REQUEST_START,

    /// The end of request handling.
    REQUEST_END,

    /// The start of sending request bytes to the server (i.e. first
    /// byte).
    SEND_START,

    /// The end of sending request bytes to the server (i.e. last
    /// byte).
    SEND_END,

    /// The start of receiving response bytes from the server
    /// (i.e. first byte).
    RECV_START,

    /// The end of receiving response bytes from the server (i.e. last
    /// byte).
    RECV_END,

    COUNT__
  };

  /// Construct a timer with zero-ed timestamps.
  RequestTimers() : timestamps_((size_t)Kind::COUNT__) { Reset(); }

  /// Reset all timestamp values to zero. Must be called before
  /// re-using the timer.
  void Reset()
  {
    memset(&timestamps_[0], 0, sizeof(uint64_t) * timestamps_.size());
  }

  /// Get the timestamp, in nanoseconds, for a kind.
  /// \param kind The timestamp kind.
  /// \return The timestamp in nanoseconds.
  uint64_t Timestamp(Kind kind) const { return timestamps_[(size_t)kind]; }

  /// Set a timestamp to the current time, in nanoseconds.
  /// \param kind The timestamp kind.
  /// \return The timestamp in nanoseconds.
  uint64_t CaptureTimestamp(Kind kind)
  {
    uint64_t& ts = timestamps_[(size_t)kind];
    ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
             .count();
    return ts;
  }

  /// Return the duration between start time point and end timepoint
  /// in nanosecond.
  /// \param start The start time point.
  /// \param end The end time point.
  /// \return Duration in nanosecond, or
  /// std::numeric_limits<uint64_t>::max to indicate that duration
  /// could not be calculated.
  uint64_t Duration(Kind start, Kind end) const
  {
    const uint64_t stime = timestamps_[(size_t)start];
    const uint64_t etime = timestamps_[(size_t)end];

    // If the start or end timestamp is 0 then can't calculate the
    // duration, so return max to indicate error.
    if ((stime == 0) || (etime == 0)) {
      return (std::numeric_limits<uint64_t>::max)();
    }

    return (stime > etime) ? (std::numeric_limits<uint64_t>::max)()
                           : etime - stime;
  }

 private:
  std::vector<uint64_t> timestamps_;
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
  /// Create a InferInput instance that describes a model input.
  /// \param infer_input Returns a new InferInput object.
  /// \param name The name of input whose data will be described by this object.
  /// \param dims The shape of the input.
  /// \param datatype The datatype of the input.
  /// \return Error object indicating success or failure.
  static Error Create(
      InferInput** infer_input, const std::string& name,
      const std::vector<int64_t>& dims, const std::string& datatype);

  const std::string& Name() const { return name_; }
  int64_t ByteSize() const { return byte_size_; }
  size_t TotalSendByteSize() const { return total_send_byte_size_; }
  const std::string& Datatype() const { return datatype_; }
  const std::vector<int64_t>& Shape() const { return shape_; }
  bool IsSharedMemory() const { return (io_type_ == SHARED_MEMORY); }

  Error SetShape(const std::vector<int64_t>& dims);

  Error Reset();
  Error SetRaw(const std::vector<uint8_t>& input);
  Error SetRaw(const uint8_t* input, size_t input_byte_size);
  Error SetSharedMemory(
      const std::string& name, size_t byte_size, size_t offset = 0);
  Error SetFromString(const std::vector<std::string>& input);

  // Copy the shared memory key, offset and batch_byte_size
  Error SharedMemoryInfo(
      std::string* name, size_t* batch_byte_size, size_t* offset) const;

  Error PrepareForRequest();

  // Copy into 'buf' up to 'size' bytes of this input's data. Return
  // the actual amount copied in 'input_bytes' and if the end of input
  // is reached in 'end_of_input'
  Error GetNext(
      uint8_t* buf, size_t size, size_t* input_bytes, bool* end_of_input);

  Error GetNext(const uint8_t** buf, size_t* input_bytes, bool* end_of_input);

 private:
  InferInput(
      const std::string& name, const std::vector<int64_t>& dims,
      const std::string& datatype);

  std::string name_;
  std::vector<int64_t> shape_;
  std::string datatype_;
  size_t byte_size_;
  size_t total_send_byte_size_;

  size_t bufs_idx_, buf_pos_;
  std::vector<const uint8_t*> bufs_;
  std::vector<size_t> buf_byte_sizes_;

  // Used only for STRING type tensors set with SetFromString(). Hold
  // the "raw" serialization of the string values for each index
  // that are then referenced by 'bufs_'. A std::list is used to avoid
  // reallocs that could invalidate the pointer references into the
  // std::string objects.
  std::list<std::string> str_bufs_;

  // Used only if working with Shared Memory
  enum IOType { NONE, RAW, SHARED_MEMORY };
  IOType io_type_;
  std::string shm_name_;
  size_t shm_offset_;
};

//==============================================================================
/// An InferRequestedOutput object is used to describe the requested model
/// output for inference.
///
class InferRequestedOutput {
 public:
  static Error Create(
      InferRequestedOutput** infer_output, const std::string& name,
      const size_t class_count = 0);

  /// Get the name of output associated with this object
  /// \param name Returns the name of output.
  /// \return Error object indicating success or failure of the
  /// request.
  const std::string& Name() const { return name_; }
  size_t ClassCount() const { return class_count_; }
  bool IsSharedMemory() const { return (io_type_ == SHARED_MEMORY); }


  /// Set the output tensor data to be written to specified shared
  /// memory region.
  /// \param region_name The name of the shared memory region.
  /// \param byte_size The size of data in bytes.
  /// \param offset The offset in shared memory region. Default value is 0.
  /// \return Error object indicating success or failure of the
  /// request.
  Error SetSharedMemory(
      const std::string& region_name, const size_t byte_size,
      const size_t offset = 0);

  // Copy the shared memory key, offset and batch_byte_size
  Error SharedMemoryInfo(
      std::string* name, size_t* batch_byte_size, size_t* offset) const;

 private:
  explicit InferRequestedOutput(
      const std::string& name, const size_t class_count = 0);

  std::string name_;
  size_t class_count_;

  // Used only if working with Shared Memory
  enum IOType { NONE, RAW, SHARED_MEMORY };
  IOType io_type_;
  std::string shm_name_;
  size_t shm_byte_size_;
  size_t shm_offset_;
};

//==============================================================================
/// An interface for InferResult object to interpret the response to an
/// inference request.
///
class InferResult {
 public:
  virtual Error ModelName(std::string* name) const = 0;
  virtual Error ModelVersion(std::string* version) const = 0;
  virtual Error Id(std::string* id) const = 0;
  virtual Error Shape(
      const std::string& output_name, std::vector<int64_t>* shape) const = 0;

  virtual Error Datatype(
      const std::string& output_name, std::string* datatype) const = 0;

  virtual Error RawData(
      const std::string& output_name, const uint8_t** buf,
      size_t* byte_size) const = 0;

  virtual std::string DebugString() const = 0;
};

}}}  // namespace nvidia::inferenceserver::client
