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

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

constexpr char kInferHeaderContentLengthHTTPHeader[] =
    "Inference-Header-Content-Length";
constexpr int MAX_GRPC_MESSAGE_SIZE = INT32_MAX;

namespace nvidia { namespace inferenceserver { namespace client {

class InferResult;
class InferRequest;
class RequestTimers;

#ifdef TRITON_INFERENCE_SERVER_CLIENT_CLASS
class TRITON_INFERENCE_SERVER_CLIENT_CLASS;
#endif

//==============================================================================
/// Error status reported by client API.
///
class Error {
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
/// Cumulative inference statistics.
///
/// \note
///   For GRPC protocol, 'cumulative_send_time_ns' represents the
///   time for marshaling infer request.
///   'cumulative_receive_time_ns' represents the time for
///   unmarshaling infer response.
struct InferStat {
  /// Total number of requests completed.
  size_t completed_request_count;

  /// Time from the request start until the response is completely
  /// received.
  uint64_t cumulative_total_request_time_ns;

  /// Time from the request start until the last byte is sent.
  uint64_t cumulative_send_time_ns;

  /// Time from receiving first byte of the response until the
  /// response is completely received.
  uint64_t cumulative_receive_time_ns;

  /// Create a new InferStat object with zero-ed statistics.
  InferStat()
      : completed_request_count(0), cumulative_total_request_time_ns(0),
        cumulative_send_time_ns(0), cumulative_receive_time_ns(0)
  {
  }
};

//==============================================================================
/// The base class for InferenceServerClients
///
class InferenceServerClient {
 public:
  using OnCompleteFn = std::function<void(InferResult*)>;

  explicit InferenceServerClient(bool verbose)
      : verbose_(verbose), exiting_(false)
  {
  }

  virtual ~InferenceServerClient() = default;

  /// Obtain the cumulative inference statistics of the client.
  /// \param Returns the InferStat object holding current statistics.
  /// \return Error object indicating success or failure.
  Error ClientInferStat(InferStat* infer_stat) const;

 protected:
  // Update the infer stat with the given timer
  Error UpdateInferStat(const RequestTimers& timer);
  // Enables verbose operation in the client.
  bool verbose_;

  // worker thread that will perform the asynchronous transfer
  std::thread worker_;
  // Avoid race condition between main thread and worker thread
  std::mutex mutex_;
  // Condition variable used for waiting on asynchronous request
  std::condition_variable cv_;
  // signal for worker thread to stop
  bool exiting_;

  // The inference statistic of the current client
  InferStat infer_stat_;
};

//==============================================================================
/// Structure to hold options for Inference Request.
///
struct InferOptions {
  explicit InferOptions(const std::string& model_name)
      : model_name_(model_name), model_version_(""), request_id_(""),
        sequence_id_(0), sequence_start_(false), sequence_end_(false),
        priority_(0), server_timeout_(0), client_timeout_(0)
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
  /// cannot be completed within the time by the server can take a
  /// model-specific action such as terminating the request. If not
  /// provided, the server will handle the request using default setting
  /// for the model.
  uint64_t server_timeout_;
  // The maximum end-to-end time, in microseconds, the request is allowed
  // to take. Note the HTTP library only offer the precision upto
  // milliseconds. The client will abort request when the specified time
  // elapses. The request will return error with message "Deadline Exceeded".
  // The default value is 0 which means client will wait for the
  // response from the server. This option is not supported for streaming
  // requests. Instead see 'stream_timeout' argument in
  // InferenceServerGrpcClient::StartStream().
  uint64_t client_timeout_;
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

  /// Gets name of the associated input tensor.
  /// \return The name of the tensor.
  const std::string& Name() const { return name_; }

  /// Gets datatype of the associated input tensor.
  /// \return The datatype of the tensor.
  const std::string& Datatype() const { return datatype_; }

  /// Gets the shape of the input tensor.
  /// \return The shape of the tensor.
  const std::vector<int64_t>& Shape() const { return shape_; }

  /// Set the shape of input associated with this object.
  /// \param dims the vector of dims representing the new shape
  /// of input.
  /// \return Error object indicating success or failure of the
  /// request.
  Error SetShape(const std::vector<int64_t>& dims);

  /// Prepare this input to receive new tensor values. Forget any
  /// existing values that were set by previous calls to SetSharedMemory()
  /// or AppendRaw().
  /// \return Error object indicating success or failure.
  Error Reset();

  /// Append tensor values for this input from a byte vector. The vector
  /// is not copied and so it must not be modified or destroyed
  /// until this input is no longer needed (that is until the Infer()
  /// call(s) that use the input have completed). Multiple calls can
  /// be made to this API to keep adding tensor data for this input.
  /// The data will be delivered in the order it was added.
  /// \param input The vector holding tensor values.
  /// \return Error object indicating success or failure.
  Error AppendRaw(const std::vector<uint8_t>& input);

  /// Append tensor values for this input from a byte array. The array
  /// is not copied and so it must not be modified or destroyed
  /// until this input is no longer needed (that is until the Infer()
  /// call(s) that use the input have completed). Multiple calls can
  /// be made to this API to keep adding tensor data for this input.
  /// The data will be delivered in the order it was added.
  /// \param input The pointer to the array holding the tensor value.
  /// \param input_byte_size The size of the array in bytes.
  /// \return Error object indicating success or failure.
  Error AppendRaw(const uint8_t* input, size_t input_byte_size);

  /// Set tensor values for this input by reference into a shared memory
  /// region. The values are not copied and so the shared memory region and
  /// its contents must not be modified or destroyed until this input is no
  /// longer needed (that is until the Infer() call(s) that use the input have
  /// completed. This function must be called a single time for an input that
  /// is using shared memory. The entire tensor data required by this input
  /// must be contiguous in a single shared memory region.
  /// \param name The user-given name for the registered shared memory region
  /// where the tensor values for this input is stored.
  /// \param byte_size The size, in bytes of the input tensor data. Must
  /// match the size expected for the input shape.
  /// \param offset The offset into the shared memory region upto the start
  /// of the input tensor values. The default value is 0.
  /// \return Error object indicating success or failure
  Error SetSharedMemory(
      const std::string& name, size_t byte_size, size_t offset = 0);

  /// \return true if this input is being provided in shared memory.
  bool IsSharedMemory() const { return (io_type_ == SHARED_MEMORY); }

  /// Get information about the shared memory being used for this
  /// input.
  /// \param name Returns the name of the shared memory region.
  /// \param byte_size Returns the size, in bytes, of the shared
  /// memory region.
  /// \param offset Returns the offset within the shared memory
  /// region.
  /// \return Error object indicating success or failure.
  Error SharedMemoryInfo(
      std::string* name, size_t* byte_size, size_t* offset) const;

  /// Append tensor values for this input from a vector or
  /// strings. This method can only be used for tensors with BYTES
  /// data-type. The strings are assigned in row-major order to the
  /// elements of the tensor. The strings are copied and so the
  /// 'input' does not need to be preserved as with AppendRaw(). Multiple
  /// calls can be made to this API to keep adding tensor data for
  /// this input. The data will be delivered in the order it was added.
  /// \param input The vector holding tensor string values.
  /// \return Error object indicating success or failure.
  Error AppendFromString(const std::vector<std::string>& input);

  /// Gets the size of data added into this input in bytes.
  /// \param byte_size The size of data added in bytes.
  /// \return Error object indicating success or failure.
  Error ByteSize(size_t* byte_size) const;

 private:
#ifdef TRITON_INFERENCE_SERVER_CLIENT_CLASS
  friend TRITON_INFERENCE_SERVER_CLIENT_CLASS;
#endif

  InferInput(
      const std::string& name, const std::vector<int64_t>& dims,
      const std::string& datatype);

  Error PrepareForRequest();
  Error GetNext(
      uint8_t* buf, size_t size, size_t* input_bytes, bool* end_of_input);
  Error GetNext(const uint8_t** buf, size_t* input_bytes, bool* end_of_input);

  std::string name_;
  std::vector<int64_t> shape_;
  std::string datatype_;
  size_t byte_size_;

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
  /// Create a InferRequestedOutput instance that describes a model output being
  /// requested.
  /// \param infer_output Returns a new InferOutputGrpc object.
  /// \param name The name of output being requested.
  /// \param class_count The number of classifications to be requested. The
  /// default value is 0 which means the classification results are not
  /// requested.
  /// \return Error object indicating success or failure.
  static Error Create(
      InferRequestedOutput** infer_output, const std::string& name,
      const size_t class_count = 0);

  /// Gets name of the associated output tensor.
  /// \return The name of the tensor.
  const std::string& Name() const { return name_; }

  /// Get the number of classifications requested for this output, or
  /// 0 if the output is not being returned as classifications.
  size_t ClassificationCount() const { return class_count_; }

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

  /// Clears the shared memory option set by the last call to
  /// InferRequestedOutput::SetSharedMemory(). After call to this
  /// function requested output will no longer be returned in a
  /// shared memory region.
  /// \return Error object indicating success or failure of the
  /// request.
  Error UnsetSharedMemory();

  /// \return true if this output is being returned in shared memory.
  bool IsSharedMemory() const { return (io_type_ == SHARED_MEMORY); }

  /// Get information about the shared memory being used for this
  /// output.
  /// \param name Returns the name of the shared memory region.
  /// \param byte_size Returns the size, in bytes, of the shared
  /// memory region.
  /// \param offset Returns the offset within the shared memory
  /// region.
  /// \return Error object indicating success or failure.
  Error SharedMemoryInfo(
      std::string* name, size_t* byte_size, size_t* offset) const;

 private:
#ifdef TRITON_INFERENCE_SERVER_CLIENT_CLASS
  friend TRITON_INFERENCE_SERVER_CLIENT_CLASS;
#endif

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
  virtual ~InferResult() = default;

  /// Get the name of the model which generated this response.
  /// \param name Returns the name of the model.
  /// \return Error object indicating success or failure.
  virtual Error ModelName(std::string* name) const = 0;

  /// Get the version of the model which generated this response.
  /// \param version Returns the version of the model.
  /// \return Error object indicating success or failure.
  virtual Error ModelVersion(std::string* version) const = 0;

  /// Get the id of the request which generated this response.
  /// \param version Returns the version of the model.
  /// \return Error object indicating success or failure.
  virtual Error Id(std::string* id) const = 0;

  /// Get the shape of output result returned in the response.
  /// \param output_name The name of the ouput to get shape.
  /// \param shape Returns the shape of result for specified output name.
  /// \return Error object indicating success or failure.
  virtual Error Shape(
      const std::string& output_name, std::vector<int64_t>* shape) const = 0;

  /// Get the datatype of output result returned in the response.
  /// \param output_name The name of the ouput to get datatype.
  /// \param shape Returns the datatype of result for specified output name.
  /// \return Error object indicating success or failure.
  virtual Error Datatype(
      const std::string& output_name, std::string* datatype) const = 0;

  /// Get access to the buffer holding raw results of specified output
  /// returned by the server. Note the buffer is owned by InferResult
  /// instance. Users can copy out the data if required to extend the
  /// lifetime.
  /// \param output_name The name of the output to get result data.
  /// \param buf Returns the pointer to the start of the buffer.
  /// \param byte_size Returns the size of buffer in bytes.
  /// \return Error object indicating success or failure of the
  /// request.
  virtual Error RawData(
      const std::string& output_name, const uint8_t** buf,
      size_t* byte_size) const = 0;

  /// Get the result data as a vector of strings. The vector will
  /// receive a copy of result data. An error will be generated if
  /// the datatype of output is not 'BYTES'.
  /// \param output_name The name of the output to get result data.
  /// \param string_result Returns the result data represented as
  /// a vector of strings. The strings are stored in the
  /// row-major order.
  /// \return Error object indicating success or failure of the
  /// request.
  virtual Error StringData(
      const std::string& output_name,
      std::vector<std::string>* string_result) const = 0;

  /// Returns the complete response as a user friendly string.
  /// \return The string describing the complete response.
  virtual std::string DebugString() const = 0;

  /// Returns the status of the request.
  /// \return Error object indicating the success or failure of the
  /// request.
  virtual Error RequestStatus() const = 0;
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
/// The base class to describe an inflight inference request.
///
class InferRequest {
 public:
  InferRequest(
      InferenceServerClient::OnCompleteFn callback = nullptr,
      const bool verbose = false)
      : callback_(callback), verbose_(verbose)
  {
  }
  virtual ~InferRequest() = default;

  RequestTimers& Timer() { return timer_; }

 protected:
  InferenceServerClient::OnCompleteFn callback_;
  const bool verbose_;

 private:
  // The timers for infer request.
  RequestTimers timer_;
};


}}}  // namespace nvidia::inferenceserver::client
