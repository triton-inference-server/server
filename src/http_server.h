// Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <re2/re2.h>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include "common.h"
#include "data_compressor.h"
#include "shared_memory_manager.h"
#include "tracer.h"
#include "triton/common/logging.h"
#include "triton/core/tritonserver.h"

#include <evhtp/evhtp.h>

namespace triton { namespace server {

// Generic HTTP server using evhtp
class HTTPServer {
 public:
  virtual ~HTTPServer() { IGNORE_ERR(Stop()); }

  TRITONSERVER_Error* Start();
  TRITONSERVER_Error* Stop();

 protected:
  explicit HTTPServer(
      const int32_t port, const std::string address, const int thread_cnt)
      : port_(port), address_(address), thread_cnt_(thread_cnt)
  {
  }


  static void Dispatch(evhtp_request_t* req, void* arg);

 protected:
  virtual void Handle(evhtp_request_t* req) = 0;

  static void StopCallback(evutil_socket_t sock, short events, void* arg);

  int32_t port_;
  std::string address_;
  int thread_cnt_;

  evhtp_t* htp_;
  struct event_base* evbase_;
  std::thread worker_;
  evutil_socket_t fds_[2];
  event* break_ev_;
};

#ifdef TRITON_ENABLE_METRICS
// Handle HTTP requests to obtain prometheus metrics
class HTTPMetricsServer : public HTTPServer {
 public:
  static TRITONSERVER_Error* Create(
      const std::shared_ptr<TRITONSERVER_Server>& server, int32_t port,
      std::string address, int thread_cnt,
      std::unique_ptr<HTTPServer>* metrics_server);

  ~HTTPMetricsServer() = default;

 private:
  explicit HTTPMetricsServer(
      const std::shared_ptr<TRITONSERVER_Server>& server, const int32_t port,
      std::string address, const int thread_cnt)
      : HTTPServer(port, address, thread_cnt), server_(server),
        api_regex_(R"(/metrics/?)")
  {
  }
  void Handle(evhtp_request_t* req) override;

  std::shared_ptr<TRITONSERVER_Server> server_;
  re2::RE2 api_regex_;
};
#endif  // TRITON_ENABLE_METRICS

// HTTP API server that implements KFServing community standard inference
// protocols and extensions used by Triton.
class HTTPAPIServer : public HTTPServer {
 public:
  static TRITONSERVER_Error* Create(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& smb_manager,
      const int32_t port, std::string address, const int thread_cnt,
      std::unique_ptr<HTTPServer>* http_server);

  virtual ~HTTPAPIServer();

  //
  // AllocPayload
  //
  // Simple structure that carries the userp payload needed for
  // allocation.
  struct AllocPayload {
    struct OutputInfo {
      enum Kind { JSON, BINARY, SHM };

      Kind kind_;
      void* base_;
      uint64_t byte_size_;
      TRITONSERVER_MemoryType memory_type_;
      int64_t device_id_;
      uint32_t class_cnt_;
      evbuffer* evbuffer_;
      char* cuda_ipc_handle_;

      // For non-shared memory
      OutputInfo(Kind k, uint32_t class_cnt)
          : kind_(k), class_cnt_(class_cnt), evbuffer_(nullptr)
      {
      }

      // For shared memory
      OutputInfo(
          void* base, uint64_t byte_size, TRITONSERVER_MemoryType memory_type,
          int64_t device_id, char* cuda_ipc_handle)
          : kind_(SHM), base_(base), byte_size_(byte_size),
            memory_type_(memory_type), device_id_(device_id), class_cnt_(0),
            evbuffer_(nullptr), cuda_ipc_handle_(cuda_ipc_handle)
      {
      }

      ~OutputInfo()
      {
        if (evbuffer_ != nullptr) {
          evbuffer_free(evbuffer_);
        }
      }
    };

    ~AllocPayload()
    {
      for (auto it : output_map_) {
        delete it.second;
      }
    }

    AllocPayload() : default_output_kind_(OutputInfo::Kind::JSON){};
    std::unordered_map<std::string, OutputInfo*> output_map_;
    AllocPayload::OutputInfo::Kind default_output_kind_;
  };

  // Object associated with an inference request. This persists
  // information needed for the request and records the evhtp thread
  // that is bound to the request. This same thread must be used to
  // send the response.
  class InferRequestClass {
   public:
    explicit InferRequestClass(
        TRITONSERVER_Server* server, evhtp_request_t* req,
        DataCompressor::Type response_compression_type);
    virtual ~InferRequestClass() = default;

    evhtp_request_t* EvHtpRequest() const { return req_; }

    static void InferRequestComplete(
        TRITONSERVER_InferenceRequest* request, const uint32_t flags,
        void* userp);
    static void InferResponseComplete(
        TRITONSERVER_InferenceResponse* response, const uint32_t flags,
        void* userp);
    TRITONSERVER_Error* FinalizeResponse(
        TRITONSERVER_InferenceResponse* response);

    // Helper function to set infer response header in the form specified by
    // the endpoint protocol
    virtual void SetResponseHeader(
        const bool has_binary_data, const size_t header_length);

    uint32_t IncrementResponseCount();

#ifdef TRITON_ENABLE_TRACING
    std::shared_ptr<TraceManager::Trace> trace_;
#endif  // TRITON_ENABLE_TRACING

    AllocPayload alloc_payload_;

    // Data that cannot be used directly from the HTTP body is first
    // serialized. Hold that data here so that its lifetime spans the
    // lifetime of the request.
    std::list<std::vector<char>> serialized_data_;

   protected:
    TRITONSERVER_Server* server_;
    evhtp_request_t* req_;
    evthr_t* thread_;

    DataCompressor::Type response_compression_type_;

    // Counter to keep track of number of responses generated.
    std::atomic<uint32_t> response_count_;
  };

 protected:
  explicit HTTPAPIServer(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      const int32_t port, const std::string address, const int thread_cnt);
  virtual void Handle(evhtp_request_t* req) override;
  virtual std::unique_ptr<InferRequestClass> CreateInferRequest(
      evhtp_request_t* req)
  {
    return std::unique_ptr<InferRequestClass>(new InferRequestClass(
        server_.get(), req, GetResponseCompressionType(req)));
  }

  // Helper function to retrieve infer request header in the form specified by
  // the endpoint protocol
  //
  // Get the inference header length. Return 0 if the whole request body is
  // the inference header.
  virtual TRITONSERVER_Error* GetInferenceHeaderLength(
      evhtp_request_t* req, int32_t content_length, size_t* header_length);
  virtual DataCompressor::Type GetRequestCompressionType(evhtp_request_t* req);
  virtual DataCompressor::Type GetResponseCompressionType(evhtp_request_t* req);

  static TRITONSERVER_Error* InferResponseAlloc(
      TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
      size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
      int64_t preferred_memory_type_id, void* userp, void** buffer,
      void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
      int64_t* actual_memory_type_id);
  static TRITONSERVER_Error* OutputBufferQuery(
      TRITONSERVER_ResponseAllocator* allocator, void* userp,
      const char* tensor_name, size_t* byte_size,
      TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id);
  static TRITONSERVER_Error* OutputBufferAttributes(
      TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
      TRITONSERVER_BufferAttributes* buffer_attributes, void* userp,
      void* buffer_userp);
  static TRITONSERVER_Error* InferResponseFree(
      TRITONSERVER_ResponseAllocator* allocator, void* buffer,
      void* buffer_userp, size_t byte_size, TRITONSERVER_MemoryType memory_type,
      int64_t memory_type_id);
  void HandleServerHealth(evhtp_request_t* req, const std::string& kind);
  void HandleServerMetadata(evhtp_request_t* req);
  void HandleModelReady(
      evhtp_request_t* req, const std::string& model_name,
      const std::string& model_version_str);
  void HandleModelMetadata(
      evhtp_request_t* req, const std::string& model_name,
      const std::string& model_version_str);
  void HandleModelConfig(
      evhtp_request_t* req, const std::string& model_name,
      const std::string& model_version_str);
  void HandleInfer(
      evhtp_request_t* req, const std::string& model_name,
      const std::string& model_version_str);
  void HandleModelStats(
      evhtp_request_t* req, const std::string& model_name = "",
      const std::string& model_version_str = "");
  void HandleRepositoryIndex(
      evhtp_request_t* req, const std::string& repository_name);
  void HandleRepositoryControl(
      evhtp_request_t* req, const std::string& repository_name,
      const std::string& model_name, const std::string& action);
  void HandleSystemSharedMemory(
      evhtp_request_t* req, const std::string& region_name,
      const std::string& action);
  void HandleCudaSharedMemory(
      evhtp_request_t* req, const std::string& region_name,
      const std::string& action);
  void HandleTrace(evhtp_request_t* req, const std::string& model_name = "");

  TRITONSERVER_Error* EVBufferToInput(
      const std::string& model_name, TRITONSERVER_InferenceRequest* irequest,
      evbuffer* input_buffer, InferRequestClass* infer_req,
      size_t header_length);
  TRITONSERVER_Error* EVBufferToRawInput(
      const std::string& model_name, TRITONSERVER_InferenceRequest* irequest,
      evbuffer* input_buffer, InferRequestClass* infer_req);

  static void OKReplyCallback(evthr_t* thr, void* arg, void* shared);
  static void BADReplyCallback(evthr_t* thr, void* arg, void* shared);

  std::shared_ptr<TRITONSERVER_Server> server_;

  // Storing server metadata as it is consistent during server running
  TRITONSERVER_Error* server_metadata_err_;
  std::string server_metadata_;

  TraceManager* trace_manager_;
  std::shared_ptr<SharedMemoryManager> shm_manager_;

  // The allocator that will be used to allocate buffers for the
  // inference result tensors.
  TRITONSERVER_ResponseAllocator* allocator_;

  re2::RE2 server_regex_;
  re2::RE2 model_regex_;
  re2::RE2 modelcontrol_regex_;
  re2::RE2 systemsharedmemory_regex_;
  re2::RE2 cudasharedmemory_regex_;
  re2::RE2 trace_regex_;
};

}}  // namespace triton::server
