// Copyright 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <evhtp/evhtp.h>
#include <re2/re2.h>

#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>

#include "common.h"
#include "data_compressor.h"
#include "orca_http.h"
#include "restricted_features.h"
#include "shared_memory_manager.h"
#include "tracer.h"
#include "triton/common/logging.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace server {

class MappingSchema {
 public:
  enum class Kind {
    EXACT_MAPPING,
    // An object of this kind means it is a nested mapping schema.
    MAPPING_SCHEMA
  };
  std::map<std::string, std::unique_ptr<MappingSchema>> children_;
  // Whether an unspecified key is allowed. If true,
  // * for requests, the unspecified key will be converted to Triton input
  //   following the EXACT_MAPPING rule.
  // * for responses, the Triton output will be converted to JSON key-value
  //   pairs at top level if the name is unspecified in the schema,
  //   following the EXACT_MAPPING rule.
  const bool allow_unspecified_{true};
  const Kind kind_{Kind::EXACT_MAPPING};

  explicit MappingSchema(
      const MappingSchema::Kind& kind = Kind::EXACT_MAPPING,
      const bool& allow_unspecified = true)
      : allow_unspecified_(allow_unspecified), kind_(kind)
  {
  }


 private:
};

// Generic HTTP server using evhtp
class HTTPServer {
 public:
  virtual ~HTTPServer() { IGNORE_ERR(Stop()); }

  TRITONSERVER_Error* Start();
  TRITONSERVER_Error* Stop(
      uint32_t* exit_timeout_secs = nullptr,
      const std::string& service_name = "HTTP");

 protected:
  explicit HTTPServer(
      const int32_t port, const bool reuse_port, const std::string& address,
      const std::string& header_forward_pattern, const int thread_cnt)
      : port_(port), reuse_port_(reuse_port), address_(address),
        header_forward_pattern_(header_forward_pattern),
        thread_cnt_(thread_cnt), header_forward_regex_(header_forward_pattern_),
        conn_cnt_(0), accepting_new_conn_(true)
  {
  }


  static void Dispatch(evhtp_request_t* req, void* arg);

 protected:
  virtual void Handle(evhtp_request_t* req) = 0;

  static void StopCallback(evutil_socket_t sock, short events, void* arg);

  static evhtp_res NewConnection(evhtp_connection_t* conn, void* arg);
  static evhtp_res EndConnection(evhtp_connection_t* conn, void* arg);

  int32_t port_;
  bool reuse_port_;
  std::string address_;
  std::string header_forward_pattern_;
  int thread_cnt_;
  re2::RE2 header_forward_regex_;

  evhtp_t* htp_;
  struct event_base* evbase_;
  std::thread worker_;
  evutil_socket_t fds_[2];
  event* break_ev_;

  std::mutex conn_mu_;
  uint32_t conn_cnt_;
  bool accepting_new_conn_;
};

#ifdef TRITON_ENABLE_METRICS
// Handle HTTP requests to obtain prometheus metrics
class HTTPMetricsServer : public HTTPServer {
 public:
  static TRITONSERVER_Error* Create(
      const std::shared_ptr<TRITONSERVER_Server>& server, int32_t port,
      std::string address, int thread_cnt,
      std::unique_ptr<HTTPServer>* metrics_server);

  static TRITONSERVER_Error* Create(
      std::shared_ptr<TRITONSERVER_Server>& server,
      const UnorderedMapType& options, std::unique_ptr<HTTPServer>* service);

  ~HTTPMetricsServer() = default;

 private:
  explicit HTTPMetricsServer(
      const std::shared_ptr<TRITONSERVER_Server>& server, const int32_t port,
      std::string address, const int thread_cnt)
      : HTTPServer(
            port, false /* reuse_port */, address,
            "" /* header_forward_pattern */, thread_cnt),
        server_(server), api_regex_(R"(/metrics/?)")
  {
  }
  void Handle(evhtp_request_t* req) override;

  std::shared_ptr<TRITONSERVER_Server> server_;
  re2::RE2 api_regex_;
};
#endif  // TRITON_ENABLE_METRICS

#if !defined(_WIN32) && defined(TRITON_ENABLE_TRACING)
class HttpTextMapCarrier : public otel_cntxt::propagation::TextMapCarrier {
 public:
  HttpTextMapCarrier(evhtp_kvs_t* headers) : headers_(headers) {}
  HttpTextMapCarrier() = default;
  virtual opentelemetry::nostd::string_view Get(
      opentelemetry::nostd::string_view key) const noexcept override
  {
    std::string key_to_compare = key.data();
    auto it = evhtp_kv_find(headers_, key_to_compare.c_str());
    if (it != NULL) {
      return opentelemetry::nostd::string_view(it);
    }
    return "";
  }
  // Not required on server side
  virtual void Set(
      opentelemetry::nostd::string_view key,
      opentelemetry::nostd::string_view value) noexcept override
  {
    return;
  }

  evhtp_kvs_t* headers_;
};
#else
using HttpTextMapCarrier = void*;
#endif


// HTTP API server that implements KFServing community standard inference
// protocols and extensions used by Triton.
class HTTPAPIServer : public HTTPServer {
 public:
  static TRITONSERVER_Error* Create(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& smb_manager,
      const int32_t port, const bool reuse_port, const std::string& address,
      const std::string& header_forward_pattern, const int thread_cnt,
      const size_t max_input_size, const RestrictedFeatures& restricted_apis,
      std::unique_ptr<HTTPServer>* http_server);

  static TRITONSERVER_Error* Create(
      std::shared_ptr<TRITONSERVER_Server>& server,
      const UnorderedMapType& options,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      const RestrictedFeatures& restricted_features,
      std::unique_ptr<HTTPServer>* service);

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
    // [FIXME] decompression / compression should be handled implicitly
    // within InferRequestClass. This alleviate the check for decompressed
    // buffer in HTTPServer code.
    explicit InferRequestClass(
        TRITONSERVER_Server* server, evhtp_request_t* req,
        DataCompressor::Type response_compression_type,
        const std::shared_ptr<TRITONSERVER_InferenceRequest>& triton_request,
        const std::shared_ptr<SharedMemoryManager>& shm_manager);

    virtual ~InferRequestClass()
    {
      if (req_ != nullptr) {
        evhtp_request_unset_hook(req_, evhtp_hook_on_request_fini);
      }
      req_ = nullptr;

      // Unregister shm regions that are waiting for the completion of an
      // inference.
      while (!shm_regions_info_.empty()) {
        auto shm_name = shm_regions_info_.back()->name_;
        auto shm_memory_type = shm_regions_info_.back()->kind_;
        auto awaiting_unregister =
            shm_regions_info_.back()->awaiting_unregister_;

        // Delete shared_ptr to decrement reference count
        shm_regions_info_.pop_back();

        if (awaiting_unregister) {
          if (shm_manager_ != nullptr) {
            auto err = shm_manager_->Unregister(shm_name, shm_memory_type);
            if (err != nullptr) {
              LOG_VERBOSE(1) << TRITONSERVER_ErrorMessage(err);
            }
          } else {
            LOG_VERBOSE(1) << "Shared memory manager is not available";
          }
        }
      }
    }

    evhtp_request_t* EvHtpRequest() const { return req_; }

    static void InferRequestComplete(
        TRITONSERVER_InferenceRequest* request, const uint32_t flags,
        void* userp);
    static void InferResponseComplete(
        TRITONSERVER_InferenceResponse* response, const uint32_t flags,
        void* userp);
    virtual TRITONSERVER_Error* FinalizeResponse(
        TRITONSERVER_InferenceResponse* response);

    // Helper function to set infer response header in the form specified by
    // the endpoint protocol
    virtual void SetResponseHeader(
        const bool has_binary_data, const size_t header_length);

    uint32_t IncrementResponseCount();

    // Only used if tracing enabled
    std::shared_ptr<TraceManager::Trace> trace_;

    AllocPayload alloc_payload_;

    // Data that cannot be used directly from the HTTP body is first
    // serialized. Hold that data here so that its lifetime spans the
    // lifetime of the request.
    std::list<std::vector<char>> serialized_data_;

    static void ReplyCallback(evthr_t* thr, void* arg, void* shared);

    void AddShmRegionInfo(
        const std::shared_ptr<const SharedMemoryManager::SharedMemoryInfo>&
            shm_info)
    {
      shm_regions_info_.push_back(shm_info);
    }

   protected:
    TRITONSERVER_Server* server_{nullptr};
    evhtp_request_t* req_{nullptr};
    evthr_t* thread_{nullptr};

    DataCompressor::Type response_compression_type_{
        DataCompressor::Type::IDENTITY};

    // Counter to keep track of number of responses generated.
    std::atomic<uint32_t> response_count_{0};

    // Event hook for called before request deletion
    static evhtp_res RequestFiniHook(evhtp_request* req, void* arg);

    // Pointer to associated Triton request, this class does not own the
    // request and must not reference it after a successful
    // TRITONSERVER_ServerInferAsync (except for cancellation).
    std::shared_ptr<TRITONSERVER_InferenceRequest> triton_request_{nullptr};

    // Maintain shared pointers(read-only reference) to the shared memory
    // block's information for the shared memory regions used by the request.
    // These pointers will automatically increase the usage count, preventing
    // unregistration of the shared memory. This vector must be cleared when no
    // longer needed to decrease the count and permit unregistration.
    std::vector<std::shared_ptr<const SharedMemoryManager::SharedMemoryInfo>>
        shm_regions_info_;

    std::shared_ptr<SharedMemoryManager> shm_manager_;

    evhtp_res response_code_{EVHTP_RES_OK};
  };

  class GenerateRequestClass : public InferRequestClass {
   public:
    explicit GenerateRequestClass(
        TRITONSERVER_Server* server, evhtp_request_t* req,
        DataCompressor::Type response_compression_type,
        const MappingSchema* request_schema,
        const MappingSchema* response_schema, bool streaming,
        const std::shared_ptr<TRITONSERVER_InferenceRequest>& triton_request,
        const std::shared_ptr<SharedMemoryManager>& shm_manager)
        : InferRequestClass(
              server, req, response_compression_type, triton_request,
              shm_manager),
          request_schema_(request_schema), response_schema_(response_schema),
          streaming_(streaming)
    {
    }
    virtual ~GenerateRequestClass();

    TRITONSERVER_Server* EvHtpServer() const { return server_; }

    // [FIXME] Specialize response complete function for now, should have
    // been a dispatcher and call into object specific response function.
    static void InferResponseComplete(
        TRITONSERVER_InferenceResponse* response, const uint32_t flags,
        void* userp);
    static void ChunkResponseCallback(evthr_t* thr, void* arg, void* shared);
    static void EndResponseCallback(evthr_t* thr, void* arg, void* shared);
    // Return whether the response is ending
    void SendChunkResponse(bool end);

    // Response preparation
    TRITONSERVER_Error* FinalizeResponse(
        TRITONSERVER_InferenceResponse* response) override;
    void AddErrorJson(TRITONSERVER_Error* error);
    static void StartResponse(evthr_t* thr, void* arg, void* shared);

    // [DLIS-5551] currently always performs basic conversion, only maps schema
    // of EXACT_MAPPING kind. MAPPING_SCHEMA and upcoming kinds are for
    // customized conversion where a detailed schema will be provided.
    TRITONSERVER_Error* ConvertGenerateRequest(
        std::map<std::string, triton::common::TritonJson::Value>&
            input_metadata,
        const MappingSchema* schema,
        triton::common::TritonJson::Value& generate_request);

    const MappingSchema* RequestSchema() { return request_schema_; }
    const MappingSchema* ResponseSchema() { return response_schema_; }

   private:
    struct TritonOutput {
      enum class Type { RESERVED, TENSOR, PARAMETER };
      TritonOutput(Type t, const std::string& val) : type(t), value(val) {}
      explicit TritonOutput(Type t, uint32_t i) : type(t), index(i) {}
      Type type;
      // RESERVED type
      std::string value;
      // TENSOR, PARAMETER type
      uint32_t index;
    };

    TRITONSERVER_Error* ExactMappingInput(
        const std::string& name, triton::common::TritonJson::Value& value,
        std::map<std::string, triton::common::TritonJson::Value>&
            input_metadata);

    // [DLIS-5551] currently always performs basic conversion, only maps schema
    // of EXACT_MAPPING kind. MAPPING_SCHEMA and upcoming kinds are for
    // customized conversion where a detailed schema will be provided.
    TRITONSERVER_Error* ConvertGenerateResponse(
        const std::map<std::string, TritonOutput>& output_metadata,
        const MappingSchema* schema,
        triton::common::TritonJson::Value* generate_response,
        std::set<std::string>* mapped_outputs);
    TRITONSERVER_Error* ExactMappingOutput(
        const std::string& name, const TritonOutput& triton_output,
        triton::common::TritonJson::Value* generate_response,
        std::set<std::string>* mapped_outputs);

    const MappingSchema* request_schema_{nullptr};
    const MappingSchema* response_schema_{nullptr};
    const bool streaming_{false};
    // Placeholder to completing response, this class does not own
    // the response.
    TRITONSERVER_InferenceResponse* triton_response_{nullptr};
    // As InferResponseComplete and ChunkResponseCallback are called in
    // different threads, need to have dedicated buffers for each response and
    // ensure mutual exclusive access.
    std::mutex res_mtx_;
    std::queue<evbuffer*> pending_http_responses_;
    bool end_{false};
  };

  // Simple structure that carries the userp payload needed for
  // request release callback.
  struct RequestReleasePayload final {
    RequestReleasePayload(
        const std::shared_ptr<TRITONSERVER_InferenceRequest>& inference_request,
        evbuffer* buffer)
        : inference_request_(inference_request), buffer_(buffer){};

    ~RequestReleasePayload()
    {
      if (buffer_ != nullptr) {
        evbuffer_free(buffer_);
      }
    };

   private:
    std::shared_ptr<TRITONSERVER_InferenceRequest> inference_request_ = nullptr;
    evbuffer* buffer_ = nullptr;
  };


 protected:
  explicit HTTPAPIServer(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      const int32_t port, const bool reuse_port, const std::string& address,
      const std::string& header_forward_pattern, const int thread_cnt,
      const size_t max_input_size = HTTP_DEFAULT_MAX_INPUT_SIZE,
      const RestrictedFeatures& restricted_apis = {});

  virtual void Handle(evhtp_request_t* req) override;
  // [FIXME] extract to "infer" class
  virtual std::unique_ptr<InferRequestClass> CreateInferRequest(
      evhtp_request_t* req,
      const std::shared_ptr<TRITONSERVER_InferenceRequest>& triton_request)
  {
    return std::unique_ptr<InferRequestClass>(new InferRequestClass(
        server_.get(), req, GetResponseCompressionType(req), triton_request,
        shm_manager_));
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


  TRITONSERVER_Error* GetModelConfig(
      const std::string& model_name, int64_t requested_model_version,
      std::string* config_json);
  TRITONSERVER_Error* GetContentLength(
      evhtp_request_t* req, evbuffer* decompressed_buffer,
      int32_t* content_length);
  TRITONSERVER_Error* DecompressBuffer(
      evhtp_request_t* req, evbuffer** decompressed_buffer);
  TRITONSERVER_Error* CheckTransactionPolicy(
      evhtp_request_t* req, const std::string& model_name,
      int64_t requested_model_version);
  std::shared_ptr<TraceManager::Trace> StartTrace(
      evhtp_request_t* req, const std::string& model_name,
      TRITONSERVER_InferenceTrace** triton_trace);
  TRITONSERVER_Error* ForwardHeaders(
      evhtp_request_t* req, TRITONSERVER_InferenceRequest* irequest);

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
  void HandleLogging(evhtp_request_t* req);

  // Text Generation / LLM format
  //'streaming' selects the schema pair to convert request / response.
  // 'streaming' also controls the response convention, if true,
  // Server-Sent Events format will be used to send responses.
  void HandleGenerate(
      evhtp_request_t* req, const std::string& model_name,
      const std::string& model_version_str, bool streaming);

  // 'meta_data_root' is the root JSON document for 'input_metadata'.
  // In TritonJson, the Value objects are references to the root document.
  // Therefore the document must stay valid.
  TRITONSERVER_Error* ModelInputMetadata(
      const std::string& model_name, const int64_t model_version,
      std::map<std::string, triton::common::TritonJson::Value>* input_metadata,
      triton::common::TritonJson::Value* meta_data_root);

  // Internal utility method for parsing evhtp request to JSON
  // Should not be called directly - use EVRequestToJson or
  // EVRequestToJsonAllowsEmpty instead
  TRITONSERVER_Error* EVRequestToJsonImpl(
      evhtp_request_t* req, std::string_view request_kind,
      bool allows_empty_body, triton::common::TritonJson::Value* request_json,
      size_t* buffer_len);

  // Parses full evhtp request and its evbuffers into JSON.
  TRITONSERVER_Error* EVRequestToJsonAllowsEmpty(
      evhtp_request_t* req, std::string_view request_kind,
      triton::common::TritonJson::Value* request_json)
  {
    size_t buffer_len = 0;
    TRITONSERVER_Error* err =
        EVRequestToJsonImpl(req, request_kind, true, request_json, &buffer_len);
    return err;
  }

  TRITONSERVER_Error* EVRequestToJson(
      evhtp_request_t* req, std::string_view request_kind,
      triton::common::TritonJson::Value* request_json, size_t* buffer_len)
  {
    TRITONSERVER_Error* err =
        EVRequestToJsonImpl(req, request_kind, false, request_json, buffer_len);
    return err;
  }

  // Parses evhtp request buffers into Triton Inference Request.
  TRITONSERVER_Error* EVRequestToTritonRequest(
      evhtp_request_t* req, const std::string& model_name,
      TRITONSERVER_InferenceRequest* irequest, evbuffer* decompressed_buffer,
      InferRequestClass* infer_req, size_t header_length);
  TRITONSERVER_Error* EVBufferToInput(
      const std::string& model_name, TRITONSERVER_InferenceRequest* irequest,
      evbuffer* input_buffer, InferRequestClass* infer_req,
      size_t header_length);
  TRITONSERVER_Error* EVBufferToRawInput(
      const std::string& model_name, TRITONSERVER_InferenceRequest* irequest,
      evbuffer* input_buffer, InferRequestClass* infer_req);
  TRITONSERVER_Error* EVBufferToJson(
      triton::common::TritonJson::Value* document, evbuffer_iovec* v,
      int* v_idx, const size_t length, int n);


  // Helpers for parsing JSON requests for Triton-specific fields
  TRITONSERVER_Error* ParseJsonTritonIO(
      triton::common::TritonJson::Value& request_json,
      TRITONSERVER_InferenceRequest* irequest, InferRequestClass* infer_req,
      const std::string& model_name, evbuffer_iovec* v, int* v_idx_ptr,
      size_t header_length, int n);
  TRITONSERVER_Error* ParseJsonTritonParams(
      triton::common::TritonJson::Value& request_json,
      TRITONSERVER_InferenceRequest* irequest, InferRequestClass* infer_req);
  TRITONSERVER_Error* ParseJsonTritonRequestID(
      triton::common::TritonJson::Value& request_json,
      TRITONSERVER_InferenceRequest* irequest);

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

  // [DLIS-5551] currently always performs basic conversion, only maps schema
  // of EXACT_MAPPING kind. MAPPING_SCHEMA and upcoming kinds are for
  // customized conversion where a detailed schema will be provided.
  std::unique_ptr<MappingSchema> generate_request_schema_{new MappingSchema()};
  std::unique_ptr<MappingSchema> generate_response_schema_{new MappingSchema()};
  std::unique_ptr<MappingSchema> generate_stream_response_schema_{
      new MappingSchema()};
  std::unique_ptr<MappingSchema> generate_stream_request_schema_{
      new MappingSchema()};

  // Provisional definition of generate mapping schema
  // to allow for parameters passing
  //
  // Note: subject to change
  void ConfigureGenerateMappingSchema()
  {
    // Reserved field parameters for generate
    // If present, parameters will be converted to tensors
    // or parameters based on model config

    const std::string parameters_field = "parameters";
    generate_stream_request_schema_->children_.emplace(
        parameters_field,
        new MappingSchema(MappingSchema::Kind::MAPPING_SCHEMA, true));
    generate_request_schema_->children_.emplace(
        parameters_field,
        new MappingSchema(MappingSchema::Kind::MAPPING_SCHEMA, true));
  }
  size_t max_input_size_;
  RestrictedFeatures restricted_apis_{};
  bool RespondIfRestricted(
      evhtp_request_t* req, const Restriction& restriction);
};

}}  // namespace triton::server
