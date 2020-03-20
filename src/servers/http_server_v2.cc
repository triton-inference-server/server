// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "src/servers/http_server_v2.h"

#include <event2/buffer.h>
#include <evhtp/evhtp.h>
#include <google/protobuf/text_format.h>
#include <re2/re2.h>
#include <algorithm>
#include <thread>
#include "src/core/api.pb.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/server_status.pb.h"
#include "src/core/trtserver.h"
#include "src/core/trtserver2.h"
#include "src/servers/common.h"

#ifdef TRTIS_ENABLE_TRACING
#include "src/servers/tracer.h"
#endif  // TRTIS_ENABLE_TRACING

namespace nvidia { namespace inferenceserver {

// Generic HTTP server using evhtp
class HTTPServerV2Impl : public HTTPServerV2 {
 public:
  explicit HTTPServerV2Impl(const int32_t port, const int thread_cnt)
      : port_(port), thread_cnt_(thread_cnt)
  {
  }

  virtual ~HTTPServerV2Impl() { Stop(); }

  static void Dispatch(evhtp_request_t* req, void* arg);

  TRTSERVER_Error* Start() override;
  TRTSERVER_Error* Stop() override;

 protected:
  virtual void Handle(evhtp_request_t* req) = 0;

  static void StopCallback(int sock, short events, void* arg);

  int32_t port_;
  int thread_cnt_;

  evhtp_t* htp_;
  struct event_base* evbase_;
  std::thread worker_;
  int fds_[2];
  event* break_ev_;
};

TRTSERVER_Error*
HTTPServerV2Impl::Start()
{
  if (!worker_.joinable()) {
    evbase_ = event_base_new();
    htp_ = evhtp_new(evbase_, NULL);
    evhtp_enable_flag(htp_, EVHTP_FLAG_ENABLE_NODELAY);
    evhtp_set_gencb(htp_, HTTPServerV2Impl::Dispatch, this);
    evhtp_use_threads_wexit(htp_, NULL, NULL, thread_cnt_, NULL);
    evhtp_bind_socket(htp_, "0.0.0.0", port_, 1024);
    // Set listening event for breaking event loop
    evutil_socketpair(AF_UNIX, SOCK_STREAM, 0, fds_);
    break_ev_ = event_new(evbase_, fds_[0], EV_READ, StopCallback, evbase_);
    event_add(break_ev_, NULL);
    worker_ = std::thread(event_base_loop, evbase_, 0);
    return nullptr;
  }

  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_ALREADY_EXISTS, "HTTP V2 server is already running.");
}

TRTSERVER_Error*
HTTPServerV2Impl::Stop()
{
  if (worker_.joinable()) {
    // Notify event loop to break via fd write
    send(fds_[1], &evbase_, sizeof(event_base*), 0);
    worker_.join();
    event_free(break_ev_);
    evutil_closesocket(fds_[0]);
    evutil_closesocket(fds_[1]);
    evhtp_unbind_socket(htp_);
    evhtp_free(htp_);
    event_base_free(evbase_);
    return nullptr;
  }

  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNAVAILABLE, "HTTP V2 server is not running.");
}

void
HTTPServerV2Impl::StopCallback(int sock, short events, void* arg)
{
  struct event_base* base = (struct event_base*)arg;
  event_base_loopbreak(base);
}

void
HTTPServerV2Impl::Dispatch(evhtp_request_t* req, void* arg)
{
  (static_cast<HTTPServerV2Impl*>(arg))->Handle(req);
}

#ifdef TRTIS_ENABLE_METRICS

// Handle HTTP requests to obtain prometheus metrics
class HTTPMetricsServerV2 : public HTTPServerV2Impl {
 public:
  explicit HTTPMetricsServerV2(
      const std::shared_ptr<TRTSERVER_Server>& server, const int32_t port,
      const int thread_cnt)
      : HTTPServerV2Impl(port, thread_cnt), server_(server),
        api_regex_(R"(/metrics/?)")
  {
  }

  ~HTTPMetricsServerV2() = default;

 private:
  void Handle(evhtp_request_t* req) override;

  std::shared_ptr<TRTSERVER_Server> server_;
  re2::RE2 api_regex_;
};

void
HTTPMetricsServerV2::Handle(evhtp_request_t* req)
{
  LOG_VERBOSE(1) << "HTTP request: " << req->method << " "
                 << req->uri->path->full;

  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  evhtp_res res = EVHTP_RES_BADREQ;

  // Call to metric endpoint should not have any trailing string
  if (RE2::FullMatch(std::string(req->uri->path->full), api_regex_)) {
    TRTSERVER_Metrics* metrics = nullptr;
    TRTSERVER_Error* err = TRTSERVER_ServerMetrics(server_.get(), &metrics);
    if (err == nullptr) {
      const char* base;
      size_t byte_size;
      err = TRTSERVER_MetricsFormatted(
          metrics, TRTSERVER_METRIC_PROMETHEUS, &base, &byte_size);
      if (err == nullptr) {
        res = EVHTP_RES_OK;
        evbuffer_add(req->buffer_out, base, byte_size);
      }
    }

    TRTSERVER_MetricsDelete(metrics);
    TRTSERVER_ErrorDelete(err);
  }

  evhtp_send_reply(req, res);
}

#endif  // TRTIS_ENABLE_METRICS

// Handle HTTP requests to inference server APIs
class HTTPAPIServerV2 : public HTTPServerV2Impl {
 public:
  explicit HTTPAPIServerV2(
      const std::shared_ptr<TRTSERVER_Server>& server,
      const std::shared_ptr<nvidia::inferenceserver::TraceManager>&
          trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      const std::vector<std::string>& endpoints, const int32_t port,
      const int thread_cnt)
      : HTTPServerV2Impl(port, thread_cnt), server_(server),
        trace_manager_(trace_manager), shm_manager_(shm_manager),
        allocator_(nullptr), server_regex_(R"(/v2(?:/health/(live|ready))?)"),
        model_regex_(
            R"(/v2/models/([^/]+)(?:/version/([0-9]+))?(?:/(infer|ready))?)"),
        modelcontrol_regex_(
            R"(/v2/repository(?:/([^/]+))?/(index|model/([^/]+)/(load|unload)))"),
        systemsharedmemory_regex_(
            R"(/v2/systemsharedmemory(?:/region/([^/]+))?/(status|register|unregister))"),
        cudasharedmemory_regex_(
            R"(/v2/cudasharedmemory(?:/region/([^/]+))?/(status|register|unregister))")
  {
    TRTSERVER_Error* err = TRTSERVER_ServerId(server_.get(), &server_id_);
    if (err != nullptr) {
      server_id_ = "unknown:0";
      TRTSERVER_ErrorDelete(err);
    }

    FAIL_IF_ERR(
        TRTSERVER_ResponseAllocatorNew(
            &allocator_, InferResponseAlloc, ResponseRelease),
        "creating response allocator");
  }

  ~HTTPAPIServerV2()
  {
    LOG_TRTSERVER_ERROR(
        TRTSERVER_ResponseAllocatorDelete(allocator_),
        "deleting response allocator");
  }

  //
  // AllocPayload
  //
  // Simple structure that carries the userp payload needed for
  // allocation.
  struct ShmInfo {
    void* base_;
    size_t byte_size_;
    TRTSERVER_Memory_Type memory_type_;
    int64_t device_id_;
  };

  using TensorShmMap = std::unordered_map<std::string, ShmInfo>;

  struct AllocPayload {
    explicit AllocPayload() : shm_map_(nullptr) {}
    ~AllocPayload()
    {
      // Don't delete 'response_buffer_' here. Destoryed as a part of the
      // InferRequestClass
      delete shm_map_;
    }

    std::vector<evbuffer*> response_buffer_;
    rapidjson::Document request_json_;
    rapidjson::Document response_json_;
    TensorShmMap* shm_map_;
  };

  // Class object associated to evhtp thread, requests received are bounded
  // with the thread that accepts it. Need to keep track of that and let the
  // corresponding thread send back the reply
  class InferRequestClass {
   public:
    InferRequestClass(
        evhtp_request_t* req, const char* server_id, uint64_t unique_id);

    ~InferRequestClass()
    {
      for (auto buffer : response_meta_data_.response_buffer_) {
        if (buffer != nullptr) {
          evbuffer_free(buffer);
        }
      }
    }

    evhtp_request_t* EvHtpRequest() const { return req_; }

    static void InferComplete(
        TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
        TRTSERVER2_InferenceRequest* request, void* userp);
    evhtp_res FinalizeResponse(TRTSERVER2_InferenceRequest* request);

#ifdef TRTIS_ENABLE_TRACING
    std::unique_ptr<TraceMetaData> trace_meta_data_;
#endif  // TRTIS_ENABLE_TRACING

    AllocPayload response_meta_data_;

   private:
    evhtp_request_t* req_;
    evthr_t* thread_;
    const char* const server_id_;
    const uint64_t unique_id_;
  };

 private:
  static TRTSERVER_Error* InferResponseAlloc(
      TRTSERVER_ResponseAllocator* allocator, const char* tensor_name,
      size_t byte_size, TRTSERVER_Memory_Type preferred_memory_type,
      int64_t preferred_memory_type_id, void* userp, void** buffer,
      void** buffer_userp, TRTSERVER_Memory_Type* actual_memory_type,
      int64_t* actual_memory_type_id);
  static TRTSERVER_Error* ResponseRelease(
      TRTSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
      size_t byte_size, TRTSERVER_Memory_Type memory_type,
      int64_t memory_type_id);

  void Handle(evhtp_request_t* req) override;
  void HandleServerHealth(evhtp_request_t* req, const std::string& kind);
  void HandleServerMetadata(evhtp_request_t* req);
  void HandleModelReady(
      evhtp_request_t* req, const std::string& model_name,
      const std::string& model_version_str);
  void HandleModelMetadata(
      evhtp_request_t* req, const std::string& model_name,
      const std::string& model_version_str);
  void HandleInfer(
      evhtp_request_t* req, const std::string& model_name,
      const std::string& model_version_str);
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

#ifdef TRTIS_ENABLE_GPU
  TRTSERVER_Error* EVBufferToCudaHandle(
      evbuffer* handle_buffer, cudaIpcMemHandle_t** cuda_shm_handle);
#endif  // TRTIS_ENABLE_GPU
  TRTSERVER_Error* EVBufferToInput(
      const std::string& model_name, TRTSERVER2_InferenceRequest* irequest,
      evbuffer* input_buffer, InferRequestClass* infer_req,
      size_t header_length);
  TRTSERVER_Error* EVBufferToJson(
      rapidjson::Document* document, evbuffer_iovec* v, int* v_idx,
      const size_t length, int n);

  static void OKReplyCallback(evthr_t* thr, void* arg, void* shared);
  static void BADReplyCallback(evthr_t* thr, void* arg, void* shared);

  std::shared_ptr<TRTSERVER_Server> server_;
  const char* server_id_;

  std::shared_ptr<TraceManager> trace_manager_;
  std::shared_ptr<SharedMemoryManager> shm_manager_;

  // The allocator that will be used to allocate buffers for the
  // inference result tensors.
  TRTSERVER_ResponseAllocator* allocator_;

  re2::RE2 server_regex_;
  re2::RE2 model_regex_;
  re2::RE2 modelcontrol_regex_;
  re2::RE2 systemsharedmemory_regex_;
  re2::RE2 cudasharedmemory_regex_;
};

TRTSERVER_Error*
HTTPAPIServerV2::InferResponseAlloc(
    TRTSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRTSERVER_Memory_Type preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRTSERVER_Memory_Type* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  AllocPayload* payload = reinterpret_cast<AllocPayload*>(userp);

  evbuffer* evhttp_buffer = evbuffer_new();
  if (evhttp_buffer == nullptr) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INTERNAL,
        std::string("failed to create evbuffer for output tensor").c_str());
  } else {
    payload->response_buffer_.push_back(evhttp_buffer);
  }

  const TensorShmMap* shm_map = payload->shm_map_;

  *buffer = nullptr;
  *buffer_userp = nullptr;
  *actual_memory_type = preferred_memory_type;
  *actual_memory_type_id = preferred_memory_type_id;

  // Don't need to do anything if no memory was requested.
  if (byte_size > 0) {
    bool use_shm = false;

    if (shm_map != nullptr) {
      const auto& pr = shm_map->find(tensor_name);
      if (pr != shm_map->end()) {
        // If the output is in shared memory then check whether the shared
        // memory size is at least the byte size of the output.
        if (byte_size > pr->second.byte_size_) {
          return TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_INTERNAL,
              std::string(
                  "shared memory size specified with the request for output '" +
                  std::string(tensor_name) + "' (" +
                  std::to_string(pr->second.byte_size_) +
                  " bytes) should be at least " + std::to_string(byte_size) +
                  " bytes to hold the results")
                  .c_str());
        }

        *buffer = const_cast<void*>(pr->second.base_);
        *actual_memory_type = pr->second.memory_type_;
        *actual_memory_type_id = pr->second.device_id_;
        use_shm = true;

        LOG_VERBOSE(1) << "HTTP: using shared-memory for '" << tensor_name
                       << "', size: " << byte_size << ", addr: " << *buffer;
      }
    }

    if (!use_shm) {
      // Can't allocate for any memory type other than CPU. If asked to
      // allocate on GPU memory then force allocation on CPU instead.
      if (*actual_memory_type != TRTSERVER_MEMORY_CPU) {
        LOG_VERBOSE(1) << "HTTP: unable to provide '" << tensor_name << "' in "
                       << MemoryTypeString(*actual_memory_type) << ", will use "
                       << MemoryTypeString(TRTSERVER_MEMORY_CPU);
        *actual_memory_type = TRTSERVER_MEMORY_CPU;
        *actual_memory_type_id = 0;
      }

      // Reserve requested space in evbuffer...
      struct evbuffer_iovec output_iovec;
      if (evbuffer_reserve_space(evhttp_buffer, byte_size, &output_iovec, 1) !=
          1) {
        return TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INTERNAL,
            std::string(
                "failed to reserve " + std::to_string(byte_size) +
                " bytes in output tensor buffer")
                .c_str());
      }

      if (output_iovec.iov_len < byte_size) {
        return TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INTERNAL,
            std::string(
                "reserved " + std::to_string(output_iovec.iov_len) +
                " bytes in output tensor buffer, need " +
                std::to_string(byte_size))
                .c_str());
      }

      output_iovec.iov_len = byte_size;
      *buffer = output_iovec.iov_base;

      // Immediately commit the buffer space. We are relying on evbuffer
      // not to relocate this space. Because we request a contiguous
      // chunk every time (above by allowing only a single entry in
      // output_iovec), this seems to be a valid assumption.
      if (evbuffer_commit_space(evhttp_buffer, &output_iovec, 1) != 0) {
        *buffer = nullptr;
        return TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INTERNAL,
            "failed to commit output tensors to output buffer");
      }

      LOG_VERBOSE(1) << "HTTP using buffer for: '" << tensor_name
                     << "', size: " << byte_size << ", addr: " << *buffer;
    }
  }

  return nullptr;  // Success
}

TRTSERVER_Error*
HTTPAPIServerV2::ResponseRelease(
    TRTSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRTSERVER_Memory_Type memory_type, int64_t memory_type_id)
{
  LOG_VERBOSE(1) << "HTTP release: "
                 << "size " << byte_size << ", addr " << buffer;

  // Don't do anything when releasing a buffer since ResponseAlloc
  // wrote directly into the response ebvuffer.
  return nullptr;  // Success
}

template <typename T>
void
ReadDataFromJsonHelper(
    std::vector<T>* data_vec, const DataType dtype,
    const rapidjson::Value& payload_data, int* counter)
{
  for (size_t i = 0; i < payload_data.Size(); i++) {
    // If last dimension
    if (!payload_data[i].IsArray()) {
      switch (dtype) {
        case TYPE_BOOL:
          data_vec->push_back((uint8_t)payload_data[i].GetBool());
          break;
        case TYPE_UINT8:
          data_vec->push_back((uint8_t)payload_data[i].GetInt());
          break;
        case TYPE_UINT16:
          data_vec->push_back((uint16_t)payload_data[i].GetInt());
          break;
        case TYPE_UINT32:
          data_vec->push_back((uint32_t)payload_data[i].GetInt());
          break;
        case TYPE_UINT64:
          data_vec->push_back((uint64_t)payload_data[i].GetInt());
          break;
        case TYPE_INT8:
          data_vec->push_back((int8_t)payload_data[i].GetInt());
          break;
        case TYPE_INT16:
          data_vec->push_back((int16_t)payload_data[i].GetInt());
          break;
        case TYPE_INT32:
          data_vec->push_back((int32_t)payload_data[i].GetInt());
          break;
        case TYPE_INT64:
          data_vec->push_back((int64_t)payload_data[i].GetInt());
          break;
        case TYPE_FP32:
          data_vec->push_back((float)payload_data[i].GetFloat());
          break;
        case TYPE_FP64:
          data_vec->push_back((double)payload_data[i].GetDouble());
          break;
        default:
          break;
      }
      *counter += 1;
    }
    // If not dimension
    else {
      ReadDataFromJsonHelper(data_vec, dtype, payload_data[i], counter);
    }
  }
}

TRTSERVER_Error*
ReadDataFromJson(
    const rapidjson::Value& request_input, char** base, size_t* byte_size)
{
  const rapidjson::Value& tensor_data = request_input["data"];
  const rapidjson::Value& shape = request_input["shape"];
  std::string dtype_str = std::string(request_input["datatype"].GetString());
  const DataType dtype =
      ProtocolStringToDataType(dtype_str.c_str(), dtype_str.size());
  int counter = 0;

  // Must be an array
  if (!tensor_data.IsArray()) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INVALID_ARG,
        "failed to parse request buffer, tensor data must be an array");
  }

  int element_cnt = 0;
  for (rapidjson::SizeType i = 0; i < shape.Size(); i++) {
    if (element_cnt == 0) {
      element_cnt = shape[i].GetInt();
    } else {
      element_cnt *= shape[i].GetInt();
    }
  }

  if (element_cnt == 0) {
    return nullptr;
  }

  switch (dtype) {
    case TYPE_BOOL: {
      std::vector<uint8_t> bool_tensor;
      ReadDataFromJsonHelper(&bool_tensor, dtype, tensor_data, &counter);
      *base = reinterpret_cast<char*>(&bool_tensor[0]);
      break;
    }
    case TYPE_UINT8: {
      std::vector<uint8_t> uint8_t_tensor;
      ReadDataFromJsonHelper(&uint8_t_tensor, dtype, tensor_data, &counter);
      *base = reinterpret_cast<char*>(&uint8_t_tensor[0]);
      break;
    }
    case TYPE_UINT16: {
      std::vector<uint16_t> uint16_t_tensor;
      ReadDataFromJsonHelper(&uint16_t_tensor, dtype, tensor_data, &counter);
      *base = reinterpret_cast<char*>(&uint16_t_tensor[0]);
      break;
    }
    case TYPE_UINT32: {
      std::vector<uint32_t> uint32_t_tensor;
      ReadDataFromJsonHelper(&uint32_t_tensor, dtype, tensor_data, &counter);
      *base = reinterpret_cast<char*>(&uint32_t_tensor[0]);
      break;
    }
    case TYPE_UINT64: {
      std::vector<uint64_t> uint64_t_tensor;
      ReadDataFromJsonHelper(&uint64_t_tensor, dtype, tensor_data, &counter);
      *base = reinterpret_cast<char*>(&uint64_t_tensor[0]);
      break;
    }
    case TYPE_INT8: {
      std::vector<int8_t> int8_t_tensor;
      ReadDataFromJsonHelper(&int8_t_tensor, dtype, tensor_data, &counter);
      *base = reinterpret_cast<char*>(&int8_t_tensor[0]);
    } break;
    case TYPE_INT16: {
      std::vector<int8_t> int16_t_tensor;
      ReadDataFromJsonHelper(&int16_t_tensor, dtype, tensor_data, &counter);
      *base = reinterpret_cast<char*>(&int16_t_tensor[0]);
    } break;
    case TYPE_INT32: {
      std::vector<int32_t> int32_t_tensor;
      ReadDataFromJsonHelper(&int32_t_tensor, dtype, tensor_data, &counter);
      *base = reinterpret_cast<char*>(&int32_t_tensor[0]);
      break;
    }
    case TYPE_INT64: {
      std::vector<int64_t> int64_t_tensor;
      ReadDataFromJsonHelper(&int64_t_tensor, dtype, tensor_data, &counter);
      *base = reinterpret_cast<char*>(&int64_t_tensor[0]);
      break;
    }
    // FP16 needs a work around
    case TYPE_FP16: {  // std::vector<float> float16_tensor;
      // ReadDataFromJsonHelper(&float16_t_tensor, dtype, tensor_data,
      // &counter); *base = reinterpret_cast<char*>(&float16_tensor[0]);
      break;
    }
    case TYPE_FP32: {
      std::vector<float> float_tensor;
      ReadDataFromJsonHelper(&float_tensor, dtype, tensor_data, &counter);
      *base = reinterpret_cast<char*>(&float_tensor[0]);
      break;
    }
    case TYPE_FP64: {
      std::vector<double> double_tensor;
      ReadDataFromJsonHelper(&double_tensor, dtype, tensor_data, &counter);
      *base = reinterpret_cast<char*>(&double_tensor[0]);
      break;
    }
    // BYTES (String) needs a work around
    case TYPE_STRING:
      break;
    case TYPE_INVALID: {
      return TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_INVALID_ARG,
          std::string(
              "invalid datatype " +
              std::string(DataTypeToProtocolString(dtype)) + " of input " +
              request_input["name"].GetString())
              .c_str());
    }
    default:
      break;
  }

  *byte_size = element_cnt * GetDataTypeByteSize(dtype);

  return nullptr;
}

template <typename T>
void
WriteDataToJsonHelper(
    rapidjson::Value* response_output_val,
    rapidjson::Document::AllocatorType& allocator,
    const rapidjson::Value& shape, int shape_index, T* base, int* counter)
{
  for (int i = 0; i < shape[shape_index].GetInt(); i++) {
    if ((shape_index + 1) != (int)shape.Size()) {
      rapidjson::Value response_output_array(rapidjson::kArrayType);
      WriteDataToJsonHelper(
          &response_output_array, allocator, shape, shape_index + 1, base,
          counter);
      response_output_val->PushBack(response_output_array, allocator);
    } else {
      rapidjson::Value data_val((T)(base[*counter]));
      response_output_val->PushBack(data_val, allocator);
      *counter += 1;
    }
  }
}

void
WriteDataToJson(
    rapidjson::Value& response_output,
    rapidjson::Document::AllocatorType& allocator, void* base)
{
  const rapidjson::Value& shape = response_output["shape"];
  std::string dtype_str = std::string(response_output["datatype"].GetString());
  const DataType dtype =
      ProtocolStringToDataType(dtype_str.c_str(), dtype_str.size());

  rapidjson::Value data_array(rapidjson::kArrayType);
  int counter = 0;

  for (int i = 0; i < shape[0].GetInt(); i++) {
    switch (dtype) {
      case TYPE_BOOL: {
        uint8_t* bool_base = reinterpret_cast<uint8_t*>(base);
        WriteDataToJsonHelper(
            &data_array, allocator, shape, 1, bool_base, &counter);
        break;
      }
      case TYPE_UINT8: {
        uint8_t* uint8_t_base = reinterpret_cast<uint8_t*>(base);
        WriteDataToJsonHelper(
            &data_array, allocator, shape, 1, uint8_t_base, &counter);
        break;
      }
      case TYPE_UINT16: {
        uint16_t* uint16_t_base = reinterpret_cast<uint16_t*>(base);
        WriteDataToJsonHelper(
            &data_array, allocator, shape, 1, uint16_t_base, &counter);
        break;
      }
      case TYPE_UINT32: {
        uint32_t* uint32_t_base = reinterpret_cast<uint32_t*>(base);
        WriteDataToJsonHelper(
            &data_array, allocator, shape, 1, uint32_t_base, &counter);
        break;
      }
      case TYPE_UINT64: {
        uint64_t* uint64_t_base = reinterpret_cast<uint64_t*>(base);
        WriteDataToJsonHelper(
            &data_array, allocator, shape, 1, uint64_t_base, &counter);
        break;
      }
      case TYPE_INT8: {
        int8_t* int8_t_base = reinterpret_cast<int8_t*>(base);
        WriteDataToJsonHelper(
            &data_array, allocator, shape, 1, int8_t_base, &counter);
      } break;
      case TYPE_INT16: {
        int16_t* int16_t_base = reinterpret_cast<int16_t*>(base);
        WriteDataToJsonHelper(
            &data_array, allocator, shape, 1, int16_t_base, &counter);
      } break;
      case TYPE_INT32: {
        int32_t* int32_t_base = reinterpret_cast<int32_t*>(base);
        WriteDataToJsonHelper(
            &data_array, allocator, shape, 1, int32_t_base, &counter);
        break;
      }
      case TYPE_INT64: {
        int64_t* int64_t_base = reinterpret_cast<int64_t*>(base);
        WriteDataToJsonHelper(
            &data_array, allocator, shape, 1, int64_t_base, &counter);
        break;
      }
      // FP16 needs a work around
      case TYPE_FP16: {
        // float16* float16_base = reinterpret_cast<float16*>(base);
        // WriteDataToJsonHelper(
        //     &data_array, allocator, shape, 1, float16_base, &counter);
        break;
      }
      case TYPE_FP32: {
        float* float_base = reinterpret_cast<float*>(base);
        WriteDataToJsonHelper(
            &data_array, allocator, shape, 1, float_base, &counter);
        break;
      }
      case TYPE_FP64: {
        double* double_base = reinterpret_cast<double*>(base);
        WriteDataToJsonHelper(
            &data_array, allocator, shape, 1, double_base, &counter);
        break;
      }
      // BYTES (String) needs a work around
      case TYPE_STRING:
        break;
      case TYPE_INVALID: {
        break;
      }
      default:
        break;
    }
  }

  response_output.AddMember("data", data_array, allocator);
}

void
EVBufferAddErrorJson(evbuffer* buffer, TRTSERVER_Error* err)
{
  std::string message = std::string(TRTSERVER_ErrorMessage(err));
  std::string message_json = "{ \"error\" : \"" + message + "\" }";
  evbuffer_add(buffer, message_json.c_str(), message_json.size());
}

void
HTTPAPIServerV2::Handle(evhtp_request_t* req)
{
  LOG_VERBOSE(1) << "HTTP V2 request: " << req->method << " "
                 << req->uri->path->full;

  std::string model_name, version, kind;
  if (RE2::FullMatch(
          std::string(req->uri->path->full), model_regex_, &model_name,
          &version, &kind)) {
    if (kind == "ready") {
      // model ready
      HandleModelReady(req, model_name, version);
      return;
    } else if (kind == "infer") {
      // model infer
      HandleInfer(req, model_name, version);
      return;
    } else if (kind == "") {
      // model metadata
      HandleModelMetadata(req, model_name, version);
      return;
    }
  }

  std::string region, action, rest, repo_name;
  if (std::string(req->uri->path->full) == "/v2") {
    // server metadata
    HandleServerMetadata(req);
    return;
  } else if (RE2::FullMatch(
                 std::string(req->uri->path->full), server_regex_, &rest)) {
    // server health
    HandleServerHealth(req, rest);
    return;
  } else if (RE2::FullMatch(
                 std::string(req->uri->path->full), systemsharedmemory_regex_,
                 &region, &action)) {
    // system shared memory
    HandleSystemSharedMemory(req, region, action);
    return;
  } else if (RE2::FullMatch(
                 std::string(req->uri->path->full), cudasharedmemory_regex_,
                 &region, &action)) {
    // cuda shared memory
    HandleCudaSharedMemory(req, region, action);
    return;
  } else if (RE2::FullMatch(
                 std::string(req->uri->path->full), modelcontrol_regex_,
                 &repo_name, &kind, &model_name, &action)) {
    // model repository
    if (kind == "index") {
      HandleRepositoryIndex(req, repo_name);
      return;
    } else if (kind.find("model", 0) == 0) {
      HandleRepositoryControl(req, repo_name, model_name, action);
      return;
    }
  }

  LOG_VERBOSE(1) << "HTTP V2 error: " << req->method << " "
                 << req->uri->path->full << " - "
                 << static_cast<int>(EVHTP_RES_BADREQ);
}

void
HTTPAPIServerV2::HandleServerHealth(
    evhtp_request_t* req, const std::string& kind)
{
  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  TRTSERVER_Error* err = nullptr;
  bool ready = false;

  if (kind == "live") {
    err = TRTSERVER_ServerIsLive(server_.get(), &ready);
  } else {
    err = TRTSERVER_ServerIsReady(server_.get(), &ready);
  }

  evhtp_send_reply(
      req, (ready && (err == nullptr)) ? EVHTP_RES_OK : EVHTP_RES_BADREQ);

  TRTSERVER_ErrorDelete(err);
}

void
HTTPAPIServerV2::HandleRepositoryIndex(
    evhtp_request_t* req, const std::string& repository_name)
{
  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  TRTSERVER_Error* err = nullptr;
  const char* const* models;
  uint64_t models_count = 0;
  TRTSERVER2_ModelIndex* model_index = nullptr;
  if (repository_name.empty()) {
    err = TRTSERVER2_ServerModelIndex(server_.get(), &model_index);
    if (err == nullptr) {
      err = TRTSERVER2_ModelIndexNames(model_index, &models, &models_count);
    }
  } else {
    err = TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_UNSUPPORTED,
        "'repository_name' specification is not yet supported");
  }

  rapidjson::Document document;
  document.SetObject();
  rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
  if (err == nullptr) {
    rapidjson::Value models_array(rapidjson::kArrayType);
    for (uint64_t i = 0; i < models_count; i++) {
      rapidjson::Value model_index;
      model_index.SetObject();
      const char* model_name = models[i];
      rapidjson::Value name_val(model_name, strlen(model_name), allocator);
      model_index.AddMember("name", name_val, allocator);
      models_array.PushBack(model_index, allocator);
    }
    document.AddMember("index", models_array, allocator);

    rapidjson::StringBuffer buffer;
    buffer.Clear();
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    document.Accept(writer);
    const char* model_metadata = buffer.GetString();
    evbuffer_add(req->buffer_out, model_metadata, strlen(model_metadata));
    err = TRTSERVER2_ModelIndexDelete(model_index);
    evhtp_send_reply(req, EVHTP_RES_OK);
  } else {
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
  }

  TRTSERVER_ErrorDelete(err);
}

void
HTTPAPIServerV2::HandleRepositoryControl(
    evhtp_request_t* req, const std::string& repository_name,
    const std::string& model_name, const std::string& action)
{
  if (req->method != htp_method_POST) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  TRTSERVER_Error* err = nullptr;
  if (!repository_name.empty()) {
    err = TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_UNSUPPORTED,
        "'repository_name' specification is not supported");
  }

  if (action == "load") {
    err = TRTSERVER_ServerLoadModel(server_.get(), model_name.c_str());
  } else if (action == "unload") {
    err = TRTSERVER_ServerUnloadModel(server_.get(), model_name.c_str());
  }

  if (err == nullptr) {
    evhtp_send_reply(req, EVHTP_RES_OK);
  } else {
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
  }

  TRTSERVER_ErrorDelete(err);
}

void
HTTPAPIServerV2::HandleModelReady(
    evhtp_request_t* req, const std::string& model_name,
    const std::string& model_version_str)
{
  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  if (model_name.empty()) {
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    return;
  }

  bool ready = false;
  TRTSERVER_Protobuf* model_status_protobuf = nullptr;
  TRTSERVER_Error* err = TRTSERVER_ServerModelStatus(
      server_.get(), model_name.c_str(), &model_status_protobuf);
  if (err == nullptr) {
    const char* status_buffer;
    size_t status_byte_size;
    err = TRTSERVER_ProtobufSerialize(
        model_status_protobuf, &status_buffer, &status_byte_size);
    if (err == nullptr) {
      ServerStatus server_status;
      if (!server_status.ParseFromArray(status_buffer, status_byte_size)) {
        err = TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_UNKNOWN, "failed to parse server status");
      } else {
        const auto& itr = server_status.model_status().find(model_name);
        if (itr == server_status.model_status().end()) {
          err = TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_INTERNAL,
              std::string(
                  "no status available for unknown model '" + model_name + "'")
                  .c_str());
        } else {
          const ModelStatus& model_status = itr->second;
          int64_t requested_version = -1;
          err =
              GetModelVersionFromString(model_version_str, &requested_version);
          if (err == nullptr) {
            // If requested_version is -1 then find the highest valued
            // version.
            if (requested_version == -1) {
              for (const auto& pr : model_status.version_status()) {
                requested_version = std::max(requested_version, pr.first);
              }
            }

            const auto& vitr =
                model_status.version_status().find(requested_version);
            if (vitr == model_status.version_status().end()) {
              err = TRTSERVER_ErrorNew(
                  TRTSERVER_ERROR_INVALID_ARG,
                  std::string(
                      "no status available for model '" + model_name +
                      "', version " + model_version_str)
                      .c_str());
            } else {
              const ModelVersionStatus& version_status = vitr->second;
              ready =
                  version_status.ready_state() == ModelReadyState::MODEL_READY;
            }
          }
        }
      }
    }
  }

  TRTSERVER_ProtobufDelete(model_status_protobuf);

  evhtp_send_reply(
      req, (ready && (err == nullptr)) ? EVHTP_RES_OK : EVHTP_RES_BADREQ);

  TRTSERVER_ErrorDelete(err);
}

void
HTTPAPIServerV2::HandleModelMetadata(
    evhtp_request_t* req, const std::string& model_name,
    const std::string& model_version_str)
{
  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  if (model_name.empty()) {
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    return;
  }

  ServerStatus server_status;
  TRTSERVER_Protobuf* model_status_protobuf = nullptr;
  TRTSERVER_Error* err = TRTSERVER_ServerModelStatus(
      server_.get(), model_name.c_str(), &model_status_protobuf);
  if (err == nullptr) {
    const char* status_buffer;
    size_t status_byte_size;
    err = TRTSERVER_ProtobufSerialize(
        model_status_protobuf, &status_buffer, &status_byte_size);
    if (err == nullptr) {
      if (!server_status.ParseFromArray(status_buffer, status_byte_size)) {
        err = TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_UNKNOWN, "failed to parse server status");
      }
    }
  }

  TRTSERVER_ProtobufDelete(model_status_protobuf);
  rapidjson::Document document;
  document.SetObject();
  rapidjson::Document::AllocatorType& allocator = document.GetAllocator();

  if (err == nullptr) {
    const auto& nitr = server_status.model_status().find(model_name);
    if (nitr == server_status.model_status().end()) {
      err = TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_INVALID_ARG,
          std::string(
              "no metadata available for unknown model '" + model_name + "'")
              .c_str());
    } else {
      // All models share the same metadata across versions so we ignore
      // model_version.
      const ModelStatus& model_status = nitr->second;
      const ModelConfig& model_config = model_status.config();
      // std::string name_str(name);
      rapidjson::Value name_val(
          model_config.name().c_str(), model_config.name().size());
      document.AddMember("name", name_val, allocator);

      rapidjson::Value versions_array(rapidjson::kArrayType);
      for (const auto& pr : model_status.version_status()) {
        std::string version_str = std::to_string(pr.first);
        rapidjson::Value version_val(version_str.c_str(), version_str.size());
        versions_array.PushBack(version_val, allocator);
      }
      document.AddMember("versions", versions_array, allocator);

      rapidjson::Value platform_val(
          model_config.platform().c_str(), model_config.platform().size());
      document.AddMember("platform", platform_val, allocator);

      rapidjson::Value inputs_array(rapidjson::kArrayType);
      rapidjson::Value input_metadata[model_config.input().size()];
      int i = 0;
      for (const auto& io : model_config.input()) {
        input_metadata[i].SetObject();
        rapidjson::Value name_val(io.name().c_str(), io.name().size());
        input_metadata[i].AddMember("name", name_val, allocator);

        std::string datatype_str = DataTypeToProtocolString(io.data_type());
        rapidjson::Value datatype_val(
            datatype_str.c_str(), datatype_str.size());
        input_metadata[i].AddMember("datatype", datatype_val, allocator);

        rapidjson::Value shape_array(rapidjson::kArrayType);
        for (const auto d : io.dims()) {
          shape_array.PushBack(d, allocator);
        }
        input_metadata[i].AddMember("shape", shape_array, allocator);

        inputs_array.PushBack(input_metadata[i], allocator);
        i++;
      }
      document.AddMember("inputs", inputs_array, allocator);

      rapidjson::Value outputs_array(rapidjson::kArrayType);
      rapidjson::Value output_metadata[model_config.output().size()];
      i = 0;
      for (const auto& io : model_config.output()) {
        output_metadata[i].SetObject();
        rapidjson::Value name_val(io.name().c_str(), io.name().size());
        output_metadata[i].AddMember("name", name_val, allocator);

        std::string datatype_str = DataTypeToProtocolString(io.data_type());
        rapidjson::Value datatype_val(
            datatype_str.c_str(), datatype_str.size());
        output_metadata[i].AddMember("datatype", datatype_val, allocator);

        rapidjson::Value shape_array(rapidjson::kArrayType);
        for (const auto d : io.dims()) {
          shape_array.PushBack(d, allocator);
        }
        output_metadata[i].AddMember("shape", shape_array, allocator);

        outputs_array.PushBack(output_metadata[i], allocator);
        i++;
      }
      document.AddMember("outputs", outputs_array, allocator);
    }
  }

  evhtp_headers_add_header(
      req->headers_out,
      evhtp_header_new("Content-Type", "application/json", 1, 1));

  if (err == nullptr) {
    rapidjson::StringBuffer buffer;
    buffer.Clear();
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    document.Accept(writer);
    std::string model_metadata(buffer.GetString());
    evbuffer_add(
        req->buffer_out, model_metadata.c_str(), model_metadata.size());
    evhtp_send_reply(req, EVHTP_RES_OK);
  } else {
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
  }

  TRTSERVER_ErrorDelete(err);
}

void
HTTPAPIServerV2::HandleServerMetadata(evhtp_request_t* req)
{
  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  rapidjson::Document document;
  document.SetObject();
  rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
  const char* name = nullptr;
  TRTSERVER_Error* err = TRTSERVER_ServerId(server_.get(), &name);
  if (err == nullptr) {
    std::string name_str(name);
    rapidjson::Value name_val(name_str.c_str(), name_str.size());
    document.AddMember("name", name_val, allocator);

    const char* version = nullptr;
    err = TRTSERVER_ServerVersion(server_.get(), &version);
    if (err == nullptr) {
      std::string version_str(version);
      rapidjson::Value version_val(version_str.c_str(), version_str.size());
      document.AddMember("version", version_val, allocator);

      uint64_t extensions_count;
      const char* const* extensions;
      err = TRTSERVER_ServerExtensions(
          server_.get(), &extensions, &extensions_count);
      rapidjson::Value extensions_array(rapidjson::kArrayType);
      if (err == nullptr) {
        for (uint64_t i = 0; i < extensions_count; ++i) {
          std::string extension_str(extensions[i]);
          rapidjson::Value extension_val(
              extension_str.c_str(), extension_str.size(), allocator);
          extensions_array.PushBack(extension_val, allocator);
        }
        document.AddMember("extensions", extensions_array, allocator);
      }
    }
  }

  evhtp_headers_add_header(
      req->headers_out,
      evhtp_header_new("Content-Type", "application/json", 1, 1));

  if (err == nullptr) {
    rapidjson::StringBuffer buffer;
    buffer.Clear();
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    document.Accept(writer);
    std::string status_buffer(buffer.GetString());
    evbuffer_add(req->buffer_out, status_buffer.c_str(), status_buffer.size());
    evhtp_send_reply(req, EVHTP_RES_OK);
  } else {
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
  }

  TRTSERVER_ErrorDelete(err);
}

void
HTTPAPIServerV2::HandleSystemSharedMemory(
    evhtp_request_t* req, const std::string& region_name,
    const std::string& action)
{
  if ((action == "status") && (req->method != htp_method_GET)) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  } else if ((action != "status") && (req->method != htp_method_POST)) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  TRTSERVER_Error* err = nullptr;
  rapidjson::Document document;
  rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
  if (action == "status") {
    document.SetObject();
    SharedMemoryStatus shm_status;
    err = shm_manager_->GetStatusV2(
        region_name, &shm_status, TRTSERVER_MEMORY_CPU);
    if (err == nullptr) {
      rapidjson::Value response_regions(rapidjson::kArrayType);
      for (int i = 0; i < shm_status.shared_memory_region_size(); i++) {
        const auto& rshm_region = shm_status.shared_memory_region(i);
        if (rshm_region.has_system_shared_memory()) {
          rapidjson::Value shm_region;
          shm_region.SetObject();
          std::string name = rshm_region.name();
          rapidjson::Value name_val(name.c_str(), name.size());
          shm_region.AddMember("name", name_val, allocator);
          std::string key =
              rshm_region.system_shared_memory().shared_memory_key();
          rapidjson::Value key_val(key.c_str(), key.size());
          shm_region.AddMember("key", key_val, allocator);
          uint64_t offset = rshm_region.system_shared_memory().offset();
          rapidjson::Value offset_val(offset);
          shm_region.AddMember("offset", offset_val, allocator);
          uint64_t byte_size = rshm_region.byte_size();
          rapidjson::Value byte_size_val(byte_size);
          shm_region.AddMember("byte_size", byte_size_val, allocator);
          response_regions.PushBack(shm_region, allocator);
        }
      }
      document.AddMember(
          "system shared memory status", response_regions, allocator);

      rapidjson::StringBuffer buffer;
      buffer.Clear();
      rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
      document.Accept(writer);
      std::string status_buffer(buffer.GetString());
      evbuffer_add(
          req->buffer_out, status_buffer.c_str(), status_buffer.size());
    }
  } else {
    if ((action == "register") && (region_name.empty())) {
      err = TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_INTERNAL,
          "'region name' is necessary to register system shared memory region");
    } else if (action == "register") {
      struct evbuffer_iovec* v = nullptr;
      int v_idx = 0;
      int n = evbuffer_peek(req->buffer_in, -1, NULL, NULL, 0);
      if (n > 0) {
        v = static_cast<struct evbuffer_iovec*>(
            alloca(sizeof(struct evbuffer_iovec) * n));
        if (evbuffer_peek(req->buffer_in, -1, NULL, v, n) != n) {
          err = TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_INTERNAL,
              "unexpected error getting register request buffers");
        }
      }
      if (err == nullptr) {
        size_t buffer_len = evbuffer_get_length(req->buffer_in);
        err = EVBufferToJson(&document, v, &v_idx, buffer_len, n);
        if (err == nullptr) {
          const char* shm_key = document["key"].GetString();
          uint64_t offset = document["offset"].GetInt();
          uint64_t byte_size = document["byte_size"].GetInt();
          err = shm_manager_->RegisterSystemSharedMemory(
              region_name.c_str(), shm_key, offset, byte_size);
        }
      }
    } else if ((action == "unregister") && (region_name.empty())) {
      err = shm_manager_->UnregisterAllV2(TRTSERVER_MEMORY_CPU);
    } else if (action == "unregister") {
      err = shm_manager_->UnregisterV2(region_name, TRTSERVER_MEMORY_CPU);
    }
  }

  if (err == nullptr) {
    evhtp_send_reply(req, EVHTP_RES_OK);
  } else {
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
  }
  TRTSERVER_ErrorDelete(err);
}

void
HTTPAPIServerV2::HandleCudaSharedMemory(
    evhtp_request_t* req, const std::string& region_name,
    const std::string& action)
{
  if ((action == "status") && (req->method != htp_method_GET)) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  } else if ((action != "status") && (req->method != htp_method_POST)) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  TRTSERVER_Error* err = nullptr;
  rapidjson::Document document;
  rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
  if (action == "status") {
    document.SetObject();
    SharedMemoryStatus shm_status;
    err = shm_manager_->GetStatusV2(
        region_name, &shm_status, TRTSERVER_MEMORY_GPU);
    if (err == nullptr) {
      rapidjson::Value response_regions(rapidjson::kArrayType);
      for (int i = 0; i < shm_status.shared_memory_region_size(); i++) {
        const auto& rshm_region = shm_status.shared_memory_region(i);
        if (rshm_region.has_cuda_shared_memory()) {
          rapidjson::Value shm_region;
          shm_region.SetObject();
          std::string name = rshm_region.name();
          rapidjson::Value name_val(name.c_str(), name.size());
          shm_region.AddMember("name", name_val, allocator);
          uint64_t device_id = rshm_region.cuda_shared_memory().device_id();
          rapidjson::Value device_id_val(device_id);
          shm_region.AddMember("device_id", device_id_val, allocator);
          uint64_t byte_size = rshm_region.byte_size();
          rapidjson::Value byte_size_val(byte_size);
          shm_region.AddMember("byte_size", byte_size_val, allocator);
          response_regions.PushBack(shm_region, allocator);
        }
      }
      document.AddMember(
          "cuda shared memory status", response_regions, allocator);

      rapidjson::StringBuffer buffer;
      buffer.Clear();
      rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
      document.Accept(writer);
      std::string status_buffer(buffer.GetString());
      evbuffer_add(
          req->buffer_out, status_buffer.c_str(), status_buffer.size());
    }
  } else {
    if ((action == "register") && (region_name.empty())) {
      err = TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_INTERNAL,
          "'region name' is necessary to register cuda shared memory region");
    } else if (action == "register") {
      struct evbuffer_iovec* v = nullptr;
      int v_idx = 0;
      int n = evbuffer_peek(req->buffer_in, -1, NULL, NULL, 0);
      if (n > 0) {
        v = static_cast<struct evbuffer_iovec*>(
            alloca(sizeof(struct evbuffer_iovec) * n));
        if (evbuffer_peek(req->buffer_in, -1, NULL, v, n) != n) {
          err = TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_INTERNAL,
              "unexpected error getting register request buffers");
        }
      }
      if (err == nullptr) {
        size_t buffer_len = evbuffer_get_length(req->buffer_in);
        err = EVBufferToJson(&document, v, &v_idx, buffer_len, n);
        if (err == nullptr) {
          const char* raw_handle = document["raw_handle"].GetString();
          uint64_t byte_size = document["byte_size"].GetInt();
          uint64_t device_id = document["device_id"].GetInt();
          err = shm_manager_->RegisterCUDASharedMemory(
              region_name.c_str(),
              reinterpret_cast<const cudaIpcMemHandle_t*>(raw_handle),
              byte_size, device_id);
        }
      }
    } else if ((action == "unregister") && (region_name.empty())) {
      err = shm_manager_->UnregisterAllV2(TRTSERVER_MEMORY_GPU);
    } else if (action == "unregister") {
      err = shm_manager_->UnregisterV2(region_name, TRTSERVER_MEMORY_GPU);
    }
  }

  if (err == nullptr) {
    evhtp_send_reply(req, EVHTP_RES_OK);
  } else {
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
  }
  TRTSERVER_ErrorDelete(err);
}

bool
CheckBinaryInputData(const rapidjson::Value& request_input, size_t* byte_size)
{
  bool binary_input = false;
  rapidjson::Value::ConstMemberIterator itr =
      request_input.FindMember("parameters");
  if (itr != request_input.MemberEnd()) {
    const rapidjson::Value& params = itr->value;
    rapidjson::Value::ConstMemberIterator iter =
        params.FindMember("binary_data_size");
    if (iter != params.MemberEnd()) {
      *byte_size = iter->value.GetInt();
      binary_input = true;
    }
  }

  return binary_input;
}

bool
CheckBinaryOutputData(const rapidjson::Value& request_output)
{
  rapidjson::Value::ConstMemberIterator itr =
      request_output.FindMember("parameters");
  if (itr != request_output.MemberEnd()) {
    const rapidjson::Value& params = itr->value;
    rapidjson::Value::ConstMemberIterator iter =
        params.FindMember("binary_data");
    if (iter != params.MemberEnd()) {
      return iter->value.GetBool();
    }
  }

  return false;
}

TRTSERVER_Error*
HTTPAPIServerV2::EVBufferToJson(
    rapidjson::Document* document, evbuffer_iovec* v, int* v_idx,
    const size_t length, int n)
{
  size_t offset = 0, remaining_length = length;
  char* json_base;
  std::vector<char> json_buffer;

  // No need to memcpy when number of iovecs is 1
  if ((n > 0) and (v[0].iov_len >= remaining_length)) {
    json_base = static_cast<char*>(v[0].iov_base);
    if (v[0].iov_len > remaining_length) {
      v[0].iov_base = static_cast<void*>(json_base + remaining_length);
      v[0].iov_len -= remaining_length;
      remaining_length = 0;
    } else if (v[0].iov_len == remaining_length) {
      remaining_length = 0;
      *v_idx += 1;
    }
  } else {
    json_buffer.resize(length);
    json_base = json_buffer.data();
    while ((remaining_length > 0) && (*v_idx < n)) {
      char* base = static_cast<char*>(v[*v_idx].iov_base);
      size_t base_size;
      if (v[*v_idx].iov_len > remaining_length) {
        base_size = remaining_length;
        v[*v_idx].iov_base = static_cast<void*>(base + remaining_length);
        v[*v_idx].iov_len -= remaining_length;
        remaining_length = 0;
      } else {
        base_size = v[*v_idx].iov_len;
        remaining_length -= v[*v_idx].iov_len;
        *v_idx += 1;
      }

      memcpy(json_base + offset, base, base_size);
      offset += base_size;
    }
  }

  if (remaining_length != 0) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected size for request JSON, expecting " +
            std::to_string(remaining_length) + " more bytes")
            .c_str());
  }

  document->Parse(json_base, length);
  if (document->HasParseError()) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INVALID_ARG,
        std::string(
            "failed to parse the request JSON buffer: " +
            std::string(GetParseError_En(document->GetParseError())) + " at " +
            std::to_string(document->GetErrorOffset()))
            .c_str());
  }

  return nullptr;
}

TRTSERVER_Error*
HTTPAPIServerV2::EVBufferToInput(
    const std::string& model_name, TRTSERVER2_InferenceRequest* irequest,
    evbuffer* input_buffer, InferRequestClass* infer_req, size_t header_length)
{
  // Extract individual input data from HTTP body and register in
  // 'request_provider'. The input data from HTTP body is not
  // necessarily contiguous so may need to register multiple input
  // "blocks" for a given input.
  //
  // Get the addr and size of each chunk of input data from the
  // evbuffer.
  struct evbuffer_iovec* v = nullptr;
  int v_idx = 0;

  int n = evbuffer_peek(input_buffer, -1, NULL, NULL, 0);
  if (n > 0) {
    v = static_cast<struct evbuffer_iovec*>(
        alloca(sizeof(struct evbuffer_iovec) * n));
    if (evbuffer_peek(input_buffer, -1, NULL, v, n) != n) {
      return TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_INTERNAL, "unexpected error getting input buffers");
    }
  }

  // Extract just the json header from the complete buffer
  rapidjson::Document& request_json =
      infer_req->response_meta_data_.request_json_;
  int buffer_len = 0;
  if (header_length == 0) {
    buffer_len = evbuffer_get_length(input_buffer);
  } else {
    buffer_len = header_length;
  }
  RETURN_IF_ERR(EVBufferToJson(&request_json, v, &v_idx, buffer_len, n));

  // Set InferenceRequest request_id
  auto itr = request_json.FindMember("id");
  if (itr != request_json.MemberEnd()) {
    const char* id = itr->value.GetString();
    RETURN_IF_ERR(TRTSERVER2_InferenceRequestSetId(irequest, id));
  }

  // Get the byte-size for each input and from that get the blocks
  // holding the data for that input
  const rapidjson::Value& inputs = request_json["inputs"];
  for (size_t i = 0; i < inputs.Size(); i++) {
    const rapidjson::Value& request_input = inputs[i];
    const char* input_name = request_input["name"].GetString();
    const char* datatype = request_input["datatype"].GetString();
    const rapidjson::Value& shape = request_input["shape"];

    std::vector<int64_t> shape_vec;
    for (rapidjson::SizeType i = 0; i < shape.Size(); i++) {
      shape_vec.push_back(shape[i].GetInt());
    }

    size_t byte_size = 0;
    bool binary_input = CheckBinaryInputData(request_input, &byte_size);
    RETURN_IF_ERR(TRTSERVER2_InferenceRequestAddInput(
        irequest, input_name, datatype, &shape_vec[0], shape_vec.size()));

    if (byte_size == 0 && binary_input) {
      RETURN_IF_ERR(TRTSERVER2_InferenceRequestAppendInputData(
          irequest, input_name, nullptr, 0 /* byte_size */,
          TRTSERVER_MEMORY_CPU, 0 /* memory_type_id */));
    } else {
      // If input is in shared memory then verify that the size is
      // correct and set input from the shared memory.
#if 0
      // FIXMEV2 handle shared memory inputs
      if (io.has_shared_memory()) {
        if (byte_size != io.shared_memory().byte_size()) {
          return TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_INVALID_ARG,
              std::string(
                  "unexpected shared-memory size " +
                  std::to_string(io.shared_memory().byte_size()) +
                  " for input '" + io.name() + "', expecting " +
                  std::to_string(byte_size) + " for model '" + model_name + "'")
                  .c_str());
        }

        void* base;
        TRTSERVER_Memory_Type memory_type = TRTSERVER_MEMORY_CPU;
        int64_t memory_type_id;
        RETURN_IF_ERR(shm_manager_->GetMemoryInfo(
            io.shared_memory().name(), io.shared_memory().offset(), &base,
            &memory_type, &memory_type_id));
        RETURN_IF_ERR(TRTSERVER_InferenceRequestProviderSetInputData(
            request_provider, io.name().c_str(), base, byte_size, memory_type,
            memory_type_id));
      } else {
#endif
      if (binary_input) {
        if (header_length == 0) {
          return TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_INVALID_ARG,
              "must specify valid 'Infer-Header-Content-Length' in request "
              "header and 'binary_data_size' when passing inputs in binary "
              "data "
              "format");
        }

        // Process one block at a time
        while ((byte_size > 0) && (v_idx < n)) {
          char* base = static_cast<char*>(v[v_idx].iov_base);
          size_t base_size;
          if (v[v_idx].iov_len > byte_size) {
            base_size = byte_size;
            v[v_idx].iov_base = static_cast<void*>(base + byte_size);
            v[v_idx].iov_len -= byte_size;
            byte_size = 0;
          } else {
            base_size = v[v_idx].iov_len;
            byte_size -= v[v_idx].iov_len;
            v_idx++;
          }

          RETURN_IF_ERR(TRTSERVER2_InferenceRequestAppendInputData(
              irequest, input_name, base, base_size, TRTSERVER_MEMORY_CPU,
              0 /* memory_type_id */));
        }

        if (byte_size != 0) {
          return TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_INVALID_ARG,
              std::string(
                  "unexpected size for input '" +
                  std::string(request_input["name"].GetString()) +
                  "', expecting " + std::to_string(byte_size) +
                  " bytes for model '" + model_name + "'")
                  .c_str());
        }
      } else {
        char* base = nullptr;
        RETURN_IF_ERR(ReadDataFromJson(request_input, &base, &byte_size));
        RETURN_IF_ERR(TRTSERVER2_InferenceRequestAppendInputData(
            irequest, input_name, base, byte_size, TRTSERVER_MEMORY_CPU,
            0 /* memory_type_id */));
      }
#if 0
    }
#endif
    }
  }

  if (v_idx != n) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected additional input data for model '" + model_name + "'")
            .c_str());
  }

  rapidjson::Value& outputs_array = request_json["outputs"];
  for (size_t i = 0; i < outputs_array.Size(); i++) {
    rapidjson::Value& output = outputs_array[i];
    std::string output_name = std::string(output["name"].GetString());
    TRTSERVER2_InferenceRequestAddRequestedOutput(
        irequest, output_name.c_str());
  }

#if 0
  // FIXMEV2 handle shared memory outputs
  // Initialize System Memory for Output if it uses shared memory
  for (const auto& io : request_header.output()) {
    if (io.has_shared_memory()) {
      void* base;
      TRTSERVER_Memory_Type memory_type;
      int64_t memory_type_id;
      RETURN_IF_ERR(shm_manager_->GetMemoryInfo(
          io.shared_memory().name(), io.shared_memory().offset(), &base,
          &memory_type, &memory_type_id));

      output_shm_map.emplace(
          io.name(),
          std::make_tuple(
              static_cast<const void*>(base), io.shared_memory().byte_size(),
              memory_type, memory_type_id));
    }
  }
#endif

  return nullptr;  // success
}

void
HTTPAPIServerV2::HandleInfer(
    evhtp_request_t* req, const std::string& model_name,
    const std::string& model_version_str)
{
  if (req->method != htp_method_POST) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  int64_t requested_model_version;
  TRTSERVER_Error* err = GetModelVersionFromString(
      model_version_str.c_str(), &requested_model_version);

#ifdef TRTIS_ENABLE_TRACING
  // Timestamps from evhtp are capture in 'req'. We record here since
  // this is the first place where we have a tracer.
  std::unique_ptr<TraceMetaData> trace_meta_data;
  if (trace_manager_ != nullptr) {
    trace_meta_data.reset(trace_manager_->SampleTrace());
    if (trace_meta_data != nullptr) {
      if (err == nullptr) {
        trace_meta_data->tracer_->SetModel(model_name, requested_model_version);
      } else {
        // If failed to retrieve the requested_model_version
        // then use the default model version just to record
        // the timestamps in the tracer
        trace_meta_data->tracer_->SetModel(model_name, -1);
      }
      trace_meta_data->tracer_->CaptureTimestamp(
          TRTSERVER_TRACE_LEVEL_MIN, "http recv start",
          TIMESPEC_TO_NANOS(req->recv_start_ts));
      trace_meta_data->tracer_->CaptureTimestamp(
          TRTSERVER_TRACE_LEVEL_MIN, "http recv end",
          TIMESPEC_TO_NANOS(req->recv_end_ts));
    }
  }
#endif  // TRTIS_ENABLE_TRACING

  uint64_t unique_id = RequestStatusUtil::NextUniqueRequestId();

  // Create the inference request object which provides all information needed
  // for an inference.
  TRTSERVER2_InferenceRequest* irequest = nullptr;
  if (err == nullptr) {
    err = TRTSERVER2_InferenceRequestNew(
        &irequest, server_.get(), model_name.c_str(),
        model_version_str.c_str());
  }

  if (err == nullptr) {
    std::unique_ptr<InferRequestClass> infer_request(
        new InferRequestClass(req, server_id_, unique_id));

    // Find Inference-Header-Content-Length in header. If missing set to 0
    size_t header_length = 0;
    const char* header_length_c_str =
        evhtp_kv_find(req->headers_in, kInferHeaderContentLengthHTTPHeader);
    if (header_length_c_str != NULL) {
      header_length = std::atoi(header_length_c_str);
    }

    err = EVBufferToInput(
        model_name, irequest, req->buffer_in, infer_request.get(),
        header_length);
    if (err == nullptr) {
      // Provide the trace manager object to use for this request, if nullptr
      // then no tracing will be performed.
      TRTSERVER_TraceManager* trace_manager = nullptr;
#ifdef TRTIS_ENABLE_TRACING
      if (trace_meta_data != nullptr) {
        infer_request->trace_meta_data_ = std::move(trace_meta_data);
        TRTSERVER_TraceManagerNew(
            &trace_manager, TraceManager::CreateTrace,
            TraceManager::ReleaseTrace, infer_request->trace_meta_data_.get());
      }
#endif  // TRTIS_ENABLE_TRACING

      rapidjson::Document& response_json =
          infer_request->response_meta_data_.response_json_;
      rapidjson::Document::AllocatorType& allocator =
          response_json.GetAllocator();
      response_json.SetObject();
      rapidjson::Value model_name_val(model_name.c_str(), model_name.size());
      response_json.AddMember("model_name", model_name_val, allocator);
      rapidjson::Value model_version_val(
          model_version_str.c_str(), model_version_str.size());
      response_json.AddMember("model_version", model_version_val, allocator);

      err = TRTSERVER2_ServerInferAsync(
          server_.get(), trace_manager, irequest, allocator_,
          reinterpret_cast<void*>(&infer_request->response_meta_data_),
          InferRequestClass::InferComplete,
          reinterpret_cast<void*>(infer_request.get()));
      if (err == nullptr) {
        infer_request.release();
      }
    }
  }

  // The request provider can be deleted before ServerInferAsync
  // callback completes.

  if (err != nullptr) {
    LOG_VERBOSE(1) << "Infer failed: " << TRTSERVER_ErrorMessage(err);
    EVBufferAddErrorJson(req->buffer_out, err);

    evhtp_headers_add_header(
        req->headers_out,
        evhtp_header_new("Content-Type", "application/json", 1, 1));

    evhtp_send_reply(req, EVHTP_RES_BADREQ);
  }

  TRTSERVER_ErrorDelete(err);
}

void
HTTPAPIServerV2::OKReplyCallback(evthr_t* thr, void* arg, void* shared)
{
  HTTPAPIServerV2::InferRequestClass* infer_request =
      reinterpret_cast<HTTPAPIServerV2::InferRequestClass*>(arg);

  evhtp_request_t* request = infer_request->EvHtpRequest();
  evhtp_send_reply(request, EVHTP_RES_OK);
  evhtp_request_resume(request);

#ifdef TRTIS_ENABLE_TRACING
  if (infer_request->trace_meta_data_ != nullptr) {
    infer_request->trace_meta_data_->tracer_->CaptureTimestamp(
        TRTSERVER_TRACE_LEVEL_MIN, "http send start",
        TIMESPEC_TO_NANOS(request->send_start_ts));
    infer_request->trace_meta_data_->tracer_->CaptureTimestamp(
        TRTSERVER_TRACE_LEVEL_MIN, "http send end",
        TIMESPEC_TO_NANOS(request->send_end_ts));
  }
#endif  // TRTIS_ENABLE_TRACING

  delete infer_request;
}

void
HTTPAPIServerV2::BADReplyCallback(evthr_t* thr, void* arg, void* shared)
{
  HTTPAPIServerV2::InferRequestClass* infer_request =
      reinterpret_cast<HTTPAPIServerV2::InferRequestClass*>(arg);

  evhtp_request_t* request = infer_request->EvHtpRequest();
  evhtp_send_reply(request, EVHTP_RES_BADREQ);
  evhtp_request_resume(request);

#ifdef TRTIS_ENABLE_TRACING
  if (infer_request->trace_meta_data_ != nullptr) {
    infer_request->trace_meta_data_->tracer_->CaptureTimestamp(
        TRTSERVER_TRACE_LEVEL_MIN, "http send start",
        TIMESPEC_TO_NANOS(request->send_start_ts));
    infer_request->trace_meta_data_->tracer_->CaptureTimestamp(
        TRTSERVER_TRACE_LEVEL_MIN, "http send end",
        TIMESPEC_TO_NANOS(request->send_end_ts));
  }
#endif  // TRTIS_ENABLE_TRACING

  delete infer_request;
}

HTTPAPIServerV2::InferRequestClass::InferRequestClass(
    evhtp_request_t* req, const char* server_id, uint64_t unique_id)
    : req_(req), server_id_(server_id), unique_id_(unique_id)
{
  evhtp_connection_t* htpconn = evhtp_request_get_connection(req);
  thread_ = htpconn->thread;
  evhtp_request_pause(req);
}

void
HTTPAPIServerV2::InferRequestClass::InferComplete(
    TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
    TRTSERVER2_InferenceRequest* request, void* userp)
{
  HTTPAPIServerV2::InferRequestClass* infer_request =
      reinterpret_cast<HTTPAPIServerV2::InferRequestClass*>(userp);

  if (infer_request->FinalizeResponse(request) == EVHTP_RES_OK) {
    evthr_defer(infer_request->thread_, OKReplyCallback, infer_request);
  } else {
    evthr_defer(infer_request->thread_, BADReplyCallback, infer_request);
  }

  // Don't need to explicitly delete 'trace_manager'. It is owned by
  // 'infer_request' which will be deleted after the response is sent
  // in ReplayCallback.
  LOG_TRTSERVER_ERROR(
      TRTSERVER2_InferenceRequestDelete(request), "deleting inference request");
}

evhtp_res
HTTPAPIServerV2::InferRequestClass::FinalizeResponse(
    TRTSERVER2_InferenceRequest* request)
{
  rapidjson::Document& response_json = response_meta_data_.response_json_;
  rapidjson::Document::AllocatorType& allocator = response_json.GetAllocator();

  const char* request_id;
  TRTSERVER2_InferenceRequestId(request, &request_id);
  std::string request_id_str = std::string(request_id);
  if (!request_id_str.empty()) {
    rapidjson::Value id_val(request_id_str.c_str(), request_id_str.size());
    response_json.AddMember("id", id_val, allocator);
  }

  TRTSERVER_Error* err;
  rapidjson::Value& request_outputs =
      response_meta_data_.request_json_["outputs"];
  rapidjson::Value response_outputs(rapidjson::kArrayType);
  rapidjson::Value output_metadata[request_outputs.Size()];
  for (size_t i = 0; i < request_outputs.Size(); i++) {
    output_metadata[i].SetObject();
    rapidjson::Value& request_output = request_outputs[i];
    std::string output_name = std::string(request_output["name"].GetString());
    rapidjson::Value name_val(output_name.c_str(), output_name.size());
    output_metadata[i].AddMember("name", name_val, allocator);

    int class_size = 0;
    rapidjson::Value::ConstMemberIterator itr =
        request_output.FindMember("parameters");
    if (itr != request_output.MemberEnd()) {
      const rapidjson::Value& params = itr->value;
      auto iter = params.FindMember("classification");
      if (iter != params.MemberEnd()) {
        class_size = iter->value.GetInt();
      }
    }

    if (class_size == 0) {
      // Get shape of Output
      // Assume max dimension of 6
      uint64_t dim_count = 6;
      std::vector<int64_t> shape_vec(dim_count);
      err = TRTSERVER2_InferenceRequestOutputShape(
          request, output_name.c_str(), &shape_vec[0], &dim_count);
      if (err != nullptr) {
        return EVHTP_RES_BADREQ;
      }

      rapidjson::Value shape_array(rapidjson::kArrayType);
      for (size_t i = 0; i < dim_count; i++) {
        shape_array.PushBack(shape_vec[i], allocator);
      }
      output_metadata[i].AddMember("shape", shape_array, allocator);

      const char* datatype;
      err = TRTSERVER2_InferenceRequestOutputDataType(
          request, output_name.c_str(), &datatype);
      if (err != nullptr) {
        return EVHTP_RES_BADREQ;
      }

      std::string datatype_str = std::string(datatype);
      rapidjson::Value datatype_val(datatype_str.c_str(), datatype_str.size());
      output_metadata[i].AddMember("datatype", datatype_val, allocator);

      const void* base;
      size_t byte_size;
      TRTSERVER_Memory_Type memory_type;
      int64_t memory_type_id;
      err = TRTSERVER2_InferenceRequestOutputData(
          request, output_name.c_str(), &base, &byte_size, &memory_type,
          &memory_type_id);
      if (err != nullptr) {
        return EVHTP_RES_BADREQ;
      }

      if (CheckBinaryOutputData(request_output)) {
        // Write outputs into binary buffer
        evbuffer_add(req_->buffer_out, base, byte_size);
        rapidjson::Value binary_size_val(byte_size);
        auto itr = output_metadata[i].FindMember("parameters");
        if (itr != output_metadata[i].MemberEnd()) {
          itr->value.AddMember("binary_data_size", binary_size_val, allocator);
        } else {
          rapidjson::Value params;
          params.SetObject();
          params.AddMember("binary_data_size", binary_size_val, allocator);
          output_metadata[i].AddMember("parameters", params, allocator);
        }
      } else {
        // Write outputs into json array
        WriteDataToJson(output_metadata[i], allocator, const_cast<void*>(base));
      }
    }
    // TODO Add case for classification
    response_outputs.PushBack(output_metadata[i], allocator);
  }
  response_json.AddMember("outputs", response_outputs, allocator);

  // write json metadata into evbuffer
  rapidjson::StringBuffer buffer;
  buffer.Clear();
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  response_meta_data_.response_json_.Accept(writer);
  std::string response_metadata(buffer.GetString());
  evbuffer_add(
      req_->buffer_out, response_metadata.c_str(), response_metadata.size());

  evhtp_headers_add_header(
      req_->headers_out,
      evhtp_header_new("Content-Type", "application/json", 1, 1));

  return (err == nullptr) ? EVHTP_RES_OK : EVHTP_RES_BADREQ;
}

TRTSERVER_Error*
HTTPServerV2::CreateAPIServer(
    const std::shared_ptr<TRTSERVER_Server>& server,
    const std::shared_ptr<nvidia::inferenceserver::TraceManager>& trace_manager,
    const std::shared_ptr<SharedMemoryManager>& shm_manager,
    const std::map<int32_t, std::vector<std::string>>& port_map, int thread_cnt,
    std::vector<std::unique_ptr<HTTPServerV2>>* http_servers)
{
  if (port_map.empty()) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INVALID_ARG,
        "HTTP V2 is enabled but none of the service endpoints have a valid "
        "port assignment");
  }
  http_servers->clear();
  for (auto const& ep_map : port_map) {
    std::string addr = "0.0.0.0:" + std::to_string(ep_map.first);
    LOG_INFO << "Starting HTTPV2Service at " << addr;
    http_servers->emplace_back(new HTTPAPIServerV2(
        server, trace_manager, shm_manager, ep_map.second, ep_map.first,
        thread_cnt));
  }

  return nullptr;
}

TRTSERVER_Error*
HTTPServerV2::CreateMetricsServer(
    const std::shared_ptr<TRTSERVER_Server>& server, const int32_t port,
    const int thread_cnt, std::unique_ptr<HTTPServerV2>* metrics_server)
{
  std::string addr = "0.0.0.0:" + std::to_string(port);
  LOG_INFO << "Starting Metrics Service at " << addr;

#ifndef TRTIS_ENABLE_METRICS
  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNAVAILABLE, "Metrics support is disabled");
#endif  // !TRTIS_ENABLE_METRICS

#ifdef TRTIS_ENABLE_METRICS
  metrics_server->reset(new HTTPMetricsServerV2(server, port, thread_cnt));
  return nullptr;
#endif  // TRTIS_ENABLE_METRICS
}

}}  // namespace nvidia::inferenceserver