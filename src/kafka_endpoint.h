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

#include <librdkafka/rdkafka.h>
#include <map>
#include <queue>
#include <set>
#include <thread>
#include "shared_memory_manager.h"
#include "triton/common/logging.h"
#include "triton/core/tritonserver.h"
// modern-cpp-kafka libraries
#include "kafka/KafkaConsumer.h"
#include "kafka/KafkaProducer.h"

namespace triton { namespace server {

class KafkaEndpoint {
 public:
  KafkaEndpoint(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      const std::string& port, const std::vector<std::string>& consumer_topics);

  ~KafkaEndpoint();

  static TRITONSERVER_Error* Create(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      const std::string& port, const std::vector<std::string>& consumer_topics,
      std::unique_ptr<KafkaEndpoint>* kafka_endpoint);

  TRITONSERVER_Error* Start();

  TRITONSERVER_Error* StartProducer();

  TRITONSERVER_Error* StartConsumer();

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
      char* buffer_;
      char* cuda_ipc_handle_;

      // For non-shared memory
      OutputInfo(Kind k, uint32_t class_cnt)
          : kind_(k), class_cnt_(class_cnt), buffer_(nullptr)
      {
      }

      // For shared memory
      OutputInfo(
          void* base, uint64_t byte_size, TRITONSERVER_MemoryType memory_type,
          int64_t device_id, char* cuda_ipc_handle)
          : kind_(SHM), base_(base), byte_size_(byte_size),
            memory_type_(memory_type), device_id_(device_id), class_cnt_(0),
            buffer_(nullptr), cuda_ipc_handle_(cuda_ipc_handle)
      {
      }

      ~OutputInfo()
      {
        if (buffer_ != nullptr) {
          free(buffer_);
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
        TRITONSERVER_Server* server): server_(server), response_count_(0) {};
    ~InferRequestClass() = default;

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
    //virtual void SetResponseHeader(
        //const bool has_binary_data, const size_t header_length);

    uint32_t IncrementResponseCount();


    AllocPayload alloc_payload_;

    // Data that cannot be used directly from the HTTP body is first
    // serialized. Hold that data here so that its lifetime spans the
    // lifetime of the request.
    //std::list<std::vector<char>> serialized_data_;

   protected:
    TRITONSERVER_Server* server_;
    // Counter to keep track of number of responses generated.
    std::atomic<uint32_t> response_count_;
  };

  TRITONSERVER_Error* ProduceInferenceResponse(
      std::unique_ptr<kafka::clients::producer::ProducerRecord>&
          producer_response_msg);

  void ConsumeRequests();

  void CreateInferenceRequestMap(
      std::map<std::string, std::string>& inference_request_map,
      const kafka::clients::consumer::ConsumerRecord& inference_request_msg);

  TRITONSERVER_Error* FindParameter(
      std::map<std::string, std::string>& inference_request_map,
      const char* parameter, std::string* value);

  TRITONSERVER_Error* HandleInferenceRequest(
      const kafka::clients::consumer::ConsumerRecord& inference_request_msg);

  TRITONSERVER_Error* ParseInferenceRequestPayload(
      std::map<std::string, std::string>& inference_request_map,
      const kafka::clients::consumer::ConsumerRecord& inference_request_msg,
      TRITONSERVER_InferenceRequest* irequest, InferRequestClass* infer_req);

  TRITONSERVER_Error* ExecuteInferenceRequest(
      std::map<std::string, std::string>& inference_request_map,
      TRITONSERVER_InferenceRequest* irequest, std::unique_ptr<InferRequestClass>& infer_req);

  /*static void InferRequestComplete(
      TRITONSERVER_InferenceRequest* request, const uint32_t flags,
      void* userp);*/

  TRITONSERVER_Error* Stop();

  void DisplayConsumerTopics();

  void CreateInferenceResponse(
      std::vector<std::pair<std::string, std::string>>& header_pair_vector,
      const std::string& val);

  static TRITONSERVER_Error* InferResponseAlloc(
      TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
      size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
      int64_t preferred_memory_type_id, void* userp, void** buffer,
      void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
      int64_t* actual_memory_type_id);

  static TRITONSERVER_Error* InferResponseFree(
      TRITONSERVER_ResponseAllocator* allocator, void* buffer,
      void* buffer_userp, size_t byte_size, TRITONSERVER_MemoryType memory_type,
      int64_t memory_type_id);

  static TRITONSERVER_Error* OutputBufferQuery(
      TRITONSERVER_ResponseAllocator* allocator, void* userp,
      const char* tensor_name, size_t* byte_size,
      TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id);

  static TRITONSERVER_Error* OutputBufferAttributes(
      TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
      TRITONSERVER_BufferAttributes* buffer_attributes, void* userp,
      void* buffer_userp);

  /*static void InferResponseComplete(
        TRITONSERVER_InferenceResponse* response, const uint32_t flags,
        void* userp);*/

  protected:
    virtual std::unique_ptr<InferRequestClass> CreateInferRequest()
    {
        return std::unique_ptr<InferRequestClass>(new InferRequestClass(
            server_.get()));
    }


 private:
  std::shared_ptr<TRITONSERVER_Server> server_;
  std::shared_ptr<SharedMemoryManager> shm_manager_;
  const std::string port_;
  std::set<std::string> consumer_topics_;
  std::thread consumer_thread_;
  bool consumer_active_;
  std::unique_ptr<kafka::clients::KafkaConsumer> consumer_;
  std::unique_ptr<kafka::clients::KafkaProducer> producer_;
  std::map<std::string, std::map<std::string, std::string>> request_header_map_;
  TRITONSERVER_ResponseAllocator* allocator_;
  AllocPayload alloc_payload_;
};

}}  // namespace triton::server