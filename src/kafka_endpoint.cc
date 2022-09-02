// Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "kafka_endpoint.h"
#include "common.h"

namespace triton { namespace server {

KafkaEndpoint::KafkaEndpoint(
    const std::shared_ptr<TRITONSERVER_Server>& server,
    const std::shared_ptr<SharedMemoryManager>& shm_manager,
    const std::string& port, const std::vector<std::string>& consumer_topics)
    : server_(server), shm_manager_(shm_manager), port_(port),
      consumer_topics_(consumer_topics.begin(), consumer_topics.end())

{
  allocator_ = nullptr;
  FAIL_IF_ERR(
      TRITONSERVER_ResponseAllocatorNew(
          &allocator_, InferResponseAlloc, InferResponseFree,
          nullptr /* start_fn */),
      "creating response allocator");
  FAIL_IF_ERR(
      TRITONSERVER_ResponseAllocatorSetQueryFunction(
          allocator_, OutputBufferQuery),
      "setting allocator's query function");
  FAIL_IF_ERR(
      TRITONSERVER_ResponseAllocatorSetBufferAttributesFunction(
          allocator_, OutputBufferAttributes),
      "setting allocator's buffer attributes function");
}

KafkaEndpoint::~KafkaEndpoint()
{
  IGNORE_ERR(Stop());
  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_ResponseAllocatorDelete(allocator_),
      "deleting response allocator");
}


TRITONSERVER_Error*
KafkaEndpoint::Create(
    const std::shared_ptr<TRITONSERVER_Server>& server,
    const std::shared_ptr<SharedMemoryManager>& shm_manager,
    const std::string& port, const std::vector<std::string>& consumer_topics,
    std::unique_ptr<KafkaEndpoint>* kafka_endpoint)
{
  if (port.empty() || consumer_topics.empty()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Kakfa [port], [producer topic], or [consumer topics] not specified");
  }
  kafka_endpoint->reset(
      new KafkaEndpoint(server, shm_manager, port, consumer_topics));

  LOG_INFO << "Started Kafka Endpoint, subscribed to port: " << port;

  return nullptr;  // Success
}

TRITONSERVER_Error*
KafkaEndpoint::Start()
{
  RETURN_IF_ERR(StartProducer());
  RETURN_IF_ERR(StartConsumer());

  return nullptr;  // Success
}

TRITONSERVER_Error*
KafkaEndpoint::StartProducer()
{
  if (producer_ != nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_ALREADY_EXISTS,
        "Kafka Endpoint producer is already running!");
  }
  kafka::Properties props({
      {"bootstrap.servers", port_},
      {"enable.idempotence", "true"},
  });
  // Create a producer instance.
  producer_ = std::make_unique<kafka::clients::KafkaProducer>(props);

  return nullptr;  // Success
}

TRITONSERVER_Error*
KafkaEndpoint::StartConsumer()
{
  if (consumer_thread_.joinable()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_ALREADY_EXISTS,
        "Kafka Endpoint consumer is already running!");
  }
  try {
    kafka::Properties props(
        {{"bootstrap.servers", port_}, {"enable.auto.commit", "true"}});

    // Create a consumer instance
    consumer_ = std::make_unique<kafka::clients::KafkaConsumer>(props);

    // Subscribe to topics
    // TODO NOTE ______
    consumer_->subscribe({consumer_topics_});
  }
  catch (const kafka::KafkaException& e) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (e.what() + std::string(" Check topics exist.")).c_str());
  }
  consumer_active_ = true;
  consumer_thread_ = std::thread(&KafkaEndpoint::ConsumeRequests, this);

  return nullptr;  // Success
}

void
KafkaEndpoint::CreateInferenceResponse(
    std::vector<std::pair<std::string, std::string>>& header_pair_vector,
    const std::string& val)
{
  LOG_INFO << "Creating inference response";
  std::unique_ptr<kafka::clients::producer::ProducerRecord> response_record(
      std::make_unique<kafka::clients::producer::ProducerRecord>(
          "output", kafka::NullKey, kafka::Value(val.c_str(), val.size())));
  LOG_INFO << "[" << &response_record << "]";

  for (auto it = header_pair_vector.begin(); it != header_pair_vector.end();
       ++it) {
    response_record->headers().emplace_back(
        it->first, kafka::Header::Value(it->second.c_str(), it->second.size()));
  }
  ProduceInferenceResponse(response_record);
}


TRITONSERVER_Error*
KafkaEndpoint::ProduceInferenceResponse(
    std::unique_ptr<kafka::clients::producer::ProducerRecord>&
        producer_response_msg)
{
  try {
    LOG_INFO << "Response Sending! "
             << producer_response_msg->value().toString();
    kafka::clients::producer::RecordMetadata metadata =
        producer_->syncSend(*producer_response_msg);
    LOG_INFO << "Response Sent! " << producer_response_msg->value().toString();
  }
  catch (const kafka::KafkaException& e) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("Message delivery failed: " + e.error().message()).c_str());
  }
  return nullptr;  // Success
}

void
KafkaEndpoint::ConsumeRequests()
{
  try {
    DisplayConsumerTopics();

    while (consumer_active_) {
      auto records = consumer_->poll(std::chrono::milliseconds(100));
      for (const auto& record : records) {
        if (!record.error()) {
          std::cout << "% Got a new message..." << std::endl;
          std::cout << "    Topic    : " << record.topic() << std::endl;
          std::cout << "    Partition: " << record.partition() << std::endl;
          std::cout << "    Offset   : " << record.offset() << std::endl;
          std::cout << "    Timestamp: " << record.timestamp().toString()
                    << std::endl;
          std::cout << "    Headers  : " << kafka::toString(record.headers())
                    << std::endl;
          std::cout << "    Key   [" << record.key().toString() << "]"
                    << std::endl;
          std::cout << "    Value [" << record.value().toString() << "]"
                    << std::endl;
          TRITONSERVER_Error* err = HandleInferenceRequest(record);
          if (err != nullptr) {
            LOG_ERROR << "Failed to parse inference request: "
                      << TRITONSERVER_ErrorMessage(err);
            TRITONSERVER_ErrorDelete(err);
          }
        } else {
          std::cerr << record.toString() << std::endl;
        }
      }
    }
  }
  catch (const kafka::KafkaException& e) {
    LOG_ERROR << "% Unexpected exception caught: " << e.what();
  }
}

TRITONSERVER_Error*
KafkaEndpoint::HandleInferenceRequest(
    const kafka::clients::consumer::ConsumerRecord& inference_request_msg)
{
  std::string model_name;
  std::string model_version;
  std::string request_id;
  std::string response_topic;
  std::string payload_header_length;
  std::map<std::string, std::string> inference_request_map;
  CreateInferenceRequestMap(inference_request_map, inference_request_msg);

  RETURN_IF_ERR(
      FindParameter(inference_request_map, "model_name", &model_name));
  RETURN_IF_ERR(
      FindParameter(inference_request_map, "model_version", &model_version));
  RETURN_IF_ERR(
      FindParameter(inference_request_map, "response_topic", &response_topic));
  RETURN_IF_ERR(FindParameter(inference_request_map, "id", &request_id));
  RETURN_IF_ERR(FindParameter(
      inference_request_map, "payload_header_length", &payload_header_length));

  LOG_INFO << "Request Parsed: [ name: " << model_name
           << ", version: " << model_version
           << ", output topic: " << response_topic << ", id: " << request_id
           << ", payload header length: " << payload_header_length << "]";

  int64_t requested_model_version;
  RETURN_IF_ERR(GetModelVersionFromString(
      inference_request_map["model_version"].c_str(),
      &requested_model_version));

  TRITONSERVER_InferenceRequest* irequest = nullptr;
  RETURN_IF_ERR(TRITONSERVER_InferenceRequestNew(
      &irequest, server_.get(), inference_request_map["model_name"].c_str(),
      requested_model_version));

  std::unique_ptr<InferRequestClass> infer_request = CreateInferRequest();

  RETURN_IF_ERR(ParseInferenceRequestPayload(
      inference_request_map, inference_request_msg, irequest, infer_request.get()));
  const char* id = inference_request_map["id"].c_str();
  LOG_INFO << "Setting id: " << id;
  // Check if this allows duplicate id
  RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetId(irequest, id));
  LOG_INFO << "Executing request " << id;
  RETURN_IF_ERR(ExecuteInferenceRequest(inference_request_map, irequest, infer_request));

  return nullptr;  // Success
}

void
KafkaEndpoint::CreateInferenceRequestMap(
    std::map<std::string, std::string>& inference_request_map,
    const kafka::clients::consumer::ConsumerRecord& inference_request_msg)
{
  LOG_INFO << "Creating inference map";
  for (unsigned long i = 0; i < inference_request_msg.headers().size(); i++) {
    std::string key = inference_request_msg.headers().at(i).key;
    std::string value = inference_request_msg.headers().at(i).value.toString();
    inference_request_map[key] = value;
  }
}

TRITONSERVER_Error*
KafkaEndpoint::FindParameter(
    std::map<std::string, std::string>& inference_request_map,
    const char* parameter, std::string* value)
{
  auto it = inference_request_map.find(parameter);
  if (it == inference_request_map.end()) {
    LOG_INFO << "Failed to find " << std::string(parameter);
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNAVAILABLE,
        ("Inference request parameter [" + std::string(parameter) +
         "] is required, but was not found.")
            .c_str());
  }
  *value = it->second;
  return nullptr;  // Success
}

TRITONSERVER_Error*
KafkaEndpoint::ParseInferenceRequestPayload(
    std::map<std::string, std::string>& inference_request_map,
    const kafka::clients::consumer::ConsumerRecord& inference_request_msg,
    TRITONSERVER_InferenceRequest* irequest, InferRequestClass* infer_req)
{
  // Convert payload header into a parseable json object
  int64_t payload_header_length;
  try {
    payload_header_length =
        std::stol(inference_request_map["payload_header_length"].c_str());
  }
  catch (std::exception& e) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "failed to get header length from specified version string '" +
            inference_request_map["payload_header_length"] + "' (details: " +
            e.what() + "), version should be an integral value > 0")
            .c_str());
  }
  // data() returns a const void* which must be cast away and
  // reinterpreted. Cannot use member function toString() 
  // because it will corrupt the binary data payload.
  const char* payload_ptr = reinterpret_cast<char*>(
      const_cast<void*>(inference_request_msg.value().data()));
  std::string complete_payload(
      payload_ptr, inference_request_msg.value().size());

  std::string payload_header =
      complete_payload.substr(0, payload_header_length);
  LOG_INFO << "Payload header: " << payload_header;
  triton::common::TritonJson::Value payload_header_json;
  payload_header_json.Parse(payload_header.c_str(), payload_header.size());

  // Parse the input characteristics
  triton::common::TritonJson::Value inputs_json;
  RETURN_MSG_IF_ERR(
      payload_header_json.MemberAsArray("inputs", &inputs_json),
      "Unable to parse 'inputs'");

  std::string binary_data = complete_payload.substr(
      payload_header_length, (complete_payload.size() - payload_header_length));
  int binary_data_offset = 0;

  // Iterate over 'input' elements
  for (size_t i = 0; i < inputs_json.ArraySize(); i++) {
    triton::common::TritonJson::Value request_input;
    RETURN_IF_ERR(inputs_json.At(i, &request_input));

    // Parse NAME
    const char* input_name;
    size_t input_name_len;
    RETURN_MSG_IF_ERR(
        request_input.MemberAsString("name", &input_name, &input_name_len),
        "Unable to parse 'name'");

    // Parse DATATYPE
    const char* datatype;
    size_t datatype_len;
    RETURN_MSG_IF_ERR(
        request_input.MemberAsString("datatype", &datatype, &datatype_len),
        "Unable to parse 'datatype'");
    const TRITONSERVER_DataType dtype = TRITONSERVER_StringToDataType(datatype);

    // Parse SHAPE
    triton::common::TritonJson::Value shape_json;
    RETURN_MSG_IF_ERR(
        request_input.MemberAsArray("shape", &shape_json),
        "Unable to parse 'shape'");
    std::vector<int64_t> shape_vec;
    for (size_t i = 0; i < shape_json.ArraySize(); i++) {
      uint64_t d = 0;
      RETURN_MSG_IF_ERR(
          shape_json.IndexAsUInt(i, &d), "Unable to parse 'shape'");
      shape_vec.push_back(d);
    }

    LOG_INFO << "Adding input, name: [" << input_name << "] type ["
             << (int)dtype << "]";
    LOG_INFO << "input shape, dim [" << shape_vec.size() << "] shape: [";
    for (auto i = shape_vec.begin(); i != shape_vec.end(); i++) {
      LOG_INFO << *i;
    }
    LOG_INFO << "]";

    RETURN_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
        irequest, input_name, dtype, &shape_vec[0], shape_vec.size()));

    // Parse BINARY_DATA_SIZE
    bool binary_payload = false;
    size_t payload_length;
    triton::common::TritonJson::Value params_json;
    if (request_input.Find("parameters", &params_json)) {
      triton::common::TritonJson::Value binary_data_size_json;
      if (params_json.Find("binary_data_size", &binary_data_size_json)) {
        RETURN_MSG_IF_ERR(
            binary_data_size_json.AsUInt(&payload_length),
            "Unable to parse 'binary_data_size'");
        binary_payload = true;
      }
    }
    if (!binary_payload) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string("Payload is not comprised of binary data").c_str());
    }
    LOG_INFO << "Payload length: " << payload_length;
    // Permitted b/c strings are stored as contiguous memory in c++11
    char* start = &binary_data[binary_data_offset];
    // TODO double check using memcpy
    unsigned char* base = (unsigned char*)malloc(payload_length + 1);
    for (size_t i = 0; i < payload_length; i++) {
      base[i] = *reinterpret_cast<unsigned char*>(start);
      start++;
    }
    base[payload_length - 1] = '\0';
    // Append the binary payload
    // base_final = base;
    RETURN_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
        irequest, input_name, base, payload_length, TRITONSERVER_MEMORY_CPU,
        0));
    binary_data_offset += payload_length;
  }

  if (payload_header_json.Find("outputs")) {
    triton::common::TritonJson::Value outputs_json;
    RETURN_MSG_IF_ERR(
        payload_header_json.MemberAsArray("outputs", &outputs_json),
        "Unable to parse 'outputs'");
    for (size_t i = 0; i < outputs_json.ArraySize(); i++) {
      triton::common::TritonJson::Value request_output;
      RETURN_IF_ERR(outputs_json.At(i, &request_output));

      const char* output_name;
      size_t output_name_len;
      RETURN_MSG_IF_ERR(
          request_output.MemberAsString("name", &output_name, &output_name_len),
          "Unable to parse 'name'");
      RETURN_IF_ERR(TRITONSERVER_InferenceRequestAddRequestedOutput(
          irequest, output_name));

      uint64_t class_count = 0;
      infer_req->alloc_payload_.output_map_.emplace(
          std::piecewise_construct, std::forward_as_tuple(output_name),
          std::forward_as_tuple(new AllocPayload::OutputInfo(
              AllocPayload::OutputInfo::BINARY, class_count)));
    }
  }
  infer_req->alloc_payload_.default_output_kind_ = AllocPayload::OutputInfo::BINARY;

  return nullptr;
}

TRITONSERVER_Error*
KafkaEndpoint::ExecuteInferenceRequest(
    std::map<std::string, std::string>& inference_request_map,
    TRITONSERVER_InferenceRequest* irequest, std::unique_ptr<InferRequestClass>& infer_req)
{
  const char* request_id = nullptr;
  TRITONSERVER_Error* err = nullptr;
  if (err == nullptr) {
    err = TRITONSERVER_InferenceRequestSetReleaseCallback(
        irequest, InferRequestClass::InferRequestComplete, nullptr);
    if (err == nullptr) {
      err = TRITONSERVER_InferenceRequestSetResponseCallback(
          irequest, allocator_, reinterpret_cast<void*>(&infer_req->alloc_payload_),
          InferRequestClass::InferResponseComplete, reinterpret_cast<void*>(infer_req.get()));
    }
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceRequestId(irequest, &request_id),
        "unable to retrieve request ID string");
    if ((request_id == nullptr) || (request_id[0] == '\0')) {
      request_id = "<id_unknown>";
    }
    LOG_INFO << "Making inference";
    if (err == nullptr) {
      err = TRITONSERVER_ServerInferAsync(server_.get(), irequest, nullptr);
    }
    if (err != nullptr) {
      LOG_INFO << "[request id: " << request_id << "] "
               << "Infer failed: " << TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
      LOG_TRITONSERVER_ERROR(
          TRITONSERVER_InferenceRequestDelete(irequest),
          "deleting HTTP/REST inference request");
    }
    if (err == nullptr) {
      infer_req.release();
    }
  }
  return nullptr;
}

void
KafkaEndpoint::InferRequestClass::InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  LOG_INFO << "KafkaEndpoint::InferRequestComplete";

  if ((flags & TRITONSERVER_REQUEST_RELEASE_ALL) != 0) {
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceRequestDelete(request),
        "deleting Kafka inference request");
  }
}

void
KafkaEndpoint::DisplayConsumerTopics()
{
  std::string print_consumer_topics = "{";
  for (auto s : consumer_topics_) {
    print_consumer_topics += s + ",";
  }
  print_consumer_topics.replace(print_consumer_topics.size() - 1, 1, "}");
  LOG_INFO << "Kafka Endpoint: consumer topics: " << print_consumer_topics;
}

TRITONSERVER_Error*
KafkaEndpoint::Stop()
{
  if (consumer_thread_.joinable()) {
    consumer_active_ = false;
    consumer_thread_.join();
    consumer_.reset();
  } else {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNAVAILABLE, "Kafka consumer not running.");
  }
  producer_.reset();

  return nullptr;  // Success
}

// Make sure to keep InferResponseAlloc, OutputBufferQuery, and
// OutputBufferAttributes logic in sync
TRITONSERVER_Error*
KafkaEndpoint::InferResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  LOG_INFO << "Kafka Infer response alloc " << tensor_name << " " << (int)byte_size;

  AllocPayload* payload = reinterpret_cast<AllocPayload*>(userp);
  std::unordered_map<std::string, AllocPayload::OutputInfo*>& output_map =
      payload->output_map_;
  const AllocPayload::OutputInfo::Kind default_output_kind =
      payload->default_output_kind_;

  LOG_INFO << "Got map";

  *buffer = nullptr;
  *buffer_userp = nullptr;
  *actual_memory_type = preferred_memory_type;
  *actual_memory_type_id = preferred_memory_type_id;

  AllocPayload::OutputInfo* info = nullptr;

  // If we don't find an output then it means that the output wasn't
  // explicitly specified in the request. In that case we create an
  // OutputInfo for it that uses default setting of JSON.
  auto pr = output_map.find(tensor_name);
  if (pr == output_map.end()) {
    LOG_INFO << "Could not find tensor name in output";
    info = new AllocPayload::OutputInfo(default_output_kind, 0);
  } else {
    // Take ownership of the OutputInfo object.
    LOG_INFO << "Found tensor name in output";
    info = pr->second;
    output_map.erase(pr);
  }

  // Don't need to do anything if no memory was requested.
  if (byte_size > 0) {
    // Can't allocate for any memory type other than CPU. If asked to
    // allocate on GPU memory then force allocation on CPU instead.
    if (*actual_memory_type != TRITONSERVER_MEMORY_CPU) {
      LOG_INFO << "Kafka: unable to provide '" << tensor_name << "' in "
               << TRITONSERVER_MemoryTypeString(*actual_memory_type)
               << ", will use "
               << TRITONSERVER_MemoryTypeString(TRITONSERVER_MEMORY_CPU);
      *actual_memory_type = TRITONSERVER_MEMORY_CPU;
      *actual_memory_type_id = 0;
    }
    LOG_INFO << "Allocating memory";
    char* transfer_buffer = (char*)malloc(byte_size+1);
    *buffer = transfer_buffer;

    // Ownership passes to 'buffer_userp' which has the same lifetime
    // as the buffer itself.
    LOG_INFO << "Transferring ownership";
    info->buffer_ = transfer_buffer;
    transfer_buffer[byte_size] = '\0';
    LOG_INFO << "Kafka using buffer for: '" << tensor_name
             << "', size: " << byte_size << ", addr: " << *buffer;
  }
  LOG_INFO << "Casting data back to userp";
  *buffer_userp = reinterpret_cast<void*>(info);

  return nullptr;  // Success
}


TRITONSERVER_Error*
KafkaEndpoint::InferResponseFree(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  LOG_INFO << "Kafka release: "
           << "size " << byte_size << ", addr " << buffer;

  // 'buffer' is backed by shared memory or evbuffer so we don't
  // delete directly.
  auto info = reinterpret_cast<AllocPayload::OutputInfo*>(buffer_userp);
  delete info;

  return nullptr;  // Success
}

// Make sure to keep InferResponseAlloc and OutputBufferQuery logic in sync
TRITONSERVER_Error*
KafkaEndpoint::OutputBufferQuery(
    TRITONSERVER_ResponseAllocator* allocator, void* userp,
    const char* tensor_name, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
{
  AllocPayload* payload = reinterpret_cast<AllocPayload*>(userp);

  if (tensor_name != nullptr) {
    auto pr = payload->output_map_.find(tensor_name);
    if ((pr != payload->output_map_.end()) &&
        (pr->second->kind_ == AllocPayload::OutputInfo::SHM)) {
      // The output is in shared memory so check that shared memory
      // size is at least large enough for the output, if byte size is provided
      if ((byte_size != nullptr) && (*byte_size > pr->second->byte_size_)) {
        // Don't return error yet and just set to the default properties for
        // GRPC buffer, error will be raised when allocation happens
        *memory_type = TRITONSERVER_MEMORY_CPU;
        *memory_type_id = 0;
      } else {
        *memory_type = pr->second->memory_type_;
        *memory_type_id = pr->second->device_id_;
      }
      return nullptr;  // Success
    }
  }

  // Not using shared memory so a evhtp buffer will be used,
  // and the type will be CPU.
  *memory_type = TRITONSERVER_MEMORY_CPU;
  *memory_type_id = 0;
  return nullptr;  // Success
}

// Make sure to keep InferResponseAlloc, OutputBufferQuery, and
// OutputBufferAttributes logic in sync
TRITONSERVER_Error*
KafkaEndpoint::OutputBufferAttributes(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    TRITONSERVER_BufferAttributes* buffer_attributes, void* userp,
    void* buffer_userp)
{
  AllocPayload::OutputInfo* info =
      reinterpret_cast<AllocPayload::OutputInfo*>(buffer_userp);

  // We only need to set the cuda ipc handle here. The rest of the buffer
  // attributes have been properly populated by triton core.
  if (tensor_name != nullptr) {
    if (info->kind_ == AllocPayload::OutputInfo::SHM &&
        info->memory_type_ == TRITONSERVER_MEMORY_GPU) {
      RETURN_IF_ERR(TRITONSERVER_BufferAttributesSetCudaIpcHandle(
          buffer_attributes, info->cuda_ipc_handle_));
    }
  }

  return nullptr;  // Success
}


void
KafkaEndpoint::InferRequestClass::InferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  // FIXME can't use InferRequestClass object here since it's lifetime
  // is different than response. For response we need to know how to
  // send each output (as json, shm, or binary) and that information
  // has to be maintained in a way that allows us to clean it up
  // appropriately if connection closed or last response sent.
  //
  // But for now userp is the InferRequestClass object and the end of
  // its life is in the OK or BAD ReplyCallback.

  LOG_INFO << "Inference response complete";
  KafkaEndpoint::InferRequestClass* infer_request =
      reinterpret_cast<KafkaEndpoint::InferRequestClass*>(userp);

  auto response_count = infer_request->IncrementResponseCount();

  // Defer to the callback with the final response
  if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) == 0) {
    LOG_ERROR << "[INTERNAL] received a response without FINAL flag";
    return;
  }

  TRITONSERVER_Error* err = nullptr;
  if (response_count != 0) {
    err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, std::string(
                                         "expected a single response, got " +
                                         std::to_string(response_count + 1))
                                         .c_str());
  } else if (response == nullptr) {
    err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "received an unexpected null response");
  } else {
    err = infer_request->FinalizeResponse(response);
  }

  if (err == nullptr) {
    LOG_INFO << "Response finalized";
  } else {
    LOG_ERROR << "Failed to parse inference request: "
                      << TRITONSERVER_ErrorMessage(err);
    LOG_INFO << "Deleting error message";
    TRITONSERVER_ErrorDelete(err);
  }
  LOG_INFO << "Deleting request";
  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceResponseDelete(response),
      "deleting inference response");
}


uint32_t
KafkaEndpoint::InferRequestClass::IncrementResponseCount()
{
  return response_count_++;
}

TRITONSERVER_Error*
KafkaEndpoint::InferRequestClass::FinalizeResponse(
    TRITONSERVER_InferenceResponse* response)
{
  LOG_INFO << "Kafka finalizing response";
  RETURN_IF_ERR(TRITONSERVER_InferenceResponseError(response));
  std::vector<std::pair<std::string, std::string>> header_pair_vector;

  triton::common::TritonJson::Value response_json(
      triton::common::TritonJson::ValueType::OBJECT);

  const char* request_id = nullptr;
  RETURN_IF_ERR(TRITONSERVER_InferenceResponseId(response, &request_id));
  header_pair_vector.push_back(std::pair<std::string,std::string>("id", request_id));

  const char* model_name;
  int64_t model_version;
  RETURN_IF_ERR(TRITONSERVER_InferenceResponseModel(
      response, &model_name, &model_version));
  header_pair_vector.push_back(std::pair<std::string,std::string>("model_name", model_name));
  header_pair_vector.push_back(std::pair<std::string,std::string>("model_version", std::to_string(model_version)));

  // If the response has any parameters, convert them to JSON.
  uint32_t parameter_count;
  RETURN_IF_ERR(
      TRITONSERVER_InferenceResponseParameterCount(response, &parameter_count));
  if (parameter_count > 0) {
    triton::common::TritonJson::Value params_json(
        response_json, triton::common::TritonJson::ValueType::OBJECT);

    for (uint32_t pidx = 0; pidx < parameter_count; ++pidx) {
      const char* name;
      TRITONSERVER_ParameterType type;
      const void* vvalue;
      RETURN_IF_ERR(TRITONSERVER_InferenceResponseParameter(
          response, pidx, &name, &type, &vvalue));
      switch (type) {
        case TRITONSERVER_PARAMETER_BOOL:
          RETURN_IF_ERR(params_json.AddBool(
              name, *(reinterpret_cast<const bool*>(vvalue))));
          break;
        case TRITONSERVER_PARAMETER_INT:
          RETURN_IF_ERR(params_json.AddInt(
              name, *(reinterpret_cast<const int64_t*>(vvalue))));
          break;
        case TRITONSERVER_PARAMETER_STRING:
          RETURN_IF_ERR(params_json.AddStringRef(
              name, reinterpret_cast<const char*>(vvalue)));
          break;
        case TRITONSERVER_PARAMETER_BYTES:
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED,
              "Response parameter of type 'TRITONSERVER_PARAMETER_BYTES' is "
              "not currently supported");
          break;
      }
    }

    RETURN_IF_ERR(response_json.Add("parameters", std::move(params_json)));
  }

  // Go through each response output and transfer information to JSON
  uint32_t output_count;
  RETURN_IF_ERR(
      TRITONSERVER_InferenceResponseOutputCount(response, &output_count));

  std::vector<char*> ordered_buffers;
  ordered_buffers.reserve(output_count);

  triton::common::TritonJson::Value response_outputs(
      response_json, triton::common::TritonJson::ValueType::ARRAY);

  std::map<int, int> byte_allocation_map;
  for (uint32_t idx = 0; idx < output_count; ++idx) {
    const char* cname;
    TRITONSERVER_DataType datatype;
    const int64_t* shape;
    uint64_t dim_count;
    const void* base;
    size_t byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    void* userp;

    RETURN_IF_ERR(TRITONSERVER_InferenceResponseOutput(
        response, idx, &cname, &datatype, &shape, &dim_count, &base, &byte_size,
        &memory_type, &memory_type_id, &userp));

    triton::common::TritonJson::Value output_json(
        response_json, triton::common::TritonJson::ValueType::OBJECT);
    RETURN_IF_ERR(output_json.AddStringRef("name", cname));

    // Handle data. SHM outputs will not have an info.
    auto info = reinterpret_cast<AllocPayload::OutputInfo*>(userp);

    size_t element_count = 1;

    const char* datatype_str = TRITONSERVER_DataTypeString(datatype);
    RETURN_IF_ERR(output_json.AddStringRef("datatype", datatype_str));

    triton::common::TritonJson::Value shape_json(
        response_json, triton::common::TritonJson::ValueType::ARRAY);
    for (size_t j = 0; j < dim_count; j++) {
      RETURN_IF_ERR(shape_json.AppendUInt(shape[j]));
      element_count *= shape[j];
    }

    RETURN_IF_ERR(output_json.Add("shape", std::move(shape_json)));

    // Add JSON data, or collect binary data.
    if (info->kind_ == AllocPayload::OutputInfo::BINARY) {
      triton::common::TritonJson::Value parameters_json;
      if (!output_json.Find("parameters", &parameters_json)) {
        parameters_json = triton::common::TritonJson::Value(
            response_json, triton::common::TritonJson::ValueType::OBJECT);
        RETURN_IF_ERR(parameters_json.AddUInt("binary_data_size", byte_size));
        RETURN_IF_ERR(
            output_json.Add("parameters", std::move(parameters_json)));
      } else {
        RETURN_IF_ERR(parameters_json.AddUInt("binary_data_size", byte_size));
      }
      if (byte_size > 0) {
        ordered_buffers.push_back(info->buffer_);
        byte_allocation_map[idx] = byte_size;
      }
    }

    RETURN_IF_ERR(response_outputs.Append(std::move(output_json)));
  }

  /* = {
      {"model_name", "python_float32_float32_float32"},
      {"model_version", "1"},
      {"response_topic", "output"},
      {"id", "1"},
      {"payload_header_length", "317"}};*/

  RETURN_IF_ERR(response_json.Add("outputs", std::move(response_outputs)));

  triton::common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERR(response_json.Write(&buffer));
  LOG_INFO << "Json response: " << buffer.Base();

  int additional_size = 0;
  if (!ordered_buffers.empty()) {
    for (size_t i = 0; i < ordered_buffers.size(); i++) {
      additional_size += byte_allocation_map[i];
    }
  }

  int total_size = buffer.Size()+additional_size+1;
  char* response_payload = (char*)malloc(total_size);
  memcpy(response_payload, (char*)&buffer, buffer.Size());

  // If there is binary data write it next in the appropriate
  // order... also need the HTTP header when returning binary data.
  int offset = buffer.Size();
  for(size_t i = 0; i < ordered_buffers.size(); i++) {
    memcpy(&response_payload[offset], &ordered_buffers[i][0], byte_allocation_map[i]);
    offset += byte_allocation_map[i];
  }
  response_payload[total_size] = '\0';
  LOG_INFO << "Total size: " << total_size;
  std::string payload(response_payload);

  LOG_INFO << response_payload;
  //CreateInferenceResponse(header_pair_vector, payload);

  free(response_payload);

  return nullptr;  // success
}

}}  // namespace triton::server
