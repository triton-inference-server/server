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

#include "src/clients/c++/perf_analyzer/client_backend/tensorflow_serving/grpc_client.h"
#include "src/clients/c++/perf_analyzer/client_backend/tensorflow_serving/tfserve_client_backend.h"


#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>

/// Type alias for string-TensorProto map.
typedef google::protobuf::Map<std::string, tensorflow::TensorProto>
    StringKeyedProtos;

namespace perfanalyzer { namespace clientbackend { namespace tfserving {

namespace {

// Use map to keep track of GRPC channels. <key, value> : <url, Channel*>
// If context is created on url that has established Channel, then reuse it.
std::map<std::string, std::shared_ptr<grpc::Channel>> grpc_channel_map_;
std::mutex grpc_channel_map_mtx_;

void
GetTensorFlowDataType(const std::string& datatype, tensorflow::DataType* dtype)
{
  if (datatype == "FP16") {
    *dtype = tensorflow::DataType::DT_HALF;
  } else if (datatype == "FP32") {
    *dtype = tensorflow::DataType::DT_FLOAT;
  } else if (datatype == "FP64") {
    *dtype = tensorflow::DataType::DT_DOUBLE;
  } else if (datatype == "INT32") {
    *dtype = tensorflow::DataType::DT_INT32;
  } else if (datatype == "INT16") {
    *dtype = tensorflow::DataType::DT_INT16;
  } else if (datatype == "UINT16") {
    *dtype = tensorflow::DataType::DT_UINT16;
  } else if (datatype == "INT8") {
    *dtype = tensorflow::DataType::DT_INT8;
  } else if (datatype == "UINT8") {
    *dtype = tensorflow::DataType::DT_UINT8;
  } else if (datatype == "BYTES") {
    *dtype = tensorflow::DataType::DT_STRING;
  } else if (datatype == "INT64") {
    *dtype = tensorflow::DataType::DT_INT64;
  } else if (datatype == "BOOL") {
    *dtype = tensorflow::DataType::DT_BOOL;
  } else if (datatype == "UINT32") {
    *dtype = tensorflow::DataType::DT_UINT32;
  } else if (datatype == "UINT64") {
    *dtype = tensorflow::DataType::DT_UINT64;
  } else {
    *dtype = tensorflow::DT_INVALID;
  }
}

void
ReadFile(const std::string& filename, std::string& data)
{
  data.clear();
  if (!filename.empty()) {
    std::ifstream file(filename.c_str(), std::ios::in);
    if (file.is_open()) {
      std::stringstream ss;
      ss << file.rdbuf();
      file.close();
      data = ss.str();
    }
  }
}

std::shared_ptr<grpc::Channel>
GetChannel(const std::string& url, bool use_ssl, const SslOptions& ssl_options)
{
  std::lock_guard<std::mutex> lock(grpc_channel_map_mtx_);

  const auto& channel_itr = grpc_channel_map_.find(url);
  if (channel_itr != grpc_channel_map_.end()) {
    return channel_itr->second;
  } else {
    grpc::ChannelArguments arguments;
    arguments.SetMaxSendMessageSize(nic::MAX_GRPC_MESSAGE_SIZE);
    arguments.SetMaxReceiveMessageSize(nic::MAX_GRPC_MESSAGE_SIZE);
    std::shared_ptr<grpc::ChannelCredentials> credentials;
    if (use_ssl) {
      std::string root;
      std::string key;
      std::string cert;
      ReadFile(ssl_options.root_certificates, root);
      ReadFile(ssl_options.private_key, key);
      ReadFile(ssl_options.certificate_chain, cert);
      grpc::SslCredentialsOptions opts = {root, key, cert};
      credentials = grpc::SslCredentials(opts);
    } else {
      credentials = grpc::InsecureChannelCredentials();
    }
    std::shared_ptr<grpc::Channel> channel =
        grpc::CreateCustomChannel(url, credentials, arguments);
    grpc_channel_map_.insert(std::make_pair(url, channel));
    return channel;
  }
}

}  // namespace

//==============================================================================
// An GrpcInferRequest represents an inflght inference request on gRPC.
//
class GrpcInferRequest {
 public:
  GrpcInferRequest(TFServeOnCompleteFn callback = nullptr)
      : callback_(callback), grpc_status_(),
        grpc_response_(std::make_shared<tensorflow::serving::PredictResponse>())
  {
  }

  nic::RequestTimers& Timer() { return timer_; }
  friend GrpcClient;

 private:
  TFServeOnCompleteFn callback_;
  // Variables for GRPC call
  grpc::ClientContext grpc_context_;
  grpc::Status grpc_status_;
  std::shared_ptr<tensorflow::serving::PredictResponse> grpc_response_;
  // The timers for infer request.
  nic::RequestTimers timer_;
};

//==============================================================================

Error
GrpcClient::Create(
    std::unique_ptr<GrpcClient>* client, const std::string& server_url,
    bool verbose, bool use_ssl, const SslOptions& ssl_options)
{
  client->reset(new GrpcClient(server_url, verbose, use_ssl, ssl_options));
  return Error::Success;
}

Error
GrpcClient::ModelMetadata(
    tensorflow::serving::GetModelMetadataResponse* model_metadata,
    const std::string& model_name, const std::string& model_version,
    const Headers& headers)
{
  model_metadata->Clear();
  Error err;

  tensorflow::serving::GetModelMetadataRequest request;
  grpc::ClientContext context;

  for (const auto& it : headers) {
    context.AddMetadata(it.first, it.second);
  }

  request.mutable_model_spec()->set_name(model_name);
  if (!model_version.empty()) {
    request.mutable_model_spec()->set_version_label(model_version);
  }
  request.add_metadata_field("signature_def");
  grpc::Status grpc_status =
      stub_->GetModelMetadata(&context, request, model_metadata);
  if (grpc_status.ok()) {
    if (verbose_) {
      std::cout << model_metadata->DebugString() << std::endl;
    }
  } else {
    err = Error(grpc_status.error_message());
  }

  return err;
}

Error
GrpcClient::Infer(
    InferResult** result, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs,
    const Headers& headers)
{
  Error err;

  grpc::ClientContext context;

  std::shared_ptr<GrpcInferRequest> sync_request(new GrpcInferRequest());

  sync_request->Timer().Reset();
  sync_request->Timer().CaptureTimestamp(
      nic::RequestTimers::Kind::REQUEST_START);
  // Use send timer to measure time for marshalling infer request
  sync_request->Timer().CaptureTimestamp(nic::RequestTimers::Kind::SEND_START);
  for (const auto& it : headers) {
    context.AddMetadata(it.first, it.second);
  }

  err = PreRunProcessing(options, inputs, outputs);
  sync_request->Timer().CaptureTimestamp(nic::RequestTimers::Kind::SEND_END);
  if (!err.IsOk()) {
    return err;
  }
  sync_request->grpc_response_->Clear();
  sync_request->grpc_status_ = stub_->Predict(
      &context, infer_request_, sync_request->grpc_response_.get());

  if (!sync_request->grpc_status_.ok()) {
    err = Error(sync_request->grpc_status_.error_message());
  }

  sync_request->Timer().CaptureTimestamp(nic::RequestTimers::Kind::RECV_START);
  InferResult::Create(result, sync_request->grpc_response_, err);
  sync_request->Timer().CaptureTimestamp(nic::RequestTimers::Kind::RECV_END);

  sync_request->Timer().CaptureTimestamp(nic::RequestTimers::Kind::REQUEST_END);

  nic::Error update_err = UpdateInferStat(sync_request->Timer());
  if (!update_err.IsOk()) {
    std::cerr << "Failed to update context stat: " << update_err << std::endl;
  }

  if (sync_request->grpc_status_.ok()) {
    if (verbose_) {
      std::cout << sync_request->grpc_response_->DebugString() << std::endl;
    }
  }

  return (*result)->RequestStatus();
}

Error
GrpcClient::AsyncInfer(
    TFServeOnCompleteFn callback, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs,
    const Headers& headers)
{
  if (callback == nullptr) {
    return Error(
        "Callback function must be provided along with AsyncInfer() call.");
  }
  if (!worker_.joinable()) {
    worker_ = std::thread(&GrpcClient::AsyncTransfer, this);
  }

  GrpcInferRequest* async_request;
  async_request = new GrpcInferRequest(std::move(callback));

  async_request->Timer().CaptureTimestamp(
      nic::RequestTimers::Kind::REQUEST_START);
  async_request->Timer().CaptureTimestamp(nic::RequestTimers::Kind::SEND_START);
  for (const auto& it : headers) {
    async_request->grpc_context_.AddMetadata(it.first, it.second);
  }

  Error err = PreRunProcessing(options, inputs, outputs);
  if (!err.IsOk()) {
    delete async_request;
    return err;
  }

  async_request->Timer().CaptureTimestamp(nic::RequestTimers::Kind::SEND_END);

  std::unique_ptr<
      grpc::ClientAsyncResponseReader<tensorflow::serving::PredictResponse>>
      rpc(stub_->PrepareAsyncPredict(
          &async_request->grpc_context_, infer_request_,
          &async_request_completion_queue_));

  rpc->StartCall();

  rpc->Finish(
      async_request->grpc_response_.get(), &async_request->grpc_status_,
      (void*)async_request);

  if (verbose_) {
    std::cout << "Sent request";
    if (options.request_id_.size() != 0) {
      std::cout << " '" << options.request_id_ << "'";
    }
    std::cout << std::endl;
  }

  return Error::Success;
}

void
GrpcClient::AsyncTransfer()
{
  while (!exiting_) {
    // GRPC async APIs are thread-safe https://github.com/grpc/grpc/issues/4486
    GrpcInferRequest* raw_async_request;
    bool ok = true;
    bool status =
        async_request_completion_queue_.Next((void**)(&raw_async_request), &ok);
    std::shared_ptr<GrpcInferRequest> async_request;
    if (!ok) {
      fprintf(stderr, "Unexpected not ok on client side.\n");
    }
    if (!status) {
      if (!exiting_) {
        fprintf(stderr, "Completion queue is closed.\n");
      }
    } else if (raw_async_request == nullptr) {
      fprintf(stderr, "Unexpected null tag received at client.\n");
    } else {
      async_request.reset(raw_async_request);
      InferResult* async_result;
      Error err;
      if (!async_request->grpc_status_.ok()) {
        err = Error(async_request->grpc_status_.error_message());
      }
      async_request->Timer().CaptureTimestamp(
          nic::RequestTimers::Kind::RECV_START);
      InferResult::Create(&async_result, async_request->grpc_response_, err);
      async_request->Timer().CaptureTimestamp(
          nic::RequestTimers::Kind::RECV_END);
      async_request->Timer().CaptureTimestamp(
          nic::RequestTimers::Kind::REQUEST_END);
      nic::Error update_err = UpdateInferStat(async_request->Timer());
      if (!update_err.IsOk()) {
        std::cerr << "Failed to update context stat: " << update_err
                  << std::endl;
      }
      if (async_request->grpc_status_.ok()) {
        if (verbose_) {
          std::cout << async_request->grpc_response_->DebugString()
                    << std::endl;
        }
      }
      async_request->callback_(async_result);
    }
  }
}

Error
GrpcClient::PreRunProcessing(
    const InferOptions& options, const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  // Populate the request protobuf

  // Describing model name and signature from remote server.
  infer_request_.mutable_model_spec()->set_name(options.model_name_);
  if (!options.model_version_.empty()) {
    infer_request_.mutable_model_spec()->set_version_label(
        options.model_version_);
  }
  if (!options.model_signature_name_.empty()) {
    infer_request_.mutable_model_spec()->set_signature_name(
        options.model_signature_name_);
  }

  // Describing remote model inputs shape.
  StringKeyedProtos& keyed_proto_inputs = *infer_request_.mutable_inputs();
  std::set<std::string> request_inputs;

  for (const auto input : inputs) {
    auto raw_input = dynamic_cast<TFServeInferInput*>(input);
    request_inputs.insert(raw_input->Name());
    // Add new TensorProto submessages only if required, otherwise
    // reuse the submessages already available.
    auto itr = keyed_proto_inputs.find(raw_input->Name());
    if (itr == keyed_proto_inputs.end()) {
      itr = keyed_proto_inputs
                .insert(google::protobuf::MapPair<
                        std::string, tensorflow::TensorProto>(
                    raw_input->Name(), tensorflow::TensorProto()))
                .first;
    }

    // Set datatype
    tensorflow::DataType tf_dtype = tensorflow::DT_INVALID;
    GetTensorFlowDataType(raw_input->Datatype(), &tf_dtype);
    itr->second.set_dtype(tf_dtype);
    if (tf_dtype == tensorflow::DT_INVALID) {
      return Error(
          "failed to retrieve the TF datatype for " + raw_input->Name());
    }

    // Populate the shape
    itr->second.mutable_tensor_shape()->Clear();
    for (const auto dim : raw_input->Shape()) {
      itr->second.mutable_tensor_shape()->add_dim()->set_size(dim);
    }

    raw_input->PrepareForRequest();
    // There is an extra copy into the buffer to collect all the input
    // batches. This is a room for improvement for later.
    bool end_of_input = false;

    // auto* raw_contents = itr->second.mutable_float_val()->mutable_data();
    size_t content_size;
    raw_input->ByteSize(&content_size);
    temp_buffer_.clear();
    temp_buffer_.reserve(content_size);
    while (!end_of_input) {
      const uint8_t* buf;
      size_t buf_size;
      raw_input->GetNext(&buf, &buf_size, &end_of_input);
      if (buf != nullptr) {
        temp_buffer_.append(reinterpret_cast<const char*>(buf), buf_size);
      }
    }
    ClearAllInputFields(&itr->second);
    PopulateInputData(raw_input, &itr->second);
  }

  // Remove extra tensor protos, if any.
  std::set<std::string> extra_inputs;
  for (const auto& iter : keyed_proto_inputs) {
    if (request_inputs.find(iter.first) == request_inputs.end()) {
      extra_inputs.insert(iter.first);
    }
  }
  for (const auto& extra_input : extra_inputs) {
    keyed_proto_inputs.erase(extra_input);
  }

  if (infer_request_.ByteSizeLong() > INT_MAX) {
    size_t request_size = infer_request_.ByteSizeLong();
    infer_request_.Clear();
    return Error(
        "Request has byte size " + std::to_string(request_size) +
        " which exceed gRPC's byte size limit " + std::to_string(INT_MAX) +
        ".");
  }

  return Error::Success;
}

Error
GrpcClient::ClearAllInputFields(tensorflow::TensorProto* input_tensor_proto)
{
  input_tensor_proto->mutable_half_val()->Clear();
  input_tensor_proto->mutable_float_val()->Clear();
  input_tensor_proto->mutable_double_val()->Clear();
  input_tensor_proto->mutable_int_val()->Clear();
  input_tensor_proto->mutable_string_val()->Clear();
  input_tensor_proto->mutable_int64_val()->Clear();
  input_tensor_proto->mutable_bool_val()->Clear();
  input_tensor_proto->mutable_uint32_val()->Clear();
  input_tensor_proto->mutable_uint64_val()->Clear();

  return Error::Success;
}

Error
GrpcClient::PopulateInputData(
    TFServeInferInput* input, tensorflow::TensorProto* input_tensor_proto)
{
  if (input->Datatype() == "FP16") {
    RETURN_IF_CB_ERROR(PopulateHalfVal(input_tensor_proto));
  } else if (input->Datatype() == "FP32") {
    RETURN_IF_CB_ERROR(PopulateFloatVal(input_tensor_proto));
  } else if (input->Datatype() == "FP64") {
    RETURN_IF_CB_ERROR(PopulateDoubleVal(input_tensor_proto));
  } else if (input->Datatype() == "INT32") {
    RETURN_IF_CB_ERROR(PopulateIntVal(input_tensor_proto));
  } else if (input->Datatype() == "INT16") {
    RETURN_IF_CB_ERROR(PopulateIntVal(input_tensor_proto, 2));
  } else if (input->Datatype() == "UINT16") {
    RETURN_IF_CB_ERROR(PopulateIntVal(input_tensor_proto, 2));
  } else if (input->Datatype() == "INT8") {
    RETURN_IF_CB_ERROR(PopulateIntVal(input_tensor_proto, 1));
  } else if (input->Datatype() == "UINT8") {
    RETURN_IF_CB_ERROR(PopulateIntVal(input_tensor_proto, 1));
  } else if (input->Datatype() == "BYTES") {
    RETURN_IF_CB_ERROR(PopulateStrVal(input_tensor_proto));
  } else if (input->Datatype() == "INT64") {
    RETURN_IF_CB_ERROR(PopulateInt64Val(input_tensor_proto));
  } else if (input->Datatype() == "BOOL") {
    RETURN_IF_CB_ERROR(PopulateBoolVal(input_tensor_proto));
  } else if (input->Datatype() == "UINT32") {
    RETURN_IF_CB_ERROR(PopulateUintVal(input_tensor_proto));
  } else if (input->Datatype() == "UINT64") {
    RETURN_IF_CB_ERROR(PopulateUint64Val(input_tensor_proto));
  } else {
    return Error("unsupported datatype for populating input data");
  }

  return Error::Success;
}

Error
GrpcClient::PopulateHalfVal(tensorflow::TensorProto* input_tensor_proto)
{
  // Building FP16 one by one. Note that since protobuf has no int16 type, we'll
  // have some pointless zero padding for each value here.
  uint64_t copied_byte_size = 0;
  while (copied_byte_size < temp_buffer_.size()) {
    int32_t elem;
    memcpy(&elem, (temp_buffer_.c_str() + copied_byte_size), 2);
    input_tensor_proto->add_half_val(elem);
    copied_byte_size += 2;
  }

  return Error::Success;
}

Error
GrpcClient::PopulateFloatVal(tensorflow::TensorProto* input_tensor_proto)
{
  input_tensor_proto->mutable_float_val()->Reserve(temp_buffer_.size());
  memcpy(
      input_tensor_proto->mutable_float_val()->mutable_data(),
      temp_buffer_.c_str(), temp_buffer_.size());

  return Error::Success;
}

Error
GrpcClient::PopulateDoubleVal(tensorflow::TensorProto* input_tensor_proto)
{
  input_tensor_proto->mutable_double_val()->Reserve(temp_buffer_.size());
  memcpy(
      input_tensor_proto->mutable_double_val()->mutable_data(),
      temp_buffer_.c_str(), temp_buffer_.size());

  return Error::Success;
}

Error
GrpcClient::PopulateIntVal(
    tensorflow::TensorProto* input_tensor_proto, size_t step_size)
{
  if (step_size == 4) {
    input_tensor_proto->mutable_int_val()->Reserve(temp_buffer_.size());
    memcpy(
        input_tensor_proto->mutable_int_val()->mutable_data(),
        temp_buffer_.c_str(), temp_buffer_.size());
  } else {
    // Note that since protobuf has no int16/int8 type, we'll
    // have some pointless zero padding for each value here and
    // need to build the tensor one element at a time
    uint64_t copied_byte_size = 0;
    while (copied_byte_size < temp_buffer_.size()) {
      int32_t elem;
      memcpy(&elem, (temp_buffer_.c_str() + copied_byte_size), step_size);
      input_tensor_proto->add_int_val(elem);
      copied_byte_size += step_size;
    }
  }

  return Error::Success;
}

Error
GrpcClient::PopulateStrVal(tensorflow::TensorProto* input_tensor_proto)
{
  uint64_t copied_byte_size = 0;
  while (copied_byte_size < temp_buffer_.size()) {
    int32_t string_length = *((int*)(temp_buffer_.c_str() + copied_byte_size));
    input_tensor_proto->add_string_val(std::string(
        (temp_buffer_.c_str() + copied_byte_size + 4), string_length));
    copied_byte_size += string_length;
  }

  return Error::Success;
}

Error
GrpcClient::PopulateBoolVal(tensorflow::TensorProto* input_tensor_proto)
{
  input_tensor_proto->mutable_bool_val()->Reserve(temp_buffer_.size());
  memcpy(
      input_tensor_proto->mutable_bool_val()->mutable_data(),
      temp_buffer_.c_str(), temp_buffer_.size());

  return Error::Success;
}

Error
GrpcClient::PopulateInt64Val(tensorflow::TensorProto* input_tensor_proto)
{
  input_tensor_proto->mutable_int64_val()->Reserve(temp_buffer_.size());
  memcpy(
      input_tensor_proto->mutable_int64_val()->mutable_data(),
      temp_buffer_.c_str(), temp_buffer_.size());

  return Error::Success;
}

Error
GrpcClient::PopulateUintVal(tensorflow::TensorProto* input_tensor_proto)
{
  input_tensor_proto->mutable_uint32_val()->Reserve(temp_buffer_.size());
  memcpy(
      input_tensor_proto->mutable_uint32_val()->mutable_data(),
      temp_buffer_.c_str(), temp_buffer_.size());

  return Error::Success;
}

Error
GrpcClient::PopulateUint64Val(tensorflow::TensorProto* input_tensor_proto)
{
  input_tensor_proto->mutable_uint64_val()->Reserve(temp_buffer_.size());
  memcpy(
      input_tensor_proto->mutable_uint64_val()->mutable_data(),
      temp_buffer_.c_str(), temp_buffer_.size());

  return Error::Success;
}

GrpcClient::GrpcClient(
    const std::string& url, bool verbose, bool use_ssl,
    const SslOptions& ssl_options)
    : InferenceServerClient(verbose),
      stub_(tensorflow::serving::PredictionService::NewStub(
          GetChannel(url, use_ssl, ssl_options)))
{
}

GrpcClient::~GrpcClient()
{
  exiting_ = true;
  // Close complete queue and wait for the worker thread to return
  async_request_completion_queue_.Shutdown();

  // thread not joinable if AsyncInfer() is not called
  // (it is default constructed thread before the first AsyncInfer() call)
  if (worker_.joinable()) {
    worker_.join();
  }

  bool has_next = true;
  GrpcInferRequest* async_request;
  bool ok;
  do {
    has_next =
        async_request_completion_queue_.Next((void**)&async_request, &ok);
    if (has_next && async_request != nullptr) {
      delete async_request;
    }
  } while (has_next);
}

//======================================================================

Error
InferResult::Create(
    InferResult** infer_result,
    std::shared_ptr<tensorflow::serving::PredictResponse> response,
    Error& request_status)
{
  *infer_result =
      reinterpret_cast<InferResult*>(new InferResult(response, request_status));
  return Error::Success;
}

Error
InferResult::RequestStatus() const
{
  return request_status_;
}

InferResult::InferResult(
    std::shared_ptr<tensorflow::serving::PredictResponse> response,
    Error& request_status)
    : response_(response), request_status_(request_status)
{
}

//======================================================================

}}}  // namespace perfanalyzer::clientbackend::tfserving
