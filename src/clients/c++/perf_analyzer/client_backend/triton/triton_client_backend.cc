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

#include "src/clients/c++/perf_analyzer/client_backend/triton/triton_client_backend.h"

#include "src/clients/c++/examples/json_utils.h"

namespace perfanalyzer { namespace clientbackend {

//==============================================================================

Error
TritonClientBackend::Create(
    const std::string& url, const ProtocolType protocol,
    std::shared_ptr<Headers> http_headers, const bool verbose,
    std::unique_ptr<ClientBackend>* client_backend)
{
  std::unique_ptr<TritonClientBackend> triton_client_backend(
      new TritonClientBackend(protocol, http_headers));
  if (protocol == ProtocolType::HTTP) {
    RETURN_IF_TRITON_ERROR(nic::InferenceServerHttpClient::Create(
        &(triton_client_backend->client_.http_client_), url, verbose));
  } else {
    RETURN_IF_TRITON_ERROR(nic::InferenceServerGrpcClient::Create(
        &(triton_client_backend->client_.grpc_client_), url, verbose));
  }

  *client_backend = std::move(triton_client_backend);

  return Error::Success;
}

Error
TritonClientBackend::ServerExtensions(std::set<std::string>* extensions)
{
  extensions->clear();
  if (protocol_ == ProtocolType::HTTP) {
    std::string server_metadata;
    FAIL_IF_TRITON_ERR(
        client_.http_client_->ServerMetadata(&server_metadata, *http_headers_),
        "unable to get server metadata");

    rapidjson::Document server_metadata_json;
    FAIL_IF_TRITON_ERR(
        nic::ParseJson(&server_metadata_json, server_metadata),
        "failed to parse server metadata");
    for (const auto& extension :
         server_metadata_json["extensions"].GetArray()) {
      extensions->insert(
          std::string(extension.GetString(), extension.GetStringLength()));
    }
  } else {
    inference::ServerMetadataResponse server_metadata;
    FAIL_IF_TRITON_ERR(
        client_.grpc_client_->ServerMetadata(&server_metadata, *http_headers_),
        "unable to get server metadata");
    for (const auto& extension : server_metadata.extensions()) {
      extensions->insert(extension);
    }
  }

  return Error::Success;
}

Error
TritonClientBackend::ModelMetadata(
    rapidjson::Document* model_metadata, const std::string& model_name,
    const std::string& model_version)
{
  if (protocol_ == ProtocolType::HTTP) {
    std::string metadata;
    RETURN_IF_TRITON_ERROR(client_.http_client_->ModelMetadata(
        &metadata, model_name, model_version, *http_headers_));
    RETURN_IF_TRITON_ERROR(nic::ParseJson(model_metadata, metadata));
  } else {
    inference::ModelMetadataResponse model_metadata_proto;
    RETURN_IF_TRITON_ERROR(client_.grpc_client_->ModelMetadata(
        &model_metadata_proto, model_name, model_version, *http_headers_));

    std::string metadata;
    ::google::protobuf::util::JsonPrintOptions options;
    options.preserve_proto_field_names = true;
    options.always_print_primitive_fields = true;
    ::google::protobuf::util::MessageToJsonString(
        model_metadata_proto, &metadata, options);

    RETURN_IF_TRITON_ERROR(nic::ParseJson(model_metadata, metadata));
  }

  return Error::Success;
}

Error
TritonClientBackend::ModelConfig(
    rapidjson::Document* model_config, const std::string& model_name,
    const std::string& model_version)
{
  if (protocol_ == ProtocolType::HTTP) {
    std::string config;
    RETURN_IF_TRITON_ERROR(client_.http_client_->ModelConfig(
        &config, model_name, model_version, *http_headers_));
    RETURN_IF_TRITON_ERROR(nic::ParseJson(model_config, config));
  } else {
    inference::ModelConfigResponse model_config_proto;
    RETURN_IF_TRITON_ERROR(client_.grpc_client_->ModelConfig(
        &model_config_proto, model_name, model_version, *http_headers_));

    std::string config;
    ::google::protobuf::util::JsonPrintOptions options;
    options.preserve_proto_field_names = true;
    options.always_print_primitive_fields = true;
    ::google::protobuf::util::MessageToJsonString(
        model_config_proto, &config, options);

    rapidjson::Document full_config;
    RETURN_IF_TRITON_ERROR(nic::ParseJson(&full_config, config));
    model_config->CopyFrom(full_config["config"], model_config->GetAllocator());
  }
  return Error::Success;
}

Error
TritonClientBackend::Infer(
    InferResult** result, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  std::vector<nic::InferInput*> triton_inputs;
  ParseInferInputToTriton(inputs, &triton_inputs);

  std::vector<const nic::InferRequestedOutput*> triton_outputs;
  ParseInferRequestedOutputToTriton(outputs, &triton_outputs);

  nic::InferOptions triton_options(options.model_name_);
  ParseInferOptionsToTriton(options, &triton_options);

  nic::InferResult* triton_result;

  if (protocol_ == ProtocolType::GRPC) {
    RETURN_IF_TRITON_ERROR(client_.grpc_client_->Infer(
        &triton_result, triton_options, triton_inputs, triton_outputs,
        *http_headers_));
  } else {
    RETURN_IF_TRITON_ERROR(client_.http_client_->Infer(
        &triton_result, triton_options, triton_inputs, triton_outputs,
        *http_headers_));
  }

  *result = new TritonInferResult(triton_result);

  return Error::Success;
}

Error
TritonClientBackend::AsyncInfer(
    OnCompleteFn callback, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  auto wrapped_callback = [callback](nic::InferResult* client_result) {
    InferResult* result = new TritonInferResult(client_result);
    callback(result);
  };

  std::vector<nic::InferInput*> triton_inputs;
  ParseInferInputToTriton(inputs, &triton_inputs);

  std::vector<const nic::InferRequestedOutput*> triton_outputs;
  ParseInferRequestedOutputToTriton(outputs, &triton_outputs);

  nic::InferOptions triton_options(options.model_name_);
  ParseInferOptionsToTriton(options, &triton_options);

  if (protocol_ == ProtocolType::GRPC) {
    RETURN_IF_TRITON_ERROR(client_.grpc_client_->AsyncInfer(
        wrapped_callback, triton_options, triton_inputs, triton_outputs,
        *http_headers_));
  } else {
    RETURN_IF_TRITON_ERROR(client_.http_client_->AsyncInfer(
        wrapped_callback, triton_options, triton_inputs, triton_outputs,
        *http_headers_));
  }

  return Error::Success;
}

Error
TritonClientBackend::StartStream(OnCompleteFn callback, bool enable_stats)
{
  auto wrapped_callback = [callback](nic::InferResult* client_result) {
    InferResult* result = new TritonInferResult(client_result);
    callback(result);
  };

  if (protocol_ == ProtocolType::GRPC) {
    RETURN_IF_TRITON_ERROR(client_.grpc_client_->StartStream(
        wrapped_callback, enable_stats, 0 /* stream_timeout */,
        *http_headers_));
  } else {
    return Error("HTTP does not support starting streams");
  }

  return Error::Success;
}

Error
TritonClientBackend::AsyncStreamInfer(
    const InferOptions& options, const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  std::vector<nic::InferInput*> triton_inputs;
  ParseInferInputToTriton(inputs, &triton_inputs);

  std::vector<const nic::InferRequestedOutput*> triton_outputs;
  ParseInferRequestedOutputToTriton(outputs, &triton_outputs);

  nic::InferOptions triton_options(options.model_name_);
  ParseInferOptionsToTriton(options, &triton_options);

  if (protocol_ == ProtocolType::GRPC) {
    RETURN_IF_TRITON_ERROR(client_.grpc_client_->AsyncStreamInfer(
        triton_options, triton_inputs, triton_outputs));
  } else {
    return Error("HTTP does not support streaming inferences");
  }

  return Error::Success;
}

Error
TritonClientBackend::ClientInferStat(InferStat* infer_stat)
{
  nic::InferStat triton_infer_stat;
  if (protocol_ == ProtocolType::GRPC) {
    RETURN_IF_TRITON_ERROR(
        client_.grpc_client_->ClientInferStat(&triton_infer_stat));
  } else {
    RETURN_IF_TRITON_ERROR(
        client_.http_client_->ClientInferStat(&triton_infer_stat));
  }

  ParseInferStat(triton_infer_stat, infer_stat);

  return Error::Success;
}

Error
TritonClientBackend::ModelInferenceStatistics(
    std::map<ModelIdentifier, ModelStatistics>* model_stats,
    const std::string& model_name, const std::string& model_version)
{
  if (protocol_ == ProtocolType::GRPC) {
    inference::ModelStatisticsResponse infer_stat;
    RETURN_IF_TRITON_ERROR(client_.grpc_client_->ModelInferenceStatistics(
        &infer_stat, model_name, model_version, *http_headers_));
    ParseStatistics(infer_stat, model_stats);
  } else {
    std::string infer_stat;
    RETURN_IF_TRITON_ERROR(client_.http_client_->ModelInferenceStatistics(
        &infer_stat, model_name, model_version, *http_headers_));
    rapidjson::Document infer_stat_json;
    RETURN_IF_TRITON_ERROR(nic::ParseJson(&infer_stat_json, infer_stat));
    ParseStatistics(infer_stat_json, model_stats);
  }

  return Error::Success;
}

Error
TritonClientBackend::UnregisterAllSharedMemory()
{
  if (protocol_ == ProtocolType::GRPC) {
    RETURN_IF_TRITON_ERROR(
        client_.grpc_client_->UnregisterSystemSharedMemory("", *http_headers_));
    RETURN_IF_TRITON_ERROR(
        client_.grpc_client_->UnregisterCudaSharedMemory("", *http_headers_));
  } else {
    RETURN_IF_TRITON_ERROR(
        client_.http_client_->UnregisterSystemSharedMemory("", *http_headers_));
    RETURN_IF_TRITON_ERROR(
        client_.http_client_->UnregisterCudaSharedMemory("", *http_headers_));
  }

  return Error::Success;
}

Error
TritonClientBackend::RegisterSystemSharedMemory(
    const std::string& name, const std::string& key, const size_t byte_size)
{
  if (protocol_ == ProtocolType::GRPC) {
    RETURN_IF_TRITON_ERROR(client_.grpc_client_->RegisterSystemSharedMemory(
        name, key, byte_size, 0 /* offset */, *http_headers_));

  } else {
    RETURN_IF_TRITON_ERROR(client_.http_client_->RegisterSystemSharedMemory(
        name, key, byte_size, 0 /* offset */, *http_headers_));
  }

  return Error::Success;
}

Error
TritonClientBackend::RegisterCudaSharedMemory(
    const std::string& name, const cudaIpcMemHandle_t& handle,
    const size_t byte_size)
{
  if (protocol_ == ProtocolType::GRPC) {
    RETURN_IF_TRITON_ERROR(client_.grpc_client_->RegisterCudaSharedMemory(
        name, handle, 0 /*device id*/, byte_size, *http_headers_));

  } else {
    RETURN_IF_TRITON_ERROR(client_.http_client_->RegisterCudaSharedMemory(
        name, handle, 0 /*device id*/, byte_size, *http_headers_));
  }

  return Error::Success;
}

//
// Shared Memory Utilities
//
Error
TritonClientBackend::CreateSharedMemoryRegion(
    std::string shm_key, size_t byte_size, int* shm_fd)
{
  RETURN_IF_TRITON_ERROR(
      nic::CreateSharedMemoryRegion(shm_key, byte_size, shm_fd));

  return Error::Success;
}


Error
TritonClientBackend::MapSharedMemory(
    int shm_fd, size_t offset, size_t byte_size, void** shm_addr)
{
  RETURN_IF_TRITON_ERROR(
      nic::MapSharedMemory(shm_fd, offset, byte_size, shm_addr));

  return Error::Success;
}


Error
TritonClientBackend::CloseSharedMemory(int shm_fd)
{
  RETURN_IF_TRITON_ERROR(nic::CloseSharedMemory(shm_fd));

  return Error::Success;
}

Error
TritonClientBackend::UnlinkSharedMemoryRegion(std::string shm_key)
{
  RETURN_IF_TRITON_ERROR(nic::UnlinkSharedMemoryRegion(shm_key));

  return Error::Success;
}

Error
TritonClientBackend::UnmapSharedMemory(void* shm_addr, size_t byte_size)
{
  RETURN_IF_TRITON_ERROR(nic::UnmapSharedMemory(shm_addr, byte_size));

  return Error::Success;
}

void
TritonClientBackend::ParseInferInputToTriton(
    const std::vector<InferInput*>& inputs,
    std::vector<nic::InferInput*>* triton_inputs)
{
  for (const auto input : inputs) {
    triton_inputs->push_back((dynamic_cast<TritonInferInput*>(input))->Get());
  }
}

void
TritonClientBackend::ParseInferRequestedOutputToTriton(
    const std::vector<const InferRequestedOutput*>& outputs,
    std::vector<const nic::InferRequestedOutput*>* triton_outputs)
{
  for (const auto output : outputs) {
    triton_outputs->push_back(
        (dynamic_cast<const TritonInferRequestedOutput*>(output))->Get());
  }
}

void
TritonClientBackend::ParseInferOptionsToTriton(
    const InferOptions& options, nic::InferOptions* triton_options)
{
  triton_options->model_name_ = options.model_name_;
  triton_options->model_version_ = options.model_version_;
  triton_options->request_id_ = options.request_id_;
  triton_options->sequence_id_ = options.sequence_id_;
  triton_options->sequence_start_ = options.sequence_start_;
  triton_options->sequence_end_ = options.sequence_end_;
}


void
TritonClientBackend::ParseStatistics(
    const inference::ModelStatisticsResponse& infer_stat,
    std::map<ModelIdentifier, ModelStatistics>* model_stats)
{
  model_stats->clear();
  for (const auto& this_stat : infer_stat.model_stats()) {
    auto it = model_stats
                  ->emplace(
                      std::make_pair(this_stat.name(), this_stat.version()),
                      ModelStatistics())
                  .first;
    it->second.inference_count_ = this_stat.inference_count();
    it->second.execution_count_ = this_stat.execution_count();
    it->second.success_count_ = this_stat.inference_stats().success().count();
    it->second.cumm_time_ns_ = this_stat.inference_stats().success().ns();
    it->second.queue_time_ns_ = this_stat.inference_stats().queue().ns();
    it->second.compute_input_time_ns_ =
        this_stat.inference_stats().compute_input().ns();
    it->second.compute_infer_time_ns_ =
        this_stat.inference_stats().compute_infer().ns();
    it->second.compute_output_time_ns_ =
        this_stat.inference_stats().compute_output().ns();
  }
}

void
TritonClientBackend::ParseStatistics(
    const rapidjson::Document& infer_stat,
    std::map<ModelIdentifier, ModelStatistics>* model_stats)
{
  model_stats->clear();
  for (const auto& this_stat : infer_stat["model_stats"].GetArray()) {
    auto it = model_stats
                  ->emplace(
                      std::make_pair(
                          this_stat["name"].GetString(),
                          this_stat["version"].GetString()),
                      ModelStatistics())
                  .first;
    it->second.inference_count_ = this_stat["inference_count"].GetUint64();
    it->second.execution_count_ = this_stat["execution_count"].GetUint64();
    it->second.success_count_ =
        this_stat["inference_stats"]["success"]["count"].GetUint64();
    it->second.cumm_time_ns_ =
        this_stat["inference_stats"]["success"]["ns"].GetUint64();
    it->second.queue_time_ns_ =
        this_stat["inference_stats"]["queue"]["ns"].GetUint64();
    it->second.compute_input_time_ns_ =
        this_stat["inference_stats"]["compute_input"]["ns"].GetUint64();
    it->second.compute_infer_time_ns_ =
        this_stat["inference_stats"]["compute_infer"]["ns"].GetUint64();
    it->second.compute_output_time_ns_ =
        this_stat["inference_stats"]["compute_output"]["ns"].GetUint64();
  }
}

void
TritonClientBackend::ParseInferStat(
    const nic::InferStat& triton_infer_stat, InferStat* infer_stat)
{
  infer_stat->completed_request_count =
      triton_infer_stat.completed_request_count;
  infer_stat->cumulative_total_request_time_ns =
      triton_infer_stat.cumulative_total_request_time_ns;
  infer_stat->cumulative_send_time_ns =
      triton_infer_stat.cumulative_send_time_ns;
  infer_stat->cumulative_receive_time_ns =
      triton_infer_stat.cumulative_receive_time_ns;
}

//==============================================================================

Error
TritonInferInput::Create(
    InferInput** infer_input, const std::string& name,
    const std::vector<int64_t>& dims, const std::string& datatype)
{
  TritonInferInput* local_infer_input = new TritonInferInput(name, datatype);

  nic::InferInput* triton_infer_input;
  RETURN_IF_TRITON_ERROR(
      nic::InferInput::Create(&triton_infer_input, name, dims, datatype));
  local_infer_input->input_.reset(triton_infer_input);

  *infer_input = local_infer_input;
  return Error::Success;
}

const std::vector<int64_t>&
TritonInferInput::Shape() const
{
  return input_->Shape();
}

Error
TritonInferInput::SetShape(const std::vector<int64_t>& shape)
{
  RETURN_IF_TRITON_ERROR(input_->SetShape(shape));
  return Error::Success;
}

Error
TritonInferInput::Reset()
{
  RETURN_IF_TRITON_ERROR(input_->Reset());
  return Error::Success;
}

Error
TritonInferInput::AppendRaw(const uint8_t* input, size_t input_byte_size)
{
  RETURN_IF_TRITON_ERROR(input_->AppendRaw(input, input_byte_size));
  return Error::Success;
}

Error
TritonInferInput::SetSharedMemory(
    const std::string& name, size_t byte_size, size_t offset)
{
  RETURN_IF_TRITON_ERROR(input_->SetSharedMemory(name, byte_size, offset));
  return Error::Success;
}

TritonInferInput::TritonInferInput(
    const std::string& name, const std::string& datatype)
    : InferInput(BackendKind::TRITON, name, datatype)
{
}


//==============================================================================

Error
TritonInferRequestedOutput::Create(
    InferRequestedOutput** infer_output, const std::string name,
    const size_t class_count)
{
  TritonInferRequestedOutput* local_infer_output =
      new TritonInferRequestedOutput();

  nic::InferRequestedOutput* triton_infer_output;
  RETURN_IF_TRITON_ERROR(nic::InferRequestedOutput::Create(
      &triton_infer_output, name, class_count));
  local_infer_output->output_.reset(triton_infer_output);

  *infer_output = local_infer_output;

  return Error::Success;
}

Error
TritonInferRequestedOutput::SetSharedMemory(
    const std::string& region_name, const size_t byte_size, const size_t offset)
{
  RETURN_IF_TRITON_ERROR(
      output_->SetSharedMemory(region_name, byte_size, offset));
  return Error::Success;
}


TritonInferRequestedOutput::TritonInferRequestedOutput()
    : InferRequestedOutput(BackendKind::TRITON)
{
}

//==============================================================================

TritonInferResult::TritonInferResult(nic::InferResult* result)
{
  result_.reset(result);
}

Error
TritonInferResult::Id(std::string* id) const
{
  RETURN_IF_TRITON_ERROR(result_->Id(id));
  return Error::Success;
}

Error
TritonInferResult::RequestStatus() const
{
  RETURN_IF_TRITON_ERROR(result_->RequestStatus());
  return Error::Success;
}

//==============================================================================

}}  // namespace perfanalyzer::clientbackend
