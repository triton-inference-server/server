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

#include "src/clients/c++/perf_analyzer/client_backend/tensorflow_serving/tfserve_client_backend.h"

#include "src/clients/c++/examples/json_utils.h"

namespace perfanalyzer { namespace clientbackend {

//==============================================================================

Error
TFServeClientBackend::Create(
    const std::string& url, const ProtocolType protocol,
    std::shared_ptr<Headers> http_headers, const bool verbose,
    std::unique_ptr<ClientBackend>* client_backend)
{
  if (protocol == ProtocolType::HTTP) {
    return Error(
        "perf_analyzer does not support http protocol with TF serving");
  }
  std::unique_ptr<TFServeClientBackend> tfserve_client_backend(
      new TFServeClientBackend(http_headers));

  RETURN_IF_CB_ERROR(tfs::GrpcClient::Create(
      &(tfserve_client_backend->grpc_client_), url, verbose));

  *client_backend = std::move(tfserve_client_backend);

  return Error::Success;
}

Error
TFServeClientBackend::ModelMetadata(
    rapidjson::Document* model_metadata, const std::string& model_name,
    const std::string& model_version)
{
  tensorflow::serving::GetModelMetadataResponse metadata_proto;
  RETURN_IF_CB_ERROR(grpc_client_->ModelMetadata(
      &metadata_proto, model_name, model_version, *http_headers_));

  std::string metadata;
  ::google::protobuf::util::JsonPrintOptions options;
  options.preserve_proto_field_names = true;
  options.always_print_primitive_fields = true;
  ::google::protobuf::util::MessageToJsonString(
      metadata_proto, &metadata, options);

  RETURN_IF_TRITON_ERROR(nic::ParseJson(model_metadata, metadata));

  return Error::Success;
}

Error
TFServeClientBackend::Infer(
    InferResult** result, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  tfs::InferResult* tfserve_result;
  RETURN_IF_CB_ERROR(grpc_client_->Infer(
      &tfserve_result, options, inputs, outputs, *http_headers_));

  *result = new TFServeInferResult(tfserve_result);

  return Error::Success;
}

Error
TFServeClientBackend::AsyncInfer(
    OnCompleteFn callback, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  auto wrapped_callback = [callback](tfs::InferResult* client_result) {
    InferResult* result = new TFServeInferResult(client_result);
    callback(result);
  };

  RETURN_IF_CB_ERROR(grpc_client_->AsyncInfer(
      wrapped_callback, options, inputs, outputs, *http_headers_));

  return Error::Success;
}


Error
TFServeClientBackend::ClientInferStat(InferStat* infer_stat)
{
  // Reusing the common library utilities to collect and report the
  // client side statistics.
  nic::InferStat client_infer_stat;

  RETURN_IF_TRITON_ERROR(grpc_client_->ClientInferStat(&client_infer_stat));

  ParseInferStat(client_infer_stat, infer_stat);

  return Error::Success;
}

void
TFServeClientBackend::ParseInferStat(
    const nic::InferStat& tfserve_infer_stat, InferStat* infer_stat)
{
  infer_stat->completed_request_count =
      tfserve_infer_stat.completed_request_count;
  infer_stat->cumulative_total_request_time_ns =
      tfserve_infer_stat.cumulative_total_request_time_ns;
  infer_stat->cumulative_send_time_ns =
      tfserve_infer_stat.cumulative_send_time_ns;
  infer_stat->cumulative_receive_time_ns =
      tfserve_infer_stat.cumulative_receive_time_ns;
}

//==============================================================================

Error
TFServeInferRequestedOutput::Create(
    InferRequestedOutput** infer_output, const std::string name)
{
  TFServeInferRequestedOutput* local_infer_output =
      new TFServeInferRequestedOutput();

  nic::InferRequestedOutput* tfserve_infer_output;
  RETURN_IF_TRITON_ERROR(
      nic::InferRequestedOutput::Create(&tfserve_infer_output, name));
  local_infer_output->output_.reset(tfserve_infer_output);

  *infer_output = local_infer_output;

  return Error::Success;
}

TFServeInferRequestedOutput::TFServeInferRequestedOutput()
    : InferRequestedOutput(BackendKind::TENSORFLOW_SERVING)
{
}

//==============================================================================

TFServeInferResult::TFServeInferResult(tfs::InferResult* result)
{
  result_.reset(result);
}

Error
TFServeInferResult::Id(std::string* id) const
{
  id->clear();
  return Error::Success;
}

Error
TFServeInferResult::RequestStatus() const
{
  RETURN_IF_CB_ERROR(result_->RequestStatus());
  return Error::Success;
}

//==============================================================================


}}  // namespace perfanalyzer::clientbackend
