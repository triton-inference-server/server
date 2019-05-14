// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "src/backends/tensorflow/savedmodel_backend.h"

#include <set>
#include "src/backends/tensorflow/graphdef_backend_factory.h"
#include "src/backends/tensorflow/tensorflow_backend_tf.h"
#include "src/backends/tensorflow/tf_utils.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/model_config_utils.h"

namespace nvidia { namespace inferenceserver {

Status
SavedModelBackend::Init(const std::string& path, const ModelConfig& config)
{
  RETURN_IF_ERROR(ValidateModelConfig(config, kTensorFlowSavedModelPlatform));
  RETURN_IF_ERROR(BaseBackend::Init(path, config));

  return Status::Success;
}

Status
SavedModelBackend::CreateWorkspace(
    const std::shared_ptr<GraphDefBackendFactory::Config>& backend_config,
    const int gpu_device, const bool has_graph_level, const int graph_level,
    const std::string& model_path, std::unique_ptr<TFWorkspace>* workspace,
    IONameMap* input_name_map, IONameMap* output_name_map)
{
  TFWorkspace* tfws = nullptr;
  TFWorkspace::Error error = TFWorkspaceCreateFromSavedModel(
      &tfws, model_path, model_path, gpu_device, has_graph_level, graph_level,
      backend_config->allow_gpu_memory_growth,
      backend_config->per_process_gpu_memory_fraction,
      backend_config->allow_soft_placement);
  if (!error.IsOk()) {
    return Status(RequestStatusCode::INTERNAL, error.Message());
  }

  // The workspace inputs are the expected inputs and the outputs are
  // the allowed outputs. Saved-model gives these explicitly so we can
  // check precisely if the model configuration matches.
  const TFWorkspace::IOList& tfwsinputs = tfws->Inputs();
  const TFWorkspace::IOList& tfwsoutputs = tfws->Outputs();

  std::set<std::string> expected_inputs, allowed_outputs;
  for (auto& io : tfwsinputs) {
    expected_inputs.insert(io.name_);
    input_name_map->insert({io.name_, io.inmodel_name_});
  }
  for (const auto& io : tfwsoutputs) {
    allowed_outputs.insert(io.name_);
    output_name_map->insert({io.name_, io.inmodel_name_});
  }

  size_t expected_input_cnt = (size_t)Config().input().size();

  // If this is a sequence model then make sure that the required
  // inputs are present in the model and have the correct shape and
  // datatype.
  if (Config().has_sequence_batching()) {
    RETURN_IF_ERROR(ValidateSequenceControl(
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_START, tfwsinputs));
    RETURN_IF_ERROR(ValidateSequenceControl(
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_READY, tfwsinputs));
    expected_input_cnt += 2;
  }

  // Verify that the model configuration input and outputs match what
  // is expected by the model.
  if (expected_inputs.size() != expected_input_cnt) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "unable to load model '" + Name() + "', configuration expects " +
            std::to_string(Config().input().size()) +
            " inputs, model provides " +
            std::to_string(expected_inputs.size()));
  }

  for (const auto& io : Config().input()) {
    RETURN_IF_ERROR(CheckAllowedModelInput(io, expected_inputs));

    const std::string& find_name = io.name();
    const auto& iitr = std::find_if(
        tfwsinputs.begin(), tfwsinputs.end(),
        [&find_name](const TFWorkspace::IO& io) {
          return io.name_ == find_name;
        });
    if (iitr == tfwsinputs.end()) {
      return Status(
          RequestStatusCode::INTERNAL,
          "unexpected inference input '" + io.name() + "'");
    }

    // If a reshape is provided for the input then use that when
    // validating that the TF model matches what is expected.
    const DimsList& dims =
        (io.has_reshape()) ? io.reshape().shape() : io.dims();

    RETURN_IF_ERROR(CompareDimsSupported(
        Name(), io.name(), iitr->shape_, dims, Config().max_batch_size() > 0));

    if (!CompareDataType(iitr->data_type_, io.data_type())) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "unable to load model '" + Name() + "', input '" + io.name() +
              "' data-type " +
              DataType_Name(ConvertDataType(iitr->data_type_)) +
              " doesn't match configuration data-type " +
              DataType_Name(io.data_type()));
    }
  }

  for (const auto& io : Config().output()) {
    RETURN_IF_ERROR(CheckAllowedModelOutput(io, allowed_outputs));

    const std::string& find_name = io.name();
    const auto& oitr = std::find_if(
        tfwsoutputs.begin(), tfwsoutputs.end(),
        [&find_name](const TFWorkspace::IO& io) {
          return io.name_ == find_name;
        });
    if (oitr == tfwsoutputs.end()) {
      return Status(
          RequestStatusCode::INTERNAL,
          "unexpected inference output '" + io.name() + "'");
    }

    // If a reshape is provided for the output then use that when
    // validating that the TF model matches what is expected.
    const DimsList& dims =
        (io.has_reshape()) ? io.reshape().shape() : io.dims();

    RETURN_IF_ERROR(CompareDimsSupported(
        Name(), io.name(), oitr->shape_, dims, Config().max_batch_size() > 0));

    if (!CompareDataType(oitr->data_type_, io.data_type())) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "unable to load model '" + Name() + "', output '" + io.name() +
              "' data-type " +
              DataType_Name(ConvertDataType(oitr->data_type_)) +
              " doesn't match configuration data-type " +
              DataType_Name(io.data_type()));
    }
  }

  return Status::Success;
}

Status
SavedModelBackend::ValidateSequenceControl(
    const ModelSequenceBatching::Control::Kind control_kind,
    const TFWorkspace::IOList& inputs)
{
  std::string tensor_name;
  DataType tensor_datatype;
  RETURN_IF_ERROR(GetSequenceControlProperties(
      Config().sequence_batching(), Name(), control_kind, true /* required */,
      &tensor_name, &tensor_datatype, nullptr, nullptr, nullptr, nullptr));

  const auto& iitr = std::find_if(
      inputs.begin(), inputs.end(), [&tensor_name](const TFWorkspace::IO& io) {
        return io.name_ == tensor_name;
      });
  if (iitr == inputs.end()) {
    return Status(
        RequestStatusCode::INTERNAL,
        "configuration specified sequence control '" + tensor_name +
            "', but model does not provide that input");
  }

  // Control tensors must have shape [1].
  DimsList dims;
  dims.Add(1);

  if (!CompareDimsExact(iitr->shape_, dims, Config().max_batch_size() > 0)) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "unable to load model '" + Name() + "', sequence control '" +
            tensor_name + "' dims " + DimsListToString(iitr->shape_) +
            " don't match expected dims [1]");
  }

  if (!CompareDataType(iitr->data_type_, tensor_datatype)) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "unable to load model '" + Name() + "', sequence control '" +
            tensor_name + "' data-type " +
            DataType_Name(ConvertDataType(iitr->data_type_)) +
            " doesn't match required data-type " +
            DataType_Name(tensor_datatype));
  }

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
