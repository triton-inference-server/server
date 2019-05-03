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
#include "src/backends/tensorflow/loader.h"
#include "src/backends/tensorflow/tf_utils.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/model_config_utils.h"
#include "tensorflow/c/c_api.h"

namespace nvidia { namespace inferenceserver {

Status
SavedModelBackend::Init(const std::string& path, const ModelConfig& config)
{
  RETURN_IF_ERROR(ValidateModelConfig(config, kTensorFlowSavedModelPlatform));
  RETURN_IF_ERROR(BaseBackend::Init(path, config));

  return Status::Success;
}

Status
SavedModelBackend::CreateSession(
    const tensorflow::SessionOptions& options, const int gpu_device,
    const std::string& model_path, tensorflow::Session** session,
    IONameMap* input_name_map, IONameMap* output_name_map)
{
  // Set the default device to control the CPU/GPU that the graph runs
  // on. This isn't foolproof since individual operations in the graph
  // could specify a specific run location. But given that
  // visible_device_list doesn't work it seems like the only option we
  // have. [DLIS-43]
  //
  // The GraphDef where we need to use this workaround is only
  // available in tensorflow/cc/saved_model/loader.cc so we use
  // visible_device_list in pass in the gpu_device we want and then
  // our (modified) loader.cc will use that to SetDefaultDevice
  // appropriately.
  tensorflow::SessionOptions session_options = options;
  if (gpu_device == Context::NO_GPU_DEVICE) {
    session_options.config.mutable_gpu_options()->set_visible_device_list(
        "/cpu:0");
  } else {
    session_options.config.mutable_gpu_options()->set_visible_device_list(
        "/gpu:" + std::to_string(gpu_device));
  }

  std::unique_ptr<tensorflow::SavedModelBundle> bundle;
  tensorflow::SignatureDef sig;
  RETURN_IF_ERROR(
      LoadSavedModel(Name(), model_path, session_options, &bundle, &sig));

  // Collect all the expected input and allowed output tensor names
  // based on the signature def.
  std::set<std::string> expected_inputs, allowed_outputs;
  for (const auto& i : sig.inputs()) {
    expected_inputs.emplace(i.first);
    input_name_map->insert({i.first, i.second.name()});
  }
  for (const auto& o : sig.outputs()) {
    allowed_outputs.emplace(o.first);
    output_name_map->insert({o.first, o.second.name()});
  }

  size_t expected_input_cnt = (size_t)Config().input().size();

  // If this is a sequence model then make sure that the required
  // inputs are present in the model and have the correct shape and
  // datatype.
  if (Config().has_sequence_batching()) {
    RETURN_IF_ERROR(ValidateSequenceControl(
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_START, sig));
    RETURN_IF_ERROR(ValidateSequenceControl(
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_READY, sig));
    expected_input_cnt += 2;
  }

  // Verify that the model configuration input and outputs match what
  // is expected by the signature def.
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

    const auto& iitr = sig.inputs().find(io.name());
    if (iitr == sig.inputs().end()) {
      return Status(
          RequestStatusCode::INTERNAL,
          "unexpected inference input '" + io.name() + "'");
    }

    // If a reshape is provided for the input then use that when
    // validating that the TF model matches what is expected.
    const DimsList& dims =
        (io.has_reshape()) ? io.reshape().shape() : io.dims();

    RETURN_IF_ERROR(CompareDimsSupported(
        Name(), io.name(), iitr->second.tensor_shape(), dims,
        Config().max_batch_size() > 0));

    if (!CompareDataType(iitr->second.dtype(), io.data_type())) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "unable to load model '" + Name() + "', input '" + io.name() +
              "' data-type " + tensorflow::DataType_Name(iitr->second.dtype()) +
              " doesn't match configuration data-type " +
              DataType_Name(io.data_type()));
    }
  }

  for (const auto& io : Config().output()) {
    RETURN_IF_ERROR(CheckAllowedModelOutput(io, allowed_outputs));

    const auto& oitr = sig.outputs().find(io.name());
    if (oitr == sig.outputs().end()) {
      return Status(
          RequestStatusCode::INTERNAL,
          "unexpected inference output '" + io.name() + "'");
    }

    // If a reshape is provided for the output then use that when
    // validating that the TF model matches what is expected.
    const DimsList& dims =
        (io.has_reshape()) ? io.reshape().shape() : io.dims();

    RETURN_IF_ERROR(CompareDimsSupported(
        Name(), io.name(), oitr->second.tensor_shape(), dims,
        Config().max_batch_size() > 0));

    if (!CompareDataType(oitr->second.dtype(), io.data_type())) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "unable to load model '" + Name() + "', output '" + io.name() +
              "' data-type " + tensorflow::DataType_Name(oitr->second.dtype()) +
              " doesn't match configuration data-type " +
              DataType_Name(io.data_type()));
    }
  }

  *session = bundle->session.release();

  return Status::Success;
}

Status
SavedModelBackend::ValidateSequenceControl(
    const ModelSequenceBatching::Control::Kind control_kind,
    const tensorflow::SignatureDef& sig)
{
  std::string tensor_name;
  DataType tensor_datatype;
  RETURN_IF_ERROR(GetSequenceControlProperties(
      Config().sequence_batching(), Name(), control_kind, true /* required */,
      &tensor_name, &tensor_datatype, nullptr, nullptr, nullptr, nullptr));

  const auto& iitr = sig.inputs().find(tensor_name);
  if (iitr == sig.inputs().end()) {
    return Status(
        RequestStatusCode::INTERNAL,
        "configuration specified sequence control '" + tensor_name +
            "', but model does not provide that input");
  }

  // Control tensors must have shape [1].
  DimsList dims;
  dims.Add(1);

  if (!CompareDimsExact(
          iitr->second.tensor_shape(), dims, Config().max_batch_size() > 0)) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "unable to load model '" + Name() + "', sequence control '" +
            tensor_name + "' dims " +
            DimsDebugString(iitr->second.tensor_shape()) +
            " don't match expected dims [1]");
  }

  if (!CompareDataType(iitr->second.dtype(), tensor_datatype)) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "unable to load model '" + Name() + "', sequence control '" +
            tensor_name + "' data-type " +
            tensorflow::DataType_Name(iitr->second.dtype()) +
            " doesn't match required data-type " +
            DataType_Name(tensor_datatype));
  }

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
