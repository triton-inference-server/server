// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
SavedModelBackend::CreateTRTISTFModel(
    const GraphDefBackendFactory::Config* backend_config, const int device_id,
    const bool has_graph_level, const int graph_level,
    const std::string& model_path, TRTISTFModelHandle* trtistf_model,
    IONameMap* input_name_map, IONameMap* output_name_map,
    const TRTISTF_TFTRTConfig* tftrt_config)
{
  TRTISTF_Model* model = nullptr;
  RETURN_IF_TRTISTF_ERROR(TRTISTF_ModelCreateFromSavedModel(
      &model, model_path.c_str(), model_path.c_str(), device_id,
      has_graph_level, graph_level, backend_config->allow_gpu_memory_growth,
      backend_config->per_process_gpu_memory_fraction,
      backend_config->allow_soft_placement, backend_config->memory_limit_mb,
      tftrt_config));

  trtistf_model->reset(model);

  // The model inputs are the expected inputs and the outputs are
  // the allowed outputs. Saved-model gives these explicitly so we can
  // check precisely if the model configuration matches.
  const TRTISTF_IOList* inputs = TRTISTF_ModelInputs(model);
  const TRTISTF_IOList* outputs = TRTISTF_ModelOutputs(model);

  std::set<std::string> expected_inputs, allowed_outputs;
  for (const TRTISTF_IOList* itr = inputs; itr != nullptr; itr = itr->next_) {
    expected_inputs.insert(itr->io_->name_);
    input_name_map->insert({itr->io_->name_, itr->io_->inmodel_name_});
  }
  for (const TRTISTF_IOList* itr = outputs; itr != nullptr; itr = itr->next_) {
    allowed_outputs.insert(itr->io_->name_);
    output_name_map->insert({itr->io_->name_, itr->io_->inmodel_name_});
  }

  size_t expected_input_cnt = (size_t)Config().input().size();

  // If this is a sequence model then make sure that the required
  // inputs are present in the model and have the correct shape and
  // datatype.
  if (Config().has_sequence_batching()) {
    bool have_start, have_end, have_ready, have_corrid;
    RETURN_IF_ERROR(ValidateBooleanSequenceControl(
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_START, inputs,
        false /* required */, &have_start));
    RETURN_IF_ERROR(ValidateBooleanSequenceControl(
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_END, inputs,
        false /* required */, &have_end));
    RETURN_IF_ERROR(ValidateBooleanSequenceControl(
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_READY, inputs,
        false /* required */, &have_ready));
    RETURN_IF_ERROR(ValidateTypedSequenceControl(
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_CORRID, inputs,
        false /* required */, &have_corrid));
    if (have_start) {
      expected_input_cnt += 1;
    }
    if (have_end) {
      expected_input_cnt += 1;
    }
    if (have_ready) {
      expected_input_cnt += 1;
    }
    if (have_corrid) {
      expected_input_cnt += 1;
    }
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

    const TRTISTF_IO* input = FindIOByName(inputs, io.name());
    if (input == nullptr) {
      return Status(
          RequestStatusCode::INTERNAL,
          "unexpected inference input '" + io.name() + "'");
    }

    // If a reshape is provided for the input then use that when
    // validating that the TF model matches what is expected.
    const DimsList& dims =
        (io.has_reshape()) ? io.reshape().shape() : io.dims();

    if (input->shape_->rank_ != 0) {
      RETURN_IF_ERROR(CompareDims(
          Name(), io.name(), input->shape_, dims, Config().max_batch_size() > 0,
          false /* compare_exact */));
    } else {
      // The savedmodel doesn't specify a shape for the input so use the shape
      // from the model configuration
      bool supports_batching = Config().max_batch_size() > 0;
      input->shape_->rank_ =
          (size_t)(dims.size() + (supports_batching ? 1 : 0));
      input->shape_->dims_ =
          (int64_t*)malloc(input->shape_->rank_ * sizeof(int64_t));
      for (int i = 0; i < dims.size(); ++i) {
        input->shape_->dims_[i + (supports_batching ? 1 : 0)] = dims[i];
      }
    }

    if (!CompareDataType(input->data_type_, io.data_type())) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "unable to load model '" + Name() + "', input '" + io.name() +
              "' data-type " +
              DataType_Name(ConvertDataType(input->data_type_)) +
              " doesn't match configuration data-type " +
              DataType_Name(io.data_type()));
    }
  }

  for (const auto& io : Config().output()) {
    RETURN_IF_ERROR(CheckAllowedModelOutput(io, allowed_outputs));

    const TRTISTF_IO* output = FindIOByName(outputs, io.name());
    if (output == nullptr) {
      return Status(
          RequestStatusCode::INTERNAL,
          "unexpected inference output '" + io.name() + "'");
    }

    // If a reshape is provided for the output then use that when
    // validating that the TF model matches what is expected.
    const DimsList& dims =
        (io.has_reshape()) ? io.reshape().shape() : io.dims();

    if (output->shape_->rank_ != 0) {
      RETURN_IF_ERROR(CompareDims(
          Name(), io.name(), output->shape_, dims,
          Config().max_batch_size() > 0, false /* compare_exact */));
    } else {
      // The savedmodel doesn't specify a shape for the output so use the shape
      // from the model configuration
      bool supports_batching = Config().max_batch_size() > 0;
      output->shape_->rank_ =
          (size_t)(dims.size() + (supports_batching ? 1 : 0));
      output->shape_->dims_ =
          (int64_t*)malloc(output->shape_->rank_ * sizeof(int64_t));
      for (int i = 0; i < dims.size(); ++i) {
        output->shape_->dims_[i + (supports_batching ? 1 : 0)] = dims[i];
      }
    }

    if (!CompareDataType(output->data_type_, io.data_type())) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "unable to load model '" + Name() + "', output '" + io.name() +
              "' data-type " +
              DataType_Name(ConvertDataType(output->data_type_)) +
              " doesn't match configuration data-type " +
              DataType_Name(io.data_type()));
    }
  }

  return Status::Success;
}

Status
SavedModelBackend::ValidateBooleanSequenceControl(
    const ModelSequenceBatching::Control::Kind control_kind,
    const TRTISTF_IOList* inputs, bool required, bool* have_control)
{
  std::string tensor_name;
  DataType tensor_datatype;
  RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
      Config().sequence_batching(), Name(), control_kind, required,
      &tensor_name, &tensor_datatype, nullptr, nullptr, nullptr, nullptr));
  *have_control = !tensor_name.empty();
  if (*have_control) {
    const TRTISTF_IO* input = FindIOByName(inputs, tensor_name);
    if (input == nullptr) {
      return Status(
          RequestStatusCode::INTERNAL,
          "configuration specified sequence control '" + tensor_name +
              "', but model does not provide that input");
    }

    // Control tensors must have shape [1].
    DimsList dims;
    dims.Add(1);

    Status compare_status = CompareDims(
        Name(), tensor_name, input->shape_, dims, Config().max_batch_size() > 0,
        true /* compare_exact */);
    if (!compare_status.IsOk()) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "unable to load model '" + Name() + "', sequence control '" +
              tensor_name + "': " + compare_status.Message());
    }

    if (!CompareDataType(input->data_type_, tensor_datatype)) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "unable to load model '" + Name() + "', sequence control '" +
              tensor_name + "': the model expects data-type " +
              DataType_Name(ConvertDataType(input->data_type_)) +
              " but the model configuration specifies data-type " +
              DataType_Name(tensor_datatype));
    }
  }

  return Status::Success;
}

Status
SavedModelBackend::ValidateTypedSequenceControl(
    const ModelSequenceBatching::Control::Kind control_kind,
    const TRTISTF_IOList* inputs, bool required, bool* have_control)
{
  std::string tensor_name;
  DataType tensor_datatype;
  RETURN_IF_ERROR(GetTypedSequenceControlProperties(
      Config().sequence_batching(), Name(), control_kind, required,
      &tensor_name, &tensor_datatype));
  *have_control = !tensor_name.empty();
  if (*have_control) {
    const TRTISTF_IO* input = FindIOByName(inputs, tensor_name);
    if (input == nullptr) {
      return Status(
          RequestStatusCode::INTERNAL,
          "configuration specified sequence control '" + tensor_name +
              "', but model does not provide that input");
    }

    // Control tensors must have shape [1].
    DimsList dims;
    dims.Add(1);

    Status compare_status = CompareDims(
        Name(), tensor_name, input->shape_, dims, Config().max_batch_size() > 0,
        true /* compare_exact */);
    if (!compare_status.IsOk()) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "unable to load model '" + Name() + "', sequence control '" +
              tensor_name + "': " + compare_status.Message());
    }

    if (!CompareDataType(input->data_type_, tensor_datatype)) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "unable to load model '" + Name() + "', sequence control '" +
              tensor_name + "', the model expects data-type " +
              DataType_Name(ConvertDataType(input->data_type_)) +
              " but the model configuration specifies data-type " +
              DataType_Name(tensor_datatype));
    }
  }

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
