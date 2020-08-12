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

#include "src/backends/backend/examples/backend_input_collector.h"
#include "src/backends/backend/examples/backend_model.h"
#include "src/backends/backend/examples/backend_model_instance.h"
#include "src/backends/backend/examples/backend_output_responder.h"
#include "src/backends/backend/examples/backend_utils.h"
#include "src/backends/backend/tensorflow/tf_utils.h"
#include "src/backends/tensorflow/tensorflow_backend_tf.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <set>
#include <thread>
#include <unordered_map>

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace ni = nvidia::inferenceserver;
namespace nib = nvidia::inferenceserver::backend;

//
// TF Backend that implements the TRITONBACKEND API.
//

namespace {

#define THROW_IF_BACKEND_MODEL_ERROR(X)            \
  do {                                             \
    TRITONSERVER_Error* tie_err__ = (X);           \
    if (tie_err__ != nullptr) {                    \
      throw nib::BackendModelException(tie_err__); \
    }                                              \
  } while (false)


#ifndef TRITON_ENABLE_GPU
using cudaStream_t = void*;
#endif  // !TRITON_ENABLE_GPU

using IONameMap = std::unordered_map<std::string, std::string>;
using TRTISTFModelHandle =
    std::unique_ptr<TRTISTF_Model, decltype(&TRTISTF_ModelDelete)>;

TRITONSERVER_Error*
ParseLongLongParameter(
    const std::string& key, const std::string& value, int64_t* parsed_value)
{
  try {
    *parsed_value = std::stoll(value);
  }
  catch (const std::invalid_argument& ia) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("failed to convert ") + key + " '" + value +
         "' to integral number")
            .c_str());
  }

  return nullptr;  // success
}

void
RequestsRespondIfError(
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    TRITONSERVER_Error* response_err)
{
  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err != nullptr) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    } else {
      std::unique_ptr<
          TRITONBACKEND_Response, decltype(&TRITONBACKEND_ResponseDelete)>
          response_handle(response, TRITONBACKEND_ResponseDelete);
      err = TRITONBACKEND_ResponseSend(
          response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, response_err);
      if (err != nullptr) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to send response");
        TRITONSERVER_ErrorDelete(err);
      }
    }
    err = TRITONBACKEND_RequestRelease(
        requests[i], TRITONSERVER_REQUEST_RELEASE_ALL);
    if (err != nullptr) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to release request");
      TRITONSERVER_ErrorDelete(err);
    }
  }
  TRITONSERVER_ErrorDelete(response_err);
}

// BackendConfig
struct BackendConfig {
  BackendConfig()
      : allow_gpu_memory_growth_(true), per_process_gpu_memory_fraction_(0.0),
        allow_soft_placement_(true), memory_limit_mb_()
  {
  }
  bool allow_gpu_memory_growth_;
  float per_process_gpu_memory_fraction_;
  bool allow_soft_placement_;
  std::map<int, std::vector<float>> memory_limit_mb_;
};

namespace GraphDef {

TRITONSERVER_Error*
ValidateSequenceControl(
    const std::string& model_name, ni::TritonJson::Value& model_config,
    const std::string& control_kind, const TRTISTF_IOList* inputs,
    bool required, bool is_boolean)
{
  ni::TritonJson::Value sequence_batching;
  RETURN_IF_ERROR(
      model_config.MemberAsObject("sequence_batching", &sequence_batching));
  std::string tensor_name;
  if (is_boolean) {
    RETURN_IF_ERROR(nib::GetBooleanSequenceControlProperties(
        sequence_batching, model_name, control_kind, required, &tensor_name,
        nullptr, nullptr, nullptr, nullptr, nullptr));
  } else {
    RETURN_IF_ERROR(nib::GetTypedSequenceControlProperties(
        sequence_batching, model_name, control_kind, required, &tensor_name,
        nullptr));
  }
  if (!tensor_name.empty()) {
    const TRTISTF_IO* input = nib::FindIOByName(inputs, tensor_name);
    if (input == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string(
               "configuration specified sequence control '" + tensor_name +
               "', but model does not provide that input")
               .c_str()));
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ValidateTRTISTFModel(
    const std::string& model_name, ni::TritonJson::Value& model_config,
    TRTISTF_Model* model)
{
  // For graphdef the model inputs and outputs are just "potential"
  // inputs and outputs since graphdef doesn't explicitly list the
  // inputs and outputs. Also, only the name is available, shape and
  // datatype are not.
  const TRTISTF_IOList* inputs = TRTISTF_ModelInputs(model);
  const TRTISTF_IOList* outputs = TRTISTF_ModelOutputs(model);

  std::set<std::string> potential_inputs, potential_outputs;
  for (const TRTISTF_IOList* itr = inputs; itr != nullptr; itr = itr->next_) {
    potential_inputs.insert(itr->io_->name_);
  }
  for (const TRTISTF_IOList* itr = outputs; itr != nullptr; itr = itr->next_) {
    potential_outputs.insert(itr->io_->name_);
  }

  ni::TritonJson::Value config_inputs;
  RETURN_IF_ERROR(model_config.MemberAsArray("input", &config_inputs));
  if (potential_inputs.size() < config_inputs.ArraySize()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "unable to load model '" + model_name +
            "', configuration expects " +
            std::to_string(config_inputs.ArraySize()) +
            " inputs, model provides at most " +
            std::to_string(potential_inputs.size()))
            .c_str());
  }

  // If this is a sequence model then make sure that the required
  // inputs are present in the model
  ni::TritonJson::Value sequence_batching;
  if (model_config.Find("sequence_batching", &sequence_batching)) {
    RETURN_IF_ERROR(ValidateSequenceControl(
        model_name, model_config, "CONTROL_SEQUENCE_START", inputs,
        false /* required */, true /* is_boolean */));
    RETURN_IF_ERROR(ValidateSequenceControl(
        model_name, model_config, "CONTROL_SEQUENCE_END", inputs,
        false /* required */, true /* is_boolean */));
    RETURN_IF_ERROR(ValidateSequenceControl(
        model_name, model_config, "CONTROL_SEQUENCE_READY", inputs,
        false /* required */, true /* is_boolean */));
    RETURN_IF_ERROR(ValidateSequenceControl(
        model_name, model_config, "CONTROL_SEQUENCE_CORRID", inputs,
        false /* required */, false /* is_boolean */));
  }

  for (size_t i = 0; i < config_inputs.ArraySize(); i++) {
    ni::TritonJson::Value io;
    RETURN_IF_ERROR(config_inputs.IndexAsObject(i, &io));
    RETURN_IF_ERROR(nib::CheckAllowedModelInput(io, potential_inputs));
  }

  ni::TritonJson::Value config_outputs;
  RETURN_IF_ERROR(model_config.MemberAsArray("output", &config_outputs));
  for (size_t i = 0; i < config_outputs.ArraySize(); i++) {
    ni::TritonJson::Value io;
    RETURN_IF_ERROR(config_outputs.IndexAsObject(i, &io));
    RETURN_IF_ERROR(nib::CheckAllowedModelOutput(io, potential_outputs));
  }

  return nullptr;  // success
}

}  // namespace GraphDef

namespace SavedModel {

TRITONSERVER_Error*
ValidateSequenceControl(
    const std::string& model_name, ni::TritonJson::Value& model_config,
    const int max_batch_size, const std::string& control_kind,
    const TRTISTF_IOList* inputs, bool required, bool is_boolean,
    bool* have_control)
{
  ni::TritonJson::Value sequence_batching;
  RETURN_IF_ERROR(
      model_config.MemberAsObject("sequence_batching", &sequence_batching));
  std::string tensor_name;
  std::string tensor_datatype;
  if (is_boolean) {
    RETURN_IF_ERROR(nib::GetBooleanSequenceControlProperties(
        sequence_batching, model_name, control_kind, required, &tensor_name,
        &tensor_datatype, nullptr, nullptr, nullptr, nullptr));
  } else {
    RETURN_IF_ERROR(nib::GetTypedSequenceControlProperties(
        sequence_batching, model_name, control_kind, required, &tensor_name,
        &tensor_datatype));
  }

  *have_control = !tensor_name.empty();
  if (*have_control) {
    const TRTISTF_IO* input = nib::FindIOByName(inputs, tensor_name);
    if (input == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string(
               "configuration specified sequence control '" + tensor_name +
               "', but model does not provide that input")
               .c_str()));
    }

    // Control tensors must have shape [1].
    std::vector<int64_t> dims{1};

    auto err = nib::CompareDims(
        model_name, tensor_name, input->shape_, dims, max_batch_size > 0,
        true /* compare_exact */);
    if (err != nullptr) {
      auto detailed_err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "unable to load model '" + model_name + "', sequence control '" +
              tensor_name + "': " + TRITONSERVER_ErrorMessage(err))
              .c_str());
      TRITONSERVER_ErrorDelete(err);
      return detailed_err;
    }

    if (!nib::CompareDataType(input->data_type_, tensor_datatype)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "unable to load model '" + model_name + "', sequence control '" +
              tensor_name + "': the model expects data-type " +
              TRITONSERVER_DataTypeString(
                  nib::ConvertDataType(input->data_type_)) +
              " but the model configuration specifies data-type " +
              tensor_datatype)
              .c_str());
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ValidateTRTISTFModel(
    const std::string& model_name, ni::TritonJson::Value& model_config,
    const int max_batch_size, TRTISTF_Model* model, IONameMap* input_name_map,
    IONameMap* output_name_map)
{
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

  ni::TritonJson::Value config_inputs;
  RETURN_IF_ERROR(model_config.MemberAsArray("input", &config_inputs));
  size_t expected_input_cnt = config_inputs.ArraySize();

  // If this is a sequence model then make sure that the required
  // inputs are present in the model and have the correct shape and
  // datatype.
  ni::TritonJson::Value sequence_batching;
  if (model_config.Find("sequence_batching", &sequence_batching)) {
    bool have_start, have_end, have_ready, have_corrid;
    RETURN_IF_ERROR(ValidateSequenceControl(
        model_name, model_config, max_batch_size, "CONTROL_SEQUENCE_START",
        inputs, false /* required */, true /* is_boolean */, &have_start));
    RETURN_IF_ERROR(ValidateSequenceControl(
        model_name, model_config, max_batch_size, "CONTROL_SEQUENCE_END",
        inputs, false /* required */, true /* is_boolean */, &have_end));
    RETURN_IF_ERROR(ValidateSequenceControl(
        model_name, model_config, max_batch_size, "CONTROL_SEQUENCE_READY",
        inputs, false /* required */, true /* is_boolean */, &have_ready));
    RETURN_IF_ERROR(ValidateSequenceControl(
        model_name, model_config, max_batch_size, "CONTROL_SEQUENCE_CORRID",
        inputs, false /* required */, false /* is_boolean */, &have_corrid));
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
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "unable to load model '" + model_name +
            "', configuration expects " +
            std::to_string(config_inputs.ArraySize()) +
            " inputs, model provides " + std::to_string(expected_inputs.size()))
            .c_str());
  }

  for (size_t i = 0; i < config_inputs.ArraySize(); i++) {
    ni::TritonJson::Value io;
    RETURN_IF_ERROR(config_inputs.IndexAsObject(i, &io));
    RETURN_IF_ERROR(nib::CheckAllowedModelInput(io, expected_inputs));

    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    const TRTISTF_IO* input = nib::FindIOByName(inputs, io_name);
    if (input == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          std::string("unexpected inference input '" + io_name + "'").c_str());
    }

    // If a reshape is provided for the input then use that when
    // validating that the TF model matches what is expected.
    std::vector<int64_t> dims;
    ni::TritonJson::Value reshape;
    if (io.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(nib::ParseShape(reshape, "shape", &dims));
    } else {
      RETURN_IF_ERROR(nib::ParseShape(io, "dims", &dims));
    }
    if (input->shape_->rank_ != 0) {
      RETURN_IF_ERROR(nib::CompareDims(
          model_name, io_name, input->shape_, dims, max_batch_size > 0,
          false /* compare_exact */));
    } else {
      // The savedmodel doesn't specify a shape for the input so use the shape
      // from the model configuration
      bool supports_batching = max_batch_size > 0;
      input->shape_->rank_ =
          (size_t)(dims.size() + (supports_batching ? 1 : 0));
      input->shape_->dims_ =
          (int64_t*)malloc(input->shape_->rank_ * sizeof(int64_t));
      for (size_t i = 0; i < dims.size(); ++i) {
        input->shape_->dims_[i + (supports_batching ? 1 : 0)] = dims[i];
      }
    }

    std::string io_data_type;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_data_type));
    if (!nib::CompareDataType(input->data_type_, io_data_type)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "unable to load model '" + model_name + "', input '" + io_name +
              "' data-type " +
              TRITONSERVER_DataTypeString(
                  nib::ConvertDataType(input->data_type_)) +
              " doesn't match configuration data-type " + io_data_type)
              .c_str());
    }
  }

  ni::TritonJson::Value config_outputs;
  RETURN_IF_ERROR(model_config.MemberAsArray("output", &config_outputs));
  for (size_t i = 0; i < config_outputs.ArraySize(); i++) {
    ni::TritonJson::Value io;
    RETURN_IF_ERROR(config_outputs.IndexAsObject(i, &io));
    RETURN_IF_ERROR(nib::CheckAllowedModelOutput(io, allowed_outputs));

    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    const TRTISTF_IO* output = nib::FindIOByName(outputs, io_name);
    if (output == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          std::string("unexpected inference output '" + io_name + "'").c_str());
    }

    // If a reshape is provided for the input then use that when
    // validating that the TF model matches what is expected.
    std::vector<int64_t> dims;
    ni::TritonJson::Value reshape;
    if (io.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(nib::ParseShape(reshape, "shape", &dims));
    } else {
      RETURN_IF_ERROR(nib::ParseShape(io, "dims", &dims));
    }

    if (output->shape_->rank_ != 0) {
      RETURN_IF_ERROR(nib::CompareDims(
          model_name, io_name, output->shape_, dims, max_batch_size > 0,
          true /* compare_exact */));
    } else {
      // The savedmodel doesn't specify a shape for the output so use the shape
      // from the model configuration
      bool supports_batching = max_batch_size > 0;
      output->shape_->rank_ =
          (size_t)(dims.size() + (supports_batching ? 1 : 0));
      output->shape_->dims_ =
          (int64_t*)malloc(output->shape_->rank_ * sizeof(int64_t));
      for (size_t i = 0; i < dims.size(); ++i) {
        output->shape_->dims_[i + (supports_batching ? 1 : 0)] = dims[i];
      }
    }

    std::string io_data_type;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_data_type));
    if (!nib::CompareDataType(output->data_type_, io_data_type)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "unable to load model '" + model_name + "', output '" + io_name +
              "' data-type " +
              TRITONSERVER_DataTypeString(
                  nib::ConvertDataType(output->data_type_)) +
              " doesn't match configuration data-type " + io_data_type)
              .c_str());
    }
  }

  return nullptr;  // success
}

class AutoCompleteHelper {
 public:
  AutoCompleteHelper(
      const std::string& model_name, TRTISTF_Model* trtistf_model)
      : model_name_(model_name),
        trtistf_model_(trtistf_model, TRTISTF_ModelDelete)
  {
  }

  TRITONSERVER_Error* Fix(
      ni::TritonJson::Value* config, const int max_batch_size);

 private:
  TRITONSERVER_Error* FixIOConfig(
      const TRTISTF_IOList* reference_list, const char* key,
      ni::TritonJson::Value* config);

  TRITONSERVER_Error* FixBatchingSupport(
      ni::TritonJson::Value* config, const int max_batch_size);

  const std::string& model_name_;
  std::unique_ptr<TRTISTF_Model, decltype(&TRTISTF_ModelDelete)> trtistf_model_;
  bool model_support_batching_;
};

TRITONSERVER_Error*
AutoCompleteHelper::Fix(ni::TritonJson::Value* config, const int max_batch_size)
{
  // Validate and fill 'max_batch_size' based on model signature and
  // config hint
  RETURN_IF_ERROR(FixBatchingSupport(config, max_batch_size));

  // Inputs
  const TRTISTF_IOList* inputs = TRTISTF_ModelInputs(trtistf_model_.get());
  RETURN_IF_ERROR(FixIOConfig(inputs, "input", config));

  // Outputs
  const TRTISTF_IOList* outputs = TRTISTF_ModelOutputs(trtistf_model_.get());
  RETURN_IF_ERROR(FixIOConfig(outputs, "output", config));

  return nullptr;  // success
}

TRITONSERVER_Error*
AutoCompleteHelper::FixBatchingSupport(
    ni::TritonJson::Value* config, const int max_batch_size)
{
  std::vector<const TRTISTF_IOList*> model_ios{
      TRTISTF_ModelInputs(trtistf_model_.get()),
      TRTISTF_ModelOutputs(trtistf_model_.get())};

  // Assume model doesn't support batching unless we see a batch
  // dimension (-1) on signature of every model input and output.
  bool sig_supports_batch = true;
  for (const auto& ios : model_ios) {
    for (const TRTISTF_IOList* itr = ios; itr != nullptr; itr = itr->next_) {
      TRTISTF_IO* io = itr->io_;
      if ((io->shape_->rank_ == 0) || (io->shape_->dims_[0] != -1)) {
        sig_supports_batch = false;
      }
    }
  }

  // If max-batch-size is explicitly set to non-zero but the model
  // signature doesn't support batching then can't autofill.
  if (!sig_supports_batch && (max_batch_size > 0)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string(
            "unable to autofill for '" + model_name_ +
            "', configuration specified max-batch " +
            std::to_string(max_batch_size) +
            " but model signature does not support batching")
            .c_str());
  }

  // 'model_support_batching_' is set to be true when all model inputs have
  // variable size first dimension, but it is not necessary to be the case
  // (i.e. non-batch model with variable size tensors). As 'max_batch_size == 0'
  // from existing config is also ambiguous, it can be either unspecified or
  // no-batch, autofill will check specified input/output (if any) for hint.
  model_support_batching_ = sig_supports_batch;
  if (model_support_batching_ && (max_batch_size == 0)) {
    bool config_batch_hint = false;
    ni::TritonJson::Value config_inputs(
        *config, ni::TritonJson::ValueType::ARRAY);
    config->Find("input", &config_inputs);
    ni::TritonJson::Value config_outputs(
        *config, ni::TritonJson::ValueType::ARRAY);
    config->Find("output", &config_outputs);
    if ((config_inputs.ArraySize() != 0) || (config_outputs.ArraySize() != 0)) {
      std::vector<ni::TritonJson::Value*> config_ios{&config_inputs,
                                                     &config_outputs};
      for (size_t ios_idx = 0; ios_idx < config_ios.size(); ios_idx++) {
        for (size_t i = 0; i < config_ios[ios_idx]->ArraySize(); i++) {
          ni::TritonJson::Value config_io;
          RETURN_IF_ERROR(config_ios[ios_idx]->IndexAsObject(i, &config_io));
          if (config_io.Find("name")) {
            std::string config_name;
            RETURN_IF_ERROR(config_io.MemberAsString("name", &config_name));
            ni::TritonJson::Value config_dims;
            if (config_io.Find("dims", &config_dims) &&
                (config_dims.ArraySize() != 0)) {
              // look up corresponding io info from model
              for (const TRTISTF_IOList* itr = model_ios[ios_idx];
                   itr != nullptr; itr = itr->next_) {
                TRTISTF_IO* io = itr->io_;
                if (config_name == io->name_) {
                  bool should_batch =
                      (io->shape_->rank_ == (config_dims.ArraySize() + 1));
                  // inconsistent hint
                  if (config_batch_hint &&
                      (model_support_batching_ != should_batch)) {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INTERNAL,
                        std::string(
                            "unable to autofill for '" + model_name_ +
                            "', model tensor configurations are "
                            "contradicting " +
                            "each other in terms of whether batching is "
                            "supported")
                            .c_str());
                  }
                  config_batch_hint = true;
                  model_support_batching_ = should_batch;
                }
              }
            }
          }
        }
      }
    }
  }

  // Set max-batch-size to 1 if the model signature and config hint agree
  if (max_batch_size == 0) {
    ni::TritonJson::Value mbs_value;
    config->Find("max_batch_size", &mbs_value);
    mbs_value.SetInt(model_support_batching_ ? 1 : 0);
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
AutoCompleteHelper::FixIOConfig(
    const TRTISTF_IOList* reference_list, const char* key,
    ni::TritonJson::Value* config)
{
  // Replace I/O even if inputs / outputs are specified in config.
  ni::TritonJson::Value ios;
  bool found_ios = config->Find(key, &ios);

  ni::TritonJson::Value auto_complete_ios(
      *config, ni::TritonJson::ValueType::ARRAY);
  for (const TRTISTF_IOList* itr = reference_list; itr != nullptr;
       itr = itr->next_) {
    TRTISTF_IO* io = itr->io_;

    ni::TritonJson::Value auto_complete_io(
        *config, ni::TritonJson::ValueType::OBJECT);
    RETURN_IF_ERROR(auto_complete_io.AddString("name", io->name_));
    RETURN_IF_ERROR(auto_complete_io.AddString(
        "data_type",
        std::string("TYPE_") +
            TRITONSERVER_DataTypeString(nib::ConvertDataType(io->data_type_))));
    ni::TritonJson::Value dims(*config, ni::TritonJson::ValueType::ARRAY);
    // The model signature supports batching then the first
    // dimension is -1 and should not appear in the model
    // configuration 'dims' that we are creating.
    for (size_t i = (model_support_batching_ ? 1 : 0); i < io->shape_->rank_;
         ++i) {
      RETURN_IF_ERROR(dims.AppendInt(io->shape_->dims_[i]));
    }

    // If io dims are empty then must use a reshape for the
    // io, since 'dims' is not allowed to be empty.
    if (dims.ArraySize() == 0) {
      RETURN_IF_ERROR(dims.AppendInt(1));
      ni::TritonJson::Value reshape(*config, ni::TritonJson::ValueType::OBJECT);
      ni::TritonJson::Value reshape_dims(
          *config, ni::TritonJson::ValueType::ARRAY);
      RETURN_IF_ERROR(reshape.Add("shape", std::move(reshape_dims)));
      RETURN_IF_ERROR(auto_complete_io.Add("reshape", std::move(reshape)));
    }
    RETURN_IF_ERROR(auto_complete_io.Add("dims", std::move(dims)));
    RETURN_IF_ERROR(auto_complete_ios.Append(std::move(auto_complete_io)));
  }
  if (found_ios) {
    ios.Swap(auto_complete_ios);
  } else {
    config->Add(key, std::move(auto_complete_ios));
  }

  return nullptr;  // success
}

}  // namespace SavedModel

// This function will return a tensor's contents as a contiguous
// chunk in system memory. In some cases this will require copying the data.
// If that  happens, 'contiguous_buffer' will be set to hold the contiguous
// chunk and 'cuda_copy' will be set to indicate whether CUDA copy is
// conducted.  The data copy can be avoided if the input is already in
// a contiguous chunk and the input is located in memory type and id
// specified.
TRITONSERVER_Error*
GetContiguousInputContent(
    TRITONBACKEND_Input* rinput, const uint32_t buffer_count,
    const char** content, size_t* content_byte_size, char** contiguous_buffer,
    cudaStream_t stream, bool* cuda_copy)
{
  *cuda_copy = false;
  *contiguous_buffer = nullptr;

  // Check input buffers to see if data copy is necessary
  size_t chunk_count = 0;
  bool type_mismatch = false;
  uint64_t total_byte_size = 0;
  for (size_t idx = 0; idx < buffer_count; ++idx) {
    TRITONSERVER_MemoryType src_memory_type;
    int64_t src_memory_type_id;
    size_t src_byte_size;
    const void* src_ptr;

    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
        rinput, idx, &src_ptr, &src_byte_size, &src_memory_type,
        &src_memory_type_id));

    if (src_ptr != nullptr) {
      chunk_count++;
      total_byte_size += src_byte_size;
      type_mismatch |= (src_memory_type == TRITONSERVER_MEMORY_GPU);
    }
  }

  if (chunk_count == 0) {
    *content = nullptr;
    *content_byte_size = 0;
  } else if ((chunk_count == 1) && !type_mismatch) {
    TRITONSERVER_MemoryType src_memory_type;
    int64_t src_memory_type_id;
    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
        rinput, 0, (const void**)content, content_byte_size, &src_memory_type,
        &src_memory_type_id));
  } else {
    *contiguous_buffer = (char*)malloc(total_byte_size);

    size_t offset = 0;
    for (size_t i = 0; i < chunk_count; i++) {
      bool cuda_used;
      TRITONSERVER_MemoryType src_memory_type;
      int64_t src_memory_type_id;
      size_t src_byte_size;
      const void* src_ptr;

      RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
          rinput, i, &src_ptr, &src_byte_size, &src_memory_type,
          &src_memory_type_id));
      RETURN_IF_ERROR(nib::CopyBuffer(
          "Contiguous input", src_memory_type, src_memory_type_id,
          TRITONSERVER_MEMORY_CPU, 0, src_byte_size, src_ptr,
          *contiguous_buffer + offset, stream, &cuda_used));
      *cuda_copy |= cuda_used;
      offset += src_byte_size;
    }

    *content = *contiguous_buffer;
    *content_byte_size = total_byte_size;
  }

  return nullptr;  // success
}

void
FillStringTensor(TRTISTF_Tensor* tensor, const size_t idx, const size_t cnt)
{
  for (size_t c = 0; c < cnt; ++c) {
    TRTISTF_TensorSetString(tensor, idx + c, nullptr, 0);
  }
}

bool
SetStringInputTensor(
    TRTISTF_Tensor* tensor, TRITONBACKEND_Input* input, const char* name,
    const uint32_t buffer_count, const size_t request_element_cnt,
    const size_t tensor_offset, TRITONBACKEND_Response** response,
    cudaStream_t stream)
{
  bool cuda_copy = false;
  size_t element_idx = 0;

  // For string data type, we always need to have the data on CPU so
  // that we can read string length and construct the string
  // properly. So if the request's input tensor is not in CPU need to
  // copy it there.
  const char* content = nullptr;
  size_t content_byte_size = 0;

  char* contiguous_buffer = nullptr;
  auto err = GetContiguousInputContent(
      input, buffer_count, &content, &content_byte_size, &contiguous_buffer,
      stream, &cuda_copy);
  if (err != nullptr) {
    RESPOND_AND_SET_NULL_IF_ERROR(response, err);
    FillStringTensor(
        tensor, tensor_offset + element_idx, request_element_cnt - element_idx);
    free(contiguous_buffer);
    return cuda_copy;
  }

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream);
    cuda_copy = false;
  }
#endif  // TRITON_ENABLE_GPU

  // Parse content and assign to 'tensor'. Each string in 'content'
  // is a 4-byte length followed by the string itself with no
  // null-terminator.
  while (content_byte_size >= sizeof(uint32_t)) {
    if (element_idx >= request_element_cnt) {
      RESPOND_AND_SET_NULL_IF_ERROR(
          response,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              std::string(
                  "unexpected number of string elements " +
                  std::to_string(element_idx + 1) + " for inference input '" +
                  name + "', expecting " + std::to_string(request_element_cnt))
                  .c_str()));
      FillStringTensor(
          tensor, tensor_offset + element_idx,
          request_element_cnt - element_idx);
      free(contiguous_buffer);
      return cuda_copy;
    }

    const uint32_t len = *(reinterpret_cast<const uint32_t*>(content));
    content += sizeof(uint32_t);
    content_byte_size -= sizeof(uint32_t);

    if (content_byte_size < len) {
      RESPOND_AND_SET_NULL_IF_ERROR(
          response,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              std::string(
                  "incomplete string data for inference input '" +
                  std::string(name) + "', expecting string of length " +
                  std::to_string(len) + " but only " +
                  std::to_string(content_byte_size) + " bytes available")
                  .c_str()));
      FillStringTensor(
          tensor, tensor_offset + element_idx,
          request_element_cnt - element_idx);
      free(contiguous_buffer);
      return cuda_copy;
    }

    TRTISTF_TensorSetString(tensor, tensor_offset + element_idx, content, len);
    content += len;
    content_byte_size -= len;
    element_idx++;
  }

  if ((*response != nullptr) && (element_idx != request_element_cnt)) {
    RESPOND_AND_SET_NULL_IF_ERROR(
        response, TRITONSERVER_ErrorNew(
                      TRITONSERVER_ERROR_INTERNAL,
                      std::string(
                          "expected " + std::to_string(request_element_cnt) +
                          " strings for inference input '" + name + "', got " +
                          std::to_string(element_idx))
                          .c_str()));
    FillStringTensor(
        tensor, tensor_offset + element_idx, request_element_cnt - element_idx);
  }

  free(contiguous_buffer);
  return cuda_copy;
}

bool
SetStringOutputBuffer(
    TRTISTF_Tensor* tensor, TRITONBACKEND_Response** response,
    TRITONBACKEND_Output* response_output, const size_t tensor_element_count,
    const size_t tensor_offset, cudaStream_t stream, std::string* serialized)
{
  bool cuda_copy = false;

  // Serialize the output tensor strings. Each string is serialized as
  // a 4-byte length followed by the string itself with no
  // null-terminator.
  serialized->clear();
  for (size_t e = 0; e < tensor_element_count; ++e) {
    size_t len;
    const char* cstr = TRTISTF_TensorString(tensor, tensor_offset + e, &len);
    serialized->append(reinterpret_cast<const char*>(&len), sizeof(uint32_t));
    if (len > 0) {
      serialized->append(cstr, len);
    }
  }

  // Allocate a buffer large enough to hold the serialized tensor.
  TRITONSERVER_MemoryType actual_memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t actual_memory_type_id = 0;

  void* buffer;
  auto err = TRITONBACKEND_OutputBuffer(
      response_output, &buffer, serialized->size(), &actual_memory_type,
      &actual_memory_type_id);
  if (err != nullptr) {
    RESPOND_AND_SET_NULL_IF_ERROR(response, err);
    return cuda_copy;
  }

  // Copy the serialized tensor into the allocated buffer.
  bool cuda_used = false;
  err = nib::CopyBuffer(
      "String output", TRITONSERVER_MEMORY_CPU /* src_memory_type */,
      0 /* src_memory_type_id */, actual_memory_type, actual_memory_type_id,
      serialized->size(), reinterpret_cast<const void*>(serialized->c_str()),
      buffer, stream, &cuda_used);
  cuda_copy |= cuda_used;

  if (err != nullptr) {
    RESPOND_AND_SET_NULL_IF_ERROR(response, err);
    return cuda_copy;
  }

  return cuda_copy;
}

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public nib::BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  ::BackendConfig* BackendConfig() const { return backend_config_; }
  const std::unordered_map<std::string, std::string>& ModelPaths() const
  {
    return model_paths_;
  }
  bool IsGraphdef() const { return is_graphdef_; }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);

  // Auto-complete the model configuration
  TRITONSERVER_Error* AutoCompleteConfig();

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

  ::BackendConfig* backend_config_;
  std::unordered_map<std::string, std::string> model_paths_;
  bool is_graphdef_;
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const nib::BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  // Auto-complete the configuration if requested...
  bool auto_complete_config = false;
  RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(
      triton_model, &auto_complete_config));
  if (auto_complete_config) {
    RETURN_IF_ERROR((*state)->AutoCompleteConfig());

    ni::TritonJson::WriteBuffer json_buffer;
    (*state)->ModelConfig().Write(&json_buffer);

    TRITONSERVER_Message* message;
    RETURN_IF_ERROR(TRITONSERVER_MessageNewFromSerializedJson(
        &message, json_buffer.Base(), json_buffer.Size()));
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetConfig(
        triton_model, 1 /* config_version */, message));
  }

  RETURN_IF_ERROR((*state)->ValidateModelConfig());

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : nib::BackendModel(triton_model)
{
  // Obtain backend configuration
  TRITONBACKEND_Backend* backend;
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelBackend(triton_model, &backend));
  void* vstate;
  THROW_IF_BACKEND_MODEL_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  backend_config_ = reinterpret_cast<::BackendConfig*>(vstate);

  std::string platform;
  THROW_IF_BACKEND_MODEL_ERROR(
      ModelConfig().MemberAsString("platform", &platform));
  if (platform == "tensorflow_graphdef") {
    is_graphdef_ = true;
  } else if (platform == "tensorflow_savedmodel") {
    is_graphdef_ = false;
  } else {
    throw nib::BackendModelException(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, (std::string("platform ") + platform +
                                         " not supported for TensorFlow "
                                         "model '" +
                                         Name() + "'")
                                            .c_str()));
  }

  THROW_IF_BACKEND_MODEL_ERROR(nib::ModelPaths(
      RepositoryPath(), Version(), is_graphdef_, !is_graphdef_, &model_paths_));
}

TRITONSERVER_Error*
ModelState::AutoCompleteConfig()
{
  // Nothing to be filled for graphdef as the model itself does not
  // provide information needed.
  if (!is_graphdef_) {
    // Attempt to auto-complete the config with first loaded model file.
    // 'default_model_filename' is the first model file to try.
    std::string default_model_filename;
    ModelConfig().MemberAsString(
        "default_model_filename", &default_model_filename);
    if (default_model_filename.empty()) {
      default_model_filename = "model.savedmodel";
    }

    TRTISTF_Model* trtistf_model = nullptr;
    TRTISTF_Error* err = nullptr;
    auto it = model_paths_.find(default_model_filename);
    if (it != model_paths_.end()) {
      err = TRTISTF_ModelCreateFromSavedModel(
          &trtistf_model, Name().c_str(), it->second.c_str(),
          TRTISTF_NO_GPU_DEVICE, false /* have_graph */, 0 /* graph_level */,
          backend_config_->allow_gpu_memory_growth_,
          backend_config_->per_process_gpu_memory_fraction_,
          backend_config_->allow_soft_placement_,
          backend_config_->memory_limit_mb_, nullptr /* tftrt_config */,
          false /* auto_mixed precision */);
    }

    // If 'default_model_filename' fails, traverse all possible model files
    if (err != nullptr) {
      for (it = model_paths_.begin(); it != model_paths_.end(); it++) {
        if (it->first == default_model_filename) {
          continue;
        }
        // Release previous error
        TRTISTF_ErrorDelete(err);
        err = TRTISTF_ModelCreateFromSavedModel(
            &trtistf_model, Name().c_str(), it->second.c_str(),
            TRTISTF_NO_GPU_DEVICE, false /* have_graph */, 0 /* graph_level */,
            backend_config_->allow_gpu_memory_growth_,
            backend_config_->per_process_gpu_memory_fraction_,
            backend_config_->allow_soft_placement_,
            backend_config_->memory_limit_mb_, nullptr /* tftrt_config */,
            false /* auto_mixed precision */);
        if (err == nullptr) {
          break;
        }
      }
    }

    if (err != nullptr) {
      std::string msg((err->msg_ == nullptr) ? "<unknown>" : err->msg_);
      TRTISTF_ErrorDelete(err);
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unable to auto-complete model configuration for '") +
           Name() + "', failed to load model: " + msg)
              .c_str());
    }

    auto ach = SavedModel::AutoCompleteHelper(Name(), trtistf_model);
    RETURN_IF_ERROR(ach.Fix(&ModelConfig(), MaxBatchSize()));
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // We have the json DOM for the model configuration...
  ni::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  ni::TritonJson::Value ios;
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    ni::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    // Check datatypes
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    RETURN_ERROR_IF_TRUE(
        nib::ConvertDataType(io_dtype) ==
            TRTISTF_DataType::TRTISTF_TYPE_INVALID,
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("unsupported datatype '") + io_dtype + "' for tensor '" +
            io_name + "' for model '" + Name() + "'");
  }
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("output", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    ni::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    // Check datatypes
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    RETURN_ERROR_IF_TRUE(
        nib::ConvertDataType(io_dtype) ==
            TRTISTF_DataType::TRTISTF_TYPE_INVALID,
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("unsupported datatype '") + io_dtype + "' for tensor '" +
            io_name + "' for model '" + Name() + "'");
  }

  return nullptr;  // success
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public nib::BackendModelInstance {
 public:
  // GPU device number that indicates that no gpu is available for a
  // context (which is an invalid state since TensorRT requires a
  // GPU).
  static constexpr int NO_GPU_DEVICE = -1;

  // GPU device number that indicates model will be loaded on GPUs
  // as specified in model graph
  static constexpr int MODEL_DEVICE = -2;

  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState() = default;

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);

  ModelState* model_state_;

  // Map from configuration name for an input to tensor name for
  // that input in the model.
  IONameMap input_name_map_;

  // Map from configuration name for an output to tensor name for
  // that output in the model.
  IONameMap output_name_map_;

  // TRTISTFModel for this context.
  TRTISTFModelHandle trtistf_model_;

  // use for GPU allocator
  int input_device_id_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const nib::BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  // If the model configuration doesn't have an explicit model file
  // specified then use the default name.
  std::string cc_model_filename = (*state)->ArtifactFilename();
  if (cc_model_filename.empty()) {
    if (model_state->IsGraphdef()) {
      cc_model_filename = "model.graphdef";
    } else {
      cc_model_filename = "model.savedmodel";
    }
  }

  const auto& gdp_itr = model_state->ModelPaths().find(cc_model_filename);
  if (gdp_itr == model_state->ModelPaths().end()) {
    RETURN_ERROR_IF_FALSE(
        false, TRITONSERVER_ERROR_INTERNAL,
        (std::string("unable to find model '") + cc_model_filename + "' for " +
         (*state)->Name()));
  }

  int gpu_device;
  switch ((*state)->Kind()) {
    case TRITONSERVER_INSTANCEGROUPKIND_CPU:
      gpu_device = ModelInstanceState::NO_GPU_DEVICE;
      break;
    case TRITONSERVER_INSTANCEGROUPKIND_MODEL:
      gpu_device = ModelInstanceState::MODEL_DEVICE;
      break;
    default:
      gpu_device = (*state)->DeviceId();
      break;
  }

  // Max batch size. A value of 0 in the config becomes NO_BATCHING.
  const int mbs = (model_state->MaxBatchSize() <= 0)
                      ? nib::BackendModel::NO_BATCHING
                      : model_state->MaxBatchSize();

  TRTISTF_TFTRTConfig* tftrt_config_ptr = nullptr;
  TRTISTF_TFTRTConfig tftrt_config;
  bool auto_mixed_precision = false;
  bool has_graph_level = false;
  int64_t graph_level = 0;
  // [TODO] this can be moved one level above
  {
    ni::TritonJson::Value optimization;
    if (model_state->ModelConfig().Find("optimization", &optimization)) {
      {
        ni::TritonJson::Value graph;
        if ((has_graph_level = optimization.Find("graph", &graph))) {
          RETURN_IF_ERROR(graph.MemberAsInt("level", &graph_level));
        }
      }
      ni::TritonJson::Value eas;
      if (optimization.Find("execution_accelerators", &eas)) {
        // Set default values. is_dynamic_op is always true for online
        // TF-TRT.
        tftrt_config.minimum_segment_size_ = 3;
        tftrt_config.max_workspace_size_bytes_ = 1 << 30;
        tftrt_config.max_cached_engines_ = 100;
        tftrt_config.max_batch_size_ = std::max(mbs, 1);
        tftrt_config.precision_mode_ = TRTISTF_MODE_FP32;
        tftrt_config.is_dynamic_op_ = true;

        ni::TritonJson::Value cpu_eas;
        RETURN_ERROR_IF_TRUE(
            eas.Find("cpu_execution_accelerator", &cpu_eas) &&
                (cpu_eas.ArraySize() != 0),
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string("CPU Execution Accelerator is not supported in "
                        "TensorFlow backend"));

        RETURN_ERROR_IF_TRUE(
            gpu_device == ModelInstanceState::NO_GPU_DEVICE,
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string(
                "GPU Execution Accelerator can only be set on non-CPU backend "
                "context"));

        ni::TritonJson::Value gpu_eas;
        if (eas.Find("gpu_execution_accelerator", &gpu_eas)) {
          for (size_t ea_idx = 0; ea_idx < gpu_eas.ArraySize(); ea_idx++) {
            ni::TritonJson::Value ea;
            RETURN_IF_ERROR(gpu_eas.IndexAsObject(ea_idx, &ea));
            std::string name;
            RETURN_IF_ERROR(ea.MemberAsString("name", &name));
            if (name == nib::kTensorRTExecutionAccelerator) {
              // Validate and set parameters
              ni::TritonJson::Value params;
              if (ea.Find("parameters", &params)) {
                std::vector<std::string> param_keys;
                RETURN_IF_ERROR(params.Members(&param_keys));
                for (const auto& param_key : param_keys) {
                  std::string value_string;
                  if (param_key == "precision_mode") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    if (value_string == "FP32") {
                      tftrt_config.precision_mode_ = TRTISTF_MODE_FP32;
                    } else if (value_string == "FP16") {
                      tftrt_config.precision_mode_ = TRTISTF_MODE_FP16;
                    } else {
                      RETURN_ERROR_IF_FALSE(
                          false, TRITONSERVER_ERROR_INVALID_ARG,
                          std::string("unsupported precision mode '") +
                              value_string + "' is requested");
                    }
                  } else if (param_key == "minimum_segment_size") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    RETURN_IF_ERROR(ParseLongLongParameter(
                        "minimum_segment_size", value_string,
                        &tftrt_config.minimum_segment_size_));
                  } else if (param_key == "max_workspace_size_bytes") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    RETURN_IF_ERROR(ParseLongLongParameter(
                        "max_workspace_size_bytes", value_string,
                        &tftrt_config.max_workspace_size_bytes_));
                  } else if (param_key == "max_cached_engines") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    RETURN_IF_ERROR(ParseLongLongParameter(
                        "max_cached_engines", value_string,
                        &tftrt_config.max_cached_engines_));
                  } else {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INVALID_ARG,
                        std::string(
                            "unknown parameter '" + param_key +
                            "' is provided for TensorRT Execution Accelerator")
                            .c_str());
                  }
                }
              }
              tftrt_config_ptr = &tftrt_config;
              LOG_MESSAGE(
                  TRITONSERVER_LOG_VERBOSE,
                  (std::string("TensorRT Execution Accelerator is set for ") +
                   (*state)->Name())
                      .c_str());
            } else if (name == nib::kGPUIOExecutionAccelerator) {
              // GPU I/O can be set, set hint
              if ((gpu_device != ModelInstanceState::NO_GPU_DEVICE) &&
                  (gpu_device != ModelInstanceState::MODEL_DEVICE)) {
                (*state)->input_device_id_ = gpu_device;
              }
            } else if (name == nib::kAutoMixedPrecisionExecutionAccelerator) {
              auto_mixed_precision = true;
            } else {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("unknown Execution Accelerator '") + name +
                   "' is requested")
                      .c_str());
            }
          }
        }
      }
    }
  }

  if (auto_mixed_precision && (tftrt_config_ptr != nullptr)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Auto mixed precision can not be set with TFTRT optimization");
  }

  TRTISTF_Model* model = nullptr;
  if (model_state->IsGraphdef()) {
    RETURN_IF_TRTISTF_ERROR(TRTISTF_ModelCreateFromGraphDef(
        &model, model_state->Name().c_str(), gdp_itr->second.c_str(),
        gpu_device, has_graph_level, graph_level,
        model_state->BackendConfig()->allow_gpu_memory_growth_,
        model_state->BackendConfig()->per_process_gpu_memory_fraction_,
        model_state->BackendConfig()->allow_soft_placement_,
        model_state->BackendConfig()->memory_limit_mb_, tftrt_config_ptr,
        auto_mixed_precision));
    (*state)->trtistf_model_.reset(model);

    RETURN_IF_ERROR(GraphDef::ValidateTRTISTFModel(
        model_state->Name(), model_state->ModelConfig(), model));
  } else {
    RETURN_IF_TRTISTF_ERROR(TRTISTF_ModelCreateFromSavedModel(
        &model, model_state->Name().c_str(), gdp_itr->second.c_str(),
        gpu_device, has_graph_level, graph_level,
        model_state->BackendConfig()->allow_gpu_memory_growth_,
        model_state->BackendConfig()->per_process_gpu_memory_fraction_,
        model_state->BackendConfig()->allow_soft_placement_,
        model_state->BackendConfig()->memory_limit_mb_, tftrt_config_ptr,
        auto_mixed_precision));
    (*state)->trtistf_model_.reset(model);

    RETURN_IF_ERROR(SavedModel::ValidateTRTISTFModel(
        model_state->Name(), model_state->ModelConfig(),
        model_state->MaxBatchSize(), model, &((*state)->input_name_map_),
        &((*state)->output_name_map_)));
  }

  if ((*state)->input_device_id_ != ModelInstanceState::MODEL_DEVICE) {
    std::vector<const char*> input_names, output_names;
    std::vector<TRTISTF_DataType> input_types, output_types;
    std::deque<std::string> io_names;

    ni::TritonJson::Value config_inputs;
    RETURN_IF_ERROR(
        model_state->ModelConfig().MemberAsArray("input", &config_inputs));
    for (size_t i = 0; i < config_inputs.ArraySize(); i++) {
      ni::TritonJson::Value io;
      RETURN_IF_ERROR(config_inputs.IndexAsObject(i, &io));
      io_names.emplace_back();
      RETURN_IF_ERROR(io.MemberAsString("name", &io_names.back()));
      std::string io_data_type;
      RETURN_IF_ERROR(io.MemberAsString("data_type", &io_data_type));

      input_names.push_back(io_names.back().c_str());
      input_types.push_back(nib::ConvertDataType(io_data_type));
    }

    ni::TritonJson::Value config_outputs;
    RETURN_IF_ERROR(
        model_state->ModelConfig().MemberAsArray("output", &config_outputs));
    for (size_t i = 0; i < config_outputs.ArraySize(); i++) {
      ni::TritonJson::Value io;
      RETURN_IF_ERROR(config_outputs.IndexAsObject(i, &io));
      io_names.emplace_back();
      RETURN_IF_ERROR(io.MemberAsString("name", &io_names.back()));
      std::string io_data_type;
      RETURN_IF_ERROR(io.MemberAsString("data_type", &io_data_type));

      output_names.push_back(io_names.back().c_str());
      output_types.push_back(nib::ConvertDataType(io_data_type));
    }
    RETURN_IF_TRTISTF_ERROR(TRTISTF_ModelMakeCallable(
        (*state)->trtistf_model_.get(), input_names.data(), input_types.data(),
        config_inputs.ArraySize(), output_names.data(), output_types.data(),
        config_outputs.ArraySize()));
  }

  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state), trtistf_model_(nullptr, TRTISTF_ModelDelete),
      input_device_id_(MODEL_DEVICE)
{
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: Running ") + Name() + " with " +
       std::to_string(request_count) + " requests")
          .c_str());

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  const int max_batch_size = StateForModel()->MaxBatchSize();

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.
  size_t total_batch_size = 0;
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondIfError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to TensorFlow runner for '" + Name() +
                  "'")
                  .c_str()));
      return;
    }

    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs,
      // if the model support batching, the first dimension size is batch size
      //
      // FIXME A backend API to get input by index would be must faster here
      const char* name;
      auto err = TRITONBACKEND_RequestInputName(requests[i], 0, &name);
      TRITONBACKEND_Input* input;
      if (err == nullptr) {
        err = TRITONBACKEND_RequestInput(requests[i], name, &input);
      }
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
      }
      if (err != nullptr) {
        RequestsRespondIfError(requests, request_count, err);
        return;
      }
    } else {
      total_batch_size += 1;
    }
  }

  // If there are no valid requests then no need to run the
  // inference. This should never happen unless called with an empty
  // 'requests' for some reason.
  if (total_batch_size == 0) {
    return;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size)) {
    RequestsRespondIfError(
        requests, request_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "dynamic batch size " + std::to_string(total_batch_size) +
                " for '" + Name() + "', max allowed is " +
                std::to_string(max_batch_size))
                .c_str()));
    return;
  }

  // At this point we are committed to running inference with all
  // 'requests'. Create a response for each request. During input
  // processing if there is an error with any request that error will
  // be sent immediately with the corresponding response (and the
  // response pointer will then be nullptr). The request object
  // itself will not be released until after all inferencing is done
  // (below) as we may need to access the request object when
  // determine how to process outputs (for example, even if we don't
  // need the outputs for a request that has an error, we do need to
  // know the size of those outputs associated with the request so we
  // can skip them in the output tensors).
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  // Create a tensor for each input sized correctly for the total
  // batch size. Concatenate input values from each request into the
  // corresponding tensor.

  // Unique pointer is TensorList** as the pointer to input head
  // (TensorList*) will be updated in SetInput()
  TRTISTF_TensorList* input_head_ptr = nullptr;
  static auto input_deleter = [](TRTISTF_TensorList** list) {
    if (list != nullptr) {
      TRTISTF_TensorListDelete(*list);
    }
  };
  std::unique_ptr<TRTISTF_TensorList*, decltype(input_deleter)> input_tensors(
      &input_head_ptr, input_deleter);

  // Collect the request inputs into contiguous input tensors. For
  // tensors with string data type we must handle ourselves since we
  // must use TF-specific string tensor APIs.
  bool cuda_copy = false;

  nib::BackendInputCollector collector(
      requests, request_count, &responses, StateForModel()->EnablePinnedInput(),
      CudaStream());
  {
    // All requests must have equally-sized input tensors so use the first
    // request as the representative for the input tensors.
    uint32_t input_count;
    TRITONBACKEND_RequestInputCount(requests[0], &input_count);
    for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
      const char* name;
      TRITONBACKEND_RequestInputName(requests[0], input_idx, &name);
      TRITONBACKEND_Input* input;
      TRITONBACKEND_RequestInput(requests[0], name, &input);
      TRITONSERVER_DataType datatype;
      const int64_t* shape;
      uint32_t dims_count;
      uint32_t buffer_count;
      TRITONBACKEND_InputProperties(
          input, nullptr, &datatype, &shape, &dims_count, nullptr,
          &buffer_count);

      // The shape for the entire input patch, [total_batch_size, ...]
      std::vector<int64_t> batchn_shape(shape, shape + dims_count);
      if (max_batch_size != nib::BackendModel::NO_BATCHING) {
        batchn_shape[0] = total_batch_size;
      }

      // The name of the input in the model can be different...
      const char* input_tensor_name = name;
      const auto& tn_itr = input_name_map_.find(input_tensor_name);
      if (tn_itr != input_name_map_.end()) {
        input_tensor_name = tn_itr->second.c_str();
      }

      // Create a TF tensor to hold the entire input batch. Only try
      // to create a tensor on a specific device if 'input_device_id_'
      // is set. If unable to create the tensor then fail all
      // requests.
      TRTISTF_Tensor* tensor = TRTISTF_TensorNew(
          input_tensor_name, nib::ConvertDataType(datatype),
          batchn_shape.size(),
          (batchn_shape.size() == 0) ? nullptr : &batchn_shape[0],
          input_device_id_);
      if (tensor == nullptr) {
        auto err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("failed to create input tensor '") + name +
             "' with shape " + nib::ShapeToString(batchn_shape) +
             " and data type " + TRITONSERVER_DataTypeString(datatype) +
             " for '" + Name() + "'")
                .c_str());
        // Send remaining responses and returned
        for (uint32_t r = 0; r < request_count; ++r) {
          if (responses[r] != nullptr) {
            LOG_IF_ERROR(
                TRITONBACKEND_ResponseSend(
                    responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
                "failed to send TensorFlow backend response");
          }

          LOG_IF_ERROR(
              TRITONBACKEND_RequestRelease(
                  requests[r], TRITONSERVER_REQUEST_RELEASE_ALL),
              "failed releasing request");
        }
        TRITONSERVER_ErrorDelete(err);
        return;
      }

      // Add the new TF tensor to the list of TF inputs.
      TRTISTF_TensorList* tlink = TRTISTF_TensorListNew(tensor, *input_tensors);
      *input_tensors = tlink;

      // Custom handling for string/bytes tensor...
      if (datatype == TRITONSERVER_TYPE_BYTES) {
        size_t tensor_offset = 0;

        for (size_t idx = 0; idx < request_count; idx++) {
          TRITONBACKEND_Input* input;
          RESPOND_AND_SET_NULL_IF_ERROR(
              &responses[idx],
              TRITONBACKEND_RequestInput(requests[idx], name, &input));
          const int64_t* shape;
          uint32_t dims_count;
          uint32_t buffer_count;
          RESPOND_AND_SET_NULL_IF_ERROR(
              &responses[idx], TRITONBACKEND_InputProperties(
                                   input, nullptr, nullptr, &shape, &dims_count,
                                   nullptr, &buffer_count));

          const int64_t batch_element_cnt =
              nib::GetElementCount(shape, dims_count);

          cuda_copy |= SetStringInputTensor(
              tensor, input, name, buffer_count, batch_element_cnt,
              tensor_offset, &responses[idx], CudaStream());
          tensor_offset += batch_element_cnt;
        }
      }
      // Use the collector for non-STRING datatype...
      else {  // datatype != DataType::TYPE_STRING
        collector.ProcessTensor(
            name, TRTISTF_TensorData(tensor),
            TRTISTF_TensorDataByteSize(tensor),
            (TRTISTF_TensorIsGPUTensor(tensor)) ? TRITONSERVER_MEMORY_GPU
                                                : TRITONSERVER_MEMORY_CPU,
            (TRTISTF_TensorIsGPUTensor(tensor)) ? DeviceId() : 0);
      }

      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string("TRITONBACKEND_ModelExecute: input '") + name +
           "' is GPU tensor: " +
           ((TRTISTF_TensorIsGPUTensor(tensor)) ? "true" : "false"))
              .c_str());
    }

    // Finalize...
    cuda_copy |= collector.Finalize();
  }

  // Collect the names of requested outputs. Do not include outputs
  // for requests that have already responded with an error.
  std::set<std::string> required_outputs;
  std::vector<std::set<std::string>> request_required_outputs(request_count);
  for (size_t idx = 0; idx < request_count; idx++) {
    const auto& request = requests[idx];
    auto& response = responses[idx];
    if (response != nullptr) {
      uint32_t output_count;
      RESPOND_AND_SET_NULL_IF_ERROR(
          &response, TRITONBACKEND_RequestOutputCount(request, &output_count));
      if (response != nullptr) {
        for (uint32_t output_idx = 0; output_idx < output_count; output_idx++) {
          const char* output_name;
          RESPOND_AND_SET_NULL_IF_ERROR(
              &response, TRITONBACKEND_RequestOutputName(
                             request, output_idx, &output_name));
          if (response != nullptr) {
            required_outputs.insert(output_name);
            request_required_outputs[idx].insert(output_name);
          }
        }
      }
    }
  }

  // Create the vector of required output names using the names
  // expected by the model.
  std::vector<std::string> model_output_names;
  const char* output_names_cstr[required_outputs.size()];
  {
    size_t oidx = 0;
    for (const auto& name : required_outputs) {
      model_output_names.push_back(name);
      const auto& tn_itr = output_name_map_.find(name);
      if (tn_itr == output_name_map_.end()) {
        output_names_cstr[oidx] = name.c_str();
      } else {
        output_names_cstr[oidx] = tn_itr->second.c_str();
      }
      oidx++;
    }
  }

  // Wait for any in-flight input tensor copies to complete.
#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(CudaStream());
  }
#endif

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  // Run. Session will update the 'output_tensors'.
  std::unique_ptr<TRTISTF_TensorList, decltype(&TRTISTF_TensorListDelete)>
      output_tensors(nullptr, TRTISTF_TensorListDelete);

  {
    TRTISTF_TensorList* rtl = nullptr;

    TRTISTF_Error* tf_err = TRTISTF_ModelRun(
        trtistf_model_.get(), *(input_tensors.release()),
        required_outputs.size(), output_names_cstr, &rtl);
    if (tf_err != nullptr) {
      auto err =
          TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, tf_err->msg_);
      TRTISTF_ErrorDelete(tf_err);
      // Send remaining responses and returned
      for (uint32_t r = 0; r < request_count; ++r) {
        if (responses[r] != nullptr) {
          LOG_IF_ERROR(
              TRITONBACKEND_ResponseSend(
                  responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
              "failed to send TensorFlow backend response");
        }

        LOG_IF_ERROR(
            TRITONBACKEND_RequestRelease(
                requests[r], TRITONSERVER_REQUEST_RELEASE_ALL),
            "failed releasing request");
      }
      TRITONSERVER_ErrorDelete(err);
      return;
    }

    output_tensors.reset(rtl);
  }

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  // Create the response tensors and copy the appropriate tensor data
  // into each. For tensors with string data type we must handle
  // ourselves since we must use TF-specific string tensor APIs.
  cuda_copy = false;
  // The serialized string buffer must be valid until output copies are done
  std::vector<std::unique_ptr<std::string>> string_buffer;
  nib::BackendOutputResponder responder(
      requests, request_count, &responses, max_batch_size,
      StateForModel()->EnablePinnedOutput(), CudaStream());
  {
    TRTISTF_TensorList* output_tensor_itr = output_tensors.get();
    for (const auto& name : model_output_names) {
      TRTISTF_Tensor* output_tensor = output_tensor_itr->tensor_;

      TRTISTF_DataType tf_datatype = TRTISTF_TensorDataType(output_tensor);
      TRTISTF_Shape* tf_shape = TRTISTF_TensorShape(output_tensor);

      const TRITONSERVER_DataType datatype = nib::ConvertDataType(tf_datatype);

      // batchn_shape holds the shape of the entire tensor batch, but
      // is overwritten below and used as the shape for each response
      // output.
      std::vector<int64_t> batchn_shape;
      batchn_shape.reserve(tf_shape->rank_);
      for (size_t itr = 0; itr < tf_shape->rank_; itr++) {
        const int64_t dim = tf_shape->dims_[itr];
        batchn_shape.push_back(dim);
      }

      // Custom handling for string/bytes tensor...
      if (datatype == TRITONSERVER_TYPE_BYTES) {
        size_t tensor_offset = 0;

        for (size_t idx = 0; idx < responses.size(); idx++) {
          auto& request = requests[idx];
          auto& response = responses[idx];

          if (max_batch_size != nib::BackendModel::NO_BATCHING) {
            // [TODO] remember some input properties on the first call
            const char* name;
            TRITONBACKEND_RequestInputName(request, 0, &name);
            TRITONBACKEND_Input* input;
            TRITONBACKEND_RequestInput(request, name, &input);
            const int64_t* shape;
            TRITONBACKEND_InputProperties(
                input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
            batchn_shape[0] = shape[0];
          }

          const size_t tensor_element_cnt = nib::GetElementCount(batchn_shape);

          // Only need an response tensor for requested outputs.
          if ((response != nullptr) &&
              (request_required_outputs[idx].find(name) !=
               request_required_outputs[idx].end())) {
            TRITONBACKEND_Output* response_output;
            RESPOND_AND_SET_NULL_IF_ERROR(
                &response,
                TRITONBACKEND_ResponseOutput(
                    response, &response_output, name.c_str(), datatype,
                    batchn_shape.data(), batchn_shape.size()));
            string_buffer.emplace_back(new std::string());
            cuda_copy |= SetStringOutputBuffer(
                output_tensor, &response, response_output, tensor_element_cnt,
                tensor_offset, CudaStream(), string_buffer.back().get());
          }

          tensor_offset += tensor_element_cnt;
        }
      }
      // Use the responder for non-STRING datatype...
      else {  // datatype != DataType::TYPE_STRING
        responder.ProcessTensor(
            name, datatype, batchn_shape, TRTISTF_TensorData(output_tensor),
            (TRTISTF_TensorIsGPUTensor(output_tensor))
                ? TRITONSERVER_MEMORY_GPU
                : TRITONSERVER_MEMORY_CPU,
            (TRTISTF_TensorIsGPUTensor(output_tensor)) ? DeviceId() : 0);
      }

      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string("TRITONBACKEND_ModelExecute: output '") + name +
           "' is GPU tensor: " +
           ((TRTISTF_TensorIsGPUTensor(output_tensor)) ? "true" : "false"))
              .c_str());

      output_tensor_itr = output_tensor_itr->next_;
    }

    // Finalize and wait for any pending buffer copies.
    cuda_copy |= responder.Finalize();
  }

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(CudaStream());
  }
#endif  // TRITON_ENABLE_GPU

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  // Send all the responses that haven't already been sent because of
  // an earlier error. Note that the responses are not set to nullptr
  // for later checking.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send TensorFlow backend response");
    }
  }

  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];
    // Report statistics for the request. Note that there could
    // still be responses that have not yet been sent but those
    // cannot be captured in the statistics as they reflect only the
    // request object. We use the execution start/end time for
    // compute also so that the entire execution time is associated
    // with the inference computation.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  // Report the entire batch statistics. This backend does not support
  // batching so the total batch size is always 1.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          TritonModelInstance(), total_batch_size, exec_start_ns,
          compute_start_ns, compute_end_ns, exec_end_ns),
      "failed reporting batch request statistics");

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: model ") + Name() +
       " released " + std::to_string(request_count) + " requests")
          .c_str());
}

}  // namespace

/////////////

extern "C" {

// Implementing TRITONBACKEND_Initialize is optional. The backend
// should initialize any global state that is intended to be shared
// across all models and model instances that use the backend. But here
// it simply verify the backend API version is compatible
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // We should check the backend API version that Triton supports
  // vs. what this backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }

  // The backend configuration may contain information needed by the
  // backend, such a command-line arguments.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  ni::TritonJson::Value backend_config;
  if (byte_size != 0) {
    RETURN_IF_ERROR(backend_config.Parse(buffer, byte_size));
  }

  std::unique_ptr<BackendConfig> lconfig(new BackendConfig());
  ni::TritonJson::Value cmdline;
  if (backend_config.Find("cmdline", &cmdline)) {
    ni::TritonJson::Value value;
    if (cmdline.Find("allow-soft-placement", &value)) {
      RETURN_IF_ERROR(value.AsBool(&lconfig->allow_soft_placement_));
    }
    if (cmdline.Find("gpu-memory-fraction", &value)) {
      double lvalue;
      RETURN_IF_ERROR(value.AsDouble(&lvalue));
      lconfig->per_process_gpu_memory_fraction_ = lvalue;
      lconfig->allow_gpu_memory_growth_ = (lvalue == 0.0);
    }
  }
  RETURN_IF_ERROR(TRITONBACKEND_BackendSetState(
      backend, reinterpret_cast<void*>(lconfig.get())));

  lconfig.release();
  return nullptr;  // success
}

// Implementing TRITONBACKEND_Finalize is optional unless state is set
// using TRITONBACKEND_BackendSetState. The backend must free this
// state and perform any other global cleanup.
TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  auto config = reinterpret_cast<BackendConfig*>(vstate);
  delete config;
  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  ModelState* model_state = nullptr;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceInitialize is optional. The
// backend should initialize any state that is required for a model
// instance.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name +
       " (device " + std::to_string(device_id) + ")")
          .c_str());

  // The instance can access the corresponding model as well... here
  // we get the model and from that get the model's state.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // With each instance we create a ModelInstanceState object and
  // associate it with the TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceFinalize is optional unless
// state is set using TRITONBACKEND_ModelInstanceSetState. The backend
// must free this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceExecute is required.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.

  // Note that access to 'requests' will be invalidated once
  // TRITONBACKEND_ModelInstanceExecute returns. If requests need to be
  // referred after TRITONBACKEND_ModelInstanceExecute returns, i.e. in
  // non-blocking model, the request array must be copied to a instance
  // owned buffer.
  instance_state->ProcessRequests(requests, request_count);

  return nullptr;  // success
}

}  // extern "C"
