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

#include "src/backends/onnx/autofill.h"

#include <NvInfer.h>
#include "src/backends/onnx/loader.h"
#include "src/backends/onnx/onnx_utils.h"
#include "src/core/autofill.h"
#include "src/core/constants.h"
#include "src/core/filesystem.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"

namespace nvidia { namespace inferenceserver {

namespace {

template <class ModelIO>
void
SetIOConfig(
    const std::string& name, const OnnxTensorInfo& info, bool batching,
    ModelIO* config_io)
{
  config_io->set_name(name);

  // only set type and shape if they are not set
  if (config_io->data_type() == DataType::TYPE_INVALID) {
    config_io->set_data_type(ConvertFromOnnxDataType(info.type_));
  }

  if (config_io->dims_size() == 0) {
    // Skip batching dimension
    size_t idx = (batching) ? 1 : 0;
    for (; idx < info.dims_.size(); ++idx) {
      config_io->mutable_dims()->Add(info.dims_[idx]);
    }

    // If tensor dims are empty then must use a reshape for the
    // tensor, since 'dims' is not allowed to be empty.
    if (config_io->dims_size() == 0) {
      config_io->mutable_dims()->Add(1);
      config_io->mutable_reshape();
    }
  }
}

Status
ValidateIOInfoType(
    const std::string& model_name, const OnnxTensorInfoMap& infos)
{
  // Validate all tensors are in supported data type
  for (const auto& io_info : infos) {
    if (ConvertFromOnnxDataType(io_info.second.type_) ==
        DataType::TYPE_INVALID) {
      return Status(
          RequestStatusCode::INTERNAL,
          "unable to autofill for '" + model_name +
              "', unsupported data-type '" +
              OnnxDataTypeName(io_info.second.type_) + "'");
    }
  }
  return Status::Success;
}

}  // namespace

class AutoFillOnnxImpl : public AutoFill {
 public:
  AutoFillOnnxImpl(
      const std::string& model_name, const std::string& onnx_filename)
      : AutoFill(model_name), onnx_filename_(onnx_filename)
  {
  }

  Status Fix(ModelConfig* config) override;

  Status SetConfigFromOrtSession(OrtSession* session, OrtAllocator* allocator);

 private:
  Status FixBatchingSupport(ModelConfig* config);
  Status FixInputConfig(ModelConfig* config);
  Status FixOutputConfig(ModelConfig* config);

  Status SetBatchingSupport();

  const std::string onnx_filename_;
  bool model_support_batching_;
  OnnxTensorInfoMap input_infos_;
  OnnxTensorInfoMap output_infos_;
};

Status
AutoFillOnnxImpl::Fix(ModelConfig* config)
{
  config->set_platform(kOnnxRuntimeOnnxPlatform);

  // Set name if not already set.
  if (config->name().empty()) {
    config->set_name(model_name_);
  }

  if (config->default_model_filename().empty()) {
    config->set_default_model_filename(onnx_filename_);
  }

  // Validate and fill 'max_batch_size' based on model info and config hint
  RETURN_IF_ERROR(FixBatchingSupport(config));

  // Validate and fill inputs
  RETURN_IF_ERROR(ValidateIOInfoType(model_name_, input_infos_));
  RETURN_IF_ERROR(FixInputConfig(config));

  // Validate and fill outputs
  RETURN_IF_ERROR(ValidateIOInfoType(model_name_, output_infos_));
  RETURN_IF_ERROR(FixOutputConfig(config));

  return Status::Success;
}

Status
AutoFillOnnxImpl::FixBatchingSupport(ModelConfig* config)
{
  if (!model_support_batching_ && (config->max_batch_size() > 0)) {
    return Status(
        RequestStatusCode::INTERNAL,
        "unable to autofill for '" + model_name_ +
            "', configuration specified max-batch " +
            std::to_string(config->max_batch_size()) +
            " but model session does not support batching");
  }

  // 'model_support_batching' is set to be true when all model inputs have
  // variable size first dimension, but it is not necessary to be the case
  // (i.e. non-batch model with variable size tensors). As 'max_batch_size == 0'
  // from existing config is also ambiguous, it can be either unspecified or
  // no-batch, autofill will check specified input/output (if any) for hint.
  if (model_support_batching_ && (config->max_batch_size() == 0)) {
    bool config_batch_hint = false;
    if ((config->input_size() != 0) || (config->output_size() != 0)) {
      for (const auto& io : config->input()) {
        if (!io.dims().empty()) {
          // look up corresponding io info from model
          const auto it = input_infos_.find(io.name());
          if (it != input_infos_.end()) {
            bool should_batch =
                (static_cast<int>(it->second.dims_.size()) ==
                 (io.dims_size() + 1));
            // inconsistent hint
            if (config_batch_hint &&
                (model_support_batching_ != should_batch)) {
              return Status(
                  RequestStatusCode::INTERNAL,
                  "unable to autofill for '" + model_name_ +
                      "', model tensor configurations are contradicting " +
                      "each other in terms of whether batching is supported");
            }
            config_batch_hint = true;
            model_support_batching_ = should_batch;
          }
        }
      }
      for (const auto& io : config->output()) {
        if (!io.dims().empty()) {
          // look up corresponding io info from model
          const auto it = output_infos_.find(io.name());
          if (it != output_infos_.end()) {
            bool should_batch =
                (static_cast<int>(it->second.dims_.size()) ==
                 (io.dims_size() + 1));
            // inconsistent hint
            if (config_batch_hint &&
                (model_support_batching_ != should_batch)) {
              return Status(
                  RequestStatusCode::INTERNAL,
                  "unable to autofill for '" + model_name_ +
                      "', model tensor configurations are contradicting " +
                      "each other in terms of whether batching is supported");
            }
            config_batch_hint = true;
            model_support_batching_ = should_batch;
          }
        }
      }
    }
  }

  if (config->max_batch_size() == 0) {
    config->set_max_batch_size(model_support_batching_ ? 1 : 0);
  }
  return Status::Success;
}

Status
AutoFillOnnxImpl::FixInputConfig(ModelConfig* config)
{
  if (config->input_size() == 0) {
    // fill all corresponding i/o tensors
    for (const auto& io_info : input_infos_) {
      ModelInput* config_io = config->add_input();
      SetIOConfig(
          io_info.first, io_info.second, model_support_batching_, config_io);
    }
  } else {
    for (auto& io : *(config->mutable_input())) {
      const auto it = input_infos_.find(io.name());
      if (it != input_infos_.end()) {
        SetIOConfig(it->first, it->second, model_support_batching_, &io);
      }
    }
  }
  return Status::Success;
}

Status
AutoFillOnnxImpl::FixOutputConfig(ModelConfig* config)
{
  if (config->output_size() == 0) {
    // fill all corresponding i/o tensors
    for (const auto& io_info : output_infos_) {
      ModelOutput* config_io = config->add_output();
      SetIOConfig(
          io_info.first, io_info.second, model_support_batching_, config_io);
    }
  } else {
    for (auto& io : *(config->mutable_output())) {
      const auto it = output_infos_.find(io.name());
      if (it != output_infos_.end()) {
        SetIOConfig(it->first, it->second, model_support_batching_, &io);
      }
    }
  }
  return Status::Success;
}

Status
AutoFillOnnxImpl::SetConfigFromOrtSession(
    OrtSession* session, OrtAllocator* allocator)
{
  RETURN_IF_ERROR(InputInfos(session, allocator, input_infos_));
  RETURN_IF_ERROR(OutputInfos(session, allocator, output_infos_));

  RETURN_IF_ERROR(SetBatchingSupport());
  return Status::Success;
}

Status
AutoFillOnnxImpl::SetBatchingSupport()
{
  model_support_batching_ = true;

  // iterate over all input tensors
  for (const auto& io_info : input_infos_) {
    const auto& dims = io_info.second.dims_;
    if ((dims.size() == 0) || (dims[0] != -1)) {
      model_support_batching_ = false;
    }
  }

  // iterate over all output tensors
  for (const auto& io_info : output_infos_) {
    const auto& dims = io_info.second.dims_;
    if ((dims.size() == 0) || (dims[0] != -1)) {
      model_support_batching_ = false;
    }
  }

  return Status::Success;
}

Status
AutoFillOnnx::Create(
    const std::string& model_name, const std::string& model_path,
    std::unique_ptr<AutoFill>* autofill)
{
  std::unique_ptr<AutoFillOnnxImpl> local_autofill;

  std::set<std::string> version_dirs;
  RETURN_IF_ERROR(GetDirectorySubdirs(model_path, &version_dirs));

  if (version_dirs.size() == 0) {
    return Status(
        RequestStatusCode::INTERNAL, "unable to autofill for '" + model_name +
                                         "' due to no version directories");
  }

  // Create resource wrapper to manage release of resource
  OrtSessionOptions* session_options;

  RETURN_IF_ORT_ERROR(OrtCreateSessionOptions(&session_options));

  OrtResourceWrapper<OrtSessionOptions*> options_wrapper(
      session_options, &OrtReleaseSessionOptions);
  RETURN_IF_ORT_ERROR(OrtSetSessionThreadPoolSize(session_options, 1));
  RETURN_IF_ORT_ERROR(OrtSetSessionGraphOptimizationLevel(session_options, ORT_DISABLE_ALL));

  OrtSession* session;

  // All versions should share the same model configuration, thus use the first
  // one that can be loaded successfully.
  Status status;
  bool unsupported_opset = false;
  const std::string opset_error(
      "onnx runtime error " + std::to_string(ORT_NOT_IMPLEMENTED));
  for (const auto& version : version_dirs) {
    const auto version_path = JoinPath({model_path, version});

    // There must be a single onnx file within the version directory...
    std::set<std::string> onnx_files;
    RETURN_IF_ERROR(GetDirectoryFiles(version_path, &onnx_files));
    if (onnx_files.size() != 1) {
      return Status(
          RequestStatusCode::INTERNAL, "unable to autofill for '" + model_name +
                                           "', unable to find onnx file");
    }

    const std::string onnx_file = *(onnx_files.begin());
    const auto onnx_path = JoinPath({version_path, onnx_file});

    // Load session
    std::string onnx_file_content;
    RETURN_IF_ERROR(ReadTextFile(onnx_path, &onnx_file_content));
    status =
        OnnxLoader::LoadSession(onnx_file_content, session_options, &session);

    if (status.IsOk()) {
      local_autofill.reset(new AutoFillOnnxImpl(model_name, onnx_file));
      break;
    } else if (
        (status.Message().compare(0, opset_error.size(), opset_error)) == 0) {
      local_autofill.reset(new AutoFillOnnxImpl(model_name, onnx_file));
      unsupported_opset = true;
      // no break in case there is a valid version
    }
  }

  // If it is due to unsupported opset, return success with limited autofill
  // capability
  if (!status.IsOk() && unsupported_opset) {
    *autofill = std::move(local_autofill);
    return Status::Success;
  }

  // Return if none of the version can be loaded successfully
  // due to reasons other than unsupported opset
  RETURN_IF_ERROR(status);

  OrtAllocator* allocator;
  OrtStatus* ort_status = OrtGetAllocatorWithDefaultOptions(&allocator);

  if (ort_status == nullptr) {
    status = local_autofill->SetConfigFromOrtSession(session, allocator);
  }
  OnnxLoader::UnloadSession(session);

  RETURN_IF_ORT_ERROR(ort_status);
  RETURN_IF_ERROR(status);

  *autofill = std::move(local_autofill);
  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
