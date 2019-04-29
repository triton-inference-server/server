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

#include "src/servables/tensorrt/autofill.h"

#include <NvInfer.h>
#include "src/core/autofill.h"
#include "src/core/constants.h"
#include "src/core/filesystem.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/servables/tensorrt/loader.h"
#include "src/servables/tensorrt/plan_utils.h"

namespace nvidia { namespace inferenceserver {

class AutoFillPlanImpl : public AutoFill {
 public:
  AutoFillPlanImpl(
      const std::string& model_name, const std::string& plan_filename,
      const int32_t max_batch_size, const ModelConfig& config)
      : AutoFill(model_name), plan_filename_(plan_filename),
        max_batch_size_(max_batch_size), config_(config)
  {
  }

  Status Fix(ModelConfig* config) override;

 private:
  const std::string plan_filename_;
  const int32_t max_batch_size_;
  const ModelConfig config_;
};

Status
AutoFillPlanImpl::Fix(ModelConfig* config)
{
  config->set_platform(kTensorRTPlanPlatform);

  // Set name if not already set.
  if (config->name().empty()) {
    config->set_name(model_name_);
  }

  if (config->default_model_filename().empty()) {
    config->set_default_model_filename(plan_filename_);
  }

  if (config->max_batch_size() == 0) {
    config->set_max_batch_size(max_batch_size_);
  }

  // Inputs
  if (config->input().size() == 0) {
    config->mutable_input()->CopyFrom(config_.input());
  }

  // Outputs
  if (config->output().size() == 0) {
    config->mutable_output()->CopyFrom(config_.output());
  }

  return Status::Success;
}

Status
AutoFillPlan::Create(
    const std::string& model_name, const std::string& model_path,
    std::unique_ptr<AutoFill>* autofill)
{
  std::set<std::string> version_dirs;
  RETURN_IF_ERROR(GetDirectorySubdirs(model_path, &version_dirs));

  // There must be at least one version directory that we can inspect
  // to attempt to determine the platform. For now we only handle the
  // case where there is one version directory.
  if (version_dirs.size() != 1) {
    return Status(
        RequestStatusCode::INTERNAL,
        "unable to autofill for '" + model_name + "' due to multiple versions");
  }

  const auto version_path = JoinPath({model_path, *(version_dirs.begin())});

  // There must be a single plan file within the version directory...
  std::set<std::string> plan_files;
  RETURN_IF_ERROR(GetDirectoryFiles(version_path, &plan_files));
  if (plan_files.size() != 1) {
    return Status(
        RequestStatusCode::INTERNAL, "unable to autofill for '" + model_name +
                                         "', unable to find plan file");
  }

  const std::string plan_file = *(plan_files.begin());
  const auto plan_path = JoinPath({version_path, plan_file});

  std::string plan_data_str;
  RETURN_IF_ERROR(ReadTextFile(plan_path, &plan_data_str));
  std::vector<char> plan_data(plan_data_str.begin(), plan_data_str.end());

  nvinfer1::IRuntime* runtime = nullptr;
  nvinfer1::ICudaEngine* engine = nullptr;
  if (!LoadPlan(plan_data, &runtime, &engine).IsOk()) {
    if (engine != nullptr) {
      engine->destroy();
    }
    if (runtime != nullptr) {
      runtime->destroy();
    }
    return Status(
        RequestStatusCode::INTERNAL,
        "unable to autofill for '" + model_name +
            "', unable to create TensorRT runtime and engine");
  }

  const int32_t max_batch_size = engine->getMaxBatchSize();

  // Inputs and outputs.
  ModelConfig config;
  for (int i = 0; i < engine->getNbBindings(); ++i) {
    if (engine->bindingIsInput(i)) {
      ModelInput* config_input = config.add_input();
      config_input->set_name(engine->getBindingName(i));
      config_input->set_data_type(
          ConvertDatatype(engine->getBindingDataType(i)));
      nvinfer1::Dims dims = engine->getBindingDimensions(i);
      for (int didx = 0; didx < dims.nbDims; ++didx) {
        config_input->mutable_dims()->Add(dims.d[didx]);
      }
    } else {
      ModelOutput* config_output = config.add_output();
      config_output->set_name(engine->getBindingName(i));
      config_output->set_data_type(
          ConvertDatatype(engine->getBindingDataType(i)));
      nvinfer1::Dims dims = engine->getBindingDimensions(i);
      for (int didx = 0; didx < dims.nbDims; ++didx) {
        config_output->mutable_dims()->Add(dims.d[didx]);
      }
    }
  }

  engine->destroy();
  runtime->destroy();

  autofill->reset(
      new AutoFillPlanImpl(model_name, plan_file, max_batch_size, config));
  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
