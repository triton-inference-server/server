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

#include "src/core/autofill.h"

#ifdef TRITON_ENABLE_TENSORFLOW
#include "src/backends/tensorflow/autofill.h"
#endif  // TRITON_ENABLE_TENSORFLOW
#ifdef TRITON_ENABLE_TENSORRT
#include "src/backends/tensorrt/autofill.h"
#endif  // TRITON_ENABLE_TENSORRT
#ifdef TRITON_ENABLE_ONNXRUNTIME
#include "src/backends/onnx/autofill.h"
#endif  // TRITON_ENABLE_ONNXRUNTIME
#ifdef TRITON_ENABLE_PYTORCH
#include "src/backends/pytorch/autofill.h"
#endif  // TRITON_ENABLE_PYTORCH
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"

namespace nvidia { namespace inferenceserver {

//
// AutoFillNull
//
class AutoFillNull : public AutoFill {
 public:
  static Status Create(std::unique_ptr<AutoFill>* autofill);
  Status Fix(inference::ModelConfig* config);

 private:
  AutoFillNull() : AutoFill(std::string()) {}
};

Status
AutoFillNull::Create(std::unique_ptr<AutoFill>* autofill)
{
  autofill->reset(new AutoFillNull);
  return Status::Success;
}

Status
AutoFillNull::Fix(inference::ModelConfig* config)
{
  return Status::Success;
}

//
// AutoFillSimple
//
class AutoFillSimple : public AutoFill {
 public:
  static Status Create(
      const std::string& model_name, std::unique_ptr<AutoFill>* autofill);
  Status Fix(inference::ModelConfig* config);

 private:
  AutoFillSimple(const std::string& model_name) : AutoFill(model_name) {}
};

Status
AutoFillSimple::Create(
    const std::string& model_name, std::unique_ptr<AutoFill>* autofill)
{
  autofill->reset(new AutoFillSimple(model_name));
  return Status::Success;
}

Status
AutoFillSimple::Fix(inference::ModelConfig* config)
{
  // Set name if not already set.
  if (config->name().empty()) {
    config->set_name(model_name_);
  }

  return Status::Success;
}

//
// AutoFill
//
Status
AutoFill::Create(
    const std::string& model_name, const BackendConfigMap& backend_config_map,
    const std::string& model_path, const inference::ModelConfig& config,
    std::unique_ptr<AutoFill>* autofill)
{
  autofill->reset();

  // If the config specifies a platform use it to create the
  // appropriate autofill object, otherwise just try creating each
  // autofill object to see if one can detect the platform.
#if defined(TRITON_ENABLE_TENSORFLOW) || defined(TRITON_ENABLE_TENSORRT) || \
    defined(TRITON_ENABLE_ONNXRUNTIME) || defined(TRITON_ENABLE_PYTORCH)
  const Platform platform = GetPlatform(config.platform());
  const BackendType backend_type = GetBackendType(config.backend());
  bool unknown_model =
      ((platform == Platform::PLATFORM_UNKNOWN) &&
       (backend_type == BackendType::BACKEND_TYPE_UNKNOWN));
#endif

  Status status;

#ifdef TRITON_ENABLE_TENSORFLOW
  if ((platform == Platform::PLATFORM_TENSORFLOW_SAVEDMODEL) ||
      (backend_type == BackendType::BACKEND_TYPE_TENSORFLOW) || unknown_model) {
    // FIXME drop the AutoFillXXX once autofill for all backends is merely
    // filling platform / backend
    std::unique_ptr<AutoFill> afsm;
    std::shared_ptr<BackendConfig> backend_config;
    status = AutoFillSavedModel::Create(
        model_name, backend_config, model_path, &afsm);
    LOG_VERBOSE(1) << "TensorFlow SavedModel autofill: " << status.AsString();
    if (status.IsOk()) {
      *autofill = std::move(afsm);
      return Status::Success;
    }
  }

  if ((platform == Platform::PLATFORM_TENSORFLOW_GRAPHDEF) ||
      (backend_type == BackendType::BACKEND_TYPE_TENSORFLOW) || unknown_model) {
    // FIXME drop the AutoFillXXX once autofill for all backends is merely
    // filling platform / backend
    std::unique_ptr<AutoFill> afgd;
    status = AutoFillGraphDef::Create(model_name, model_path, &afgd);
    LOG_VERBOSE(1) << "TensorFlow GraphDef autofill: " << status.AsString();
    if (status.IsOk()) {
      *autofill = std::move(afgd);
      return Status::Success;
    }
  }
#endif  // TRITON_ENABLE_TENSORFLOW

#ifdef TRITON_ENABLE_PYTORCH
  if ((platform == Platform::PLATFORM_PYTORCH_LIBTORCH) ||
      (backend_type == BackendType::BACKEND_TYPE_PYTORCH) || unknown_model) {
    std::unique_ptr<AutoFill> afpt;
    status = AutoFillPyTorch::Create(model_name, model_path, &afpt);
    LOG_VERBOSE(1) << "PyTorch autofill: " << status.AsString();
    if (status.IsOk()) {
      *autofill = std::move(afpt);
      return Status::Success;
    }
  }
#endif  // TRITON_ENABLE_PYTORCH

#ifdef TRITON_ENABLE_ONNXRUNTIME
  // Check for ONNX model must be done before check for TensorRT plan
  // because TensorRT deserializeCudaEngine() function will cause program
  // to exit when it tries to deserialize an ONNX model.
  // However this is not bulletproof as ONNX Runtime does not support
  // ONNX models with opset < 8, thus under AutoFillOnnx class, there
  // is additional check on reason of loading failure.
  //
  // [TODO] remove additional checking once TensorRT provides
  // an elegent way to handle passing incorrect model format (i.e. ONNX model)
  // to deserializeCudaEngine()
  if ((platform == Platform::PLATFORM_ONNXRUNTIME_ONNX) ||
      (backend_type == BackendType::BACKEND_TYPE_ONNXRUNTIME) ||
      unknown_model) {
    std::unique_ptr<AutoFill> afox;

    // If model operations is specified, use it to set the session options for
    // ONNX.
    auto model_ops = config.model_operations();
    std::vector<std::string> op_libraries;
    for (const auto& lib_filename : model_ops.op_library_filename()) {
      op_libraries.push_back(lib_filename);
    }
    status = AutoFillOnnx::Create(model_name, model_path, &afox, op_libraries);
    LOG_VERBOSE(1) << "ONNX autofill: " << status.AsString();
    if (status.IsOk()) {
      *autofill = std::move(afox);
      return Status::Success;
    }
  }
#endif  // TRITON_ENABLE_ONNXRUNTIME

#ifdef TRITON_ENABLE_TENSORRT
  if ((platform == Platform::PLATFORM_TENSORRT_PLAN) ||
      (backend_type == BackendType::BACKEND_TYPE_TENSORRT) || unknown_model) {
    std::unique_ptr<AutoFill> afp;
    status = AutoFillPlan::Create(model_name, model_path, &afp);
    LOG_VERBOSE(1) << "TensorRT autofill: " << status.AsString();
    if (status.IsOk()) {
      *autofill = std::move(afp);
      return Status::Success;
    }
  }
#endif  // TRITON_ENABLE_TENSORRT

  // Unable to determine the platform so just use the simple autofill,
  // or null if that fails.
  {
#if defined(TRITON_ENABLE_TENSORFLOW) || defined(TRITON_ENABLE_TENSORRT) || \
    defined(TRITON_ENABLE_ONNXRUNTIME) || defined(TRITON_ENABLE_PYTORCH)
    bool print_warning = true;
    if (!LOG_VERBOSE_IS_ON(1)) {
      if (platform == Platform::PLATFORM_UNKNOWN) {
        LOG_WARNING << "Autofiller failed to detect the platform for "
                    << model_name
                    << " (verify contents of model directory or use "
                       "--log-verbose=1 for more details)";
      } else {
#ifdef TRITON_ENABLE_ENSEMBLE
        if (platform == Platform::PLATFORM_ENSEMBLE) {
          print_warning = false;
        }
#endif

        if (print_warning) {
          LOG_WARNING << "Autofiller failed to retrieve model. Error Details: "
                      << status.AsString();
        }
      }
    }
    if (print_warning) {
      LOG_WARNING << "Proceeding with simple config for now";
    }
#endif
    std::unique_ptr<AutoFill> afs;
    status = AutoFillSimple::Create(model_name, &afs);
    if (status.IsOk()) {
      *autofill = std::move(afs);
    } else {
      std::unique_ptr<AutoFill> afn;
      RETURN_IF_ERROR(AutoFillNull::Create(&afn));
      *autofill = std::move(afn);
    }
  }

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
