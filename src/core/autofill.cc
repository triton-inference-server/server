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

#include <set>
#include "src/core/constants.h"
#include "src/core/filesystem.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"

namespace nvidia { namespace inferenceserver {

Status
AutoFill::Fix(
    const std::string& model_name, const std::string& model_path,
    inference::ModelConfig* config)
{
  std::set<std::string> version_dirs;
  RETURN_IF_ERROR(GetDirectorySubdirs(model_path, &version_dirs));

  // There must be at least one version directory that we can inspect to
  // attempt to determine the platform. If not, we skip autofill with file name.
  // For now we allow multiple versions and only inspect the first verison
  // directory to ensure it is valid. We can add more aggressive checks later.
  const bool has_version = (version_dirs.size() != 0);
  const auto version_path =
      has_version ? JoinPath({model_path, *(version_dirs.begin())}) : "";
  std::set<std::string> version_dir_content;
  if (has_version) {
    RETURN_IF_ERROR(GetDirectoryContents(version_path, &version_dir_content));
  }

  if (config->name().empty()) {
    config->set_name(model_name);
  }

  // Trying to fill the 'backend', 'default_model_filename' field.
#ifdef TRITON_ENABLE_TENSORFLOW
  // For TF backend, the platform is required
  if (config->platform().empty()) {
    // Check 'backend', 'default_model_filename', and the actual directory
    // to determine the platform
    if (config->backend().empty() ||
        (config->backend() == kTensorFlowBackend)) {
      if (config->default_model_filename() == kTensorFlowSavedModelFilename) {
        config->set_platform(kTensorFlowSavedModelPlatform);
      } else if (
          config->default_model_filename() == kTensorFlowGraphDefFilename) {
        config->set_platform(kTensorFlowGraphDefPlatform);
      } else if (config->default_model_filename().empty() && has_version) {
        bool is_dir = false;
        if (version_dir_content.find(kTensorFlowSavedModelFilename) !=
            version_dir_content.end()) {
          RETURN_IF_ERROR(IsDirectory(
              JoinPath({version_path, kTensorFlowSavedModelFilename}),
              &is_dir));
          if (is_dir) {
            config->set_platform(kTensorFlowSavedModelPlatform);
          }
        }
        if (version_dir_content.find(kTensorFlowGraphDefFilename) !=
            version_dir_content.end()) {
          RETURN_IF_ERROR(IsDirectory(
              JoinPath({version_path, kTensorFlowGraphDefFilename}), &is_dir));
          if (!is_dir) {
            config->set_platform(kTensorFlowGraphDefPlatform);
          }
        }
      }
    }
  }
  // Fill 'backend' and 'default_model_filename' if missing
  if ((config->platform() == kTensorFlowSavedModelPlatform) ||
      (config->platform() == kTensorFlowGraphDefPlatform)) {
    if (config->backend().empty()) {
      config->set_backend(kTensorFlowBackend);
    }
    if (config->default_model_filename().empty()) {
      if (config->platform() == kTensorFlowSavedModelPlatform) {
        config->set_default_model_filename(kTensorFlowSavedModelFilename);
      } else {
        config->set_default_model_filename(kTensorFlowGraphDefFilename);
      }
    }
    return Status::Success;
  }
#endif  // TRITON_ENABLE_TENSORFLOW
#ifdef TRITON_ENABLE_TENSORRT
  if (config->backend().empty()) {
    if ((config->platform() == kTensorRTPlanPlatform) ||
        (config->default_model_filename() == kTensorRTPlanFilename)) {
      config->set_backend(kTensorRTBackend);
    } else if (
        config->platform().empty() &&
        config->default_model_filename().empty() && has_version) {
      bool is_dir = false;
      if (version_dir_content.find(kTensorRTPlanFilename) !=
          version_dir_content.end()) {
        RETURN_IF_ERROR(IsDirectory(
            JoinPath({version_path, kTensorRTPlanFilename}), &is_dir));
        if (!is_dir) {
          config->set_backend(kTensorRTBackend);
        }
      }
    }
  }
  if (config->backend() == kTensorRTBackend) {
    if (config->platform().empty()) {
      config->set_platform(kTensorRTPlanPlatform);
    }
    if (config->default_model_filename().empty()) {
      config->set_default_model_filename(kTensorRTPlanFilename);
    }
    return Status::Success;
  }
#endif  // TRITON_ENABLE_TENSORRT
#ifdef TRITON_ENABLE_ONNXRUNTIME
  if (config->backend().empty()) {
    if ((config->platform() == kOnnxRuntimeOnnxPlatform) ||
        (config->default_model_filename() == kOnnxRuntimeOnnxFilename)) {
      config->set_backend(kOnnxRuntimeBackend);
    } else if (
        config->platform().empty() &&
        config->default_model_filename().empty() && has_version) {
      bool is_dir = false;
      if (version_dir_content.find(kOnnxRuntimeOnnxFilename) !=
          version_dir_content.end()) {
        RETURN_IF_ERROR(IsDirectory(
            JoinPath({version_path, kOnnxRuntimeOnnxFilename}), &is_dir));
        if (!is_dir) {
          config->set_backend(kOnnxRuntimeBackend);
        }
      }
    }
  }
  if (config->backend() == kOnnxRuntimeBackend) {
    if (config->platform().empty()) {
      config->set_platform(kOnnxRuntimeOnnxPlatform);
    }
    if (config->default_model_filename().empty()) {
      config->set_default_model_filename(kOnnxRuntimeOnnxFilename);
    }
    return Status::Success;
  }
#endif  // TRITON_ENABLE_ONNXRUNTIME
#ifdef TRITON_ENABLE_PYTORCH
  if (config->backend().empty()) {
    if ((config->platform() == kPyTorchLibTorchPlatform) ||
        (config->default_model_filename() == kPyTorchLibTorchFilename)) {
      config->set_backend(kPyTorchBackend);
    } else if (
        config->platform().empty() &&
        config->default_model_filename().empty() && has_version) {
      bool is_dir = false;
      if (version_dir_content.find(kPyTorchLibTorchFilename) !=
          version_dir_content.end()) {
        RETURN_IF_ERROR(IsDirectory(
            JoinPath({version_path, kPyTorchLibTorchFilename}), &is_dir));
        if (!is_dir) {
          config->set_backend(kPyTorchBackend);
        }
      }
    }
  }
  if (config->backend() == kPyTorchBackend) {
    if (config->platform().empty()) {
      config->set_platform(kPyTorchLibTorchPlatform);
    }
    if (config->default_model_filename().empty()) {
      config->set_default_model_filename(kPyTorchLibTorchFilename);
    }
    return Status::Success;
  }
#endif  // TRITON_ENABLE_PYTORCH
  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
