// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include <string>
#include <unordered_map>

#include "src/backends/custom/custom.h"
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/custom/sdk/error_codes.h"

namespace nvidia { namespace inferenceserver { namespace custom {

//==============================================================================
/// Base class for custom backend instances. CustomInstance is responsible for
/// the state of the instance, provide a C++ wrapper around the C-API, and
/// provide common helper functions that can be used for initialization and
/// execution.
///
class CustomInstance {
 public:
  /// Create a custom backend. This static method is declared here, but must be
  /// defined in the child custom class. This create method is used
  /// CustomInitialization C-API method.
  ///
  /// \param instance A return pointer for the custom instance object created by
  /// this method. Note that the custom object pointer should be returned even
  /// in the case of failure, so that the error string associated to the custom
  /// backend can be retrieved.
  /// \param name The name of the custom instance
  /// \param model_config The model configuration
  /// \param gpu_device The GPU device ID
  /// \return Error code indicating success or the type of failure
  static int Create(
      CustomInstance** instance, const std::string& name,
      const ModelConfig& model_config, int gpu_device,
      const CustomInitializeData* data);

  virtual ~CustomInstance() = default;

  /// Execute the custom instance. User should override this function if
  /// version 1 of the custom interface is used or the version is not specified.
  ///
  /// \param payload_cnt The number of payloads to execute.
  /// \param payloads The payloads to execute.
  /// \param input_fn The callback function to get tensor input (see
  /// CustomGetNextInputFn_t).
  /// \param output_fn The callback function to get buffer for tensor
  /// output (see CustomGetOutputFn_t).
  /// \return Error code indicating success or the type of failure
  virtual int Execute(
      const uint32_t payload_cnt, CustomPayload* payloads,
      CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
  {
    return ErrorCodes::InvalidInvocationV1;
  }

  /// Execute the custom instance. User should override this function
  /// if version 2 of the custom interface is used.
  ///
  /// \param payload_cnt The number of payloads to execute.
  /// \param payloads The payloads to execute.
  /// \param input_fn The callback function to get tensor input (see
  /// CustomGetNextInputV2Fn_t).
  /// \param output_fn The callback function to get buffer for tensor
  /// output (see CustomGetOutputV2Fn_t).
  /// \return Error code indicating success or the type of failure
  virtual int Execute(
      const uint32_t payload_cnt, CustomPayload* payloads,
      CustomGetNextInputV2Fn_t input_fn, CustomGetOutputV2Fn_t output_fn)
  {
    return ErrorCodes::InvalidInvocationV2;
  }

  /// Get the string for an error code.
  ///
  /// /param error Error code returned by a CustomInstance function
  /// /return Descriptive error message for a specific error code.
  inline const char* ErrorString(uint32_t error) const
  {
    return errors_.ErrorString(error);
  }

 protected:
  /// Base constructor for CustomInstance
  ///
  /// \param name The name of the custom instance
  /// \param model_config The model configuration
  /// \param gpu_device The GPU device ID
  /// \return Error code indicating success or the type of failure
  CustomInstance(
      const std::string& instance_name, const ModelConfig& model_config,
      int gpu_device);

  /// Register a custom error and error message.
  ///
  /// \param error_message A descriptive error message string
  /// \return The unique error code registered to this error message
  inline int RegisterError(const std::string& error_message)
  {
    return errors_.RegisterError(error_message);
  }

  /// The name of this backend instance
  const std::string instance_name_;

  /// The model configuration
  const ModelConfig model_config_;

  /// The GPU device ID to execute on or CUSTOM_NO_GPU_DEVICE if should
  /// execute on CPU.
  const int gpu_device_;

 private:
  /// Error code manager.
  ErrorCodes errors_{};
};

}}}  // namespace nvidia::inferenceserver::custom
