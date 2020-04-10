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
#pragma once

#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

/// Get the integral version from a string, or fail if string does not
/// represent a valid version.
/// \param version_string The string version.
/// \param version Returns the integral version.
/// \return The error status. Failure if 'version_string' doesn't
/// convert to valid version.
Status GetModelVersionFromString(
    const std::string& version_string, int64_t* version);

/// Get version of a model from the path containing the model
/// definition file.
/// \param path The path to the model definition file.
/// \param version Returns the version.
/// \return The error status.
Status GetModelVersionFromPath(const std::string& path, int64_t* version);

/// Get the tensor name, false value, and true value for a boolean
/// sequence batcher control kind. If 'required' is true then must
/// find a tensor for the control. If 'required' is false, return
/// 'tensor_name' as empty-string if the control is not mapped to any
/// tensor.
Status GetBooleanSequenceControlProperties(
    const ModelSequenceBatching& batcher, const std::string& model_name,
    const ModelSequenceBatching::Control::Kind control_kind,
    const bool required, std::string* tensor_name, DataType* tensor_datatype,
    float* fp32_false_value, float* fp32_true_value, int32_t* int32_false_value,
    int32_t* int32_true_value);

/// Get the tensor name and datatype for a non-boolean sequence
/// batcher control kind. If 'required' is true then must find a
/// tensor for the control. If 'required' is false, return
/// 'tensor_name' as empty-string if the control is not mapped to any
/// tensor. 'tensor_datatype' returns the required datatype for the
/// control.
Status GetTypedSequenceControlProperties(
    const ModelSequenceBatching& batcher, const std::string& model_name,
    const ModelSequenceBatching::Control::Kind control_kind,
    const bool required, std::string* tensor_name, DataType* tensor_datatype);

/// Read a ModelConfig and normalize it as expected by model backends.
/// \param path The full-path to the directory containing the
/// model configuration.
/// \param backend_config_map Map from platform name to the backend
/// configuration for that platform.
/// \param autofill If true attempt to determine any missing required
/// configuration from the model definition.
/// \param min_compute_capability The minimum support CUDA compute
/// capability.
/// \param config Returns the normalized model configuration.
/// \return The error status.
Status GetNormalizedModelConfig(
    const std::string& path, const BackendConfigMap& backend_config_map,
    const bool autofill, const double min_compute_capability,
    ModelConfig* config);

/// Validate that a model is specified correctly.
/// \param config The model configuration to validate.
/// \param expected_platform If non-empty the model will be checked
/// to make sure its platform matches this value.
/// \param min_compute_capability The minimum support CUDA compute
/// capability.
/// \return The error status. A non-OK status indicates the configuration
/// is not valid.
Status ValidateModelConfig(
    const ModelConfig& config, const std::string& expected_platform,
    const double min_compute_capability);

/// Validate that input is specified correctly in a model
/// configuration.
/// \param io The model input.
/// \param max_batch_size The max batch size specified in model configuration.
/// \param platform The platform name
/// \return The error status. A non-OK status indicates the input
/// is not valid.
Status ValidateModelInput(
    const ModelInput& io, int32_t max_batch_size, const std::string& platform);

/// Validate that an input matches one of the allowed input names.
/// \param io The model input.
/// \param allowed The set of allowed input names.
/// \return The error status. A non-OK status indicates the input
/// is not valid.
Status CheckAllowedModelInput(
    const ModelInput& io, const std::set<std::string>& allowed);

/// Validate that an output is specified correctly in a model
/// configuration.
/// \param io The model output.
/// \param max_batch_size The max batch size specified in model configuration.
/// \param platform The platform name
/// \return The error status. A non-OK status indicates the output
/// is not valid.
Status ValidateModelOutput(
    const ModelOutput& io, int32_t max_batch_size, const std::string& platform);

/// Validate that an output matches one of the allowed output names.
/// \param io The model output.
/// \param allowed The set of allowed output names.
/// \return The error status. A non-OK status indicates the output
/// is not valid.
Status CheckAllowedModelOutput(
    const ModelOutput& io, const std::set<std::string>& allowed);

/// Parse the 'value' of the parameter 'key' into a boolean value.
/// \param key The name of the parameter.
/// \param value The value of the parameter in string.
/// \param parsed_value Return the boolean of the parameter.
/// \return The error status. A non-OK status indicates failure on parsing the
/// value.
Status ParseBoolParameter(
    const std::string& key, std::string value, bool* parsed_value);

/// Parse the 'value' of the parameter 'key' into a long long integer value.
/// \param key The name of the parameter.
/// \param value The value of the parameter in string.
/// \param parsed_value Return the numerical value of the parameter.
/// \return The error status. A non-OK status indicates failure on parsing the
/// value.
Status ParseLongLongParameter(
    const std::string& key, const std::string& value, int64_t* parsed_value);

/// Obtain the 'profile_index' of the 'profile_name'.
/// \param profile_name The name of the profile.
/// \param profile_index Return the index of the profile.
/// \return The error status. A non-OK status indicates failure on getting the
/// value.
Status GetProfileIndex(const std::string& profile_name, int* profile_index);

}}  // namespace nvidia::inferenceserver
