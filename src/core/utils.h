// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include "src/core/model_config.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace nvidia { namespace inferenceserver {

/// Get version of a model from the path containing the model
/// definition file.
/// \param path The path to the model definition file.
/// \param version Returns the version.
/// \return The error status.
tensorflow::Status GetModelVersionFromPath(
    const tensorflow::StringPiece& path, int64_t* version);

/// Read a ModelConfig and normalize it as expected by model servables.
/// \param path The full-path to the directory containing the
/// model configuration.
/// \param autofill If true attempt to determine any missing required
/// configuration from the model definition.
/// \param config Returns the normalized model configuration.
/// \return The error status.
tensorflow::Status GetNormalizedModelConfig(
    const tensorflow::StringPiece& path, const bool autofill,
    ModelConfig* config);

/// Validate that a model is specified correctly (excluding inputs and
/// outputs which are validated via ValidateModelInput() and
/// ValidateModelOutput()).
/// \param config The model configuration to validate.
/// \param expected_platform If non-empty the model will be checked
/// to make sure its platform matches this value.
/// \return The error status. A non-OK status indicates the configuration
/// is not valid.
tensorflow::Status ValidateModelConfig(
    const ModelConfig& config, const std::string& expected_platform);

/// Validate that input is specified correctly in a model
/// configuration.
/// \param io The model input.
/// \return The error status. A non-OK status indicates the input
/// is not valid.
tensorflow::Status ValidateModelInput(const ModelInput& io);

/// Validate that an input is specified correctly in a model
/// configuration and matches one of the allowed input names.
/// \param io The model input.
/// \param allowed The set of allowed input names.
/// \return The error status. A non-OK status indicates the input
/// is not valid.
tensorflow::Status ValidateModelInput(
    const ModelInput& io, const std::set<std::string>& allowed);

/// Validate that an output is specified correctly in a model
/// configuration.
/// \param io The model output.
/// \return The error status. A non-OK status indicates the output
/// is not valid.
tensorflow::Status ValidateModelOutput(const ModelOutput& io);

/// Validate that an output is specified correctly in a model
/// configuration and matches one of the allowed output names.
/// \param io The model output.
/// \param allowed The set of allowed output names.
/// \return The error status. A non-OK status indicates the output
/// is not valid.
tensorflow::Status ValidateModelOutput(
    const ModelOutput& io, const std::set<std::string>& allowed);

}}  // namespace nvidia::inferenceserver
