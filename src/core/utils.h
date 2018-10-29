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

// Get version of a model from the path containing the model file.
tensorflow::Status GetModelVersionFromPath(
  const tensorflow::StringPiece& path, uint32_t* version);

// Read a ModelConfig and normalize it as expected by model servables.
// 'path' should be the full-path to the directory containing the
// model configuration. If 'autofill' then attempt to determine any
// missing required configuration from the model definition.
tensorflow::Status GetNormalizedModelConfig(
  const tensorflow::StringPiece& path, const bool autofill,
  ModelConfig* config);

// Validate that a model is specified correctly (excluding inputs and
// outputs which are validated via ValidateModelInput() and
// ValidateModelOutput()). If 'expected_platform' is non-empty the
// model will be checked to make sure it matches that platform.
tensorflow::Status ValidateModelConfig(
  const ModelConfig& config, const std::string& expected_platform);

// Validate that input is specified correctly in a model
// configuration. 'allowed' is the set of allowed input names, or
// empty if inputs names should not be checked against an allowed set.
tensorflow::Status ValidateModelInput(const ModelInput& io);
tensorflow::Status ValidateModelInput(
  const ModelInput& io, const std::set<std::string>& allowed);

// Validate that output is specified correctly in a model
// configuration. 'allowed' is the set of allowed output names, or
// empty if outputs names should not be checked against an allowed
// set.
tensorflow::Status ValidateModelOutput(const ModelOutput& io);
tensorflow::Status ValidateModelOutput(
  const ModelOutput& io, const std::set<std::string>& allowed);

}}  // namespace nvidia::inferenceserver
