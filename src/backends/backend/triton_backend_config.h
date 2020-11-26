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
#pragma once

#include "src/core/model_config.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

/// Get a key's string value from a backend configuration.
Status BackendConfiguration(
    const BackendCmdlineConfig& config, const std::string& key,
    std::string* val);

/// Convert a backend configuration string  value into a double.
Status BackendConfigurationParseStringToDouble(
    const std::string& str, double* val);

/// Convert a backend configuration string  value into a bool.
Status BackendConfigurationParseStringToBool(const std::string& str, bool* val);

/// Get the global backends directory from the backend configuration.
Status BackendConfigurationGlobalBackendsDirectory(
    const BackendCmdlineConfigMap& config_map, std::string* dir);

/// Get the minimum compute capability from the backend configuration.
Status BackendConfigurationMinComputeCapability(
    const BackendCmdlineConfigMap& config_map, double* mcc);

/// Get the model configuration auto-complete setting from the backend
/// configuration.
Status BackendConfigurationAutoCompleteConfig(
    const BackendCmdlineConfigMap& config_map, bool* acc);

/// Convert a backend name to the specialized version of that name
/// based on the backend configuration. For example, "tensorflow" will
/// convert to either "tensorflow1" or "tensorflow2" depending on how
/// tritonserver is run.
Status BackendConfigurationSpecializeBackendName(
    const BackendCmdlineConfigMap& config_map, const std::string& backend_name,
    std::string* specialized_name);

/// Return the shared library name for a backend.
Status BackendConfigurationBackendLibraryName(
    const std::string& backend_name, std::string* libname);

}}  // namespace nvidia::inferenceserver
