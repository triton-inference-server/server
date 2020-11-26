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

#include "src/backends/backend/triton_backend_config.h"

#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

Status
BackendConfiguration(
    const BackendCmdlineConfig& config, const std::string& key,
    std::string* val)
{
  for (const auto& pr : config) {
    if (pr.first == key) {
      *val = pr.second;
      return Status::Success;
    }
  }

  return Status(
      Status::Code::INTERNAL,
      std::string("unable to find common backend configuration for '") + key +
          "'");
}

Status
BackendConfigurationParseStringToDouble(const std::string& str, double* val)
{
  try {
    *val = std::stod(str);
  }
  catch (...) {
    return Status(
        Status::Code::INTERNAL,
        "unable to parse common backend configuration as double");
  }

  return Status::Success;
}

Status
BackendConfigurationParseStringToBool(const std::string& str, bool* val)
{
  try {
    std::string lowercase_str{str};
    std::transform(
        lowercase_str.begin(), lowercase_str.end(), lowercase_str.begin(),
        [](unsigned char c) { return std::tolower(c); });
    *val = (lowercase_str == "true");
  }
  catch (...) {
    return Status(
        Status::Code::INTERNAL,
        "unable to parse common backend configuration as bool");
  }

  return Status::Success;
}

Status
BackendConfigurationGlobalBackendsDirectory(
    const BackendCmdlineConfigMap& config_map, std::string* dir)
{
  const auto& itr = config_map.find(std::string());
  if (itr == config_map.end()) {
    return Status(
        Status::Code::INTERNAL,
        "unable to find global backends directory configuration");
  }

  RETURN_IF_ERROR(BackendConfiguration(itr->second, "backend-directory", dir));

  return Status::Success;
}

Status
BackendConfigurationMinComputeCapability(
    const BackendCmdlineConfigMap& config_map, double* mcc)
{
#ifdef TRITON_ENABLE_GPU
  *mcc = TRITON_MIN_COMPUTE_CAPABILITY;
#else
  *mcc = 0;
#endif  // TRITON_ENABLE_GPU

  const auto& itr = config_map.find(std::string());
  if (itr == config_map.end()) {
    return Status(
        Status::Code::INTERNAL, "unable to find common backend configuration");
  }

  std::string min_compute_capability_str;
  RETURN_IF_ERROR(BackendConfiguration(
      itr->second, "min-compute-capability", &min_compute_capability_str));
  RETURN_IF_ERROR(
      BackendConfigurationParseStringToDouble(min_compute_capability_str, mcc));

  return Status::Success;
}

Status
BackendConfigurationAutoCompleteConfig(
    const BackendCmdlineConfigMap& config_map, bool* acc)
{
  const auto& itr = config_map.find(std::string());
  if (itr == config_map.end()) {
    return Status(
        Status::Code::INTERNAL, "unable to find auto-complete configuration");
  }

  std::string auto_complete_config_str;
  RETURN_IF_ERROR(BackendConfiguration(
      itr->second, "auto-complete-config", &auto_complete_config_str));
  RETURN_IF_ERROR(
      BackendConfigurationParseStringToBool(auto_complete_config_str, acc));

  return Status::Success;
}

Status
BackendConfigurationSpecializeBackendName(
    const BackendCmdlineConfigMap& config_map, const std::string& backend_name,
    std::string* specialized_name)
{
  *specialized_name = backend_name;
  if (backend_name == "tensorflow") {
    std::string tf_version_str = "1";
    const auto& itr = config_map.find("tensorflow");
    if (itr != config_map.end()) {
      if (BackendConfiguration(itr->second, "version", &tf_version_str)
              .IsOk()) {
        if ((tf_version_str != "1") && (tf_version_str != "2")) {
          return Status(
              Status::Code::INVALID_ARG,
              "unexpected TensorFlow library version '" + tf_version_str +
                  "', expects 1 or 2.");
        }
      }
    }

    *specialized_name += tf_version_str;
  }

  return Status::Success;
}

Status
BackendConfigurationBackendLibraryName(
    const std::string& backend_name, std::string* libname)
{
#ifdef _WIN32
  *libname = "triton_" + backend_name + ".dll";
#else
  *libname = "libtriton_" + backend_name + ".so";
#endif

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
