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

#include <unordered_map>
#include "src/core/constants.h"
#include "src/core/ensemble_utils.h"
#include "src/core/filesystem.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/model_config_utils.h"
#include "src/test/model_config_test_base.h"

namespace nvidia { namespace inferenceserver { namespace test {

class EnsembleBackendTest : public ModelConfigTestBase {
 public:
  bool GetModelConfigsInRepository(
      const std::string& model_base_path,
      std::unordered_map<std::string, ModelConfig>& config_map,
      std::string* result);
};

bool
EnsembleBackendTest::GetModelConfigsInRepository(
    const std::string& model_base_path,
    std::unordered_map<std::string, ModelConfig>& config_map,
    std::string* result)
{
  result->clear();
  config_map.clear();

  std::set<std::string> models;
  CHECK_IF_ERROR(GetDirectorySubdirs(model_base_path, &models));

  for (const auto& model_name : models) {
    const std::string model_path = JoinPath({model_base_path, model_name});

    ModelConfig config;
    PlatformConfigMap platform_map;
    Status status =
        GetNormalizedModelConfig(model_path, platform_map, false, &config);
    if (!status.IsOk()) {
      result->append(status.AsString());
      return false;
    }

    status = ValidateModelConfig(config, std::string());
    if (!status.IsOk()) {
      result->append(status.AsString());
      return false;
    }

    config_map[config.name()] = config;
  }
  return true;
}

TEST_F(EnsembleBackendTest, ModelConfigSanity)
{
  BackendInitFunc init_func = [](const std::string& path,
                                 const ModelConfig& config) -> Status {
    return Status::Success;
  };

  // Standard testing...
  ValidateAll(kEnsemblePlatform, init_func);

  // Check model config sanity against ensemble's own test cases
  ValidateOne(
      "inference_server/src/backends/ensemble/testdata/model_config_sanity",
      true /* autofill */, std::string() /* platform */, init_func);
}

TEST_F(EnsembleBackendTest, EnsembleConfigSanity)
{
  std::string error;
  std::unordered_map<std::string, ModelConfig> config_map;

  const std::string test_repo_path =
      JoinPath({getenv("TEST_SRCDIR"),
                "inference_server/src/backends/ensemble/testdata/"
                "ensemble_config_sanity"});
  std::set<std::string> model_repos;
  CHECK_IF_ERROR(GetDirectorySubdirs(test_repo_path, &model_repos));

  for (const auto& repo : model_repos) {
    const std::string model_base_path = JoinPath({test_repo_path, repo});
    if (GetModelConfigsInRepository(model_base_path, config_map, &error) ==
        false) {
      EXPECT_TRUE(error.empty());
      LOG_ERROR << "Unexpected error while loading model configs in " << repo
                << ":" << std::endl
                << error;
      continue;
    }

    std::string actual;
    Status status = ValidateEnsembleConfig(config_map);
    if (!status.IsOk()) {
      actual.append(status.AsString());
    }

    std::string fail_expected;
    CompareActualWithExpected(model_base_path, actual, &fail_expected);

    EXPECT_TRUE(fail_expected.empty());
    if (!fail_expected.empty()) {
      LOG_ERROR << "Expected:" << std::endl << fail_expected;
      LOG_ERROR << "Actual:" << std::endl << actual;
    }
  }
}

}}}  // namespace nvidia::inferenceserver::test
