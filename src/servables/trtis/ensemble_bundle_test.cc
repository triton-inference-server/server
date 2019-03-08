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
#include "src/core/logging.h"
#include "src/core/utils.h"
#include "src/test/model_config_test_base.h"

namespace nvidia { namespace inferenceserver { namespace test {

class EnsembleBundleTest : public ModelConfigTestBase {
 public:
  bool GetModelConfigsInRepository(
    const std::string& model_base_path,
    std::unordered_map<std::string, ModelConfig>& config_map,
    std::string* result);
};

bool
EnsembleBundleTest::GetModelConfigsInRepository(
  const std::string& model_base_path,
  std::unordered_map<std::string, ModelConfig>& config_map,
  std::string* result)
{
  result->clear();
  config_map.clear();

  std::vector<std::string> models;
  TF_CHECK_OK(
      tensorflow::Env::Default()->GetChildren(model_base_path, &models));

  for (const auto& model_name : models) {
    const auto model_path =
        tensorflow::io::JoinPath(model_base_path, model_name);

    if (!tensorflow::Env::Default()->IsDirectory(model_path).ok()) {
      continue;
    }

    ModelConfig config;
    tfs::PlatformConfigMap platform_map;
    tensorflow::Status status =
        GetNormalizedModelConfig(model_path, platform_map, false, &config);
    if (!status.ok()) {
      result->append(status.ToString());
      return false;
    }

    status = ValidateModelConfig(config, std::string());
    if (!status.ok()) {
      result->append(status.ToString());
      return false;
    }

    config_map[config.name()] = config;
  }
  return true;
}

TEST_F(EnsembleBundleTest, ModelConfigSanity)
{
  BundleInitFunc init_func =
      [](const std::string& path,
         const ModelConfig& config) -> tensorflow::Status {
    return tensorflow::Status::OK();
  };

  // Standard testing...
  ValidateAll(kEnsemblePlatform, init_func);

  // Check model config sanity against ensemble's own test cases
  ValidateOne(
      "inference_server/src/servables/trtis/testdata/model_config_sanity",
      true /* autofill */, std::string() /* platform */, init_func);
}

TEST_F(EnsembleBundleTest, EnsembleConfigSanity)
{
  std::string error;
  std::unordered_map<std::string, ModelConfig> config_map;

  const std::string test_repo_path = tensorflow::io::JoinPath(getenv("TEST_SRCDIR"),
          "inference_server/src/servables/trtis/testdata/ensemble_config_sanity");
  std::vector<std::string> model_repos;
  TF_CHECK_OK(
      tensorflow::Env::Default()->GetChildren(test_repo_path, &model_repos));
    
  for (const auto& repo : model_repos) {
    const std::string model_base_path =
        tensorflow::io::JoinPath(test_repo_path, repo);
    if (GetModelConfigsInRepository(model_base_path, config_map, &error) == false) {
      EXPECT_TRUE(error.empty());
      LOG_ERROR << "Unexpected error while loading model configs:" << std::endl
                << error;
    }

    std::string actual;
    tensorflow::Status status = ValidateEnsembleConfig(config_map);
    if (!status.ok()) {
      actual.append(status.ToString());
    }
  }
}

}}}  // namespace nvidia::inferenceserver::test
