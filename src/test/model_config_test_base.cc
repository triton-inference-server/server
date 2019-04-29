// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "src/test/model_config_test_base.h"

#include <stdlib.h>
#include <fstream>
#include <memory>
#include "src/core/constants.h"
#include "src/core/filesystem.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/model_config_utils.h"

namespace nvidia { namespace inferenceserver { namespace test {

bool
ModelConfigTestBase::ValidateInit(
    const std::string& model_path, bool autofill, BundleInitFunc init_func,
    std::string* result)
{
  result->clear();

  ModelConfig config;
  PlatformConfigMap platform_map;
  Status status =
      GetNormalizedModelConfig(model_path, platform_map, autofill, &config);
  if (!status.IsOk()) {
    result->append(status.AsString());
    return false;
  }

  status = ValidateModelConfig(config, std::string());
  if (!status.IsOk()) {
    result->append(status.AsString());
    return false;
  }

  // ModelConfig unit tests assume model version "1"
  const std::string version_path = JoinPath({model_path, "1"});

  status = init_func(version_path, config);
  if (!status.IsOk()) {
    result->append(status.AsString());
    return false;
  }

  *result = config.DebugString();
  return true;
}

void
ModelConfigTestBase::ValidateAll(
    const std::string& platform, BundleInitFunc init_func)
{
  // Sanity tests without autofill and forcing the platform.
  ValidateOne(
      "inference_server/src/test/testdata/model_config_sanity",
      false /* autofill */, platform, init_func);

  // Sanity tests with autofill and no platform.
  ValidateOne(
      "inference_server/src/test/testdata/autofill_sanity", true /* autofill */,
      std::string() /* platform */, init_func);
}

void
ModelConfigTestBase::ValidateOne(
    const std::string& test_repository_rpath, bool autofill,
    const std::string& platform, BundleInitFunc init_func)
{
  const std::string model_base_path =
      JoinPath({getenv("TEST_SRCDIR"), test_repository_rpath});

  std::set<std::string> models;
  CHECK_IF_ERROR(GetDirectorySubdirs(model_base_path, &models));

  for (const auto& model_name : models) {
    const auto model_path = JoinPath({model_base_path, model_name});

    // If a platform is specified and there is a configuration file
    // then must change the configuration to use that platform. We
    // modify the config file in place... not ideal but for how our CI
    // testing is done it is not a problem.
    if (!platform.empty()) {
      const auto config_path = JoinPath({model_path, kModelConfigPbTxt});

      bool config_exists;
      CHECK_IF_ERROR(FileExists(config_path, &config_exists));
      if (config_exists) {
        ModelConfig config;
        CHECK_IF_ERROR(ReadTextProto(config_path, &config));
        config.set_platform(platform);
        CHECK_IF_ERROR(WriteTextProto(config_path, config));
      }
    }

    LOG_INFO << "Testing " << model_name;
    std::string actual, fail_expected;
    ValidateInit(model_path, autofill, init_func, &actual);

    CompareActualWithExpected(model_path, actual, &fail_expected);

    EXPECT_TRUE(fail_expected.empty());
    if (!fail_expected.empty()) {
      LOG_ERROR << "Expected:" << std::endl << fail_expected;
      LOG_ERROR << "Actual:" << std::endl << actual;
    }
  }
}

void
ModelConfigTestBase::CompareActualWithExpected(
    const std::string& expected_path, const std::string& actual,
    std::string* fail_expected)
{
  // The actual output must match *one of* the "expected*" files.
  std::set<std::string> children;
  if (GetDirectoryFiles(expected_path, &children).IsOk()) {
    for (const auto& child : children) {
      if (child.find("expected") == 0) {
        const auto expected_file_path = JoinPath({expected_path, child});
        LOG_INFO << "Comparing with " << expected_file_path;

        std::ifstream expected_file(expected_file_path);
        std::string expected(
            (std::istreambuf_iterator<char>(expected_file)),
            (std::istreambuf_iterator<char>()));
        std::string truncated_actual;
        if (expected.size() < actual.size()) {
          truncated_actual = actual.substr(0, expected.size());
        } else {
          truncated_actual = actual;
        }

        if (expected != truncated_actual) {
          *fail_expected = expected;
        } else {
          fail_expected->clear();
          break;
        }
      }
    }
  }
}
}}}  // namespace nvidia::inferenceserver::test
