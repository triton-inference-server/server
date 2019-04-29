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

#include "src/servables/tensorrt/plan_bundle.h"
#include "src/core/constants.h"
#include "src/core/filesystem.h"
#include "src/core/status.h"
#include "src/test/model_config_test_base.h"

namespace nvidia { namespace inferenceserver { namespace test {

class PlanBundleTest : public ModelConfigTestBase {
 public:
};

TEST_F(PlanBundleTest, ModelConfigSanity)
{
  BundleInitFunc init_func = [](const std::string& path,
                                const ModelConfig& config) -> Status {
    std::unique_ptr<PlanBundle> bundle(new PlanBundle());
    Status status = bundle->Init(path, config);
    if (status.IsOk()) {
      std::unordered_map<std::string, std::vector<char>> plan_blobs;

      for (const auto& filename :
           std::vector<std::string>{kTensorRTPlanFilename}) {
        const auto plan_path = JoinPath({path, filename});
        tensorflow::string blob_str;
        ReadTextFile(plan_path, &blob_str);
        std::vector<char> blob(blob_str.begin(), blob_str.end());
        plan_blobs.emplace(filename, std::move(blob));
      }

      status = bundle->CreateExecutionContexts(plan_blobs);
    }

    return status;
  };

  // Standard testing...
  ValidateAll(kTensorRTPlanPlatform, init_func);

  // Sanity tests with autofill and not providing the platform.
  ValidateOne(
      "inference_server/src/servables/tensorrt/testdata/autofill_sanity",
      true /* autofill */, std::string() /* platform */, init_func);
}

}}}  // namespace nvidia::inferenceserver::test
