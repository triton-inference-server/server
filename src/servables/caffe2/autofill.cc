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

#include "src/servables/caffe2/autofill.h"

#include "src/core/autofill.h"
#include "src/core/constants.h"
#include "src/core/filesystem.h"
#include "tensorflow/core/lib/io/path.h"

namespace nvidia { namespace inferenceserver {

class AutoFillNetDefImpl : public AutoFill {
 public:
  AutoFillNetDefImpl(const std::string& model_name) : AutoFill(model_name) {}
  Status Fix(ModelConfig* config) override;
};

Status
AutoFillNetDefImpl::Fix(ModelConfig* config)
{
  config->set_platform(kCaffe2NetDefPlatform);

  // Set name if not already set.
  if (config->name().empty()) {
    config->set_name(model_name_);
  }

  return Status::Success;
}

Status
AutoFillNetDef::Create(
    const std::string& model_name, const std::string& model_path,
    std::unique_ptr<AutoFill>* autofill)
{
  std::set<std::string> version_dirs;
  RETURN_IF_ERROR(GetDirectorySubdirs(model_path, &version_dirs));

  // There must be at least one version directory that we can inspect
  // to attempt to determine the platform. For now we only handle the
  // case where there is one version directory.
  if (version_dirs.size() != 1) {
    return Status(
        RequestStatusCode::INTERNAL,
        "unable to autofill for '" + model_name + "' due to multiple versions");
  }

  const auto version_path = JoinPath({model_path, *(version_dirs.begin())});

  // There must be a single netdef model (which is spread across two
  // files) within the version directory...
  std::set<std::string> netdef_files;
  RETURN_IF_ERROR(GetDirectoryFiles(version_path, &netdef_files));
  if (netdef_files.size() != 2) {
    return Status(
        RequestStatusCode::INTERNAL, "unable to autofill for '" + model_name +
                                         "', unable to find netdef files");
  }

  const std::string netdef0_file = *(netdef_files.begin());
  const auto netdef0_path = JoinPath({version_path, netdef0_file});

  const std::string netdef1_file = *(std::next(netdef_files.begin()));
  const auto netdef1_path = JoinPath({version_path, netdef1_file});

  const std::string expected_init_filename =
      std::string(kCaffe2NetDefInitFilenamePrefix) +
      std::string(kCaffe2NetDefFilename);

  // If find both files named with the default netdef names then
  // assume it is a netdef. In the future we can be smarter here and
  // try to parse to see if it really is a netdef, and then try to
  // derive more of the configuration...
  if (!(((netdef0_file == kCaffe2NetDefFilename) &&
         (netdef1_file == expected_init_filename)) ||
        ((netdef1_file == kCaffe2NetDefFilename) &&
         (netdef0_file == expected_init_filename)))) {
    return Status(
        RequestStatusCode::INTERNAL,
        "unable to autofill for '" + model_name +
            "', unable to find netdef files named '" + kCaffe2NetDefFilename +
            "' and '" + expected_init_filename + "'");
  }

  autofill->reset(new AutoFillNetDefImpl(model_name));
  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
