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
#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include "src/core/constants.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

// Provides classification labels.
class LabelProvider {
 public:
  LabelProvider() = default;

  // Return the label associated with 'name' for a given
  // 'index'. Return empty string if no label is available.
  const std::string& GetLabel(const std::string& name, size_t index) const;

  // Associate with 'name' a set of labels initialized from a given
  // 'filepath'. Within the file each label is specified on its own
  // line. The first label (line 0) is the index-0 label, the second
  // label (line 1) is the index-1 label, etc.
  Status AddLabels(const std::string& name, const std::string& filepath);

  // Return the labels associated with 'name'. Return empty vector if no labels
  // are available.
  const std::vector<std::string>& GetLabels(const std::string& name);

  // Associate with 'name' a set of 'labels'
  Status AddLabels(
      const std::string& name, const std::vector<std::string>& labels);

 private:
  DISALLOW_COPY_AND_ASSIGN(LabelProvider);

  std::unordered_map<std::string, std::vector<std::string>> label_map_;
};

}}  // namespace nvidia::inferenceserver
