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

#include "src/core/label_provider.h"

#include <iostream>
#include <iterator>
#include <sstream>
#include "src/core/filesystem.h"

namespace nvidia { namespace inferenceserver {

const std::string&
LabelProvider::GetLabel(const std::string& name, size_t index) const
{
  static const std::string not_found;

  auto itr = label_map_.find(name);
  if (itr == label_map_.end()) {
    return not_found;
  }

  if (itr->second.size() <= index) {
    return not_found;
  }

  return itr->second[index];
}

Status
LabelProvider::AddLabels(const std::string& name, const std::string& filepath)
{
  std::string label_file_contents;
  RETURN_IF_ERROR(ReadTextFile(filepath, &label_file_contents));

  auto p = label_map_.insert(std::make_pair(name, std::vector<std::string>()));
  if (!p.second) {
    return Status(
        RequestStatusCode::INTERNAL, "multiple label files for '" + name + "'");
  }

  auto itr = p.first;

  std::istringstream label_file_stream(label_file_contents);
  std::string line;
  while (std::getline(label_file_stream, line)) {
    itr->second.push_back(line);
  }

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
