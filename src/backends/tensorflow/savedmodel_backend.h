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

#include "src/backends/tensorflow/base_backend.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class SavedModelBackend : public BaseBackend {
 public:
  SavedModelBackend() = default;
  SavedModelBackend(SavedModelBackend&&) = default;

  Status Init(const std::string& path, const ModelConfig& config);

  Status CreateTRTISTFModel(
      const std::shared_ptr<GraphDefBackendFactory::Config>& backend_config,
      const int device_id, const bool has_graph_level, const int graph_level,
      const std::string& model_path, TRTISTFModelHandle* trtistf_model,
      IONameMap* input_name_map, IONameMap* output_name_map,
      const TRTISTF_TFTRTConfig* tftrt_config) override;

 private:
  Status ValidateSequenceControl(
      const ModelSequenceBatching::Control::Kind control_kind,
      const TRTISTF_IOList* inputs);

  DISALLOW_COPY_AND_ASSIGN(SavedModelBackend);
};

}}  // namespace nvidia::inferenceserver
