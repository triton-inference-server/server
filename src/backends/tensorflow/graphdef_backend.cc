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

#include "src/backends/tensorflow/graphdef_backend.h"

#include <set>
#include "src/backends/tensorflow/tensorflow_backend_tf.h"
#include "src/backends/tensorflow/tf_utils.h"
#include "src/core/constants.h"
#include "src/core/filesystem.h"
#include "src/core/model_config.h"
#include "src/core/model_config_utils.h"

namespace nvidia { namespace inferenceserver {

Status
GraphDefBackend::Init(const std::string& path, const ModelConfig& config)
{
  RETURN_IF_ERROR(ValidateModelConfig(config, kTensorFlowGraphDefPlatform));
  RETURN_IF_ERROR(BaseBackend::Init(path, config));
  return Status::Success;
}

Status
GraphDefBackend::CreateTRTISTFModel(
    const std::shared_ptr<GraphDefBackendFactory::Config>& backend_config,
    const int device_id, const bool has_graph_level, const int graph_level,
    const std::string& model_path, TRTISTFModelHandle* trtistf_model,
    IONameMap* input_name_map, IONameMap* output_name_map,
    const TRTISTF_TFTRTConfig* tftrt_config)
{
  TRTISTF_Model* model = nullptr;
  RETURN_IF_TRTISTF_ERROR(TRTISTF_ModelCreateFromGraphDef(
      &model, model_path.c_str(), model_path.c_str(), device_id,
      has_graph_level, graph_level, backend_config->allow_gpu_memory_growth,
      backend_config->per_process_gpu_memory_fraction,
      backend_config->allow_soft_placement, backend_config->memory_limit_mb,
      tftrt_config));

  trtistf_model->reset(model);

  // For graphdef the model inputs and outputs are just "potential"
  // inputs and outputs since graphdef doesn't explicitly list the
  // inputs and outputs. Also, only the name is available, shape and
  // datatype are not.
  const TRTISTF_IOList* inputs = TRTISTF_ModelInputs(model);
  const TRTISTF_IOList* outputs = TRTISTF_ModelOutputs(model);

  std::set<std::string> potential_inputs, potential_outputs;
  for (const TRTISTF_IOList* itr = inputs; itr != nullptr; itr = itr->next_) {
    potential_inputs.insert(itr->io_->name_);
  }
  for (const TRTISTF_IOList* itr = outputs; itr != nullptr; itr = itr->next_) {
    potential_outputs.insert(itr->io_->name_);
  }

  if (potential_inputs.size() < (size_t)Config().input().size()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "unable to load model '" + Name() + "', configuration expects " +
            std::to_string(Config().input().size()) +
            " inputs, model provides at most " +
            std::to_string(potential_inputs.size()));
  }

  for (const auto& io : Config().input()) {
    RETURN_IF_ERROR(CheckAllowedModelInput(io, potential_inputs));
  }
  for (const auto& io : Config().output()) {
    RETURN_IF_ERROR(CheckAllowedModelOutput(io, potential_outputs));
  }

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
