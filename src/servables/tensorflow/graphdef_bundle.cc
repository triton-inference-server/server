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

#include "src/servables/tensorflow/graphdef_bundle.h"

#include <set>
#include "src/core/constants.h"
#include "src/core/filesystem.h"
#include "src/core/model_config.h"
#include "src/core/model_config_utils.h"
#include "src/servables/tensorflow/tf_utils.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/lib/io/path.h"

namespace nvidia { namespace inferenceserver {

Status
GraphDefBundle::Init(const std::string& path, const ModelConfig& config)
{
  RETURN_IF_ERROR(ValidateModelConfig(config, kTensorFlowGraphDefPlatform));
  RETURN_IF_ERROR(BaseBundle::Init(path, config));
  return Status::Success;
}

Status
GraphDefBundle::CreateSession(
    const tensorflow::SessionOptions& options, const int gpu_device,
    const std::string& model_path, tensorflow::Session** session,
    IONameMap* input_name_map, IONameMap* output_name_map)
{
  RETURN_IF_TF_ERROR(tensorflow::NewSession(options, session));

  tensorflow::GraphDef graph_def;
  RETURN_IF_ERROR(ReadBinaryProto(model_path, &graph_def));
  if (graph_def.node_size() == 0) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "model " + Name() + " has an empty network");
  }

  // Set the default device to control the CPU/GPU that the graph runs
  // on. This isn't foolproof since individual operations in the graph
  // could specify a specific run location. But given that
  // visible_device_list doesn't work it seems like the only option we
  // have. [DLIS-43]
  if (gpu_device == Context::NO_GPU_DEVICE) {
    tensorflow::graph::SetDefaultDevice("/cpu:0", &graph_def);
  } else {
    tensorflow::graph::SetDefaultDevice(
        "/gpu:" + std::to_string(gpu_device), &graph_def);
  }

  RETURN_IF_TF_ERROR((*session)->Create(graph_def));

  // Go through all graph nodes and collect the possible inputs and
  // outputs. We use this to verify the requested inputs and outputs
  // when initializing. Unfortunately graphdef isn't explicit in
  // indicating inputs and outputs so we assume any Placeholder can be
  // an input and any node can be an output.
  std::set<std::string> potential_inputs, potential_outputs;
  for (const auto& node : graph_def.node()) {
    if (node.op() == "Placeholder") {
      potential_inputs.emplace(node.name());
    } else {
      potential_outputs.emplace(node.name());
    }
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
