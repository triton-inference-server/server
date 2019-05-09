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

#include "src/backends/tensorflow/loader.h"

#include "src/backends/tensorflow/tf_utils.h"
#include "src/core/logging.h"

namespace nvidia { namespace inferenceserver {

Status
LoadSavedModel(
    const std::string& model_name, const std::string& model_path,
    const tensorflow::SessionOptions& session_options,
    std::unique_ptr<tensorflow::SavedModelBundle>* bundle,
    tensorflow::SignatureDef* sig)
{
  bundle->reset(new tensorflow::SavedModelBundle);

  std::unordered_set<std::string> saved_model_tags;
  saved_model_tags.insert(tensorflow::kSavedModelTagServe);

  tensorflow::RunOptions run_options;
  RETURN_IF_TF_ERROR(tensorflow::LoadSavedModel(
      session_options, run_options, model_path, saved_model_tags,
      bundle->get()));

  LOG_VERBOSE(1) << "Loaded saved-model: "
                 << (*bundle)->meta_graph_def.DebugString();

  // Verify that the bundle has the "serve" tag
  bool found_serve_tag = false;
  for (const auto& tag : (*bundle)->meta_graph_def.meta_info_def().tags()) {
    if (tag == tensorflow::kSavedModelTagServe) {
      found_serve_tag = true;
      break;
    }
  }
  if (!found_serve_tag) {
    return Status(
        RequestStatusCode::INTERNAL,
        "unable to load model '" + model_name + "', expected '" +
            tensorflow::kSavedModelTagServe + "' tag");
  }

  // Verify that a "serving_default" signature exists, that is what
  // will be used to verify the inputs and outputs.
  static const std::string DEFAULT_SERVING_SIGNATURE_DEF_KEY("serving_default");
  const auto& sig_itr = (*bundle)->meta_graph_def.signature_def().find(
      DEFAULT_SERVING_SIGNATURE_DEF_KEY);
  if (sig_itr == (*bundle)->meta_graph_def.signature_def().end()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "unable to load model '" + model_name + "', expected '" +
            DEFAULT_SERVING_SIGNATURE_DEF_KEY + "' signature");
  }

  if (sig != nullptr) {
    *sig = sig_itr->second;
  }

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
