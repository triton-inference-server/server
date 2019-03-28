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
#pragma once

#include "src/clients/c++/request.h"
#include "src/core/constants.h"

namespace nvidia { namespace inferenceserver { namespace client {

//==============================================================================
/// InferenceServerContext is the in-process instantiation of an
/// inference server.
///
class InferenceServerContext {
 public:
  /// Options used when creating the inference server.
  class Options {
   public:
    virtual ~Options() = 0;

    // Create Options.
    static Error Create(std::unique_ptr<Options>* options);

    /// Set the model repository path for the server.
    /// \param path The model repository path.
    virtual void SetModelRepositoryPath(const std::string& path) = 0;
  };

  /// Create the singleton inference server. This method may be called only
  /// once as there can only be a single inference server in a process.
  /// \param ctx Returns the singleton InferenceServerContext object.
  /// \param options The options to use when creating the server.
  /// \return Error object indicating success or failure.
  static Error Create(
      std::unique_ptr<InferenceServerContext>* ctx,
      const std::unique_ptr<Options>& options);
};

//==============================================================================
/// ServerHealthInProcessContext is the in-process instantiation of
/// ServerHealthContext.
///
class ServerHealthInProcessContext {
 public:
  /// Create a context that returns health information about server.
  /// \param ctx Returns a new ServerHealthContext object.
  /// \param server_ctx The inference server.
  /// \param verbose If true generate verbose output when contacting
  /// the inference server.
  /// \return Error object indicating success or failure.
  static Error Create(
      std::unique_ptr<ServerHealthContext>* ctx,
      const std::unique_ptr<InferenceServerContext>& server_ctx,
      bool verbose = false);
};

//==============================================================================
/// ServerStatusInProcessContext is the in-process instantiation of
/// ServerStatusContext.
///
class ServerStatusInProcessContext {
 public:
  /// Create a context that returns information about an inference
  /// server and all models on the server.
  /// \param ctx Returns a new ServerStatusContext object.
  /// \param server_ctx The inference server.
  /// \param verbose If true generate verbose output when contacting
  /// the inference server.
  /// \return Error object indicating success or failure.
  static Error Create(
      std::unique_ptr<ServerStatusContext>* ctx,
      const std::unique_ptr<InferenceServerContext>& server_ctx,
      bool verbose = false);

  /// Create a context that returns information about an inference
  /// server and one model on the server.
  /// \param ctx Returns a new ServerStatusContext object.
  /// \param server_ctx The inference server.
  /// \param model_name The name of the model to get status for.
  /// \param verbose If true generate verbose output when contacting
  /// the inference server.
  /// \return Error object indicating success or failure.
  static Error Create(
      std::unique_ptr<ServerStatusContext>* ctx,
      const std::unique_ptr<InferenceServerContext>& server_ctx,
      const std::string& model_name, bool verbose = false);
};

//==============================================================================
/// InferInProcessContext is the in-process instantiation of InferContext.
///
class InferInProcessContext {
 public:
  /// Create context that performs inferencing for a non-sequence
  /// model.
  ///
  /// \param ctx Returns a new InferInProcessContext object.
  /// \param server_ctx The inference server.
  /// \param model_name The name of the model to get status for.
  /// \param model_version The version of the model to use for inference,
  /// or -1 to indicate that the latest (i.e. highest version number)
  /// version should be used.
  /// \param verbose If true generate verbose output when contacting
  /// the inference server.
  /// \return Error object indicating success or failure.
  static Error Create(
      std::unique_ptr<InferContext>* ctx,
      const std::unique_ptr<InferenceServerContext>& server_ctx,
      const std::string& model_name, int64_t model_version = -1,
      bool verbose = false);

  /// Create context that performs inference for a sequence model
  /// using a given correlation ID.
  ///
  /// \param ctx Returns a new InferInProcessContext object.
  /// \param server_ctx The inference server.
  /// \param correlation_id The correlation ID to use for all
  /// inferences performed with this context. A value of 0 (zero)
  /// indicates that no correlation ID should be used.
  /// \param model_name The name of the model to get status for.
  /// \param model_version The version of the model to use for inference,
  /// or -1 to indicate that the latest (i.e. highest version number)
  /// version should be used.
  /// \param verbose If true generate verbose output when contacting
  /// the inference server.
  /// \return Error object indicating success or failure.
  static Error Create(
      std::unique_ptr<InferContext>* ctx,
      const std::unique_ptr<InferenceServerContext>& server_ctx,
      CorrelationID correlation_id, const std::string& model_name,
      int64_t model_version = -1, bool verbose = false);
};

}}}  // namespace nvidia::inferenceserver::client
