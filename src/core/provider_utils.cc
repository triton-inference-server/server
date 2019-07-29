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

#include "src/core/provider_utils.h"

#include <google/protobuf/text_format.h>
#include <deque>
#include "src/core/backend.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/model_config_utils.h"

namespace nvidia { namespace inferenceserver {

Status
NormalizeRequestHeader(
    const InferenceBackend& is, InferRequestHeader& request_header)
{
  const std::string& model_name = is.Name();
  const ModelConfig& model_config = is.Config();

  // Make sure the request has a batch-size > 0. Even for models that
  // don't support batching the requested batch size must be 1.
  if (request_header.batch_size() < 1) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "inference request batch-size must be >= 1 for '" + model_name + "'");
  }

  // Make sure request batch-size doesn't exceed what is supported by
  // the model. For models that don't support batching the request
  // batch-size will still be 1.
  if ((request_header.batch_size() != 1) &&
      ((int)request_header.batch_size() > model_config.max_batch_size())) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "inference request batch-size must be <= " +
            std::to_string(model_config.max_batch_size()) + " for '" +
            model_name + "'");
  }

  // Make sure that the request is providing the same number of inputs
  // as is expected by the model.
  if (request_header.input_size() != model_config.input_size()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "expected " + std::to_string(model_config.input_size()) +
            " inputs but got " + std::to_string(request_header.input_size()) +
            " inputs for model '" + model_name + "'");
  }

  // Update each input to have shape and batch-byte-size.
  for (InferRequestHeader::Input& io : *request_header.mutable_input()) {
    const ModelInput* input_config;
    RETURN_IF_ERROR(is.GetInput(io.name(), &input_config));

    // If the inference request specifies a shape for an input, make
    // sure it matches what the model expects.
    if (io.dims_size() > 0) {
      if (!CompareDimsWithWildcard(io.dims(), input_config->dims())) {
        return Status(
            RequestStatusCode::INVALID_ARG,
            "unexpected shape for input '" + io.name() + "' for model '" +
                model_name + "'. Expected " +
                DimsListToString(input_config->dims()) + ", got " +
                DimsListToString(io.dims()));
      }

      // If there is a reshape for this input then clear the dims and
      // set them to the reshape. As reshape may have variable-size dimensions,
      // we need to record corresponding value
      // so that we can set the value correctly for reshape.
      if (input_config->has_reshape()) {
        std::deque<int64_t> variable_size_values;
        for (int64_t idx = 0; idx < input_config->dims_size(); idx++) {
          if (input_config->dims(idx) == -1) {
            variable_size_values.push_back(io.dims(idx));
          }
        }

        io.clear_dims();
        for (const auto& dim : input_config->reshape().shape()) {
          if (dim == -1) {
            io.add_dims(variable_size_values.front());
            variable_size_values.pop_front();
          } else {
            io.add_dims(dim);
          }
        }
      }
    }

    // If we don't have shape for the input at this point then the
    // request didn't specify it, or it has a reshape that we must use
    // instead.
    if (io.dims_size() == 0) {
      const DimsList& dims = (input_config->has_reshape())
                                 ? input_config->reshape().shape()
                                 : input_config->dims();

      // Inference request doesn't specify shape, make sure input
      // shape is fully specified in the model and calculate expected
      // size from the model configuration.
      for (auto dim : dims) {
        if (dim < 0) {
          return Status(
              RequestStatusCode::INVALID_ARG,
              "model supports variable-size for input '" + io.name() +
                  "', request must specify input shape for model '" +
                  model_name + "'");
        }

        io.add_dims(dim);
      }
    }

    // For fixed-size datatype the tensor used to calculate byte-size
    // is:
    //
    //   [ batch-size, tensor-shape ] : for batching model and
    //   non-zero-rank tensor. For example, batch-size 4 and dims [ 1,
    //   2 ] the full tensor shape is [ 4, 1, 2 ].
    //
    //   [ tensor-shape ] : for non-batching model and non-zero-rank
    //   tensor. For example, dims [ 1, 2 ] the full tensor shape is [
    //   1, 2 ].
    //
    //   [ batch-size ] : for batching model and zero-rank tensor. For
    //   example, batch-size 4 with dims [ 1 ] and reshape [ ], the
    //   full tensor shape is [ 4 ].
    //
    // Note that non-batching zero-rank tensor is not allowed since
    // that will always be shape [], i.e. a tensor with no contents.
    //
    uint64_t bs = 0;
    if (IsFixedSizeDataType(input_config->data_type())) {
      bs = GetByteSize(input_config->data_type(), io.dims());
      if (model_config.max_batch_size() > 0) {
        if (io.dims_size() == 0) {
          bs = GetDataTypeByteSize(input_config->data_type()) *
               request_header.batch_size();
        } else {
          bs *= request_header.batch_size();
        }
      }

      // If batch-byte-size is given check to make sure that the
      // calculated batch size matches
      if ((io.batch_byte_size() != 0) && (io.batch_byte_size() != bs)) {
        return Status(
            RequestStatusCode::INVALID_ARG,
            "specific batch-byte-size for input '" + io.name() +
                "' does not match expected byte-size calculated from shape and "
                "datatype for model '" +
                model_name + "'");
      }
    } else {
      // The input's datatype is not fixed-sized (like TYPE_STRING),
      // use the full-batch size specified by the request.
      bs = io.batch_byte_size();
    }

    io.set_batch_byte_size(bs);
  }

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
