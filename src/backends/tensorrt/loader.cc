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

#include "src/backends/tensorrt/loader.h"

#include <NvInferPlugin.h>
#include <memory>
#include <mutex>
#include "src/backends/tensorrt/logging.h"
#include "src/core/logging.h"

namespace nvidia { namespace inferenceserver {

Status
LoadPlan(
    const std::vector<char>& model_data, int64_t dla_core_id,
    std::shared_ptr<nvinfer1::IRuntime>* runtime,
    std::shared_ptr<nvinfer1::ICudaEngine>* engine)
{
  // Create runtime only if it is not provided
  if (*runtime == nullptr) {
    runtime->reset(nvinfer1::createInferRuntime(tensorrt_logger));
    if (*runtime == nullptr) {
      return Status(
          Status::Code::INTERNAL, "unable to create TensorRT runtime");
    }

    // Report error if 'dla_core_id' >= number of DLA cores
    if (dla_core_id != -1) {
      if (dla_core_id < (*runtime)->getNbDLACores()) {
        (*runtime)->setDLACore(dla_core_id);
      } else {
        return Status(
            Status::Code::INVALID_ARG,
            ("unable to create TensorRT runtime with DLA Core ID: " +
             std::to_string(dla_core_id))
                .c_str());
      }
    }
  }

  engine->reset(
      (*runtime)->deserializeCudaEngine(&model_data[0], model_data.size()));
  if (*engine == nullptr) {
    return Status(Status::Code::INTERNAL, "unable to create TensorRT engine");
  }

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
