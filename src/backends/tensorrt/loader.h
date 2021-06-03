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

#include <NvInfer.h>
#include <memory>
#include <vector>
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

/// Load a TensorRT plan from a binary blob and return the
/// corresponding runtime and engine. It is the caller's
/// responsibility to destroy any returned runtime or engine object
/// even if an error is returned.
///
/// \param model_data The binary blob of the plan data.
/// \param dla_core_id The DLA core to use for this runtime. Does not
/// use DLA when set to -1.
/// \param runtime Returns the IRuntime object, or nullptr if failed
/// to create.
/// \param engine Returns the ICudaEngine object, or nullptr if failed
/// to create.
/// \return Error status.
Status LoadPlan(
    const std::vector<char>& model_data, int64_t dla_core_id,
    std::shared_ptr<nvinfer1::IRuntime>* runtime,
    std::shared_ptr<nvinfer1::ICudaEngine>* engine);

}}  // namespace nvidia::inferenceserver
