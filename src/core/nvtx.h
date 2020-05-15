// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifdef TRITON_ENABLE_NVTX

#include <nvtx3/nvToolsExt.h>
#include "src/core/logging.h"

namespace nvidia { namespace inferenceserver {

// Updates a server stat with duration measured by a C++ scope.
class NvtxRange {
 public:
  explicit NvtxRange(const std::string& label)
  {
    depth_ = nvtxRangePushA(label.c_str());
    if (depth_ < 0) {
      LOG_ERROR << "Unable to start NVTX range '" << label << "'";
    }
  }

  ~NvtxRange()
  {
    if (depth_ >= 0) {
      nvtxRangePop();
    }
  }

 private:
  int depth_;
};

}}  // namespace nvidia::inferenceserver

#endif  // TRITON_ENABLE_NVTX

//
// Macros to access NVTX functionality
//
#ifdef TRITON_ENABLE_NVTX
#define NVTX_INITIALIZE nvtxInitialize(nullptr)
#define NVTX_RANGE(V, L) nvidia::inferenceserver::NvtxRange V(L)
#define NVTX_MARKER(L) nvtxMarkA(L)
#else
#define NVTX_INITIALIZE
#define NVTX_RANGE(V, L)
#define NVTX_MARKER(L)
#endif  // TRITON_ENABLE_NVTX
