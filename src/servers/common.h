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

#include "src/core/request_status.pb.h"
#include "src/core/trtserver.h"

namespace nvidia { namespace inferenceserver {

#define FAIL(MSG)                                 \
  do {                                            \
    std::cerr << "error: " << (MSG) << std::endl; \
    exit(1);                                      \
  } while (false)

#define FAIL_IF_ERR(X, MSG)                                  \
  do {                                                       \
    TRTSERVER_Error* err = (X);                              \
    if (err != nullptr) {                                    \
      std::cerr << "error: " << (MSG) << ": "                \
                << TRTSERVER_ErrorCodeString(err) << " - "   \
                << TRTSERVER_ErrorMessage(err) << std::endl; \
      TRTSERVER_ErrorDelete(err);                            \
      exit(1);                                               \
    }                                                        \
  } while (false)

#define RETURN_IF_ERR(X)        \
  do {                          \
    TRTSERVER_Error* err = (X); \
    if (err != nullptr) {       \
      return err;               \
    }                           \
  } while (false)

// Return the RequestStatusCode corresponding to a TRTSERVER error
// code.
RequestStatusCode CodeToStatus(TRTSERVER_Error_Code code);

}}  // namespace nvidia::inferenceserver
