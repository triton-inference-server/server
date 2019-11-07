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

#include "src/core/status.h"

#ifdef TRTIS_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRTIS_ENABLE_GPU

namespace nvidia { namespace inferenceserver {

#ifndef TRTIS_ENABLE_GPU
using cudaStream_t = void*;
#endif  // TRTIS_ENABLE_GPU

/// Enable peer access for all GPU device pairs
/// \return The error status. A non-OK status means not all pairs are enabled
Status EnablePeerAccess();

/// Copy buffer from 'src' to 'dst' for given 'byte_size'. The buffer location
/// is identified by the memory type and id, and the corresponding copy will be
/// initiated.
/// 'msg' is the message to be prepended in error message.
/// 'cuda_stream' specifies the stream to be associated with, and 0 can be
/// passed for default stream.
/// 'cuda_used' returns whether a CUDA memory copy is initiated. If true,
/// the caller should synchronize on the given 'cuda_stream' to ensure data copy
/// is completed.
/// \return The error status.
Status CopyBuffer(
    const std::string& msg, const TRTSERVER_Memory_Type src_memory_type,
    const int64_t src_memory_type_id,
    const TRTSERVER_Memory_Type dst_memory_type,
    const int64_t dst_memory_type_id, const size_t byte_size, const void* src,
    void* dst, cudaStream_t cuda_stream, bool* cuda_used);

}}  // namespace nvidia::inferenceserver
