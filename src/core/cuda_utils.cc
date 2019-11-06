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

#include "src/core/cuda_utils.h"

#include "src/core/model_config_utils.h"
#include "src/core/request_status.pb.h"

namespace nvidia { namespace inferenceserver {

Status
EnablePeerAccess()
{
#ifdef TRTIS_ENABLE_GPU
  // If we can't enable peer access for one device pair, the best we can
  // do is skipping it...
  std::set<int> supported_gpus;
  bool all_enabled = false;
  if (GetSupportedGPUs(supported_gpus).IsOk()) {
    all_enabled = true;
    int can_access_peer = false;
    for (const auto& host : supported_gpus) {
      auto cuerr = cudaSetDevice(host);

      if (cuerr == cudaSuccess) {
        for (const auto& peer : supported_gpus) {
          if (host == peer) {
            continue;
          }

          cuerr = cudaDeviceCanAccessPeer(&can_access_peer, host, peer);
          if ((cuerr == cudaSuccess) && (can_access_peer == 1)) {
            cuerr = cudaDeviceEnablePeerAccess(peer, 0);
          }

          all_enabled &= ((cuerr == cudaSuccess) && (can_access_peer == 1));
        }
      }
    }
  }
  if (!all_enabled) {
    return Status(
        RequestStatusCode::UNSUPPORTED,
        "failed to enable peer access for some device pairs");
  }
#endif  // TRTIS_ENABLE_GPU
  return Status::Success;
}

Status
CopyBuffer(
    const std::string& msg, const TRTSERVER_Memory_Type src_memory_type,
    const int64_t src_memory_type_id,
    const TRTSERVER_Memory_Type dst_memory_type,
    const int64_t dst_memory_type_id, const size_t byte_size, const void* src,
    void* dst, cudaStream_t cuda_stream, bool* cuda_used)
{
  *cuda_used = false;

  if ((src_memory_type == TRTSERVER_MEMORY_CPU) &&
      (dst_memory_type == TRTSERVER_MEMORY_CPU)) {
    memcpy(dst, src, byte_size);
  } else {
#ifdef TRTIS_ENABLE_GPU
    // [TODO] use cudaMemcpyDefault if UVM is supported for the device
    auto copy_kind = cudaMemcpyDeviceToDevice;
    if (src_memory_type == TRTSERVER_MEMORY_CPU) {
      copy_kind = cudaMemcpyHostToDevice;
    } else if (dst_memory_type == TRTSERVER_MEMORY_CPU) {
      copy_kind = cudaMemcpyDeviceToHost;
    }

    cudaError_t err;
    if ((src_memory_type_id != dst_memory_type_id) &&
        (copy_kind == cudaMemcpyDeviceToDevice)) {
      err = cudaMemcpyPeerAsync(
          dst, dst_memory_type_id, src, src_memory_type_id, byte_size,
          cuda_stream);
    } else {
      err = cudaMemcpyAsync(dst, src, byte_size, copy_kind, cuda_stream);
    }

    if (err != cudaSuccess) {
      return Status(
          RequestStatusCode::INTERNAL,
          msg + ": failed to use CUDA copy : " +
              std::string(cudaGetErrorString(err)));
    } else {
      *cuda_used = true;
    }
#else
    return Status(
        RequestStatusCode::INTERNAL,
        msg + ": try to use CUDA copy while GPU is not supported");
#endif  // TRTIS_ENABLE_GPU
  }
  return Status::Success;
}

}}  // namespace nvidia::inferenceserver