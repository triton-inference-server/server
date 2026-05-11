<!--
Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Speculative Decoding with TensorRT-LLM Backend

## Overview

TensorRT-LLM backend provides highly optimized speculative decoding support for NVIDIA GPUs. This guide covers configuration and deployment of speculative decoding using the TensorRT-LLM backend.

## Documentation

For comprehensive documentation on speculative decoding with TensorRT-LLM, please refer to the official TensorRT-LLM backend documentation:

[TensorRT-LLM Backend Decoding Documentation](https://github.com/triton-inference-server/tensorrtllm_backend/tree/main?tab=readme-ov-file#decoding)

## Key Features

- Maximum performance on NVIDIA GPUs with optimized kernels
- Support for INT8/FP8 quantization
- Advanced scheduling and batching
- Medusa and standard speculative decoding modes

## Quick Reference

For speculative decoding setup with TensorRT-LLM:

1. Build TensorRT-LLM engines for both target and draft models
2. Configure the model repository with appropriate parameters
3. Deploy using Triton with TensorRT-LLM backend

See the [TensorRT-LLM backend documentation](https://github.com/triton-inference-server/tensorrtllm_backend) for detailed instructions.

## See Also

- [Speculative Decoding Overview](../README.md)
- [vLLM Speculative Decoding](../vLLM/README.md) (alternative backend)
- [TensorRT-LLM Backend](https://github.com/triton-inference-server/tensorrtllm_backend)
