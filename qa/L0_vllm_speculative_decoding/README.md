<!--
Copyright 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

# vLLM Speculative Decoding Test

This directory contains test configurations for vLLM speculative decoding feature.

## Model Configuration

The example model `llama-speculative` demonstrates how to configure speculative decoding:

```json
{
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "speculative_model": "meta-llama/Llama-3.2-1B-Instruct",
  "num_speculative_tokens": 5,
  "gpu_memory_utilization": 0.9,
  "max_model_len": 2048
}
```

## Running the Test

```bash
# Set your HuggingFace token if using gated models
export HF_TOKEN="your_token_here"

# Launch container
docker run -it --net=host --gpus all --rm \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/model_repository:/model_repository \
  -e HF_TOKEN \
  nvcr.io/nvidia/tritonserver:26.04-vllm-python-py3

# Inside container, start Triton
tritonserver --model-repository=/model_repository
```

## Sending Test Requests

```bash
# Using Triton's native API
curl -X POST http://localhost:8000/v2/models/llama-speculative/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text_input": "What is speculative decoding?",
    "parameters": {
      "max_tokens": 100,
      "temperature": 0.7
    }
  }'
```

## Expected Behavior

- Both target and draft models should load successfully
- Inference should complete with reduced latency compared to non-speculative mode
- Output quality should be identical to standard decoding
- Server logs should show speculative decoding is enabled

## See Also

- [vLLM Speculative Decoding Tutorial](../../../docs/tutorials/Feature_Guide/Speculative_Decoding/vLLM/README.md)
