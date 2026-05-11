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

# Speculative Decoding with vLLM Backend

## Overview

Speculative decoding is an inference optimization technique that accelerates text generation by using a smaller, faster "draft" model to propose tokens, which are then verified by the larger "target" model. This approach can significantly reduce latency while maintaining the same output quality as standard decoding.

The vLLM backend in Triton Inference Server supports speculative decoding, allowing you to leverage this optimization technique for your LLM deployments.

## How It Works

1. A small draft model quickly generates multiple candidate tokens
2. The larger target model verifies these candidates in parallel
3. Accepted tokens are returned; rejected tokens are corrected by the target model
4. This process continues until the full response is generated

The speedup comes from the fact that the draft model is much faster than the target model, and verification can be done in parallel for multiple tokens.

## Prerequisites

- Triton Inference Server with vLLM backend support
- A target model (the main model you want to serve)
- A draft model (a smaller, compatible model from the same family)
- Docker with NVIDIA Container Runtime
- Access to HuggingFace models (HF_TOKEN if using gated models)

## Configuration

Speculative decoding with vLLM is configured through the `model.json` file in your model repository. The vLLM backend passes the configuration parameters directly to vLLM's `AsyncEngineArgs`.

### Key Parameters

The following parameters in `model.json` control speculative decoding:

- **`speculative_model`**: The name or path of the draft model to use
- **`num_speculative_tokens`**: Number of tokens to generate speculatively (default: depends on model)
- **`speculative_draft_tensor_parallel_size`**: Tensor parallelism size for draft model (optional)
- **`ngram_prompt_lookup_max`**: For n-gram based speculation (alternative to draft model)
- **`ngram_prompt_lookup_min`**: Minimum n-gram size for prompt lookup

### Example 1: Basic Speculative Decoding Configuration

Here's a simple example using Llama models:

```json
{
  "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
  "speculative_model": "meta-llama/Meta-Llama-3.2-1B-Instruct",
  "num_speculative_tokens": 5,
  "gpu_memory_utilization": 0.9
}
```

### Example 2: Speculative Decoding with Tensor Parallelism

For larger deployments with multi-GPU setups:

```json
{
  "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
  "tensor_parallel_size": 4,
  "speculative_model": "meta-llama/Meta-Llama-3.2-1B-Instruct",
  "num_speculative_tokens": 5,
  "speculative_draft_tensor_parallel_size": 1,
  "gpu_memory_utilization": 0.85
}
```

### Example 3: N-gram Prompt Lookup (Alternative Approach)

Instead of using a separate draft model, you can use n-gram based speculation:

```json
{
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "ngram_prompt_lookup_max": 4,
  "ngram_prompt_lookup_min": 1,
  "gpu_memory_utilization": 0.9
}
```

## Model Repository Structure

Your model repository should follow this structure:

```
model_repository/
└── llama-3.1-70b-speculative/
    ├── config.pbtxt
    └── 1/
        └── model.json
```

### config.pbtxt

```
# Copyright 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# (License text omitted for brevity)

name: "llama-3.1-70b-speculative"
backend: "vllm"
max_batch_size: 0
model_transaction_policy {
  decoupled: True
}

input [
  {
    name: "text_input"
    data_type: TYPE_STRING
    dims: [ -1 ]
  },
  {
    name: "stream"
    data_type: TYPE_BOOL
    dims: [ 1 ]
    optional: true
  }
]

output [
  {
    name: "text_output"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_MODEL
  }
]
```

## Running the Example

### Step 1: Launch Container

```bash
docker run -it --net=host --gpus all --rm \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  -v ${PWD}/model_repository:/model_repository \
  -e HF_TOKEN \
  nvcr.io/nvidia/tritonserver:26.04-vllm-python-py3
```

### Step 2: Start Triton Server

For native vLLM backend:

```bash
tritonserver --model-repository=/model_repository
```

Or using the OpenAI-compatible frontend:

```bash
cd /opt/tritonserver/python/openai
python3 openai_frontend/main.py \
  --model-repository /model_repository \
  --tokenizer meta-llama/Meta-Llama-3.1-70B-Instruct
```

### Step 3: Send Inference Requests

Using the OpenAI API:

```bash
MODEL="llama-3.1-70b-speculative"
curl -s http://localhost:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'${MODEL}'",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "max_tokens": 256
  }' | jq
```

Or using the native Triton API:

```bash
curl -X POST http://localhost:8000/v2/models/llama-3.1-70b-speculative/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text_input": "Explain quantum computing in simple terms.",
    "parameters": {
      "max_tokens": 256,
      "temperature": 0.7
    }
  }'
```

## Choosing the Right Draft Model

For best results:

1. **Same Model Family**: Use a draft model from the same family as your target model (e.g., Llama 3.2 1B for Llama 3.1 70B)
2. **Size Ratio**: Aim for a draft model that is 10-50x smaller than the target model
3. **Architecture Compatibility**: Ensure the draft model has compatible architecture (same tokenizer, similar attention mechanisms)

### Popular Model Combinations

| Target Model | Draft Model | Expected Speedup |
|--------------|-------------|------------------|
| Llama-3.1-70B | Llama-3.2-1B | 1.5-2.5x |
| Llama-3.1-8B | Llama-3.2-1B | 1.3-2.0x |
| Mixtral-8x7B | Mistral-7B-v0.1 | 1.4-2.2x |

> **Note**: Actual speedup depends on hardware, batch size, sequence length, and the similarity between draft and target model outputs.

## Performance Tuning

### Adjusting num_speculative_tokens

- **Higher values (5-10)**: Better speedup potential but higher memory usage
- **Lower values (2-4)**: More conservative, lower memory overhead
- **Start with 5** and adjust based on your specific use case

### Memory Considerations

Speculative decoding requires loading both models into GPU memory. Adjust `gpu_memory_utilization` accordingly:

```json
{
  "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
  "speculative_model": "meta-llama/Meta-Llama-3.2-1B-Instruct",
  "num_speculative_tokens": 5,
  "gpu_memory_utilization": 0.85
}
```

If you encounter OOM errors, try:
1. Reducing `gpu_memory_utilization` to 0.8 or lower
2. Decreasing `num_speculative_tokens`
3. Using a smaller draft model
4. Enabling tensor parallelism for the target model

## Monitoring and Debugging

### Check Model Loading

When Triton starts, you should see log messages indicating both models are loaded:

```
I0511 00:00:00.000000 1 llm_engine.py:123] Initializing an LLM engine with config: ...
I0511 00:00:00.000000 1 llm_engine.py:456] Using speculative decoding with draft model: meta-llama/Meta-Llama-3.2-1B-Instruct
```

### Metrics

Monitor these metrics to evaluate speculative decoding performance:

- **Acceptance Rate**: Percentage of draft tokens accepted (higher is better)
- **Time to First Token (TTFT)**: Should be similar to non-speculative
- **Inter-Token Latency**: Should be significantly lower
- **Throughput**: Overall tokens/second should increase

Access metrics at `http://localhost:8002/metrics` (or `:9000/metrics` for OpenAI frontend).

## Troubleshooting

### Common Issues

**Issue**: Model fails to load with OOM error
```
Solution: Reduce gpu_memory_utilization or use a smaller draft model
```

**Issue**: No speedup observed
```
Solution: 
- Ensure draft model is from the same family as target model
- Check that num_speculative_tokens > 0
- Verify both models loaded successfully in logs
- Try increasing num_speculative_tokens
```

**Issue**: Different outputs compared to non-speculative mode
```
Solution: This should not happen - speculative decoding guarantees identical outputs.
Check vLLM backend logs for errors. This may indicate a configuration issue.
```

**Issue**: Draft model not found or fails to load
```
Solution:
- Verify the speculative_model path/name is correct
- Ensure HF_TOKEN is set if using gated models
- Check that the draft model is cached or can be downloaded
```

## Limitations

1. **Memory Overhead**: Requires loading both target and draft models
2. **Model Compatibility**: Draft model must be compatible with target model
3. **Batch Size**: Effectiveness may vary with different batch sizes
4. **Sequence Length**: Longer sequences may see different speedup characteristics

## Additional Resources

- [vLLM Backend Documentation](https://github.com/triton-inference-server/vllm_backend)
- [vLLM Speculative Decoding](https://docs.vllm.ai/)
- [Speculative Decoding Paper](https://arxiv.org/abs/2211.17192)
- [TRT-LLM Speculative Decoding](../TRT-LLM/README.md) (alternative backend)

## References

- Chen, C., et al. (2023). "Accelerating Large Language Model Decoding with Speculative Sampling"
- Leviathan, Y., et al. (2023). "Fast Inference from Transformers via Speculative Decoding"

## Feedback and Support

For issues or questions:
- [Triton GitHub Issues](https://github.com/triton-inference-server/server/issues)
- [vLLM Backend Issues](https://github.com/triton-inference-server/vllm_backend/issues)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
