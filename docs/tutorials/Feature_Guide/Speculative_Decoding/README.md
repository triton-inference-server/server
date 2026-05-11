<!--
Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

# Speculative Decoding

## Overview

Speculative decoding (also known as speculative sampling or assisted generation) is an inference optimization technique that accelerates Large Language Model (LLM) text generation without compromising output quality. It leverages a smaller, faster "draft" model to propose candidate tokens, which are then verified in parallel by the larger "target" model.

## Why Use Speculative Decoding?

### Key Benefits

1. **Reduced Latency**: Generate tokens faster by using a lightweight draft model for initial proposals
2. **Identical Outputs**: Maintains mathematically identical outputs to standard autoregressive decoding
3. **Lower Cost**: Reduces inference costs by decreasing GPU time for generation
4. **Better GPU Utilization**: Improves batch processing efficiency

### Typical Performance Improvements

- **1.5-3x speedup** in generation latency (varies by model and use case)
- **No quality degradation** - outputs are identical to standard decoding
- **Most effective** for generation-heavy workloads (longer output sequences)

## How It Works

The speculative decoding process follows these steps:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Draft Model generates K candidate tokens                │
│    (Fast, small model - e.g., 1B parameters)                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Target Model verifies all K tokens in PARALLEL          │
│    (Slower, large model - e.g., 70B parameters)             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Accept matching tokens, correct first mismatch          │
│    Continue from corrected position                         │
└─────────────────────────────────────────────────────────────┘
```

### Example

Suppose we want to generate the sentence: "The cat sat on the mat."

1. **Draft Model proposes**: "The cat sat on the"
2. **Target Model verifies**:
   - ✓ "The" - Accept
   - ✓ "cat" - Accept
   - ✓ "sat" - Accept
   - ✓ "on" - Accept
   - ✓ "the" - Accept
3. **Result**: All 5 tokens accepted, continue from "the"
4. **Next round**: Draft proposes "mat .", Target verifies and accepts

Instead of 7 sequential forward passes through the large model, we only needed 2, reducing latency by ~3.5x.

## Supported Backends

Triton Inference Server supports speculative decoding with multiple backends:

| Backend | Support Status | Documentation |
|---------|---------------|---------------|
| **vLLM** | ✅ Supported | [vLLM Guide](vLLM/README.md) |
| **TensorRT-LLM** | ✅ Supported | [TRT-LLM Guide](TRT-LLM/README.md) |
| **Python Backend** | ⚠️ Custom Implementation | Manual integration required |

## Choosing a Backend

### vLLM Backend

**Best for:**
- Quick deployment and prototyping
- Dynamic batching with speculative decoding
- Flexible model support (any HuggingFace model)
- Easier configuration via JSON

**Considerations:**
- Slightly higher memory usage
- Less optimized for specific hardware compared to TRT-LLM

[→ See vLLM Speculative Decoding Guide](vLLM/README.md)

### TensorRT-LLM Backend

**Best for:**
- Maximum performance on NVIDIA GPUs
- Production deployments requiring lowest latency
- INT8/FP8 quantization support
- Tightly optimized kernels

**Considerations:**
- Requires engine compilation
- More complex setup process
- Specific model architecture support

[→ See TensorRT-LLM Speculative Decoding Guide](TRT-LLM/README.md)

## Quick Start

Choose your backend and follow the corresponding guide:

### vLLM (Recommended for Getting Started)

```bash
# 1. Create model configuration with speculative decoding
cat > model_repository/my_model/1/model.json << EOF
{
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "speculative_model": "meta-llama/Llama-3.2-1B-Instruct",
  "num_speculative_tokens": 5,
  "gpu_memory_utilization": 0.9
}
EOF

# 2. Launch Triton with vLLM backend
docker run --gpus all --rm \
  -v $(pwd)/model_repository:/models \
  -e HF_TOKEN=$HF_TOKEN \
  nvcr.io/nvidia/tritonserver:26.04-vllm-python-py3 \
  tritonserver --model-repository=/models

# 3. Send inference request
curl -X POST http://localhost:8000/v2/models/my_model/generate \
  -d '{"text_input": "Explain speculative decoding"}'
```

[→ Full vLLM Setup Guide](vLLM/README.md)

### TensorRT-LLM

TensorRT-LLM requires building engines for both target and draft models. See the [TensorRT-LLM guide](TRT-LLM/README.md) for detailed instructions.

## Model Selection Guidelines

### Choosing a Draft Model

For optimal performance, your draft model should be:

1. **From the same model family** as the target model
   - Example: Use Llama 3.2 1B as draft for Llama 3.1 8B/70B
   - Example: Use Mistral 7B as draft for Mixtral 8x7B

2. **10-50x smaller** than the target model
   - Too small: Low acceptance rate
   - Too large: Insufficient speedup

3. **Using the same tokenizer**
   - Ensures token-level compatibility

### Recommended Model Pairs

| Target Model | Draft Model | Size Ratio | Expected Speedup |
|--------------|-------------|------------|------------------|
| Llama-3.1-70B | Llama-3.2-1B | 70x | 2.0-2.8x |
| Llama-3.1-8B | Llama-3.2-1B | 8x | 1.5-2.2x |
| Mixtral-8x7B | Mistral-7B-v0.1 | ~8x | 1.6-2.4x |
| CodeLlama-34B | CodeLlama-7B | 4.8x | 1.4-2.0x |
| Qwen2.5-72B | Qwen2.5-7B | 10x | 1.8-2.5x |

## Performance Tuning

### Key Parameters

1. **num_speculative_tokens** (vLLM) / **num_draft_tokens** (TRT-LLM)
   - Controls how many tokens the draft model generates per iteration
   - Higher values: Better speedup potential, more memory usage
   - Typical range: 3-10
   - Start with 5 and adjust based on results

2. **gpu_memory_utilization**
   - Both models must fit in GPU memory simultaneously
   - Reduce if encountering OOM errors
   - Typical range: 0.8-0.9

3. **Acceptance Rate**
   - Higher is better (target: >70%)
   - Low acceptance (<50%) indicates draft model is too different
   - Monitor via metrics endpoint

### Memory Optimization

```
Total GPU Memory = Target Model + Draft Model + KV Cache + Activations
```

If memory is constrained:
- Use a smaller draft model
- Reduce `max_model_len` or `max_batch_size`
- Enable tensor parallelism for target model only
- Use quantization (INT8/FP8) where supported

## Monitoring and Metrics

Triton exposes metrics for speculative decoding performance:

```bash
# Access metrics endpoint
curl http://localhost:8002/metrics
```

### Key Metrics to Monitor

- **`speculative_acceptance_rate`**: Percentage of draft tokens accepted (target: >70%)
- **`inter_token_latency`**: Time between tokens (should decrease significantly)
- **`time_to_first_token`**: Should remain similar to non-speculative mode
- **`throughput_tokens_per_second`**: Overall tokens/sec (should increase)

## When NOT to Use Speculative Decoding

Speculative decoding may not be beneficial in these scenarios:

1. **Very short outputs**: Overhead may exceed benefits for <10 token generations
2. **Extremely high batch sizes**: May be memory-constrained with both models
3. **No suitable draft model**: If no compatible smaller model exists
4. **Maximum batch utilization**: When already fully utilizing GPU with standard decoding

## Troubleshooting

### Low or No Speedup

**Possible causes:**
- Draft model too different from target model
- `num_speculative_tokens` too low (try increasing to 7-10)
- Acceptance rate too low (<50%)
- Batch size too small to hide overhead

**Solutions:**
- Use a draft model from the same family
- Increase speculative token count
- Monitor acceptance rate metrics
- Adjust batch size

### Out of Memory Errors

**Solutions:**
- Reduce `gpu_memory_utilization` to 0.8 or lower
- Use a smaller draft model
- Decrease `max_model_len`
- Enable tensor parallelism for target model
- Reduce `num_speculative_tokens`

### Different Outputs Than Expected

**Note:** Speculative decoding should produce **identical outputs** to standard decoding. If outputs differ:
- Check that both models loaded successfully
- Verify model versions match expected
- Review logs for errors or warnings
- This may indicate a bug - please report it

## Additional Resources

### Research Papers

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan et al., 2023)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) (Chen et al., 2023)
- [SpecInfer: Accelerating LLM Serving with Tree-based Speculative Inference](https://arxiv.org/abs/2305.09781) (Miao et al., 2023)

### Documentation

- [vLLM Backend Repository](https://github.com/triton-inference-server/vllm_backend)
- [TensorRT-LLM Backend Repository](https://github.com/triton-inference-server/tensorrtllm_backend)
- [Triton Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)

### Community

- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
- [Triton GitHub Discussions](https://github.com/triton-inference-server/server/discussions)
- [Report Issues](https://github.com/triton-inference-server/server/issues)

## Next Steps

1. **Choose your backend**: [vLLM](vLLM/README.md) or [TensorRT-LLM](TRT-LLM/README.md)
2. **Follow the setup guide** for your chosen backend
3. **Experiment with different draft models** and parameters
4. **Monitor metrics** to optimize performance
5. **Deploy to production** once you've validated the configuration

---

**Questions or feedback?** Please open an issue on the [Triton GitHub repository](https://github.com/triton-inference-server/server/issues).
