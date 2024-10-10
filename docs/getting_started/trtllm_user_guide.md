<!--
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# TensorRT-LLM User Guide

## What is TensorRT-LLM

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
(TRT-LLM) is an open-source library designed to accelerate and optimize the
inference performance of large language models (LLMs) on NVIDIA GPUs. TRT-LLM
offers users an easy-to-use Python API to build TensorRT engines for LLMs,
incorporating state-of-the-art optimizations to ensure efficient inference on
NVIDIA GPUs.

## How to run TRT-LLM models with Triton Server via TensorRT-LLM backend

The
[TensorRT-LLM Backend](https://github.com/triton-inference-server/tensorrtllm_backend)
lets you serve TensorRT-LLM models with Triton Inference Server. Check out the
[Getting Started](https://github.com/triton-inference-server/tensorrtllm_backend?tab=readme-ov-file#getting-started)
section in the TensorRT-LLM Backend repo to learn how to utlize the
[NGC Triton TRT-LLM container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
to prepare engines for your LLM models and serve them with Triton.

## How to use your custom TRT-LLM model

All the supported models can be found in the
[examples](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples) folder in
the TRT-LLM repo. Follow the examples to convert your models to TensorRT
engines.

After the engine is built, [prepare the model repository](https://github.com/triton-inference-server/tensorrtllm_backend?tab=readme-ov-file#prepare-the-model-repository)
for Triton, and
[modify the model configuration](https://github.com/triton-inference-server/tensorrtllm_backend?tab=readme-ov-file#modify-the-model-configuration).

Only the *mandatory parameters* need to be set in the model config file. Feel free
to modify the optional parameters as needed. To learn more about the
parameters, model inputs, and outputs, see the
[model config documentation](ttps://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/model_config.md) for more details.

## Advanced Configuration Options and Deployment Strategies

Explore advanced configuration options and deployment strategies to optimize
and run Triton with your TRT-LLM models effectively:

- [Model Deployment](https://github.com/triton-inference-server/tensorrtllm_backend/tree/main?tab=readme-ov-file#model-deployment): Techniques for efficiently deploying and managing your models in various environments.
- [Multi-Instance GPU (MIG) Support](https://github.com/triton-inference-server/tensorrtllm_backend/tree/main?tab=readme-ov-file#mig-support): Run Triton and TRT-LLM models with MIG to optimize GPU resource management.
- [Scheduling](https://github.com/triton-inference-server/tensorrtllm_backend/tree/main?tab=readme-ov-file#scheduling): Configure scheduling policies to control how requests are managed and executed.
- [Key-Value Cache](https://github.com/triton-inference-server/tensorrtllm_backend/tree/main?tab=readme-ov-file#key-value-cache): Utlizte KV cache and KV cache reuse to optimize memory usage and improve performance.
- [Decoding](https://github.com/triton-inference-server/tensorrtllm_backend/tree/main?tab=readme-ov-file#decoding): Advanced methods for generating text, including top-k, top-p, top-k top-p, beam search, Medusa, and speculative decoding.
- [Chunked Context](https://github.com/triton-inference-server/tensorrtllm_backend/tree/main?tab=readme-ov-file#chunked-context): Splitting the context into several chunks and batching them during generation phase to increase overall throughput.
- [Quantization](https://github.com/triton-inference-server/tensorrtllm_backend/tree/main?tab=readme-ov-file#quantization): Apply quantization techniques to reduce model size and enhance inference speed.
- [LoRa (Low-Rank Adaptation)](https://github.com/triton-inference-server/tensorrtllm_backend/tree/main?tab=readme-ov-file#lora): Use LoRa for efficient model fine-tuning and adaptation.

## Tutorials

Make sure to check out the
[tutorials](https://github.com/triton-inference-server/tutorials) repo to see
more guides on serving popular LLM models with Triton Server and TensorRT-LLM,
as well as deploying them on Kubernetes.

## Benchmark

[GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer/tree/main/genai-perf)
is a command line tool for measuring the throughput and latency of LLMs served
by Triton Inference Server. Check out the
[Quick Start](https://github.com/triton-inference-server/perf_analyzer/tree/main/genai-perf#quick-start)
to learn how to use GenAI-Perf to benchmark your LLM models.

## Performance Best Practices

Check out the
[Performance Best Practices guide](https://nvidia.github.io/TensorRT-LLM/performance/perf-best-practices.html)
to learn how to optimize your TensorRT-LLM models for better performance.

## Metrics

Triton Server provides
[metrics](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/metrics.md)
indicating GPU and request statistics.
See the
[Triton Metrics](https://github.com/triton-inference-server/tensorrtllm_backend?tab=readme-ov-file#triton-metrics)
section in the TensorRT-LLM Backend repo to learn how to query the Triton
metrics endpoint to obtain TRT-LLM statistics.

## Ask questions or report issues

Can't find what you're looking for, or have a question or issue? Feel free to
ask questions or report issues in the GitHub issues page:

- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/issues)
- [TensorRT-LLM Backend](https://github.com/triton-inference-server/tensorrtllm_backend/issues)
- [Triton Inference Server](https://github.com/triton-inference-server/server/issues)
