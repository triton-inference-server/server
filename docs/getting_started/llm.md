<!--
# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
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

# Deploying an LLM with Triton and TRT-LLM

This guide walks through serving a Hugging Face LLM with Triton Inference Server
using the TensorRT-LLM PyTorch backend (LLM API), and shows how to use GenAI-Perf
to benchmark throughput and latency. The PyTorch backend serves any Hugging Face
model directly — no TensorRT engine compilation required.

> [!NOTE]
> The legacy TensorRT engine-build workflow (`convert_checkpoint.py` +
> `trtllm-build` and the `inflight_batcher_llm` ensemble model layout) is
> deprecated and is being removed from TensorRT-LLM. This guide uses the modern
> LLM API / PyTorch backend instead. See the
> [TensorRT-LLM Backend README](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/README.md)
> for the full set of configuration and deployment options.

This guide uses [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) as the
example model, but you can serve any Hugging Face model by changing a single line
in `model.yaml`.

- [Serve the model with Triton](#serve-the-model-with-triton)
- [Send an inference request](#send-an-inference-request)
- [Benchmark with GenAI-Perf](#benchmark-with-genai-perf)
- [References](#references)

## Serve the model with Triton

### 1. Launch the container

```bash
docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    nvcr.io/nvidia/tritonserver:26.03-trtllm-python-py3 bash
```

Replace `26.03` with the latest tag from
[NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags).
For gated models, set your token first: `export HF_TOKEN=hf_...`

### 2. Configure your model

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
```

Edit `TensorRT-LLM/triton_backend/all_models/llmapi/tensorrt_llm/1/model.yaml`
and set `model:` to any Hugging Face model ID or local path:

```yaml
model: Qwen/Qwen3-8B
```

All keys in `model.yaml` map directly to the
[`LLM()` constructor arguments](https://nvidia.github.io/TensorRT-LLM/llm-api/).
This is where you configure KV cache, quantization, and parallelism. Qwen3-8B
runs on a single GPU. For larger flagship models — for example the
[Qwen/Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B)
Mixture-of-Experts model — set the parallelism to match your hardware:

```yaml
model: Qwen/Qwen3-235B-A22B
tensor_parallel_size: 8
```

### 3. Launch the server

Run the launch script from the parent of `TensorRT-LLM/` (running it from inside
the cloned folder causes `ModuleNotFoundError: No module named
'tensorrt_llm.bindings'`):

```bash
python3 TensorRT-LLM/triton_backend/scripts/launch_triton_server.py \
    --model_repo=TensorRT-LLM/triton_backend/all_models/llmapi/
```

You should see the following logs once the server is ready:

```
I0503 22:01:25.210518 1175 grpc_server.cc:2463] Started GRPCInferenceService at 0.0.0.0:8001
I0503 22:01:25.211612 1175 http_server.cc:4692] Started HTTPService at 0.0.0.0:8000
I0503 22:01:25.254914 1175 http_server.cc:362] Started Metrics Service at 0.0.0.0:8002
```

To stop Triton Server inside the container, run `pkill tritonserver`.

## Send an inference request

```bash
curl -X POST localhost:8000/v2/models/tensorrt_llm/generate \
    -d '{"text_input": "How do I count to nine in French?", "sampling_param_max_tokens": 256}' | jq
```

## Benchmark with GenAI-Perf

### 1. Launch the SDK container

```bash
export RELEASE="26.03"
docker run -it --net=host --gpus '"device=0"' nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk
```

### 2. Run GenAI-Perf

```bash
export INPUT_SEQUENCE_LENGTH=128
export OUTPUT_SEQUENCE_LENGTH=128
export CONCURRENCY=25

genai-perf profile \
  -m tensorrt_llm \
  --service-kind triton \
  --backend tensorrtllm \
  --random-seed 123 \
  --synthetic-input-tokens-mean $INPUT_SEQUENCE_LENGTH \
  --synthetic-input-tokens-stddev 0 \
  --streaming \
  --output-tokens-mean $OUTPUT_SEQUENCE_LENGTH \
  --output-tokens-stddev 0 \
  --output-tokens-mean-deterministic \
  --concurrency $CONCURRENCY \
  --tokenizer Qwen/Qwen3-8B \
  --measurement-interval 4000 \
  --url localhost:8001
```

More details on performance benchmarking with GenAI-Perf can be found
[here](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/README.md).

## References

- [TensorRT-LLM User Guide](trtllm_user_guide.md)
- [TensorRT-LLM Backend README](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/README.md)
- [LLM API guide](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llmapi.md)
- [LLM API reference](https://nvidia.github.io/TensorRT-LLM/llm-api/)
