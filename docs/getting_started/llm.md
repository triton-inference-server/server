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
using the TensorRT-LLM PyTorch backend (LLM API). The PyTorch backend serves
supported Hugging Face models directly — no TensorRT engine compilation required.

> [!NOTE]
> The legacy TensorRT engine-build workflow (`convert_checkpoint.py` +
> `trtllm-build` and the `inflight_batcher_llm` ensemble model layout) is
> deprecated and is being removed from TensorRT-LLM. This guide uses the modern
> LLM API / PyTorch backend instead. See the
> [TensorRT-LLM Backend README](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/README.md)
> for the full set of configuration and deployment options.

This guide uses
[`nvidia/DeepSeek-R1-0528-FP4-V2`](https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4-V2)
as the example model.

- [Serve the model with Triton](#serve-the-model-with-triton)
- [Send an inference request](#send-an-inference-request)
- [Streaming responses](#streaming-responses)
- [Benchmark](#benchmark)
- [References](#references)

## Serve the model with Triton

### 1. Launch the container

```bash
export RELEASE=26.07
docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    nvcr.io/nvidia/tritonserver:${RELEASE}-trtllm-python-py3 bash
```

Check [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags)
for the latest `-trtllm-python-py3` tag. For gated models, set your token first:
`export HF_TOKEN=hf_...`

If the server later fails with `ImportError: cannot import name
'PartReasoningText'`, the container's `openai` package is too old — run
`pip install -U openai` and try again.

### 2. Get the model repository

The Triton model repository for the LLM API backend lives in the TensorRT-LLM
repo. Clone the tag matching the `tensorrt_llm` version in your container:

```bash
TRTLLM_VERSION=$(python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)")
git clone --depth 1 --branch "v${TRTLLM_VERSION}" https://github.com/NVIDIA/TensorRT-LLM.git
```

Deriving the tag this way keeps the model repository in step with the container,
so changing `RELEASE` above needs no second edit here.

### 3. Configure your model

Edit `TensorRT-LLM/triton_backend/all_models/llmapi/tensorrt_llm/1/model.yaml`
and set `model:` to a Hugging Face model ID or a local path:

```yaml
model: nvidia/DeepSeek-R1-0528-FP4-V2
backend: "pytorch"
tensor_parallel_size: 8
moe_expert_parallel_size: 8
max_seq_len: 4096
max_num_tokens: 8192
kv_cache_config:
  free_gpu_memory_fraction: 0.7

triton_config:
  max_batch_size: 0
  decoupled: False
```

All keys outside `triton_config` map directly to the
[`LLM()` constructor arguments](https://nvidia.github.io/TensorRT-LLM/llm-api/).
This is where you configure KV cache, quantization, and parallelism.

### 4. Launch the server

Run the launch script from the parent of `TensorRT-LLM/`. Running it from inside
the cloned folder makes Python import the source tree instead of the installed
package and fails with `ModuleNotFoundError: No module named
'tensorrt_llm.bindings'`:

```bash
python3 TensorRT-LLM/triton_backend/scripts/launch_triton_server.py \
    --model_repo=TensorRT-LLM/triton_backend/all_models/llmapi/
```

You should see the following logs once the server is ready:

```
I0803 18:43:44.778509 1525575 grpc_server.cc:2579] "Started GRPCInferenceService at 0.0.0.0:8001"
I0803 18:43:44.778681 1525575 http_server.cc:4961] "Started HTTPService at 0.0.0.0:8000"
I0803 18:43:44.819624 1525575 http_server.cc:400] "Started Metrics Service at 0.0.0.0:8002"
```

On an 8x B200 node this configuration takes about 6 minutes to become ready and
uses roughly 145 GiB of each GPU's 183 GiB, with 81.83 GiB left for the paged KV
cache (2,500,608 tokens).

## Send an inference request

`generate` is a raw completion endpoint and does not apply the model's chat
template, so format the prompt using the template from the checkpoint's
`tokenizer_config.json`:

```bash
curl -X POST localhost:8000/v2/models/tensorrt_llm/generate -d '{
  "text_input": "<｜begin▁of▁sentence｜><｜User｜>How do I count to nine in French?<｜Assistant｜>",
  "sampling_param_max_tokens": 512,
  "sampling_param_exclude_input_from_output": true }' | jq
```

DeepSeek-R1 emits its reasoning in a `<think>` block before the answer:

```
<think>
Okay, the user is asking how to count to nine in French. That seems
straightforward—they probably need the French numbers from one to nine.
...
```

Sampling options are passed as `sampling_param_*` inputs — see the `input`
section of
`TensorRT-LLM/triton_backend/all_models/llmapi/tensorrt_llm/config.pbtxt` for the
full list.

## Streaming responses

Streaming requires Triton's decoupled transaction policy. Setting
`decoupled: True` in `model.yaml` alone is **not** enough: the launch script
passes `--disable-auto-complete-config`, which skips the `auto_complete_config()`
hook where `model.yaml`'s `triton_config` is applied. The server then streams
while Triton core still treats the model as non-decoupled, and the request hangs
with `Streaming is only supported in decoupled mode.` in the server log.

To enable streaming, also append the policy to
`TensorRT-LLM/triton_backend/all_models/llmapi/tensorrt_llm/config.pbtxt`:

```
model_transaction_policy {
  decoupled: True
}
```

Then restart the server and use the `generate_stream` endpoint:

```bash
curl -N -X POST localhost:8000/v2/models/tensorrt_llm/generate_stream \
    -d '{"text_input": "Count to three:", "sampling_param_max_tokens": 10, "streaming": true}'
```

> [!NOTE]
> Token-by-token events require the model repository from TensorRT-LLM 1.3 or
> newer. With `v1.2.x`, a `streaming: true` request is accepted but the server
> emits a single event containing the full response. A decoupled model also
> cannot be used over the non-streaming `generate` endpoint, which returns
> `[501] HTTP end point doesn't support models with decoupled transaction
> policy` — keep `decoupled: False` unless you need streaming.

## Benchmark

Install the Triton client in the server container and run the backend's
benchmarking client against the model, with `decoupled: False`:

```bash
pip install "tritonclient[grpc,http]"

python3 TensorRT-LLM/triton_backend/tools/inflight_batcher_llm/benchmark_core_model.py \
  --max-input-len 500 \
  --tensorrt-llm-model-name tensorrt_llm \
  --test-llmapi \
  dataset --dataset TensorRT-LLM/triton_backend/tools/dataset/mini_cnn_eval.json \
  --tokenizer-dir nvidia/DeepSeek-R1-0528-FP4-V2
```

```
[INFO] Warm up for benchmarking.
[INFO] Start benchmarking on 37 prompts.
[INFO] Total Latency: <ms>
```

> [!NOTE]
> The shipped `model.yaml` sets `max_batch_size: 0`, so the backend serves
> requests without batching. Concurrency sweeps will not show throughput scaling
> until batching support lands.

## References

- [TensorRT-LLM User Guide](trtllm_user_guide.md)
- [TensorRT-LLM Backend README](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/README.md)
- [LLM API guide](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llmapi.md)
- [LLM API reference](https://nvidia.github.io/TensorRT-LLM/llm-api/)
- [TensorRT-LLM supported models](https://nvidia.github.io/TensorRT-LLM/models/supported-models.html)
