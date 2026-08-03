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

This guide uses [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) as the
example model. You can serve any model listed in the TensorRT-LLM
[support matrix](https://nvidia.github.io/TensorRT-LLM/models/supported-models.html)
by changing a single line in `model.yaml`.

- [Serve the model with Triton](#serve-the-model-with-triton)
- [Send an inference request](#send-an-inference-request)
- [Streaming responses](#streaming-responses)
- [Multi-GPU models](#multi-gpu-models)
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

> [!IMPORTANT]
> The `26.03` and `26.07` containers (verified; the tags in between are likely
> affected too) ship `openai==1.107.3`, which is too old for the bundled
> `tensorrt_llm` package — TensorRT-LLM declares `openai` as a dependency with no
> lower bound. Without the fix below, loading the model fails with:
>
> ```
> ImportError: cannot import name 'PartReasoningText' from
> 'openai.types.responses.response_content_part_added_event'
> ```
>
> Upgrade the package inside the container before starting the server:
>
> ```bash
> pip install -U openai
> ```

### 2. Get the model repository

The Triton model repository for the LLM API backend lives in the TensorRT-LLM
repo. Clone the tag that matches the `tensorrt_llm` version inside your
container, so that the `model.py` you run matches the library it imports:

```bash
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"   # e.g. 1.2.1
git clone --depth 1 --branch v1.2.1 https://github.com/NVIDIA/TensorRT-LLM.git
```

> [!NOTE]
> Prefer the matching tag over the default branch. `main` tracks the next release
> (currently `1.3.0rc*`); it happens to work against a `1.2.1` container today,
> but nothing guarantees that, since its `model.py` is developed against the
> unreleased library. Pinning keeps the guide reproducible.

### 3. Configure your model

Edit `TensorRT-LLM/triton_backend/all_models/llmapi/tensorrt_llm/1/model.yaml`
and set `model:` to a Hugging Face model ID or a local path:

```yaml
model: Qwen/Qwen3-8B
backend: "pytorch"
tensor_parallel_size: 1
pipeline_parallel_size: 1

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
I0803 04:19:38.396545 3606543 grpc_server.cc:2579] "Started GRPCInferenceService at 0.0.0.0:8001"
I0803 04:19:38.396742 3606543 http_server.cc:4961] "Started HTTPService at 0.0.0.0:8000"
I0803 04:19:38.437685 3606543 http_server.cc:400] "Started Metrics Service at 0.0.0.0:8002"
```

> [!NOTE]
> `launch_triton_server.py` starts Triton as a background process and returns
> immediately. It is designed for an interactive shell. If you wrap it in a
> script, a batch job, or `srun`, keep the parent process alive or the server is
> killed when your script exits.

To stop Triton Server inside the container, run `pkill tritonserver`.

## Send an inference request

```bash
curl -X POST localhost:8000/v2/models/tensorrt_llm/generate \
    -d '{"text_input": "How do I count to nine in French?", "sampling_param_max_tokens": 256}' | jq
```

Sampling options are passed as `sampling_param_*` inputs — see the `input`
section of
`TensorRT-LLM/triton_backend/all_models/llmapi/tensorrt_llm/config.pbtxt` for the
full list. Add `"sampling_param_exclude_input_from_output": true` to get only the
generated text back instead of prompt + completion.

> [!NOTE]
> This endpoint performs raw text completion and does **not** apply the model's
> chat template. For instruct and reasoning models you must format the prompt
> yourself — see
> [Example: DeepSeek-R1 in NVFP4 on 8x B200](#example-deepseek-r1-in-nvfp4-on-8x-b200).

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

```
data: {"model_name":"tensorrt_llm","model_version":"1","text_output":" 1"}
data: {"model_name":"tensorrt_llm","model_version":"1","text_output":" 1,"}
data: {"model_name":"tensorrt_llm","model_version":"1","text_output":" 1, 2"}
```

> [!NOTE]
> Incremental, token-by-token events require the model repository from
> TensorRT-LLM 1.3 or newer. With the `v1.2.x` `model.py`, a `streaming: true`
> request is accepted but the server emits a single event containing the full
> response.

> [!NOTE]
> A decoupled model cannot be used over the non-streaming HTTP `generate`
> endpoint — Triton returns
> `[501] HTTP end point doesn't support models with decoupled transaction policy`.
> Keep `decoupled: False` unless you need streaming.

## Multi-GPU models

Larger models run across the GPUs of a single node by setting the parallelism in
`model.yaml`. The LLM API launches its own worker processes, so no extra
`--world_size` argument is needed on `launch_triton_server.py`:

```yaml
model: <your-model>
tensor_parallel_size: 8
```

For Mixture-of-Experts models, also set the expert parallelism. A bare
`tensor_parallel_size` is often not sufficient — check the model's row in the
[support matrix](https://nvidia.github.io/TensorRT-LLM/models/supported-models.html)
and its example README for required settings.

### Example: DeepSeek-R1 in NVFP4 on 8x B200

[`nvidia/DeepSeek-R1-0528-FP4-V2`](https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4-V2)
is a 671B-parameter MoE quantized to NVFP4 with an FP8 KV cache. Quantization
brings the checkpoint from 642 GB (FP8) down to 385 GB, so it fits comfortably on
one 8x B200 node with room left for the KV cache. NVFP4 requires Blackwell
(`SM100+`).

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

The quantization format is read from `hf_quant_config.json` in the checkpoint
(`quant_algo: NVFP4`, `kv_cache_quant_algo: FP8`) — you do not set it in
`model.yaml`.

Loading 385 GB takes several minutes on first start (about 6 minutes from a
warm local cache). Watch for `Started HTTPService` before sending requests. Once
resident, this configuration uses roughly 145 GiB of each B200's 183 GiB,
leaving headroom for the KV cache.

> [!IMPORTANT]
> `generate` is a **raw completion** endpoint — it does not apply the model's
> chat template. Sending a bare question to an instruct or reasoning model makes
> it continue the text rather than answer it. DeepSeek-R1 replies to
> `"How do I count to nine in French?"` with a list of scraped page titles.
> Format the prompt yourself using the model's template:
>
> ```bash
> curl -X POST localhost:8000/v2/models/tensorrt_llm/generate -d '{
>   "text_input": "<｜begin▁of▁sentence｜><｜User｜>How do I count to nine in French?<｜Assistant｜>",
>   "sampling_param_max_tokens": 512,
>   "sampling_param_exclude_input_from_output": true }'
> ```
>
> R1 then emits its reasoning trace in a `<think>` block before the answer:
>
> ```
> <think>
> Okay, the user is asking how to count to nine in French. That seems
> straightforward—they probably need the French numbers from one to nine.
> ...
> ```
>
> Each model family uses a different template — read `chat_template` in the
> checkpoint's `tokenizer_config.json`.

> [!NOTE]
> DeepSeek-V4 (`DeepSeek-V4-Flash` / `-Pro`) is **not** usable with these
> containers. It requires TensorRT-LLM 1.3 or newer, which is not yet GA and is
> not in `26.07` or earlier — those ship TensorRT-LLM 1.2.x, which does not
> register `DeepseekV4ForCausalLM`. Loading it fails with
> `The checkpoint you are trying to load has model type 'deepseek_v4' but
> Transformers does not recognize this architecture.` The same applies to GLM-5.x
> (`glm_moe_dsa`). Use DeepSeek-R1 or GLM-4.7 until a container ships 1.3.

## Benchmark

The LLM API backend ships its own benchmarking client. Install the Triton client
in the server container and run it against the model, with `decoupled: False`:

```bash
pip install "tritonclient[grpc,http]"

python3 TensorRT-LLM/triton_backend/tools/inflight_batcher_llm/benchmark_core_model.py \
  --max-input-len 500 \
  --tensorrt-llm-model-name tensorrt_llm \
  --test-llmapi \
  dataset --dataset TensorRT-LLM/triton_backend/tools/dataset/mini_cnn_eval.json \
  --tokenizer-dir Qwen/Qwen3-8B
```

```
Tokenizer: Tokens per word =  1.324
[INFO] Warm up for benchmarking.
[INFO] Start benchmarking on 37 prompts.
[INFO] Total Latency: 853.993 ms
```

> [!NOTE]
> GenAI-Perf's `--backend tensorrtllm` mode targets the legacy
> `inflight_batcher_llm` model, whose inputs are named `max_tokens` and `stream`.
> The LLM API backend names them `sampling_param_max_tokens` and `streaming`, so
> GenAI-Perf fails with
> `Failed to init manager inputs: The input or output 'max_tokens' is not found in
> the model configuration`. Use the client above until GenAI-Perf adds LLM API
> support.

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
