---
name: triton-onnx-model-repository
description: Deploy a user's ONNX file as a working Triton Inference Server endpoint, end to end. Use when the user has a .onnx file and wants it served, wants a config.pbtxt generated from the ONNX graph, or wants to debug a Triton load or inference failure on an ONNX model. Use triton-pytorch-model-repository for .pt files and triton-tensorrt-model-repository for .plan engines.
license: Apache-2.0
metadata:
  author: Vinya Kestur <vinyak@nvidia.com>
  tags:
  - triton
  - inference
  - onnx
  - model-repository
  - config-pbtxt
---

# Triton ONNX Model Repository

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## When to Use This Skill

Trigger when the user:

- Has a `.onnx` file and wants it served on Triton Inference Server.
- Wants a `config.pbtxt` generated from an ONNX graph.
- Is debugging a Triton load or inference failure on an ONNX model.
- Is writing a Python or REST client for a served ONNX model.

Do not trigger for `.pt` (TorchScript, use triton-pytorch-model-repository), `.plan` (TensorRT engines), LLM serving on Triton, Kubernetes deployment, or ensembles.

## Purpose

Turn a user's `.onnx` file into a running Triton Inference Server endpoint the user's code can call. Read the graph, build the repository, write `config.pbtxt` only when needed, start the server, verify the load, return a working client snippet.

## Prerequisites

- Read access to the `.onnx` file.
- Docker installed. For GPU targets, the NVIDIA Container Toolkit set up so `docker run --gpus all nvidia-smi` prints a device.
- Python 3.10+ with the `onnx` package.
- Either `tritonclient` or `curl` for the smoke test.
- Host ports 8000, 8001, 8002 free, or overrides supplied in `ports`.

## Required Inputs

Collect or infer these before changing anything. Ask only for what cannot be inferred:

- `onnx_path`: absolute path to the `.onnx` file.
- `model_name`: default is the file's stem in lowercase, hyphenated.
- `repo`: absolute path to the model repository root. Default: `${PWD}/triton-repo`.
- `target`: `cpu` or `gpu`. Default: `gpu` if the host has one and the user did not say otherwise, else `cpu`.
- `batching`: `none` | `auto` | `explicit:<max_batch_size>`. Default: `auto` if the ONNX graph has a dynamic leading axis, else `none`.
- `versioning`: `single` | `canary:<v1,v2>`. Default: `single`.
- `cache`: `on` | `off`. Default: `off`.
- `accelerator`: `none` | `tensorrt` | `openvino`. Default: `none`. Set `tensorrt` for GPU production to get the ORT-TRT 2x speedup. Set `openvino` for Intel CPU.
- `ports`: three integers `<http>,<grpc>,<metrics>`. Default: `8000,8001,8002`. Change if the defaults are occupied.
- `execute`: `true` to run commands here, `false` to only produce the exact commands. Default: `true`.
- `review`: `true` to pause and show the generated `config.pbtxt` before starting the server, `false` to skip the pause. Default: `true`.

## Instructions

### 1. Preflight

Run read-only checks first. Do not mutate anything yet.

```bash
test -f "${onnx_path}"
python3 -c "import onnx; onnx.checker.check_model('${onnx_path}')"
docker version >/dev/null
# GPU target only:
docker run --rm --gpus all nvcr.io/nvidia/tritonserver:26.10-py3 nvidia-smi -L
```

Stop and ask the user if any check fails.

### 2. Inspect the ONNX graph

Read names, dtypes, and shapes from the graph. Do not assume them.

```python
import onnx
m = onnx.load(onnx_path)
inputs  = [(i.name, i.type.tensor_type.elem_type, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
outputs = [(o.name, o.type.tensor_type.elem_type, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]
```

If any input's leading axis is a string (e.g., `"N"`), treat the graph as batchable.

### 3. Build the model repository

```
${repo}/
  ${model_name}/
    1/
      model.onnx
```

The file must be named `model.onnx`. The version directory must be a positive integer.

If the ONNX file has sidecar files (external weights, custom ops), either keep them next to `model.onnx` in the version directory, or wrap the whole set in a directory also named `model.onnx/` with the main file inside:

```
${model_name}/
  1/
    model.onnx/
      model.onnx
      <sidecar files>
```

### 4. Write config.pbtxt only if required

Skip when all of these hold: `batching` is `none` or `auto`, `versioning` is `single`, `cache` is `off`, `accelerator` is `none`, and no graph name contains `:0` or `/`. Auto-complete handles the rest.

Otherwise, write the file using `references/config-pbtxt.md`. Use `backend: "onnxruntime"`, not `platform:`. Match input and output names to the graph exactly.

When `accelerator` is `tensorrt`, append:

```
optimization { execution_accelerators {
  gpu_execution_accelerator: [{ name: "tensorrt"
    parameters { key: "precision_mode" value: "FP16" }
    parameters { key: "max_workspace_size_bytes" value: "1073741824" }
  }]
}}
```

ORT-TRT compiles the graph on first load, which can take minutes for large models. Pair with `model_warmup` so the compile happens at load time, not on the first request.

When `accelerator` is `openvino`, append:

```
optimization { execution_accelerators {
  cpu_execution_accelerator: [{ name: "openvino" }]
}}
```

When this step writes a `config.pbtxt`, print the full file contents to the user and pause for confirmation before moving to Step 5. Skip the pause when `review` is `false` or `execute` is `false`.

### 5. Start Triton

```bash
HTTP_PORT="$(echo ${ports} | cut -d, -f1)"
GRPC_PORT="$(echo ${ports} | cut -d, -f2)"
METRICS_PORT="$(echo ${ports} | cut -d, -f3)"

docker run -d --rm --name triton \
  --gpus all \
  -p "${HTTP_PORT}":8000 -p "${GRPC_PORT}":8001 -p "${METRICS_PORT}":8002 \
  -v "${repo}:/models:ro" \
  nvcr.io/nvidia/tritonserver:26.10-py3 \
  tritonserver --model-repository=/models --exit-on-error=false \
  $( [ "${cache}" = "on" ] && echo --cache-config local,size=1048576 )
```

For CPU targets, drop `--gpus all`. Wait for `/v2/health/live` to return 200 before continuing.

### 6. Verify the load

```bash
curl -s http://localhost:"${HTTP_PORT}"/v2/health/ready
curl -s -X POST http://localhost:"${HTTP_PORT}"/v2/repository/index
curl -s http://localhost:"${HTTP_PORT}"/v2/models/"${model_name}"/config | jq
```

Confirm the model appears with `"state": "READY"`. On any other state, open `references/errors.md`.

### 7. Smoke test from a client

Generate a Python snippet from the graph read in Step 2. Use the graph's exact names, the matching numpy dtype, and shape with a leading `1` when the model is batchable.

```python
import numpy as np, tritonclient.http as hc
c = hc.InferenceServerClient("localhost:<http_port>")
x0 = np.ones(<input0_shape>, dtype=np.float32)
inp = [hc.InferInput("<input0_name>", x0.shape, "FP32")]
inp[0].set_data_from_numpy(x0)
r = c.infer("<model_name>", inp)
print(r.as_numpy("<output_name>"))
```

Use the KFServing v2 datatype spelling (`"FP32"`, not `TYPE_FP32`) on the client side. See `references/datatypes.md`.

### 8. Return the Output Contract

Produce the fields listed under "Output Contract" below. If `execute` is `false`, produce the commands without running them.

## Output Contract

Return to the user:

- **Model repository path.** Absolute path on disk.
- **config.pbtxt.** The file contents if written. "Auto-completed" otherwise.
- **Server command.** The literal `docker run` or `tritonserver` command.
- **Client snippet.** Python code that matches the graph's names and shapes exactly.
- **Smoke-test result.** The output tensor (shape, dtype, first few values). "Not executed" if `execute` is `false`.
- **Metrics URL.** `http://<host>:8002/metrics`.
- **Blockers.** Any step that failed, with the exact error and the row from `references/errors.md` that matches.

## Available Scripts

None. Use built-in Python and shell. For config validation, consult `references/config-pbtxt.md` against the Step 2 graph read.

## Examples

**Fixed-shape CPU, no config.**

```bash
mkdir -p repo/add/1 && cp model.onnx repo/add/1/model.onnx
docker run -d --rm --name triton -p 8000:8000 -v $(pwd)/repo:/models:ro \
  nvcr.io/nvidia/tritonserver:26.10-py3 tritonserver --model-repository=/models
curl -s http://localhost:8000/v2/models/add/config | jq
```

**Dynamic-shape GPU, batching, cache.** Write `repo/mymodel/config.pbtxt` with `backend: "onnxruntime"`, `max_batch_size: 16`, `dynamic_batching { preferred_batch_size: [8] }`, `instance_group [{ kind: KIND_GPU }]`, `response_cache { enable: true }`. Start with `--cache-config local,size=1048576 --gpus all`.

**Canary v1 + v3.** Place `1/model.onnx` and `3/model.onnx`. Add `version_policy: { specific: { versions: [1, 3] } }` to `config.pbtxt`. Client picks via `model_version="3"`.

## Limitations

- ONNX only. For PyTorch (`.pt`), use `triton-pytorch-model-repository`. For TensorRT (`.plan`), use `triton-tensorrt-model-repository`.
- Single-model focus. Ensembles and the Triton model orchestrator are out of scope.
- Development setup. Does not configure TLS, authentication, rate limiting, or multi-tenant isolation.
- Does not install Docker, the NVIDIA Container Toolkit, the `onnx` Python package, or `tritonclient`. Prerequisites must already be satisfied.
- Does not run the model repository on object storage (S3/GCS). Local filesystem only.
- Does not create or manage Kubernetes resources. For Kubernetes deployment, use a separate Triton Kubernetes skill.

## Troubleshooting

Short pointers only; the full table with exact error strings is in `references/errors.md`.

| Symptom | Where to look |
|---|---|
| Model missing entirely from `/v2/repository/index` | `references/errors.md` row "silent parse failure" |
| `UNAVAILABLE` with `failed to stat file ...` | File is not named `model.onnx`. Rename, or set `default_model_filename`. |
| `UNAVAILABLE` with `unexpected inference input/output ...` | Name in `config.pbtxt` does not match the ONNX graph. Re-read step 2. |
| `[400] unexpected shape ... Expected [-1,N], got [N]` | Request shape missing batch dim. Reshape client input. |
| `[404] no available versions` | Model failed to load. Check the index. |
| `is_server_ready()` False while models work | Server ready is strict. Use `is_server_live()` and per-model readiness. |
| Load fails with `NOT_IMPLEMENTED : Could not find an implementation for <OpType>(<opset>)` | The op at that opset is beyond what Triton's bundled ONNX Runtime covers. Re-export at a lower opset, or upgrade Triton. |
| Load fails with `ONNXRuntimeError ... not implemented for ...` | Graph contains an operator the backend cannot handle on the target device. Re-export without the op, or run on CPU. |

## Benchmark

See `BENCHMARK.md`. Auto-generated by the NVSkills CI pipeline on `/nvskills-ci`.

## References

- `references/config-pbtxt.md`: every `config.pbtxt` field this skill may write, with the exact rule and example.
- `references/datatypes.md`: the `TYPE_FP32` to `"FP32"` to `np.float32` crosswalk.
- `references/errors.md`: exact Triton 26.10 error strings, cause, and fix.
- `docs/user_guide/metrics.md` in the Triton server repo: per-model and per-GPU counters exposed on port 8002.
- Triton docs in the server repo: `docs/user_guide/model_repository.md`, `docs/user_guide/model_configuration.md`.
- Working examples in the server repo: `qa/L0_rhel_tutorial_cpu/gen_onnx.py` and its `models/add_onnx/config.pbtxt`.
