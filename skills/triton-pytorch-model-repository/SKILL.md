---
name: triton-pytorch-model-repository
description: Deploy a user's PyTorch model as a working Triton Inference Server endpoint, end to end. Use when the user has a TorchScript .pt file or a PT2 archive (.pt2 from torch.export + AOT Inductor) and wants it served, wants a config.pbtxt written from scratch, or wants to debug a Triton load or inference failure on a PyTorch model. Use triton-onnx-model-repository for .onnx files and triton-tensorrt-model-repository for .plan engines.
license: Apache-2.0
metadata:
  author: Vinya Kestur <vinyak@nvidia.com>
  tags:
    - triton
    - inference
    - pytorch
    - torchscript
    - torch-aoti
    - model-repository
    - config-pbtxt
---

# Triton PyTorch Model Repository

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## When to Use

Trigger on `.pt`, `.pt2`, `config.pbtxt` for a PyTorch model, or a Triton load or inference failure on a PyTorch model. Hand off `.onnx` to `triton-onnx-model-repository` and `.plan` to `triton-tensorrt-model-repository`. Not for training.

## Hard Constraints

1. **No auto-complete.** Every input, output, datatype, and shape must be declared in `config.pbtxt`. Source: `pt/model_state.cc`, `pt2/model_state.cc`.
2. **No BF16, no FP8, no UINT16+.** Cast to FP16 or FP32, or use the ONNX or TensorRT backend. Source: `libtorch_utils.cc`. See [`references/datatypes.md`](references/datatypes.md).
3. **PT2 archives are device-baked.** CPU export cannot serve on `KIND_GPU` and vice versa. Re-export to change device. See [`references/torch-aoti.md`](references/torch-aoti.md).
4. **Misnamed inputs load but produce wrong results.** If names match neither `forward()` nor `<name>__<index>`, the backend falls back to strict config order. See [`references/naming.md`](references/naming.md).
5. **Cache-cleaning parameter is named differently per platform.** `ENABLE_CACHE_CLEANING` on TorchScript, `CACHE_CLEANING_ENABLED` on PT2. See [`references/parameters.md`](references/parameters.md).

## Platform Form

| Form | `config.pbtxt` | Model file |
|---|---|---|
| TorchScript, C++ runtime | `platform: "pytorch_libtorch"` | `model.pt` |
| TorchScript, Python runtime | `backend: "pytorch"` + `runtime: "model.py"` | `model.py` + `model.pt` |
| AOT Inductor (PT2) | `backend: "pytorch"` + `platform: "torch_aoti"` | `model.pt2` |

Default filenames overridable with `default_model_filename`.

## Deploy TorchScript

Layout:

```
model_repository/
  <name>/
    config.pbtxt
    1/model.pt
```

Minimal `config.pbtxt`:

```
name: "my_model"
platform: "pytorch_libtorch"
max_batch_size: 8
input  [ { name: "INPUT__0"  data_type: TYPE_FP32 dims: [ 3, 224, 224 ] } ]
output [ { name: "OUTPUT__0" data_type: TYPE_FP32 dims: [ 1000 ] } ]
instance_group [ { kind: KIND_GPU } ]
```

Launch:

```bash
tritonserver --model-repository=/models
```

`--disable-auto-complete-config` is **optional** and only silences the PyTorch backend's "auto-complete not supported" warning at startup. Do not add it when the same model repository also holds ONNX, TensorFlow, or TensorRT models that rely on auto-complete: the flag is server-wide and those models will fail to load.

## Deploy PT2

Build the archive with a dynamic batch dim; sample input must have batch > 1 so `torch.export` does not specialize:

```python
import torch

m = Model().eval()                              # add .cuda() for KIND_GPU
sample = (torch.randn(2, 4), torch.randn(2, 4)) # .cuda() each for KIND_GPU
batch = torch.export.Dim("batch", min=1, max=8)
exported = torch.export.export(
    m, sample, dynamic_shapes=({0: batch}, {0: batch})
)
torch._inductor.aoti_compile_and_package(exported, package_path="model.pt2")
```

Minimal `config.pbtxt`:

```
name: "my_model"
backend: "pytorch"
platform: "torch_aoti"
max_batch_size: 8
input  [ { name: "INPUT__0" data_type: TYPE_FP32 dims: [ 4 ] } ]
output [ { name: "OUTPUT__0" data_type: TYPE_FP32 dims: [ 4 ] } ]
instance_group [ { kind: KIND_CPU } ]
```

`Dim("batch", max=N)` must be `>= max_batch_size`. `instance_group` kind must match the export device.

## Build vs Serve Containers

`nvcr.io/nvidia/tritonserver:26.10-py3` ships libtorch but no Python `torch`. Build artifacts in `nvcr.io/nvidia/pytorch:26.10-py3` or install `torch` before building. Match build and serve to 26.10 to avoid opset drift.

## Inference (HTTP KFServing v2)

```bash
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"inputs":[{"name":"INPUT__0","shape":[1,4],"datatype":"FP32","data":[1,1,1,1]}]}' \
  http://localhost:8000/v2/models/my_model/infer
```

Large tensors: use the binary tensor extension. See [`references/config-pbtxt.md`](references/config-pbtxt.md).

## Debug a Failed Load

Validation order: file exists → input count matches → datatype supported → naming resolves. Fix in order; each later check only runs if earlier ones pass.

| Error (server log) | Fix |
|---|---|
| `unable to find '/models/<name>/1/model.pt'` | Place `model.pt` at `1/` or set `default_model_filename`. |
| `PyTorch inductor model file "..." is unreachable or inaccessible.` | Place `model.pt2` at `1/`. |
| `unsupported datatype TYPE_<X>` (TorchScript) | See [`references/datatypes.md`](references/datatypes.md). Cast or switch backend. |
| `Unsupported datatype TYPE_<X> ... model instance "<name>_0_0".` (PT2) | Same recovery; note PT2 wording differs. |
| `configuration expects N inputs, but model expects M inputs.` | Match input count in `config.pbtxt` to the exported call spec. |
| `Incorrect device passed to aoti_runner_cpu` | Re-export PT2 on the target device. |
| `PyTorch execute failure: output must be of type Tensor, List[str] or Tuple ...` (inference-time) | Return Tensor, 1-D `List[str]`, or a Tuple. Wrap scalars with `torch.tensor(...)`. |
| `PyTorch execute failure: output ... received List[Tensor]` (inference-time) | Return a Tuple of Tensors, not a List. |
| `failed to split the output tensor '<name>' in responses: expected batch size of at least N ...` (PT2 inference-time) | PT2 archive lacks dynamic batch Dim. Re-export with `torch.export.Dim`. |
| `sequence batching with state but not all ... <name>__<index>` | Rename every input, output, and `initial_state.name` to `<name>__<index>`. |

Full catalog: [`references/errors.md`](references/errors.md).

## Performance

- Backend is BLOCKING (`libtorch.cc`). Add instance-group entries, not threads.
- `INTRA_OP_THREAD_COUNT` and `INTER_OP_THREAD_COUNT` are process-global; first model wins (`pt/model_state.cc`).
- `INFERENCE_MODE` default `true` is stronger than NoGrad; leave on.
- TorchScript JIT compiles on first invocation; use `model_warmup` to amortize.
- `KIND_MODEL` lets the model's own `.to(device)` win; inputs land on CPU, output path forces `cudaStreamSynchronize` (`pt/model_instance_state.cc`).

See [`references/parameters.md`](references/parameters.md).

## Metrics

Standard Triton metrics at `GET /metrics` on port 8002. No PyTorch-specific metrics. See [`docs/user_guide/metrics.md`](../../docs/user_guide/metrics.md).

## Hand Off

- User needs BF16, FP8, UINT16+: ONNX or TensorRT backend.
- User wants Dynamo (vLLM, TRT-LLM router): Dynamo skills.
- User wants TorchServe features: different product.
- CUDA OOM or device-side assertion: model issue, not Triton config.

## References

- [`references/config-pbtxt.md`](references/config-pbtxt.md): field reference and examples.
- [`references/datatypes.md`](references/datatypes.md): supported crosswalk, unsupported list.
- [`references/errors.md`](references/errors.md): error catalog with recovery.
- [`references/parameters.md`](references/parameters.md): TorchScript and PT2 parameters.
- [`references/naming.md`](references/naming.md): naming resolver rules.
- [`references/torch-aoti.md`](references/torch-aoti.md): PT2 export, device-baked, `MODEL_INIT_LIBRARY`.
- [`references/sequence-batching.md`](references/sequence-batching.md): sequence batching on `pytorch_libtorch` and `torch_aoti`.
- Canonical docs: [`docs/user_guide/model_configuration.md`](../../docs/user_guide/model_configuration.md), [`docs/user_guide/model_repository.md`](../../docs/user_guide/model_repository.md).
- Backend source: `triton-inference-server/pytorch_backend` on branch `r26.10`.
