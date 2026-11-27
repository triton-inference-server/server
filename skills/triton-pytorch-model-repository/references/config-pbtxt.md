<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Canonical source: [`docs/user_guide/model_configuration.md`](../../../docs/user_guide/model_configuration.md). This file covers only PyTorch-specific usage on Triton 26.10.

# `config.pbtxt` for PyTorch

## Platform form

| Form | `config.pbtxt` lines | Default file | Source |
|---|---|---|---|
| TorchScript, C++ runtime | `platform: "pytorch_libtorch"` | `model.pt` | `pt/model_state.cc` |
| TorchScript, Python runtime | `backend: "pytorch"` + `runtime: "model.py"` | `model.py` + `model.pt` | `qa/L0_pytorch_python_runtime/test.sh` |
| AOT Inductor (PT2) | `backend: "pytorch"` + `platform: "torch_aoti"` | `model.pt2` | `pt2/model_state.hh` |

Platform dispatch: `libtorch.cc`. Default filenames overridable with `default_model_filename`.

## Required fields

- `name`: must match the directory.
- `backend` and `platform`: pick a form from the table above.
- `max_batch_size`: maximum request batch size, or `0` for no batching.
- `input` and `output`: one block per tensor.

## `input` / `output`

```
input [
  {
    name: "INPUT__0"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]
```

- `name`: pick one naming scheme consistently (see [`naming.md`](naming.md)).
- `data_type`: see [`datatypes.md`](datatypes.md).
- `dims`: shape excluding batch; use `-1` for dynamic.
- `reshape { shape: [ ... ] }`: optional reshape between Triton and the model.
- `optional: true`: input may be omitted (TorchScript dict-input mode only).

## `instance_group`

```
instance_group [
  { kind: KIND_GPU, count: 2, gpus: [ 0 ] },
  { kind: KIND_CPU, count: 1 }
]
```

- `KIND_GPU`: inputs on the instance's GPU.
- `KIND_CPU`: inputs on CPU.
- `KIND_MODEL`: the model's own `.to(device)` wins; inputs land on CPU, output path forces `cudaStreamSynchronize` (`pt/model_instance_state.cc`).

More instances adds concurrency. The backend is BLOCKING (`libtorch.cc`).

## `dynamic_batching`

```
dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 100
}
```

- Supported on `pytorch_libtorch` and `torch_aoti`.
- PT2 also requires `torch.export.Dim` on every input's batch axis.
- PT2 does not support batching of nested `dict` / `tuple` / `list` I/O (`pytorch_backend/README.md`).

## `version_policy`, `response_cache`, `model_warmup`, `default_model_filename`

All standard Triton semantics. Use `model_warmup` to amortize TorchScript first-request JIT cost:

```
model_warmup [
  {
    name: "warm_16"
    batch_size: 16
    inputs { key: "INPUT__0" value { data_type: TYPE_FP32 dims: [ 3, 224, 224 ] zero_data: true } }
  }
]
```

## `parameters`

```
parameters { key: "INFERENCE_MODE" value { string_value: "true" } }
```

See [`parameters.md`](parameters.md) for every key, default, and per-platform availability. `INTRA_OP_THREAD_COUNT` and `INTER_OP_THREAD_COUNT` are process-global.

## `sequence_batching`

Supported on `pytorch_libtorch` and `torch_aoti` with different naming requirements. See [`sequence-batching.md`](sequence-batching.md).

## Full TorchScript example

```
name: "resnet50_ts"
platform: "pytorch_libtorch"
max_batch_size: 32

input  [ { name: "INPUT__0"  data_type: TYPE_FP32 dims: [ 3, 224, 224 ] } ]
output [ { name: "OUTPUT__0" data_type: TYPE_FP32 dims: [ 1000 ] } ]

instance_group [ { kind: KIND_GPU, count: 2, gpus: [ 0 ] } ]

dynamic_batching {
  preferred_batch_size: [ 8, 16 ]
  max_queue_delay_microseconds: 100
}

parameters { key: "INFERENCE_MODE" value { string_value: "true" } }
```

## Full PT2 example

```
name: "resnet50_pt2"
backend: "pytorch"
platform: "torch_aoti"
max_batch_size: 32

input  [ { name: "INPUT__0"  data_type: TYPE_FP32 dims: [ 3, 224, 224 ] } ]
output [ { name: "OUTPUT__0" data_type: TYPE_FP32 dims: [ 1000 ] } ]

instance_group [ { kind: KIND_GPU, count: 1, gpus: [ 0 ] } ]
```

The archive at `1/model.pt2` must be exported with `torch.export.Dim("batch", min=1, max=32)` on a CUDA model.
