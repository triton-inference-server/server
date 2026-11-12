<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `config.pbtxt` field reference

Canonical source: [`docs/user_guide/model_configuration.md`](../../../docs/user_guide/model_configuration.md). This file is a quick reference of ONNX-specific usage and examples for the skill. Defer to the canonical doc when a field is added or changed.

## Backend and platform

Pick one, not both.

```
backend: "onnxruntime"
```

Preferred. Modern form. Used in `qa/L0_rhel_tutorial_cpu/models/add_onnx/config.pbtxt`.

```
platform: "onnxruntime_onnx"
```

Older form. Still works. Appears in `docs/examples/model_repository/simple/config.pbtxt`.

## Batching

```
max_batch_size: 0
```

No batching. `dims` is the full tensor shape. Required when the ONNX graph has fixed-size inputs.

```
max_batch_size: 16
dynamic_batching { preferred_batch_size: [8] max_queue_delay_microseconds: 100000 }
```

Dynamic batching with explicit scheduler. Triton prepends a batch dim: a `dims: [4]` input becomes `[-1, 4]` at the wire.

Auto-complete behavior: with auto-complete enabled (the default), a `max_batch_size > 1` and no scheduler causes Triton to attach a `dynamic_batching { }` block automatically. To take the batcher explicitly, write the block as shown above so the parameters are not left to the defaults. To opt one model out of batching, write `max_batch_size: 0` and provide full `input` and `output` blocks in that model's `config.pbtxt`; a complete config leaves nothing for auto-complete to override. Do not use `--disable-auto-complete-config` to opt a single model out: that flag is server-wide and will break any other model that relies on auto-complete.

## Inputs and outputs

Names must match the ONNX graph exactly. Datatypes use `TYPE_*` spelling (see `references/datatypes.md`).

```
input [
  { name: "INPUT0", data_type: TYPE_FP32, dims: [4] },
  { name: "INPUT1", data_type: TYPE_FP32, dims: [4] }
]
output [ { name: "OUTPUT0", data_type: TYPE_FP32, dims: [4] } ]
```

For TensorFlow-converted graphs with `:0` suffixes or `/` paths, quote the name:

```
input  [ { name: "image:0", data_type: TYPE_FP32, dims: [1, 3] } ]
output [ { name: "resnet50/predictions/Softmax:0", data_type: TYPE_FP32, dims: [1, 3] } ]
```

Dynamic axes use `-1`:

```
input [ { name: "INPUT", data_type: TYPE_FP32, dims: [-1] } ]
```

## Instance group

```
instance_group [ { kind: KIND_CPU } ]
```

CPU-only.

```
instance_group [ { count: 2, kind: KIND_GPU, gpus: [0, 1] } ]
```

Explicit GPU placement.

Omit the field entirely for automatic placement: KIND_GPU on every visible GPU (if GPUs are visible), else KIND_CPU.

## Default model filename

To keep the original filename instead of renaming to `model.onnx`:

```
default_model_filename: "mymodel.onnx"
```

Place the file at `<model>/<version>/mymodel.onnx`.

## Version policy

Default (highest numeric version wins): omit the field.

```
version_policy: { specific: { versions: [1, 3] } }
```

Serve v1 and v3 side by side. Client picks via `model_version="3"` in the API call. Other forms: `version_policy: { all: {} }`, `version_policy: { latest: { num_versions: 2 } }`.

## Response cache

Requires BOTH a server flag and a model config entry:

Server: `tritonserver --cache-config local,size=1048576`

Model:
```
response_cache { enable: true }
```

Without the server flag, the model entry is a no-op (no cache is allocated).

## Model warmup

```
model_warmup [
  { name: "zero_sample"
    batch_size: 1
    inputs { key: "INPUT0" value: { data_type: TYPE_FP32 dims: 4 zero_data: true } }
    inputs { key: "INPUT1" value: { data_type: TYPE_FP32 dims: 4 zero_data: true } }
  }
]
```

Model is not READY until every warmup sample completes. A failed warmup fails the load. Triton does not log explicit "warming up" lines; the signal is that READY is delayed.

Other data sources: `random_data: true`, `input_data_file: "sample.bin"`.

## Classification labels

```
output [ { name: "LOGITS", data_type: TYPE_FP32, dims: [1, 5],
           label_filename: "labels.txt" } ]
```

`labels.txt` sits next to `config.pbtxt`, one label per line. Client asks for top-K with `InferRequestedOutput("LOGITS", class_count=3)` and gets back `b"<prob>:<class_id>:<label>"` strings.

## String tensors

```
input  [ { name: "TEXT_IN",  data_type: TYPE_STRING, dims: [-1] } ]
output [ { name: "TEXT_OUT", data_type: TYPE_STRING, dims: [-1] } ]
```

Client sends `np.array(["..."], dtype=object)` with `"BYTES"` datatype. Output comes back as `bytes` objects; decode with `.decode("utf-8")`.

## Acceleration

### ONNX-TensorRT (ORT-TRT), GPU

Published speedup from the Triton docs: 2x throughput, half the latency on a DenseNet ONNX baseline.

```
optimization { execution_accelerators {
  gpu_execution_accelerator: [{ name: "tensorrt"
    parameters { key: "precision_mode" value: "FP16" }
    parameters { key: "max_workspace_size_bytes" value: "1073741824" }
  }]
}}
```

Precision modes: `FP32`, `FP16`, `INT8`. `max_workspace_size_bytes` sets the TRT builder workspace.

ORT-TRT compiles the graph on first load and the compile can take minutes for large models. Pair with `model_warmup` so the compile is amortized into the load time instead of the first request.

### ONNX-OpenVINO, Intel CPU

```
optimization { execution_accelerators {
  cpu_execution_accelerator: [{ name: "openvino" }]
}}
```

Use on Intel CPUs. Non-Intel CPUs gain nothing or fail.

## Minimum viable template

Starting point for most cases. Fill the names, dtypes, and dims from the ONNX graph.

```
name: "<model_name>"
backend: "onnxruntime"
max_batch_size: 0
input [
  { name: "<name>", data_type: TYPE_FP32, dims: [<shape>] }
]
output [
  { name: "<name>", data_type: TYPE_FP32, dims: [<shape>] }
]
instance_group [ { kind: KIND_CPU } ]
```
