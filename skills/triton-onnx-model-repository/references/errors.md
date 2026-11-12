<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Canonical source for loading and inference behavior: [`docs/user_guide/model_management.md`](../../../docs/user_guide/model_management.md) and [`docs/user_guide/model_repository.md`](../../../docs/user_guide/model_repository.md). This file lists the exact error strings and recovery steps the skill matches on.

# Error catalog (Triton 26.10)

Match the symptom, apply the fix.

## Load-time errors

### Silent parse failure

**Symptom.** Model is MISSING ENTIRELY from `/v2/repository/index`. No entry, no `UNAVAILABLE`. Server log has:

```
[libprotobuf ERROR ...] Error parsing text-format inference.ModelConfig:
    N:N: Unknown enumeration value of "float32" for field "data_type".
[E ... model_repository_manager.cc:1506] "Poll failed for model directory 'X':
    failed to read text proto from /models/X/config.pbtxt"
```

**Cause.** A field value does not parse. Common: `data_type: float32` (lowercase) instead of `TYPE_FP32`.

**Fix.** Use enum spellings from `references/datatypes.md`. Re-read `config.pbtxt` for any bare-word value where an `enum` is expected (`data_type`, `kind`, policy fields).

### Non-integer version directory

**Symptom.** Model is MISSING from `/v2/repository/index`. Server log has:

```
[W ...] "ignore version directory 'v1' which fails to convert to integral number"
[E ...] "failed to load model '<model>':
    at least one version must be available under the version policy of model '<model>'"
```

**Cause.** Version directory is not an integer.

**Fix.** Rename `v1/` to `1/`. Only non-negative integers are version names.

### Wrong model filename

**Symptom.** Index shows `UNAVAILABLE` with:

```
Internal: failed to stat file /models/<model>/1/model.onnx
```

**Cause.** The ONNX file in the version directory is not named `model.onnx`.

**Fix.** Rename to `model.onnx`, or add `default_model_filename: "<your-name>.onnx"` to `config.pbtxt`.

### No model file

**Symptom.** Index shows name only, no `state`. Server log has:

```
Could not determine backend for model '<model>' with no backend in model configuration.
    Expected model name of the form 'model.<backend_name>'.
```

**Cause.** The version directory is empty or has no file, AND the config does not set `backend:`.

**Fix.** Put `model.onnx` in the version directory, or add `backend: "onnxruntime"` to `config.pbtxt`.

### Input name mismatch

**Symptom.** Index shows `UNAVAILABLE` with:

```
Invalid argument: unexpected inference input 'X', allowed inputs are: INPUT0, INPUT1
```

**Cause.** Input name in `config.pbtxt` does not match the ONNX graph. Common with TF-converted graphs (`:0` suffixes).

**Fix.** Rename the input in `config.pbtxt` to one of the listed names. Re-inspect the graph with `onnx.load(path).graph.input`.

### Output name mismatch

**Symptom.** Index shows `UNAVAILABLE` with:

```
Invalid argument: unexpected inference output 'X', allowed outputs are: OUTPUT0
```

**Cause.** Same as input mismatch, for output names.

**Fix.** Rename to one of the listed names.

## Inference-time errors

### Missing batch dim

**Symptom.** HTTP 400 at inference:

```
unexpected shape for input 'X' for model '<model>'.
    Expected [-1,N], got [N].
    NOTE: Setting a non-zero max_batch_size in the model config requires a batch dim
```

**Cause.** Model has `max_batch_size > 0` (set explicitly or by auto-complete). The wire shape must include a batch dim.

**Fix.** Send shape `[1, N]` instead of `[N]`. Or set `max_batch_size: 0` in `config.pbtxt` if batching is not wanted.

### Calling an unavailable model

**Symptom.** HTTP 404:

```
Request for unknown model: '<model>' has no available versions
```

**Cause.** The model failed to load. Its index entry is `UNAVAILABLE`.

**Fix.** Check `/v2/repository/index` for the reason. Fix the load error (see load-time rows). Retry the call.

### Calling a version not covered by policy

**Symptom.** HTTP 404:

```
Request for unknown model: '<model>' version <v> is not found
```

**Cause.** `version_policy` does not include this version.

**Fix.** Add the version to `version_policy: { specific: { versions: [...] } }`, or ask for a version that is covered.

## Server-wide readiness oddities

### `is_server_ready()` False while models work

**Symptom.** `is_server_ready()` returns False, but `/v2/models/<model>/ready` for a model is True and inference works.

**Cause.** Server ready is strict. It is True only when EVERY model is READY. One failed model makes it False.

**Fix.** Use `is_server_live()` for liveness probes, `is_model_ready("<model>")` for routing decisions. Server ready is a catalog-level boolean, not a routing signal.

## ONNX-specific load failures

### Unsupported opset version

**Symptom.** Model load fails with an ORT error:

```
[ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for
    <OpType>(<opset>) node with name '<n>'
```

or

```
Fatal error: <OpType> is not a registered function/op
```

**Cause.** The ONNX graph uses an op at an opset the bundled ONNX Runtime does not implement. Each Triton release ships a specific ORT version with a specific per-op opset coverage; the failure threshold varies per op, not just the model's import opset.

**Fix.** Re-export the model at a lower opset. On recent Triton containers common ops at opset 23 still load; the failure appears on ops beyond what the bundled ORT covers. Or upgrade Triton to a container that bundles a newer ORT.

### Unsupported operator on target device

**Symptom.** Model load fails with:

```
[ONNXRuntimeError] : 1 : FAIL : ... <OpType> is not implemented for <device>
```

**Cause.** The graph contains an op the ORT backend cannot run on the target. Common when a CUDA-only or TRT-only op is routed to CPU, or vice versa.

**Fix.** Re-export the model without the unsupported op, or switch the target. For GPU-only ops on a CPU target, set `instance_group: [{ kind: KIND_GPU }]`.

## Informational

### Autofill warning

**Symptom.** Server log (not an error):

```
autofilled max_batch_size to 4 for model '<model>' since batching is supported
    but no max_batch_size is specified in model configuration.
    Must specify max_batch_size to utilize autofill with a larger max batch size
```

**Cause.** Auto-complete used the server default of 4 for a batchable model.

**Fix.** If 4 is fine, ignore. Otherwise set `max_batch_size` explicitly in `config.pbtxt`, or pass `--backend-config=default-max-batch-size=<N>` at startup.
