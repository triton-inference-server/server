<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Canonical source: `pytorch_backend/src/libtorch_utils.cc`. Both `pytorch_libtorch` and `torch_aoti` share this mapping.

# Datatype crosswalk

## Supported

| Triton `TYPE_*` | PyTorch | File:line |
|---|---|---|
| `TYPE_BOOL`   | `kBool`   | `libtorch_utils.cc` |
| `TYPE_UINT8`  | `kByte`   | `libtorch_utils.cc` |
| `TYPE_INT8`   | `kChar`   | `libtorch_utils.cc` |
| `TYPE_INT16`  | `kShort`  | `libtorch_utils.cc` |
| `TYPE_INT32`  | `kInt`    | `libtorch_utils.cc` |
| `TYPE_INT64`  | `kLong`   | `libtorch_utils.cc` |
| `TYPE_FP16`   | `kHalf`   | `libtorch_utils.cc` |
| `TYPE_FP32`   | `kFloat`  | `libtorch_utils.cc` |
| `TYPE_FP64`   | `kDouble` | `libtorch_utils.cc` |
| `TYPE_STRING` | special-cased, TorchScript only | `pt/model_instance_state.cc` |

The shipped [`docs/user_guide/model_configuration.md`](../../../docs/user_guide/model_configuration.md) datatype table shows the FP16 cell blank for PyTorch. The backend accepts it; `qa/L0_torch_aoti/torch_aoti_infer_test.py` exercises it on PT2.

## Not supported

`TYPE_BF16`, `TYPE_FP8_E4M3`, `TYPE_FP8_E5M2`, `TYPE_UINT16`, `TYPE_UINT32`, `TYPE_UINT64`, `TYPE_BYTES`. All hit the default fallthrough at `libtorch_utils.cc`. `TYPE_UINT32` and `TYPE_UINT64` explicitly rejected in `qa/L0_libtorch_io_types/test.sh`.

Error at load:

- TorchScript: `unsupported datatype TYPE_<X> for input '<name>' for model '<model>'` (`pt/model_instance_state.cc`).
- PT2: `Unsupported datatype TYPE_<X> for input "<name>" for model instance "<model>_0_0".` (`pt2/model_instance_state.cc`).

## Per-platform string handling

- TorchScript: accepts `TYPE_STRING` only as 1-D `List[str]` (`pt/model_instance_state.cc`, backend `README.md`).
- PT2: no `TYPE_STRING` support. `CONTROL_SEQUENCE_CORRID` must be `TYPE_INT32` or `TYPE_INT64` (`pt2/model_instance_state.cc`).

## Known upstream bug

Traced TorchScript models can overflow when `int8` tensor values are transformed to `int32` on GPU. Script instead of trace, or cast inputs to `int32` or `int64`. Reference: Triton 26.10 release notes and upstream `pytorch/pytorch#66930`.

## When the model needs BF16 or FP8

Not supported here. Options:

1. Cast weights and activations to FP16 or FP32.
2. Export to ONNX and use `triton-onnx-model-repository`.
3. Build a TensorRT engine and use `triton-tensorrt-model-repository`.
