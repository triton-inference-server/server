<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

Canonical source: [`docs/customization_guide/inference_protocols.md`](../../../docs/customization_guide/inference_protocols.md) and [`docs/user_guide/model_configuration.md`](../../../docs/user_guide/model_configuration.md). Quick reference only.

# Datatype crosswalk

Three different spellings for every datatype. Mismatch is the single most common client-side bug.

| In `config.pbtxt` | In KFServing v2 client (`InferInput`) | In numpy |
|---|---|---|
| `TYPE_FP32` | `"FP32"` | `np.float32` |
| `TYPE_FP16` | `"FP16"` | `np.float16` |
| `TYPE_FP64` | `"FP64"` | `np.float64` |
| `TYPE_INT8` | `"INT8"` | `np.int8` |
| `TYPE_INT16` | `"INT16"` | `np.int16` |
| `TYPE_INT32` | `"INT32"` | `np.int32` |
| `TYPE_INT64` | `"INT64"` | `np.int64` |
| `TYPE_UINT8` | `"UINT8"` | `np.uint8` |
| `TYPE_UINT16` | `"UINT16"` | `np.uint16` |
| `TYPE_UINT32` | `"UINT32"` | `np.uint32` |
| `TYPE_UINT64` | `"UINT64"` | `np.uint64` |
| `TYPE_BOOL` | `"BOOL"` | `np.bool_` |
| `TYPE_STRING` | `"BYTES"` | `np.object_` |

## Server-side rules

Writing `data_type: float32` (lowercase, numpy-style) in `config.pbtxt` fails the protobuf parse. The model disappears from `/v2/repository/index` entirely. The only clue is a line in the server log: `Error parsing text-format inference.ModelConfig: ... Unknown enumeration value of "float32" for field "data_type"`.

## Client-side rules

Writing `httpclient.InferInput("x", shape, "float32")` or `"TYPE_FP32"` on the client side fails. Use the middle column.

## ONNX-to-config mapping

The ONNX `TensorProto.DataType` enum values map to the config spelling:

| ONNX enum | Config |
|---|---|
| `TensorProto.FLOAT` (1) | `TYPE_FP32` |
| `TensorProto.UINT8` (2) | `TYPE_UINT8` |
| `TensorProto.INT8` (3) | `TYPE_INT8` |
| `TensorProto.UINT16` (4) | `TYPE_UINT16` |
| `TensorProto.INT16` (5) | `TYPE_INT16` |
| `TensorProto.INT32` (6) | `TYPE_INT32` |
| `TensorProto.INT64` (7) | `TYPE_INT64` |
| `TensorProto.STRING` (8) | `TYPE_STRING` |
| `TensorProto.BOOL` (9) | `TYPE_BOOL` |
| `TensorProto.FLOAT16` (10) | `TYPE_FP16` |
| `TensorProto.DOUBLE` (11) | `TYPE_FP64` |
| `TensorProto.UINT32` (12) | `TYPE_UINT32` |
| `TensorProto.UINT64` (13) | `TYPE_UINT64` |
