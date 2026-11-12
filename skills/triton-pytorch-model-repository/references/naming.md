<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Canonical sources: `pytorch_backend/src/pt/model_instance_state.cc` (TorchScript resolver), `src/pt2/model_state.cc` (PT2 call-spec parser). Server docs: [`docs/user_guide/model_configuration.md`](../../../docs/user_guide/model_configuration.md).

# Naming

## TorchScript resolver

The backend tries three rules in order. Source comment at `pt/model_instance_state.cc`:

| Priority | Rule | Matches |
|---|---|---|
| 1 | FORWARD_ARGUMENT | Every input name equals a `forward()` argument name. |
| 2 | NAMED_INDEX | Every input name matches `<name>__<index>` with a non-negative integer index. Delimiter `"__"` (double underscore). |
| 3 | STRICT_CONFIG_ORDERING | Fallback. Config order is model order; names are ignored for matching. |

Outputs cannot use FORWARD_ARGUMENT (`libtorch_utils.h`).

## Silent fallback

If names match neither rule 1 nor rule 2, the model loads `READY` and inference maps tensors by config order with no warning.

Concrete example: a `forward(self, a, b)` model that returns `a - b`. Configured with inputs named `b_input` (position 0) and `a_input` (position 1): neither name matches `forward()` arg names (`a`, `b`), neither follows `<name>__<index>`. STRICT_CONFIG_ORDERING fires. The client sends `a_input=[5,5,5,5]` and `b_input=[2,2,2,2]` matching their intent of `a=5, b=2`, expecting `a-b = 3`. The backend assigns config position 0 (`b_input`) to forward arg `a` and position 1 (`a_input`) to forward arg `b`. The model computes `2 - 5 = -3`. The client sees `-3` and no error anywhere.

Avoid by picking one scheme and stating it:

- Clean `forward()` argument names: use FORWARD_ARGUMENT (`name: "image"`, `name: "mask"`).
- Otherwise: use NAMED_INDEX (`INPUT__0`, `INPUT__1`, ... in order).

Do not mix.

## Dictionary input (TorchScript only)

If `forward(self, x)` takes a single `Dict[str, Tensor]`, input names in `config.pbtxt` map to dictionary keys. Source: `pt/model_instance_state.cc`. Multiple Dict inputs and non-Tensor/non-Dict inputs are rejected; see [`errors.md`](errors.md).

```
# forward(self, x: Dict[str, Tensor])
input [
  { name: "image"    data_type: TYPE_FP32 dims: [ 3, 224, 224 ] },
  { name: "metadata" data_type: TYPE_FP32 dims: [ 8 ] }
]
```

## PT2 call-spec and ordinal names

Both schemes work and may coexist in one `config.pbtxt`:

- Call-spec: `ARGS[0]`ARGS[1]`KWARGS[kwarg_name]`RESULT` or `RESULT[0]`.
- Ordinal: `INPUT__0`, `INPUT__1`, `OUTPUT__0`.

A PT2 model returning a bare Tensor has a single output named `RESULT` by default (`pt2/model_state.cc`). Configure as `name: "RESULT"` or `name: "OUTPUT__0"`.

## Sequence batching naming

Sequence control and implicit-state tensors on PT2 accept either ordinal `INPUT__<index>` / `OUTPUT__<index>` names or the PT2 call-spec forward-argument names (`ARGS[...]`, `KWARGS[...]`, `RESULT[...]`). Source: `pt2/model_instance_state.cc`. See [`sequence-batching.md`](sequence-batching.md).

## PT2 input count check

The regular `input [ ... ]` count must equal the model's call-spec input count. Sequence control, implicit-state, and batch inputs are counted separately but contribute to the total expected by the model. See [`errors.md`](errors.md).

## Recommendations

1. New models: NAMED_INDEX (`INPUT__0`, `OUTPUT__0`).
2. Never mix schemes within one model.
3. Model loads `READY` but inference returns garbage: suspect STRICT_CONFIG_ORDERING fallback. Rename to NAMED_INDEX.
