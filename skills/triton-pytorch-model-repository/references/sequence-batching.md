<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Canonical sources: `pytorch_backend/src/pt/model_instance_state.cc` (TorchScript) and `src/pt2/model_instance_state.cc` (PT2). Server docs do not cover PT2 sequence batching.

# Sequence batching

`pytorch_libtorch` and `torch_aoti` both support sequence batching on 26.10. Naming constraints differ.

## TorchScript

NAMED_INDEX naming is required on every `input`, every `output`, and the `initial_state.name` inside the `state` block when `sequence_batching { state { ... } }` is set. A `name: "initial state"` (with a space) fails the check. Mixing schemes is rejected (`pt/model_instance_state.cc`). Not compatible with `ENABLE_WEIGHT_SHARING: true` (`README.md`).

Minimal example:

```
name: "stateful_ts"
platform: "pytorch_libtorch"
max_batch_size: 4

input  [ { name: "INPUT__0"  data_type: TYPE_FP32 dims: [ 16 ] } ]
output [ { name: "OUTPUT__0" data_type: TYPE_FP32 dims: [ 16 ] } ]

sequence_batching {
  max_sequence_idle_microseconds: 5000000
  control_input [
    {
      name: "CONTROL_SEQUENCE_START__1"
      control [ { kind: CONTROL_SEQUENCE_START, int32_false_true: [ 0, 1 ] } ]
    },
    {
      name: "CONTROL_SEQUENCE_END__2"
      control [ { kind: CONTROL_SEQUENCE_END, int32_false_true: [ 0, 1 ] } ]
    },
    {
      name: "CONTROL_SEQUENCE_CORRID__3"
      control [ { kind: CONTROL_SEQUENCE_CORRID, data_type: TYPE_UINT64 } ]
    }
  ]
}

instance_group [ { kind: KIND_CPU, count: 1 } ]
```

With implicit state the config adds a `state` block. Every name in it must also follow `<name>__<index>` including `initial_state.name`:

```
sequence_batching {
  max_sequence_idle_microseconds: 5000000
  state [
    {
      input_name: "INPUT__1"
      output_name: "OUTPUT__1"
      data_type: TYPE_FP32
      dims: [ 4 ]
      initial_state {
        data_type: TYPE_FP32
        dims: [ 4 ]
        zero_data: true
        name: "initial_state__9"
      }
    }
  ]
}
```

## PT2 (implicit state)

The sequence scheduler delivers control and state tensors as regular request inputs. The PT2 model's `forward` must accept them as positional tensor arguments and return any new state as an output. Source comments at `pt2/model_instance_state.cc`.

Constraints:

- **Naming**: sequence control and implicit-state tensors must resolve to one of the model's inputs. Two naming schemes are accepted: ordinal `INPUT__<index>` / `OUTPUT__<index>` names, or the PT2 call-spec forward-argument names (`ARGS[0]`, `KWARGS[kwarg_name]`, `RESULT[0]`, etc.). The QA test `qa/L0_torch_aoti/torch_aoti_infer_test.py` exercises both schemes against the same stateful model. Source: `pt2/model_instance_state.cc`.
- **Correlation ID**: must be numeric. `TYPE_STRING` CORRID is rejected on PT2 (`pt2/model_instance_state.cc`). Use `TYPE_INT32` or `TYPE_INT64`.
- **Input accounting**: the regular `input [ ... ]` array only lists data inputs. Sequence control inputs and batch inputs are declared separately and each contributes one input toward the model's forward arity. Each `state` block contributes one input (the `input_name`) and one output (the `output_name`). Source: `pt2/model_instance_state.cc`.
- **Dynamic batch dim** still required on the PT2 archive. See [`torch-aoti.md`](torch-aoti.md).

## Debugging

If `forward()` takes 5 tensors (2 data + 1 state + 2 controls), the config must declare 2 regular inputs, 1 state block (contributing 1 input), and 2 control inputs. A short sequence (3-4 requests on one correlation ID) is the quickest way to validate the plumbing before scaling.

## Recommendations

1. Prefer TorchScript unless PT2 has a specific advantage.
2. On PT2:
   - Design `forward()` to take data, state, and control as positional args.
   - Return new state as explicit outputs.
   - `TYPE_INT64` for `CONTROL_SEQUENCE_CORRID`.
   - Ordinal `INPUT__<i>` / `OUTPUT__<i>` naming throughout.
