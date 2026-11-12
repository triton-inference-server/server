<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Canonical sources: [`docs/user_guide/model_management.md`](../../../docs/user_guide/model_management.md), [`docs/user_guide/model_repository.md`](../../../docs/user_guide/model_repository.md), and the `pytorch_backend` source files cited below.

# Error catalog

Validation order: file exists → input count matches → datatype supported → naming resolves → runtime shape/dtype. Fix in order; later checks only run after earlier ones pass.

## Load

| Error (server log) | Source | Fix |
|---|---|---|
| `unable to find '<path>' for model instance '<name>'` | `pt/model_state.cc` | Place `model.pt` at `1/` or set `default_model_filename`. |
| `PyTorch inductor model file "<path>" is unreachable or inaccessible.` | `pt2/model_state.cc` | Place `model.pt2` at `1/` or set `default_model_filename`. |
| `failed to load model '<name>': <exception>` | `pt/model_state.cc` | Opset drift or invalid archive. Rebuild in `nvcr.io/nvidia/pytorch:26.10-py3`. |
| `Failed to parse input call specification for model "<name>".` | `pt2/model_state.cc` | Malformed `.pt2`. Re-export with matching `torch`. |
| `Unexpected input call specification type "<type>" ... Expected type is "builtins_tuple".` | `pt2/model_state.cc` | Top-level call spec must be a tuple. Re-export. |
| `Failed to dlopen MODEL_INIT_LIBRARY "<path>": <dlerror>` | `pt2/model_state.cc` | Check path, architecture, dependencies (`ldd`). |
| `MODEL_INIT_LIBRARY "<path>" does not export 'triton_pytorch_model_init' for model "<name>": ...` | `pt2/model_state.cc` | Library must export `extern "C" void triton_pytorch_model_init(const char*, int)`. See [`parameters.md`](parameters.md). |

## Config validation

| Error (server log) | Source | Fix |
|---|---|---|
| `model configuration must contain at least one input, none were specified.` | `pt/model_instance_state.cc` | Add an `input [ { ... } ]` block. |
| `model configuration must contain at least one output, none were specified.` | `pt/model_instance_state.cc` | Add an `output [ { ... } ]` block. |
| `unsupported datatype TYPE_<X> for input '<name>' for model '<model>'` | `pt/model_instance_state.cc` | See [`datatypes.md`](datatypes.md). Cast or switch backend. |
| `Unsupported datatype TYPE_<X> for input "<name>" for model instance "<model>_0_0".` | `pt2/model_instance_state.cc` | Same recovery; note PT2 wording differs. |
| `Model instance "<name>" has mismatching input count ... config input count = X, model definition input count = Y` | `pt2/model_instance_state.cc` | Match input count. Control, implicit-state, batch inputs count toward the total but live in separate blocks. |
| `Incorrect device passed to aoti_runner_cpu` | upstream libtorch at `model_container_runner_cpu.cpp:39` | PT2 archive is device-baked. Re-export on the target device. See [`torch-aoti.md`](torch-aoti.md). |

## Naming

| Error (server log) | Source | Fix |
|---|---|---|
| `input '<name>' does not follow <name>__<index> naming convention.` | `pt/model_instance_state.cc` | Use NAMED_INDEX or `forward()` argument names consistently. See [`naming.md`](naming.md). |
| `PyTorch model '<name>' is using sequence batching with state but not all inputs and outputs follow the <name>__<index> naming convention.` | `pt/model_instance_state.cc` | Rename every input and output to `<name>__<index>`. |
| `Sequence control input (<kind>) "<name>" for model "<model>" does not correspond to any model input. ... ordinal "INPUT__<index>" naming convention.` | `pt2/model_instance_state.cc` | Rename sequence control and state to `INPUT__<i>` / `OUTPUT__<i>`. |

## TorchScript model shape (load)

| Error (server log) | Source | Fix |
|---|---|---|
| `Multiple inputs of kind DictType were detected. Only a single input of type Dict(str, Tensor) is supported.` | `pt/model_instance_state.cc` | At most one `Dict[str, Tensor]` input. |
| `An input of type '<type>' was detected in the model. Only a single input of type Dict(str, Tensor) or input(s) of type Tensor are supported.` | `pt/model_instance_state.cc` | Only Tensor and `Dict[str, Tensor]` accepted. |
| `unable to load model '<name>', configuration expects X inputs, model provides Y` | `pt/model_instance_state.cc` | Match the `config.pbtxt` input count to the `forward()` arity. Sequence control, state, and batch inputs count toward the total. |
| `configuration expects datatype TYPE_<X> for input '<name>', model provides TYPE_<Y>` | `pt/model_instance_state.cc` | Config's `data_type` for the input does not match the exported model's dtype. Align the `config.pbtxt` to the model, or re-export the model with the target dtype. |
| `output tensor '<name>' is not found` | `pt/model_instance_state.cc` | Rename per [`naming.md`](naming.md). |

## Inference (TorchScript)

Scalar and nested-list output errors fire when a request is served, not at load. The model loads `READY` and only fails when inference runs.

| Error (API response) | Source | Fix |
|---|---|---|
| `PyTorch execute failure: output must be of type Tensor, List[str] or Tuple containing one of these two types. It should not be a List / Dictionary of Tensors or a Scalar` | `pt/model_instance_state.cc` | Wrap scalar returns with `torch.tensor(...)`. |
| `PyTorch execute failure: output must be of type Tensor or List[str], received List[Tensor]` | `pt/model_instance_state.cc` | Return a Tuple of Tensors, not a Python list. |
| `only 1-dimensional List[str] input / output supported for TorchScript PyTorch backend` | `pt/model_instance_state.cc`, backend `README.md` | Flatten strings to 1-D. PT2 has no string support. |

## Inference (PT2)

| Error (API response) | Source | Fix |
|---|---|---|
| `failed to split the output tensor '<name>' in responses: expected batch size of at least N in model output, got M` | Triton core response splitter | PT2 archive was exported without a dynamic batch dim (`torch.export.Dim`). Re-export with `Dim("batch", min=1, max=N)` so the model handles the requested batch size. |

## Runtime note

CUDA errors in TorchScript's async execution may surface on a later request. If an unrelated inference starts failing, inspect the model that last ran. Server restart may be required. Reference: `pytorch_backend/README.md`.
