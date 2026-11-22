<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Canonical sources: `pytorch_backend/src/pt/model_state.cc` (TorchScript), `src/pt2/model_state.cc` (PT2).

# `parameters` block

Every value is a `string_value` even when the semantic is a boolean or integer:

```
parameters { key: "INFERENCE_MODE" value { string_value: "true" } }
```

## Full table

| Key | TorchScript | PT2 | Default | Notes | Source |
|---|---|---|---|---|---|
| `INFERENCE_MODE`             | yes | yes | `true`   | Guards execution with `c10::InferenceMode`; stronger than `NoGrad`. | `pt/model_state.cc` |
| `DISABLE_OPTIMIZED_EXECUTION`| yes | yes | `false`  | Disables TorchScript optimizer / AOTI optimizations. | `pt/model_state.cc` |
| `DISABLE_CUDNN`              | yes | yes | `false`  | Disables cuDNN kernels. | `pt/model_state.cc` |
| `ENABLE_WEIGHT_SHARING`      | yes | yes | `false`  | Reuses one weight copy across instances on the same device. **Not compatible with sequence batching** (`README.md`). | `pt/model_state.cc` |
| `INTRA_OP_THREAD_COUNT`      | yes | yes | `-1`     | `at::set_num_threads`. **Process-global**; first model wins. | `pt/model_state.cc` |
| `INTER_OP_THREAD_COUNT`      | yes | yes | `-1`     | `at::set_num_interop_threads`. **Process-global, throws on second attempt**; first model wins. | `pt/model_state.cc` |
| `ENABLE_CACHE_CLEANING`      | yes | no  | `false`  | Clears CUDA caching allocator after each `execute`. GPU-only. | `pt/model_state.cc` |
| `CACHE_CLEANING_ENABLED`     | no  | yes | `false`  | Same semantics as above; **different name on PT2**. | `pt2/model_state.cc` |
| `ENABLE_TENSOR_FUSER`        | yes | no  | unset    | Tri-state: unset leaves libtorch default. | `pt/model_state.cc` |
| `ENABLE_JIT_PROFILING`       | yes | no  | unset    | Tri-state. | `pt/model_state.cc` |
| `ENABLE_JIT_EXECUTOR`        | yes | no  | unset    | Tri-state. | `pt/model_state.cc` |
| `MODEL_INIT_LIBRARY`         | no  | yes | unset    | Path to a `.so`; backend `dlopen`s it and calls `triton_pytorch_model_init`. | `pt2/model_state.cc` |

## Name asymmetry for cache cleaning

`ENABLE_CACHE_CLEANING` on TorchScript. `CACHE_CLEANING_ENABLED` on PT2. Using the wrong spelling is silently ignored.

## Process-global threads

`INTRA_OP_THREAD_COUNT` and `INTER_OP_THREAD_COUNT` are applied once via `std::call_once`; the first model to load wins. Setting them differently per model has no effect after the first set. The INTER variant throws on a second attempt; the backend guards with try/catch.

## `MODEL_INIT_LIBRARY` (PT2)

Use to run one-time native setup before the PT2 archive loads (e.g. load external embedding weights). The backend `dlopen`s the library and calls `triton_pytorch_model_init(model_dir, device_index)`; optional `triton_pytorch_model_fini(handle)` is called on unload. The backend does not link against it.

Minimal stub (`init_hook.cc`, build with `g++ -shared -fPIC`):

```cpp
extern "C" void triton_pytorch_model_init(const char* dir, int device_index) {}
extern "C" void triton_pytorch_model_fini(void* handle) {}
```

Point to it:

```
parameters { key: "MODEL_INIT_LIBRARY" value { string_value: "/models/my_model/1/libinit.so" } }
```

Errors: see [`errors.md`](errors.md).

## Python-runtime path parameters

Only apply when using `backend: "pytorch"` + `runtime: "model.py"` (not `pt/` or `pt2/`):

- `NUM_THREADS`
- `NUM_INTEROP_THREADS`
- `TORCH_COMPILE_OPTIONAL_PARAMETERS` (JSON of `torch.compile` kwargs)

Reference: `pytorch_backend/README.md`.
