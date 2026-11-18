<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Canonical sources: `pytorch_backend/src/pt2/` and `pytorch_backend/README.md`. Server docs do not cover PT2.

# PT2 archives (`torch_aoti`)

A `.pt2` is a self-contained archive from `torch.export.export` + `torch._inductor.aoti_compile_and_package`. It contains the call spec, compiled AOT Inductor kernels, and constants. Available in Triton since 26.03 (`README.md`).

## Device-baked

The archive captures the device it was exported on. CPU export runs only on `KIND_CPU`; CUDA export runs only on `KIND_GPU`. Mixing throws `Incorrect device passed to aoti_runner_cpu` at load (upstream libtorch `model_container_runner_cpu.cpp:39`). Re-export to change device.

## Layout

```
model_repository/
  <name>/
    config.pbtxt
    1/
      model.pt2            # override with default_model_filename
      [libinit.so]         # only if MODEL_INIT_LIBRARY is set
```

Default filename: `model.pt2` (`pt2/model_state.hh`).

## Export recipe

CPU:

```python
import torch

m = Model().eval()
sample = (torch.randn(2, 4), torch.randn(2, 4))
batch = torch.export.Dim("batch", min=1, max=8)

exported = torch.export.export(
    m, sample, dynamic_shapes=({0: batch}, {0: batch})
)
torch._inductor.aoti_compile_and_package(exported, package_path="model.pt2")
```

GPU:

```python
m = Model().eval().cuda()
sample = (torch.randn(2, 4).cuda(), torch.randn(2, 4).cuda())
# rest identical
```

Constraints:

- Sample batch must be `> 1`. A batch=1 sample causes `torch.export` to specialize the Dim: `Constraints violated (batch)`.
- `Dim("batch", max=N)` must be `>= max_batch_size` in `config.pbtxt`.

## Dynamic batching

`pytorch_backend/README.md`: for `max_batch_size > 0`, every input must have a dynamic first dim via `torch.export.Dim`. An archive without this loads `READY` and serves correctly for batch 1 but rejects any larger batch at inference time with:

```
failed to split the output tensor '<name>' in responses: expected batch size of at least N in model output, got M
```

The fix is to re-export with `torch.export.Dim("batch", min=1, max=max_batch_size)` applied to the first dim of every input.

## Naming

Two schemes, mixable. See [`naming.md`](naming.md):

- Call-spec: `ARGS[0]`KWARGS[kwarg_name]`RESULT` for a single bare Tensor output.
- Ordinal: `INPUT__0`, `OUTPUT__0`.

## Thread-safety at load

The backend serializes PT2 construction with a static mutex (`pt2/model_state.cc`). Upstream libtorch's `AOTIModelPackageLoader` uses a shared static `nlohmann::json`; concurrent loads would corrupt metadata. Many PT2 models loading at startup therefore load serially.

## `MODEL_INIT_LIBRARY`

Per-model native init hook. The backend `dlopen`s the library and calls `triton_pytorch_model_init(model_dir, device_index)` before loading the archive; optional `triton_pytorch_model_fini(handle)` on unload. See [`parameters.md`](parameters.md).

## Not supported in PT2 (26.10)

From `pytorch_backend/README.md`:

- `TYPE_STRING` inputs and outputs.
- `TYPE_STRING` `CONTROL_SEQUENCE_CORRID` (use `TYPE_INT32` or `TYPE_INT64`).
- Default batching of nested `dict` / `tuple` / `list` I/O.
- RHEL (manylinux) builds.

Supported with care:

- Sequence batching with implicit state: see [`sequence-batching.md`](sequence-batching.md).
- Multi-GPU: use `KIND_MODEL`.

## Checklist

1. Export on the target device with matching `torch` version.
2. `torch.export.Dim("batch", min=1, max=N)` on every input's dim 0 if `max_batch_size > 0`; `N >= max_batch_size`.
3. Place `model.pt2` at `<repo>/<name>/1/model.pt2`.
4. `backend: "pytorch"` + `platform: "torch_aoti"`.
5. `instance_group` kind matches export device.
6. Name inputs `INPUT__<i>` or call-spec; output `RESULT` for a single Tensor or `OUTPUT__<i>`.
