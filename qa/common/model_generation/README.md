<!--
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# QA Model Size Classification

Every model in the QA corpus is labelled with a **t-shirt size** derived from its on-disk
footprint. The label exists so a consumer can decide *what to fetch* without fetching it first:
the corpus is extremely top-heavy, and almost all of the transfer cost is concentrated in a
handful of models.

## Tiers

The tier is a pure function of the model directory's total size in bytes:

| tier | range | models | % of models | bytes | % of bytes |
|------|-------|-------:|------------:|------:|-----------:|
| `xs` | < 10 KiB | 857 | 87.8% | 1.3 MiB | 0.01% |
| `s`  | 10 KiB – 1 MiB | 69 | 7.1% | 5.8 MiB | 0.03% |
| `m`  | 1 – 100 MiB | 36 | 3.7% | 693.5 MiB | 4.06% |
| `l`  | 100 MiB – 1 GiB | 13 | 1.3% | 3.20 GiB | 19.21% |
| `xl` | > 1 GiB | 1 | 0.1% | 12.78 GiB | 76.69% |
| | **total** | **976** | | **16.66 GiB** | |

Measured on the assembled `26.07` tree (`/data/inferenceserver/26.07`). Counts shift between
trains; the proportions do not.

Boundaries sit on real gaps in the distribution rather than round numbers. Between the largest
`l` model (685 MiB) and the single `xl` there is nothing at all, and only 3 models fall within
2x of the `l|xl` cut. The `xs|s` and `s|m` cuts are noisier (34 and 28 models within 2x), but
both tiers are rounding errors by volume, so misclassification there costs nothing.

## Why the distribution looks like this

**`xs` is not "small models" — it is models with no weights.** 498 of them are ensembles
(`platform: "ensemble"`) and 106 are `nop_*` models on the `identity` backend. They consist of a
`config.pbtxt` and nothing else.

**20 models hold 99% of the bytes.** The median model is under 1 KiB.

**One model is three quarters of the corpus.** `onnx_model_store2/large_onnx` is a DLRM with
6,858,905,473 FP16 parameters (opset 11, produced by PyTorch 1.4), stored with ONNX external
tensor data — so `model.onnx` is a *directory* of 31 weight files plus a 26 KB graph proto, not
a file. 24 embedding tables account for ~97% of the parameters; five of them are 10M x 128.
Its dense compute is trivial by comparison — the bottom MLP is 13 -> 512 -> 256 -> 128.

The next largest model, `c2_model_store/vgg19_netdef`, is 685 MiB — nineteen times smaller.

## Everything at `l` and above

These 14 models are 95.9% of the corpus. The list is short enough to curate by hand, and it is
the only part of the corpus where per-model fetch granularity pays for itself.

| tier | size | % corpus | backend | origin | model |
|------|------|---------:|---------|--------|-------|
| `xl` | 12.78 GiB | 76.7% | onnx | static | `onnx_model_store2/large_onnx` |
| `l` | 685 MiB | 4.0% | netdef | static | `c2_model_store/vgg19_netdef` |
| `l` | 548 MiB | 3.2% | libtorch | static | `libtorch_model_store/vgg19_libtorch` |
| `l` | 548 MiB | 3.2% | onnx | dynamic | `qa_dynamic_batch_image_model_repository/vgg19_onnx` |
| `l` | 229 MiB | 1.3% | onnx | static | `onnx_model_store/resnet_v1_152` |
| `l` | 229 MiB | 1.3% | onnx | dynamic | `qa_dynamic_batch_image_model_repository/resnet152_onnx` |
| `l` | 160 MiB | 0.9% | libtorch | static | `libtorch_model_store/fasterrcnn_resnet50_libtorch` |
| `l` | 160 MiB | 0.9% | libtorch | static | `libtorch_model_store/fasterrcnn_libtorch_unsupported` |
| `l` | 160 MiB | 0.9% | libtorch | static | `libtorch_model_store/fasterrcnn_libtorch_cpu` |
| `l` | 130 MiB | 0.8% | libtorch | dynamic | `torchtrt_model_store/resnet50_libtorch` |
| `l` | 122 MiB | 0.7% | netdef | static | `c2_model_store/resnet50_netdef` |
| `l` | 104 MiB | 0.6% | libtorch | static | `libtorch_model_store/inception_v3_libtorch` |
| `l` | 100 MiB | 0.6% | torch_aoti | dynamic | `qa_model_repository/torchvision_aoti` |
| `l` | 100 MiB | 0.6% | plan | dynamic | `qa_identity_model_repository/plan_compatible_zero_1_float32` |

Two things stand out. **Nine of the fourteen are `static`** — pinned to
`STATIC_MODELS_SOURCE_VERSION`, arch-independent, never regenerated, yet carried in every
assembled tree on every architecture. And **the same networks recur across backends**: vgg19
three times (686 + 549 + 549 MiB), resnet152 twice, fasterrcnn three times. The fasterrcnn trio
are not blind copies — `_unsupported` and `_cpu` have different checksums — but
`fasterrcnn_resnet50_libtorch` contains only a 676-byte artifact despite reporting 160 MiB,
which suggests a symlink or sparse layout worth checking before any dedup work.

## Framework version fields

Unrelated to size, but the other axis worth recording per model, and the two frameworks do not
express it the same way:

| | ONNX | OpenVINO IR |
|---|---|---|
| format revision | `ir_version` (e.g. 6) | `<net version="...">` — 10 or 11 in this corpus |
| operator-set revision | `opset_import` — one integer per model | per-layer `version="opsetN"` |

ONNX gives a single model-wide opset. OpenVINO declares an opset **per operation**
(`opset1`…`opset15`), so reducing it to one field means taking the max across layers.

The two OpenVINO stores currently disagree: the pinned `openvino_model_store/resnet50_int8_openvino`
is IR version 10, while models freshly generated by `gen_qa_models.py --openvino` are IR
version 11 — expected, since the static store is copied forward and never regenerated.

Neither value is derivable from `config.pbtxt`; both require reading the artifact
(`model.onnx` graph proto, or the `<net>` element of `model.xml`).

## Where the classification is stored

Two places, and both are needed:

1. **`manifest.json`**, next to each model's `config.pbtxt`. Records `size_bytes` (the truth)
   and `size_tier` (the index key). This is the on-disk record, useful once a tree is assembled.

2. **Artifactory properties** on the uploaded artifact, as `triton.size_bytes` and
   `triton.size_tier`.

Point 2 is the load-bearing one. A manifest stored *inside* an artifact cannot inform the
decision to download that artifact — you would have to fetch 12.78 GiB to discover it is `xl`.
Only an AQL-queryable property lets a job filter before transfer:

```
items.find({"repo":"sw-dl-triton-generic-local",
            "@triton.size_tier":{"$in":["xs","s","m"]}})
```

Keep `size_bytes` authoritative and treat `size_tier` as derived. If the boundaries are ever
retuned, tiers can be recomputed from bytes without re-measuring the corpus.

## Computing the tier

```python
KIB, MIB, GIB = 1024, 1024**2, 1024**3

def size_tier(total_bytes):
    if total_bytes <  10 * KIB: return "xs"
    if total_bytes <   1 * MIB: return "s"
    if total_bytes < 100 * MIB: return "m"
    if total_bytes <   1 * GIB: return "l"
    return "xl"
```

`total_bytes` is the recursive sum of every file under the model directory, including the
version subdirectories and any external tensor data, and excluding `manifest.json` itself so
the value is stable across regeneration.

Note the model tree is **not** uniformly two levels deep. Most models are at
`<repository>/<model>/`, but 606 of the 976 sit one level deeper under
`qa_ensemble_model_repository/<subrepo>/<model>/` or `qa_custom_ops/libtorch_custom_ops/`.
Walk for directories containing `config.pbtxt` rather than globbing a fixed depth.

## Practical consequences

- **`xs` + `s` should not be fetched individually.** 926 models, 95% of the corpus by count,
  about 7 MiB in total. Per-model addressing costs more in metadata and round-trips than the
  payload is worth. Ship them as one blob.
- **`l` + `xl` are the entire problem.** 14 models, 96% of the bytes. Small enough to curate by
  hand; large enough that skipping even one changes a job's transfer time materially.
- **`xl` is arch-independent and pinned.** `onnx_model_store2` is in `STATIC_MODELS_LIST`, so it
  is copied forward from `STATIC_MODELS_SOURCE_VERSION` and never regenerated — yet it is
  currently carried in every assembled tree on every architecture. Letting jobs that do not need
  it opt out recovers ~77% of the transfer on its own.
