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

# QA Model Generation

The QA model corpus, how it is built, and how it is described once built.

Everything lives directly in `qa/common`, beside the helpers the generators import. They were
briefly split into a `model_generation/` subdirectory, but `test_util.py` is imported by 14 of
them *and* by the `L0_*` suites, so it could not follow — and a split that leaves the most-used
import behind buys nothing. Keeping them together means the staged source directory is just a
copy of this one, with nothing named individually.

| file | role |
|---|---|
| `gen_qa_model_repository` | the original shell driver — still the one CI calls |
| `gen_qa_model_repository.py` | Python wrapper over the same work, with per-framework selection |
| `gen_qa_*.py` | the generators, one per model family |
| `gen_common.py` | dtype helpers shared by every generator |
| `gen_manifest.py` | builds and maintains each model's `manifest.json` |
| `gen_archive.py` | packs each model into its own archive, with a fetch index |
| `gen_artifactory.py` | uploads those archives, with properties to select on |

## Usage

### Generate models

```bash
# everything, exactly what the shell driver does
./gen_qa_model_repository.py

# one framework
./gen_qa_model_repository.py --openvino

# a subset, with an image override and a specific opset
./gen_qa_model_repository.py --onnx --pytorch \
    --ubuntu-image ubuntu:24.04 --onnx-opset 17

# see what a flag combination would do, without doing it
./gen_qa_model_repository.py --all --list
./gen_qa_model_repository.py --tensorrt --dry-run
```

Stages always run in the order OpenVINO → ONNX → PyTorch → TensorRT, whatever order the flags
are given in, and never in parallel. That is not cosmetic: all four write into the *same*
repositories — `qa_model_repository` receives models from every one of them.

With no framework flag, all four run. `MODEL_TYPE=igpu` drops the TensorRT stage entirely and
`torch_tensorrt` within the PyTorch stage.

Stage selection also comes from the environment, for a CI job that sets variables rather than
composing a command line:

```bash
TRITON_MODELS_FRAMEWORKS=onnx,openvino ./gen_qa_model_repository.py
TRITON_MODELS_FRAMEWORKS="pytorch tensorrt" ./gen_qa_model_repository.py
TRITON_MODELS_FRAMEWORKS=all ./gen_qa_model_repository.py
```

Comma or whitespace separated, case-insensitive. The backend names used in `config.pbtxt` and
in the `L0_*` suites' `BACKENDS` are accepted too — `plan`, `libtorch`, `onnxruntime`, `trt`,
`torch` — since asking for models *for* a backend is the same request as asking for the stage
that builds them. Precedence runs `--all`, then the per-stage flags, then the variable, then
all four.

The value is verified, and an unrecognised name is refused rather than skipped:

```
$ TRITON_MODELS_FRAMEWORKS=onx ./gen_qa_model_repository.py
error: --frameworks/TRITON_MODELS_FRAMEWORKS: unknown framework 'onx' (did you mean 'onnx'?);
       valid: openvino, onnx, pytorch, tensorrt, all
```

That refusal is the point of the check. A typo that quietly generated nothing would not surface
until a test suite failed on missing models, hours later and far from the cause.

### Every environment variable is also a flag

The wrapper reads the same variables the shell driver does, so existing CI works unchanged, and
each has a flag that overrides it for one invocation:

| flag | variable | default |
|---|---|---|
| `--frameworks` | `TRITON_MODELS_FRAMEWORKS` | all four |
| `--container-version` | `TRITON_CONTAINER_VERSION` | `build.py` `triton_container_version` |
| `--upstream-version` | `NVIDIA_UPSTREAM_VERSION` | `build.py` `upstream_container_version` |
| `--semver` | `TRITON_SEMVER` | `build.py` `release_version` |
| `--ubuntu-image` | `UBUNTU_IMAGE` | `ubuntu:22.04` |
| `--pytorch-image` | `PYTORCH_IMAGE` | `nvcr.io/nvidia/pytorch:<version>-py3` |
| `--tensorrt-image` | `TENSORRT_IMAGE` | `nvcr.io/nvidia/tensorrt:<version>-py3` |
| `--onnx-version` | `ONNX_VERSION` | `1.20.1` |
| `--onnx-opset` | `ONNX_OPSET` | `0` |
| `--openvino-version` | `OPENVINO_VERSION` | `2024.5.0` |
| `--model-type` | `MODEL_TYPE` | unset |
| `--runtime`, `--use-docker`, `--use-enroot` | `TRITON_MODELS_USE_DOCKER`, `TRITON_MODELS_USE_ENROOT` | auto |
| `--nvidia-visible-devices` | `NVIDIA_VISIBLE_DEVICES` | `all` |
| `--docker-volume` | `DOCKER_VOLUME` | `volume.gen_qa_model_repository.<job id>` |
| `--build-dir` | `TRITON_MDLS_BLD_DIR` | `<mount>/<job id>` |
| `--job-id` | `CI_JOB_ID` | timestamp |

### Container engines

Both engines the shell driver supports are kept. `--runtime auto` follows its precedence —
docker if enabled and installed, else enroot — and `--runtime docker|enroot` forces one, failing
loudly rather than falling back if it is not installed.

|  | docker | enroot |
|---|---|---|
| storage | named volume at `/mnt` | host `/tmp`, bind-mounted |
| output | `docker cp` to `--output-dir` at the end | written straight to `/tmp/<job id>` |
| privilege | container default | `--root` for the apt-based stages only |
| GPU | `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES` | `-e NVIDIA_VISIBLE_DEVICES` |

enroot is not vestigial — it is how the SLURM B200 job builds, with `TRITON_MODELS_USE_DOCKER=0`.

`--nvidia-visible-devices` reaches both engines. That matters more on enroot than it looks:
enroot exposes GPUs through its `98-nvidia.sh` hook, which keys entirely off
`NVIDIA_VISIBLE_DEVICES` in the container environment and does nothing at all when it is unset.

The values are the container toolkit's, not ours: `all` for every GPU, `0,1` or a `GPU-…` UUID
for specific ones, `none` for no GPU with driver capabilities still enabled, and `void` — which
empty and unset also mean — for a runtime that behaves like `runc`, exposing neither. `all` is
the default here because it is the documented way to ask for whatever GPUs exist, where `0`
assumed an index that a given node need not have. An *empty* value falls back to that default
rather than being honoured as `void`: it is indistinguishable from a caller forwarding a
variable its environment never defined, which is exactly how a CI job came to build without a
GPU and fail minutes later inside torch.
The shell driver never sets it on the enroot path, so enroot builds ran without a GPU and
recorded `gpu: null` — invisible for OpenVINO, but TensorRT plan files are compute-capability
specific, so the field has to be real. The hook's own opt-out value, `none`, is passed through.

### Archive the models

```bash
./gen_qa_model_repository.py --all --archive
./gen_qa_model_repository.py --all --archive --archive-dir /mnt/artifacts
```

Archiving runs **on the host, after generation finishes** — it is not a stage and never happens
inside a generation container. A finished tree can therefore be re-archived, under different
naming or after a failed upload, without regenerating a single model, and the generation images
carry no archiving concern. A packaging failure is warned about rather than failing the run,
since by then the models are already generated and correct.

Archives go in `archives/`, a folder of their own at the root of the model tree, so they travel
with it. They are safe there because every walk — sizing, manifesting, archiving — keys on a
directory holding a `config.pbtxt`, and this one holds only tarballs. `--archive-dir` puts them
somewhere else entirely.

**One archive per framework.** A test suite runs against one backend and the four stages are
already split that way, so the framework is the unit a consumer actually fetches:

```
onnx-26.07-2.72.0dev-57638316.360696202.tar       735 models       4.2 MiB  14 properties
openvino-26.07-2.72.0dev-57638316.360696202.tar    16 models       0.2 MiB  15 properties
pytorch-26.07-2.72.0dev-57638316.360696202.tar    113 models    1164.3 MiB  14 properties
tensorrt-26.07-2.72.0dev-57638316.360696202.tar    84 models     106.5 MiB  14 properties
```

The trailing components are `<upstream>-<semver>`, then `<pipeline>.<job>` when CI supplies
them — a dot between the CI pair, matching the names already in Artifactory, and a dash between
the versions. Absent parts are dropped rather than left as empty segments, so a local archive is
not named after a pipeline that does not exist.

Inside, each model keeps its full path relative to the tree:

```
qa_model_repository/openvino_int8_int8_int8/config.pbtxt
qa_model_repository/openvino_int8_int8_int8/manifest.json
```

so unpacking a bundle over a tree root restores every model to its own repository — necessary
because the same model name appears in more than one.

Each bundle gets a `manifest-<framework>.json` beside it carrying the properties **every model
in it agrees on**, already flattened:

```
container.image_name=ubuntu:22.04   model.framework=onnx        platform.arch=x86_64
container.runtime=docker            model.framework_version=1.20.1
```

A field the models *disagree* on is dropped rather than resolved to a first or majority value: a
property true of only some of an archive's contents is worse than an absent one. That is why the
ONNX bundle has one property fewer than OpenVINO — it also carries ensemble models, whose
`model.provenance` differs.

#### One archive per model instead

`--granularity model` packs each model separately rather than bundling a framework:

```bash
python3 gen_archive.py --tree /tmp/26.08dev --dest /tmp/26.08dev/archives \
    --granularity model
```

```
qa_model_repository-openvino_int8_int8_int8-26.07-2.72.0dev.tar   1 model   13.9 KiB
```

The choice is what a consumer has to download, and what the properties can say:

| | fetch cost | properties |
|---|---|---|
| `framework` (default) | a whole backend to get one model | only the fields its models agree on |
| `model` | exactly the model wanted | everything, since each is true of its artifact |

A bundle publishes only what its models agree on, so `model.name`, the sizes and the format
versions drop out the moment it holds more than one — they would be false of the bundle. The
static stores carry 17 properties bundled and 23 per model for exactly that reason.

Framework granularity suits the generated corpus, where a suite runs against one backend and
wants all of it. Model granularity suits the static stores, where the point is pulling
individual objects rather than a repository to find out what is in it.

Labels become `<repository>/<model>`, which names the directory an upload lands in; the archive
and its manifest are files, so both flatten the separator into their names and still say where
they came from once detached.

`gen_archive.py` runs on its own against any finished tree:

```bash
python3 gen_archive.py --tree /tmp/26.08dev --dest /tmp/26.08dev/archives
python3 gen_archive.py --tree /tmp/26.08dev --dest /tmp/26.08dev/archives --framework onnx
```

Pass `--compress` for `.tar.gz` instead of `.tar`; plan files compress ~96%, which matters for
the TensorRT and PyTorch bundles. Archives are byte-reproducible — entries sorted, ownership and
timestamps normalised — so identical content yields an identical checksum and a runner can skip a
download it already has, provided the name components match.

### Upload the archives

```bash
./gen_qa_model_repository.py --all --artifactory-upload
```

`--artifactory-upload` implies `--archive` and needs four settings, each a flag or a variable:

| flag | variable |
|---|---|
| `--artifactory-url` | `ARTIFACTORY_URL` |
| `--artifactory-token` | `ARTIFACTORY_TOKEN` |
| `--artifactory-path` | `ARTIFACTORY_PATH` |
| `--artifactory-properties` | `ARTIFACTORY_PROPERTIES` |

They are checked **before the first container starts**, so a missing token costs a second rather
than a multi-hour run that cannot deliver its output, and the error names every absent setting
rather than the first. The token reaches the uploader through the environment, never argv: the
driver logs every command it runs, and argv is readable by any process on the host.

`gen_artifactory.py` also runs on its own:

```bash
python3 gen_artifactory.py --archives /tmp/26.08dev/archives \
    --path sw-dl-triton-generic-local/triton/models \
    --model-type DGX-Spark-sbsa-12.1 \
    --nvidia-upstream-version 26.07 --triton-semver 2.72.0dev
```

Uploads land one level per thing a consumer narrows by — device, release, then bundle:

```
sw-dl-triton-generic-local/triton/models/DGX-Spark-sbsa-12.1/26.07-2.72.0dev/onnx/onnx-26.07-2.72.0dev-57638316.360696202.tar
```

`--flat` drops the bundle directory, putting every archive directly under the release:

```
sw-dl-triton-generic-local/triton/models/STATIC/26.07-2.71.2/openvino_model_store-resnet50_int8_openvino-26.07-2.71.2-manual.tar
```

The directory earns its place when a handful of bundles share a release. Per-model archives
already carry repository and model in their own names, so nesting them buys a directory per
model and nothing else — which is why the static stores are published flat.

`--url` and `--token` fall back to the jfrog CLI config, so a developer who has already run
`jf config add` need not restate either. It is read from `$JFROG_CLI_HOME_DIR/jfrog-cli.conf.v6`,
defaulting to `~/.jfrog`, and the server is chosen by `isDefault` rather than by position — the
servers are a JSON array and a second `jf config add` reorders them. `--jfrog-server-id` picks one
by name. Which source supplied each value is logged, by server id; the token never is.

Failed PUTs are retried `--retries` times, default 3. A 4xx is not retried — a bad token or path
will not fix itself. The `sha256` from the index is sent as `X-Checksum-Sha256`, so Artifactory
rejects a truncated upload rather than leaving it for a test job to find.

#### Which properties an upload carries

Three layers, each winning over the one before:

1. **The bundle's own**, flattened from `manifest-<framework>.json` — `model.framework`,
   `platform.arch`, `container.runtime` and the rest.
2. **The bundle's identity** — `bundle`, `model_count`, `sha256`. The grouping key is published
   as `bundle`, not `framework`: for the generated corpus the two coincide, but the static stores
   group by repository, where `framework=c2_model_store` would contradict the
   `model.framework=netdef` the manifest supplies.
3. **Declared properties**, merged from four sources, lowest scope first:

   ```
   ARTIFACTORY_PKG_MODELS_COMMON_PROPERTIES   what is true of every model package
   ARTIFACTORY_PROPERTIES                     this upload
   ARTIFACTORY_PROPERTIES_CI_JOB_PROPERTIES   this CI job
   --properties                               this invocation
   ```

   All four are **merged**, not chosen between — they are scopes of the same thing, and a key set
   by more than one resolves to the narrowest scope that sets it.

That third layer is how provenance survives that no artifact can supply. The static stores carry
`NVIDIA_UPSTREAM_VERSION=25.09` and `TRITON_VERSION=2.61.0` — what actually built them, older
than the directory they sit in — alongside `ISSUE`, `MODEL_TYPE` and the `CI_*` set.

## Size classification

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

## Writing the manifest: `gen_manifest.py`

`gen_manifest.py` builds and maintains the per-model `manifest.json`. It sits beside
`gen_common.py` and is imported the same way, by every generator script.

### From a generator script

Two calls, either side of generation:

```python
import gen_manifest

if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()

    manifest_baseline = gen_manifest.snapshot_model_dirs(FLAGS.models_dir)

    ...generate models...

    gen_manifest.emit_manifests(FLAGS.models_dir, manifest_baseline)
```

The baseline is the load-bearing part. Every stage writes into the *same* repositories —
`qa_model_repository` receives models from all four — so a script that stamped everything it
found would relabel the previous stage's models with its own image and framework.
`snapshot_model_dirs()` fingerprints each model directory (file count, total bytes, newest
mtime) before generation; `emit_manifests()` writes only where that fingerprint changed. A
model another script later adds files to is correctly re-stamped by that script.

`emit_manifests()` never raises. A run that has already produced its models must not fail over
a metadata file, so problems are logged and counted in the `[manifest]` summary line instead.
Missing manifests are caught by the phase-2 pass below, not by killing the build.

### From the command line

```bash
# phase 1 -- stamp a tree with build provenance
python3 gen_manifest.py --tree /tmp/26.08 --image ubuntu:24.04 --framework openvino

# phase 2 -- (re)measure sizes once every stage has finished
python3 gen_manifest.py --tree /tmp/26.08 --update-sizes
```

Sizes are a second pass because a model directory receives files from more than one stage: the
size recorded by the stage that created a model is only final once the later stages have run.
Phase 2 is idempotent — `size_bytes` excludes `manifest.json`, so re-running produces a
byte-identical file.

`--repository` and `--provenance` narrow the walk; `--dry-run` reports without writing.

### What it reports while it runs

The properties this pass will stamp into every manifest it writes, once, before it starts —
so a generation log answers "which openvino was this built against?" without unpacking a
model and reading its manifest:

```
[manifest] properties recorded by this pass:
[manifest]   tree                         /mnt/26.07
[manifest]   generator                    gen_qa_models.py
[manifest]   framework                    openvino
[manifest]   framework version            2024.5.0-17288-7975fa5da0c
[manifest]   container image              ubuntu:22.04
[manifest]   container runtime            docker
[manifest]   platform                     Ubuntu:22.04 x86_64
[manifest]   gpu                          NVIDIA RTX 5880 Ada Generation
[manifest]   gpu compute capability       8.9
[manifest]   driver version               590.44.01
[manifest]   triton version               2.72.0dev
[manifest]   upstream container version   26.07
```

A field the driver did not set prints as `-` and is recorded as null rather than guessed at.
When no stage is declared, `framework` reads `per model (from config.pbtxt)` — it and its
version are then resolved per model from the backend that serves it, so neither can be
reported up front. `--update-sizes` skips the block: that pass rewrites only `size_bytes` and
`size_tier`, so reporting the rest would describe the invocation rather than the manifests.

Then one line per model as it is stamped — a full tree is ~976 models and the pass is not
instantaneous, so without this it looks hung:

```
[manifest]   qa_model_repository/openvino_float32_float32_float32   openvino   13.9 KiB  s
[manifest]   qa_model_repository/openvino_nobatch_int32_int8_int8   openvino    4.7 KiB  xs
[manifest] updated 16 manifest(s), skipped 0
```

The fields are the ones a reader decides on: what serves the model, how big it is, and which
tier that puts it in. `--quiet` keeps only the summary line.

`--summary PATH` additionally writes the whole pass as JSON, so CI can assert on counts and
totals instead of grepping log text. `--summary -` sends it to stdout.

```json
{
  "kind": "triton-qa-model-manifest-summary",
  "written": 16, "skipped": 0, "model_count": 16, "total_bytes": 128424,
  "by_size_tier": {"s": 6, "xs": 10},
  "by_provenance": {"openvino": 16},
  "models": [{"name": "...", "path": "...", "repository": "...",
              "provenance": "openvino", "size_bytes": 14220, "size_tier": "s"}]
}
```

`gen_qa_model_repository.py` writes one to `<tree>/manifest-summary.json` at the end of a run.
It sits at the *root* of the tree, not inside a model, so it travels with the tree while no walk
of it — sizing, archiving or manifesting — ever picks the file up.

### What gets recorded, and where it comes from

| field | source |
|---|---|
| `model.name`, `platform`, `backend` | top-level keys in `config.pbtxt`, anchored at column 0 so tensor `name:` inside an input block cannot win |
| `model.provenance` | normalised from `platform`, else `backend`, else the artifact suffix |
| `model.generator` | the script that wrote it — `provenance` alone does not identify it, since eight generators emit onnx models |
| `model.framework`, `framework_version` | `TRITON_MODEL_GEN_FRAMEWORK`, else the model's own provenance; the version is read in-process from the installed library, not from a pin |
| `model.gpu` | `cuda-python` + NVML if importable, else `nvidia-smi`; probed once per run, not once per model |
| `model.origin` | `static` for the pinned stores listed in `STATIC_REPOSITORIES`, else `dynamic`. `perf_model_store` is listed under all three of its spellings — it ships as two artifacts split by backend, and a tree laid out to match them has the split names |
| `container.image_name` | `TRITON_MODEL_GEN_IMAGE` |
| `container.runtime` | `TRITON_MODEL_GEN_RUNTIME`, else detected |
| `platform.*` | `platform.system()`, `/etc/os-release`, `platform.machine()` |

#### Static models record almost none of that

A `static` model is copied forward, not generated: it was built elsewhere, at an unknown time, on
unknown hardware. So for `origin: static` the host is not probed at all — `gpu`, `platform.platform`,
`platform.os_release`, `framework_version` and both `container` fields are recorded as null,
even when the caller passes an image or version explicitly, because those describe the process
doing the stamping rather than whatever produced the artifact.

```json
"container": { "image_name": null, "runtime": null },
"platform":  { "platform": null, "os_release": null, "arch": "x86_64" },
"model":     { "framework": "openvino", "provenance": "openvino", "origin": "static",
               "framework_version": null, "gpu": null, "size_tier": "m" }
```

Stamping the `26.07-2.71.0` static stores on a dev box would otherwise assert they were built
against that box's GPU and OS — and compute capability is exactly what a consumer filters on to
decide whether a plan or AOTI artifact is portable to their hardware, so a confident wrong answer
there is worse than none. Architecture survives: a static artifact is fetched for an arch and its
tree is arch-specific regardless.

What the build environment actually was lives in the Artifactory properties instead — see
[Which properties an upload carries](#which-properties-an-upload-carries).

The container block is named `container`, not `docker` or `enroot`. Naming it after the engine
would make the schema shape depend on the data, forcing every consumer to probe both keys to
find the image; enroot pulls `docker://` references anyway, so only `runtime` differs.

Runtime detection is best-effort. Docker leaves `/.dockerenv` in every image. enroot leaves
nothing — it exports no marker and strips the `ENROOT_*` variables it was invoked with, so
testing the environment both misses real enroot containers *and* false-positives on any host
whose shell has `ENROOT_CONFIG_PATH` set. What it does leave is the mount shape: `/` is
bind-mounted from `$ENROOT_DATA_PATH/<container>`. Retuning `ENROOT_DATA_PATH` defeats that, so
set `TRITON_MODEL_GEN_RUNTIME` when the value has to be exact.

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
