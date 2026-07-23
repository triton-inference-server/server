<!--
# Copyright 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# L2_build_presets

Validates the experimental **build presets** feature
(`docs/customization_guide/build.md`) by running `build.py --dryrun` and checking
the generated `build_presets.json`. No GPU, container, or real build needed.

## Run

```bash
cd qa/L2_build_presets
bash test.sh                              # installs deps, runs pytest, writes logs
```

Or directly: `python3 -m pytest build_presets_test.py`.

## Finding build.py

`build.py` lives in the server repo. It is located in-tree (a checkout), or
cloned when only this directory is present. Override via:

| Env var | Meaning | Default |
|---|---|---|
| `TRITON_BUILD_PY` | explicit path to `build.py` | — |
| `TRITON_SERVER_REPO` | repo to clone when not found | `.../triton-inference-server/server.git` |
| `TRITON_SERVER_BRANCH_NAME` | branch/tag to clone — **must exist on the remote** | `main` |

Bare container, mounting only this directory:

```bash
cd server/
docker run --rm -v "$PWD":/workspace -w /workspace \
  -e TRITON_SERVER_BRANCH_NAME=<remote-branch> \
  -w /workspace/qa/L2_build_presets \
  nvcr.io/nvidia/tritonserver:26.06-py3 python3 build_presets_test.py
```

## Output

`test.sh` writes `build_presets_test.log` (full console) and
`build_presets_test.report.xml` (JUnit) — both git-ignored — and prints the
`*** Test Passed ***` / `*** Test FAILED ***` markers.
