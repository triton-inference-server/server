<!--
SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
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
