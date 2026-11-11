<!--
# Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
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
# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This repository is the **NVIDIA Triton Inference Server** — a C++ inference serving
platform with Python frontends. The main developable components are:

| Component | Path | Language | Notes |
|---|---|---|---|
| Core server | `src/` | C++ (CMake) | Requires NVIDIA Docker container + GPU to build/run |
| OpenAI-compatible frontend | `python/openai/` | Python (FastAPI) | Developable for Python-only changes; runtime requires `tritonserver` |
| Build system | `build.py`, `compose.py` | Python | Docker-based build orchestration |
| QA tests | `qa/L0_*` | Shell/Python | Designed to run inside NVIDIA containers |

### Linting

Run linting through pre-commit (the authoritative lint check enforced in CI):

```bash
pre-commit run --files <files>        # check specific files
pre-commit run --all-files            # check everything
```

Pre-commit hooks include: isort, black (v23.1.0), flake8, clang-format (v16), codespell,
and a custom license-header adder (`tools/add_copyright.py`). Configuration is in
`.pre-commit-config.yaml` and `pyproject.toml`.

### Tests

The OpenAI frontend tests (`python/openai/tests/`) require:
- The `tritonserver` Python package (only available inside NVIDIA NGC containers)
- A GPU-backed ML framework (`vllm` or `tensorrt_llm`)
- A running Triton Inference Server with loaded models

These tests **cannot run** in a standard Cloud Agent VM without GPU/NVIDIA infrastructure.
To run them, use an NVIDIA Docker container as described in `python/openai/README.md`.

### Key gotchas

- The `tritonserver` Python package cannot be pip-installed normally; it requires
  the Triton C library from NVIDIA NGC containers.
- `build.py` and `compose.py` both orchestrate Docker-based builds; they will fail
  without Docker and NVIDIA container runtime.
- The pre-commit `add-license` hook modifies files by adding/updating copyright headers —
  if it reports "Failed", run `git add` on the modified files and commit again.
- The globally-installed `black` version may differ from the pre-commit pinned version
  (23.1.0). Always use `pre-commit run` for authoritative formatting checks.
- `compose.py` emits `SyntaxWarning` about invalid escape sequences — this is a
  known issue in the existing code and does not affect functionality.
