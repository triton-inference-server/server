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

### Project overview

NVIDIA Triton Inference Server — a C++ inference serving platform with Python frontends. The core server binary (`tritonserver`) is built via CMake inside Docker containers using `build.py`. See `README.md` and `docs/customization_guide/build.md` for build details.

### Key components

| Component | Path | Language | Notes |
|---|---|---|---|
| Core server | `src/` | C++ | Requires GPU + Docker to build/run |
| Build orchestrator | `build.py` | Python | Generates Docker/CMake build steps; requires `distro`, `requests` |
| Container composer | `compose.py` | Python | Builds custom Triton Docker images |
| OpenAI-compatible frontend | `python/openai/` | Python (FastAPI) | Requires `tritonserver` Python bindings (C++ extension, only in Triton containers) |
| tritonfrontend bindings | `src/python/` | Python/C++ (pybind11) | Built as a wheel from C++ |
| QA tests | `qa/` | Shell scripts | ~140 `L0_*` integration tests; require GPU + pre-built Triton images |

### Linting

Pre-commit hooks are defined in `.pre-commit-config.yaml` and enforce: `isort`, `black`, `flake8`, `clang-format`, `codespell`, and misc checks (trailing whitespace, YAML/JSON validation, etc.). Run with:

```bash
pre-commit run --all-files
```

Or on staged files only (automatically on `git commit` if hooks are installed):

```bash
pre-commit install
```

### Running tests

- **OpenAI frontend tests** (`python/openai/tests/`): require `tritonserver` Python bindings and a GPU. These only work inside an official Triton Docker container (e.g. `nvcr.io/nvidia/tritonserver:26.04-vllm-python-py3`).
- **QA integration tests** (`qa/L0_*/`): shell-script-based, require GPU + Docker + pre-built Triton images. See `docs/customization_guide/test.md`.

### Environment constraints (Cloud VM)

- **No GPU** available in the Cloud Agent VM. The core C++ server cannot be built or run natively.
- **No `tritonserver` Python module** available outside Docker containers. The OpenAI frontend `main.py` and its tests will fail to import without it.
- Python linting (`pre-commit`, `flake8`, `black`, `isort`) and `build.py --help` / `compose.py --help` work without GPU.
- The Python OpenAI frontend schemas (`python/openai/openai_frontend/schemas/`) can be imported and validated without `tritonserver`.

### Build script dependencies

`build.py` and `compose.py` require Python packages `distro` and `requests` to be installed.

### OpenAI frontend dependencies

Install with:
```bash
pip install -r python/openai/requirements.txt
pip install -r python/openai/requirements-test.txt
```
