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
"""Fixtures for the `python_based_backends` scenario: a hand-rolled
"add_sub" backend registered directly under BACKEND_DIR (not the python
backend), the python-backend add_sub model, and a generated pytorch-backend
model -- all loaded on demand by python_based_backends_test.py itself
against a --model-control-mode=explicit server with nothing preloaded.

Ported from qa/L0_backend_python/python_based_backends/test.sh. Preserves
the TRI-1744 fix: gen_manifest.py must be copied alongside
gen_qa_pytorch_model.py before running it, or the generator's own
`import gen_manifest` fails.

python_based_backends_test.py is copied in unchanged (see
../../model_control/tests/conftest.py for why that's safe).
"""

import contextlib
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIO_DIR = os.path.dirname(HERE)
QA_DIR = os.path.dirname(os.path.dirname(SCENARIO_DIR))
PYTHON_MODELS_DIR = os.path.join(QA_DIR, "python_models")
COMMON_DIR = os.path.join(QA_DIR, "common")

sys.path.append(COMMON_DIR)
import action_log  # noqa: E402

TRITON_DIR = os.environ.get("TRITON_DIR", "/opt/tritonserver")
SERVER = os.environ.get("SERVER", os.path.join(TRITON_DIR, "bin", "tritonserver"))
BACKEND_DIR = os.environ.get("BACKEND_DIR", os.path.join(TRITON_DIR, "backends"))
TRITONSERVER_IPADDR = os.environ.get("TRITONSERVER_IPADDR", "localhost")
HTTP_PORT = int(os.environ.get("TRITONSERVER_HTTP_PORT", "8000"))
GRPC_PORT = int(os.environ.get("TRITONSERVER_GRPC_PORT", "8001"))

STARTUP_TIMEOUT_S = 120
OUTPUT_DIR = os.path.join(SCENARIO_DIR, "output")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")


@pytest.fixture(scope="session")
def pinned_torch():
    if os.environ.get("PBB_SKIP_TORCH_INSTALL") == "1":
        yield
        return
    action_log.run(
        ["pip3", "install", "torch"],
        "Installing torch for the generated add_sub_pytorch model",
        check=True,
    )
    yield


@pytest.fixture(scope="session")
def model_repository(pinned_torch):
    if os.path.isdir(MODELS_DIR):
        shutil.rmtree(MODELS_DIR)
    os.makedirs(MODELS_DIR)

    # Register the hand-rolled "add_sub" backend directly under BACKEND_DIR
    # (not a model in the repository -- a whole new TRITONBACKEND_* backend).
    add_sub_backend_dir = os.path.join(BACKEND_DIR, "add_sub")
    os.makedirs(add_sub_backend_dir, exist_ok=True)
    shutil.copy(
        os.path.join(
            PYTHON_MODELS_DIR, "python_based_backends", "add_sub_backend", "model.py"
        ),
        os.path.join(add_sub_backend_dir, "model.py"),
    )

    add_v1 = os.path.join(MODELS_DIR, "add", "1")
    os.makedirs(add_v1, exist_ok=True)
    with open(os.path.join(add_v1, "model.json"), "w") as f:
        f.write('{ "operation": "add" }')
    with open(os.path.join(MODELS_DIR, "add", "config.pbtxt"), "w") as f:
        f.write('backend: "add_sub"')
    shutil.copytree(add_v1, os.path.join(MODELS_DIR, "add", "2"))

    sub_v1 = os.path.join(MODELS_DIR, "sub", "1")
    os.makedirs(sub_v1, exist_ok=True)
    with open(os.path.join(sub_v1, "model.json"), "w") as f:
        f.write('{ "operation": "sub" }')
    with open(os.path.join(MODELS_DIR, "sub", "config.pbtxt"), "w") as f:
        f.write('backend: "add_sub"')

    # Python-backend add_sub, two identical versions.
    add_sub_v1 = os.path.join(MODELS_DIR, "add_sub", "1")
    os.makedirs(add_sub_v1, exist_ok=True)
    shutil.copy(
        os.path.join(PYTHON_MODELS_DIR, "add_sub", "model.py"),
        os.path.join(add_sub_v1, "model.py"),
    )
    shutil.copy(
        os.path.join(PYTHON_MODELS_DIR, "add_sub", "config.pbtxt"),
        os.path.join(MODELS_DIR, "add_sub", "config.pbtxt"),
    )
    shutil.copytree(add_sub_v1, os.path.join(MODELS_DIR, "add_sub", "2"))

    # Generated pytorch-backend model (add_sub_pytorch). gen_manifest.py
    # must sit next to gen_qa_pytorch_model.py -- TRI-1744.
    shutil.copy(
        os.path.join(COMMON_DIR, "gen_qa_pytorch_model.py"),
        os.path.join(OUTPUT_DIR, "gen_qa_pytorch_model.py"),
    )
    shutil.copy(
        os.path.join(COMMON_DIR, "gen_manifest.py"),
        os.path.join(OUTPUT_DIR, "gen_manifest.py"),
    )
    action_log.run(
        ["python3", "gen_qa_pytorch_model.py", "-m", MODELS_DIR],
        "Generating the add_sub_pytorch model (gen_qa_pytorch_model.py)",
        cwd=OUTPUT_DIR,
        check=True,
    )

    return MODELS_DIR


def _ready(base, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/v2/health/ready", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


@pytest.fixture(scope="session", autouse=True)
def _running_server(model_repository):
    server_log = os.path.join(OUTPUT_DIR, "python_based_backends_server.log")
    cmd = [
        SERVER,
        "--model-repository=%s" % model_repository,
        "--backend-directory=%s" % BACKEND_DIR,
        "--model-control-mode=explicit",
        "--http-port=%d" % HTTP_PORT,
        "--grpc-port=%d" % GRPC_PORT,
        "--log-verbose=1",
    ]
    with open(server_log, "wb") as log_fh:
        proc = action_log.popen(
            cmd,
            "Starting tritonserver for python_based_backends (see %s)" % server_log,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
    base = "http://%s:%d" % (TRITONSERVER_IPADDR, HTTP_PORT)
    try:
        if not _ready(base, STARTUP_TIMEOUT_S):
            proc.terminate()
            with contextlib.suppress(Exception):
                proc.wait(timeout=30)
            pytest.fail(
                "server did not become ready; see %s" % server_log, pytrace=False
            )
        yield
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=30)
