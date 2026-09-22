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
"""Fixtures for the `async_execute` scenario: one server, one test module.

Ported from qa/L0_backend_python/async_execute/test.sh: unlike bls, the
original script runs its whole client (concurrency_test.py) as a single
pytest invocation against one already-running server -- so this stays one
session-scoped server for the whole module, not one server per test, to
match the original's behavior (and because the concurrency assertions
in test_async_execute.py depend on request timing that a fresh server
per test would not change, so per-test isolation buys nothing here).

Self-contained: everything needed lives under
qa/L0_backend_python_pytest/async_execute/, except the model sources
(qa/python_models/async_execute_decouple*) and the tritonserver binary
itself, both shared, read-only assets.
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

sys.path.append(os.path.join(QA_DIR, "common"))
import action_log  # noqa: E402

TRITON_DIR = os.environ.get("TRITON_DIR", "/opt/tritonserver")
SERVER = os.environ.get("SERVER", os.path.join(TRITON_DIR, "bin", "tritonserver"))
BACKEND_DIR = os.environ.get("BACKEND_DIR", os.path.join(TRITON_DIR, "backends"))
TRITONSERVER_IPADDR = os.environ.get("TRITONSERVER_IPADDR", "localhost")
HTTP_PORT = int(os.environ.get("TRITONSERVER_HTTP_PORT", "8000"))
GRPC_PORT = int(os.environ.get("TRITONSERVER_GRPC_PORT", "8001"))

STARTUP_TIMEOUT_S = 120
MODEL_NAMES = ["async_execute_decouple", "async_execute_decouple_bls"]

OUTPUT_DIR = os.path.join(SCENARIO_DIR, "output")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")


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


@pytest.fixture(scope="session")
def server_log_path():
    return os.path.join(OUTPUT_DIR, "concurrency_test.server.log")


@pytest.fixture(scope="session", autouse=True)
def running_server(server_log_path):
    if os.path.isdir(MODELS_DIR):
        shutil.rmtree(MODELS_DIR)
    for name in MODEL_NAMES:
        src = os.path.join(QA_DIR, "python_models", name)
        dst = os.path.join(MODELS_DIR, name)
        os.makedirs(os.path.join(dst, "1"))
        shutil.copy(os.path.join(src, "config.pbtxt"), dst)
        shutil.copy(os.path.join(src, "model.py"), os.path.join(dst, "1"))

    cmd = [
        SERVER,
        "--model-repository=%s" % MODELS_DIR,
        "--backend-directory=%s" % BACKEND_DIR,
        "--http-port=%d" % HTTP_PORT,
        "--grpc-port=%d" % GRPC_PORT,
        "--log-verbose=1",
    ]
    with open(server_log_path, "wb") as log_fh:
        proc = action_log.popen(
            cmd,
            "Starting tritonserver for async_execute's concurrency tests "
            "(see %s)" % server_log_path,
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
                "server did not become ready; see %s" % server_log_path,
                pytrace=False,
            )
        os.environ["SERVER_LOG"] = server_log_path
        yield base
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=30)
