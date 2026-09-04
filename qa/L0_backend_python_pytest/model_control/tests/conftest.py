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
"""Fixtures for the `model_control` scenario: explicit load/unload of
identity_fp32 (+ its simple_identity_fp32 ensemble wrapper) and model-ID
validation, both against one server started with
--model-control-mode=explicit and nothing preloaded.

Ported from qa/L0_backend_python/model_control/test.sh. The client test
code (model_control_test.py) is copied in unchanged: it is a plain
unittest.TestCase file whose sys.path.append("../../common") is relative to
the process cwd (pytest.py's subprocess.call sets cwd=<scenario_dir>), which
resolves to qa/common exactly as it did in the original layout.
"""

import contextlib
import os
import shutil
import subprocess
import time
import urllib.request

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIO_DIR = os.path.dirname(HERE)
QA_DIR = os.path.dirname(os.path.dirname(SCENARIO_DIR))
PYTHON_MODELS_DIR = os.path.join(QA_DIR, "python_models")

TRITON_DIR = os.environ.get("TRITON_DIR", "/opt/tritonserver")
SERVER = os.environ.get("SERVER", os.path.join(TRITON_DIR, "bin", "tritonserver"))
BACKEND_DIR = os.environ.get("BACKEND_DIR", os.path.join(TRITON_DIR, "backends"))
TRITONSERVER_IPADDR = os.environ.get("TRITONSERVER_IPADDR", "localhost")
HTTP_PORT = int(os.environ.get("TRITONSERVER_HTTP_PORT", "8000"))

STARTUP_TIMEOUT_S = 120
OUTPUT_DIR = os.path.join(SCENARIO_DIR, "output")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")


def _copy_model(name, model_py=True):
    src = os.path.join(PYTHON_MODELS_DIR, name)
    dst = os.path.join(MODELS_DIR, name)
    os.makedirs(os.path.join(dst, "1"), exist_ok=True)
    shutil.copy(os.path.join(src, "config.pbtxt"), dst)
    if model_py:
        shutil.copy(os.path.join(src, "model.py"), os.path.join(dst, "1"))


@pytest.fixture(scope="session")
def model_repository():
    if os.path.isdir(MODELS_DIR):
        shutil.rmtree(MODELS_DIR)
    os.makedirs(MODELS_DIR)
    _copy_model("identity_fp32")
    # simple_identity_fp32 is an ensemble wrapper around identity_fp32: only
    # a config.pbtxt, no model.py, matching the original test.sh setup.
    _copy_model("simple_identity_fp32", model_py=False)
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
    """One server for the whole session -- matches test.sh's single run_server."""
    server_log = os.path.join(OUTPUT_DIR, "model_control_server.log")
    cmd = [
        SERVER,
        "--model-repository=%s" % model_repository,
        "--backend-directory=%s" % BACKEND_DIR,
        "--model-control-mode=explicit",
        "--http-port=%d" % HTTP_PORT,
        "--log-verbose=1",
    ]
    with open(server_log, "wb") as log_fh:
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
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
