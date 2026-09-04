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
"""Fixtures for the `custom_metrics` scenario: one self-testing model.

Ported from qa/L0_backend_python/custom_metrics/test.sh.

Self-contained: everything needed to run this scenario lives under
qa/L0_backend_python_pytest/custom_metrics/, except the model source
(qa/python_models/custom_metrics) and the tritonserver binary itself, both
shared, read-only assets -- not orchestration state from a sibling scenario.
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

TRITON_DIR = os.environ.get("TRITON_DIR", "/opt/tritonserver")
SERVER = os.environ.get("SERVER", os.path.join(TRITON_DIR, "bin", "tritonserver"))
BACKEND_DIR = os.environ.get("BACKEND_DIR", os.path.join(TRITON_DIR, "backends"))
TRITONSERVER_IPADDR = os.environ.get("TRITONSERVER_IPADDR", "localhost")
HTTP_PORT = int(os.environ.get("TRITONSERVER_HTTP_PORT", "8000"))
GRPC_PORT = int(os.environ.get("TRITONSERVER_GRPC_PORT", "8001"))

STARTUP_TIMEOUT_S = 120
MODEL_NAME = "custom_metrics"

OUTPUT_DIR = os.path.join(SCENARIO_DIR, "output")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")


@pytest.fixture(scope="session")
def model_repository():
    src = os.path.join(QA_DIR, "python_models", MODEL_NAME)
    dst = os.path.join(MODELS_DIR, MODEL_NAME)
    if os.path.isdir(MODELS_DIR):
        shutil.rmtree(MODELS_DIR)
    os.makedirs(os.path.join(dst, "1"))
    shutil.copy(os.path.join(src, "config.pbtxt"), dst)
    shutil.copy(os.path.join(src, "model.py"), os.path.join(dst, "1"))
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


@contextlib.contextmanager
def serve(model_repository, server_log, extra_args=()):
    cmd = [
        SERVER,
        "--model-repository=%s" % model_repository,
        "--backend-directory=%s" % BACKEND_DIR,
        "--http-port=%d" % HTTP_PORT,
        "--grpc-port=%d" % GRPC_PORT,
        "--log-verbose=1",
        *extra_args,
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
        yield base
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=30)
