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
"""Fixtures for the `model_readiness` scenario.

Ported from qa/L0_backend_python/model_readiness/test.sh: two independent
phases, each with its own model repository and server --
(1) TRITONBACKEND_ModelInstanceReady after the model's python stub is
killed with SIGSEGV/SIGKILL, (2) the user-defined is_ready() function
under various return values/delays. test_model_readiness_client.py is
carried over verbatim as the assertion source; test_model_readiness.py
drives it directly via _run(), since each test needs its own server
lifecycle -- something auto-collected unittest methods can't ask for.

collect_ignore keeps pytest from also auto-collecting
test_model_readiness_client.py on its own (its filename matches pytest's
default test_*.py pattern) -- without it, every TestModelReadiness/
TestUserDefinedModelReadinessFunction method runs twice: once via
test_model_readiness.py's deliberate per-test server setup, once via
pytest's unittest auto-discovery with no server running at all.
"""

import contextlib
import os
import shutil
import subprocess
import time
import urllib.request

import pytest

collect_ignore = ["test_model_readiness_client.py"]

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIO_DIR = os.path.dirname(HERE)
QA_DIR = os.path.dirname(os.path.dirname(SCENARIO_DIR))
PYTHON_MODELS_DIR = os.path.join(QA_DIR, "python_models")

TRITON_DIR = os.environ.get("TRITON_DIR", "/opt/tritonserver")
SERVER = os.environ.get("SERVER", os.path.join(TRITON_DIR, "bin", "tritonserver"))
BACKEND_DIR = os.environ.get("BACKEND_DIR", os.path.join(TRITON_DIR, "backends"))
TRITONSERVER_IPADDR = os.environ.get("TRITONSERVER_IPADDR", "localhost")
HTTP_PORT = int(os.environ.get("TRITONSERVER_HTTP_PORT", "8000"))
GRPC_PORT = int(os.environ.get("TRITONSERVER_GRPC_PORT", "8001"))

STARTUP_TIMEOUT_S = 120
OUTPUT_DIR = os.path.join(SCENARIO_DIR, "output")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")


def identity_model_repository():
    """Just identity_fp32 -- used by the signal-kill readiness phase."""
    if os.path.isdir(MODELS_DIR):
        shutil.rmtree(MODELS_DIR)
    os.makedirs(MODELS_DIR)
    src = os.path.join(PYTHON_MODELS_DIR, "identity_fp32")
    dst = os.path.join(MODELS_DIR, "identity_fp32")
    os.makedirs(os.path.join(dst, "1"))
    shutil.copy(os.path.join(src, "model.py"), os.path.join(dst, "1"))
    shutil.copy(os.path.join(src, "config.pbtxt"), dst)
    return MODELS_DIR


# (model_name, source .py under tests/test_models/, READINESS_FN_RETURN_VALUE,
# READINESS_FN_DELAY_SECS)
_READINESS_MODELS = [
    ("is_ready_fn_returns_true", "readiness_model.py", "true", "0.1"),
    ("is_ready_fn_returns_false", "readiness_model.py", "false", "0.1"),
    ("is_ready_fn_raises_error", "readiness_model.py", "exception", "0.1"),
    ("is_ready_fn_returns_non_boolean", "readiness_model.py", "non_boolean", "0.1"),
    ("is_ready_fn_timeout", "readiness_model.py", "true", "8"),
    (
        "is_ready_fn_coroutine_returns_true",
        "readiness_coroutine_model.py",
        "coroutine",
        "0.1",
    ),
]


def readiness_fn_model_repository():
    """identity_fp32 config as a base, plus the decoupled is_ready model."""
    if os.path.isdir(MODELS_DIR):
        shutil.rmtree(MODELS_DIR)
    os.makedirs(MODELS_DIR)

    base_config = os.path.join(PYTHON_MODELS_DIR, "identity_fp32", "config.pbtxt")
    with open(base_config) as f:
        base_config_text = f.read()

    for name, source, return_value, delay_secs in _READINESS_MODELS:
        dst = os.path.join(MODELS_DIR, name)
        os.makedirs(os.path.join(dst, "1"))
        shutil.copy(
            os.path.join(HERE, "test_models", source),
            os.path.join(dst, "1", "model.py"),
        )
        config_lines = []
        for line in base_config_text.splitlines():
            if line.startswith("name:"):
                config_lines.append('name: "%s"' % name)
            else:
                config_lines.append(line)
        config_lines.append("parameters: {")
        config_lines.append('  key: "READINESS_FN_RETURN_VALUE"')
        config_lines.append('  value: { string_value: "%s" }' % return_value)
        config_lines.append("}")
        config_lines.append("parameters: {")
        config_lines.append('  key: "READINESS_FN_DELAY_SECS"')
        config_lines.append('  value: { string_value: "%s" }' % delay_secs)
        config_lines.append("}")
        with open(os.path.join(dst, "config.pbtxt"), "w") as f:
            f.write("\n".join(config_lines) + "\n")

    decoupled_src = os.path.join(
        HERE, "test_models", "is_ready_fn_returns_true_decoupled"
    )
    decoupled_dst = os.path.join(MODELS_DIR, "is_ready_fn_returns_true_decoupled")
    os.makedirs(os.path.join(decoupled_dst, "1"))
    shutil.copy(
        os.path.join(decoupled_src, "model.py"), os.path.join(decoupled_dst, "1")
    )
    shutil.copy(os.path.join(decoupled_src, "config.pbtxt"), decoupled_dst)

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
