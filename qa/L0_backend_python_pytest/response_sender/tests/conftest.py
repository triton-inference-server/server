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
"""Fixtures for the `response_sender` scenario.

Ported from qa/L0_backend_python/response_sender/test.sh. The bash version
runs two separate server sessions (one for the 8 response_sender
model-config variants, one -- with the models directory wiped and rebuilt
-- for response_sender_complete_final alone). There is no functional
dependency between the two model sets, so this port builds one combined
repository and runs both test modules against a single session-scoped
server instead: same coverage, fewer server restarts.

Self-contained: everything needed to run this scenario lives under
qa/L0_backend_python_pytest/response_sender/, except the model sources
(qa/python_models/response_sender*), which are shared, read-only assets
already present in the QA image.
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

sys.path.append(os.path.join(QA_DIR, "common"))
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
SERVER_LOG = os.path.join(OUTPUT_DIR, "response_sender.server.log")

# (model_name, model.py source, extra config.pbtxt lines)
_VARIANTS = [
    ("response_sender", "model.py", []),
    (
        "response_sender_decoupled",
        "model.py",
        ["model_transaction_policy { decoupled: True }"],
    ),
    ("response_sender_async", "model_async.py", []),
    (
        "response_sender_decoupled_async",
        "model_async.py",
        ["model_transaction_policy { decoupled: True }"],
    ),
    (
        "response_sender_batching",
        "model.py",
        ["dynamic_batching { max_queue_delay_microseconds: 500000 }"],
    ),
    (
        "response_sender_decoupled_batching",
        "model.py",
        [
            "model_transaction_policy { decoupled: True }",
            "dynamic_batching { max_queue_delay_microseconds: 500000 }",
        ],
    ),
    (
        "response_sender_async_batching",
        "model_async.py",
        ["dynamic_batching { max_queue_delay_microseconds: 500000 }"],
    ),
    (
        "response_sender_decoupled_async_batching",
        "model_async.py",
        [
            "model_transaction_policy { decoupled: True }",
            "dynamic_batching { max_queue_delay_microseconds: 500000 }",
        ],
    ),
]


@pytest.fixture(scope="session")
def model_repository():
    src = os.path.join(PYTHON_MODELS_DIR, "response_sender")
    if os.path.isdir(MODELS_DIR):
        shutil.rmtree(MODELS_DIR)
    os.makedirs(MODELS_DIR)

    for name, model_source, extra_config in _VARIANTS:
        dst = os.path.join(MODELS_DIR, name)
        os.makedirs(os.path.join(dst, "1"))
        shutil.copy(os.path.join(src, "model_common.py"), os.path.join(dst, "1"))
        shutil.copy(os.path.join(src, model_source), os.path.join(dst, "1", "model.py"))
        shutil.copy(os.path.join(src, "config.pbtxt"), dst)
        if extra_config:
            with open(os.path.join(dst, "config.pbtxt"), "a") as f:
                f.write("\n" + "\n".join(extra_config) + "\n")

    cf_src = os.path.join(PYTHON_MODELS_DIR, "response_sender_complete_final")
    cf_dst = os.path.join(MODELS_DIR, "response_sender_complete_final")
    os.makedirs(os.path.join(cf_dst, "1"))
    shutil.copy(os.path.join(cf_src, "model.py"), os.path.join(cf_dst, "1"))
    shutil.copy(os.path.join(cf_src, "config.pbtxt"), cf_dst)

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
def server(model_repository):
    """One server for both test modules; sets SERVER_LOG for the tests that read it."""
    os.environ["SERVER_LOG"] = SERVER_LOG
    cmd = [
        SERVER,
        "--model-repository=%s" % model_repository,
        "--backend-directory=%s" % BACKEND_DIR,
        "--http-port=%d" % HTTP_PORT,
        "--grpc-port=%d" % GRPC_PORT,
        "--log-verbose=1",
    ]
    with open(SERVER_LOG, "wb") as log_fh:
        proc = action_log.popen(
            cmd,
            "Starting tritonserver for response_sender (see %s)" % SERVER_LOG,
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
                "server did not become ready; see %s" % SERVER_LOG, pytrace=False
            )
        yield base
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=30)
