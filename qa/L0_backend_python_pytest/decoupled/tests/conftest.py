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
"""Fixtures for the `decoupled` scenario: identity_fp32, execute_cancel
(decoupled), response_sender_until_cancelled, square_int32 (from a
python_backend clone, like ../../bls/tests/conftest.py's square model), and
dlpack_add_sub, served by one default-model-control server.

Ported from qa/L0_backend_python/decoupled/test.sh. decoupled_test.py is
copied in unchanged (see ../../model_control/tests/conftest.py for why
that's safe).

The bash version's post-kill_server verify_log_counts (grep thresholds on
the server log, including exact "Finalize invoked"/"Finalize complete..."
counts that only exist once the server has actually exited) is reproduced
in the model_repository fixture's teardown -- see its docstring, and
../../ensemble/tests/conftest.py's shm_pages_before fixture for the same
"this must run after server teardown, not as a test function" reasoning.
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

# qa/L0_backend_python/decoupled/models/decoupled_* are statically
# checked-in model repos (not sourced from qa/python_models/) -- the
# original test.sh never copies them explicitly because it runs with them
# already present in its cwd. Every decoupled_bls*/decoupled_execute_error/
# etc. test method needs one of these; without them the whole suite fails
# (verified live: all 11 fail identically, model not found).
STATIC_MODELS_DIR = os.path.join(QA_DIR, "L0_backend_python", "decoupled", "models")

TRITON_DIR = os.environ.get("TRITON_DIR", "/opt/tritonserver")
SERVER = os.environ.get("SERVER", os.path.join(TRITON_DIR, "bin", "tritonserver"))
BACKEND_DIR = os.environ.get("BACKEND_DIR", os.path.join(TRITON_DIR, "backends"))
TRITONSERVER_IPADDR = os.environ.get("TRITONSERVER_IPADDR", "localhost")
HTTP_PORT = int(os.environ.get("TRITONSERVER_HTTP_PORT", "8000"))
GRPC_PORT = int(os.environ.get("TRITONSERVER_GRPC_PORT", "8001"))

STARTUP_TIMEOUT_S = 120
OUTPUT_DIR = os.path.join(SCENARIO_DIR, "output")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
SERVER_LOG = os.path.join(OUTPUT_DIR, "decoupled_server.log")

TRITON_REPO_ORGANIZATION = os.environ.get(
    "TRITON_REPO_ORGANIZATION", "https://github.com/triton-inference-server"
)
PYTHON_BACKEND_REPO_TAG = os.environ.get("PYTHON_BACKEND_REPO_TAG", "main")

TORCH_SPEC = os.environ.get("DECOUPLED_TORCH_SPEC", "torch==2.3.1+cu118")
TORCH_INDEX_URL = os.environ.get(
    "DECOUPLED_TORCH_INDEX_URL", "https://download.pytorch.org/whl/torch_stable.html"
)


@pytest.fixture(scope="session")
def pinned_torch():
    if os.environ.get("DECOUPLED_SKIP_TORCH_PIN") == "1":
        yield
        return
    subprocess.run(
        ["pip3", "uninstall", "-y", "torch"], check=False, capture_output=True
    )
    subprocess.run(
        ["pip3", "install", *TORCH_SPEC.split(), "-f", TORCH_INDEX_URL], check=True
    )
    yield


def _copy_model(name, extra_config_lines=()):
    src = os.path.join(PYTHON_MODELS_DIR, name)
    dst = os.path.join(MODELS_DIR, name)
    os.makedirs(os.path.join(dst, "1"), exist_ok=True)
    shutil.copy(os.path.join(src, "config.pbtxt"), dst)
    shutil.copy(os.path.join(src, "model.py"), os.path.join(dst, "1"))
    if extra_config_lines:
        with open(os.path.join(dst, "config.pbtxt"), "a") as f:
            for line in extra_config_lines:
                f.write(line + "\n")


@pytest.fixture(scope="session")
def model_repository(pinned_torch):
    if os.path.isdir(MODELS_DIR):
        shutil.rmtree(MODELS_DIR)
    shutil.copytree(STATIC_MODELS_DIR, MODELS_DIR)

    _copy_model("identity_fp32")
    _copy_model(
        "execute_cancel",
        extra_config_lines=["model_transaction_policy { decoupled: True }"],
    )
    _copy_model("response_sender_until_cancelled")
    _copy_model("dlpack_add_sub")

    clone_dir = os.path.join(OUTPUT_DIR, "python_backend")
    if os.path.isdir(clone_dir):
        shutil.rmtree(clone_dir)
    subprocess.run(
        [
            "git",
            "clone",
            "%s/python_backend" % TRITON_REPO_ORGANIZATION,
            "-b",
            PYTHON_BACKEND_REPO_TAG,
            clone_dir,
        ],
        check=True,
    )
    square_dst = os.path.join(MODELS_DIR, "square_int32", "1")
    os.makedirs(square_dst)
    shutil.copy(
        os.path.join(clone_dir, "examples", "decoupled", "square_model.py"),
        os.path.join(square_dst, "model.py"),
    )
    shutil.copy(
        os.path.join(clone_dir, "examples", "decoupled", "square_config.pbtxt"),
        os.path.join(MODELS_DIR, "square_int32", "config.pbtxt"),
    )

    yield MODELS_DIR

    # verify_log_counts, from test.sh -- must run after the server (a
    # dependent of this fixture, via _running_server) has exited, since
    # Finalize invoked/complete only appear in the log once every python
    # backend instance has actually shut down.
    with open(SERVER_LOG) as f:
        log_text = f.read()
    for marker in ("Specific Msg!", "Info Msg!", "Warning Msg!", "Error Msg!"):
        assert log_text.count(marker) >= 1, "%r count incorrect in server log" % marker
    for marker in ("Finalize invoked", "Finalize complete..."):
        count = log_text.count(marker)
        assert count == 3, "expected 3 occurrences of %r, found %d" % (marker, count)


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
    # test_decoupled_execute_cancel reads the server log directly; point it
    # at SERVER_LOG rather than hardcoding a path relative to the original
    # bash script's cwd (which no longer matches ours -- see
    # ../../response_sender/tests/conftest.py for the same pattern).
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
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    base = "http://%s:%d" % (TRITONSERVER_IPADDR, HTTP_PORT)
    try:
        if not _ready(base, STARTUP_TIMEOUT_S):
            proc.terminate()
            with contextlib.suppress(Exception):
                proc.wait(timeout=30)
            pytest.fail(
                "server did not become ready; see %s" % SERVER_LOG, pytrace=False
            )
        yield
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=30)
