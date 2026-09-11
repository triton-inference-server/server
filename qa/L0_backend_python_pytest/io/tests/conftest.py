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
"""Fixtures for the `io` scenario.

Ported from qa/L0_backend_python/io/test.sh, which runs six independent
model repos, each against its own tritonserver instance (except
test_requested_output_default/decoupled, which share one server), plus a
torch pin for the dlpack GPU/CPU round trips.

io_test.py is carried over verbatim (not rewritten) since it is already a
plain unittest.TestCase client; test_io.py below drives it directly per
scenario instead of relying on pytest's unittest auto-collection, because
each test method needs its own model repository built and server started
first -- something auto-collected unittest methods can't ask for.

collect_ignore keeps pytest from also auto-collecting io_test.py on its
own (its filename matches pytest's default *_test.py pattern) -- without
it, every IOTest method would run twice: once via test_io.py's deliberate
per-method server setup, once via pytest's unittest auto-discovery with no
server running at all.
"""

import contextlib
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import pytest

collect_ignore = ["io_test.py"]

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

TORCH_SPEC = os.environ.get("IO_TORCH_SPEC", "torch==2.3.1+cu118")
TORCH_INDEX_URL = os.environ.get(
    "IO_TORCH_INDEX_URL", "https://download.pytorch.org/whl/torch_stable.html"
)


@pytest.fixture(scope="session")
def pinned_torch():
    if os.environ.get("IO_SKIP_TORCH_PIN") == "1":
        yield
        return
    action_log.run(
        ["pip3", "uninstall", "-y", "torch"],
        "Uninstalling the image's default torch before pinning io's version",
        check=False,
        capture_output=True,
    )
    action_log.run(
        ["pip3", "install", *TORCH_SPEC.split(), "-f", TORCH_INDEX_URL],
        "Installing pinned torch for io's dlpack GPU/CPU round trips",
        check=True,
    )
    yield


def _fresh_models_dir():
    if os.path.isdir(MODELS_DIR):
        shutil.rmtree(MODELS_DIR)
    os.makedirs(MODELS_DIR)
    return MODELS_DIR


def _copy_model(name, dst_name=None, dst_root=MODELS_DIR):
    dst_name = dst_name or name
    src = os.path.join(PYTHON_MODELS_DIR, name)
    dst = os.path.join(dst_root, dst_name)
    os.makedirs(os.path.join(dst, "1"))
    shutil.copy(os.path.join(src, "model.py"), os.path.join(dst, "1"))
    shutil.copy(os.path.join(src, "config.pbtxt"), dst)
    return dst


def _rename_model(config_path, new_name):
    with open(config_path) as f:
        lines = f.readlines()
    with open(config_path, "w") as f:
        for line in lines:
            if line.startswith("name:"):
                f.write('name: "%s"\n' % new_name)
            else:
                f.write(line)


def ensemble_repo(trial):
    """default/decoupled dlpack_io_identity x3 + ensemble_io."""
    _fresh_models_dir()
    source = (
        "dlpack_io_identity" if trial == "default" else "dlpack_io_identity_decoupled"
    )
    for i in (1, 2, 3):
        name = "dlpack_io_identity_%d" % i
        dst = _copy_model(source, dst_name=name)
        _rename_model(os.path.join(dst, "config.pbtxt"), name)
    ensemble_dst = os.path.join(MODELS_DIR, "ensemble_io")
    os.makedirs(os.path.join(ensemble_dst, "1"))
    shutil.copy(
        os.path.join(PYTHON_MODELS_DIR, "ensemble_io", "config.pbtxt"), ensemble_dst
    )
    return MODELS_DIR


def empty_gpu_output_repo():
    _fresh_models_dir()
    _copy_model("dlpack_empty_output")
    return MODELS_DIR


def variable_gpu_output_repo():
    _fresh_models_dir()
    _copy_model("variable_gpu_output")
    return MODELS_DIR


def requested_output_repo():
    _fresh_models_dir()
    _copy_model("add_sub")
    _copy_model("dlpack_io_identity_decoupled")
    return MODELS_DIR


def requested_output_decoupled_prior_crash_repo():
    _fresh_models_dir()
    dst = os.path.join(MODELS_DIR, "llm")
    os.makedirs(os.path.join(dst, "1"))
    local_model_dir = os.path.join(HERE, "requested_output_model")
    shutil.copy(os.path.join(local_model_dir, "model.py"), os.path.join(dst, "1"))
    shutil.copy(os.path.join(local_model_dir, "config.pbtxt"), dst)
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
def serve(model_repository, server_log):
    cmd = [
        SERVER,
        "--model-repository=%s" % model_repository,
        "--backend-directory=%s" % BACKEND_DIR,
        "--http-port=%d" % HTTP_PORT,
        "--grpc-port=%d" % GRPC_PORT,
        "--log-verbose=1",
    ]
    with open(server_log, "wb") as log_fh:
        proc = action_log.popen(
            cmd,
            "Starting tritonserver for io (model_repository=%s)" % model_repository,
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
        yield base
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=30)
