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
"""Fixtures for the `ensemble` scenario: a CPU ensemble (add_sub_1 ->
add_sub_2) and a GPU ensemble (DATADIR-provided libtorch model, patched to
KIND_GPU), served by one default-model-control server (everything auto-loads
at startup, matching test.sh -- no --model-control-mode override).

Ported from qa/L0_backend_python/ensemble/test.sh. ensemble_test.py is
copied in unchanged (see ../../model_control/tests/conftest.py for why
that's safe).

The bash version's before/after `get_shm_pages` (/dev/shm entry count) leak
check is reproduced in the shm_pages_before fixture below (see its
docstring for why it has to be a fixture teardown, not a test function).
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

STARTUP_TIMEOUT_S = 120
OUTPUT_DIR = os.path.join(SCENARIO_DIR, "output")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")


def _repo_version():
    version = os.environ.get("REPO_VERSION") or os.environ.get(
        "NVIDIA_TRITON_SERVER_VERSION"
    )
    if not version:
        pytest.fail(
            "REPO_VERSION or NVIDIA_TRITON_SERVER_VERSION must be set "
            "(selects the QA model repository under DATADIR)",
            pytrace=False,
        )
    arch = os.environ.get("TEST_REPO_ARCH")
    return "%s_%s" % (version, arch) if arch else version


def get_shm_pages():
    return len(os.listdir("/dev/shm"))


@pytest.fixture(scope="session")
def model_repository():
    if os.path.isdir(MODELS_DIR):
        shutil.rmtree(MODELS_DIR)
    os.makedirs(MODELS_DIR)

    ensemble_dst = os.path.join(MODELS_DIR, "ensemble", "1")
    os.makedirs(ensemble_dst, exist_ok=True)
    shutil.copy(
        os.path.join(PYTHON_MODELS_DIR, "ensemble", "config.pbtxt"),
        os.path.join(MODELS_DIR, "ensemble", "config.pbtxt"),
    )

    for variant in ("add_sub_1", "add_sub_2"):
        dst = os.path.join(MODELS_DIR, variant, "1")
        os.makedirs(dst, exist_ok=True)
        shutil.copy(
            os.path.join(PYTHON_MODELS_DIR, "add_sub", "config.pbtxt"),
            os.path.join(MODELS_DIR, variant, "config.pbtxt"),
        )
        shutil.copy(
            os.path.join(PYTHON_MODELS_DIR, "add_sub", "model.py"),
            os.path.join(dst, "model.py"),
        )

    ensemble_gpu_dst = os.path.join(MODELS_DIR, "ensemble_gpu", "1")
    os.makedirs(ensemble_gpu_dst, exist_ok=True)
    shutil.copy(
        os.path.join(PYTHON_MODELS_DIR, "ensemble_gpu", "config.pbtxt"),
        os.path.join(MODELS_DIR, "ensemble_gpu", "config.pbtxt"),
    )

    datadir = os.environ.get("DATADIR") or os.path.join(
        "/data/inferenceserver", _repo_version()
    )
    libtorch_src = os.path.join(
        datadir, "qa_model_repository", "libtorch_float32_float32_float32"
    )
    libtorch_dst = os.path.join(MODELS_DIR, "libtorch_float32_float32_float32")
    shutil.copytree(libtorch_src, libtorch_dst)
    shutil.rmtree(os.path.join(libtorch_dst, "2"), ignore_errors=True)
    shutil.rmtree(os.path.join(libtorch_dst, "3"), ignore_errors=True)
    config_path = os.path.join(libtorch_dst, "config.pbtxt")
    with open(config_path) as f:
        lines = f.readlines()
    with open(config_path, "w") as f:
        for line in lines:
            if line.startswith("max_batch_size:"):
                f.write("max_batch_size: 0\n")
            elif line.startswith("version_policy:"):
                continue
            else:
                f.write(line)
        f.write("instance_group [ { kind: KIND_GPU }]\n")

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


@pytest.fixture(scope="session")
def shm_pages_before():
    """Assert /dev/shm is back to its pre-server entry count once the
    server (a dependent of this fixture) has fully torn down.

    Fixture teardown runs in reverse dependency order, so this yields
    *before* _running_server starts and its post-yield assertion runs
    *after* _running_server has stopped -- the same ordering as test.sh's
    prev_num_pages=$(get_shm_pages) / run_server / ... / kill_server /
    current_num_pages=$(get_shm_pages) check. A plain test function can't
    reproduce this: every test runs before session-fixture teardown, i.e.
    before the server (and thus its shm segments) has actually exited.
    """
    before = get_shm_pages()
    yield before
    after = get_shm_pages()
    assert after == before, (
        "shared memory pages were not cleaned up properly: %d before "
        "starting triton, %d after stopping it (see /dev/shm)" % (before, after)
    )


@pytest.fixture(scope="session", autouse=True)
def _running_server(model_repository, shm_pages_before):
    server_log = os.path.join(OUTPUT_DIR, "ensemble_server.log")
    cmd = [
        SERVER,
        "--model-repository=%s" % model_repository,
        "--backend-directory=%s" % BACKEND_DIR,
        "--http-port=%d" % HTTP_PORT,
        "--log-verbose=1",
    ]
    with open(server_log, "wb") as log_fh:
        proc = action_log.popen(
            cmd,
            "Starting tritonserver for ensemble (CPU add_sub_1->add_sub_2 "
            "and GPU libtorch ensembles, see %s)" % server_log,
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
