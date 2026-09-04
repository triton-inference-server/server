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
"""Fixtures for the `examples` scenario: the docs/examples/*.py sample
models and clients from the python_backend repo, each served one model at
a time.

Ported from qa/L0_backend_python/examples/test.sh. Every example lives in
a git clone of python_backend (not qa/python_models/*, unlike every other
scenario) -- the clone is this scenario's real dependency, not a shared QA
asset, so it lives under this scenario's own output/, not qa/.

Self-contained: everything needed to run this scenario lives under
qa/L0_backend_python_pytest/examples/, except the tritonserver binary
itself.
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

TRITON_DIR = os.environ.get("TRITON_DIR", "/opt/tritonserver")
SERVER = os.environ.get("SERVER", os.path.join(TRITON_DIR, "bin", "tritonserver"))
BACKEND_DIR = os.environ.get("BACKEND_DIR", os.path.join(TRITON_DIR, "backends"))
TRITONSERVER_IPADDR = os.environ.get("TRITONSERVER_IPADDR", "localhost")
HTTP_PORT = int(os.environ.get("TRITONSERVER_HTTP_PORT", "8000"))

STARTUP_TIMEOUT_S = 120

# Everything this run generates (python_backend clone, per-example model
# repos, server logs) lives here, kept out of tests/ and off of source, so
# cleanup is one `rm -rf output`.
OUTPUT_DIR = os.path.join(SCENARIO_DIR, "output")

TRITON_REPO_ORGANIZATION = os.environ.get(
    "TRITON_REPO_ORGANIZATION", "https://github.com/triton-inference-server"
)
PYTHON_BACKEND_REPO_TAG = os.environ.get("PYTHON_BACKEND_REPO_TAG", "main")
TEST_JETSON = os.environ.get("TEST_JETSON", "0") == "1"

CLONE_MAX_RETRIES = 3


@pytest.fixture(scope="session")
def pinned_deps():
    """Match qa/L0_backend_python/examples/test.sh's environment: torch
    2.5.0/torchvision 0.20.0 + numpy>=2 (this scenario is also used as the
    numpy-2.x coverage case) + validators (instance_kind example) + jax
    (skipped on Jetson, matching the original)."""
    if os.environ.get("EXAMPLES_SKIP_DEPS") == "1":
        yield
        return
    subprocess.run(["pip3", "uninstall", "-y", "torch"], check=False)
    subprocess.run(["pip3", "uninstall", "-y", "numpy"], check=False)
    subprocess.run(["pip3", "install", "numpy>=2"], check=True)
    subprocess.run(
        [
            "pip3",
            "install",
            "torch==2.5.0",
            "torchvision==0.20.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ],
        check=True,
    )
    subprocess.run(["pip3", "install", "validators"], check=True)
    if not TEST_JETSON:
        subprocess.run(["pip3", "install", "-U", "jax[cuda12]"], check=True)
    yield


@pytest.fixture(scope="session")
def python_backend_clone(pinned_deps):
    clone_dir = os.path.join(OUTPUT_DIR, "python_backend")
    if os.path.isdir(clone_dir):
        shutil.rmtree(clone_dir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    last_err = None
    for attempt in range(1, CLONE_MAX_RETRIES + 1):
        if os.path.isdir(clone_dir):
            shutil.rmtree(clone_dir)
        try:
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
            return clone_dir
        except subprocess.CalledProcessError as e:
            last_err = e
            time.sleep(10)
    pytest.fail(
        "Failed to clone python_backend after %d attempts: %s"
        % (CLONE_MAX_RETRIES, last_err),
        pytrace=False,
    )


def build_model_repo(clone_dir, model_name, model_py_rel, config_rel, extra_models=()):
    """One-model repo for a single example, matching the original's
    `mkdir -p models/<name>/1 && cp examples/.../model.py ... && cp
    examples/.../config.pbtxt ...` per-example setup.

    extra_models: (name, model_py_rel, config_rel) tuples for models this
    example calls into via BLS. The original test.sh never clears its
    models/ directory between examples, so add_sub (the very first example
    it sets up) stays loaded for every later example -- including every
    bls_* example, all of which BLS-call into it. Verified live: without
    this, every bls_* example fails with "Model add_sub is not ready."
    """
    models_dir = os.path.join(OUTPUT_DIR, "models_%s" % model_name)
    if os.path.isdir(models_dir):
        shutil.rmtree(models_dir)
    dst = os.path.join(models_dir, model_name, "1")
    os.makedirs(dst)
    shutil.copy(os.path.join(clone_dir, model_py_rel), os.path.join(dst, "model.py"))
    shutil.copy(
        os.path.join(clone_dir, config_rel),
        os.path.join(models_dir, model_name, "config.pbtxt"),
    )
    for extra_name, extra_model_py_rel, extra_config_rel in extra_models:
        extra_dst = os.path.join(models_dir, extra_name, "1")
        os.makedirs(extra_dst)
        shutil.copy(
            os.path.join(clone_dir, extra_model_py_rel),
            os.path.join(extra_dst, "model.py"),
        )
        shutil.copy(
            os.path.join(clone_dir, extra_config_rel),
            os.path.join(models_dir, extra_name, "config.pbtxt"),
        )
    return models_dir


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
        "--log-verbose=1",
        "--http-port=%d" % HTTP_PORT,
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


def run_client(clone_dir, client_rel, log_path, extra_args=()):
    """Run one example's client.py from inside the clone (matches the
    original's `cd python_backend && python3 examples/.../client.py`), and
    return its combined stdout+stderr text."""
    with open(log_path, "wb") as log_fh:
        rv = subprocess.call(
            ["python3", client_rel, *extra_args],
            cwd=clone_dir,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
    with open(log_path) as f:
        return rv, f.read()
