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
"""Fixtures for the `bls` scenario: the core BLS chaining matrix.

Ported from qa/L0_backend_python/bls/test.sh, scoped to the
non_decoupled/decoupled x 64/256-MB-CUDA-pool x
{bls,bls_memory,bls_memory_async,bls_async} matrix -- the part that
exercises the regression this pilot was built to catch (TRI-1744). The
init_error / finalize_error / model_loading / parameters sub-scenarios in
the original test.sh are each independent in their own right and are left
for a follow-up split, not carried into this pilot.

Self-contained: everything needed to run this scenario lives under
qa/L0_backend_python_pytest/bls/, except the model sources
(qa/python_models/*, a python_backend clone, and the DATADIR-provided
libtorch/onnx QA model repository), which are shared, read-only/generated
assets already present in the QA image and its data volume -- not
orchestration state from a sibling scenario.
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

# Everything this run generates (model repo, python_backend clone, server
# logs) lives here, kept out of tests/ and off of source, so cleanup is one
# `rm -rf output`. Also exactly the depth the CI log-collection step scans
# (TEST_WORK_DIR/<one dir>/<file>) -- see ci/templates/utility in the
# tritonserver (GitLab) repo.
OUTPUT_DIR = os.path.join(SCENARIO_DIR, "output")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

TRITON_REPO_ORGANIZATION = os.environ.get(
    "TRITON_REPO_ORGANIZATION", "https://github.com/triton-inference-server"
)
PYTHON_BACKEND_REPO_TAG = os.environ.get("PYTHON_BACKEND_REPO_TAG", "main")

# The dlpack CPU/GPU round-trip covered by this scenario needs a specific
# torch build; the QA image's default torch does not necessarily match it.
TORCH_SPEC = os.environ.get("BLS_TORCH_SPEC", "torch==2.3.1+cu118")
TORCH_INDEX_URL = os.environ.get(
    "BLS_TORCH_INDEX_URL", "https://download.pytorch.org/whl/torch_stable.html"
)

_SIMPLE_MODELS = [
    "bls",
    "dlpack_add_sub",
    "bls_async",
    "bls_memory",
    "bls_memory_async",
    "add_sub",
    "execute_error",
    "identity_fp32",
    "dlpack_identity",
    "dlpack_square",
    "identity_fp32_timeout",
]


def _repo_version():
    """Mirrors DATADIR construction in qa/L0_backend_python/test.sh: plain
    REPO_VERSION (== NVIDIA_TRITON_SERVER_VERSION), no arch suffix."""
    version = os.environ.get("REPO_VERSION") or os.environ.get(
        "NVIDIA_TRITON_SERVER_VERSION"
    )
    if not version:
        pytest.fail(
            "REPO_VERSION or NVIDIA_TRITON_SERVER_VERSION must be set "
            "(selects the QA model repository under DATADIR)",
            pytrace=False,
        )
    return version


def _copy_model(name, dst_root=MODELS_DIR):
    src = os.path.join(PYTHON_MODELS_DIR, name)
    dst = os.path.join(dst_root, name)
    os.makedirs(os.path.join(dst, "1"))
    shutil.copy(os.path.join(src, "config.pbtxt"), dst)
    shutil.copy(os.path.join(src, "model.py"), os.path.join(dst, "1"))


@pytest.fixture(scope="session")
def pinned_torch():
    """Match qa/L0_backend_python/bls/test.sh's torch pin for this scenario."""
    if os.environ.get("BLS_SKIP_TORCH_PIN") == "1":
        yield
        return
    action_log.run(
        ["pip3", "uninstall", "-y", "torch"],
        "Uninstalling whatever torch the QA image shipped, before pinning",
        check=False,
        capture_output=True,
    )
    action_log.run(
        ["pip3", "install", *TORCH_SPEC.split(), "-f", TORCH_INDEX_URL],
        "Installing pinned torch (%s) for BLS's dlpack CPU/GPU round-trip" % TORCH_SPEC,
        check=True,
    )
    yield


@pytest.fixture(scope="session")
def model_repository(pinned_torch):
    """Build the model repository the whole bls matrix serves."""
    datadir = os.environ.get("DATADIR") or os.path.join(
        "/data/inferenceserver", _repo_version()
    )

    if os.path.isdir(MODELS_DIR):
        shutil.rmtree(MODELS_DIR)
    os.makedirs(MODELS_DIR)

    for name in _SIMPLE_MODELS:
        _copy_model(name)

    shutil.copytree(
        os.path.join(
            datadir,
            "qa_sequence_implicit_model_repository",
            "onnx_nobatch_sequence_int32",
        ),
        os.path.join(MODELS_DIR, "onnx_nobatch_sequence_int32"),
    )

    clone_dir = os.path.join(OUTPUT_DIR, "python_backend")
    if os.path.isdir(clone_dir):
        shutil.rmtree(clone_dir)
    action_log.run(
        [
            "git",
            "clone",
            "%s/python_backend" % TRITON_REPO_ORGANIZATION,
            "-b",
            PYTHON_BACKEND_REPO_TAG,
            clone_dir,
        ],
        "Cloning python_backend (%s) for the square_int32 example model"
        % PYTHON_BACKEND_REPO_TAG,
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

    libtorch_src = os.path.join(
        datadir, "qa_model_repository", "libtorch_nobatch_float32_float32_float32"
    )
    for variant, kind in (("libtorch_gpu", "KIND_GPU"), ("libtorch_cpu", "KIND_CPU")):
        dst = os.path.join(MODELS_DIR, variant)
        shutil.copytree(libtorch_src, dst)
        config_path = os.path.join(dst, "config.pbtxt")
        with open(config_path) as f:
            config = f.read()
        config = config.replace("libtorch_nobatch_float32_float32_float32", variant)
        with open(config_path, "w") as f:
            f.write(config)
        with open(config_path, "a") as f:
            f.write("\ninstance_group [ { kind: %s} ]\n" % kind)

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
def serve(model_repository, server_log, bls_kind, extra_args=()):
    """Run tritonserver with BLS_KIND set for the model.py it will load."""
    cmd = [
        SERVER,
        "--model-repository=%s" % model_repository,
        "--backend-directory=%s" % BACKEND_DIR,
        "--http-port=%d" % HTTP_PORT,
        "--grpc-port=%d" % GRPC_PORT,
        "--log-verbose=1",
        *extra_args,
    ]
    env = dict(os.environ, BLS_KIND=bls_kind)
    with open(server_log, "wb") as log_fh:
        proc = action_log.popen(
            cmd,
            "Starting tritonserver for bls (BLS_KIND=%s, see %s)"
            % (bls_kind, server_log),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
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
