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
"""Fixtures for the `lifecycle` scenario.

Ported from qa/L0_backend_python/lifecycle/test.sh: one main model repo
(error_code, execute_cancel, execute_error, execute_grpc_error,
decoupled_grpc_error, execute_return_error, wrong_model) exercised 5x in a
row to catch intermittent segfaults, plus three separate single-model
repos (init_error, fini_error, auto_complete_error) whose model.py raises
during initialize()/finalize()/auto_complete_config() on purpose -- the
test asserts the server surfaces the right error and leaks no shared
memory pages, not that it starts cleanly.

Self-contained: everything needed to run this scenario lives under
qa/L0_backend_python_pytest/lifecycle/, except qa/python_models/* and the
tritonserver binary itself, which are shared, read-only assets already
present in the QA image.
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
GRPC_PORT = int(os.environ.get("TRITONSERVER_GRPC_PORT", "8001"))
METRICS_PORT = int(os.environ.get("TRITONSERVER_METRICS_PORT", "8002"))

STARTUP_TIMEOUT_S = 120
EXIT_TIMEOUT_S = 30

# Generated model repos + server logs live here, kept out of tests/ and off
# of source, so cleanup is one `rm -rf output`.
OUTPUT_DIR = os.path.join(SCENARIO_DIR, "output")

MAIN_MODELS = [
    "error_code",
    "execute_cancel",
    "execute_error",
    "execute_grpc_error",
    "decoupled_grpc_error",
    "execute_return_error",
    "wrong_model",
]


def _copy_model(name, dst_root):
    src = os.path.join(PYTHON_MODELS_DIR, name)
    dst = os.path.join(dst_root, name)
    os.makedirs(os.path.join(dst, "1"))
    shutil.copy(os.path.join(src, "config.pbtxt"), dst)
    shutil.copy(os.path.join(src, "model.py"), os.path.join(dst, "1"))
    return dst


def _rewrite_config(config_path, replacements, append=None):
    with open(config_path) as f:
        text = f.read()
    for old, new in replacements:
        text = old.sub(new, text) if hasattr(old, "sub") else text.replace(old, new)
    with open(config_path, "w") as f:
        f.write(text)
        if append:
            f.write("\n" + append + "\n")


@pytest.fixture(scope="session")
def main_model_repository():
    """error_code, execute_cancel, execute_error, execute_grpc_error,
    decoupled_grpc_error, execute_return_error, wrong_model -- the models
    the 5x-repeated main test phase serves."""
    import re

    models_dir = os.path.join(OUTPUT_DIR, "main_models")
    if os.path.isdir(models_dir):
        shutil.rmtree(models_dir)
    os.makedirs(models_dir)

    for name in MAIN_MODELS:
        _copy_model(name, models_dir)

    execute_error_cfg = os.path.join(models_dir, "execute_error", "config.pbtxt")
    _rewrite_config(
        execute_error_cfg,
        [
            (re.compile(r"^name:.*$", re.M), 'name: "execute_error"'),
            (re.compile(r"^max_batch_size:.*$", re.M), "max_batch_size: 8"),
        ],
        append="dynamic_batching { preferred_batch_size: [8], "
        "max_queue_delay_microseconds: 12000000 }",
    )

    execute_grpc_error_cfg = os.path.join(
        models_dir, "execute_grpc_error", "config.pbtxt"
    )
    _rewrite_config(
        execute_grpc_error_cfg,
        [
            (re.compile(r"^name:.*$", re.M), 'name: "execute_grpc_error"'),
            (re.compile(r"^max_batch_size:.*$", re.M), "max_batch_size: 8"),
        ],
        append="dynamic_batching { preferred_batch_size: [8], "
        "max_queue_delay_microseconds: 1200000 }",
    )

    wrong_model_cfg = os.path.join(models_dir, "wrong_model", "config.pbtxt")
    _rewrite_config(
        wrong_model_cfg,
        [
            (re.compile(r"^name:.*$", re.M), 'name: "wrong_model"'),
            ("TYPE_FP32", "TYPE_UINT32"),
        ],
    )

    return models_dir


def _single_model_repository(name, subdir, with_config=True):
    models_dir = os.path.join(OUTPUT_DIR, subdir)
    if os.path.isdir(models_dir):
        shutil.rmtree(models_dir)
    dst = os.path.join(models_dir, name, "1")
    os.makedirs(dst)
    shutil.copy(os.path.join(PYTHON_MODELS_DIR, name, "model.py"), os.path.join(dst))
    if with_config:
        shutil.copy(
            os.path.join(PYTHON_MODELS_DIR, name, "config.pbtxt"),
            os.path.join(models_dir, name),
        )
    return models_dir


@pytest.fixture
def init_error_model_repository():
    return _single_model_repository("init_error", "init_error_models")


@pytest.fixture
def fini_error_model_repository():
    return _single_model_repository("fini_error", "fini_error_models")


@pytest.fixture
def auto_complete_error_model_repository():
    # No config.pbtxt on purpose: exercises auto-complete-config, which
    # requires --strict-model-config=false (see serve_expect_exit callers).
    return _single_model_repository(
        "auto_complete_error", "auto_complete_error_models", with_config=False
    )


def shm_page_count():
    return len(os.listdir("/dev/shm"))


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


def _cmd(model_repository, extra_args):
    return [
        SERVER,
        "--model-repository=%s" % model_repository,
        "--backend-directory=%s" % BACKEND_DIR,
        "--http-port=%d" % HTTP_PORT,
        "--grpc-port=%d" % GRPC_PORT,
        "--metrics-port=%d" % METRICS_PORT,
        "--log-verbose=1",
        *extra_args,
    ]


@contextlib.contextmanager
def serve(model_repository, server_log, extra_args=()):
    """Normal server lifecycle: wait for ready, yield, terminate on exit."""
    with open(server_log, "wb") as log_fh:
        proc = subprocess.Popen(
            _cmd(model_repository, extra_args), stdout=log_fh, stderr=subprocess.STDOUT
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


def run_and_wait_for_exit(model_repository, server_log, extra_args=()):
    """For init_error/auto_complete_error: the model's init failure is
    expected to bring the whole server process down on its own -- match
    qa/L0_backend_python/lifecycle/test.sh's `run_server_nowait` + `wait
    $SERVER_PID` (no readiness poll, no explicit kill)."""
    with open(server_log, "wb") as log_fh:
        proc = subprocess.Popen(
            _cmd(model_repository, extra_args), stdout=log_fh, stderr=subprocess.STDOUT
        )
    proc.wait(timeout=EXIT_TIMEOUT_S)
