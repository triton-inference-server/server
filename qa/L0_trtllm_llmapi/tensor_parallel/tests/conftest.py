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


"""Tensor parallelism. Each case prepares and serves its own repository, and
only when the machine has the GPUs for it (see test_tensor_parallel.py), so
there is no session server here.
"""

import contextlib
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIO_DIR = os.path.dirname(HERE)
SUITE_DIR = os.path.dirname(SCENARIO_DIR)
# Set by pytest.py (--output-dir); same default when pytest is run directly.
OUTPUT_DIR = os.environ.get("LLMAPI_OUTPUT_DIR") or os.path.join(
    SUITE_DIR, "output", os.path.basename(SCENARIO_DIR)
)

# llmapi_test_util (the tests' client helpers) and qa/common/action_log.
sys.path.append(SUITE_DIR)
sys.path.append(os.path.join(os.path.dirname(SUITE_DIR), "common"))
import action_log  # noqa: E402

TRITON_DIR = os.environ.get("TRITON_DIR", "/opt/tritonserver")
TENSORRTLLM_DIR = os.environ.get(
    "TENSORRTLLM_DIR", os.path.join(TRITON_DIR, "tensorrtllm")
)
MODEL_NAME = os.environ.get("LLMAPI_MODEL_NAME", "tensorrt_llm")
HF_MODEL = os.environ.get("LLMAPI_HF_MODEL", "Qwen/Qwen3-0.6B")
# The LLM API default KV cache claims most of the GPU.
KV_CACHE_FRACTION = os.environ.get("LLMAPI_KV_CACHE_FRACTION", "0.2")
# Covers a first-time model download as well as the load.
SERVER_TIMEOUT_S = int(os.environ.get("LLMAPI_SERVER_TIMEOUT", "900"))
HTTP_URL = "http://%s:%s" % (
    os.environ.get("TRITON_HOST", "localhost"),
    os.environ.get("TRITON_HTTP_PORT", "8000"),
)
LAUNCHER = os.path.join(TENSORRTLLM_DIR, "scripts", "launch_triton_server.py")


def prepare_repository(repo, tp):
    """Create `repo` with the LLM API model at tensor_parallel_size `tp`."""
    if os.path.isdir(repo):
        shutil.rmtree(repo)
    os.makedirs(repo)
    model_dir = os.path.join(repo, MODEL_NAME)
    shutil.copytree(
        os.path.join(TENSORRTLLM_DIR, "all_models", "llmapi", "tensorrt_llm"),
        model_dir,
    )

    yaml_path = os.path.join(model_dir, "1", "model.yaml")
    with open(yaml_path) as f:
        model_yaml = f.read()
    model_yaml = re.sub(r"(?m)^model: .*$", "model: %s" % HF_MODEL, model_yaml)
    model_yaml = re.sub(
        r"(?m)^tensor_parallel_size: .*$", "tensor_parallel_size: %d" % tp, model_yaml
    )
    model_yaml += "kv_cache_config:\n  free_gpu_memory_fraction: %s\n" % (
        KV_CACHE_FRACTION
    )
    with open(yaml_path, "w") as f:
        f.write(model_yaml)
    return repo


class ServerStartError(Exception):
    pass


def _server_pids():
    """PIDs of live tritonserver processes; zombies are excluded."""
    result = subprocess.run(
        ["pgrep", "-x", "-r", "R,S,D,I", "tritonserver"],
        capture_output=True,
        text=True,
    )
    return [int(pid) for pid in result.stdout.split()]


def _model_ready(model):
    url = "%s/v2/models/%s/ready" % (HTTP_URL, model)
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def _wait_for_ready(models):
    """Wait until every model in `models` is ready; raise if the server dies.

    The launcher detaches the server, so liveness is checked by process name.
    """
    start = time.time()
    while time.time() - start < SERVER_TIMEOUT_S:
        if all(_model_ready(model) for model in models):
            return
        # Allow a few seconds for the detached server process to appear.
        if time.time() - start >= 10 and not _server_pids():
            raise ServerStartError(
                "tritonserver exited before %s were ready" % ", ".join(models)
            )
        time.sleep(2)
    raise ServerStartError(
        "timed out after %ds waiting for %s" % (SERVER_TIMEOUT_S, ", ".join(models))
    )


def _stop():
    """Stop the server and every worker it started, gracefully first."""
    for pid in _server_pids():
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGINT)
    deadline = time.time() + 60
    while _server_pids() and time.time() < deadline:
        time.sleep(2)
    for match in (
        ["-x", "tritonserver"],
        ["-f", "triton_python_backend_stub"],
        ["-f", "mpi4py.futures"],
    ):
        subprocess.run(["pkill", "-9"] + match, capture_output=True)


def _tail(path, lines=30):
    if not os.path.exists(path):
        return ""
    with open(path, errors="replace") as f:
        return "".join(f.readlines()[-lines:])


def _start(repo, server_log, tp):
    """Start the server for `repo` with the image's launcher, as users do."""
    os.makedirs(os.path.dirname(server_log), exist_ok=True)
    # llmapi_test_util reads the log of the running server from SERVER_LOG.
    os.environ["SERVER_LOG"] = server_log
    launcher_log = server_log.replace(".log", "_launcher.log")
    cmd = (
        [sys.executable, LAUNCHER]
        + [
            "--world_size",
            str(tp),
            "--trtllm_llmapi_launch",
            "--oversubscribe",
        ]
        + ["--model_repo", repo, "--log", "--log-file", server_log]
    )
    with open(launcher_log, "wb") as log_fh:
        returncode = action_log.call(
            cmd,
            "Launching tritonserver for %s (see %s)" % (repo, server_log),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
    if returncode != 0:
        raise ServerStartError(
            "launcher exited with %d:\n%s" % (returncode, _tail(launcher_log))
        )
    try:
        _wait_for_ready([MODEL_NAME])
    except ServerStartError as e:
        _stop()
        raise ServerStartError("%s; server log tail:\n%s" % (e, _tail(server_log)))


@contextlib.contextmanager
def serve(repo, server_log, tp):
    """Serve `repo` for the duration of the block."""
    try:
        _start(repo, server_log, tp)
        yield
    finally:
        _stop()
