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

# Regression test for TRI-1698: a shared-memory inference request sent to a
# frontend started through the in-process Python bindings (tritonfrontend) --
# where the SharedMemoryManager is null -- must NOT crash the server process.
# The expected behavior is a graceful HTTP 503 (TRITONSERVER_ERROR_UNAVAILABLE).
#
# The server is deliberately run in a *subprocess* so that, on an unpatched
# build, the SIGSEGV kills only the child (which this test then detects as a
# clean failure) instead of taking down the pytest process itself.

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_REPO = os.path.join(HERE, "test_model_repository")
MODEL = "identity"

# Launched with:  python3 -c <SERVER_SNIPPET> <model_repo> <http_port>
# Imports only KServeHttp (never KServeGrpc) to avoid the known model-load
# interaction documented on the other tests in this directory.
SERVER_SNIPPET = """
import sys, time
import tritonserver
from tritonfrontend import KServeHttp

model_repo, port = sys.argv[1], int(sys.argv[2])
options = tritonserver.Options(
    server_id="TRI1698-regression",
    model_repository=model_repo,
    log_error=True, log_warn=True, log_info=False,
)
server = tritonserver.Server(options).start(wait_until_ready=True)
service = KServeHttp(server=server, options=KServeHttp.Options(port=port))
service.start()
sys.stderr.write("SERVER_READY\\n")
sys.stderr.flush()
while True:
    time.sleep(3600)
"""

NORMAL_REQUEST = {
    "inputs": [
        {"name": "INPUT0", "shape": [1], "datatype": "BYTES", "data": ["testing"]}
    ]
}

# Same input, but referencing a shared-memory region that was never registered.
# On a vulnerable build this dereferences a null SharedMemoryManager -> SIGSEGV.
SHM_REQUEST = {
    "inputs": [
        {
            "name": "INPUT0",
            "shape": [1],
            "datatype": "BYTES",
            "parameters": {
                "shared_memory_region": "fake_region",
                "shared_memory_byte_size": 4,
                "shared_memory_offset": 0,
            },
        }
    ]
}


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _post(base, path, payload, timeout=10):
    """POST JSON; return (status_code, body). Raises on connection errors."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        # Non-2xx responses (e.g. the expected 503) arrive here.
        return e.code, e.read().decode()


def _get(base, path, timeout=10):
    """GET; return (status_code, body). Raises urllib.error.URLError on reset."""
    req = urllib.request.Request(base + path, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def _is_ready(base, timeout=2):
    try:
        req = urllib.request.Request(base + "/v2/health/ready", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, ConnectionError):
        return False


def _wait_ready(proc, base, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                "server subprocess exited during startup with code %s" % proc.returncode
            )
        if _is_ready(base):
            return
        time.sleep(0.5)
    raise RuntimeError("server did not become ready within %ss" % timeout)


@pytest.fixture
def http_server():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-c", SERVER_SNIPPET, MODEL_REPO, str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    base = f"http://localhost:{port}"
    try:
        _wait_ready(proc, base)
        yield proc, base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def test_shm_inference_request_does_not_crash_python_frontend(http_server):
    """TRI-1698: an unauthenticated shared-memory inference request must return
    a graceful error, not crash the tritonfrontend-launched server."""
    proc, base = http_server

    # Sanity: the server is healthy and serves a normal inference.
    status, body = _post(base, f"/v2/models/{MODEL}/infer", NORMAL_REQUEST)
    assert status == 200, f"baseline inference failed: {status} {body}"

    # Attack: reference shared memory. Must not crash the process.
    try:
        status, body = _post(base, f"/v2/models/{MODEL}/infer", SHM_REQUEST)
    except (urllib.error.URLError, ConnectionError) as e:
        proc.poll()
        pytest.fail(
            "shared-memory inference reset the connection -- the server "
            "process most likely crashed (TRI-1698 regression). "
            f"error={e!r}, subprocess_returncode={proc.returncode}"
        )

    # The server must have answered gracefully with UNAVAILABLE (HTTP 503) ...
    assert status == 503, (
        f"expected HTTP 503 (UNAVAILABLE) for shared-memory request on a "
        f"frontend with no shared-memory manager, got {status}: {body}"
    )
    assert "shared memory is not supported" in body.lower(), body

    # ... and must still be alive and healthy afterwards.
    assert proc.poll() is None, (
        "server process died after a shared-memory request "
        f"(TRI-1698 regression); returncode={proc.returncode}"
    )
    assert _is_ready(base), "server is no longer ready after the attack request"


def test_shm_control_endpoint_does_not_crash_python_frontend(http_server):
    """TRI-1698 (hardening): the shared-memory status control endpoint must also
    fail gracefully rather than dereferencing a null manager. A GET reaches the
    shm_manager_ dereference (a POST would be rejected by the method check
    first), so GET is what actually exercises the vulnerable path."""
    proc, base = http_server

    try:
        status, body = _get(base, "/v2/systemsharedmemory/status")
    except (urllib.error.URLError, ConnectionError) as e:
        proc.poll()
        pytest.fail(
            "shared-memory status request reset the connection -- the server "
            "process most likely crashed (TRI-1698 regression). "
            f"error={e!r}, subprocess_returncode={proc.returncode}"
        )

    assert status == 503, (
        f"expected HTTP 503 (UNAVAILABLE) for a shared-memory control request "
        f"on a frontend with no shared-memory manager, got {status}: {body}"
    )

    # The server must still be alive and healthy afterwards.
    assert proc.poll() is None, (
        "server process died after a shared-memory control request "
        f"(TRI-1698 regression); returncode={proc.returncode}"
    )
    assert _is_ready(base), "server is no longer ready after the control request"
