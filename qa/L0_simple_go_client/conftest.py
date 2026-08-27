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

"""Fixtures for the Go gRPC client example test.

Ported from the previous test.sh. The environment variables, repository
tags, clone commands and server lifecycle are carried over unchanged so
that pass/fail semantics match the shell version.
"""

import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request

import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def _env(name, default):
    """Read name, falling back to default when unset *or empty*.

    The CI template passes these through unconditionally, e.g.
    -e TRITON_COMMON_REPO_TAG="${TRITON_COMMON_REPO_TAG}", so an unset
    pipeline variable arrives as an empty string rather than as an absent
    one. os.environ.get's default would not apply, and 'git clone -b ""'
    fails. The shell version used ${VAR:="main"}, which substitutes on
    unset or empty; this keeps that behaviour.
    """
    return os.environ.get(name) or default


# Matches the defaults in the shell version of this test.
REPO_ORGANIZATION = _env(
    "TRITON_REPO_ORGANIZATION", "http://github.com/triton-inference-server"
)
COMMON_REPO_TAG = _env("TRITON_COMMON_REPO_TAG", "main")
SERVER_BRANCH_NAME = _env("TRITON_SERVER_BRANCH_NAME", "main")
# Exported by the CI template alongside the tags above, but the client clone
# below does not pass -b yet, so this is currently read and not used. Kept so
# that wiring it up is a one-line change rather than a rediscovery.
CLIENT_REPO_TAG = _env("TRITON_CLIENT_REPO_TAG", "main")

GO_CLIENT_DIR = os.path.join(TEST_DIR, "client", "src", "grpc_generated", "go")
STUB_PACKAGE_DIR = os.path.join(GO_CLIENT_DIR, "grpc-client")

SERVER_CLONE_DIR = os.path.join(TEST_DIR, "server")
MODEL_REPOSITORY = os.path.join(TEST_DIR, "model_repository")
EXAMPLE_MODEL = "simple"
EXAMPLE_MODEL_SRC = os.path.join(
    SERVER_CLONE_DIR, "docs", "examples", "model_repository", EXAMPLE_MODEL
)

SERVER = "/opt/tritonserver/bin/tritonserver"
SERVER_LOG = os.path.join(TEST_DIR, "inference_server.log")
CLIENT_LOG = os.path.join(GO_CLIENT_DIR, "client.log")

# util.sh: SERVER_IPADDR=${TRITONSERVER_IPADDR:=localhost}, SERVER_TIMEOUT=120.
SERVER_IPADDR = _env("TRITONSERVER_IPADDR", "localhost")
SERVER_TIMEOUT = int(_env("SERVER_TIMEOUT", "120"))

READY_URL = f"http://{SERVER_IPADDR}:8000/v2/health/ready"


def _run(cmd, cwd=None, log=None):
    """Run cmd, echoing the command and its exit status.

    Output is not captured, so each sub-action streams into the job log the
    way it did under `bash -ex ./test.sh`. pytest.ini disables capturing so
    this stays visible on a passing run too, not only on failure.
    """
    print(
        f"=== Running {' '.join(cmd)}" + (f"  (cwd {cwd})" if cwd else ""), flush=True
    )
    if log is None:
        result = subprocess.run(cmd, cwd=cwd, check=False)
    else:
        with open(log, "ab") as handle:
            result = subprocess.run(
                cmd, cwd=cwd, check=False, stdout=handle, stderr=subprocess.STDOUT
            )
    print(f"=== Exit {result.returncode}: {cmd[0]}", flush=True)
    return result


def _gdb_backtrace(pid):
    """Dump a backtrace of a server that started but never became ready.

    Ports the "Server Hang" half of gdb_helper() in qa/common/util.sh, which
    run_server called on a readiness timeout. The core-dump half is omitted:
    the CI template runs the container with --ulimit core=0, so there are no
    core files to load. The log is written beside the test so the CI log
    collector picks it up.
    """
    if shutil.which("gdb") is None:
        print("=== WARNING: gdb not installed", flush=True)
        return
    gdb_log = os.path.join(TEST_DIR, f"gdb_bt.{pid}.log")
    print(
        f"=== WARNING: SERVER HANG DETECTED, DUMPING GDB BACKTRACE TO [{gdb_log}] ===",
        flush=True,
    )
    with open(gdb_log, "wb") as handle:
        subprocess.run(
            ["gdb", "-batch", "-ex", "thread apply all bt", "-p", str(pid)],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    with open(gdb_log, errors="replace") as handle:
        print(handle.read(), flush=True)


def _run_checked(cmd, cwd=None, message=""):
    """Run cmd and fail the fixture if it returns non-zero."""
    result = _run(cmd, cwd=cwd)
    assert result.returncode == 0, f"{message} (exit {result.returncode})"
    return result


def _wait_for_server_ready(proc, timeout_secs):
    """Poll /v2/health/ready, aborting early if the server process died.

    Mirrors wait_for_server_ready() in qa/common/util.sh.
    """
    for _ in range(timeout_secs):
        if proc.poll() is not None:
            return False
        # Sleep before polling, as util.sh does. Without this the loop is not
        # rate limited: while nothing is listening yet urlopen fails on
        # connection refused immediately, so all timeout_secs iterations are
        # consumed in a fraction of a second and the wait effectively becomes
        # zero rather than timeout_secs.
        time.sleep(1)
        try:
            with urllib.request.urlopen(READY_URL, timeout=1) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
    return False


@pytest.fixture(scope="session")
def model_repository():
    """Build a model repository holding the 'simple' example model.

    The model is taken from the server repo rather than from a models/
    directory staged by Dockerfile.QA, so the test provisions everything it
    needs and runs identically in CI and standalone.
    """
    shutil.rmtree(SERVER_CLONE_DIR, ignore_errors=True)
    _run_checked(
        [
            "git",
            "clone",
            "--single-branch",
            "--depth=1",
            "-b",
            SERVER_BRANCH_NAME,
            f"{REPO_ORGANIZATION}/server.git",
        ],
        cwd=TEST_DIR,
        message="failed to clone server repo",
    )
    assert os.path.isdir(
        EXAMPLE_MODEL_SRC
    ), f"missing example model: {EXAMPLE_MODEL_SRC}"

    shutil.rmtree(MODEL_REPOSITORY, ignore_errors=True)
    os.makedirs(MODEL_REPOSITORY)
    shutil.copytree(EXAMPLE_MODEL_SRC, os.path.join(MODEL_REPOSITORY, EXAMPLE_MODEL))
    return MODEL_REPOSITORY


@pytest.fixture(scope="session")
def triton_server(model_repository):
    """Start tritonserver against the provisioned model repository."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    assert os.path.isfile(SERVER), f"{SERVER} does not exist"

    args = [SERVER, f"--model-repository={model_repository}"]
    print(f"=== Running {' '.join(args)}", flush=True)
    with open(SERVER_LOG, "wb") as log:
        proc = subprocess.Popen(
            args, stdout=log, stderr=subprocess.STDOUT, cwd=TEST_DIR
        )

    if not _wait_for_server_ready(proc, SERVER_TIMEOUT):
        if proc.poll() is None:
            # Still running but never became ready: capture why before killing.
            _gdb_backtrace(proc.pid)
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=60)
        with open(SERVER_LOG) as log:
            print(log.read(), flush=True)
        pytest.fail(f"Failed to start {SERVER}")

    yield proc

    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@pytest.fixture(scope="session")
def go_stubs():
    """Clone the client and common repos and generate the Go stubs.

    The clone commands are carried over verbatim from test.sh, including
    the fact that client.git is cloned without a branch flag and so always
    resolves to the remote default branch.
    """
    for stale in ("client", "common"):
        shutil.rmtree(os.path.join(TEST_DIR, stale), ignore_errors=True)

    _run_checked(
        ["git", "clone", f"{REPO_ORGANIZATION}/client.git"],
        cwd=TEST_DIR,
        message="failed to clone client repo",
    )

    _run_checked(
        ["go", "install", "google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest"],
        cwd=TEST_DIR,
        message="failed to install protoc-gen-go-grpc",
    )

    _run_checked(
        [
            "git",
            "clone",
            "--single-branch",
            "--depth=1",
            "-b",
            COMMON_REPO_TAG,
            f"{REPO_ORGANIZATION}/common.git",
        ],
        cwd=GO_CLIENT_DIR,
        message="failed to clone common repo",
    )

    _run_checked(
        ["bash", "gen_go_stubs.sh"],
        cwd=GO_CLIENT_DIR,
        message="gen_go_stubs.sh failed",
    )

    return STUB_PACKAGE_DIR


@pytest.fixture(scope="session")
def client_log():
    """Path to the Go client's captured output."""
    return CLIENT_LOG


@pytest.fixture(scope="session")
def go_client_run(triton_server, go_stubs):
    """Run the Go example client once and return its CompletedProcess.

    Output goes to client.log so the existing log-collection glob in CI
    still picks it up.
    """
    if os.path.exists(CLIENT_LOG):
        os.unlink(CLIENT_LOG)
    result = _run(
        ["go", "run", "grpc_simple_client.go"], cwd=GO_CLIENT_DIR, log=CLIENT_LOG
    )
    # The client's own output only reaches client.log, which sits too deep for
    # the CI log collector to pick up. Echo it so the inference results are
    # visible in the job log whether the run passed or failed.
    if os.path.exists(CLIENT_LOG):
        with open(CLIENT_LOG, errors="replace") as log:
            print(f"=== {CLIENT_LOG}\n{log.read()}", flush=True)
    return result
