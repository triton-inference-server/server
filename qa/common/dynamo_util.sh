#!/bin/bash
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

# Dynamo launch mode for the Triton QA suite. Sourced in place of util.sh when
# SERVER_LAUNCH_MODE=dynamo. Overrides run_server / kill_server to launch Triton
# through Dynamo (frontend on KServe gRPC + in-process Triton worker) instead of
# the standalone tritonserver binary.
source "$(dirname "${BASH_SOURCE[0]}")/util.sh"

SERVER_LAUNCH_MODE=${SERVER_LAUNCH_MODE:=dynamo}
DYN_FRONTEND_LOG=${DYN_FRONTEND_LOG:=./frontend.log}
DYN_FRONTEND_ARGS=${DYN_FRONTEND_ARGS:="--kserve-grpc-server"}
DYN_WORKER_PY=${DYN_WORKER_PY:=/workspace/components/src/dynamo/triton/tritonworker.py}
DYN_WORKER_ARGS=${DYN_WORKER_ARGS:=""}
DYN_DISCOVERY_BACKEND=${DYN_DISCOVERY_BACKEND:=file}
# The Dynamo frontend binds KServe gRPC to --http-port; serve it on 8001.
DYN_HTTP_PORT=${DYN_HTTP_PORT:=8001}
DYN_FRONTEND_PID=0
# etcd + NATS back the etcd discovery backend. The Dynamo container has no docker
# but ships the etcd and nats-server binaries, so run_server launches them
# directly; kill_server tears down only what it started.
DYN_ETCD_LOG=${DYN_ETCD_LOG:=./etcd.log}
DYN_NATS_LOG=${DYN_NATS_LOG:=./nats.log}
DYN_ETCD_DATA_DIR=${DYN_ETCD_DATA_DIR:=/tmp/dynamo_qa_etcd.data}
DYN_ETCD_PID=0
DYN_NATS_PID=0
# KServe ServerReady flips after the first model wires up, but the worker
# registers a whole repo as a burst. When DYN_WAIT_ALL_MODELS=1, run_server also
# waits (per-model is_model_ready) until every model is ready or readiness
# settles, so the registration burst drains before tests run.
DYN_WAIT_ALL_MODELS=${DYN_WAIT_ALL_MODELS:=1}
DYN_MODELS_SETTLE_SECS=${DYN_MODELS_SETTLE_SECS:=15}

# Wait until the KServe gRPC server (Dynamo frontend on DYN_HTTP_PORT) reports
# ready. Sets WAIT_RET to 0 on success, 1 on failure.
function wait_for_grpc_server_ready() {
    local spid="$1"; shift
    local wait_time_secs="${1:-30}"; shift

    WAIT_RET=0

    local wait_secs=$wait_time_secs
    until test $wait_secs -eq 0 ; do
        if ! kill -0 $spid > /dev/null 2>&1; then
            echo "=== Server not running."
            WAIT_RET=1
            return
        fi

        # If the frontend dies, the gRPC endpoint is gone; stop probing.
        if [ "${DYN_FRONTEND_PID:-0}" != "0" ] && \
           ! kill -0 $DYN_FRONTEND_PID > /dev/null 2>&1; then
            echo "=== Dynamo frontend not running (see $DYN_FRONTEND_LOG)."
            WAIT_RET=1
            return
        fi

        sleep 1;

        set +e
        python3 - "$SERVER_IPADDR" "$DYN_HTTP_PORT" <<'PY' > /dev/null 2>&1
import sys
import tritonclient.grpc as grpcclient

try:
    client = grpcclient.InferenceServerClient(url=f"{sys.argv[1]}:{sys.argv[2]}", verbose=False)
    sys.exit(0 if client.is_server_ready() else 1)
except Exception:
    sys.exit(1)
PY
        local ready=$?
        set -e
        if [ "$ready" == "0" ]; then
            return
        fi

        ((wait_secs--));
    done

    echo "=== Timeout $wait_time_secs secs. gRPC server not ready."
    WAIT_RET=1
}

# Convert the given standalone tritonserver args (e.g. $SERVER_ARGS) into Triton
# worker (tritonworker.py) flags on stdout, to auto-populate DYN_WORKER_ARGS.
# Most flags pass through unchanged; the exception is --allow-* endpoint flags
# (owned by the frontend) which are dropped. Accepts both --flag=value and
# --flag value forms.
# Args: <server_arg>...
function dynamo_worker_args_from_server_args () {
    local out=()
    local tokens=("$@")
    local n=${#tokens[@]}
    local i=0

    while [ $i -lt $n ]; do
        local tok="${tokens[$i]}"
        case "$tok" in
            --allow-*)
                # Endpoints are owned by the Dynamo frontend, not the worker;
                # drop the flag and its value (--allow-x=v or --allow-x v).
                if [[ "$tok" != *=* ]] && [ $((i + 1)) -lt $n ] \
                        && [[ "${tokens[$((i + 1))]}" != --* ]]; then
                    ((i++))
                fi ;;
            *)
                out+=("$tok") ;;
        esac
        ((i++))
    done

    echo "${out[*]}"
}

# Start etcd + NATS for the etcd discovery backend. No-op unless
# DYN_DISCOVERY_BACKEND is etcd. Each service is reused if already listening on
# its default port (etcd 2379, NATS 4222); only services started here record a
# PID, so kill_server tears down only those. Returns 0 once both accept
# connections (or were reused / not needed), non-zero on any failure.
function start_dynamo_discovery () {
    DYN_ETCD_PID=0
    DYN_NATS_PID=0

    if [ "$DYN_DISCOVERY_BACKEND" != "etcd" ]; then
        return 0
    fi

    if (exec 3<>/dev/tcp/localhost/2379) 2>/dev/null; then
        echo "=== Reusing etcd already listening on localhost:2379"
    else
        if ! command -v etcd > /dev/null 2>&1; then
            echo "=== etcd not found; cannot start it for DYN_DISCOVERY_BACKEND=etcd"
            return 1
        fi
        echo "=== Starting etcd (log: $DYN_ETCD_LOG)"
        rm -rf "$DYN_ETCD_DATA_DIR"
        etcd --data-dir "$DYN_ETCD_DATA_DIR" \
            --listen-client-urls http://0.0.0.0:2379 \
            --advertise-client-urls http://localhost:2379 \
            > "$DYN_ETCD_LOG" 2>&1 &
        DYN_ETCD_PID=$!
    fi

    if (exec 3<>/dev/tcp/localhost/4222) 2>/dev/null; then
        echo "=== Reusing NATS already listening on localhost:4222"
    else
        if ! command -v nats-server > /dev/null 2>&1; then
            echo "=== nats-server not found; cannot start it for DYN_DISCOVERY_BACKEND=etcd"
            stop_dynamo_discovery
            return 1
        fi
        echo "=== Starting NATS with JetStream (log: $DYN_NATS_LOG)"
        nats-server -js > "$DYN_NATS_LOG" 2>&1 &
        DYN_NATS_PID=$!
    fi

    # Wait for both ports, but fail fast if a service we launched dies.
    local wait_secs=30
    until test $wait_secs -eq 0 ; do
        if (exec 3<>/dev/tcp/localhost/2379) 2>/dev/null && \
           (exec 4<>/dev/tcp/localhost/4222) 2>/dev/null; then
            return 0
        fi
        if [ "$DYN_ETCD_PID" != "0" ] && ! kill -0 "$DYN_ETCD_PID" 2>/dev/null; then
            echo "=== etcd exited early (see $DYN_ETCD_LOG)"
            stop_dynamo_discovery
            return 1
        fi
        if [ "$DYN_NATS_PID" != "0" ] && ! kill -0 "$DYN_NATS_PID" 2>/dev/null; then
            echo "=== NATS exited early (see $DYN_NATS_LOG)"
            stop_dynamo_discovery
            return 1
        fi
        sleep 1; ((wait_secs--))
    done

    echo "=== Timeout waiting for etcd + NATS to accept connections"
    stop_dynamo_discovery
    return 1
}

# Stop the etcd + NATS services started by start_dynamo_discovery. Reused
# services have no recorded PID and are left untouched.
function stop_dynamo_discovery () {
    if [ "${DYN_NATS_PID:-0}" != "0" ]; then
        kill $DYN_NATS_PID > /dev/null 2>&1 || true
        wait $DYN_NATS_PID 2>/dev/null || true
        DYN_NATS_PID=0
    fi
    if [ "${DYN_ETCD_PID:-0}" != "0" ]; then
        kill $DYN_ETCD_PID > /dev/null 2>&1 || true
        wait $DYN_ETCD_PID 2>/dev/null || true
        DYN_ETCD_PID=0
    fi
}

# Wait until every model in the worker's repository reports ready on the
# frontend, not just the first. Best-effort: returns once all models are ready,
# readiness settles (ready count unchanged for $settle_secs), or the wait times
# out, since the server is already up.
# Args: <wait_time_secs> <settle_secs> <model_repository>...
function wait_for_dynamo_models_ready() {
    local wait_time_secs="${1:-300}"; shift
    local settle_secs="${1:-15}"; shift

    if [ "$#" -eq 0 ]; then
        echo "=== wait_for_dynamo_models_ready: no model repository given; skipping"
        return
    fi

    python3 - "$SERVER_IPADDR" "$DYN_HTTP_PORT" "$wait_time_secs" "$settle_secs" "$@" <<'PY'
import os
import sys
import time

import tritonclient.grpc as grpcclient

host, port = sys.argv[1], sys.argv[2]
wait_secs, settle_secs = float(sys.argv[3]), float(sys.argv[4])
repos = sys.argv[5:]


def model_name(repo, d):
    # Triton model name defaults to the directory name; honor a `name:` override
    # in config.pbtxt when present.
    try:
        with open(os.path.join(repo, d, "config.pbtxt")) as f:
            for line in f:
                line = line.strip()
                if line.startswith("name:"):
                    return line.split(":", 1)[1].strip().strip('"')
    except OSError:
        pass
    return d


expected = set()
for repo in repos:
    try:
        for d in sorted(os.listdir(repo)):
            if os.path.isdir(os.path.join(repo, d)):
                expected.add(model_name(repo, d))
    except OSError:
        pass
expected = sorted(expected)
if not expected:
    sys.exit(0)

client = grpcclient.InferenceServerClient(url=f"{host}:{port}", verbose=False)
deadline = time.time() + wait_secs
last_count = -1
stable_since = time.time()
while time.time() < deadline:
    ready = 0
    missing = []
    for name in expected:
        try:
            if client.is_model_ready(name):
                ready += 1
                continue
        except Exception:
            pass
        missing.append(name)
    if ready == len(expected):
        print(f"=== All {ready} models ready")
        sys.exit(0)
    if ready != last_count:
        last_count = ready
        stable_since = time.time()
    elif time.time() - stable_since >= settle_secs:
        print(
            f"=== Model readiness settled at {ready}/{len(expected)}; proceeding "
            f"(missing e.g. {missing[:5]})"
        )
        sys.exit(0)
    time.sleep(2)
print(f"=== Timeout waiting for all models: {last_count}/{len(expected)} ready")
sys.exit(0)
PY
}

# Launch Triton through Dynamo: the frontend (KServe gRPC on DYN_HTTP_PORT) and
# the in-process Triton worker, both wired to DYN_DISCOVERY_BACKEND. Sets
# SERVER_PID to the worker pid (so the existing kill/debug paths apply) and
# DYN_FRONTEND_PID to the frontend pid. Returns once the gRPC server is ready or
# the timeout expires (SERVER_PID reset to 0 on failure).
function run_server () {
    SERVER_PID=0
    DYN_FRONTEND_PID=0

    # Auto-infer worker args from the standard SERVER_ARGS when not provided.
    if [ -z "$DYN_WORKER_ARGS" ]; then
        DYN_WORKER_ARGS="$(dynamo_worker_args_from_server_args $SERVER_ARGS)"
        echo "=== Inferred DYN_WORKER_ARGS from SERVER_ARGS: $DYN_WORKER_ARGS"
    fi

    if [ -z "$DYN_WORKER_ARGS" ]; then
        echo "=== DYN_WORKER_ARGS is empty (SERVER_ARGS must include --model-repository <path>)"
        return
    fi

    if [ ! -f "$DYN_WORKER_PY" ]; then
        echo "=== $DYN_WORKER_PY does not exist"
        return
    fi

    # start_dynamo_discovery cleans up after itself on failure.
    if ! start_dynamo_discovery; then
        echo "=== Failed to bring up the etcd + NATS discovery services"
        return
    fi

    echo "=== Running dynamo.frontend $DYN_FRONTEND_ARGS --http-port $DYN_HTTP_PORT --discovery-backend $DYN_DISCOVERY_BACKEND"
    python3 -m dynamo.frontend $DYN_FRONTEND_ARGS --http-port $DYN_HTTP_PORT --discovery-backend $DYN_DISCOVERY_BACKEND \
        > $DYN_FRONTEND_LOG 2>&1 &
    DYN_FRONTEND_PID=$!

    echo "=== Running $DYN_WORKER_PY $DYN_WORKER_ARGS --discovery-backend $DYN_DISCOVERY_BACKEND"
    python3 $DYN_WORKER_PY $DYN_WORKER_ARGS --discovery-backend $DYN_DISCOVERY_BACKEND \
        > $SERVER_LOG 2>&1 &
    SERVER_PID=$!

    wait_for_grpc_server_ready $SERVER_PID $SERVER_TIMEOUT
    if [ "$WAIT_RET" != "0" ]; then
        # Get further debug information about server startup failure.
        gdb_helper || true

        kill_server
        SERVER_PID=0
        return
    fi

    # server_ready only guarantees one model; wait for the whole repo to wire up.
    if [ "$DYN_WAIT_ALL_MODELS" != "0" ]; then
        local repos
        repos=$(echo "$DYN_WORKER_ARGS" | grep -oE -- '--model-repository[ =][^ ]+' | sed -E 's/--model-repository[ =]//')
        wait_for_dynamo_models_ready $SERVER_TIMEOUT $DYN_MODELS_SETTLE_SECS $repos
    fi
}

# Tear down the Triton worker, the Dynamo frontend, and the etcd + NATS
# discovery services. Dynamo mode runs on Linux only, so the WSL/MSYS teardown
# branches do not apply here.
function kill_server () {
    # Signal the worker and frontend first, then reap them, so they shut down in
    # parallel instead of serially.
    kill $SERVER_PID > /dev/null 2>&1 || true
    if [ "${DYN_FRONTEND_PID:-0}" != "0" ]; then
        kill $DYN_FRONTEND_PID > /dev/null 2>&1 || true
    fi

    # Tolerate a non-zero wait status (e.g. the SIGTERM exit code) so teardown
    # still runs under a caller's `set -e`.
    wait $SERVER_PID 2>/dev/null || true
    if [ "${DYN_FRONTEND_PID:-0}" != "0" ]; then
        wait $DYN_FRONTEND_PID 2>/dev/null || true
        DYN_FRONTEND_PID=0
    fi

    stop_dynamo_discovery
}
