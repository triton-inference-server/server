#!/bin/bash
# Copyright 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Helpers to serve a TensorRT-LLM model through the LLM API (Python) model
# that build.py installs with '--backend tensorrtllm': the model code is the
# Python-based backend backends/tensorrtllm, and a model repository only holds
# config.pbtxt and model.yaml. It is launched with launch_triton_server.py
# (without MPI), so nothing here builds a TensorRT engine.
#
# Environment:
#   TENSORRTLLM_DIR      directory build.py installed the sample model
#                        repository and the launcher to
#   LLMAPI_HF_MODEL      Hugging Face model the LLM API loads
#   LLMAPI_KV_CACHE_FRACTION
#                        free_gpu_memory_fraction for the KV cache. The LLM API
#                        default claims most of the GPU, which leaves no room
#                        for a second model instance or another test server.

TENSORRTLLM_DIR=${TENSORRTLLM_DIR:="/opt/tritonserver/tensorrtllm"}
LLMAPI_MODEL_NAME=${LLMAPI_MODEL_NAME:="tensorrt_llm"}
LLMAPI_HF_MODEL=${LLMAPI_HF_MODEL:="Qwen/Qwen3-0.6B"}
LLMAPI_KV_CACHE_FRACTION=${LLMAPI_KV_CACHE_FRACTION:="0.2"}
LLMAPI_LAUNCHER=${TENSORRTLLM_DIR}/scripts/launch_triton_server.py
# Launcher arguments. Multi-instance and TP=1 use a single process without MPI;
# tensor-parallel runs need MPI ranks, e.g.
#   LLMAPI_LAUNCH_ARGS="--world_size 2 --trtllm_llmapi_launch --oversubscribe"
LLMAPI_LAUNCH_ARGS=${LLMAPI_LAUNCH_ARGS:="--no-mpi --world_size 1"}
# Seconds to wait for the model to load. Not SERVER_TIMEOUT: util.sh defaults
# that to 120 first, which is too short for a model download and load.
LLMAPI_SERVER_TIMEOUT=${LLMAPI_SERVER_TIMEOUT:="900"}

# PIDs of live tritonserver processes. Zombies are excluded: a server that was
# killed stays listed by plain pgrep until its parent reaps it.
function llmapi_server_pids {
    pgrep -x -r R,S,D,I tritonserver
}

# Create an LLM API model repository.
#
#   prepare_llmapi_model_repository <repo-dir> [decoupled=0|1] [policy=0|1] [tp=1]
#
# decoupled: set triton_config.decoupled in model.yaml.
# policy:    also declare model_transaction_policy in config.pbtxt. The launcher
#            passes --disable-auto-complete-config, so auto_complete_config()
#            never applies model.yaml's decoupled flag to the served model
#            configuration; without the policy a decoupled model cannot send
#            more than one response and streaming requests hang.
# tp:        tensor_parallel_size.
function prepare_llmapi_model_repository {
    local repo=$1
    local decoupled=${2:-0}
    local policy=${3:-${decoupled}}
    local tp=${4:-1}
    local model_dir=${repo}/${LLMAPI_MODEL_NAME}
    local yaml=${model_dir}/1/model.yaml

    rm -rf ${repo} && mkdir -p ${repo}
    cp -r ${TENSORRTLLM_DIR}/all_models/llmapi/tensorrt_llm ${model_dir}

    sed -i "s|^model: .*|model: ${LLMAPI_HF_MODEL}|" ${yaml}
    sed -i "s|^tensor_parallel_size: .*|tensor_parallel_size: ${tp}|" ${yaml}
    printf 'kv_cache_config:\n  free_gpu_memory_fraction: %s\n' \
        "${LLMAPI_KV_CACHE_FRACTION}" >> ${yaml}

    if [ "${decoupled}" == "1" ]; then
        sed -i "s|^\(\s*\)decoupled:.*|\1decoupled: True|" ${yaml}
    fi
    if [ "${policy}" == "1" ]; then
        sed -i "0,/^instance_group \[/s//model_transaction_policy {\n  decoupled: true\n}\n\ninstance_group [/" \
            ${model_dir}/config.pbtxt
    fi
}

# Wait until the LLM API model reports ready, tritonserver exits, or the timeout
# expires. Sets WAIT_RET to 0 only when the model is ready.
#
# launch_triton_server.py starts tritonserver detached and returns at once, so
# the launcher's own exit says nothing; watch the server process instead.
function wait_for_llmapi_ready {
    local timeout=${1:-${LLMAPI_SERVER_TIMEOUT}}
    local waited=0
    WAIT_RET=1

    while [ ${waited} -lt ${timeout} ]; do
        if curl -sf -m 3 localhost:8000/v2/models/${LLMAPI_MODEL_NAME}/ready \
            >/dev/null 2>&1; then
            WAIT_RET=0
            return
        fi
        # Allow a few seconds for the detached server process to appear.
        if [ ${waited} -ge 10 ] && ! llmapi_server_pids >/dev/null; then
            echo "*** tritonserver exited before ${LLMAPI_MODEL_NAME} was ready"
            return
        fi
        sleep 2
        waited=$((waited + 2))
    done
    echo "*** timed out after ${timeout}s waiting for ${LLMAPI_MODEL_NAME}"
}

# Start the server on <repo-dir>. Extra arguments are passed to the launcher.
# Sets SERVER_PID to the tritonserver PID, or to 0 when it did not come up.
function run_llmapi_server {
    local repo=$1
    shift
    SERVER_PID=0

    local launcher_log=${SERVER_LOG%.log}_launcher.log
    python3 ${LLMAPI_LAUNCHER} ${LLMAPI_LAUNCH_ARGS} --model_repo ${repo} \
        --log --log-file ${SERVER_LOG} "$@" > ${launcher_log} 2>&1
    if [ $? -ne 0 ]; then
        cat ${launcher_log}
        return 1
    fi

    wait_for_llmapi_ready
    if [ "${WAIT_RET}" != "0" ]; then
        tail -n 20 ${launcher_log}
        kill_llmapi_server
        return 1
    fi
    SERVER_PID=$(llmapi_server_pids | head -n 1)
}

# Stop the server gracefully, then make sure nothing the LLM API spawned
# (Python backend stubs, MPI helper processes) survives: a leftover worker keeps
# the GPU memory and the next server cannot start.
function kill_llmapi_server {
    local pids=$(llmapi_server_pids)
    if [ -n "${pids}" ]; then
        kill -SIGINT ${pids} 2>/dev/null
    fi

    local waited=0
    while [ ${waited} -lt 60 ] && llmapi_server_pids >/dev/null; do
        sleep 2
        waited=$((waited + 2))
    done

    pkill -9 -x tritonserver 2>/dev/null || true
    pkill -9 -f triton_python_backend_stub 2>/dev/null || true
    pkill -9 -f mpi4py.futures 2>/dev/null || true
    return 0
}

# Start tritonserver directly, with the flags launch_triton_server.py would use,
# for tests that need server options the launcher does not expose (for example
# explicit model control). Extra arguments are passed to tritonserver.
# Sets SERVER_PID, or 0 when the model did not become ready.
function run_llmapi_server_direct {
    local repo=$1
    shift
    local server=${TRITON_DIR:-/opt/tritonserver}/bin/tritonserver
    SERVER_PID=0

    ${server} --model-repository=${repo} --log-verbose=3 \
        --disable-auto-complete-config \
        --backend-config=python,shm-region-prefix-name=prefix0_ "$@" \
        > ${SERVER_LOG} 2>&1 &
    wait_for_llmapi_ready
    if [ "${WAIT_RET}" != "0" ]; then
        tail -n 20 ${SERVER_LOG}
        kill_llmapi_server
        return 1
    fi
    SERVER_PID=$(llmapi_server_pids | head -n 1)
}

# Serve <count> instances of the model: gpu_device_ids is the per-instance GPU
# assignment, e.g. "0;0" for two instances on GPU 0 or "0;1" for one per GPU.
# Passing an empty gpu_device_ids leaves the parameter out, which the model
# rejects when count > 1.
function set_llmapi_instance_group {
    local repo=$1
    local count=$2
    local gpu_device_ids=${3:-}
    python3 - "${repo}/${LLMAPI_MODEL_NAME}/config.pbtxt" "${count}" \
        "${gpu_device_ids}" <<'PYEOF'
import re
import sys

path, count, gpu_device_ids = sys.argv[1:4]
with open(path) as f:
    config = f.read()
config, replaced = re.subn(r"count:\s*1\b", f"count: {count}", config, count=1)
assert replaced == 1, "instance_group count not found in config.pbtxt"
if gpu_device_ids:
    config += (
        "\nparameters {\n"
        '  key: "gpu_device_ids"\n'
        f'  value: {{ string_value: "{gpu_device_ids}" }}\n'
        "}\n"
    )
with open(path, "w") as f:
    f.write(config)
PYEOF
}
