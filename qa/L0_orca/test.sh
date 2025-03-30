#!/bin/bash
# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

RET=0
BASE_DIR=$(pwd)
NUM_GPUS=${NUM_GPUS:=1}
TENSORRTLLM_BACKEND_REPO_TAG=${TENSORRTLLM_BACKEND_REPO_TAG:="main"}
TRITON_REPO_ORG=${TRITON_REPO_ORG:="https://github.com/triton-inference-server"}
TRT_ROOT="/usr/local/tensorrt"

MODEL_NAME="gpt2_tensorrt_llm"
NAME="tensorrt_llm_benchmarking_test"
MODEL_REPOSITORY="$(pwd)/triton_model_repo"
TENSORRTLLM_BACKEND_DIR="/workspace/tensorrtllm_backend"
GPT_DIR="$TENSORRTLLM_BACKEND_DIR/tensorrt_llm/examples/gpt"
TOKENIZER_DIR="$GPT_DIR/gpt2"
ENGINES_DIR="${BASE_DIR}/engines/inflight_batcher_llm/${NUM_GPUS}-gpu"
TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
BACKEND_DIR=${TRITON_DIR}/backends
SERVER_LOG="${NAME}_server.log"
SERVER_TIMEOUT=${SERVER_TIMEOUT:=120}
CLIENT_PY=${BASE_DIR}/orca_http_test.py
CLIENT_LOG="${NAME}_orca_http_test.log"
source ../common/util.sh

function prepare_model_repository {
    rm -rf ${MODEL_REPOSITORY} && mkdir ${MODEL_REPOSITORY}
    cp -r ${TENSORRTLLM_BACKEND_DIR}/all_models/inflight_batcher_llm/* ${MODEL_REPOSITORY}
    rm -rf ${MODEL_REPOSITORY}/tensorrt_llm_bls
    mv "${MODEL_REPOSITORY}/ensemble" "${MODEL_REPOSITORY}/${MODEL_NAME}"

    replace_config_tags "model_version: -1" "model_version: 1" "${MODEL_REPOSITORY}/${MODEL_NAME}/config.pbtxt"
    replace_config_tags '${triton_max_batch_size}' "128" "${MODEL_REPOSITORY}/${MODEL_NAME}/config.pbtxt"
    replace_config_tags 'name: "ensemble"' "name: \"$MODEL_NAME\"" "${MODEL_REPOSITORY}/${MODEL_NAME}/config.pbtxt"
    replace_config_tags '${logits_datatype}' "TYPE_FP32" "${MODEL_REPOSITORY}/${MODEL_NAME}/config.pbtxt"

    replace_config_tags '${triton_max_batch_size}' "128" "${MODEL_REPOSITORY}/preprocessing/config.pbtxt"
    replace_config_tags '${preprocessing_instance_count}' '1' "${MODEL_REPOSITORY}/preprocessing/config.pbtxt"
    replace_config_tags '${tokenizer_dir}' "${TOKENIZER_DIR}/" "${MODEL_REPOSITORY}/preprocessing/config.pbtxt"
    replace_config_tags '${logits_datatype}' "TYPE_FP32" "${MODEL_REPOSITORY}/preprocessing/config.pbtxt"
    replace_config_tags '${max_queue_delay_microseconds}' "1000000" "${MODEL_REPOSITORY}/preprocessing/config.pbtxt"
    replace_config_tags '${max_queue_size}' "0" "${MODEL_REPOSITORY}/preprocessing/config.pbtxt"

    replace_config_tags '${triton_max_batch_size}' "128" "${MODEL_REPOSITORY}/postprocessing/config.pbtxt"
    replace_config_tags '${postprocessing_instance_count}' '1' "${MODEL_REPOSITORY}/postprocessing/config.pbtxt"
    replace_config_tags '${tokenizer_dir}' "${TOKENIZER_DIR}/" "${MODEL_REPOSITORY}/postprocessing/config.pbtxt"
    replace_config_tags '${logits_datatype}' "TYPE_FP32" "${MODEL_REPOSITORY}/postprocessing/config.pbtxt"

    replace_config_tags '${triton_max_batch_size}' "128" "${MODEL_REPOSITORY}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${decoupled_mode}' 'true' "${MODEL_REPOSITORY}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${max_queue_delay_microseconds}' "1000000" "${MODEL_REPOSITORY}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${batching_strategy}' 'inflight_fused_batching' "${MODEL_REPOSITORY}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${engine_dir}' "${ENGINES_DIR}" "${MODEL_REPOSITORY}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${triton_backend}' "tensorrtllm" "${MODEL_REPOSITORY}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${max_queue_size}' "0" "${MODEL_REPOSITORY}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${logits_datatype}' "TYPE_FP32" "${MODEL_REPOSITORY}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${encoder_input_features_data_type}' "TYPE_FP32" "${MODEL_REPOSITORY}/tensorrt_llm/config.pbtxt"
}

# Wait until server health endpoint shows ready. Sets WAIT_RET to 0 on
# success, 1 on failure
function wait_for_server_ready() {
    local wait_time_secs="${1:-30}"
    shift
    local spids=("$@")

    WAIT_RET=0

    for _ in $(seq "$wait_time_secs"); do
        for pid in "${spids[@]}"; do
            if ! kill -0 "$pid" >/dev/null 2>&1; then
                echo "=== Server not running."
                WAIT_RET=1
                return
            fi
        done

        sleep 1

        if curl -s --fail localhost:8000/v2/health/ready &&
            curl -s --fail -w "%{http_code}" -o /dev/null -d '{"log_verbose_level":1}' localhost:8000/v2/logging; then
            return
        fi
    done

    echo "=== Timeout $wait_time_secs secs. Server not ready."
    WAIT_RET=1
}

function run_server {
    python3 ${TENSORRTLLM_BACKEND_DIR}/scripts/launch_triton_server.py --world_size="${NUM_GPUS}" --model_repo="${MODEL_REPOSITORY}" >${SERVER_LOG} 2>&1 &
    sleep 2 # allow time to obtain the pid(s)
    # Read PIDs into an array, trimming whitespaces
    readarray -t SERVER_PID < <(pgrep "tritonserver")

    wait_for_server_ready ${SERVER_TIMEOUT} "${SERVER_PID[@]}"
    if [ "$WAIT_RET" != "0" ]; then
        # Cleanup
        kill "${SERVER_PID[@]}" >/dev/null 2>&1 || true
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi
}

function kill_server {
    pgrep tritonserver | xargs kill -SIGINT
    for pid in "${SERVER_PID[@]}"; do
        echo "Waiting for proc ${pid} to terminate..."
        while kill -0 $pid >/dev/null 2>&1; do
            sleep 1
        done
    done
}

clone_tensorrt_llm_backend_repo
build_gpt2_base_model
build_gpt2_tensorrt_engine
prepare_model_repository

set +e
run_server

if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

RET=0

python3 $CLIENT_PY "http://localhost:8000/v2/models/${MODEL_NAME}/generate" >>$CLIENT_LOG 2>&1

if [ $? -ne 0 ]; then
    echo "Failed: Client test had a non-zero return code."
    RET=1
fi

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** ORCA Test Passed\n***"
else
    cat $SERVER_LOG
    cat $CLIENT_LOG
    echo -e "\n***\n*** ORCA Test FAILED\n***"
fi

kill_server
set -e
exit $RET
