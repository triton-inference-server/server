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

# Tests for TensorRT-LLM served through the LLM API (Python) model repository
# that build.py installs with '--backend tensorrtllm'.
#
# Environment:
#   TENSORRTLLM_DIR   installed triton_backend/ layout (default
#                     /opt/tritonserver/tensorrtllm)
#   LLMAPI_HF_MODEL   model to serve (default Qwen/Qwen3-0.6B)
#   REQUIRE_GPUS=1    fail, instead of skipping, tests that need more GPUs than
#                     the machine has (set this on the multi-GPU CI runner)
#   HF_TOKEN          only needed for gated models

source ../common/util.sh
source ../common/trtllm_util.sh

RET=0
SKIPPED=""
TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
TEST_RESULT_FILE="test_results.txt"
MODEL_REPOS="$(pwd)/model_repos"

# The unit tests read these.
export TENSORRTLLM_DIR LLMAPI_MODEL_NAME

function gpu_count {
    nvidia-smi -L 2>/dev/null | grep -c '^GPU '
}

# Run one test file against the running server and check its result count.
#   run_client_tests <test-file> <expected-test-count>
function run_client_tests {
    local test_file=$1
    local expected=$2
    local client_log="${test_file%.py}.log"

    rm -f ${TEST_RESULT_FILE}
    # Commands are tested with 'if' so that errexit, which the caller may have
    # enabled or disabled, neither aborts the run nor is changed here.
    if ! python3 ${test_file} -v > ${client_log} 2>&1; then
        cat ${client_log}
        echo -e "\n***\n*** ${test_file} failed\n***"
        RET=1
    elif ! check_test_results ${TEST_RESULT_FILE} ${expected}; then
        cat ${TEST_RESULT_FILE}
        echo -e "\n***\n*** ${test_file}: unexpected number of tests\n***"
        RET=1
    fi
    # Skips are legitimate only when they are visible: print them.
    local skipped=$(grep -oE 'skipped=[0-9]+' ${client_log} | cut -d= -f2 || true)
    if [ -n "${skipped}" ]; then
        SKIPPED="${SKIPPED}\n***   ${test_file}: ${skipped} test(s)"
        grep -E ' \.\.\. skipped ' ${client_log} || true
    fi
}

# Serve one repository, run one test file, stop the server.
#   serve_and_test <name> <test-file> <expected-test-count> <decoupled> <policy> [tp]
function serve_and_test {
    local name=$1
    local test_file=$2
    local expected=$3
    local decoupled=$4
    local policy=$5
    local tp=${6:-1}

    SERVER_LOG="${name}_server.log"
    export SERVER_LOG
    prepare_llmapi_model_repository ${MODEL_REPOS}/${name} ${decoupled} ${policy} ${tp}

    set +e
    run_llmapi_server ${MODEL_REPOS}/${name}
    if [ "${SERVER_PID}" == "0" ]; then
        echo -e "\n***\n*** Failed to start the server for ${name}\n***"
        cat ${SERVER_LOG%.log}_launcher.log
        RET=1
    else
        run_client_tests ${test_file} ${expected}
    fi
    kill_llmapi_server
    set -e
}

# The server must refuse to load the model and say why.
#   expect_load_failure <name> <repo-dir> <log-pattern>
function expect_load_failure {
    local name=$1
    local repo=$2
    local pattern=$3

    SERVER_LOG="${name}_server.log"
    export SERVER_LOG
    set +e
    run_llmapi_server ${repo} > ${name}_start.log 2>&1
    if [ "${SERVER_PID}" != "0" ]; then
        echo -e "\n***\n*** ${name}: the model loaded but should have been rejected\n***"
        RET=1
    elif ! grep -q "${pattern}" ${SERVER_LOG}; then
        cat ${SERVER_LOG}
        echo -e "\n***\n*** ${name}: '${pattern}' not found in the server log\n***"
        RET=1
    fi
    kill_llmapi_server
    set -e
}

# Tests read the server logs, so none from an earlier run may be left behind.
rm -rf ${MODEL_REPOS} *.log ${TEST_RESULT_FILE}
mkdir -p ${MODEL_REPOS}
# No --upgrade: the server's own numpy and friends must stay as the image has them.
pip3 install "tritonclient[all]" requests > /dev/null

set -e

# Non-decoupled model: HTTP and gRPC unary requests, error paths, metrics.
serve_and_test unary test_unary.py 13 0 0

# Decoupled model: streaming, cancellation and stop requests.
serve_and_test decoupled test_decoupled.py 9 1 1

# Decoupled flag in model.yaml only: documents why config.pbtxt needs the policy.
serve_and_test decoupled_config test_decoupled_config.py 3 1 0

# Explicit model control, started the way the launcher cannot.
SERVER_LOG="model_control_server.log"
export SERVER_LOG
prepare_llmapi_model_repository ${MODEL_REPOS}/model_control 0 0 1
set +e
run_llmapi_server_direct ${MODEL_REPOS}/model_control \
    --model-control-mode=explicit --load-model=${LLMAPI_MODEL_NAME}
if [ "${SERVER_PID}" == "0" ]; then
    echo -e "\n***\n*** Failed to start the model control server\n***"
    RET=1
else
    run_client_tests test_model_control.py 2
fi
kill_llmapi_server
set -e

# Two instances on one GPU, and the configurations that must be rejected.
SERVER_LOG="multi_instance_server.log"
export SERVER_LOG
prepare_llmapi_model_repository ${MODEL_REPOS}/multi_instance 0 0 1
set_llmapi_instance_group ${MODEL_REPOS}/multi_instance 2 "0;0"
set +e
run_llmapi_server ${MODEL_REPOS}/multi_instance
if [ "${SERVER_PID}" == "0" ]; then
    echo -e "\n***\n*** Failed to start the multi-instance server\n***"
    RET=1
else
    run_client_tests test_multi_instance.py 2
fi
kill_llmapi_server
set -e

prepare_llmapi_model_repository ${MODEL_REPOS}/no_gpu_ids 0 0 1
set_llmapi_instance_group ${MODEL_REPOS}/no_gpu_ids 2 ""
expect_load_failure no_gpu_ids ${MODEL_REPOS}/no_gpu_ids \
    "gpu_device_ids parameter required"

prepare_llmapi_model_repository ${MODEL_REPOS}/short_gpu_ids 0 0 1
set_llmapi_instance_group ${MODEL_REPOS}/short_gpu_ids 2 "0"
expect_load_failure short_gpu_ids ${MODEL_REPOS}/short_gpu_ids \
    "gpu_device_ids mismatch"

# Tensor parallelism. The server is only started when the GPUs exist; the test
# still runs, so that a missing GPU is reported as a skip (or a failure when
# REQUIRE_GPUS=1) instead of the case disappearing from the results.
for TP in 2 4; do
    export LLMAPI_TP=${TP}
    SERVER_LOG="tp${TP}_server.log"
    export SERVER_LOG
    if [ "$(gpu_count)" -ge ${TP} ]; then
        prepare_llmapi_model_repository ${MODEL_REPOS}/tp${TP} 0 0 ${TP}
        set +e
        LLMAPI_LAUNCH_ARGS="--world_size ${TP} --trtllm_llmapi_launch --oversubscribe" \
            run_llmapi_server ${MODEL_REPOS}/tp${TP}
        if [ "${SERVER_PID}" == "0" ]; then
            echo -e "\n***\n*** Failed to start the TP=${TP} server\n***"
            RET=1
        fi
        set -e
    fi
    run_client_tests test_tp.py 1
    set +e
    kill_llmapi_server
    set -e
done

if [ -n "${SKIPPED}" ]; then
    echo -e "\n***\n*** SKIPPED tests (see the logs for the reason):${SKIPPED}\n***"
fi

if [ ${RET} -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit ${RET}
