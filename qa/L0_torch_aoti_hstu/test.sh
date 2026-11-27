#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# HSTU (Generative Recommenders) Torch AOTI serving test, mirroring the serving
# half of Devtech-Compute/distributed-recommender: ci/tritonserver_test.sh.
#
# Serves an exported HSTU ranking package with `platform: "torch_aoti"` and runs
# the HSTU client against it, replaying dumped input batches through a warmup and
# through KV-cache miss and hit phases. The test image carries the HSTU runtime
# and the LD_PRELOAD that resolves its ops; the AOTI package and the input dump
# come from the CI job's export step, in a volume at EXPORT_DIR.

source ../common/util.sh

if [[ "${DEBUG}" == "true" ]]; then
    set -x
else
    set +x
fi

# The CI harness runs this with `bash -ex`. The test checks exit codes itself and
# has to reach the FlexKV and tritonserver teardown on failure, so errexit stays
# off; util.sh helpers turn it back on, hence the repeated `set +e`.
set +e

COLOR_DARK="\033[90m"
COLOR_ERROR="\033[31m"
COLOR_INFO="\033[94m"
COLOR_RESET="\033[0m"
COLOR_SUCCESS="\033[32m"
RET=0

export CUDA_VISIBLE_DEVICES=0

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_TIMEOUT=${SERVER_TIMEOUT:=300}

RECSYS_DIR=${RECSYS_DIR:="/workspace/recsys-examples/examples/hstu"}
AOTI_DIR=${AOTI_DIR:=${RECSYS_DIR}/inference_aoti}

EXPORT_DIR=${EXPORT_DIR:="/exported_hstu_model"}
MODEL_NAME=${MODEL_NAME:="hstu_gr_ranking_kvcache"}
EXPORTED_MODEL=${EXPORTED_MODEL:=${EXPORT_DIR}/${MODEL_NAME}_model}
DUMP_DIR=${DUMP_DIR:=${EXPORT_DIR}/export_test_dump}

MODELDIR=${MODELDIR:=./models}
CLIENT_LOG="./${MODEL_NAME}-client.log"
SERVER_LOG="./${MODEL_NAME}-server.log"
KVCACHE_LOG="./${MODEL_NAME}-kvcache.log"
EXPORT_KVCACHE_LOG="./cpp_kvcache_server.log"

export FLEXKV_LOG_LEVEL=${FLEXKV_LOG_LEVEL:="WARNING"}

export KVCACHE_MANAGER_CONFIG_FILE=${KVCACHE_MANAGER_CONFIG_FILE:=${AOTI_DIR}/kvcache_cpp_runtime.yaml}

KVCACHE_PID=0

# The torch_aoti model talks to a FlexKV server for its KV cache.
function start_kvcache_server () {
    KVCACHE_PID=0
    python3 ${AOTI_DIR}/start_flexkv_server_for_kvcache_cpp.py \
        --config_file ${KVCACHE_MANAGER_CONFIG_FILE} >> ${KVCACHE_LOG} 2>&1 &
    local pid=$!
    sleep 10
    if ! kill -0 ${pid} > /dev/null 2>&1; then
        echo -e "${COLOR_ERROR}\n***\n*** Failed to start FlexKV KV-cache server\n***${COLOR_RESET}" 1>&2
        cat ${KVCACHE_LOG} 1>&2
        return 1
    fi
    KVCACHE_PID=${pid}
    echo -e "${COLOR_DARK}FlexKV KV-cache server running (pid: ${KVCACHE_PID})${COLOR_RESET}"
}

function stop_kvcache_server () {
    if [[ "${KVCACHE_PID}" -ne 0 ]]; then
        echo -e "${COLOR_DARK}Killing FlexKV KV-cache server (pid: ${KVCACHE_PID})${COLOR_RESET}"
        kill ${KVCACHE_PID} > /dev/null 2>&1 || true
        wait ${KVCACHE_PID} > /dev/null 2>&1 || true
        KVCACHE_PID=0
    fi
}

for artifact in ${EXPORTED_MODEL} ${DUMP_DIR}; do
    if [[ ! -d ${artifact} ]]; then
        echo -e "${COLOR_ERROR}\n***\n*** Missing export artifact ${artifact}\n***${COLOR_RESET}" 1>&2
        echo -e "${COLOR_ERROR}\n***\n*** Test Suite FAILED\n***${COLOR_RESET}" 1>&2
        exit 1
    fi
done

# The export step's KV-cache server log, alongside this test's own logs so CI
# collects it with them.
if [[ -f ${EXPORT_DIR}/cpp_kvcache_server.log ]]; then
    cp ${EXPORT_DIR}/cpp_kvcache_server.log ${EXPORT_KVCACHE_LOG}
fi

echo -e "${COLOR_DARK}Setting up model repository in ${MODELDIR}${COLOR_RESET}"
rm -rf ${MODELDIR}
mkdir -p ${MODELDIR}
cp -apr ${AOTI_DIR}/triton_aoti/${MODEL_NAME} ${MODELDIR}
cp -apr ${EXPORTED_MODEL} ${MODELDIR}/${MODEL_NAME}/1
echo -e "${COLOR_DARK}ls ${MODELDIR}/${MODEL_NAME}${COLOR_RESET}"
ls -lha ${MODELDIR}/${MODEL_NAME}

start_kvcache_server || exit 1

SERVER_ARGS="--model-repository=${MODELDIR} --log-verbose=1"
run_server
set +e
if [[ "${SERVER_PID}" -eq 0 ]]; then
    echo -e "${COLOR_ERROR}\n***\n*** Failed to start ${SERVER}\n***${COLOR_RESET}" 1>&2
    cat ${SERVER_LOG} 1>&2
    stop_kvcache_server
    echo -e "${COLOR_ERROR}\n***\n*** Test Suite FAILED\n***${COLOR_RESET}" 1>&2
    exit 1
fi

# The client replays the dump from the recsys tree, as DevTech's script does.
rm -rf ${AOTI_DIR}/$(basename ${DUMP_DIR})
cp -apr ${DUMP_DIR} ${AOTI_DIR}/
TEST_NAME="test_tritonserver_aoti_hstu_model"
python3 ${AOTI_DIR}/${TEST_NAME}.py --dump_dir ${AOTI_DIR}/$(basename ${DUMP_DIR}) > ${CLIENT_LOG} 2>&1
EXIT_CODE=$?
if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo -e "${COLOR_ERROR}\n***\n*** Test '${TEST_NAME}' Failed with exit code ${EXIT_CODE}\n***${COLOR_RESET}" 1>&2
    cat ${CLIENT_LOG} 1>&2
    RET=1
else
    echo -e "${COLOR_INFO}\n***\n*** Test '${TEST_NAME}' Passed\n***${COLOR_RESET}"
fi

echo -e "${COLOR_DARK}Killing server (pid: ${SERVER_PID})${COLOR_RESET}"
kill -s SIGINT ${SERVER_PID}
wait ${SERVER_PID} || true
stop_kvcache_server

if [[ ${RET} -ne 0 ]]; then
    echo -e "${COLOR_ERROR}\n***\n*** Test Suite FAILED\n***${COLOR_RESET}" 1>&2
else
    echo -e "${COLOR_SUCCESS}\n***\n*** Test Suite PASSED\n***${COLOR_RESET}"
fi

exit ${RET}
