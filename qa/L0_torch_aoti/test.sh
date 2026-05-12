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

source ../common/util.sh

if [[ "${DEBUG}" == "true" ]]; then
    set -x
else
    set +x
fi

COLOR_DARK="\033[90m"
COLOR_ERROR="\033[31m"
COLOR_INFO="\033[94m"
COLOR_RESET="\033[0m"
COLOR_STATUS="\033[36m"
COLOR_SUCCESS="\033[32m"
COLOR_WARNING="\033[33m"
RET=0

REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
if [[ "$#" -ge 1 ]]; then
    REPO_VERSION=$1
fi
if [[ -z "$REPO_VERSION" ]]; then
    echo -e "${COLOR_ERROR}Repository version must be specified${COLOR_RESET}" 1>&2
    echo -e "${COLOR_ERROR}\n***\n*** Test Failed\n***${COLOR_RESET}" 1>&2
    exit 1
fi
if [[ ! -z "$TEST_REPO_ARCH" ]]; then
    REPO_VERSION=${REPO_VERSION}_${TEST_REPO_ARCH}
fi

export CUDA_VISIBLE_DEVICES=0

MODELDIR=${MODELDIR:=`pwd`/models}
DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
BACKEND_DIR=${TRITON_DIR}/backends

# PyTorch on SBSA requires libgomp to be loaded first. See the following
# GitHub issue for more information:
# https://github.com/pytorch/pytorch/issues/2575
arch=`uname -m`
echo -e "${COLOR_DARK}Detected architecture: ${arch}${COLOR_RESET}"
if [[ "${arch}" == "aarch64" ]]; then
    SERVER_LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libgomp.so.1
    echo -e "${COLOR_DARK}SERVER_LD_PRELOAD=${SERVER_LD_PRELOAD}${COLOR_RESET}"
fi

# If BACKENDS not specified, set to all
BACKENDS=${BACKENDS:="pytorch"}
export BACKENDS

# Copy the models into the model repository
echo -e "${COLOR_DARK}Setting up model repository in ${MODELDIR}${COLOR_RESET}"
rm -rf ${MODELDIR} && mkdir -p ${MODELDIR}
models=(
    "torch_aoti_complex_index"
    "torch_aoti_complex_named"
    "torch_aoti_int8_int8"
    "torch_aoti_int16_int16"
    "torch_aoti_int32_int32"
    "torch_aoti_int64_int64"
    "torch_aoti_float16_float16"
    "torch_aoti_float32_float32"
    "torchvision_aoti"
)
for model in "${models[@]}"; do
    cp -r ${DATADIR}/qa_model_repository/${model} ${MODELDIR}/${model}
    echo -e "${COLOR_DARK}ls ${MODELDIR}/${model}${COLOR_RESET}"
    ls -lha ${MODELDIR}/${model}
done
echo -e "${COLOR_DARK}ls ${MODELDIR}${COLOR_RESET}"
ls -lha ${MODELDIR}

SERVER_ARGS="--model-repository=${MODELDIR} --log-verbose=1"
SERVER_LOG="./torch_aoti_complex_named-server.log"
CLIENT_LOG="./torch_aoti_complex_named-client.log"

echo -e "${COLOR_DARK}Running ${SERVER} with model repository ${MODELDIR}${COLOR_RESET}"
run_server
if [[ "${SERVER_PID}" -eq 0 ]]; then
    echo -e "${COLOR_ERROR}\n***\n*** Failed to start ${SERVER}\n***${COLOR_RESET}" &1>2
    cat ${SERVER_LOG} &1>2
    echo -e "\n" &1>2
    exit 1
fi

# Install torch framework
echo -e "${COLOR_DARK}Installing PyTorch framework required by tests${COLOR_RESET}"
pip install torch

# Run the Tests
TEST_NAME="torch_aoti_infer_test"
python3 ./${TEST_NAME}.py >> ${CLIENT_LOG} 2>&1
EXIT_CODE=$?
if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo -e "${COLOR_ERROR}\n***\n*** Test '${TEST_NAME}' Failed with exit code ${EXIT_CODE}\n***${COLOR_RESET}" &1>2
    cat ${CLIENT_LOG} &1>2
    echo -e "\n" &1>2
    RET=1
else
    echo -e "${COLOR_INFO}\n***\n*** Test '${TEST_NAME}' Passed\n***${COLOR_RESET}"
fi

# Cleanup
echo -e "${COLOR_DARK}Killing server (pid: ${SERVER_PID})${COLOR_RESET}"
kill -s SIGINT ${SERVER_PID}
wait ${SERVER_PID} || true
echo -e "${COLOR_DARK}Removing model repository${COLOR_RESET}"
for model in "${models[@]}"; do
    rm -rf ${MODELDIR}/${model}
done

# Report results and exit.
if [[ ${RET} -ne 0 ]]; then
    echo -e "${COLOR_ERROR}\n***\n*** Test Suite FAILED\n***${COLOR_RESET}" &1>2
else
    echo -e "${COLOR_SUCCESS}\n***\n*** Test Suite PASSED\n***${COLOR_RESET}"
fi

exit ${RET}
