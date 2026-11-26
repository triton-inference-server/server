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
SERVER_TIMEOUT=${SERVER_TIMEOUT:=120}

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
BAD_MODELDIR=`pwd`/bad_models
rm -rf ${MODELDIR} ${BAD_MODELDIR}
mkdir -p ${MODELDIR}
models=(
    "torch_aoti_complex_index"
    "torch_aoti_complex_named"
    "torch_aoti_int8_int8"
    "torch_aoti_int16_int16"
    "torch_aoti_int32_int32"
    "torch_aoti_int64_int64"
    "torch_aoti_float16_float16"
    "torch_aoti_float32_float32"
    "torch_aoti_variable_float32"
    "torch_aoti_multi_instance_float32"
    "torchvision_aoti"
)
for model in "${models[@]}"; do
    cp -r ${DATADIR}/qa_model_repository/${model} ${MODELDIR}/${model}
    echo -e "${COLOR_DARK}ls ${MODELDIR}/${model}${COLOR_RESET}"
    ls -lha ${MODELDIR}/${model}
done

# Sequence-batching AOTI models live in the implicit-state sequence repository.
sequence_models=(
    "torch_aoti_sequence_float32"
    "torch_aoti_sequence_initstate_float32"
    "torch_aoti_sequence_forward_float32"
)
for model in "${sequence_models[@]}"; do
    cp -r ${DATADIR}/qa_sequence_implicit_model_repository/${model} ${MODELDIR}/${model}
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

# Negative tests: these models declare unsupported types (TYPE_STRING CORRID /
# state) and must fail to load. Start a separate server (exit-on-error=false) so
# it stays up despite the load failures, then assert the models are not ready.
# Two models that load are added to check request-time errors.
echo -e "${COLOR_DARK}Negative (load-failure) tests${COLOR_RESET}"
mkdir -p ${BAD_MODELDIR}
bad_models=(
    "torch_aoti_sequence_bad_corrid"
    "torch_aoti_sequence_bad_state"
)
for model in "${bad_models[@]}"; do
    cp -r ${DATADIR}/qa_sequence_implicit_model_repository/${model} ${BAD_MODELDIR}/${model}
done

# An optional input is not supported for AOT Inductor models: the model must
# fail to load without taking the server down.
model="torch_aoti_optional_input"
cp -r ${DATADIR}/qa_model_repository/torch_aoti_float32_float32 ${BAD_MODELDIR}/${model}
sed -i -e "s/name: \"torch_aoti_float32_float32\"/name: \"${model}\"/" \
       -e 's/name: "ARGS\[1\]"/name: "ARGS[1]"\n    optional: true/' ${BAD_MODELDIR}/${model}/config.pbtxt
bad_models+=("${model}")

# A model whose configuration declares another output datatype than the model
# produces loads, but a request must get an error response and the server must
# keep running.
BAD_DTYPE_MODEL="torch_aoti_bad_output_dtype"
cp -r ${DATADIR}/qa_model_repository/torch_aoti_float32_float32 ${BAD_MODELDIR}/${BAD_DTYPE_MODEL}
sed -i -e "s/name: \"torch_aoti_float32_float32\"/name: \"${BAD_DTYPE_MODEL}\"/" \
       -e '/name: "RESULT"/{n;s/TYPE_FP32/TYPE_INT32/}' ${BAD_MODELDIR}/${BAD_DTYPE_MODEL}/config.pbtxt

# The same for a batch input whose target is not an input of the model (it
# replaces the second input in the configuration).
BAD_BATCH_INPUT_MODEL="torch_aoti_bad_batch_input"
cp -r ${DATADIR}/qa_model_repository/torch_aoti_float32_float32 ${BAD_MODELDIR}/${BAD_BATCH_INPUT_MODEL}
sed -i -e "s/name: \"torch_aoti_float32_float32\"/name: \"${BAD_BATCH_INPUT_MODEL}\"/" \
       -e '/^  },$/,/^  }$/c\  }' ${BAD_MODELDIR}/${BAD_BATCH_INPUT_MODEL}/config.pbtxt
echo 'batch_input [{ kind: BATCH_ELEMENT_COUNT target_name: "NOT_AN_INPUT" data_type: TYPE_FP32 source_input: "ARGS[0]" }]' \
    >> ${BAD_MODELDIR}/${BAD_BATCH_INPUT_MODEL}/config.pbtxt

SERVER_ARGS="--model-repository=${BAD_MODELDIR} --exit-on-error=false --log-verbose=1"
SERVER_LOG="./torch_aoti_negative-server.log"
run_server_tolive
if [[ "${SERVER_PID}" -eq 0 ]]; then
    echo -e "${COLOR_ERROR}\n***\n*** Failed to start ${SERVER} (negative phase)\n***${COLOR_RESET}" 1>&2
    cat ${SERVER_LOG} 1>&2
    RET=1
else
    wait_for_model_stable ${SERVER_TIMEOUT}
    for model in "${bad_models[@]}"; do
        code=$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v2/models/${model}/ready)
        if [[ "${code}" == "200" ]]; then
            echo -e "${COLOR_ERROR}*** Negative model '${model}' unexpectedly loaded (ready)${COLOR_RESET}" 1>&2
            RET=1
        else
            echo -e "${COLOR_INFO}*** Negative model '${model}' correctly failed to load${COLOR_RESET}"
        fi
    done
    # Request-time errors: the model must be ready, the request must get an
    # error response with the expected message, and the server must stay live.
    row="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]"
    input0="{\"name\":\"ARGS[0]\",\"shape\":[1,16],\"datatype\":\"FP32\",\"data\":${row}}"
    input1="{\"name\":\"ARGS[1]\",\"shape\":[1,16],\"datatype\":\"FP32\",\"data\":${row}}"
    declare -A error_requests=(
        ["${BAD_DTYPE_MODEL}"]="{\"inputs\":[${input0},${input1}]}"
        ["${BAD_BATCH_INPUT_MODEL}"]="{\"inputs\":[${input0}]}"
    )
    declare -A error_messages=(
        ["${BAD_DTYPE_MODEL}"]="TYPE_INT32"
        ["${BAD_BATCH_INPUT_MODEL}"]="NOT_AN_INPUT"
    )
    for model in "${BAD_DTYPE_MODEL}" "${BAD_BATCH_INPUT_MODEL}"; do
        set +e
        ready=$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v2/models/${model}/ready)
        code=$(curl -s -o ./${model}-response.log -w "%{http_code}" -X POST -d "${error_requests[${model}]}" localhost:8000/v2/models/${model}/infer)
        grep -q "${error_messages[${model}]}" ./${model}-response.log
        message=$?
        live=$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v2/health/live)
        set -e
        if [[ "${ready}" != "200" ]] || [[ "${code}" != "400" && "${code}" != "500" ]] || [[ ${message} -ne 0 ]] || [[ "${live}" != "200" ]]; then
            echo -e "${COLOR_ERROR}*** Negative model '${model}': expected a ready model, an error response naming '${error_messages[${model}]}' and a live server, got ready ${ready}, HTTP ${code} and live ${live}${COLOR_RESET}" 1>&2
            cat ./${model}-response.log 1>&2
            RET=1
        else
            echo -e "${COLOR_INFO}*** Negative model '${model}' correctly returned an error (HTTP ${code}), server still live${COLOR_RESET}"
        fi
    done
    kill -s SIGINT ${SERVER_PID}
    wait ${SERVER_PID} || true
fi

# Multi-GPU tests: with two GPUs visible, the multi-instance model (KIND_GPU
# without 'gpus') gets model instances on each GPU. Every instance must run
# the model on its own device. Start a separate server for this phase.
echo -e "${COLOR_DARK}Multi-GPU tests${COLOR_RESET}"
export CUDA_VISIBLE_DEVICES=0,1
gpu_count=`nvidia-smi -L | grep GPU | wc -l`
if [[ "${gpu_count}" -lt 2 ]]; then
    echo -e "${COLOR_ERROR}\n***\n*** Multi-GPU tests require 2 GPUs, found ${gpu_count}\n***${COLOR_RESET}" 1>&2
    RET=1
else
    MULTI_GPU_MODELDIR=`pwd`/multi_gpu_models
    rm -rf ${MULTI_GPU_MODELDIR}
    mkdir -p ${MULTI_GPU_MODELDIR}
    cp -r ${DATADIR}/qa_model_repository/torch_aoti_multi_instance_float32 ${MULTI_GPU_MODELDIR}/

    SERVER_ARGS="--model-repository=${MULTI_GPU_MODELDIR} --log-verbose=1"
    SERVER_LOG="./torch_aoti_multi_gpu-server.log"
    CLIENT_LOG="./torch_aoti_multi_gpu-client.log"
    run_server
    if [[ "${SERVER_PID}" -eq 0 ]]; then
        echo -e "${COLOR_ERROR}\n***\n*** Failed to start ${SERVER} (multi-GPU phase)\n***${COLOR_RESET}" 1>&2
        cat ${SERVER_LOG} 1>&2
        RET=1
    else
        set +e
        python3 ./${TEST_NAME}.py TorchAotiTest.test_multi_instance >> ${CLIENT_LOG} 2>&1
        EXIT_CODE=$?
        if [[ ${EXIT_CODE} -ne 0 ]]; then
            echo -e "${COLOR_ERROR}\n***\n*** Test '${TEST_NAME}' (multi-GPU phase) Failed with exit code ${EXIT_CODE}\n***${COLOR_RESET}" 1>&2
            cat ${CLIENT_LOG} 1>&2
            RET=1
        else
            echo -e "${COLOR_INFO}\n***\n*** Test '${TEST_NAME}' (multi-GPU phase) Passed\n***${COLOR_RESET}"
        fi
        # The server must survive the test: a failing instance must not take it down.
        if ! kill -0 ${SERVER_PID} > /dev/null 2>&1; then
            echo -e "${COLOR_ERROR}\n***\n*** ${SERVER} exited during the multi-GPU phase\n***${COLOR_RESET}" 1>&2
            RET=1
        else
            kill -s SIGINT ${SERVER_PID}
            wait ${SERVER_PID}
            if [[ $? -ne 0 ]]; then
                echo -e "${COLOR_ERROR}\n***\n*** ${SERVER} did not shut down cleanly after the multi-GPU phase\n***${COLOR_RESET}" 1>&2
                RET=1
            fi
        fi
        set -e
    fi
fi

# Report results and exit.
if [[ ${RET} -ne 0 ]]; then
    echo -e "${COLOR_ERROR}\n***\n*** Test Suite FAILED\n***${COLOR_RESET}" &1>2
else
    echo -e "${COLOR_SUCCESS}\n***\n*** Test Suite PASSED\n***${COLOR_RESET}"
fi

exit ${RET}
