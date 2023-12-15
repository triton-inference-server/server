#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "${REPO_VERSION}" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi
if [ ! -z "$TEST_REPO_ARCH" ]; then
    REPO_VERSION=${REPO_VERSION}_${TEST_REPO_ARCH}
fi

source ../common/util.sh

# Setup client/perf_analyzer
CLIENT_LOG="./perf_analyzer.log"
PERF_ANALYZER=../clients/perf_analyzer

function check_perf_analyzer_error {
    ERROR_STRING="error | Request count: 0 | : 0 infer/sec"
    CLIENT_RET="$1"
    if [ ${CLIENT_RET} -ne 0 ]; then
        cat ${CLIENT_LOG}
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    if [ $(cat ${CLIENT_LOG} |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
        cat ${CLIENT_LOG}
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
}

# Checks that the model infer/sec performance is equal to an expected value
# +/- some tolerance.
# $1: csv result file from PA run
# $2: expected infer/sec value
# $3: tolerance for expected value equality
function check_performance {
    # get the boundary values based on the tolerance percentage
    MIN=$(python3 -c "print(${2} * (1 - ${3}))")
    MAX=$(python3 -c "print(${2} * (1 + ${3}))")

    # delete all but the 2nd line in the resulting file
    # then get the 2nd column value which is the infer/sec measurement
    report_val=$(sed '2!d' $1 | awk -F ',' {'print $2'})

    # check if within tolerance
    ret=$(python3 -c "print(${report_val} >= ${MIN} and ${report_val} <= ${MAX})")
    if [ "$ret" = "False" ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
}

# Iterate over the grpc results to ensure gRPC times are greater than 0
# $1: client log file
# example line: Avg gRPC time: 42648 usec (marshal 6 usec + response wait 42640 usec + unmarshal 2 usec)
function check_grpc_time {
    grep "gRPC" $1 | awk '{print $4}' | while read -r line; do
        if [ $line -eq 0 ]; then
            RET=1
        fi
    done
}

# Create input_data.json to communicate the requested model delay
# $1: desired model delay
function create_input_data {
    echo "{\"data\":[{\"INPUT0\" : [${1}]}]}" > input_data.json
}

# Setup server
export CUDA_VISIBLE_DEVICES=0
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=`pwd`/models"
SERVER_LOG="./inference_server.log"

rm -f $SERVER_LOG $CLIENT_LOG
MODEL_DIR="./models"
rm -fr ${MODEL_DIR} && mkdir ${MODEL_DIR}
MODELS="ground_truth"

for model in ${MODELS}; do
    # Add version directory to each model if non-existent
    mkdir -p "${MODEL_DIR}/${model}/1"
    cp ../python_models/${model}/model.py     ./models/${model}/1/model.py
    cp ../python_models/${model}/config.pbtxt ./models/${model}/config.pbtxt
done

# Run server
run_server
if [ "${SERVER_PID}" == "0" ]; then
    echo -e "\n***\n*** Failed to start ${SERVER}\n***"
    cat ${SERVER_LOG}
    exit 1
fi

# Run perf_analyzer
set +e
RET=0
PROTOCOLS="http grpc"
OUTPUT_FILE="results"
MODEL_DELAYS=(0.05 0.5)
TOLERANCE="0.05"

for model_delay in ${MODEL_DELAYS[@]}; do
    create_input_data ${model_delay}
    EXPECTED_RESULT=$(python3 -c "print(1 / ${model_delay})")
    for protocol in ${PROTOCOLS}; do
        for model in ${MODELS}; do
        echo "================================================================"
        echo "[PERMUTATION] Protocol=${protocol} Model=${model}"
        echo "================================================================"

            ${PERF_ANALYZER} -v -i ${protocol} --concurrency-range 2 --input-data input_data.json -m ${model} -f ${OUTPUT_FILE} | tee ${CLIENT_LOG} 2>&1
            check_perf_analyzer_error $?

            check_performance ${OUTPUT_FILE} ${EXPECTED_RESULT} ${TOLERANCE}

            if [ "${protocol}" == "grpc" ]; then
                check_grpc_time ${CLIENT_LOG}
            fi
        done;
    done;
done;


set -e

# Cleanup
kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo "=== START SERVER LOG ==="
  cat ${SERVER_LOG}
  echo "=== END SERVER LOG ==="
  echo "=== START CLIENT LOG ==="
  cat ${CLIENT_LOG}
  echo "=== END CLIENT LOG ==="
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit ${RET}
