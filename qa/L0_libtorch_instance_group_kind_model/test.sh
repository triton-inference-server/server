#!/bin/bash
# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi
if [ ! -z "$TEST_REPO_ARCH" ]; then
    REPO_VERSION=${REPO_VERSION}_${TEST_REPO_ARCH}
fi

pip3 uninstall -y torch
pip3 install torch==2.3.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

DATADIR=/data/inferenceserver/${REPO_VERSION}/qa_model_repository
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=models --log-verbose=1"
SERVER_LOG="./inference_server.log"

CLIENT_PY=./client.py
CLIENT_LOG="./client.log"
EXPECTED_NUM_TESTS="1"
TEST_RESULT_FILE='test_results.txt'

source ../common/util.sh

RET=0

rm -f *.log *.txt

mkdir -p models/libtorch_multi_device/1
mkdir -p models/libtorch_multi_gpu/1
cp models/libtorch_multi_device/config.pbtxt models/libtorch_multi_gpu/.
(cd models/libtorch_multi_gpu && \
    sed -i "s/name: \"libtorch_multi_device\"/name: \"libtorch_multi_gpu\"/" config.pbtxt)

# Generate the models which are partitioned across multiple devices
set +e
python3 gen_models.py >> $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Error when generating models. \n***"
    cat $CLIENT_LOG
    exit 1
fi
set -e

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

export MODEL_NAME='libtorch_multi_device'
python3 $CLIENT_PY >> $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Model $MODEL_NAME FAILED. \n***"
    cat $CLIENT_LOG
    RET=1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi

MESSAGES=("SumModule - INPUT0 device: cpu, INPUT1 device: cpu"
    "DiffModule - INPUT0 device: cuda:3, INPUT1 device: cuda:3")
for MESSAGE in "${MESSAGES[@]}"; do
    if grep -q "$MESSAGE" "$SERVER_LOG"; then
        echo -e "Found \"$MESSAGE\"" >> "$CLIENT_LOG"
    else
        echo -e "Not found \"$MESSAGE\"" >> "$CLIENT_LOG"
        RET=1
    fi
done

export MODEL_NAME='libtorch_multi_gpu'
python3 $CLIENT_PY >> $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Model $MODEL_NAME FAILED. \n***"
    cat $CLIENT_LOG
    RET=1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi

MESSAGES=("SumModule - INPUT0 device: cuda:2, INPUT1 device: cuda:2"
    "DiffModule - INPUT0 device: cuda:0, INPUT1 device: cuda:0")
for MESSAGE in "${MESSAGES[@]}"; do
    if grep -q "$MESSAGE" "$SERVER_LOG"; then
        echo -e "Found \"$MESSAGE\"" >> "$CLIENT_LOG"
    else
        echo -e "Not found \"$MESSAGE\"" >> "$CLIENT_LOG"
        RET=1
    fi
done

set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
