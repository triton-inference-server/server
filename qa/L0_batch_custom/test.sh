#!/bin/bash
# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

## This test tests the ability to use custom batching strategies with models.

## Clone git core repo
## 
## Run each of the models
## Confirm they run
## See what outputs...
## Confirm batch output is as expected

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

export CUDA_VISIBLE_DEVICES=0

CLIENT_LOG="./client.log"
BATCH_INPUT_TEST=batch_input_test.py
EXPECTED_NUM_TESTS="1"

DATADIR=/data/inferenceserver/${REPO_VERSION}/qa_model_repository
IDENTITY_DATADIR=/data/inferenceserver/${REPO_VERSION}/qa_identity_model_repository

TEST_RESULT_FILE='test_results.txt'
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=models --exit-timeout-secs=120"
SERVER_LOG="./inference_server.log"
source ../common/util.sh
RET=0

# TODO: First, only do this for one model with it loaded in the version folder
# Add to model config: batching library path (for relevant tests), max_batch_volume
# Test cases: batching via config, version folder, model folder, backend folder
# Try with number of requests in perf_analyzer less than max_batch_volume and more than
# Grep for loading library, then executing n requests... not executing 1 requests
# Alternatively, could use L0_batch_input approach with .py file

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

export CUDA_VISIBLE_DEVICES=0

CLIENT_LOG="./client.log"
BATCH_CUSTOM_TEST=batch_custom_test.py
EXPECTED_NUM_TESTS="1"

DATADIR=/data/inferenceserver/${REPO_VERSION}/qa_identity_model_repository

TEST_RESULT_FILE='test_results.txt'
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=models --exit-timeout-secs=120 --log-verbose 1"
SERVER_LOG="./inference_server.log"
MODEL_NAME="onnx_zero_1_float16"
source ../common/util.sh

rm -f $SERVER_LOG $CLIENT_LOG
rm -rf models && mkdir models
cp -r $DATADIR/$MODEL_NAME models

# TODO: Generate or change script to copy from elsewhere
CONFIG_PATH="models/${MODEL_NAME}/config.pbtxt"
cp -r /git/core/examples/batch_strategy/volume_batching/build/libtriton_volumebatching.so models/$MODEL_NAME
echo "parameters: {key: \"TRITON_BATCH_STRATEGY_PATH\", value: {string_value: \"models/${MODEL_NAME}/libtriton_volumebatching.so\"}}" >> ${CONFIG_PATH}
echo "parameters { key: \"MAX_BATCH_VOLUME_BYTES\" value: {string_value: \"96\"}}" >> ${CONFIG_PATH}
echo "dynamic_batching { max_queue_delay_microseconds: 10000}" >> ${CONFIG_PATH}

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $BATCH_CUSTOM_TEST >$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
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


## Afterwards:
## 1. Update Docker script to move shared library files to this container
## 2. Move test into L0_batcher?