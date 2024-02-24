#!/bin/bash
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

export CUDA_VISIBLE_DEVICES=0

SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

RET=0

rm -rf models && mkdir models
mkdir -p models/square_int32/1 && (cd models/square_int32 && \
    echo 'backend: "square"' >> config.pbtxt && \
    echo 'max_batch_size: 0' >> config.pbtxt && \
    echo 'model_transaction_policy { decoupled: True }' >> config.pbtxt && \
    echo -e 'input [{ name: "IN" \n data_type: TYPE_INT32 \n dims: [ 1 ] }]' >> config.pbtxt && \
    echo -e 'output [{ name: "OUT" \n data_type: TYPE_INT32 \n dims: [ 1 ] }]' >> config.pbtxt && \
    echo -e 'parameters [{ key: "CUSTOM_INFER_DELAY_NS" \n value: { string_value: "400000000" } }]' >> config.pbtxt && \
    echo -e 'parameters [{ key: "CUSTOM_OUTPUT_DELAY_NS" \n value: { string_value: "200000000" } }]' >> config.pbtxt && \
    echo -e 'parameters [{ key: "CUSTOM_FAIL_COUNT" \n value: { string_value: "2" } }]' >> config.pbtxt && \
    echo -e 'parameters [{ key: "CUSTOM_EMPTY_COUNT" \n value: { string_value: "1" } }]' >> config.pbtxt)
mkdir -p models/square_int32_slow/1 && (cd models/square_int32_slow && \
    echo 'backend: "square"' >> config.pbtxt && \
    echo 'max_batch_size: 0' >> config.pbtxt && \
    echo 'model_transaction_policy { decoupled: True }' >> config.pbtxt && \
    echo -e 'input [{ name: "IN" \n data_type: TYPE_INT32 \n dims: [ 1 ] }]' >> config.pbtxt && \
    echo -e 'output [{ name: "OUT" \n data_type: TYPE_INT32 \n dims: [ 1 ] }]' >> config.pbtxt && \
    echo -e 'parameters [{ key: "CUSTOM_INFER_DELAY_NS" \n value: { string_value: "1200000000" } }]' >> config.pbtxt && \
    echo -e 'parameters [{ key: "CUSTOM_OUTPUT_DELAY_NS" \n value: { string_value: "800000000" } }]' >> config.pbtxt && \
    echo -e 'parameters [{ key: "CUSTOM_CANCEL_DELAY_NS" \n value: { string_value: "400000000" } }]' >> config.pbtxt)

TEST_LOG="response_statistics_test.log"
SERVER_LOG="./response_statistics_test.server.log"

SERVER_ARGS="--model-repository=`pwd`/models"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python response_statistics_test.py > $TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed response statistics test\n***"
    cat $TEST_LOG
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi
exit $RET
