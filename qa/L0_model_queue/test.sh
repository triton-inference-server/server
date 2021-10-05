#!/bin/bash
# Copyright 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
TEST_RESULT_FILE='test_results.txt'
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

CLIENT_LOG="./client.log"
MODEL_QUEUE_TEST=model_queue_test.py

DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver

SERVER_ARGS="--model-repository=`pwd`/models"

source ../common/util.sh

RET=0

# Must run on a single device or else the TRITONSERVER_DELAY_SCHEDULER
# can fail when the requests are distributed to multiple devices.
export CUDA_VISIBLE_DEVICES=0

# Prepare base model. Only test with custom backend as it is sufficient
rm -fr *.log *.serverlog models custom_zero_1_float32
cp -r ../custom_models/custom_zero_1_float32 . && \
    mkdir -p ./custom_zero_1_float32/1 && \
    mkdir -p ./ensemble_zero_1_float32/1

(cd custom_zero_1_float32 && \
        sed -i "s/dims:.*\[.*\]/dims: \[ -1 \]/g" config.pbtxt && \
        sed -i "s/max_batch_size:.*/max_batch_size: 32/g" config.pbtxt && \
        echo "instance_group [ { kind: KIND_CPU count: 1 }]" >> config.pbtxt)

# test_max_queue_size
# For testing max queue size, we use delay in the custom model to
# create backlogs, "TRITONSERVER_DELAY_SCHEDULER" is not desired as queue size
# is capped by max queue size.
rm -fr models && mkdir models && \
    cp -r ensemble_zero_1_float32 models/. && \
    cp -r custom_zero_1_float32 models/. && \
    (cd models/custom_zero_1_float32 && \
        echo "dynamic_batching { " >> config.pbtxt && \
        echo "    preferred_batch_size: [ 4, 8 ]" >> config.pbtxt && \
        echo "    default_queue_policy {" >> config.pbtxt && \
        echo "        max_queue_size: 8" >> config.pbtxt && \
        echo "    }" >> config.pbtxt && \
        echo "}" >> config.pbtxt && \
        echo "parameters [" >> config.pbtxt && \
        echo "{ key: \"execute_delay_ms\"; value: { string_value: \"1000\" }}" >> config.pbtxt && \
        echo "]" >> config.pbtxt)

TEST_CASE=test_max_queue_size
SERVER_LOG="./$TEST_CASE.serverlog"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

echo "Test: $TEST_CASE" >>$CLIENT_LOG

set +e
python $MODEL_QUEUE_TEST ModelQueueTest.$TEST_CASE >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $TEST_RESULT_FILE 1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# test_policy_delay
rm -fr models && mkdir models && \
    cp -r ensemble_zero_1_float32 models/. && \
    cp -r custom_zero_1_float32 models/. && \
    (cd models/custom_zero_1_float32 && \
        echo "dynamic_batching { " >> config.pbtxt && \
        echo "    preferred_batch_size: [ 4, 8 ]" >> config.pbtxt && \
        echo "    max_queue_delay_microseconds: 10000000" >> config.pbtxt && \
        echo "    default_queue_policy {" >> config.pbtxt && \
        echo "        timeout_action: DELAY" >> config.pbtxt && \
        echo "        default_timeout_microseconds: 100000" >> config.pbtxt && \
        echo "    }" >> config.pbtxt && \
        echo "}" >> config.pbtxt)

TEST_CASE=test_policy_delay
SERVER_LOG="./$TEST_CASE.serverlog"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

echo "Test: $TEST_CASE" >>$CLIENT_LOG

set +e
python $MODEL_QUEUE_TEST ModelQueueTest.$TEST_CASE >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $TEST_RESULT_FILE 1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# test_policy_reject
rm -fr models && mkdir models && \
    cp -r ensemble_zero_1_float32 models/. && \
    cp -r custom_zero_1_float32 models/. && \
    (cd models/custom_zero_1_float32 && \
        echo "dynamic_batching { " >> config.pbtxt && \
        echo "    preferred_batch_size: [ 4, 8 ]" >> config.pbtxt && \
        echo "    max_queue_delay_microseconds: 10000000" >> config.pbtxt && \
        echo "    default_queue_policy {" >> config.pbtxt && \
        echo "        default_timeout_microseconds: 100000" >> config.pbtxt && \
        echo "    }" >> config.pbtxt && \
        echo "}" >> config.pbtxt)

TEST_CASE=test_policy_reject
SERVER_LOG="./$TEST_CASE.serverlog"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

echo "Test: $TEST_CASE" >>$CLIENT_LOG

set +e
python $MODEL_QUEUE_TEST ModelQueueTest.$TEST_CASE >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $TEST_RESULT_FILE 1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# test_timeout_override
rm -fr models && mkdir models && \
    cp -r ensemble_zero_1_float32 models/. && \
    cp -r custom_zero_1_float32 models/. && \
    (cd models/custom_zero_1_float32 && \
        echo "dynamic_batching { " >> config.pbtxt && \
        echo "    preferred_batch_size: [ 4, 8 ]" >> config.pbtxt && \
        echo "    max_queue_delay_microseconds: 10000000" >> config.pbtxt && \
        echo "    default_queue_policy {" >> config.pbtxt && \
        echo "        allow_timeout_override: true" >> config.pbtxt && \
        echo "        default_timeout_microseconds: 1000000" >> config.pbtxt && \
        echo "    }" >> config.pbtxt && \
        echo "}" >> config.pbtxt)

TEST_CASE=test_timeout_override
SERVER_LOG="./$TEST_CASE.serverlog"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

echo "Test: $TEST_CASE" >>$CLIENT_LOG

set +e
python $MODEL_QUEUE_TEST ModelQueueTest.$TEST_CASE >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $TEST_RESULT_FILE 1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# test_priority_levels
rm -fr models && mkdir models && \
    cp -r ensemble_zero_1_float32 models/. && \
    cp -r custom_zero_1_float32 models/. && \
    (cd models/custom_zero_1_float32 && \
        echo "dynamic_batching { " >> config.pbtxt && \
        echo "    preferred_batch_size: [ 4, 8 ]" >> config.pbtxt && \
        echo "    max_queue_delay_microseconds: 10000000" >> config.pbtxt && \
        echo "    priority_levels: 2" >> config.pbtxt && \
        echo "    default_priority_level: 2" >> config.pbtxt && \
        echo "}" >> config.pbtxt)

TEST_CASE=test_priority_levels
SERVER_LOG="./$TEST_CASE.serverlog"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

echo "Test: $TEST_CASE" >>$CLIENT_LOG

set +e
python $MODEL_QUEUE_TEST ModelQueueTest.$TEST_CASE >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $TEST_RESULT_FILE 1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# test_priority_with_policy
# 2 levels and 2 policies:
#     priority 1: delay
#     priority 2: reject
rm -fr models && mkdir models && \
    cp -r ensemble_zero_1_float32 models/. && \
    cp -r custom_zero_1_float32 models/. && \
    (cd models/custom_zero_1_float32 && \
        echo "dynamic_batching { " >> config.pbtxt && \
        echo "    preferred_batch_size: [ 4, 8, 32 ]" >> config.pbtxt && \
        echo "    max_queue_delay_microseconds: 10000000" >> config.pbtxt && \
        echo "    priority_levels: 2" >> config.pbtxt && \
        echo "    default_priority_level: 2" >> config.pbtxt && \
        echo "    default_queue_policy {" >> config.pbtxt && \
        echo "        timeout_action: DELAY" >> config.pbtxt && \
        echo "        allow_timeout_override: true" >> config.pbtxt && \
        echo "        default_timeout_microseconds: 11000000" >> config.pbtxt && \
        echo "    }" >> config.pbtxt && \
        echo "    priority_queue_policy {" >> config.pbtxt && \
        echo "        key: 2" >> config.pbtxt && \
        echo "        value: {" >> config.pbtxt && \
        echo "            timeout_action: REJECT" >> config.pbtxt && \
        echo "            allow_timeout_override: true" >> config.pbtxt && \
        echo "            default_timeout_microseconds: 11000000" >> config.pbtxt && \
        echo "        }" >> config.pbtxt && \
        echo "    }" >> config.pbtxt && \
        echo "}" >> config.pbtxt)

TEST_CASE=test_priority_with_policy
SERVER_LOG="./$TEST_CASE.serverlog"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

echo "Test: $TEST_CASE" >>$CLIENT_LOG

set +e
python $MODEL_QUEUE_TEST ModelQueueTest.$TEST_CASE >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $TEST_RESULT_FILE 1
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
