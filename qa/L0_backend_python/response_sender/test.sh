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

source ../../common/util.sh

RET=0

#
# Test response sender under decoupled / non-decoupled
#
rm -rf models && mkdir models
mkdir -p models/response_sender/1 && \
    cp ../../python_models/response_sender/model_common.py models/response_sender/1 && \
    cp ../../python_models/response_sender/model.py models/response_sender/1 && \
    cp ../../python_models/response_sender/config.pbtxt models/response_sender
mkdir -p models/response_sender_decoupled/1 && \
    cp ../../python_models/response_sender/model_common.py models/response_sender_decoupled/1 && \
    cp ../../python_models/response_sender/model.py models/response_sender_decoupled/1 && \
    cp ../../python_models/response_sender/config.pbtxt models/response_sender_decoupled && \
    echo "model_transaction_policy { decoupled: True }" >> models/response_sender_decoupled/config.pbtxt
mkdir -p models/response_sender_async/1 && \
    cp ../../python_models/response_sender/model_common.py models/response_sender_async/1 && \
    cp ../../python_models/response_sender/model_async.py models/response_sender_async/1/model.py && \
    cp ../../python_models/response_sender/config.pbtxt models/response_sender_async
mkdir -p models/response_sender_decoupled_async/1 && \
    cp ../../python_models/response_sender/model_common.py models/response_sender_decoupled_async/1 && \
    cp ../../python_models/response_sender/model_async.py models/response_sender_decoupled_async/1/model.py && \
    cp ../../python_models/response_sender/config.pbtxt models/response_sender_decoupled_async && \
    echo "model_transaction_policy { decoupled: True }" >> models/response_sender_decoupled_async/config.pbtxt
mkdir -p models/response_sender_batching/1 && \
    cp ../../python_models/response_sender/model_common.py models/response_sender_batching/1 && \
    cp ../../python_models/response_sender/model.py models/response_sender_batching/1 && \
    cp ../../python_models/response_sender/config.pbtxt models/response_sender_batching && \
    echo "dynamic_batching { max_queue_delay_microseconds: 500000 }" >> models/response_sender_batching/config.pbtxt
mkdir -p models/response_sender_decoupled_batching/1 && \
    cp ../../python_models/response_sender/model_common.py models/response_sender_decoupled_batching/1 && \
    cp ../../python_models/response_sender/model.py models/response_sender_decoupled_batching/1 && \
    cp ../../python_models/response_sender/config.pbtxt models/response_sender_decoupled_batching && \
    echo "model_transaction_policy { decoupled: True }" >> models/response_sender_decoupled_batching/config.pbtxt && \
    echo "dynamic_batching { max_queue_delay_microseconds: 500000 }" >> models/response_sender_decoupled_batching/config.pbtxt
mkdir -p models/response_sender_async_batching/1 && \
    cp ../../python_models/response_sender/model_common.py models/response_sender_async_batching/1 && \
    cp ../../python_models/response_sender/model_async.py models/response_sender_async_batching/1/model.py && \
    cp ../../python_models/response_sender/config.pbtxt models/response_sender_async_batching && \
    echo "dynamic_batching { max_queue_delay_microseconds: 500000 }" >> models/response_sender_async_batching/config.pbtxt
mkdir -p models/response_sender_decoupled_async_batching/1 && \
    cp ../../python_models/response_sender/model_common.py models/response_sender_decoupled_async_batching/1 && \
    cp ../../python_models/response_sender/model_async.py models/response_sender_decoupled_async_batching/1/model.py && \
    cp ../../python_models/response_sender/config.pbtxt models/response_sender_decoupled_async_batching && \
    echo "model_transaction_policy { decoupled: True }" >> models/response_sender_decoupled_async_batching/config.pbtxt && \
    echo "dynamic_batching { max_queue_delay_microseconds: 500000 }" >> models/response_sender_decoupled_async_batching/config.pbtxt

TEST_LOG="response_sender_test.log"
SERVER_LOG="response_sender_test.server.log"
SERVER_ARGS="--model-repository=${MODELDIR}/response_sender/models --backend-directory=${BACKEND_DIR} --log-verbose=1"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
SERVER_LOG=$SERVER_LOG python3 -m pytest --junitxml=concurrency_test.report.xml response_sender_test.py > $TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** response sender test FAILED\n***"
    cat $TEST_LOG
    RET=1
fi
set -e

kill_server

#
# Test response sender to raise exception on response after complete final flag
#
rm -rf models && mkdir models
mkdir -p models/response_sender_complete_final/1 && \
    cp ../../python_models/response_sender_complete_final/model.py models/response_sender_complete_final/1 && \
    cp ../../python_models/response_sender_complete_final/config.pbtxt models/response_sender_complete_final

TEST_LOG="response_sender_complete_final_test.log"
SERVER_LOG="response_sender_complete_final_test.server.log"
SERVER_ARGS="--model-repository=${MODELDIR}/response_sender/models --backend-directory=${BACKEND_DIR} --log-verbose=1"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
SERVER_LOG=$SERVER_LOG python3 -m pytest --junitxml=concurrency_test.report.xml response_sender_complete_final_test.py > $TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** response sender complete final test FAILED\n***"
    cat $TEST_LOG
    RET=1
fi
set -e

kill_server

#
# Test async response sender under decoupled / non-decoupled
#

# TODO

if [ $RET -eq 1 ]; then
    echo -e "\n***\n*** Response sender test FAILED\n***"
else
    echo -e "\n***\n*** Response sender test Passed\n***"
fi
exit $RET
