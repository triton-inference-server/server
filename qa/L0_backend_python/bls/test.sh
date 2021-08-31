#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1"
CLIENT_PY=../python_unittest.py
CLIENT_LOG="./client.log"
EXPECTED_NUM_TESTS="1"
TEST_RESULT_FILE='test_results.txt'
SERVER_LOG="./inference_server.log"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}

RET=0
rm -fr *.log ./models

source ../../common/util.sh

mkdir -p models/bls/1/
cp ../../python_models/bls/model.py models/bls/1/
cp ../../python_models/bls/config.pbtxt models/bls

mkdir -p models/bls_async/1/
cp ../../python_models/bls_async/model.py models/bls_async/1/
cp ../../python_models/bls_async/config.pbtxt models/bls_async

mkdir -p models/bls_memory/1/
cp ../../python_models/bls_memory/model.py models/bls_memory/1/
cp ../../python_models/bls_memory/config.pbtxt models/bls_memory

mkdir -p models/bls_memory_async/1/
cp ../../python_models/bls_memory_async/model.py models/bls_memory_async/1/
cp ../../python_models/bls_memory_async/config.pbtxt models/bls_memory_async

mkdir -p models/add_sub/1/
cp ../../python_models/add_sub/model.py models/add_sub/1/
cp ../../python_models/add_sub/config.pbtxt models/add_sub

mkdir -p models/execute_error/1/
cp ../../python_models/execute_error/model.py models/execute_error/1/
cp ../../python_models/execute_error/config.pbtxt models/execute_error

mkdir -p models/identity_fp32/1/
cp ../../python_models/identity_fp32/model.py models/identity_fp32/1/
cp ../../python_models/identity_fp32/config.pbtxt models/identity_fp32

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

export MODEL_NAME='bls'
python3 $CLIENT_PY >> $CLIENT_LOG 2>&1 
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** 'bls' test FAILED. \n***"
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

export MODEL_NAME='bls_memory'
python3 $CLIENT_PY > $CLIENT_LOG 2>&1 
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** 'bls_memory' test FAILED. \n***"
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

export MODEL_NAME='bls_memory_async'
python3 $CLIENT_PY >> $CLIENT_LOG 2>&1 
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** 'bls_async_memory' test FAILED. \n***"
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

export MODEL_NAME='bls_async'
python3 $CLIENT_PY >> $CLIENT_LOG 2>&1 
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** 'bls_async' test FAILED. \n***"
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
set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 1 ]; then
    cat $CLIENT_LOG
    cat $SERVER_LOG
    echo -e "\n***\n*** BLS test FAILED. \n***"
else
    echo -e "\n***\n*** BLS test PASSED. \n***"
fi

exit $RET
