#!/bin/bash
# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

CLIENT_PY=../python_unittest.py
CLIENT_LOG="./request_rescheduling_client.log"
EXPECTED_NUM_TESTS="1"
TEST_RESULT_FILE='test_results.txt'
source ../../common/util.sh

TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
BACKEND_DIR=${TRITON_DIR}/backends

RET=0
# This variable is used to print out the correct server log for each sub-test.
SUB_TEST_RET=0
rm -fr *.log ./models *.txt

# pip3 uninstall -y torch
# pip3 install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html

mkdir -p models/request_rescheduling/1/
cp ../../python_models/request_rescheduling/model.py models/request_rescheduling/1/
cp ../../python_models/request_rescheduling/config.pbtxt models/request_rescheduling

mkdir -p models/request_rescheduling_error/1/
cp ../../python_models/request_rescheduling_error/model.py models/request_rescheduling_error/1/
cp ../../python_models/request_rescheduling_error/config.pbtxt models/request_rescheduling_error

mkdir -p models/request_rescheduling_addsub/1/
cp ../../python_models/request_rescheduling_addsub/model.py models/request_rescheduling_addsub/1/
cp ../../python_models/request_rescheduling_addsub/config.pbtxt models/request_rescheduling_addsub

mkdir -p models/generate_sequence/1/
cp ../../python_models/generate_sequence/model.py models/generate_sequence/1/
cp ../../python_models/generate_sequence/config.pbtxt models/generate_sequence

SERVER_LOG="./request_rescheduling_server.log"
# SERVER_ARGS="--model-repository=`pwd`/models --backend-directory=${BACKEND_DIR} --log-verbose=1 --model-control-mode=explicit --load-model=request_rescheduling"
SERVER_ARGS="--model-repository=`pwd`/models --backend-directory=${BACKEND_DIR} --log-verbose=1"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

export MODEL_NAME='request_rescheduling'

set +e
python3 $CLIENT_PY >> $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** request_rescheduling unit test FAILED. \n***"
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
    cat $SERVER_LOG
    echo -e "\n***\n*** Request Rescheduling test FAILED. \n***"
else
    echo -e "\n***\n*** Request Rescheduling test PASSED. \n***"
fi

exit $RET
