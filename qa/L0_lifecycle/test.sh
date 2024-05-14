#!/bin/bash
# Copyright 2018-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

CLIENT_LOG="./client.log"
DATADIR=/data/inferenceserver/${REPO_VERSION}
LC_TEST=lifecycle_test.py
SLEEP_TIME=10
SERVER=/opt/tritonserver/bin/tritonserver
TEST_RESULT_FILE='test_results.txt'
source ../common/util.sh

function check_unit_test() {
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
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
}

# corresponds to r23.06
pip install tritonclient[all]==2.35.0

RET=0
rm -fr *.log

LOG_IDX=0

if [ `ps | grep -c "tritonserver"` != "0" ]; then
    echo -e "Tritonserver already running"ls 
    echo -e `ps | grep tritonserver`
    exit 1
fi

# LifeCycleTest.test_file_override
rm -fr models config.pbtxt.*
mkdir models
cp -r $DATADIR/qa_model_repository/onnx_float32_float32_float32 models/.
# Make only version 2, 3 is valid version directory while config requests 1, 3
rm -rf models/onnx_float32_float32_float32/1

# Start with EXPLICIT mode and load onnx_float32_float32_float32
SERVER_ARGS="--model-repository=`pwd`/models \
             --model-control-mode=explicit \
             --load-model=onnx_float32_float32_float32 \
             --strict-model-config=false"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_file_override >>$CLIENT_LOG 2>&1
check_unit_test
python $LC_TEST LifeCycleTest.test_file_override_security >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
fi

exit $RET
