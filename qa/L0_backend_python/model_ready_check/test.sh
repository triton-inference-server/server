#!/bin/bash
# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

CLIENT_LOG="./model_ready_check_client.log"
TEST_RESULT_FILE='test_results.txt'
source ../common.sh
source ../../common/util.sh

SERVER_ARGS="--model-repository=${MODELDIR}/model_ready_check/models --backend-directory=${BACKEND_DIR} --log-verbose=1"
SERVER_LOG="./model_ready_check_server.log"

RET=0
rm -fr *.log ./models

mkdir -p models/identity_fp32/1/
cp ../../python_models/identity_fp32/model.py ./models/identity_fp32/1/model.py
cp ../../python_models/identity_fp32/config.pbtxt ./models/identity_fp32/config.pbtxt

#
# Test Model Ready Check (TRITONBACKEND_ModelInstanceReady)
# Test with different signals to simulate various crash/exit scenarios
# 11 (SIGSEGV) - Segmentation fault / crash
# 9  (SIGKILL) - Force kill
for SIGNAL in 11 9; do
    echo -e "\n***\n*** Testing Model Ready Check with Signal $SIGNAL\n***"

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        cat $SERVER_LOG
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        exit 1
    fi

    set +e

    # 1. Verify model is initially ready
    echo "Checking Initial Readiness..."
    python3 -m unittest check_model_ready.ModelReadyTest.test_model_ready
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Model Ready Check Failed (Signal $SIGNAL): Initial readiness check failed \n***"
        RET=1
        kill_server
        exit 1
    fi

    # 2. Find the stub process PID
    stub_pid=$(pgrep -f "triton_python_backend_stub*")

    if [ -z "$stub_pid" ]; then
        echo -e "\n***\n*** Model Ready Check Failed (Signal $SIGNAL): Could not find stub process \n***"
        RET=1
        kill_server
    else
        echo "Found stub process: $stub_pid"

        # 3. Kill the stub process
        echo "Killing stub with signal $SIGNAL..."
        kill -$SIGNAL $stub_pid
        sleep 1

        # 4. Verify model is now NOT ready
        echo "Checking Not Ready Status..."
        python3 -m unittest check_model_ready.ModelReadyTest.test_model_not_ready
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Model Ready Check Failed (Signal $SIGNAL): Model reported ready after kill \n***"
            RET=1
        else
            echo "***\n  Model Ready Check Passed for Signal $SIGNAL"
        fi
    fi

    set -e
    kill_server
done

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET

