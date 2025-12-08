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

TEST_RESULT_FILE='test_results.txt'
source ../common.sh
source ../../common/util.sh

SERVER_ARGS="--model-repository=${MODELDIR}/model_readiness/models --backend-directory=${BACKEND_DIR} --log-verbose=1"

RET=0
rm -fr *.log ./models

MODEL_NAME="identity_fp32"
mkdir -p models/$MODEL_NAME/1/
cp ../../python_models/$MODEL_NAME/model.py ./models/$MODEL_NAME/1/model.py
cp ../../python_models/$MODEL_NAME/config.pbtxt ./models/$MODEL_NAME/config.pbtxt

#
# Test Model Ready Check (TRITONBACKEND_ModelInstanceReady)
# Test with different signals to simulate various crash/exit scenarios
# 11 (SIGSEGV) - Segmentation fault / crash
# 9  (SIGKILL) - Force kill
for SIGNAL in 11 9; do
    echo -e "\n***\n*** Testing model_readiness with Signal $SIGNAL\n***"
    SERVER_LOG="./model_readiness_signal_${SIGNAL}_server.log"
    CLIENT_LOG="./model_readiness_${SIGNAL}_client.log"

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        cat $SERVER_LOG
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        exit 1
    fi

    set +e

    # Verify model is initially ready
    echo "Checking Initial Readiness..."
    python3 -m unittest check_model_ready.ModelReadyTest.test_model_ready >> ${CLIENT_LOG} 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test model_readiness Failed (Signal $SIGNAL): Initial readiness check failed \n***"
        RET=1
        kill_server
        exit 1
    fi

    # Find the stub process PID
    stub_pid=$(pgrep -f "triton_python_backend_stub")

    if [ -z "$stub_pid" ]; then
        echo -e "\n***\n*** Test model_readiness Failed (Signal $SIGNAL): Could not find stub process \n***"
        RET=1
        kill_server
    else
        echo "Found stub process: $stub_pid"

        # Kill the stub process
        echo "Killing stub with signal $SIGNAL..."
        kill -$SIGNAL $stub_pid
        sleep 1

        # Verify model is now NOT ready
        echo "Checking Not Ready Status..."
        python3 -m unittest check_model_ready.ModelReadyTest.test_model_not_ready >> ${CLIENT_LOG} 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test model_readiness Failed (Signal $SIGNAL): Model reported ready after kill \n***"
            RET=1
        else
            # Verify correct error message in logs
            # Expect 2 occurrences: HTTP and gRPC checks
            error_count=$(grep -c "Model '${MODEL_NAME}' version 1 is not ready: Stub process '${MODEL_NAME}_0_0' is not healthy." $SERVER_LOG)
            if [ "$error_count" -eq 2 ]; then
                 echo -e "\n***\n Test model_readiness Passed for Signal $SIGNAL \n***"
            else
                 echo -e "\n***\n*** Test model_readiness Failed (Signal $SIGNAL): Expected 2 error messages, found $error_count \n***"
                 cat $SERVER_LOG
                 RET=1
            fi
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

