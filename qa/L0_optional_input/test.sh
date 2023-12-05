#!/bin/bash
# Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

export CUDA_VISIBLE_DEVICES=0

TEST_PY=./optional_input_test.py
TEST_LOG="./test.log"
TEST_RESULT_FILE='test_results.txt'

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

rm -fr *.log

mkdir -p ./models/identity_2_float32/1
mkdir -p ./models/ensemble_identity_2_float32/1
mkdir -p ./models/pipeline_identity_2_float32/1
mkdir -p ./models/optional_connecting_tensor/1

# Basic test cases
TEST_CASES=${TEST_CASES:="test_all_inputs \
                            test_optional_same_input \
                            test_optional_mix_inputs \
                            test_optional_mix_inputs_2 \
                            test_ensemble_all_inputs \
                            test_ensemble_optional_same_input \
                            test_ensemble_optional_mix_inputs \
                            test_ensemble_optional_mix_inputs_2 \
                            test_ensemble_optional_pipeline \
                            test_ensemble_optional_connecting_tensor"}
RET=0
for i in $TEST_CASES ; do
    # Restart server for every test to clear model stats
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    echo "Test: $i" >>$TEST_LOG

    set +e
    python $TEST_PY OptionalInputTest.$i >>$TEST_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    else
        check_test_results $TEST_RESULT_FILE 1
        if [ $? -ne 0 ]; then
            cat $TEST_LOG
            echo -e "\n***\n*** Test Result Verification Failed\n***"
            RET=1
        fi
    fi
    set -e

    kill $SERVER_PID
    wait $SERVER_PID
done

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
    cat $SERVER_LOG
    cat $TEST_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
