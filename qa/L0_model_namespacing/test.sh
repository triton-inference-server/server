#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TRITON_QA_ROOT_DIR=${TRITON_QA_ROOT_DIR:="/opt/tritonserver/qa"}
source $TRITON_QA_ROOT_DIR/common/util.sh

RET=0

TEST_PY=./test.py
# tests are run individually
EXPECTED_NUM_TESTS="1"
TEST_RESULT_FILE='test_results.txt'


export CUDA_VISIBLE_DEVICES=0
export TRITON_QA_ROOT_DIR=$TRITON_QA_ROOT_DIR
export TRITON_QA_PYTHON_MODEL_DIR=$TRITON_QA_ROOT_DIR/L0_model_namespacing

rm -fr *.log

REPO_ARGS="--model-namespacing=true --model-repository=`pwd`/test_dir/addsub_repo --model-repository=`pwd`/test_dir/subadd_repo"
POLL_ARGS="--model-control-mode=POLL --repository-poll-secs=2"
EXPLICIT_ARGS="--model-control-mode=EXPLICIT"

SERVER=/opt/tritonserver/bin/tritonserver

# List all tests as each test will use different repo configuration
TEST_LIST=${TEST_LIST:="test_duplication \
                            test_dynamic_resolution \
                            test_ensemble_duplication \
                            test_no_duplication"}

# Helper to make sure all ensemble have version directory
CURR_DIR=`pwd`
for test_name in $TEST_LIST; do
    for model_dir in $CURR_DIR/$test_name/*/*; do
        mkdir -p $model_dir/1
    done
done

# Set this variable to avoid generation of '__pycache__' in the model directory,
# which will cause unintended model reload in POLLING model as Triton sees
# changes in the model directory
export PYTHONDONTWRITEBYTECODE=1

# Polling
for test_name in $TEST_LIST; do
    TEST_SUITE="ModelNamespacePoll"
    TEST_LOG="`pwd`/test.$TEST_SUITE.$test_name.log"
    SERVER_LOG="./server.$TEST_SUITE.$test_name.log"

    rm -fr `pwd`/test_dir
    cp -r `pwd`/$test_name `pwd`/test_dir
    SERVER_ARGS="$REPO_ARGS $POLL_ARGS"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e
    # Pass in the test directory as the test may modify the structure
    NAMESPACE_TESTING_DIRCTORY=`pwd`/test_dir python $TEST_PY $TEST_SUITE.$test_name >>$TEST_LOG 2>&1
    if [ $? -ne 0 ]; then
        RET=1
        cat $TEST_LOG
    else
        check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
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

# Explicit
for test_name in $TEST_LIST; do
    TEST_SUITE="ModelNamespaceExplicit"
    TEST_LOG="`pwd`/test.$TEST_SUITE.$test_name.log"
    SERVER_LOG="./server.$TEST_SUITE.$test_name.log"

    rm -fr `pwd`/test_dir
    cp -r `pwd`/$test_name `pwd`/test_dir
    SERVER_ARGS="$REPO_ARGS $EXPLICIT_ARGS"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e
    # Pass in the test directory as the test may modify the structure
    NAMESPACE_TESTING_DIRCTORY=`pwd`/test_dir python $TEST_PY $TEST_SUITE.$test_name >>$TEST_LOG 2>&1
    if [ $? -ne 0 ]; then
        RET=1
        cat $TEST_LOG
    else
        check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
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
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
