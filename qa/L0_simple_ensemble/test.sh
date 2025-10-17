#!/bin/bash
# Copyright 2019-2025, NVIDIA CORPORATION. All rights reserved.
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

SIMPLE_TEST_PY=./ensemble_test.py

CLIENT_LOG="./client.log"

TEST_MODEL_DIR="`pwd`/models"
TEST_RESULT_FILE='test_results.txt'
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=${TEST_MODEL_DIR}"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

# ensure ensemble models have version sub-directory
mkdir -p ${TEST_MODEL_DIR}/ensemble_add_sub_int32_int32_int32/1
mkdir -p ${TEST_MODEL_DIR}/ensemble_partial_add_sub/1

rm -f $CLIENT_LOG $SERVER_LOG

# Run ensemble model with all outputs requested
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

RET=0

set +e
python $SIMPLE_TEST_PY EnsembleTest.test_ensemble_add_sub >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
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

# Run ensemble model with sequence flags and verify response sequence
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $SIMPLE_TEST_PY EnsembleTest.test_ensemble_sequence_flags >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
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

# Run ensemble model with only one output requested
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $SIMPLE_TEST_PY EnsembleTest.test_ensemble_add_sub_one_output >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
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

# Run partial ensemble model with all outputs requested
SERVER_ARGS="$SERVER_ARGS --log-verbose=1"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
SERVER_LOG=$SERVER_LOG python $SIMPLE_TEST_PY EnsembleTest.test_ensemble_partial_add_sub >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
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

######## Test ensemble backpressure feature (max_inflight_responses parameter)
MODEL_DIR="`pwd`/backpressure_test_models"
mkdir -p ${MODEL_DIR}/ensemble_disabled_max_inflight_responses/1

rm -rf ${MODEL_DIR}/slow_consumer
mkdir -p ${MODEL_DIR}/slow_consumer/1
cp ../python_models/ground_truth/model.py ${MODEL_DIR}/slow_consumer/1
cp ../python_models/ground_truth/config.pbtxt ${MODEL_DIR}/slow_consumer/
sed -i 's/name: "ground_truth"/name: "slow_consumer"/g' ${MODEL_DIR}/slow_consumer/config.pbtxt

# Create ensemble with "max_inflight_responses = 4"
rm -rf ${MODEL_DIR}/ensemble_enabled_max_inflight_responses
mkdir -p ${MODEL_DIR}/ensemble_enabled_max_inflight_responses/1
cp ${MODEL_DIR}/ensemble_disabled_max_inflight_responses/config.pbtxt ${MODEL_DIR}/ensemble_enabled_max_inflight_responses/
sed -i 's/ensemble_scheduling {/ensemble_scheduling {\n  max_inflight_responses: 4/g' \
  ${MODEL_DIR}/ensemble_enabled_max_inflight_responses/config.pbtxt

BACKPRESSURE_TEST_PY=./ensemble_backpressure_test.py
SERVER_LOG="./ensemble_backpressure_test_server.log"
CLIENT_LOG="./ensemble_backpressure_test_client.log"
rm -f $SERVER_LOG $CLIENT_LOG

SERVER_ARGS="--model-repository=${MODEL_DIR}"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $BACKPRESSURE_TEST_PY -v >> $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    RET=1
else
    check_test_results $TEST_RESULT_FILE 4
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

set +e
# Verify valid config was loaded successfully
if ! grep -q "Ensemble model 'ensemble_enabled_max_inflight_responses' configured with max_inflight_responses: 4" $SERVER_LOG; then
    echo -e "\n***\n*** FAILED: Valid model did not load successfully\n***"
    RET=1
fi
set -e


######## Test invalid value for "max_inflight_responses"
INVALID_PARAM_MODEL_DIR="`pwd`/invalid_param_test_models"
SERVER_ARGS="--model-repository=${INVALID_PARAM_MODEL_DIR}"
SERVER_LOG="./invalid_max_inflight_responses_server.log"
rm -rf $SERVER_LOG ${INVALID_PARAM_MODEL_DIR}

mkdir -p ${INVALID_PARAM_MODEL_DIR}/ensemble_invalid_negative_limit/1
mkdir -p ${INVALID_PARAM_MODEL_DIR}/ensemble_invalid_string_limit/1
mkdir -p ${INVALID_PARAM_MODEL_DIR}/ensemble_invalid_large_value_limit/1
cp -r ${MODEL_DIR}/decoupled_producer ${MODEL_DIR}/slow_consumer ${INVALID_PARAM_MODEL_DIR}/

# max_inflight_responses = -5
cp ${MODEL_DIR}/ensemble_disabled_max_inflight_responses/config.pbtxt ${INVALID_PARAM_MODEL_DIR}/ensemble_invalid_negative_limit/
sed -i 's/ensemble_scheduling {/ensemble_scheduling {\n  max_inflight_responses: -5/g' \
  ${INVALID_PARAM_MODEL_DIR}/ensemble_invalid_negative_limit/config.pbtxt

# max_inflight_responses = "invalid_value"
cp ${MODEL_DIR}/ensemble_disabled_max_inflight_responses/config.pbtxt ${INVALID_PARAM_MODEL_DIR}/ensemble_invalid_string_limit/
sed -i 's/ensemble_scheduling {/ensemble_scheduling {\n  max_inflight_responses: "invalid_value"/g' \
  ${INVALID_PARAM_MODEL_DIR}/ensemble_invalid_string_limit/config.pbtxt

# max_inflight_responses = 12345678901
cp ${MODEL_DIR}/ensemble_disabled_max_inflight_responses/config.pbtxt ${INVALID_PARAM_MODEL_DIR}/ensemble_invalid_large_value_limit/
sed -i 's/ensemble_scheduling {/ensemble_scheduling {\n  max_inflight_responses: 12345678901/g' \
  ${INVALID_PARAM_MODEL_DIR}/ensemble_invalid_large_value_limit/config.pbtxt


run_server
if [ "$SERVER_PID" != "0" ]; then
    echo -e "\n***\n*** FAILED: unexpected success starting $SERVER\n***"
    kill $SERVER_PID
    wait $SERVER_PID
    cat $SERVER_LOG
    RET=1
fi

set +e
# Verify negative value caused model load failure
if ! grep -q "Expected integer, got: -" $SERVER_LOG; then
    echo -e "\n***\n*** FAILED: Negative value should fail model load\n***"
    RET=1
fi

# Verify invalid string caused model load failure
if ! grep -q 'Expected integer, got: "invalid_value"' $SERVER_LOG; then
    echo -e "\n***\n*** FAILED: Invalid string should fail model load\n***"
    RET=1
fi

# Verify very large value caused model load failure
if ! grep -q "Integer out of range (12345678901)" $SERVER_LOG; then
    echo -e "\n***\n*** FAILED: Large value should fail model load\n***"
    RET=1
fi
set -e


if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
