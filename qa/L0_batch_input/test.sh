#!/bin/bash
# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
BATCH_INPUT_TEST=batch_input_test.py
EXPECTED_NUM_TESTS="8"

DATADIR=/data/inferenceserver/${REPO_VERSION}/qa_ragged_model_repository
IDENTITY_DATADIR=/data/inferenceserver/${REPO_VERSION}/qa_identity_model_repository

TEST_RESULT_FILE='test_results.txt'
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=models --exit-timeout-secs=120"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

# If BACKENDS not specified, set to all
BACKENDS=${BACKENDS:="onnx savedmodel plan"}

rm -f $SERVER_LOG $CLIENT_LOG

RET=0
for BACKEND in $BACKENDS; do
    rm -rf models && mkdir models
    cp -r $DATADIR/${BACKEND}_batch_input models/ragged_element_count_acc_zero
    (cd models/ragged_element_count_acc_zero && \
          sed -i "s/${BACKEND}_batch_input/ragged_element_count_acc_zero/" config.pbtxt)
    cp -r $DATADIR/${BACKEND}_batch_input models/ragged_acc_shape
    (cd models/ragged_acc_shape && \
          sed -i "s/${BACKEND}_batch_input/ragged_acc_shape/" config.pbtxt && \
          sed -i "s/BATCH_ELEMENT_COUNT/BATCH_ACCUMULATED_ELEMENT_COUNT/" config.pbtxt && \
          sed -i "s/BATCH_ACCUMULATED_ELEMENT_COUNT_WITH_ZERO/BATCH_MAX_ELEMENT_COUNT_AS_SHAPE/" config.pbtxt)
    cp -r $DATADIR/${BACKEND}_batch_input models/batch_item_flatten
    (cd models/batch_item_flatten && \
          sed -i "s/${BACKEND}_batch_input/batch_item_flatten/" config.pbtxt && \
          sed -i "0,/-1/{s/-1/-1, -1/}" config.pbtxt && \
          sed -i "s/BATCH_ELEMENT_COUNT/BATCH_ACCUMULATED_ELEMENT_COUNT/" config.pbtxt && \
          sed -i "s/BATCH_ACCUMULATED_ELEMENT_COUNT_WITH_ZERO/BATCH_ITEM_SHAPE_FLATTEN/" config.pbtxt)
    cp -r $DATADIR/${BACKEND}_batch_item models/batch_item
    (cd models/batch_item && \
          sed -i "s/${BACKEND}_batch_item/batch_item/" config.pbtxt)
    # Use nobatch model to showcase ragged input, identity model to verify
    # batch input is generated properly
    cp -r $IDENTITY_DATADIR/${BACKEND}_nobatch_zero_1_float32 models/ragged_io
    (cd models/ragged_io && \
          sed -i "s/${BACKEND}_nobatch_zero_1_float32/ragged_io/" config.pbtxt && \
          sed -i "s/^max_batch_size:.*/max_batch_size: 4/" config.pbtxt && \
          sed -i "s/name: \"INPUT0\"/name: \"INPUT0\"\\nallow_ragged_batch: true/" config.pbtxt && \
          echo "batch_output [{target_name: \"OUTPUT0\" \
                                 kind: BATCH_SCATTER_WITH_INPUT_SHAPE \
                                 source_input: \"INPUT0\" }] \
                dynamic_batching { max_queue_delay_microseconds: 1000000 }" >> config.pbtxt)


    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e
    python $BATCH_INPUT_TEST >$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
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
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
