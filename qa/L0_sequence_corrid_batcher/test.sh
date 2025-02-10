#!/bin/bash
# Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
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

TEST_RESULT_FILE='test_results.txt'

CLIENT_LOG="./client.log"
BATCHER_TEST=sequence_corrid_batcher_test.py

DATADIR=/data/inferenceserver/${REPO_VERSION}

SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

RET=0

# Must run on a single device or else the TRITONSERVER_DELAY_SCHEDULER
# can fail when the requests are distributed to multiple devices.
export CUDA_VISIBLE_DEVICES=0

# Setup non-variable-size model repositories. The same models are in each
# repository but they are configured as:
#   models4 - four instances with batch-size 1
rm -fr *.log  models{0,1,2,4} && mkdir models4
for m in \
        $DATADIR/qa_dyna_sequence_model_repository/graphdef_dyna_sequence_int32 \
        $DATADIR/qa_dyna_sequence_model_repository/savedmodel_dyna_sequence_int32 \
        $DATADIR/qa_dyna_sequence_model_repository/plan_dyna_sequence_int32 \
        $DATADIR/qa_dyna_sequence_model_repository/onnx_dyna_sequence_int32 \
        $DATADIR/qa_dyna_sequence_model_repository/libtorch_dyna_sequence_int32; do
    cp -r $m models4/. && \
        (cd models4/$(basename $m) && \
            sed -i -z "s/oldest.*{.*}.*control_input/control_input/" config.pbtxt && \
            sed -i "s/^max_batch_size:.*/max_batch_size: 1/" config.pbtxt && \
            sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 4/" config.pbtxt && \
            sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 4/" config.pbtxt)
done

# Same test work on all models since they all have same total number
# of batch slots.
for model_trial in 4; do
    export NO_BATCHING=1 &&
        [[ "$model_trial" != "0" ]] && export NO_BATCHING=0
    export MODEL_INSTANCES=1 &&
        [[ "$model_trial" != "0" ]] && export MODEL_INSTANCES=$model_trial

    MODEL_DIR=models${model_trial}

    # Tests that require TRITONSERVER_DELAY_SCHEDULER so that the
    # scheduler is delayed and requests can collect in the queue.
    for i in test_skip_batch ; do
        export TRITONSERVER_BACKLOG_DELAY_SCHEDULER=0
        export TRITONSERVER_DELAY_SCHEDULER=12
        SERVER_ARGS="--model-repository=`pwd`/$MODEL_DIR"
        SERVER_LOG="./$i.$MODEL_DIR.server.log"
        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        echo "Test: $i, repository $MODEL_DIR" >>$CLIENT_LOG

        set +e
        python $BATCHER_TEST SequenceCorrIDBatcherTest.$i >>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test $i Failed\n***" >>$CLIENT_LOG
            echo -e "\n***\n*** Test $i Failed\n***"
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

        unset TRITONSERVER_DELAY_SCHEDULER
        unset TRITONSERVER_BACKLOG_DELAY_SCHEDULER
        kill $SERVER_PID
        wait $SERVER_PID
    done
done

# Test correlation ID data type
mkdir -p corrid_data_type/add_sub/1
cp ../python_models/add_sub/model.py corrid_data_type/add_sub/1

for corrid_data_type in TYPE_STRING TYPE_UINT32 TYPE_INT32 TYPE_UINT64 TYPE_INT64; do
    (cd corrid_data_type/add_sub && \
    cp ../../../python_models/add_sub/config.pbtxt . && \
    echo "sequence_batching { \
        control_input [{ \
            name: \"CORRID\" \
            control [{ \
            kind: CONTROL_SEQUENCE_CORRID \
            data_type: $corrid_data_type \
            }]
        }] \
        }" >> config.pbtxt)
    MODEL_DIR=corrid_data_type

    for i in test_corrid_data_type ; do
        export TRITONSERVER_CORRID_DATA_TYPE=$corrid_data_type
        SERVER_ARGS="--model-repository=`pwd`/$MODEL_DIR"
        SERVER_LOG="./$i.$MODEL_DIR.server.log"
        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        echo "Test: $i, repository $MODEL_DIR" >>$CLIENT_LOG

        set +e
        python $BATCHER_TEST SequenceCorrIDBatcherTest.$i >>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test $i Failed\n***" >>$CLIENT_LOG
            echo -e "\n***\n*** Test $i Failed\n***"
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

        unset TRITONSERVER_CORRID_DATA_TYPE
        kill $SERVER_PID
        wait $SERVER_PID
    done
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
