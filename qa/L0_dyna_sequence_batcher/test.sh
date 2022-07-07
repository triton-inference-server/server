#!/bin/bash
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

CLIENT_LOG="./client.log"
BATCHER_TEST=dyna_sequence_batcher_test.py

DATADIR=/data/inferenceserver/${REPO_VERSION}

SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

export CUDA_VISIBLE_DEVICES=0

# If IMPLICIT_STATE not specified, set to 0
IMPLICIT_STATE=${IMPLICIT_STATE:="0"}
export IMPLICIT_STATE

# If BACKENDS not specified, set to all
BACKENDS=${BACKENDS:="graphdef savedmodel libtorch onnx plan custom custom_string"}
export BACKENDS

MODEL_REPOSITORY=''
if [ "$IMPLICIT_STATE" == "1" ]; then
  MODEL_REPOSITORY="qa_dyna_sequence_implicit_model_repository"
else
  MODEL_REPOSITORY="qa_dyna_sequence_model_repository"
fi

RET=0

rm -fr *.log *.serverlog

# models
rm -fr models && mkdir models
for MODEL in ${DATADIR}/$MODEL_REPOSITORY/* ; do
    cp -r $MODEL models/. && \
        (cd models/$(basename $MODEL) && \
            sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 1/" config.pbtxt)
done

# Implicit state models for custom backend do not exist.
if [ $IMPLICIT_STATE == "0" ]; then
    cp -r ../custom_models/custom_dyna_sequence_int32 models/.
    sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 1/" models/custom_dyna_sequence_int32/config.pbtxt
    # Construct custom dyna_sequence_model with STRING sequence ID. Copy model and edit config.pbtxt
    cp -r models/custom_dyna_sequence_int32 models/custom_string_dyna_sequence_int32
    sed -i "s/custom_dyna_sequence_int32/custom_string_dyna_sequence_int32/g" models/custom_string_dyna_sequence_int32/config.pbtxt
    sed -i "/CONTROL_SEQUENCE_CORRID/{n;s/data_type:.*/data_type: TYPE_STRING/}" models/custom_string_dyna_sequence_int32/config.pbtxt
fi

# Implicit state models that support ragged batching do not exist.
if [ $IMPLICIT_STATE == "0" ]; then
    # ragged models
    rm -fr ragged_models && mkdir ragged_models
    cp -r ../custom_models/custom_dyna_sequence_int32 ragged_models/.
    (cd ragged_models/custom_dyna_sequence_int32 && \
            sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 1/" config.pbtxt && \
            sed -i "s/name:.*\"INPUT\"/name: \"INPUT\"\\nallow_ragged_batch: true/" config.pbtxt)
fi

# Need to launch the server for each test so that the model status is
# reset (which is used to make sure the correct batch size was used
# for execution). Test everything with fixed-tensor-size models and
# variable-tensor-size models.
export NO_BATCHING=1
for i in \
        test_simple_sequence \
        test_length1_sequence \
         ; do
    SERVER_LOG="./$i.serverlog"
    SERVER_ARGS="--model-repository=`pwd`/models"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    echo "Test: $i" >>$CLIENT_LOG

    set +e
    python $BATCHER_TEST DynaSequenceBatcherTest.$i >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test $i Failed\n***" >>$CLIENT_LOG
        echo -e "\n***\n*** Test $i Failed\n***"
        RET=1
    fi
    set -e

    kill $SERVER_PID
    wait $SERVER_PID
done

# Tests that require max_queue_delay_microseconds to be non-zero so
# that batching is delayed until a full preferred batch is available.
for m in `ls models`; do
    (cd models/$m && \
            sed -i "s/max_candidate_sequences:.*/max_candidate_sequences:4/" config.pbtxt && \
            sed -i "s/max_queue_delay_microseconds:.*/max_queue_delay_microseconds:5000000/" config.pbtxt)
done

export NO_BATCHING=0
for i in \
        test_multi_sequence_different_shape \
        test_multi_sequence \
        test_multi_parallel_sequence \
        test_backlog \
        test_backlog_fill \
        test_backlog_fill_no_end \
        test_backlog_sequence_timeout \
    ; do

    SERVER_LOG="./$i.serverlog"
    SERVER_ARGS="--model-repository=`pwd`/models"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    echo "Test: $i" >>$CLIENT_LOG

    set +e
    python $BATCHER_TEST DynaSequenceBatcherTest.$i >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test $i Failed\n***" >>$CLIENT_LOG
        echo -e "\n***\n*** Test $i Failed\n***"
        RET=1
    fi
    set -e

    kill $SERVER_PID
    wait $SERVER_PID
done

if [ $IMPLICIT_STATE == "0" ]; then
    # Ragged-batch tests that require max_queue_delay_microseconds to be
    # non-zero so that batching is delayed until a full preferred batch is
    # available.
    for m in `ls ragged_models`; do
        (cd ragged_models/$m && \
                sed -i "s/max_candidate_sequences:.*/max_candidate_sequences:4/" config.pbtxt && \
                sed -i "s/max_queue_delay_microseconds:.*/max_queue_delay_microseconds:5000000/" config.pbtxt)
    done

    export NO_BATCHING=0
    for i in \
        test_multi_sequence_different_shape_allow_ragged \
        ; do

        SERVER_LOG="./$i.serverlog"
        SERVER_ARGS="--model-repository=`pwd`/ragged_models"
        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        echo "Test: $i" >>$CLIENT_LOG

        set +e
        python $BATCHER_TEST DynaSequenceBatcherTest.$i >>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test $i Failed\n***" >>$CLIENT_LOG
            echo -e "\n***\n*** Test $i Failed\n***"
            RET=1
        fi
        set -e

        kill $SERVER_PID
        wait $SERVER_PID
    done
fi

# python unittest seems to swallow ImportError and still return 0 exit
# code. So need to explicitly check CLIENT_LOG to make sure we see
# some running tests
grep -c "HTTPSocketPoolResponse status=200" $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed To Run\n***"
    RET=1
fi

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
