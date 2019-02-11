#!/bin/bash
# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

CLIENT_LOG="./client.log"
BATCHER_TEST=sequence_batcher_test.py

DATADIR=/data/inferenceserver

SERVER=/opt/tensorrtserver/bin/trtserver
source ../common/util.sh

RET=0

# Setup non-variable-size model stores. The same models are in each
# store but they are configured as:
#   models1 - one instance with batch-size 4
#   models2 - two instances with batch-size 2
#   models4 - four instances with batch-size 1
rm -fr *.log *.serverlog models{1,2,4} && mkdir models{1,2,4}
for m in \
        ../custom_models/custom_sequence_int32 ; do
    cp -r $m models1/. && \
        (cd models1/$(basename $m) && \
            sed -i "s/max_queue_delay_microseconds:.*/max_queue_delay_microseconds: 10000000/" config.pbtxt && \
            sed -i "s/^max_batch_size:.*/max_batch_size: 4/" config.pbtxt && \
            sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 1/" config.pbtxt)
    cp -r $m models2/. && \
        (cd models2/$(basename $m) && \
            sed -i "s/max_queue_delay_microseconds:.*/max_queue_delay_microseconds: 10000000/" config.pbtxt && \
            sed -i "s/^max_batch_size:.*/max_batch_size: 2/" config.pbtxt && \
            sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 2/" config.pbtxt)
    cp -r $m models4/. && \
        (cd models4/$(basename $m) && \
            sed -i "s/max_queue_delay_microseconds:.*/max_queue_delay_microseconds: 10000000/" config.pbtxt && \
            sed -i "s/^max_batch_size:.*/max_batch_size: 1/" config.pbtxt && \
            sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 4/" config.pbtxt)
done

# Same test work on all models since they all have same total number
# of batch slots.
for model_instances in 1 2 4; do
    export MODEL_INSTANCES=$model_instances
    MODEL_DIR=models${model_instances}

    # Need to launch the server for each test so that the model status is
    # reset (which is used to make sure the correctly batch size was used
    # for execution). Test everything with fixed-tensor-size models and
    # variable-tensor-size models.
    for model_type in FIXED; do    # add VARIABLE
        export BATCHER_TYPE=$model_type
        MODEL_PATH=$MODEL_DIR && [[ "$model_type" == "VARIABLE" ]] && MODEL_PATH=var_models
        for i in \
                test_simple_sequence \
                test_length1_sequence \
                test_batch_size \
                test_no_sequence_start \
                test_no_sequence_start2 \
                test_no_sequence_end \
                test_no_correlation_id ; do
            SERVER_ARGS="--model-store=`pwd`/$MODEL_PATH"
            SERVER_LOG="./$i.$MODEL_DIR.$model_type.serverlog"
            run_server
            if [ "$SERVER_PID" == "0" ]; then
                echo -e "\n***\n*** Failed to start $SERVER\n***"
                cat $SERVER_LOG
                exit 1
            fi

            echo "Test: $i" >>$CLIENT_LOG

            set +e
            python $BATCHER_TEST SequenceBatcherTest.$i >>$CLIENT_LOG 2>&1
            if [ $? -ne 0 ]; then
                echo -e "\n***\n*** Test Failed\n***"
                RET=1
            fi
            set -e

            kill $SERVER_PID
            wait $SERVER_PID
        done

        # Tests that require TRTSERVER_DELAY_SCHEDULER so that the
        # scheduler is delayed and requests can collect in the queue.
        for i in \
                test_backlog_fill \
                test_backlog_fill_no_end \
                test_backlog_same_correlation_id \
                test_backlog_same_correlation_id_no_end \
                test_half_batch \
                test_skip_batch \
                test_full_batch \
                test_backlog ; do
            export TRTSERVER_BACKLOG_DELAY_SCHEDULER=3 &&
                [[ "$i" != "test_backlog_fill_no_end" ]] && export TRTSERVER_BACKLOG_DELAY_SCHEDULER=2 &&
                [[ "$i" != "test_backlog_fill" ]] &&
                [[ "$i" != "test_backlog_same_correlation_id" ]] && export TRTSERVER_BACKLOG_DELAY_SCHEDULER=0
            export TRTSERVER_DELAY_SCHEDULER=10 &&
                [[ "$i" != "test_backlog_fill_no_end" ]] &&
                [[ "$i" != "test_backlog_fill" ]] && export TRTSERVER_DELAY_SCHEDULER=16 &&
                [[ "$i" != "test_backlog_same_correlation_id_no_end" ]] && export TRTSERVER_DELAY_SCHEDULER=8 &&
                [[ "$i" != "test_half_batch" ]] && export TRTSERVER_DELAY_SCHEDULER=12
            SERVER_ARGS="--model-store=`pwd`/$MODEL_PATH"
            SERVER_LOG="./$i.$MODEL_DIR.$model_type.serverlog"
            run_server
            if [ "$SERVER_PID" == "0" ]; then
                echo -e "\n***\n*** Failed to start $SERVER\n***"
                cat $SERVER_LOG
                exit 1
            fi

            echo "Test: $i" >>$CLIENT_LOG

            set +e
            python $BATCHER_TEST SequenceBatcherTest.$i >>$CLIENT_LOG 2>&1
            if [ $? -ne 0 ]; then
                echo -e "\n***\n*** Test Failed\n***"
                RET=1
            fi
            set -e

            unset TRTSERVER_DELAY_SCHEDULER
            unset TRTSERVER_BACKLOG_DELAY_SCHEDULER
            kill $SERVER_PID
            wait $SERVER_PID
        done
    done
done

# python unittest seems to swallow ImportError and still return 0 exit
# code. So need to explicitly check CLIENT_LOG to make sure we see
# some running tests
grep -c "HTTP/1.1 200 OK" $CLIENT_LOG
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
