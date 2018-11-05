#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
BATCHER_TEST=batcher_test.py

DATADIR=/data/inferenceserver

SERVER=/opt/tensorrtserver/bin/trtserver
source ../common/util.sh

RET=0

# Setup model store
rm -fr *.log *.serverlog models && mkdir models
for m in \
        graphdef_float32_float32_float32 \
        netdef_float32_float32_float32 \
        plan_float32_float32_float32 ; do
    cp -r $DATADIR/qa_model_repository/$m models/. &&
        (cd models/$m && \
                sed -i "s/^max_batch_size:.*/max_batch_size: 8/" config.pbtxt && \
                echo "dynamic_batching { preferred_batch_size: [ 2, 6 ], max_queue_delay_microseconds: 10000000 }" >> config.pbtxt)
done

# Need to launch the server for each test so that the model status is
# reset (which is used to make sure the correctly batch size was used
# for execution)
for i in \
        test_static_batch_preferred \
        test_static_batch_lt_any_preferred \
        test_static_batch_not_preferred \
        test_static_batch_gt_max_preferred \
        test_multi_batch_not_preferred \
        test_multi_batch_gt_max_preferred \
        test_multi_batch_sum_gt_max_preferred \
        test_multi_same_output0 \
        test_multi_same_output1 \
        test_multi_different_outputs ; do
    SERVER_ARGS="--model-store=`pwd`/models"
    SERVER_LOG="./$i.serverlog"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    echo "Test: $i" >>$CLIENT_LOG

    set +e
    python $BATCHER_TEST BatcherTest.$i >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    set -e

    kill $SERVER_PID
    wait $SERVER_PID
done

# Tests that require TRTSERVER_DELAY_SCHEDULER so that the scheduler
# is delayed and requests can collect in the queue.
for i in \
        test_multi_batch_use_biggest_preferred \
        test_multi_batch_use_best_preferred ; do
    export TRTSERVER_DELAY_SCHEDULER=6 &&
        [[ "$i" == "test_multi_batch_use_best_preferred" ]] && export TRTSERVER_DELAY_SCHEDULER=3
    SERVER_ARGS="--model-store=`pwd`/models"
    SERVER_LOG="./$i.serverlog"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    echo "Test: $i" >>$CLIENT_LOG

    set +e
    python $BATCHER_TEST BatcherTest.$i >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    set -e

    unset TRTSERVER_DELAY_SCHEDULER
    kill $SERVER_PID
    wait $SERVER_PID
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
