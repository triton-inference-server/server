#!/bin/bash
# Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
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

CLIENT_LOG="./client.log"
STRESS_TEST=sequence_stress.py

SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

RET=0

# Setup model repository.
#   models1 - one instance with batch-size 4
#   models2 - two instances with batch-size 2
#   models4 - four instances with batch-size 1
rm -fr *.log  models{1,2,4} && mkdir models{1,2,4}
for m in ../custom_models/custom_sequence_int32 ; do
    cp -r $m models1/. && \
        (cd models1/$(basename $m) && \
            sed -i "s/max_sequence_idle_microseconds:.*/max_sequence_idle_microseconds: 1000000/" config.pbtxt && \
            sed -i "s/^max_batch_size:.*/max_batch_size: 4/" config.pbtxt && \
            sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 1/" config.pbtxt && \
            sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 1/" config.pbtxt)
    cp -r $m models2/. && \
        (cd models2/$(basename $m) && \
            sed -i "s/max_sequence_idle_microseconds:.*/max_sequence_idle_microseconds: 1000000/" config.pbtxt && \
            sed -i "s/^max_batch_size:.*/max_batch_size: 2/" config.pbtxt && \
            sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 2/" config.pbtxt && \
            sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 2/" config.pbtxt)
    cp -r $m models4/. && \
        (cd models4/$(basename $m) && \
            sed -i "s/max_sequence_idle_microseconds:.*/max_sequence_idle_microseconds: 1000000/" config.pbtxt && \
            sed -i "s/^max_batch_size:.*/max_batch_size: 1/" config.pbtxt && \
            sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 4/" config.pbtxt && \
            sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 4/" config.pbtxt)
done

# Stress-test each model repository
for model_trial in 1 2 4 ; do
    MODEL_DIR=models${model_trial}
    SERVER_ARGS="--model-repository=`pwd`/$MODEL_DIR"
    SERVER_LOG="./$MODEL_DIR.server.log"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e
    python $STRESS_TEST >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    set -e

    kill $SERVER_PID
    wait $SERVER_PID
done

# Test invalid gRPC infer handler thread count
for thread_cnt in -1 0 1 129; do
    MODEL_DIR=models1
    SERVER_ARGS="--model-repository=`pwd`/$MODEL_DIR --grpc-infer-thread-count=$thread_cnt"
    SERVER_LOG="./$MODEL_DIR.server.log"
    run_server
    if [ "$SERVER_PID" != "0" ]; then
        echo -e "\n***\n*** Failed: $SERVER started successfully when it was expected to fail\n***"
        RET=1
        kill SERVER_PID
        wait $SERVER_PID
    fi
done

# Test gRPC infer handler thread count under stress
thread_cnt=128
for model_trial in 1 2 4 ; do
    MODEL_DIR=models${model_trial}
    SERVER_ARGS="--model-repository=`pwd`/$MODEL_DIR --grpc-infer-thread-count=$thread_cnt"
    SERVER_LOG="./$MODEL_DIR.server.log"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e
    python $STRESS_TEST >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
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
