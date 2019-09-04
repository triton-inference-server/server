#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

REPO_VERSION=${NVIDIA_TENSORRT_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi

CLIENT_LOG_BASE="./client"

DATADIR=`pwd`/models
MODEL_SRCDIR=/data/inferenceserver/${REPO_VERSION}/qa_custom_ops

SERVER=/opt/tensorrtserver/bin/trtserver
# Allow more time to exit. Ensemble brings in too many models
SERVER_LOG_BASE="./inference_server.log"

CLIENT_LOG="./client.log"
MULTI_STREAM_CLIENT=multi_stream_client.py

TOTAL_GPUS=${TOTAL_GPUS:=4}
TOTAL_MEM=${TOTAL_MEM:=10000}
source ../common/util.sh

# A standard TITAN V does 1200 MHz. So this value
# allows the busy loop kernel to run for over
# 2 seconds. Note: This is close to the limit for
# ints. The input to the busyloop model is an int
NUM_DELAY_CYCLES=${NUM_DELAY_CYCLES:=2100000000}

rm -f $SERVER_LOG_BASE* $CLIENT_LOG_BASE*

export LD_PRELOAD=/data/inferenceserver/${REPO_VERSION}/qa_custom_ops/libbusyop.so

for NUM_GPUS in $(seq 1 $TOTAL_GPUS); do
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(( NUM_GPUS - 1 )))
  for INSTANCE_CNT in 2 4 8; do
    for MODEL in graphdef_busyop savedmodel_busyop; do
        # Create local model repository
        rm -fr models && \
          mkdir models && \
          cp -r ${MODEL_SRCDIR}/${MODEL} models/

        # Establish baseline
        echo "instance_group [ { kind: KIND_GPU, count: ${INSTANCE_CNT} } ]" >> models/${MODEL}/config.pbtxt
        SERVER_ARGS="--model-repository=${DATADIR} --exit-timeout-secs=120"
        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        # The first run of the client warms up TF/CUDA
        set +e
        python $MULTI_STREAM_CLIENT -v -i grpc -u localhost:8001 -m $MODEL -c $INSTANCE_CNT -n $NUM_DELAY_CYCLES >> /dev/null
        python $MULTI_STREAM_CLIENT -v -i grpc -u localhost:8001 -m $MODEL -c $INSTANCE_CNT -n $NUM_DELAY_CYCLES >> $CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            exit 1
        fi
        set -e

        kill $SERVER_PID
        wait $SERVER_PID

        # Test with multi-stream
        SERVER_ARGS="--model-repository=${DATADIR} --exit-timeout-secs=120"
        PER_VGPU_MEM_LIMIT_MBYTES=$(( TOTAL_MEM / INSTANCE_CNT ))

        for i in $(seq 0 $(( NUM_GPUS - 1 ))); do
           VGPU_ARG=--tf-add-vgpu="${i};${INSTANCE_CNT};${PER_VGPU_MEM_LIMIT_MBYTES}"
           SERVER_ARGS=${SERVER_ARGS}" "${VGPU_ARG}
        done
        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        # The first run of the client warms up TF/CUDA
        set +e
        python $MULTI_STREAM_CLIENT -v -i grpc -u localhost:8001 -m $MODEL -c $INSTANCE_CNT -n $NUM_DELAY_CYCLES >> /dev/null
        python $MULTI_STREAM_CLIENT -v -i grpc -u localhost:8001 -m $MODEL -c $INSTANCE_CNT -n $NUM_DELAY_CYCLES >> $CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            exit 1
        fi
        set -e

        kill $SERVER_PID
        wait $SERVER_PID

        SCALE_FACTOR=$(grep -i "Completion time for ${INSTANCE_CNT}" $CLIENT_LOG | awk '{printf "%s ",$6}' | awk '{print $1/$2}') 
        if [ $(awk -v a="$SCALE_FACTOR" -v b="$INSTANCE_CNT" 'BEGIN{print(a<b-1)}') -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            exit 1
        fi
    done
  done
done
echo -e "\n***\n*** Test Passed\n***"
unset LD_PRELOAD
unset CUDA_VISIBLE_DEVICES
