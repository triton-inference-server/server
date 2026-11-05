#!/bin/bash
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

DATADIR=/data/inferenceserver/${REPO_VERSION}
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_LOG="./inference_server.log"
CLIENT_LOG="./test.log"
source ../common/util.sh

rm -f *.log
rm -rf models io_pooling_repo

RET=0

# BFLOAT16 test
mkdir -p models
cp -r ${DATADIR}/qa_model_repository/onnx_bf16_bf16_bf16 models/.

SERVER_ARGS="--model-repository=`pwd`/models"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

for client_type in http grpc; do
    export CLIENT_TYPE=$client_type
    CLIENT_LOG="./test_${client_type}.log"
    python test.py >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed ($client_type)\n***"
        RET=1
    fi
done
unset CLIENT_TYPE

set -e

kill $SERVER_PID
wait $SERVER_PID

# io_pooling test: every variant is a copy of a base model with its own
# instance kind and io_pooling value; "default" leaves the parameter unset,
# which must behave like false since pooling is opt-in
mkdir -p io_pooling_repo
KINDS="CPU"
if [ "$TRITON_SERVER_CPU_ONLY" != "1" ]; then
    KINDS="CPU GPU"
fi
for base in add_sub_static add_sub_dynamic passthrough int4_initializer shape_mismatch; do
    for kind in $KINDS; do
        for pooling in on off default; do
            model=io_pooling_repo/${base}_${kind,,}_${pooling}
            cp -r io_pooling_models/$base $model
            echo "instance_group [ { kind: KIND_$kind count: 1 } ]" >> $model/config.pbtxt
            if [ "$pooling" == "on" ]; then
                echo "parameters { key: \"io_pooling\" value: { string_value: \"true\" } }" >> $model/config.pbtxt
            elif [ "$pooling" == "off" ]; then
                echo "parameters { key: \"io_pooling\" value: { string_value: \"false\" } }" >> $model/config.pbtxt
            fi
        done
    done
done
cp -r io_pooling_models/add_sub_static io_pooling_repo/bad_pooling_value
echo "instance_group [ { kind: KIND_CPU count: 1 } ]" >> io_pooling_repo/bad_pooling_value/config.pbtxt
echo "parameters { key: \"io_pooling\" value: { string_value: \"maybe\" } }" >> io_pooling_repo/bad_pooling_value/config.pbtxt

# The second run has no pinned and (almost) no CUDA pool memory, so pooled
# buffers must take the direct allocation fallback.
NO_POOL_ARGS="--pinned-memory-pool-byte-size=0 --cuda-memory-pool-byte-size=0:1024"
if [ "$TRITON_SERVER_CPU_ONLY" == "1" ]; then
    NO_POOL_ARGS="--pinned-memory-pool-byte-size=0"
fi
for run in 1 2; do
    POOL_ARGS=""
    POOL_DESC="default memory pools"
    if [ $run -eq 2 ]; then
        POOL_ARGS=$NO_POOL_ARGS
        POOL_DESC="no memory pools"
    fi
    SERVER_ARGS="--model-repository=`pwd`/io_pooling_repo --model-control-mode=explicit --allow-client-shm=true --log-verbose=1 $POOL_ARGS"
    SERVER_LOG="./io_pooling_server_${run}.log"
    CLIENT_LOG="./io_pooling_test_${run}.log"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e

    python3 io_pooling_test.py >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed (io_pooling, $POOL_DESC)\n***"
        RET=1
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
