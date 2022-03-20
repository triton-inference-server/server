#!/bin/bash
# Copyright 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

export CUDA_VISIBLE_DEVICES=0

IMAGE_CLIENT=../clients/image_client.py
IMAGE=../images/vulture.jpeg

DATADIR=/data/inferenceserver/${REPO_VERSION}/libtorch_model_store

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=models --log-verbose=1"
SERVER_LOG_PREFIX="./inference_server.log"
CLIENT_LOG_PREFIX="./client.log"
source ../common/util.sh

rm -f *.log.*

RET=0

for FLAG in true false none; do
    CLIENT_LOG=$CLIENT_LOG_PREFIX.$FLAG
    SERVER_LOG=$SERVER_LOG_PREFIX.$FLAG
    mkdir -p models && cp -r $DATADIR/resnet50_libtorch models/.

    if [ "$FLAG" != "none" ]; then
        echo """
parameters: {
    key: \"ENABLE_NVFUSER\"
    value: {
        string_value: \"$FLAG\"
    }
}""" >> models/resnet50_libtorch/config.pbtxt
    fi

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e

    # Send requests of 2 different batch sizes to catch nvfuser issues.
    python $IMAGE_CLIENT -m resnet50_libtorch -s INCEPTION -c 1 -b 1 $IMAGE >> $CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        FAILED_LOG=$CLIENT_LOG
        RET=1
    fi

    python $IMAGE_CLIENT -m resnet50_libtorch -s INCEPTION -c 1 -b 8 $IMAGE >> $CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        FAILED_LOG=$CLIENT_LOG
        RET=1
    fi

    NVFUSER_LOG="NvFuser is "
    if [ "$FLAG" == "true" ]; then
        # NvFuser support has been disabled. Change to 'enabled' when fixed.
        NVFUSER_LOG+="disabled"
    elif [ "$FLAG" == "false" ]; then
        NVFUSER_LOG+="disabled"
    else
        NVFUSER_LOG+="not specified"
    fi

    if [ `grep -c "$NVFUSER_LOG" $SERVER_LOG` != "1" ]; then
        echo -e "\n***\n*** Failed. Expected 1 $NVFUSER_LOG in log\n***"
        RET=1
    fi

    if [ `grep -c VULTURE $CLIENT_LOG` != "9" ]; then
        echo -e "\n***\n*** Failed. Expected 9 VULTURE results\n***"
        FAILED_LOG=$CLIENT_LOG
        RET=1
    fi

    set -e

    kill $SERVER_PID
    wait $SERVER_PID

    rm -rf models
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $FAILED_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
