#!/bin/bash
# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

LIBTORCH_INFER_CLIENT_PY=../common/libtorch_infer_client.py

DATADIR=/data/inferenceserver/${REPO_VERSION}/qa_model_repository

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=models --log-verbose=1"
SERVER_LOG="./inference_server.log"
CLIENT_LOG="./client.log"
source ../common/util.sh

RET=0

for FLAG in true false; do
    rm -f *.log
    mkdir -p models && cp -r $DATADIR/libtorch_int32_int32_int32 models/.

    echo """
    parameters: {
        key: \"INFERENCE_MODE\"
        value: {
            string_value: \"$FLAG\"
        }
    }""" >> models/libtorch_int32_int32_int32/config.pbtxt

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e

    python $SIMPLE_INFER_CLIENT_PY >> $CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    INFERMODE_LOG="Inference Mode is "
    if [ "$FLAG" == "true" ]; then
        INFERMODE_LOG+=enabled
    else
        INFERMODE_LOG+=disabled
    fi

    if [ `grep -c "$INFERMODE_LOG" $SERVER_LOG` != "3" ]; then
        echo -e "\n***\n*** Failed. Expected 3 $INFERMODE_LOG in log\n***"
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
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
