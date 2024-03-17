#!/bin/bash
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This L0_model_update test should make changes to models without restarting the
# server, unless restarting the server is the only way of accomplishing the
# change.

export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE="True"
export MODEL_LOG_DIR="`pwd`"

SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

function setup_models() {
    rm -rf models && mkdir models
    # Basic model that log instance creation and destruction
    cp -r ../python_models/model_init_del models/model_init_del && \
        mkdir models/model_init_del/1 && \
        mv models/model_init_del/model.py models/model_init_del/1
}

RET=0

# Test model instance update with rate limiting on/off and explicit resource
for RATE_LIMIT_MODE in "off" "execution_count" "execution_count_with_explicit_resource"; do

    RATE_LIMIT_ARGS="--rate-limit=$RATE_LIMIT_MODE"
    if [ "$RATE_LIMIT_MODE" == "execution_count_with_explicit_resource" ]; then
        RATE_LIMIT_ARGS="--rate-limit=execution_count --rate-limit-resource=R1:10"
    fi

    export RATE_LIMIT_MODE=$RATE_LIMIT_MODE
    TEST_LOG="instance_update_test.rate_limit_$RATE_LIMIT_MODE.log"
    SERVER_LOG="./instance_update_test.rate_limit_$RATE_LIMIT_MODE.server.log"

    setup_models
    SERVER_ARGS="--model-repository=models --model-control-mode=explicit $RATE_LIMIT_ARGS --log-verbose=2"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e
    python instance_update_test.py > $TEST_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed model instance update test on rate limit mode $RATE_LIMIT_MODE\n***"
        cat $TEST_LOG
        RET=1
    fi
    set -e

    kill $SERVER_PID
    wait $SERVER_PID

    set +e
    grep "Should not print this" $SERVER_LOG
    if [ $? -eq 0 ]; then
        echo -e "\n***\n*** Found \"Should not print this\" on \"$SERVER_LOG\"\n***"
        cat $SERVER_LOG
        RET=1
    fi
    set -e

done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi
exit $RET
