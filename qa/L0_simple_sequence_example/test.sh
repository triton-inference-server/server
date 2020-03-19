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

export CUDA_VISIBLE_DEVICES=0

SIMPLE_CLIENT=../clients/simple_sequence_client
SIMPLE_CLIENT_PY=../clients/simple_sequence_client.py

SERVER=/opt/tritonserver/bin/trtserver
source ../common/util.sh

rm -f *.log

RET=0

for TRIAL in simple dyna; do
    if [ "$TRIAL" == "simple" ]; then
        CLIENT_ARGS="-v"
        CLIENT_LOG="./client.log"
        SERVER_ARGS="--model-repository=`pwd`/models"
        SERVER_LOG="./inference_server.log"
    else
        CLIENT_ARGS="-v -d"
        CLIENT_LOG="./client_dyna.log"
        SERVER_ARGS="--model-repository=`pwd`/models_dyna"
        SERVER_LOG="./inference_server_dyna.log"
    fi

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e

    $SIMPLE_CLIENT $CLIENT_ARGS >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    python $SIMPLE_CLIENT_PY $CLIENT_ARGS >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    $SIMPLE_CLIENT $CLIENT_ARGS -a >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    python $SIMPLE_CLIENT_PY $CLIENT_ARGS -a >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    $SIMPLE_CLIENT $CLIENT_ARGS -r -a >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    python $SIMPLE_CLIENT_PY $CLIENT_ARGS -r -a >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
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
