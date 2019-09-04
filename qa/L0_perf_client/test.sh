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

CLIENT_LOG="./perf_client.log"
PERF_CLIENT=../clients/perf_client

DATADIR=/data/inferenceserver/${REPO_VERSION}/qa_model_repository

SERVER=/opt/tensorrtserver/bin/trtserver
SERVER_ARGS=--model-repository=$DATADIR
SERVER_LOG="./inference_server.log"
source ../common/util.sh

rm -f $SERVER_LOG $CLIENT_LOG

RET=0

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# Sanity check on measurements are not all zero
set +e
$PERF_CLIENT -v -i grpc -u localhost:8001 -m graphdef_int32_int32_int32 -t 1 -p2000 -b 1 >$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ $(cat $CLIENT_LOG | grep ": 0 infer/sec\|: 0 usec" | wc -l) -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

# Test perf client behavior on different model with different batch size
for MODEL in graphdef_nobatch_int32_int32_int32 graphdef_int32_int32_int32; do
    # Valid batch size
    set +e
    $PERF_CLIENT -v -i grpc -u localhost:8001 -m $MODEL -t 1 -p20000 -b 1 >$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    set -e

    # Invalid batch sizes
    for STATIC_BATCH in 0 10; do
        set +e
        $PERF_CLIENT -v -i grpc -u localhost:8001 -m $MODEL -t 1 -p20000 -b $STATIC_BATCH >$CLIENT_LOG 2>&1
        if [ $? -eq 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        set -e
    done
done

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
fi

exit $RET
