#!/bin/bash
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

SIMPLE_CLIENT=../clients/simple_http_infer_client
SIMPLE_SEQ_CLIENT=../clients/simple_grpc_sequence_stream_infer_client

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=`pwd`/models"
source ../common/util.sh

export CUDA_VISIBLE_DEVICES=0

RET=0

rm -fr *.log

# This is a test of the schedulers to make sure they correctly release
# their own backend so don't need to test across all frameworks.  Set
# the delay, in milliseconds, that will cause the scheduler to be the
# last holding the backend handle.
export TRITONSERVER_DELAY_SCHEDULER_BACKEND_RELEASE=5000

# dynamic batcher - 1 instance
rm -fr models && cp -r simple_models models
(cd models/simple && echo "instance_group [{ count: 1 }]" >> config.pbtxt)

SERVER_LOG="./inference_server_1.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

$SIMPLE_CLIENT -v >> client_simple.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

# dynamic batcher - 4 instance
rm -fr models && cp -r simple_models models
(cd models/simple && echo "instance_group [{ count: 4 }]" >> config.pbtxt)

SERVER_LOG="./inference_server_4.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

$SIMPLE_CLIENT -v >> client_simple.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

# sequence batcher - 1 instance
rm -fr models && cp -r simple_seq_models models
(cd models/simple_sequence && \
        sed -i "s/sequence_batching.*{.*/sequence_batching { max_sequence_idle_microseconds: 10000000/" \
            config.pbtxt)

SERVER_LOG="./inference_server_seq_1.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

$SIMPLE_SEQ_CLIENT -v >> client_simple_seq.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

# sequence batcher - 4 instance
rm -fr models && cp -r simple_seq_models models
(cd models/simple_sequence && \
        echo "instance_group [{ count: 3 }]" >> config.pbtxt && \
        sed -i "s/sequence_batching.*{.*/sequence_batching { max_sequence_idle_microseconds: 10000000/" \
            config.pbtxt)

SERVER_LOG="./inference_server_seq_4.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

$SIMPLE_SEQ_CLIENT -v >> client_simple_seq.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
