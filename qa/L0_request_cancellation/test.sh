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

export CUDA_VISIBLE_DEVICES=0

SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

RET=0

#
# Unit tests
#
rm -rf models && mkdir models
mkdir -p models/model/1 && (cd models/model && \
    echo 'name: "model"' >> config.pbtxt && \
    echo 'backend: "identity"' >> config.pbtxt && \
    echo 'max_batch_size: 64' >> config.pbtxt && \
    echo -e 'input [{ name: "INPUT0" \n data_type: TYPE_INT32 \n dims: [ 1000 ] }]' >> config.pbtxt && \
    echo -e 'output [{ name: "OUTPUT0" \n data_type: TYPE_INT32 \n dims: [ 1000 ] }]' >> config.pbtxt && \
    echo 'instance_group [{ kind: KIND_CPU }]' >> config.pbtxt)

SERVER_LOG=server.log
LD_LIBRARY_PATH=/opt/tritonserver/lib:$LD_LIBRARY_PATH ./request_cancellation_test > $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Unit Tests Failed\n***"
    cat $SERVER_LOG
    RET=1
fi

#
# gRPC cancellation tests
#
rm -rf models && mkdir models
mkdir -p models/custom_identity_int32/1 && (cd models/custom_identity_int32 && \
    echo 'name: "custom_identity_int32"' >> config.pbtxt && \
    echo 'backend: "identity"' >> config.pbtxt && \
    echo 'max_batch_size: 1024' >> config.pbtxt && \
    echo -e 'input [{ name: "INPUT0" \n data_type: TYPE_INT32 \n dims: [ -1 ] }]' >> config.pbtxt && \
    echo -e 'output [{ name: "OUTPUT0" \n data_type: TYPE_INT32 \n dims: [ -1 ] }]' >> config.pbtxt && \
    echo 'instance_group [{ kind: KIND_CPU }]' >> config.pbtxt && \
    echo -e 'parameters [{ key: "execute_delay_ms" \n value: { string_value: "10000" } }]' >> config.pbtxt)

for TEST_CASE in "test_grpc_async_infer" "test_grpc_stream_infer" "test_aio_grpc_async_infer" "test_aio_grpc_stream_infer"; do

    TEST_LOG="./grpc_cancellation_test.$TEST_CASE.log"
    SERVER_LOG="grpc_cancellation_test.$TEST_CASE.server.log"

    SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e
    python grpc_cancellation_test.py GrpcCancellationTest.$TEST_CASE > $TEST_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** gRPC Cancellation Tests Failed on $TEST_CASE\n***"
        cat $TEST_LOG
        RET=1
    fi
    grep "Cancellation notification received for" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Cancellation not received by server on $TEST_CASE\n***"
        cat $SERVER_LOG
        RET=1
    fi
    set -e

    kill $SERVER_PID
    wait $SERVER_PID
done

#
# End-to-end scheduler tests
#
rm -rf models && mkdir models
mkdir -p models/dynamic_batch/1 && (cd models/dynamic_batch && \
    echo 'name: "dynamic_batch"' >> config.pbtxt && \
    echo 'backend: "identity"' >> config.pbtxt && \
    echo 'max_batch_size: 2' >> config.pbtxt && \
    echo -e 'input [{ name: "INPUT0" \n data_type: TYPE_FP32 \n dims: [ -1 ] }]' >> config.pbtxt && \
    echo -e 'output [{ name: "OUTPUT0" \n data_type: TYPE_FP32 \n dims: [ -1 ] }]' >> config.pbtxt && \
    echo -e 'instance_group [{ count: 1 \n kind: KIND_CPU }]' >> config.pbtxt && \
    echo -e 'dynamic_batching { max_queue_delay_microseconds: 600000 }' >> config.pbtxt && \
    echo -e 'parameters [{ key: "execute_delay_ms" \n value: { string_value: "6000" } }]' >> config.pbtxt)
mkdir -p models/sequence_direct/1 && (cd models/sequence_direct && \
    echo 'name: "sequence_direct"' >> config.pbtxt && \
    echo 'backend: "identity"' >> config.pbtxt && \
    echo 'max_batch_size: 1' >> config.pbtxt && \
    echo -e 'input [{ name: "INPUT0" \n data_type: TYPE_FP32 \n dims: [ -1 ] }]' >> config.pbtxt && \
    echo -e 'output [{ name: "OUTPUT0" \n data_type: TYPE_FP32 \n dims: [ -1 ] }]' >> config.pbtxt && \
    echo -e 'instance_group [{ count: 1 \n kind: KIND_CPU }]' >> config.pbtxt && \
    echo -e 'sequence_batching { direct { } \n max_sequence_idle_microseconds: 6000000 }' >> config.pbtxt && \
    echo -e 'parameters [{ key: "execute_delay_ms" \n value: { string_value: "6000" } }]' >> config.pbtxt)
mkdir -p models/sequence_oldest/1 && (cd models/sequence_oldest && \
    echo 'name: "sequence_oldest"' >> config.pbtxt && \
    echo 'backend: "identity"' >> config.pbtxt && \
    echo 'max_batch_size: 1' >> config.pbtxt && \
    echo -e 'input [{ name: "INPUT0" \n data_type: TYPE_FP32 \n dims: [ -1 ] }]' >> config.pbtxt && \
    echo -e 'output [{ name: "OUTPUT0" \n data_type: TYPE_FP32 \n dims: [ -1 ] }]' >> config.pbtxt && \
    echo -e 'instance_group [{ count: 1 \n kind: KIND_CPU }]' >> config.pbtxt && \
    echo -e 'sequence_batching { oldest { max_candidate_sequences: 1 } \n max_sequence_idle_microseconds: 6000000 }' >> config.pbtxt && \
    echo -e 'parameters [{ key: "execute_delay_ms" \n value: { string_value: "6000" } }]' >> config.pbtxt)
mkdir -p models/ensemble_model/1 && (cd models/ensemble_model && \
    echo 'name: "ensemble_model"' >> config.pbtxt && \
    echo 'platform: "ensemble"' >> config.pbtxt && \
    echo 'max_batch_size: 1' >> config.pbtxt && \
    echo -e 'input [{ name: "INPUT0" \n data_type: TYPE_FP32 \n dims: [ -1 ] }]' >> config.pbtxt && \
    echo -e 'output [{ name: "OUTPUT0" \n data_type: TYPE_FP32 \n dims: [ -1 ] }]' >> config.pbtxt && \
    echo 'ensemble_scheduling { step [' >> config.pbtxt && \
    echo -e '{ model_name: "dynamic_batch" \n model_version: -1 \n input_map { key: "INPUT0" \n value: "INPUT0" } \n output_map { key: "OUTPUT0" \n value: "out" } },' >> config.pbtxt && \
    echo -e '{ model_name: "dynamic_batch" \n model_version: -1 \n input_map { key: "INPUT0" \n value: "out" } \n output_map { key: "OUTPUT0" \n value: "OUTPUT0" } }' >> config.pbtxt && \
    echo '] }' >> config.pbtxt)

TEST_LOG="scheduler_test.log"
SERVER_LOG="./scheduler_test.server.log"

SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=2"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python scheduler_test.py > $TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Scheduler Tests Failed\n***"
    cat $TEST_LOG
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
