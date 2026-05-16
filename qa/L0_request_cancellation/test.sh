#!/bin/bash
# Copyright 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
CANCEL_LOG_LINE="Cancellation notification received for"

RET=0
rm -f *.log

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
LD_LIBRARY_PATH=/opt/tritonserver/lib:$LD_LIBRARY_PATH ./request_cancellation_test > $SERVER_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Unit Tests Failed\n***"
    cat $SERVER_LOG
    RET=1
fi

#
# Python gRPC cancellation tests
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

for TEST_CASE in "test_grpc_async_infer" \
                    "test_grpc_stream_infer" \
                    "test_aio_grpc_async_infer" \
                    "test_aio_grpc_stream_infer" \
                    "test_grpc_async_infer_cancellation_at_step_start" \
                    "test_grpc_async_infer_response_complete_during_cancellation" \
                    "test_grpc_async_infer_cancellation_before_finish_0" \
                    "test_grpc_async_infer_cancellation_before_finish_1" \
                    "test_grpc_async_infer_cancellation_before_response_complete_and_process_after_final_response"; do
    TEST_LOG="./grpc_cancellation_test.$TEST_CASE.log"
    SERVER_LOG="grpc_cancellation_test.$TEST_CASE.server.log"
    if [ "$TEST_CASE" == "test_grpc_async_infer_cancellation_at_step_start" ]; then
        export TRITONSERVER_DELAY_GRPC_PROCESS=5000
    elif [ "$TEST_CASE" == "test_grpc_async_infer_response_complete_during_cancellation" ]; then
        export TRITONSERVER_DELAY_GRPC_NOTIFICATION=5000
        export TRITONSERVER_DELAY_GRPC_ENQUEUE=5000
    elif [ "$TEST_CASE" == "test_grpc_async_infer_cancellation_before_finish_0" ]; then
        export TRITONSERVER_DELAY_GRPC_NOTIFICATION=5000
        export TRITONSERVER_DELAY_RESPONSE_COMPLETION=5000
    elif [ "$TEST_CASE" == "test_grpc_async_infer_cancellation_before_finish_1" ]; then
        export TRITONSERVER_DELAY_GRPC_PROCESS_ENTRY=1000
        export TRITONSERVER_DELAY_RESPONSE_COMPLETION=5000
    elif [ "$TEST_CASE" == "test_grpc_async_infer_cancellation_before_response_complete_and_process_after_final_response" ]; then
        export TRITONSERVER_DELAY_GRPC_NOTIFICATION=5000
        export TRITONSERVER_DELAY_RESPONSE_COMPLETE_EXEC=5000
    fi

    SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=2"
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

    count=$(grep -o "$CANCEL_LOG_LINE" $SERVER_LOG | wc -l)
    if [ $count == 0 ]; then
        echo -e "\n***\n*** Cancellation not received by server on $TEST_CASE\n***"
        cat $SERVER_LOG
        RET=1
    elif [ $count -ne 1 ]; then
        echo -e "\n***\n*** Unexpected cancellation received by server on $TEST_CASE. Expected 1 but received $count.\n***"
        cat $SERVER_LOG
        RET=1
    fi

    # Tests "test_grpc_async_infer" and "test_aio_grpc_async_infer" ends
    # prematurely before state is released.
    if [[ "$TEST_CASE" != "test_grpc_async_infer" && "$TEST_CASE" != "test_aio_grpc_async_infer" ]]; then
        count=$(grep -o "StateRelease" $SERVER_LOG | wc -l)
        state_released=${state_released:=1}
        if [ $count == 0 ]; then
            echo -e "\n***\n*** State not released by server on $TEST_CASE\n***"
            cat $SERVER_LOG
            RET=1
        elif [ $count -ne $state_released ]; then
            echo -e "\n***\n*** Unexpected states released by server on $TEST_CASE. Expected $state_released but released $count.\n***"
            cat $SERVER_LOG
            RET=1
        fi
        unset state_released
    fi
    set -e

    kill $SERVER_PID
    wait $SERVER_PID

    if [ "$TEST_CASE" == "test_grpc_async_infer_cancellation_at_step_start" ]; then
        unset TRITONSERVER_DELAY_GRPC_PROCESS
    elif [ "$TEST_CASE" == "test_grpc_async_infer_response_complete_during_cancellation" ]; then
        unset TRITONSERVER_DELAY_GRPC_NOTIFICATION
        unset TRITONSERVER_DELAY_GRPC_ENQUEUE
    elif [ "$TEST_CASE" == "test_grpc_async_infer_cancellation_before_finish_0" ]; then
        unset TRITONSERVER_DELAY_GRPC_NOTIFICATION
        unset TRITONSERVER_DELAY_RESPONSE_COMPLETION
    elif [ "$TEST_CASE" == "test_grpc_async_infer_cancellation_before_finish_1" ]; then
        unset TRITONSERVER_DELAY_GRPC_PROCESS_ENTRY
        unset TRITONSERVER_DELAY_RESPONSE_COMPLETION
    elif [ "$TEST_CASE" == "test_grpc_async_infer_cancellation_before_response_complete_and_process_after_final_response" ]; then
        unset TRITONSERVER_DELAY_GRPC_NOTIFICATION
        unset TRITONSERVER_DELAY_RESPONSE_COMPLETE_EXEC
    fi
done

#
# C++ gRPC cancellation tests
#
GRPC_CANCELLATION_TEST_CPP=../clients/grpc_cancellation_test

for ENTRY in "TestGrpcAsyncInfer 1" \
             "TestGrpcAsyncInferCancelAfterCompletionIsNoOp 0" \
             "TestGrpcAsyncInferWithoutContextStillCompletes 0" \
             "TestGrpcAsyncInferMulti 2" \
             "TestGrpcStreamInfer 1" \
             "TestGrpcStreamCancelWithoutInfer 1" \
             "TestGrpcStreamCancelThenRestart 1"; do
    read -r TEST_CASE EXPECTED_CANCEL_COUNT <<< "$ENTRY"

    TEST_LOG="./grpc_cancellation_test_cpp.$TEST_CASE.log"
    SERVER_LOG="./grpc_cancellation_test_cpp.$TEST_CASE.server.log"

    # AsyncInferMulti fans out N concurrent requests; bump to 3 CPU
    # instances so each can execute in parallel. Reverted after the test
    # so subsequent cases keep the default single-instance config.
    if [ "$TEST_CASE" == "TestGrpcAsyncInferMulti" ]; then
        sed -i 's|instance_group .*|instance_group [{ count: 3, kind: KIND_CPU }]|' \
            models/custom_identity_int32/config.pbtxt
    fi

    SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=2"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e
    LD_LIBRARY_PATH=/opt/tritonserver/lib:$LD_LIBRARY_PATH \
        $GRPC_CANCELLATION_TEST_CPP \
            --gtest_filter="GrpcCancellationTest.$TEST_CASE" > $TEST_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** C++ gRPC Cancellation Tests Failed on $TEST_CASE\n***"
        cat $TEST_LOG
        RET=1
    fi

    cancel_count=$(grep -c "$CANCEL_LOG_LINE" $SERVER_LOG || true)
    if [ $cancel_count -ne $EXPECTED_CANCEL_COUNT ]; then
        echo -e "\n***\n*** Unexpected cancellation count on $TEST_CASE. Expected $EXPECTED_CANCEL_COUNT but received $cancel_count.\n***"
        cat $SERVER_LOG
        RET=1
    fi
    set -e

    kill $SERVER_PID
    wait $SERVER_PID

    if [ "$TEST_CASE" == "TestGrpcAsyncInferMulti" ]; then
        sed -i 's|instance_group .*|instance_group [{ kind: KIND_CPU }]|' \
            models/custom_identity_int32/config.pbtxt
    fi
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

#
# Implicit state tests
#
rm -rf models && mkdir models
mkdir -p models/sequence_state/1 && (cd models/sequence_state && \
    cp ../../implicit_state_model/config.pbtxt . && \
    cp ../../implicit_state_model/model.pt 1)

TEST_LOG="implicit_state_test.log"
SERVER_LOG="implicit_state_test.server.log"

SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
SERVER_LOG=$SERVER_LOG python implicit_state_test.py > $TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Implicit State Tests Failed\n***"
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
