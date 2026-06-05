#!/bin/bash
# Copyright 2019-2024, NVIDIA CORPORATION. All rights reserved.
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

CLIENT_LOG="./client.log"
SHM_TEST=cuda_shared_memory_test.py

TEST_RESULT_FILE='test_results.txt'
SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

RET=0
rm -fr *.log

for i in \
        test_invalid_create_shm \
        test_valid_create_set_register \
        test_unregister_before_register \
        test_unregister_after_register \
        test_reregister_after_register \
        test_unregister_after_inference \
        test_register_after_inference \
        test_too_big_shm \
        test_mixed_raw_shm \
        test_unregisterall \
        test_register_out_of_bound \
        test_infer_offset_out_of_bound \
        test_infer_byte_size_out_of_bound; do
    for client_type in http grpc; do
        SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1"
        SERVER_LOG="./$i.$client_type.server.log"
        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        export CLIENT_TYPE=$client_type
        echo "Test: $i, client type: $client_type" >>$CLIENT_LOG

        set +e
        python $SHM_TEST CudaSharedMemoryTest.$i >>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        else
            check_test_results $TEST_RESULT_FILE 1
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG
                echo -e "\n***\n*** Test Result Verification Failed\n***"
                RET=1
            fi
        fi
        set -e

        kill $SERVER_PID
        wait $SERVER_PID
    done
done

for i in \
        test_exceeds_cshm_handle_size_limit \
        test_invalid_small_cshm_handle \
        test_valid_cshm_handle; do
    SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1"
    SERVER_LOG="./$i.server.log"
    CLIENT_LOG="./$i.client.log"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi
    echo "Test: $i, client type: HTTP" >>$CLIENT_LOG
    set +e
    python $SHM_TEST CudaSharedMemoryTestRawHttpRequest.$i >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    else
        check_test_results $TEST_RESULT_FILE 1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Result Verification Failed\n***"
            RET=1
        fi
    fi
    set -e
    kill $SERVER_PID
    wait $SERVER_PID
done

mkdir -p python_models/simple/1/
cp ../python_models/execute_delayed_model/model.py ./python_models/simple/1/
cp ../python_models/execute_delayed_model/config.pbtxt ./python_models/simple/
sed -i 's/KIND_CPU/KIND_GPU/g' ./python_models/simple/config.pbtxt


for test_case in \
        test_unregister_shm_during_inference_single_req \
        test_unregister_shm_during_inference_multiple_req \
        test_unregister_shm_after_inference; do
    for client_type in http grpc; do
        SERVER_ARGS="--model-repository=`pwd`/python_models --log-verbose=1 ${SERVER_ARGS_EXTRA}"
        SERVER_LOG="./${test_case}_${client_type}.server.log"
        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        export CLIENT_TYPE=$client_type
        CLIENT_LOG="./${test_case}_${client_type}.client.log"
        set +e
        python3 $SHM_TEST "TestCudaSharedMemoryUnregister.${test_case}_${client_type}" >>"$CLIENT_LOG" 2>&1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed - ${test_case}_${client_type}\n***"
            RET=1
        else
            check_test_results $TEST_RESULT_FILE 1
            if [ $? -ne 0 ]; then
                cat $TEST_RESULT_FILE
                echo -e "\n***\n*** Test Result Verification Failed - ${test_case}_${client_type}\n***"
                RET=1
            fi
        fi

        kill $SERVER_PID
        wait $SERVER_PID
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Server shut down non-gracefully\n***"
            RET=1
        fi
        set -e
    done
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
fi

exit $RET
