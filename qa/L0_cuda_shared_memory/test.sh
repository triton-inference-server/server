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
        test_unregisterall; do
    for client_type in http grpc; do
        SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1"
        SERVER_LOG="./$i.$client_type.serverlog"
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

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
fi

exit $RET
