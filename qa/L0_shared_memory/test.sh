#!/bin/bash
# Copyright 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

CLIENT_LOG="./client.log"
SHM_TEST=shared_memory_test.py
TEST_RESULT_FILE='test_results.txt'

# Configure to support test on jetson as well
TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
DATADIR=/data/inferenceserver/${REPO_VERSION}
SERVER=${TRITON_DIR}/bin/tritonserver
BACKEND_DIR=${TRITON_DIR}/backends
SERVER_ARGS_EXTRA="--backend-directory=${BACKEND_DIR}"
source ../common/util.sh
pip3 install psutil

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
        test_infer_offset_out_of_bound \
        test_infer_byte_size_out_of_bound \
        test_infer_integer_overflow \
        test_register_out_of_bound \
        test_register_reserved_names \
        test_python_client_leak; do
    for client_type in http grpc; do
        SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1 ${SERVER_ARGS_EXTRA}"
        SERVER_LOG="./$i.$client_type.server.log"
        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        export CLIENT_TYPE=$client_type
        TMP_CLIENT_LOG="./tmp_client.log"
        echo "Test: $i, client type: $client_type" >>$TMP_CLIENT_LOG

        set +e
        python3 $SHM_TEST SharedMemoryTest.$i >>$TMP_CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            cat $TMP_CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        else
            check_test_results $TEST_RESULT_FILE 1
            if [ $? -ne 0 ]; then
                cat $TEST_RESULT_FILE
                echo -e "\n***\n*** Test Result Verification Failed\n***"
                RET=1
            fi
        fi
        cat $TMP_CLIENT_LOG >>$CLIENT_LOG
        rm $TMP_CLIENT_LOG
        kill $SERVER_PID
        wait $SERVER_PID
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Server shut down non-gracefully\n***"
            RET=1
        fi
        set -e
    done
done

mkdir -p python_models/simple/1/
cp ../python_models/execute_delayed_model/model.py ./python_models/simple/1/
cp ../python_models/execute_delayed_model/config.pbtxt ./python_models/simple/

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
        python3 $SHM_TEST "TestSharedMemoryUnregister.${test_case}_${client_type}" >>"$CLIENT_LOG" 2>&1
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

# Test large system shared memory offset
rm -rf models/*
# prepare add_sub model of various backends
BACKENDS="python onnx libtorch plan openvino"
for backend in ${BACKENDS} ; do
    model="${backend}_int32_int32_int32"
    model_dir="models/${model}"
    if [[ $backend == "python" ]]; then
        mkdir -p ${model_dir}/1
        cp ../python_models/add_sub/model.py ${model_dir}/1/
        cp ../python_models/add_sub/config.pbtxt ${model_dir}/
        sed -i 's/TYPE_FP32/TYPE_INT32/g' ${model_dir}/config.pbtxt
        echo "max_batch_size: 8" >> ${model_dir}/config.pbtxt
    else
        mkdir -p ${model_dir}
        cp -r $DATADIR/qa_model_repository/${model}/1 ${model_dir}/1
        cp $DATADIR/qa_model_repository/${model}/config.pbtxt ${model_dir}/
        cp $DATADIR/qa_model_repository/${model}/output0_labels.txt ${model_dir}/
        if [ $backend == "openvino" ]; then
            echo 'parameters { key: "ENABLE_BATCH_PADDING" value { string_value: "YES" } }' >> models/${model}/config.pbtxt
        fi
    fi
done

test_case="test_large_shm_register_offset"
for client_type in http grpc; do
    SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1 ${SERVER_ARGS_EXTRA}"
    SERVER_LOG="./${test_case}.${client_type}.server.log"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    export CLIENT_TYPE=$client_type
    CLIENT_LOG="./${test_case}.${client_type}.client.log"
    set +e
    python3 $SHM_TEST SharedMemoryTest.${test_case} >>"$CLIENT_LOG" 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed - ${client_type}\n***"
        RET=1
    fi

    kill $SERVER_PID
    wait $SERVER_PID
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Server shut down non-gracefully\n***"
        RET=1
    fi
    set -e
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test Failed\n***"
fi

exit $RET
