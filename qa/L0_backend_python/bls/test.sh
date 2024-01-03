#!/bin/bash
# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

CLIENT_PY=../python_unittest.py
CLIENT_LOG="./bls_client.log"
EXPECTED_NUM_TESTS="1"
TEST_RESULT_FILE='test_results.txt'
source ../../common/util.sh

TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
BACKEND_DIR=${TRITON_DIR}/backends

RET=0
# This variable is used to print out the correct server log for each sub-test.
SUB_TEST_RET=0
rm -fr *.log ./models *.txt

pip3 uninstall -y torch
pip3 install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html

mkdir -p models/bls/1/
cp ../../python_models/bls/model.py models/bls/1/
cp ../../python_models/bls/config.pbtxt models/bls

mkdir -p models/dlpack_add_sub/1/
cp ../../python_models/dlpack_add_sub/model.py models/dlpack_add_sub/1/
cp ../../python_models/dlpack_add_sub/config.pbtxt models/dlpack_add_sub

mkdir -p models/bls_async/1/
cp ../../python_models/bls_async/model.py models/bls_async/1/
cp ../../python_models/bls_async/config.pbtxt models/bls_async

mkdir -p models/bls_memory/1/
cp ../../python_models/bls_memory/model.py models/bls_memory/1/
cp ../../python_models/bls_memory/config.pbtxt models/bls_memory

mkdir -p models/bls_memory_async/1/
cp ../../python_models/bls_memory_async/model.py models/bls_memory_async/1/
cp ../../python_models/bls_memory_async/config.pbtxt models/bls_memory_async

mkdir -p models/add_sub/1/
cp ../../python_models/add_sub/model.py models/add_sub/1/
cp ../../python_models/add_sub/config.pbtxt models/add_sub

mkdir -p models/execute_error/1/
cp ../../python_models/execute_error/model.py models/execute_error/1/
cp ../../python_models/execute_error/config.pbtxt models/execute_error

mkdir -p models/identity_fp32/1/
cp ../../python_models/identity_fp32/model.py models/identity_fp32/1/
cp ../../python_models/identity_fp32/config.pbtxt models/identity_fp32

mkdir -p models/dlpack_identity/1/
cp ../../python_models/dlpack_identity/model.py models/dlpack_identity/1/
cp ../../python_models/dlpack_identity/config.pbtxt models/dlpack_identity

cp -r ${DATADIR}/qa_sequence_implicit_model_repository/onnx_nobatch_sequence_int32/ ./models

git clone https://github.com/triton-inference-server/python_backend -b $PYTHON_BACKEND_REPO_TAG
mkdir -p models/square_int32/1/
cp python_backend/examples/decoupled/square_model.py models/square_int32/1/model.py
cp python_backend/examples/decoupled/square_config.pbtxt models/square_int32/config.pbtxt

mkdir -p models/dlpack_square/1/
cp ../../python_models/dlpack_square/model.py models/dlpack_square/1/
cp ../../python_models/dlpack_square/config.pbtxt models/dlpack_square

mkdir -p models/identity_fp32_timeout/1/
cp ../../python_models/identity_fp32_timeout/model.py models/identity_fp32_timeout/1/
cp ../../python_models/identity_fp32_timeout/config.pbtxt models/identity_fp32_timeout

cp -r ${DATADIR}/qa_model_repository/libtorch_nobatch_float32_float32_float32/ ./models/libtorch_gpu && \
    sed -i 's/libtorch_nobatch_float32_float32_float32/libtorch_gpu/' models/libtorch_gpu/config.pbtxt && \
    echo "instance_group [ { kind: KIND_GPU} ]" >> models/libtorch_gpu/config.pbtxt

cp -r ${DATADIR}/qa_model_repository/libtorch_nobatch_float32_float32_float32/ ./models/libtorch_cpu && \
    sed -i 's/libtorch_nobatch_float32_float32_float32/libtorch_cpu/' models/libtorch_cpu/config.pbtxt && \
    echo "instance_group [ { kind: KIND_CPU} ]" >> models/libtorch_cpu/config.pbtxt

# Test with different sizes of CUDA memory pool
for CUDA_MEMORY_POOL_SIZE_MB in 64 128 ; do
    CUDA_MEMORY_POOL_SIZE_BYTES=$((CUDA_MEMORY_POOL_SIZE_MB * 1024 * 1024))
    SERVER_ARGS="--model-repository=`pwd`/models --backend-directory=${BACKEND_DIR} --log-verbose=1 --cuda-memory-pool-byte-size=0:${CUDA_MEMORY_POOL_SIZE_BYTES}"
    for TRIAL in non_decoupled decoupled ; do
        export BLS_KIND=$TRIAL
        SERVER_LOG="./bls_$TRIAL.$CUDA_MEMORY_POOL_SIZE_MB.inference_server.log"

        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        set +e

        export MODEL_NAME='bls'
        python3 $CLIENT_PY >> $CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** 'bls' $BLS_KIND test FAILED. \n***"
            cat $CLIENT_LOG
            RET=1
            SUB_TEST_RET=1
        else
            check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG
                echo -e "\n***\n*** Test Result Verification Failed\n***"
                RET=1
                SUB_TEST_RET=1
            fi
        fi

        export MODEL_NAME='bls_memory'
        python3 $CLIENT_PY >> $CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** 'bls_memory' $BLS_KIND test FAILED. \n***"
            cat $CLIENT_LOG
            RET=1
            SUB_TEST_RET=1
        else
            check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG
                echo -e "\n***\n*** Test Result Verification Failed\n***"
                RET=1
                SUB_TEST_RET=1
            fi
        fi

        export MODEL_NAME='bls_memory_async'
        python3 $CLIENT_PY >> $CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** 'bls_async_memory' $BLS_KIND test FAILED. \n***"
            cat $CLIENT_LOG
            RET=1
            SUB_TEST_RET=1
        else
            check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG
                echo -e "\n***\n*** Test Result Verification Failed\n***"
                RET=1
                SUB_TEST_RET=1
            fi
        fi

        export MODEL_NAME='bls_async'
        python3 $CLIENT_PY >> $CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** 'bls_async' $BLS_KIND test FAILED. \n***"
            cat $CLIENT_LOG
            RET=1
            SUB_TEST_RET=1
        else
            check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG
                echo -e "\n***\n*** Test Result Verification Failed\n***"
                RET=1
                SUB_TEST_RET=1
            fi
        fi

        set -e

        kill $SERVER_PID
        wait $SERVER_PID

        if [ $SUB_TEST_RET -eq 1 ]; then
            cat $CLIENT_LOG
            cat $SERVER_LOG
        fi

        # Check for bls 'test_timeout' to ensure timeout value is being correctly passed
        if [ `grep -c "Request timeout: 11000000000" $SERVER_LOG` == "0" ]; then
            echo -e "\n***\n*** BLS timeout value not correctly passed to model: line ${LINENO}\n***"
            cat $SERVER_LOG
            RET=1
        fi

        if [[ $CUDA_MEMORY_POOL_SIZE_MB -eq 128 ]]; then
            if [ `grep -c "Failed to allocate memory from CUDA memory pool" $SERVER_LOG` != "0" ]; then
                echo -e "\n***\n*** Expected to use CUDA memory pool for all tests when CUDA_MEMOY_POOL_SIZE_MB is 128 MB for 'bls' $BLS_KIND test\n***"
                cat $SERVER_LOG
                RET=1
            fi
        fi
    done
done

# Test error handling when BLS is used in "initialize" or "finalize" function
ERROR_MESSAGE="BLS is only supported during the 'execute' function."

rm -fr ./models
mkdir -p models/bls_init_error/1/
cp ../../python_models/bls_init_error/model.py models/bls_init_error/1/
cp ../../python_models/bls_init_error/config.pbtxt models/bls_init_error
SERVER_LOG="./bls_init_error_server.log"
SUB_TEST_RET=0

run_server
if [ "$SERVER_PID" != "0" ]; then
    echo -e "*** FAILED: unexpected success starting $SERVER" >> $CLIENT_LOG
    RET=1
    SUB_TEST_RET=1
    kill $SERVER_PID
    wait $SERVER_PID
else
    if grep "$ERROR_MESSAGE" $SERVER_LOG; then
        echo -e "Found \"$ERROR_MESSAGE\"" >> $CLIENT_LOG
    else
        echo -e "Not found \"$ERROR_MESSAGE\"" >> $CLIENT_LOG
        RET=1
        SUB_TEST_RET=1
    fi
fi

if [ $SUB_TEST_RET -eq 1 ]; then
    cat $CLIENT_LOG
    cat $SERVER_LOG
fi

rm -fr ./models
mkdir -p models/bls_finalize_error/1/
cp ../../python_models/bls_finalize_error/model.py models/bls_finalize_error/1/
cp ../../python_models/bls_finalize_error/config.pbtxt models/bls_finalize_error/
SERVER_LOG="./bls_finalize_error_server.log"
SUB_TEST_RET=0

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

kill $SERVER_PID
wait $SERVER_PID

if grep "$ERROR_MESSAGE" $SERVER_LOG; then
    echo -e "Found \"$ERROR_MESSAGE\"" >> $CLIENT_LOG
else
    echo -e "Not found \"$ERROR_MESSAGE\"" >> $CLIENT_LOG
    RET=1
    SUB_TEST_RET=1
fi

if [ $SUB_TEST_RET -eq 1 ]; then
    cat $CLIENT_LOG
    cat $SERVER_LOG
fi

# Test model loading API with BLS
SUB_TEST_RET=0
rm -fr ./models
mkdir -p models/bls_model_loading/1/
cp ../../python_models/bls_model_loading/model.py models/bls_model_loading/1/
cp ../../python_models/bls_model_loading/config.pbtxt models/bls_model_loading/
cp -fr ${DATADIR}/qa_model_repository/onnx_int32_int32_int32 models/.
# Make only version 2, 3 is valid version directory
rm -rf models/onnx_int32_int32_int32/1

SERVER_LOG="./bls_model_loading_server.log"
SERVER_ARGS="--model-repository=`pwd`/models --backend-directory=${BACKEND_DIR} --model-control-mode=explicit --log-verbose=1"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

export MODEL_NAME='bls_model_loading'

set +e
code=`curl -s -w %{http_code} -X POST localhost:8000/v2/repository/models/${MODEL_NAME}/load`
set -e
if [ "$code" != "200" ]; then
    echo -e "\n***\n*** Failed to load model '${MODEL_NAME}'\n***"
    RET=1
    SUB_TEST_RET=1
fi

set +e

python3 $CLIENT_PY >> $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** 'bls_model_loading' test FAILED. \n***"
    cat $CLIENT_LOG
    RET=1
    SUB_TEST_RET=1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
        SUB_TEST_RET=1
    fi
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $SUB_TEST_RET -eq 1 ]; then
    cat $CLIENT_LOG
    cat $SERVER_LOG
fi

# Test model loading API with BLS warmup
(cd models/bls_model_loading && \
        echo "model_warmup [{" >> config.pbtxt && \
        echo "    name : \"regular sample\"" >> config.pbtxt && \
        echo "    batch_size: 1" >> config.pbtxt && \
        echo "    inputs {" >> config.pbtxt && \
        echo "        key: \"INPUT0\"" >> config.pbtxt && \
        echo "        value: {" >> config.pbtxt && \
        echo "            data_type: TYPE_FP32" >> config.pbtxt && \
        echo "            dims: 4" >> config.pbtxt && \
        echo "            zero_data: false" >> config.pbtxt && \
        echo "        }" >> config.pbtxt && \
        echo "    }" >> config.pbtxt && \
        echo "    inputs {" >> config.pbtxt && \
        echo "        key: \"INPUT1\"" >> config.pbtxt && \
        echo "        value: {" >> config.pbtxt && \
        echo "            data_type: TYPE_FP32" >> config.pbtxt && \
        echo "            dims: 4" >> config.pbtxt && \
        echo "            zero_data: false" >> config.pbtxt && \
        echo "        }" >> config.pbtxt && \
        echo "    }" >> config.pbtxt && \
        echo "}]" >> config.pbtxt )

SUB_TEST_RET=0
SERVER_LOG="./bls_model_loading_server_warmup.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
code=`curl -s -w %{http_code} -X POST localhost:8000/v2/repository/models/${MODEL_NAME}/load`
set -e
if [ "$code" != "200" ]; then
    echo -e "\n***\n*** Failed to load model '${MODEL_NAME}'\n***"
    RET=1
    SUB_TEST_RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

if [ $SUB_TEST_RET -eq 1 ]; then
    cat $CLIENT_LOG
    cat $SERVER_LOG
fi

# Test BLS parameters
rm -rf params_models && mkdir -p params_models/bls_parameters/1
cp ../../python_models/bls_parameters/model.py ./params_models/bls_parameters/1
cp ../../python_models/bls_parameters/config.pbtxt ./params_models/bls_parameters

TEST_LOG="./bls_parameters.log"
SERVER_LOG="./bls_parameters.server.log"

SERVER_ARGS="--model-repository=`pwd`/params_models --backend-directory=${BACKEND_DIR} --log-verbose=1"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python3 bls_parameters_test.py > $TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** bls_parameters_test.py FAILED. \n***"
    cat $TEST_LOG
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 1 ]; then
    echo -e "\n***\n*** BLS test FAILED. \n***"
else
    echo -e "\n***\n*** BLS test PASSED. \n***"
fi

exit $RET
