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

## This test tests the ability to use custom batching strategies with models.

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

BATCH_CUSTOM_TEST=batch_custom_test.py
CLIENT_LOG_BASE="./client.log"
DATADIR=/data/inferenceserver/${REPO_VERSION}/qa_identity_model_repository
EXPECTED_NUM_TESTS="1"
MODEL_NAME="onnx_zero_1_float16"
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=models --log-verbose 1"
SERVER_LOG_BASE="./inference_server.log"
TEST_RESULT_FILE='test_results.txt'
TRITON_BACKEND_REPO_TAG=${TRITON_BACKEND_REPO_TAG:="main"}
TRITON_CORE_REPO_TAG=${TRITON_CORE_REPO_TAG:="main"}

source ../common/util.sh
RET=0

# Batch strategy build requires recent version of CMake (FetchContent required)
# Using CMAKE installation instruction from:: https://apt.kitware.com/
apt update -q=2 \
    && apt install -y gpg wget \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - |  tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
    && . /etc/os-release \
    && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $UBUNTU_CODENAME main" | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
    && apt-get update -q=2 \
    && apt-get install -y --no-install-recommends cmake=3.27.7* cmake-data=3.27.7* rapidjson-dev
cmake --version

# Set up repository
rm -fr *.log* ./backend
rm -fr models && mkdir models
cp -r $DATADIR/$MODEL_NAME models

CONFIG_PATH="models/${MODEL_NAME}/config.pbtxt"
echo "dynamic_batching { max_queue_delay_microseconds: 10000}" >> ${CONFIG_PATH}
echo "instance_group [ { kind: KIND_GPU count: 2 }]" >> ${CONFIG_PATH}
echo "parameters { key: \"MAX_BATCH_VOLUME_BYTES\" value: {string_value: \"96\"}}" >> ${CONFIG_PATH}

# Create custom batching libraries
git clone --single-branch --depth=1 -b $TRITON_BACKEND_REPO_TAG \
    https://github.com/triton-inference-server/backend.git

(cd backend/examples/batching_strategies/volume_batching &&
 mkdir build &&
 cd build &&
 cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
       -DTRITON_CORE_REPO_TAG=$TRITON_CORE_REPO_TAG .. &&
 make -j4 install)

 (cd backend/examples/batching_strategies/single_batching &&
 mkdir build &&
 cd build &&
 cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
       -DTRITON_CORE_REPO_TAG=$TRITON_CORE_REPO_TAG .. &&
 make -j4 install)

cp -r backend/examples/batching_strategies/volume_batching/build/libtriton_volumebatching.so models
cp -r backend/examples/batching_strategies/single_batching/build/libtriton_singlebatching.so models

# Run a test to validate the single batching strategy example.
# Then, run tests to validate the volume batching example being passed in via the backend dir, model dir, version dir, and model config.
BACKEND_DIR="/opt/tritonserver/backends/onnxruntime"
MODEL_DIR="models/$MODEL_NAME"
VERSION_DIR="$MODEL_DIR/1/"

test_types=('single_batching_backend' 'backend_directory' 'model_directory' 'version_directory' 'model_config')
test_setups=("cp models/libtriton_singlebatching.so ${BACKEND_DIR}/batchstrategy.so && sed -i \"s/(4, 5, 6))/(12))/\" ${BATCH_CUSTOM_TEST}"
    "cp models/libtriton_volumebatching.so ${BACKEND_DIR}/batchstrategy.so && sed -i \"s/(12))/(4, 5, 6))/\" ${BATCH_CUSTOM_TEST}"
    "mv ${BACKEND_DIR}/batchstrategy.so ${MODEL_DIR} && cp models/libtriton_singlebatching.so ${BACKEND_DIR}"
    "mv ${MODEL_DIR}/batchstrategy.so ${VERSION_DIR}/batchstrategy.so"
    "mv ${VERSION_DIR}/batchstrategy.so models/${MODEL_NAME}/libtriton_volumebatching.so && echo \"parameters: {key: \\"TRITON_BATCH_STRATEGY_PATH\\", value: {string_value: \\"${MODEL_DIR}/libtriton_volumebatching.so\\"}}\" >> ${CONFIG_PATH}")

for i in "${!test_setups[@]}"; do
    echo "Running ${test_types[$i]} test"
    eval ${test_setups[$i]}

    SERVER_LOG=${SERVER_LOG_BASE}_${test_types[$i]}
    CLIENT_LOG=${CLIENT_LOG_BASE}_${test_types[$i]}

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi
    if [ `grep -c "Loading custom batching strategy" $SERVER_LOG` != "1" ]; then
        cat $SERVER_LOG
        echo -e "\n***\n*** Failed to load custom batching strategy.***"
        RET=1
    else
        set +e
        python $BATCH_CUSTOM_TEST >$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** ${test_types[$i]} Test Failed\n***"
            RET=1
        else
            check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG
                echo -e "\n***\n*** ${test_types[$i]} Test Result Verification Failed\n***"
                RET=1
            fi
        fi
        set -e
    fi

    kill $SERVER_PID
    wait $SERVER_PID
done

# Test ModelBatchInitialize failure
FILE_PATH="backend/examples/batching_strategies/volume_batching/src/volume_batching.cc"
OLD_STRING="\/\/ Batcher will point to an unsigned integer representing the maximum"
NEW_STRING="return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_NOT_FOUND,\"Failure test case\");"

sed -i "s/${OLD_STRING}/${NEW_STRING}/g" ${FILE_PATH}

(cd backend/examples/batching_strategies/volume_batching &&
 cd build &&
 cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
       -DTRITON_CORE_REPO_TAG=$TRITON_CORE_REPO_TAG .. &&
 make -j4 install)

cp -r backend/examples/batching_strategies/volume_batching/build/libtriton_volumebatching.so models/${MODEL_NAME}/libtriton_volumebatching.so

SERVER_LOG=${SERVER_LOG_BASE}_batching_init_failure

run_server
if [ "$SERVER_PID" != "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** ModelBatchInit Error Test: unexpected successful server start $SERVER\n***"
    kill_server
    RET=1
else
    if [ `grep -c "Failure test case" $SERVER_LOG` -lt 1 ] || [ `grep -c "Not found" $SERVER_LOG` -lt 1 ]; then
        cat $SERVER_LOG
        echo -e "\n***\n*** ModelBatchInit Error Test: failed to find \"Failure test case\" message and/or \"Not found\" error type"
        RET=1
    fi
fi


if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
