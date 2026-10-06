#!/bin/bash
# Copyright 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

UNITTEST_PY=./io_test.py
CLIENT_LOG="./io_client.log"
TEST_RESULT_FILE='test_results.txt'
source ../common.sh
source ../../common/util.sh

SERVER_ARGS="--model-repository=${MODELDIR}/io/models --backend-directory=${BACKEND_DIR} --log-verbose=1"
SERVER_LOG="./io_server.log"

RET=0
rm -fr *.log ./models

pip3 uninstall -y torch
pip3 install torch==2.3.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# IOTest.test_ensemble_io
TRIALS="default decoupled"

# Set up the ensemble_io models for the given trial (default|decoupled).
setup_ensemble_io_models() {
    rm -rf ./models
    if [ $1 = "default" ]; then
        src_model=dlpack_io_identity
    else
        src_model=dlpack_io_identity_decoupled
    fi
    for i in {1..3}; do
        model_name=dlpack_io_identity_$i
        mkdir -p models/$model_name/1/
        cp ../../python_models/$src_model/model.py ./models/$model_name/1/
        cp ../../python_models/$src_model/config.pbtxt ./models/$model_name/
        (cd models/$model_name && \
                  sed -i "s/^name:.*/name: \"$model_name\"/" config.pbtxt)
    done
    mkdir -p models/ensemble_io/1/
    cp ../../python_models/ensemble_io/config.pbtxt ./models/ensemble_io
}

for trial in $TRIALS; do
    export TRIAL=$trial
    setup_ensemble_io_models $trial

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        RET=1
    fi

    set +e
    SUBTEST="test_ensemble_io"
    python3 -m pytest --junitxml=${SUBTEST}.${TRIAL}.report.xml ${UNITTEST_PY}::IOTest::${SUBTEST} >> ${CLIENT_LOG}.${SUBTEST}
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** IOTest.${SUBTEST} FAILED. \n***"
        cat $CLIENT_LOG.${SUBTEST}
        RET=1
    fi
    set -e

    kill $SERVER_PID
    wait $SERVER_PID
done

# IOTest.test_ensemble_io with GPU outputs falling back to pinned memory
CUDA_MEMORY_POOL_SIZE_BYTES=1024
SAVED_SERVER_ARGS="${SERVER_ARGS}"
SERVER_ARGS="${SERVER_ARGS} --cuda-memory-pool-byte-size=0:${CUDA_MEMORY_POOL_SIZE_BYTES}"
SUBTEST="test_ensemble_io"
for trial in $TRIALS; do
    export TRIAL=$trial
    setup_ensemble_io_models $trial
    SERVER_LOG="./io_server.cuda_pool_fallback.${TRIAL}.log"

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        RET=1
        continue
    fi

    set +e
    python3 -m pytest --junitxml=${SUBTEST}.cuda_pool_fallback.${TRIAL}.report.xml ${UNITTEST_PY}::IOTest::${SUBTEST} > ${CLIENT_LOG}.${SUBTEST}.cuda_pool_fallback.${TRIAL}
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** IOTest.${SUBTEST} (CUDA pool fallback, ${TRIAL}) FAILED. \n***"
        cat ${CLIENT_LOG}.${SUBTEST}.cuda_pool_fallback.${TRIAL}
        RET=1
    fi

    kill $SERVER_PID
    wait $SERVER_PID
    set -e

    if ! grep -q "falling back to pinned system memory" $SERVER_LOG; then
        echo -e "\n***\n*** Expected the CUDA memory pool to be exhausted and fall back to pinned memory (${TRIAL}). \n***"
        RET=1
    fi
done
SERVER_ARGS="${SAVED_SERVER_ARGS}"
SERVER_LOG="./io_server.log"

# IOTest.test_empty_gpu_output
rm -rf models && mkdir models
mkdir -p models/dlpack_empty_output/1/
cp ../../python_models/dlpack_empty_output/model.py ./models/dlpack_empty_output/1/
cp ../../python_models/dlpack_empty_output/config.pbtxt ./models/dlpack_empty_output/

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

set +e
SUBTEST="test_empty_gpu_output"
python3 -m pytest --junitxml=${SUBTEST}.report.xml ${UNITTEST_PY}::IOTest::${SUBTEST} > ${CLIENT_LOG}.${SUBTEST}

if [ $? -ne 0 ]; then
    echo -e "\n***\n*** IOTest.${SUBTEST} FAILED. \n***"
    cat $CLIENT_LOG.${SUBTEST}
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# IOTest.test_variable_gpu_output
rm -rf models && mkdir models
mkdir -p models/variable_gpu_output/1/
cp ../../python_models/variable_gpu_output/model.py ./models/variable_gpu_output/1/
cp ../../python_models/variable_gpu_output/config.pbtxt ./models/variable_gpu_output/

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

set +e
SUBTEST="test_variable_gpu_output"
python3 -m pytest --junitxml=${SUBTEST}.report.xml ${UNITTEST_PY}::IOTest::${SUBTEST} > ${CLIENT_LOG}.${SUBTEST}

if [ $? -ne 0 ]; then
    echo -e "\n***\n*** IOTest.${SUBTEST} FAILED. \n***"
    cat $CLIENT_LOG.${SUBTEST}
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# IOTest.test_requested_output_default & IOTest.test_requested_output_decoupled
rm -rf models && mkdir models
mkdir -p models/add_sub/1/
cp ../../python_models/add_sub/model.py ./models/add_sub/1/
cp ../../python_models/add_sub/config.pbtxt ./models/add_sub/
mkdir -p models/dlpack_io_identity_decoupled/1/
cp ../../python_models/dlpack_io_identity_decoupled/model.py ./models/dlpack_io_identity_decoupled/1/
cp ../../python_models/dlpack_io_identity_decoupled/config.pbtxt ./models/dlpack_io_identity_decoupled/

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

SUBTESTS="test_requested_output_default test_requested_output_decoupled"
for SUBTEST in $SUBTESTS; do
    set +e
    python3 -m pytest --junitxml=${SUBTEST}.report.xml ${UNITTEST_PY}::IOTest::${SUBTEST} > ${CLIENT_LOG}.${SUBTEST}
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** IOTest.${SUBTEST} FAILED. \n***"
        cat $CLIENT_LOG.${SUBTEST}
        RET=1
    fi
    set -e
done

kill $SERVER_PID
wait $SERVER_PID

# IOTest.test_requested_output_decoupled_prior_crash
rm -rf models && mkdir models
mkdir -p models/llm/1/
cp requested_output_model/config.pbtxt models/llm/
cp requested_output_model/model.py models/llm/1/

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

SUBTEST="test_requested_output_decoupled_prior_crash"
set +e
python3 -m pytest --junitxml=${SUBTEST}.report.xml ${UNITTEST_PY}::IOTest::${SUBTEST} > ${CLIENT_LOG}.${SUBTEST}
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** IOTest.${SUBTEST} FAILED. \n***"
    cat $CLIENT_LOG.${SUBTEST}
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** IO test PASSED.\n***"
else
    echo -e "\n***\n*** IO test FAILED.\n***"
fi

exit $RET
