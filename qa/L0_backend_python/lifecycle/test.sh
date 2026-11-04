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

CLIENT_LOG="./lifecycle_client.log"
TEST_RESULT_FILE='test_results.txt'
source ../common.sh
source ../../common/util.sh

SERVER_ARGS="--model-repository=${MODELDIR}/lifecycle/models --backend-directory=${BACKEND_DIR} --log-verbose=1"
if [ "${TEST_JETSON}" == "1" ]; then
    SERVER_ARGS="${SERVER_ARGS} --exit-timeout-secs=${SERVER_TIMEOUT} --backend-config=python,stub-timeout-seconds=${SERVER_TIMEOUT}"
fi
SERVER_LOG="./lifecycle_server.log"

RET=0
rm -fr *.log ./models

mkdir -p models/error_code/1/
cp ../../python_models/error_code/model.py ./models/error_code/1/
cp ../../python_models/error_code/config.pbtxt ./models/error_code/

mkdir -p models/execute_cancel/1/
cp ../../python_models/execute_cancel/model.py ./models/execute_cancel/1/
cp ../../python_models/execute_cancel/config.pbtxt ./models/execute_cancel/

mkdir -p models/execute_error/1/
cp ../../python_models/execute_error/model.py ./models/execute_error/1/
cp ../../python_models/execute_error/config.pbtxt ./models/execute_error/
(cd models/execute_error && \
          sed -i "s/^name:.*/name: \"execute_error\"/" config.pbtxt && \
          sed -i "s/^max_batch_size:.*/max_batch_size: 8/" config.pbtxt && \
          echo "dynamic_batching { preferred_batch_size: [8], max_queue_delay_microseconds: 12000000 }" >> config.pbtxt)

mkdir -p models/execute_grpc_error/1/
cp ../../python_models/execute_grpc_error/model.py ./models/execute_grpc_error/1/
cp ../../python_models/execute_grpc_error/config.pbtxt ./models/execute_grpc_error/
(cd models/execute_grpc_error && \
          sed -i "s/^name:.*/name: \"execute_grpc_error\"/" config.pbtxt && \
          sed -i "s/^max_batch_size:.*/max_batch_size: 8/" config.pbtxt && \
          echo "dynamic_batching { preferred_batch_size: [8], max_queue_delay_microseconds: 1200000 }" >> config.pbtxt)

mkdir -p models/decoupled_grpc_error/1/
cp ../../python_models/decoupled_grpc_error/model.py ./models/decoupled_grpc_error/1/
cp ../../python_models/decoupled_grpc_error/config.pbtxt ./models/decoupled_grpc_error/

mkdir -p models/execute_return_error/1/
cp ../../python_models/execute_return_error/model.py ./models/execute_return_error/1/
cp ../../python_models/execute_return_error/config.pbtxt ./models/execute_return_error/

mkdir -p models/wrong_model/1/
cp ../../python_models/wrong_model/model.py ./models/wrong_model/1/
cp ../../python_models/wrong_model/config.pbtxt ./models/wrong_model/
(cd models/wrong_model && \
          sed -i "s/^name:.*/name: \"wrong_model\"/" config.pbtxt && \
          sed -i "s/TYPE_FP32/TYPE_UINT32/g" config.pbtxt)

prev_num_pages=`get_shm_pages`

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

set +e

# Run this multiple times to catch any intermittent segfault.
for i in {0..4}; do
    python3 -m pytest --junitxml=lifecycle.iter${i}.report.xml lifecycle_test.py >> $CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** lifecycle_test.py FAILED. \n***"
        RET=1
    fi
done

set -e

kill_server

current_num_pages=`get_shm_pages`
if [ $current_num_pages -ne $prev_num_pages ]; then
    ls /dev/shm
    echo -e "\n***\n*** Test Failed. Shared memory pages were not cleaned properly.
Shared memory pages before starting triton equals to $prev_num_pages
and shared memory pages after starting triton equals to $current_num_pages \n***"
    RET=1
fi

# These models have errors in the initialization and finalization
# steps and we want to ensure that correct error is being returned

rm -rf models/
mkdir -p models/init_error/1/
cp ../../python_models/init_error/model.py ./models/init_error/1/
cp ../../python_models/init_error/config.pbtxt ./models/init_error/

set +e
prev_num_pages=`get_shm_pages`
run_server_nowait

wait $SERVER_PID
current_num_pages=`get_shm_pages`
if [ $current_num_pages -ne $prev_num_pages ]; then
    ls /dev/shm
    echo -e "\n***\n*** Test Failed. Shared memory pages were not cleaned properly.
Shared memory pages before starting triton equals to $prev_num_pages
and shared memory pages after starting triton equals to $current_num_pages \n***"
    RET=1
fi

grep "name 'lorem_ipsum' is not defined" $SERVER_LOG

if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** init_error model test failed \n***"
    RET=1
fi
set -e

# FIXME: Until we find a way to simulate Ctrl^C on windows, this
# test will not pass.
if [[ ${TEST_WINDOWS} == 0 ]]; then
    rm -rf models/
    mkdir -p models/fini_error/1/
    cp ../../python_models/fini_error/model.py ./models/fini_error/1/
    cp ../../python_models/fini_error/config.pbtxt ./models/fini_error/

    prev_num_pages=`get_shm_pages`
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        RET=1
    fi

    kill_server

    current_num_pages=`get_shm_pages`
    if [ $current_num_pages -ne $prev_num_pages ]; then
        cat $CLIENT_LOG
        ls /dev/shm
        echo -e "\n***\n*** Test Failed. Shared memory pages were not cleaned properly.
    Shared memory pages before starting triton equals to $prev_num_pages
    and shared memory pages after starting triton equals to $current_num_pages \n***"
        RET=1
    fi

    set +e
    grep "name 'undefined_variable' is not defined" $SERVER_LOG

    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** fini_error model test failed \n***"
        RET=1
    fi
    set -e
fi

# A stub that stops responding while its model initializes must fail the model
# load instead of hanging the server. The model freezes its own stub until the
# server gives up on it, so the server has to terminate a stub that still
# considers itself initialized and then exit because a model failed to load.
# SIGSTOP is not available on Windows.
if [[ ${TEST_WINDOWS} == 0 ]]; then
    rm -rf models/
    mkdir -p models/init_unresponsive/1/
    cp ../../python_models/init_unresponsive/model.py ./models/init_unresponsive/1/
    cp ../../python_models/init_unresponsive/config.pbtxt ./models/init_unresponsive/

    # Use a fixed stub timeout, independent of platform defaults, that is long
    # enough for the stub to resume and sit idle before it is killed.
    UNRESPONSIVE_STUB_TIMEOUT=20
    SAVED_SERVER_ARGS="${SERVER_ARGS}"
    SERVER_ARGS="--model-repository=${MODELDIR}/lifecycle/models --backend-directory=${BACKEND_DIR} --log-verbose=1 --backend-config=python,stub-timeout-seconds=${UNRESPONSIVE_STUB_TIMEOUT}"

    set +e
    prev_num_pages=`get_shm_pages`
    run_server_nowait
    SERVER_ARGS="${SAVED_SERVER_ARGS}"

    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        RET=1
    else
        # Startup, the 15 second stall and the stub timeout fit well inside
        # this deadline.
        for i in $(seq 1 $((UNRESPONSIVE_STUB_TIMEOUT + 100))); do
            kill -0 $SERVER_PID 2> /dev/null || break
            sleep 1
        done
        if kill -0 $SERVER_PID 2> /dev/null; then
            cat $SERVER_LOG
            echo -e "\n***\n*** init_unresponsive: server did not exit after the model failed to load \n***"
            kill -9 $SERVER_PID
            pkill -9 -f "init_unresponsive/1/model.py"
            sleep 1
            RET=1
        fi
        wait $SERVER_PID
    fi

    current_num_pages=`get_shm_pages`
    if [ $current_num_pages -ne $prev_num_pages ]; then
        ls /dev/shm
        echo -e "\n***\n*** Test Failed. Shared memory pages were not cleaned properly.
    Shared memory pages before starting triton equals to $prev_num_pages
    and shared memory pages after starting triton equals to $current_num_pages \n***"
        RET=1
    fi

    # The load must fail because the server gave up on the stub, not because
    # of some other error in the model. If the stub is stopped while holding
    # the health mutex, the server reports that instead.
    grep -E "failed to load 'init_unresponsive'.*(Stub process 'init_unresponsive_0_0' is not healthy|Failed to obtain the health mutex)" $SERVER_LOG
    if [ $? -ne 0 ]; then
        cat $SERVER_LOG
        echo -e "\n***\n*** init_unresponsive model test failed \n***"
        RET=1
    fi

    if pgrep -f "init_unresponsive/1/model.py" > /dev/null; then
        ps -eo pid,stat,args | grep "[i]nit_unresponsive"
        echo -e "\n***\n*** init_unresponsive: stub process was left running \n***"
        RET=1
    fi
    set -e
fi

rm -rf models/
mkdir -p models/auto_complete_error/1/
cp ../../python_models/auto_complete_error/model.py ./models/auto_complete_error/1/

SERVER_ARGS="${SERVER_ARGS} --strict-model-config=false"

set +e
prev_num_pages=`get_shm_pages`
run_server_nowait

wait $SERVER_PID
current_num_pages=`get_shm_pages`
if [ $current_num_pages -ne $prev_num_pages ]; then
    ls /dev/shm
    echo -e "\n***\n*** Test Failed. Shared memory pages were not cleaned properly.
Shared memory pages before starting triton equals to $prev_num_pages
and shared memory pages after starting triton equals to $current_num_pages \n***"
    RET=1
fi

set +e
grep "name 'undefined_variable' is not defined" $SERVER_LOG

if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** auto_complete_error model test failed \n***"
    RET=1
fi
set -e

if [ $RET -eq 1 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Lifecycle test FAILED. \n***"
else
    echo -e "\n***\n*** Lifecycle test PASSED. \n***"
fi

exit $RET
