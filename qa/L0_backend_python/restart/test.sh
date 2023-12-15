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

CLIENT_LOG="./restart_client.log"
EXPECTED_NUM_TESTS="7"
TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
BACKEND_DIR=${TRITON_DIR}/backends
SERVER_ARGS="--model-repository=`pwd`/models --backend-directory=${BACKEND_DIR} --log-verbose=1"
SERVER_LOG="./restart_server.log"
source ../../common/util.sh
source ../common.sh

rm -fr *.log free_memory.txt

RET=0

prev_num_pages=`get_shm_pages`
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python3 restart_test.py RestartTest.test_infer >> $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    cat $SERVER_LOG
    echo -e "\n***\n*** test_infer test FAILED. \n***"
    RET=1
fi
set -e

triton_procs=`pgrep --parent $SERVER_PID`
echo $triton_procs

set +e
for proc in $triton_procs; do
    kill -9 $proc
done

python3 restart_test.py RestartTest.test_restart >> $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    cat $SERVER_LOG
    echo -e "\n***\n*** test_restart test FAILED. \n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

current_num_pages=`get_shm_pages`
if [ $current_num_pages -ne $prev_num_pages ]; then
    cat $CLIENT_LOG
    ls /dev/shm
    echo -e "\n***\n*** Test Failed. Shared memory pages where not cleaned properly.
Shared memory pages before starting triton equals to $prev_num_pages
and shared memory pages after starting triton equals to $current_num_pages \n***"
    exit 1
fi

# Test if the Triton server exits gracefully when the stub has been killed.
rm $SERVER_LOG
prev_num_pages=`get_shm_pages`
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

triton_procs=`pgrep --parent $SERVER_PID`
echo $triton_procs

set +e
for proc in $triton_procs; do
    kill -9 $proc
done
set -e

kill $SERVER_PID
wait $SERVER_PID

current_num_pages=`get_shm_pages`
if [ $current_num_pages -ne $prev_num_pages ]; then
    cat $CLIENT_LOG
    ls /dev/shm
    echo -e "\n***\n*** Test Failed. Shared memory pages where not cleaned properly.
Shared memory pages before starting triton equals to $prev_num_pages
and shared memory pages after starting triton equals to $current_num_pages \n***"
    exit 1
fi

if [ $RET -eq 1 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Restart test FAILED. \n***"
else
    echo -e "\n***\n*** Restart test PASSED. \n***"
fi

exit $RET
