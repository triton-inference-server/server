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

CLIENT_LOG="./ensemble_client.log"
EXPECTED_NUM_TESTS="2"
TEST_RESULT_FILE='test_results.txt'
source ../common.sh
source ../../common/util.sh

TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
BACKEND_DIR=${TRITON_DIR}/backends
SERVER_ARGS="--model-repository=`pwd`/models --backend-directory=${BACKEND_DIR} --log-verbose=1"
SERVER_LOG="./ensemble_server.log"

RET=0
rm -rf models/ $CLIENT_LOG

# Ensemble Model
mkdir -p models/ensemble/1/
cp ../../python_models/ensemble/config.pbtxt ./models/ensemble

mkdir -p models/add_sub_1/1/
cp ../../python_models/add_sub/config.pbtxt ./models/add_sub_1
cp ../../python_models/add_sub/model.py ./models/add_sub_1/1/

mkdir -p models/add_sub_2/1/
cp ../../python_models/add_sub/config.pbtxt ./models/add_sub_2/
cp ../../python_models/add_sub/model.py ./models/add_sub_2/1/

# Ensemble GPU Model
mkdir -p models/ensemble_gpu/1/
cp ../../python_models/ensemble_gpu/config.pbtxt ./models/ensemble_gpu
cp -r ${DATADIR}/qa_model_repository/libtorch_float32_float32_float32/ ./models
(cd models/libtorch_float32_float32_float32 && \
          echo "instance_group [ { kind: KIND_GPU }]" >> config.pbtxt)
(cd models/libtorch_float32_float32_float32 && \
          sed -i "s/^max_batch_size:.*/max_batch_size: 0/" config.pbtxt)
(cd models/libtorch_float32_float32_float32 && \
          sed -i "s/^version_policy:.*//" config.pbtxt)
rm -rf models/libtorch_float32_float32_float32/2
rm -rf models/libtorch_float32_float32_float32/3

prev_num_pages=`get_shm_pages`

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

set +e
python3 ensemble_test.py 2>&1 > $CLIENT_LOG

if [ $? -ne 0 ]; then
    echo -e "\n***\n*** ensemble_test.py FAILED. \n***"
    RET=1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

current_num_pages=`get_shm_pages`
if [ $current_num_pages -ne $prev_num_pages ]; then
    ls /dev/shm
    echo -e "\n***\n*** Test Failed. Shared memory pages where not cleaned properly.
Shared memory pages before starting triton equals to $prev_num_pages
and shared memory pages after starting triton equals to $current_num_pages \n***"
    RET=1
fi

if [ $RET -eq 1 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Ensemble test FAILED. \n***"
else
    echo -e "\n***\n*** Ensemble test PASSED. \n***"
fi

exit $RET
