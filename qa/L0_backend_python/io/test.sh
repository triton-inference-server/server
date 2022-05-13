#!/bin/bash
# Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
CLIENT_LOG="./client.log"
EXPECTED_NUM_TESTS="1"
TEST_RESULT_FILE='test_results.txt'
source ../common.sh
source ../../common/util.sh

TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
BACKEND_DIR=${TRITON_DIR}/backends

SERVER_ARGS="--model-repository=`pwd`/models --backend-directory=${BACKEND_DIR} --log-verbose=1"
SERVER_LOG="./inference_server.log"

RET=0
rm -fr *.log ./models

pip3 uninstall -y torch
pip3 install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

TRIALS="default decoupled"

for trial in $TRIALS; do
    export TRIAL=$trial
    rm -rf ./models

    if [ $trial = "default" ]; then
        for i in {1..3}; do
            model_name=dlpack_io_identity_$i
            mkdir -p models/$model_name/1/
            cp ../../python_models/dlpack_io_identity/model.py ./models/$model_name/1/
            cp ../../python_models/dlpack_io_identity/config.pbtxt ./models/$model_name/
            (cd models/$model_name && \
                      sed -i "s/^name:.*/name: \"$model_name\"/" config.pbtxt)
        done
    else
        for i in {1..3}; do
            model_name=dlpack_io_identity_$i
            mkdir -p models/$model_name/1/
            cp ../../python_models/dlpack_io_identity_decoupled/model.py ./models/$model_name/1/
            cp ../../python_models/dlpack_io_identity_decoupled/config.pbtxt ./models/$model_name/
            (cd models/$model_name && \
                      sed -i "s/^name:.*/name: \"$model_name\"/" config.pbtxt)
        done
    fi

    mkdir -p models/ensemble_io/1/
    cp ../../python_models/ensemble_io/config.pbtxt ./models/ensemble_io

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        RET=1
    fi

    set +e
    python3 $UNITTEST_PY > $CLIENT_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** io_test.py FAILED. \n***"
        cat $CLIENT_LOG
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
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** IO test PASSED.\n***"
else
    echo -e "\n***\n*** IO test FAILED.\n***"
fi

exit $RET
