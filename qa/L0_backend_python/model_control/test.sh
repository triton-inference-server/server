#!/bin/bash
# Copyright 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

CLIENT_LOG="./model_control_client.log"
TEST_RESULT_FILE='test_results.txt'
SERVER_ARGS="--model-repository=${MODELDIR}/model_control/models --model-control-mode=explicit --backend-directory=${BACKEND_DIR} --log-verbose=1"
SERVER_LOG="./model_control_server.log"

RET=0
rm -fr *.log ./models

source ../../common/util.sh

mkdir -p models/identity_fp32/1/
mkdir -p models/simple_identity_fp32/1/
cp ../../python_models/identity_fp32/model.py ./models/identity_fp32/1/model.py
cp ../../python_models/identity_fp32/config.pbtxt ./models/identity_fp32/config.pbtxt
cp ../../python_models/simple_identity_fp32/config.pbtxt ./models/simple_identity_fp32/config.pbtxt

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python3 -m pytest --junitxml=model_control.report.xml model_control_test.py 2>&1 > $CLIENT_LOG

if [ $? -ne 0 ]; then
    echo -e "\n***\n*** model_control_test.py FAILED. \n***"
    RET=1
fi
set -e

kill_server

if [ $RET -eq 1 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** model_control_test FAILED. \n***"
else
    echo -e "\n***\n*** model_control_test PASSED. \n***"
fi

exit $RET
