#!/bin/bash
# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

CLIENT_PY=./bls_test.py
CLIENT_LOG="./client.log"

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=`pwd`/all_models --model-control-mode=explicit --load-model=* --log-verbose=1"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

RET=0
rm -fr *.log ./all_models

git clone https://github.com/triton-inference-server/bls_backend -b $BLS_BACKEND_REPO_TAG
cp -r ./models ./all_models/
cp -r ./bls_backend/models/* ./all_models/

# Run the server with all the required models.
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $CLIENT_PY >> $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo "Failed: Client test had a non-zero return code."
    RET=1
fi

grep "PASS" $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** bls_test.py FAILED. \n***"
    cat $CLIENT_LOG
    cat $SERVER_LOG
    RET=1
fi
set -e

# Run the server without model 'addsub_python'.
code=`curl -s -w %{http_code} -o ./curl.out -X POST localhost:8000/v2/repository/models/addsub_python/unload`
if [ "$code" != "200" ]; then
    echo "Failed to unload 'addsub_python' model."
    RET=1
fi

set +e
python $CLIENT_PY >> $CLIENT_LOG 2>&1
grep "Failed to execute the inference request. Model 'addsub_python' is not ready." $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify model 'addsub_python' is unready. \n***"
    cat $SERVER_LOG
    RET=1
fi

# Run the server without model 'addsub_tf'.
code=`curl -s -w %{http_code} -o ./curl.out -X POST localhost:8000/v2/repository/models/addsub_python/load`
if [ "$code" != "200" ]; then
    echo "Failed to load 'addsub_python' model."
    RET=1
fi
code=`curl -s -w %{http_code} -o ./curl.out -X POST localhost:8000/v2/repository/models/addsub_tf/unload`
if [ "$code" != "200" ]; then
    echo "Failed to unload 'addsub_tf' model."
    RET=1
fi

set +e
python $CLIENT_PY >> $CLIENT_LOG 2>&1
grep "Failed to execute the inference request. Model 'addsub_tf' is not ready." $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify model 'addsub_tf' is unready. \n***"
    cat $SERVER_LOG
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
