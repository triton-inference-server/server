#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

source ../common.sh
source ../../common/util.sh

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_LOG="./inference_server.log"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}

RET=0
rm -fr *.log python_backend/

pip3 uninstall -y torch
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

git clone https://github.com/triton-inference-server/python_backend -b $PYTHON_BACKEND_REPO_TAG
cd python_backend
SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1"

# Example 1
mkdir -p models/add_sub/1/
cp examples/add_sub/model.py models/add_sub/1/model.py
cp examples/add_sub/config.pbtxt models/add_sub/config.pbtxt
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

set +e
python3 examples/add_sub/client.py > add_sub_client.log
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify add_sub example. \n***"
    RET=1
fi

grep "PASS" add_sub_client.log
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify pytorch example. \n***"
    cat pytorch_client.log
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# Example 2
mkdir -p models/pytorch/1/
cp examples/pytorch/model.py models/pytorch/1/model.py
cp examples/pytorch/config.pbtxt models/pytorch/config.pbtxt
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

set +e
python3 examples/pytorch/client.py > pytorch_client.log
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify pytorch example. \n***"
    RET=1
fi

grep "PASS" pytorch_client.log
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify pytorch example. \n***"
    cat pytorch_client.log
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# Example 3

# BLS Sync
mkdir -p models/bls_sync/1
cp examples/bls/sync_model.py models/bls_sync/1/model.py
cp examples/bls/sync_config.pbtxt models/bls_sync/config.pbtxt
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

set +e
python3 examples/bls/sync_client.py > sync_client.log
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify BLS sync example. \n***"
    RET=1
fi

grep "PASS" sync_client.log
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify BLS sync example. \n***"
    cat sync_client.log
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# BLS Async
mkdir -p models/bls_async/1
cp examples/bls/async_model.py models/bls_async/1/model.py
cp examples/bls/async_config.pbtxt models/bls_async/config.pbtxt
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

set +e
python3 examples/bls/async_client.py > async_client.log
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify BLS sync example. \n***"
    cat async_client.log
    RET=1
fi

grep "PASS" async_client.log
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify BLS sync example. \n***"
    cat async_client.log
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Example verification test PASSED.\n***"
else
    echo -e "\n***\n*** Example verification test FAILED.\n***"
fi

exit $RET
