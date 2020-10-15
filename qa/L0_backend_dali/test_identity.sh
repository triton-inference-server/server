#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

DALI_BACKEND_DIR=$1

CLIENT_PY=$DALI_BACKEND_DIR/qa/identity/identity_client.py
CLIENT_LOG="./client.log"

MODEL_REPO=$DALI_BACKEND_DIR/docs/examples/identity/model_repository

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=$MODEL_REPO"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

pushd $MODEL_REPO/..
sh setup_identity_example.sh
popd

run_server
if [ "$SERVER_PID" == "0" ]; then
  echo -e "\n***\n*** Failed to start $SERVER $SERVER_ARGS\n***"
  cat $SERVER_LOG
  exit 1
fi

RET=0

set +e
python $CLIENT_PY --model_name dali_identity -v --batch_size 32 >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
  RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  cat $CLIENT_LOG
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
