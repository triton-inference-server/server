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

export CUDA_VISIBLE_DEVICES=0

CLIENT_PY=./identity_test.py
CLIENT_LOG="./client.log"

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=`pwd`/models"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

rm -fr *.log

RET=0

# The config is set with invalid checksum, so expect server failed to
# load all models
run_server
if [ "$SERVER_PID" == "0" ]; then
    set +e
    grep "'identity_int32': Mismatched MD5 hash for file 1/libtriton_identity.so" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected error on mismatched MD5 hash\n***"
        cat $SERVER_LOG
        RET=1
    fi
    set -e
else
    echo -e "\n***\n*** Expect fail to start $SERVER\n***"
    cat $SERVER_LOG
    kill $SERVER_PID
    wait $SERVER_PID
    exit 1
fi

# Set correct md5sum
(cd models/identity_int32 && \
    model_hash=$(md5sum 1/libtriton_identity.so | cut -d' ' -f 1); sed -i "s/invalid_checksum/${model_hash}/" config.pbtxt
)

# Server should run successfully
rm -fr *.log
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** fail to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

for PROTOCOL in http grpc; do
    set +e
    python $CLIENT_PY -i $PROTOCOL -v >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi
    set -e
done

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
    cat $SERVER_LOG
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
