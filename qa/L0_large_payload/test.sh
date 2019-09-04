#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

LARGE_PAYLOAD_TEST_PY=large_payload_test.py

CLIENT_LOG_BASE="./client.log"

DATADIR=`pwd`/models

SERVER=/opt/tensorrtserver/bin/trtserver
SERVER_ARGS=--model-repository=$DATADIR
SERVER_LOG_BASE="./inference_server.log"
source ../common/util.sh

rm -f $SERVER_LOG_BASE* $CLIENT_LOG_BASE*

RET=0

MODEL_SUFFIX=nobatch_zero_1_float32
rm -fr models && \
    mkdir models && \
    cp -r /data/inferenceserver/$1/qa_identity_model_repository/graphdef_$MODEL_SUFFIX models/. && \
    cp -r /data/inferenceserver/$1/qa_identity_model_repository/netdef_$MODEL_SUFFIX models/. && \
    cp -r /data/inferenceserver/$1/qa_identity_model_repository/onnx_$MODEL_SUFFIX models/. && \
    cp -r /data/inferenceserver/$1/qa_identity_model_repository/savedmodel_$MODEL_SUFFIX models/. && \
    cp -r /data/inferenceserver/$1/qa_identity_model_repository/libtorch_$MODEL_SUFFIX models/. 
cp -r ../custom_models/custom_zero_1_float32 models/. && \
    mkdir -p models/custom_zero_1_float32/1 && \
    cp `pwd`/libidentity.so models/custom_zero_1_float32/1/. && \
    (cd models/custom_zero_1_float32 && \
            echo "default_model_filename: \"libidentity.so\"" >> config.pbtxt && \
            echo "instance_group [ { kind: KIND_CPU }]" >> config.pbtxt && \
            sed -i "s/dims: \[ 1 \]/dims: \[ -1 \]/" config.pbtxt)

# Restart server before every test to make sure server state
# is invariant to previous test
#
# Skipping TensorRT Plan model for now as it only supports fixed size
# tensor and it fails to generate layer with large dimension size
# [TODO] Revisit this once TensorRT supports variable size tensor
for TARGET in graphdef savedmodel netdef onnx libtorch custom; do
    SERVER_LOG=$SERVER_LOG_BASE.$TARGET
    CLIENT_LOG=$CLIENT_LOG_BASE.$TARGET

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e

    python $LARGE_PAYLOAD_TEST_PY LargePayLoadTest.test_$TARGET >$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi

    grep -c "HTTP/1.1 200 OK" $CLIENT_LOG
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed To Run\n***"
        RET=1
    fi

    set -e

    kill $SERVER_PID
    wait $SERVER_PID
done

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
fi

exit $RET
