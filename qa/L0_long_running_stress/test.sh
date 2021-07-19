#!/bin/bash
# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0

MODEL_DIR=all_models
BACKENDS=${BACKENDS:="graphdef savedmodel onnx libtorch plan python"}

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_LOG="./inference_server.log"
SERVER_ARGS="--model-repository=`pwd`/$MODEL_DIR"
source ../common/util.sh

rm -fr all_models/*float32 *.log *.serverlog RET.txt

RET=0
echo "$RET" > RET.txt

# Get Parallel for sendling multiple requests simultaneously
apt-get -y update
apt-get install -y parallel

# Setup nobatch models
NOBATCH_MODEL_SUFFIX=nobatch_zero_1_float32
for TARGET in $BACKENDS; do
    if [ "$TARGET" != "python" ]; then
        cp -r /data/inferenceserver/${REPO_VERSION}/qa_identity_model_repository/${TARGET}_$NOBATCH_MODEL_SUFFIX \
            all_models/.
    else
        mkdir -p all_models/python_$NOBATCH_MODEL_SUFFIX/1/
        cp ../python_models/identity_fp32/config.pbtxt all_models/python_$NOBATCH_MODEL_SUFFIX/
        (cd all_models/python_$NOBATCH_MODEL_SUFFIX && \
                    sed -i "s/max_batch_size: 64/max_batch_size: 0/" config.pbtxt && \
                    sed -i "s/name: \"identity_fp32\"/name: \"python_$NOBATCH_MODEL_SUFFIX\"/" config.pbtxt)

        cp ../python_models/identity_fp32/model.py all_models/python_$NOBATCH_MODEL_SUFFIX/1/model.py
    fi
done

# Setup sequence models - four instances with batch-size 1
for m in ../custom_models/custom_sequence_int32 ; do
    cp -r $m all_models/ && \
        (cd all_models/custom_sequence_int32 && \
            sed -i "s/max_sequence_idle_microseconds:.*/max_sequence_idle_microseconds: 1000000/" config.pbtxt && \
            sed -i "s/^max_batch_size:.*/max_batch_size: 1/" config.pbtxt && \
            sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 4/" config.pbtxt && \
            sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 4/" config.pbtxt)
done

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

# Send multiple requests simultaneously
parallel -u ::: 'bash stress.sh' 'bash nobatch_stress.sh' 'bash timeout_stress.sh'

set -e

kill $SERVER_PID
wait $SERVER_PID

RET=`cat RET.txt`
if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $SEQUENCE_CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
