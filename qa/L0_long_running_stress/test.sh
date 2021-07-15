#!/bin/bash
# Copyright 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

SEQUENCE_CLIENT_LOG="./sequnce_client.log"
NOBATCH_CLIENT_LOG_BASE="./nobatch_client"
MODEL_DIR=all_models

SEQUENCE_STRESS_TEST=sequence_stress.py
NOBATCH_STRESS_TEST=nobatch_stress.py

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_LOG="./inference_server.log"
SERVER_ARGS="--model-repository=`pwd`/$MODEL_DIR"
source ../common/util.sh

RET=0

rm -fr all_models *.log *.serverlog && mkdir all_models

# Setup nobatch models
NOBATCH_MODEL_SUFFIX=nobatch_zero_1_float32
for TARGET in graphdef savedmodel onnx libtorch plan; do
    cp -r /data/inferenceserver/${REPO_VERSION}/qa_identity_model_repository/${TARGET}_$NOBATCH_MODEL_SUFFIX \
       all_models/.
done
mkdir -p all_models/python_$NOBATCH_MODEL_SUFFIX/1/
cp ../python_models/identity_fp32/config.pbtxt all_models/python_$NOBATCH_MODEL_SUFFIX/
(cd all_models/python_$NOBATCH_MODEL_SUFFIX && \
            sed -i "s/max_batch_size: 64/max_batch_size: 0/" config.pbtxt && \
            sed -i "s/name: \"identity_fp32\"/name: \"python_$NOBATCH_MODEL_SUFFIX\"/" config.pbtxt)

cp ../python_models/identity_fp32/model.py all_models/python_$NOBATCH_MODEL_SUFFIX/1/model.py

# Setup sequence models - four instances with batch-size 1
for m in ../custom_models/custom_sequence_int32 ; do
    cp -r $m all_models/ && \
        (cd all_models/custom_sequence_int32 && \
            sed -i "s/max_sequence_idle_microseconds:.*/max_sequence_idle_microseconds: 1000000/" config.pbtxt && \
            sed -i "s/^max_batch_size:.*/max_batch_size: 1/" config.pbtxt && \
            sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 4/" config.pbtxt && \
            sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 4/" config.pbtxt)
done

#Setup batched models



run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

for TARGET in graphdef savedmodel onnx libtorch plan python; do
    set +e

    NOBATCH_CLIENT_LOG="$NOBATCH_CLIENT_LOG_BASE.$TARGET.log"
    python $NOBATCH_STRESS_TEST NoBatchStressTest.test_$TARGET >$NOBATCH_CLIENT_LOG 2>&1 &
    python $SEQUENCE_STRESS_TEST >>$SEQUENCE_CLIENT_LOG 2>&1 &
    
    if [ $? -ne 0 ]; then
        cat $NOBATCH_CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    else
        check_test_results $NOBATCH_CLIENT_LOG 1
        if [ $? -ne 0 ]; then
            cat $NOBATCH_CLIENT_LOG
            echo -e "\n***\n*** Test Result Verification Failed\n***"
            RET=1
        fi
    fi

    set -e
done

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
