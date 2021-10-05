#!/bin/bash
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
if [ ! -z "$TEST_REPO_ARCH" ]; then
    REPO_VERSION=${REPO_VERSION}_${TEST_REPO_ARCH}
fi

export CUDA_VISIBLE_DEVICES=0

CLIENT=../clients/perf_client
BACKENDS=${BACKENDS:="graphdef savedmodel"}
TENSOR_SIZE=16384

DATADIR=/data/inferenceserver/${REPO_VERSION}

SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

RET=0

#
# Use "identity" model for all model types.
#
rm -f ./*.log
for BACKEND in $BACKENDS; do
    MODEL_NAME=${BACKEND}_zero_1_float32
    rm -fr models && mkdir -p models
    cp -r $DATADIR/qa_identity_model_repository/${MODEL_NAME} \
       models/${MODEL_NAME}_def && \
    (cd models/${MODEL_NAME}_def && \
            sed -i 's/_zero_1_float32/&_def/' config.pbtxt) && \
    # Enable GPU I/O for TensorFlow model
    cp -r models/${MODEL_NAME}_def models/${MODEL_NAME}_gpu && \
    (cd models/${MODEL_NAME}_gpu && \
            sed -i 's/_zero_1_float32_def/_zero_1_float32_gpu/' \
                config.pbtxt && \
            echo "optimization { execution_accelerators { gpu_execution_accelerator : [ { name : \"gpu_io\"} ] } }" >> config.pbtxt)

    SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1"
    SERVER_LOG="${MODEL_NAME}.serverlog"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e

    $CLIENT -m${MODEL_NAME}_def --shape INPUT0:${TENSOR_SIZE} \
                >> ${BACKEND}.sanity.log 2>&1
    if (( $? != 0 )); then
        RET=1
    fi

    grep "is GPU tensor: true" $SERVER_LOG
    if [ $? -eq 0 ]; then
        echo -e "\n***\n*** Failed. Expected neither input or output is GPU tensor\n***"
        RET=1
    fi

    $CLIENT -m${MODEL_NAME}_gpu  --shape INPUT0:${TENSOR_SIZE} \
             >> ${BACKEND}.gpu.sanity.log 2>&1
    if (( $? != 0 )); then
        RET=1
    fi

    grep "is GPU tensor: true" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected input and output are GPU tensors\n***"
        RET=1
    fi

    # Sample latency results
    $CLIENT -m${MODEL_NAME}_def --shape INPUT0:${TENSOR_SIZE} \
             >> ${BACKEND}.log 2>&1
    if (( $? != 0 )); then
        RET=1
    fi

    $CLIENT -m${MODEL_NAME}_gpu --shape INPUT0:${TENSOR_SIZE} \
            >> ${BACKEND}.gpu.log 2>&1
    if (( $? != 0 )); then
        RET=1
    fi

    set -e

    kill $SERVER_PID
    wait $SERVER_PID
done

for BACKEND in $BACKENDS; do
    echo -e "\n${BACKEND}\n************"
    cat ${BACKEND}.log
    echo -e "\n${BACKEND} with GPU I/O\n************"
    cat ${BACKEND}.gpu.log
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
