#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

DATADIR=/data/inferenceserver/${REPO_VERSION}

CLIENT_LOG="./client.log"
ONNXTRT_OPTIMIZATION_TEST=onnxtrt_optimization_test.py

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1 --exit-on-error=false"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

RET=0

for MODEL in \
        onnx_float32_float32_float32; do
    rm -f ./*.log
    rm -fr models && mkdir -p models
    cp -r $DATADIR/qa_model_repository/${MODEL} \
       models/${MODEL}_test && \
    rm -fr models/${MODEL}_test/2 && \
    rm -fr models/${MODEL}_test/3 && \
    (cd models/${MODEL}_test && \
            sed -i 's/_float32_float32_float32/&_test/' config.pbtxt) && \
    # GPU execution accelerators with default setting
    cp -r models/${MODEL}_test models/${MODEL}_trt && \
    (cd models/${MODEL}_trt && \
            sed -i 's/_float32_test/_float32_trt/' \
                config.pbtxt && \
            echo "optimization { execution_accelerators { gpu_execution_accelerator : [ { name : \"tensorrt\"} ] } }" >> config.pbtxt) && \
    # GPU execution accelerators with correct parameters
    cp -r models/${MODEL}_test models/${MODEL}_param && \
    (cd models/${MODEL}_param && \
            sed -i 's/_float32_test/_float32_param/' \
                config.pbtxt && \
            echo "optimization { execution_accelerators { gpu_execution_accelerator : [ { name : \"tensorrt\" \
            parameters { key: \"precision_mode\" value: \"FP16\" } \
            parameters { key: \"max_workspace_size_bytes\" value: \"1073741824\" } }]}}" \
            >> config.pbtxt) && \
    # GPU execution accelerators with unknown parameters
    cp -r models/${MODEL}_test models/${MODEL}_unknown_param && \
    (cd models/${MODEL}_unknown_param && \
            sed -i 's/_float32_test/_float32_unknown_param/' \
                config.pbtxt && \
            echo "optimization { execution_accelerators { gpu_execution_accelerator : [ { name : \"tensorrt\" \
            parameters { key: \"precision_mode\" value: \"FP16\" } \
            parameters { key: \"segment_size\" value: \"1\" } }]}}" \
            >> config.pbtxt) && \
    # GPU execution accelerators with invalid parameters
    cp -r models/${MODEL}_test models/${MODEL}_invalid_param && \
    (cd models/${MODEL}_invalid_param && \
            sed -i 's/_float32_test/_float32_invalid_param/' \
                config.pbtxt && \
            echo "optimization { execution_accelerators { gpu_execution_accelerator : [ { name : \"tensorrt\" \
            parameters { key: \"precision_mode\" value: \"FP16\" } \
            parameters { key: \"max_workspace_size_bytes\" value: \"abc\" } }]}}" \
            >> config.pbtxt) && \
    # Unknown GPU execution accelerator
    cp -r models/${MODEL}_test models/${MODEL}_unknown_gpu && \
    (cd models/${MODEL}_unknown_gpu && \
            sed -i 's/_float32_test/_float32_unknown_gpu/' \
                config.pbtxt && \
            echo "optimization { execution_accelerators { gpu_execution_accelerator : [ { name : \"unknown_gpu\" } ] } }" >> config.pbtxt) && \

    run_server_tolive
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e

    grep "TensorRT Execution Accelerator is set for '${MODEL}_trt'" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected TensorRT Execution Accelerator is set for '${MODEL}_trt'\n***"
        RET=1
    fi

    grep "TensorRT Execution Accelerator is set for '${MODEL}_param'" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected TensorRT Execution Accelerator is set\n***"
        RET=1
    fi

    grep "failed to load '${MODEL}_unknown_param' version 1: Invalid argument: unknown parameter 'segment_size' is provided for TensorRT Execution Accelerator" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected unknown parameter 'segment_size' returns error\n***"
        RET=1
    fi

    grep "failed to load '${MODEL}_invalid_param' version 1: Invalid argument: failed to convert 'abc' to long long integral number" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected invalid parameter 'abc' returns error\n***"
        RET=1
    fi

    grep "failed to load '${MODEL}_unknown_gpu' version 1: Invalid argument: unknown Execution Accelerator 'unknown_gpu' is requested" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected 'unknown_gpu' Execution Accelerator returns error\n***"
        RET=1
    fi

    set -e

    kill $SERVER_PID
    wait $SERVER_PID
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
