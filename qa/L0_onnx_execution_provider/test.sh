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

REPO_VERSION=${NVIDIA_TENSORRT_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi

DATADIR=/data/inferenceserver/${REPO_VERSION}

SERVER=/opt/tensorrtserver/bin/trtserver
SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

rm -f ./*.log
rm -fr models && mkdir -p models && \
    cp -r $DATADIR/qa_model_repository/onnx_float32_float32_float32 \
       models/onnx_float32_float32_float32_def && \
    rm -fr models/onnx_float32_float32_float32_def/2 && \
    rm -fr models/onnx_float32_float32_float32_def/3 && \
    (cd models/onnx_float32_float32_float32_def && \
            sed -i 's/^name: "onnx_float32_float32_float32"/name: "onnx_float32_float32_float32_def"/' \
                config.pbtxt) && \
    # GPU execution providers
    cp -r models/onnx_float32_float32_float32_def models/onnx_float32_float32_float32_trt && \
    (cd models/onnx_float32_float32_float32_trt && \
            sed -i 's/^name: "onnx_float32_float32_float32_def"/name: "onnx_float32_float32_float32_trt"/' \
                config.pbtxt && \
            echo "optimization { execution_providers { gpu_execution_provider : [\"tensorrt\"] } }" >> config.pbtxt) && \
    # CPU execution providers
    cp -r models/onnx_float32_float32_float32_def models/onnx_float32_float32_float32_openvino && \
    (cd models/onnx_float32_float32_float32_openvino && \
            sed -i 's/^name: "onnx_float32_float32_float32_def"/name: "onnx_float32_float32_float32_openvino"/' \
                config.pbtxt && \
            echo "optimization { execution_providers { cpu_execution_provider : [\"openvino\"] } }" >> config.pbtxt) && \
    # CPU execution providers on CPU context
    cp -r models/onnx_float32_float32_float32_openvino models/onnx_float32_float32_float32_cpu_openvino && \
    (cd models/onnx_float32_float32_float32_cpu_openvino && \
            sed -i 's/^name: "onnx_float32_float32_float32_openvino"/name: "onnx_float32_float32_float32_cpu_openvino"/' \
                config.pbtxt && \
            echo "instance_group [ { kind: KIND_CPU }]" >> config.pbtxt) && \
    # Unknown GPU execution provider
    cp -r models/onnx_float32_float32_float32_def models/onnx_float32_float32_float32_unknown_gpu && \
    (cd models/onnx_float32_float32_float32_unknown_gpu && \
            sed -i 's/^name: "onnx_float32_float32_float32_def"/name: "onnx_float32_float32_float32_unknown_gpu"/' \
                config.pbtxt && \
            echo "optimization { execution_providers { gpu_execution_provider : [\"unknown_gpu\"] } }" >> config.pbtxt) && \
    # Unknown CPU execution providers
    cp -r models/onnx_float32_float32_float32_def models/onnx_float32_float32_float32_unknown_cpu && \
    (cd models/onnx_float32_float32_float32_unknown_cpu && \
            sed -i 's/^name: "onnx_float32_float32_float32_def"/name: "onnx_float32_float32_float32_unknown_cpu"/' \
                config.pbtxt && \
            echo "optimization { execution_providers { cpu_execution_provider : [\"unknown_cpu\"] } }" >> config.pbtxt)

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

RET=0

set +e

grep "TensorRT Execution Provider is set for onnx_float32_float32_float32_trt" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected TensorRT Execution Provider is set\n***"
    RET=1
fi
grep "CUDA Execution Provider is set for onnx_float32_float32_float32_trt" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected CUDA Execution Provider is set\n***"
    RET=1
fi

grep "OpenVINO Execution Provider is not supported for onnx_float32_float32_float32_openvino" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected OpenVINO is not supported\n***"
    RET=1
fi
grep "CUDA Execution Provider is set for onnx_float32_float32_float32_openvino" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected CUDA Execution Provider is set\n***"
    RET=1
fi

grep "OpenVINO Execution Provider is not supported for onnx_float32_float32_float32_cpu_openvino" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected OpenVINO Execution Provider is not supported\n***"
    RET=1
fi
grep "CUDA Execution Provider is set for onnx_float32_float32_float32_cpu_openvino" $SERVER_LOG
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** Failed. Expected CUDA Execution Provider is not set\n***"
    RET=1
fi

grep "Ignore unknown Execution Provider 'unknown_gpu' for onnx_float32_float32_float32_unknown_gpu" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected 'unknown_gpu' Execution Provider is ignored\n***"
    RET=1
fi
grep "Ignore unknown Execution Provider 'unknown_cpu' for onnx_float32_float32_float32_unknown_cpu" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected 'unknown_cpu' Execution Provider is ignored\n***"
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
