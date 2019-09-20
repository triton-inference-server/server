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
SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1 --exit-on-error=false"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

rm -f ./*.log
rm -fr models && mkdir -p models && \
    cp -r $DATADIR/onnx_model_store/resnet50 \
       models/resnet50_def && \
    (cd models/resnet50_def && \
            sed -i 's/^name: "resnet50"/name: "resnet50_def"/' \
                config.pbtxt) && \
    # GPU execution accelerators
    # [DLIS-729] For TensorRT, only deploy it on gpu 0 as deploying on
    # other devices will cause segfault
    # https://github.com/microsoft/onnxruntime/issues/1881
    cp -r models/resnet50_def models/resnet50_trt && \
    (cd models/resnet50_trt && \
            sed -i 's/^name: "resnet50_def"/name: "resnet50_trt"/' \
                config.pbtxt && \
            echo "optimization { execution_accelerators { gpu_execution_accelerator : [\"tensorrt\"] } }" >> config.pbtxt && \
            echo "instance_group [ { gpus: [0] } ]" >> config.pbtxt) && \
    # CPU execution accelerators
    cp -r models/resnet50_def models/resnet50_openvino && \
    (cd models/resnet50_openvino && \
            sed -i 's/^name: "resnet50_def"/name: "resnet50_openvino"/' \
                config.pbtxt && \
            echo "optimization { execution_accelerators { cpu_execution_accelerator : [\"openvino\"] } }" >> config.pbtxt) && \
    # CPU execution accelerators on CPU context
    cp -r models/resnet50_openvino models/resnet50_cpu_openvino && \
    (cd models/resnet50_cpu_openvino && \
            sed -i 's/^name: "resnet50_openvino"/name: "resnet50_cpu_openvino"/' \
                config.pbtxt && \
            echo "instance_group [ { kind: KIND_CPU }]" >> config.pbtxt) && \
    # Unknown GPU execution accelerator
    cp -r models/resnet50_def models/resnet50_unknown_gpu && \
    (cd models/resnet50_unknown_gpu && \
            sed -i 's/^name: "resnet50_def"/name: "resnet50_unknown_gpu"/' \
                config.pbtxt && \
            echo "optimization { execution_accelerators { gpu_execution_accelerator : [\"unknown_gpu\"] } }" >> config.pbtxt) && \
    # Unknown CPU execution accelerators
    cp -r models/resnet50_def models/resnet50_unknown_cpu && \
    (cd models/resnet50_unknown_cpu && \
            sed -i 's/^name: "resnet50_def"/name: "resnet50_unknown_cpu"/' \
                config.pbtxt && \
            echo "optimization { execution_accelerators { cpu_execution_accelerator : [\"unknown_cpu\"] } }" >> config.pbtxt)

run_server_tolive
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

RET=0

set +e

grep "TensorRT Execution Accelerator is set for resnet50_trt" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected TensorRT Execution Accelerator is set\n***"
    RET=1
fi
grep "CUDA Execution Accelerator is set for resnet50_trt" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected CUDA Execution Accelerator is set\n***"
    RET=1
fi


grep "OpenVINO Execution Accelerator is set for resnet50_openvino" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected OpenVINO Execution Accelerator is set\n***"
    RET=1
fi
grep "CUDA Execution Accelerator is set for resnet50_openvino" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected CUDA Execution Accelerator is set\n***"
    RET=1
fi

grep "OpenVINO Execution Accelerator is set for resnet50_cpu_openvino" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected OpenVINO Execution Accelerator is set\n***"
    RET=1
fi
grep "CUDA Execution Accelerator is set for resnet50_cpu_openvino" $SERVER_LOG
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** Failed. Expected CUDA Execution Accelerator is not set\n***"
    RET=1
fi

grep "\[OpenVINO-EP\] Rejecting" $SERVER_LOG
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** Failed. Expected OpenVINO is set on models that support it\n***"
    RET=1
fi
grep "failed to load 'resnet50_unknown_gpu' version 1: Invalid argument: unknown Execution Accelerator 'unknown_gpu' is requested" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected 'unknown_gpu' Execution Accelerator returns error\n***"
    RET=1
fi
grep "failed to load 'resnet50_unknown_cpu' version 1: Invalid argument: unknown Execution Accelerator 'unknown_cpu' is requested" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected 'unknown_cpu' Execution Accelerator returns error\n***"
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
