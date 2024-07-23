#!/bin/bash
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TEST_LOG="./test.log"
TEST_PY=test.py

DATADIR=/data/inferenceserver/${REPO_VERSION}
rm -f *.log

TRTEXEC=/usr/src/tensorrt/bin/trtexec
TEST_RESULT_FILE='test_results.txt'
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_LOG="./server.log"

source ../common/util.sh

RET=0

# prepare model repository, only contains ONNX and TRT models as the
# corresponding backend are known to be memory.
rm -rf models && mkdir models
# ONNX
cp -r /data/inferenceserver/${REPO_VERSION}/onnx_model_store/* models/.
rm -r models/*cpu

set +e

# VGG19 plan
rm -fr models/vgg19_plan && mkdir -p models/vgg19_plan/1 && \
cp $DATADIR/qa_dynamic_batch_image_model_repository/vgg19_onnx/1/model.onnx models/vgg19_plan/ && \
cp $DATADIR/qa_dynamic_batch_image_model_repository/vgg19_onnx/labels.txt models/vgg19_plan/

$TRTEXEC --onnx=models/vgg19_plan/model.onnx --saveEngine=models/vgg19_plan/1/model.plan \
         --minShapes=input:1x3x224x224 --optShapes=input:32x3x224x224 \
         --maxShapes=input:32x3x224x224

if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to generate vgg19 PLAN\n***"
    exit 1
fi

rm models/vgg19_plan/model.onnx
cp $DATADIR/qa_dynamic_batch_image_model_repository/vgg19_onnx/config.pbtxt models/vgg19_plan/ && \
sed -i "s/^name: .*/name: \"vgg19_plan\"/g" models/vgg19_plan/config.pbtxt && \
sed -i 's/^platform: .*/platform: "tensorrt_plan"/g' models/vgg19_plan/config.pbtxt

# Resnet50 plan
rm -fr models/resnet50_plan && mkdir -p models/resnet50_plan/1 && \
cp $DATADIR/qa_dynamic_batch_image_model_repository/resnet50_onnx/1/model.onnx models/resnet50_plan/ && \
cp $DATADIR/qa_dynamic_batch_image_model_repository/resnet50_onnx/labels.txt models/resnet50_plan/

$TRTEXEC --onnx=models/resnet50_plan/model.onnx --saveEngine=models/resnet50_plan/1/model.plan \
         --minShapes=input:1x3x224x224 --optShapes=input:32x3x224x224 \
         --maxShapes=input:32x3x224x224

if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to generate resnet50 PLAN\n***"
    exit 1
fi

rm models/resnet50_plan/model.onnx
cp $DATADIR/qa_dynamic_batch_image_model_repository/resnet50_onnx/config.pbtxt models/resnet50_plan/ && \
sed -i "s/^name: .*/name: \"resnet50_plan\"/g" models/resnet50_plan/config.pbtxt && \
sed -i 's/^platform: .*/platform: "tensorrt_plan"/g' models/resnet50_plan/config.pbtxt


# Resnet152 plan
rm -fr models/resnet152_plan && mkdir -p models/resnet152_plan/1 && \
cp $DATADIR/qa_dynamic_batch_image_model_repository/resnet152_onnx/1/model.onnx models/resnet152_plan/ && \
cp $DATADIR/qa_dynamic_batch_image_model_repository/resnet152_onnx/labels.txt models/resnet152_plan/

$TRTEXEC --onnx=models/resnet152_plan/model.onnx --saveEngine=models/resnet152_plan/1/model.plan \
         --minShapes=input:1x3x224x224 --optShapes=input:32x3x224x224 \
         --maxShapes=input:32x3x224x224

if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to generate resnet152 PLAN\n***"
    exit 1
fi

rm models/resnet152_plan/model.onnx
cp $DATADIR/qa_dynamic_batch_image_model_repository/resnet152_onnx/config.pbtxt models/resnet152_plan/ && \
sed -i "s/^name: .*/name: \"resnet152_plan\"/g" models/resnet152_plan/config.pbtxt && \
sed -i 's/^platform: .*/platform: "tensorrt_plan"/g' models/resnet152_plan/config.pbtxt

set -e

# Set multiple instances on selected model to test instance-wise collection
# and accumulation.
echo "instance_group [{ count: 2; kind: KIND_GPU }]" >> models/resnet152_plan/config.pbtxt
echo "instance_group [{ count: 2; kind: KIND_GPU }]" >> models/densenet/config.pbtxt

# testing use nvidia-smi for Python to validate the reported usage
pip install nvidia-ml-py3

# Start server to load all models (in parallel), then gradually unload
# the models and expect the memory usage changes matches what are reported
# in statistic.
SERVER_ARGS="--backend-config=triton-backend-memory-tracker=true --model-repository=models --model-control-mode=explicit --load-model=*"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $TEST_PY > $TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi
set -e
kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $SERVER_LOG
    cat $TEST_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
