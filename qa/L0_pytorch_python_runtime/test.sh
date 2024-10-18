#!/bin/bash
# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TRITON_REPO_ORGANIZATION=${TRITON_REPO_ORGANIZATION:="https://github.com/triton-inference-server"}
DATA_DIR=/data/inferenceserver/${REPO_VERSION}
IMAGE_DIR="/opt/tritonserver/qa/images"
SERVER=/opt/tritonserver/bin/tritonserver
IMAGE_CLIENT="/opt/tritonserver/qa/clients/image_client.py"
BACKENDS="/opt/tritonserver/backends"
source ../common/util.sh

if [ ! -f "$BACKENDS/pytorch/pb_exec_env_model.py.tar.gz" ]; then
    PYTORCH_BACKEND_REPO_TAG=${PYTORCH_BACKEND_REPO_TAG:="main"}
    rm -rf pytorch_backend
    git clone --single-branch --depth=1 -b $PYTORCH_BACKEND_REPO_TAG ${TRITON_REPO_ORGANIZATION}/pytorch_backend
    (cd pytorch_backend/tools && \
        ./gen_pb_exec_env.sh && \
        mv pb_exec_env_model.py.tar.gz $BACKENDS/pytorch)
fi

rm -f *.log
RET=0

#
# Unit tests
#
rm -rf py_runtime_exec_env py_runtime_exec_env.tar.gz py_runtime.py
cp $BACKENDS/pytorch/model.py py_runtime.py
cp $BACKENDS/pytorch/pb_exec_env_model.py.tar.gz py_runtime_exec_env.tar.gz
mkdir py_runtime_exec_env && tar -xzf py_runtime_exec_env.tar.gz -C py_runtime_exec_env

set +e

UNIT_TEST_ENV="source py_runtime_exec_env/bin/activate && exec env LD_LIBRARY_PATH=`pwd`/py_runtime_exec_env/lib:$LD_LIBRARY_PATH"
UNIT_TEST_LOG="./unit_test.log"
bash -c "$UNIT_TEST_ENV python3 unit_test.py" > $UNIT_TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed PyTorch Python backend based runtime unit test\n***"
    cat $UNIT_TEST_LOG
    RET=1
fi

set -e

#
# End-to-end inference tests
#
rm -rf models && mkdir models
cp -r $DATA_DIR/pytorch_model_store/* models
cp -r $DATA_DIR/libtorch_model_store/resnet50_libtorch models && \
    sed -i "/platform/d" models/resnet50_libtorch/config.pbtxt && \
    echo "backend: \"pytorch\"" >> models/resnet50_libtorch/config.pbtxt && \
    echo "runtime: \"model.py\"" >> models/resnet50_libtorch/config.pbtxt && \
    echo "instance_group: [{ kind: KIND_MODEL }]" >> models/resnet50_libtorch/config.pbtxt
mv models/neuralnet/1/test_data.json neuralnet_test_data.json

SERVER_ARGS="--model-repository=models --log-verbose=1"
SERVER_LOG="./infer.server.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
else
    set +e

    # Check correct model instance initialization
    EXPECTED_LOG_MSGS=(
        'Loading '"'"'resnet50_libtorch'"'"' as TorchScript'
        'Torch parallelism settings for '"'"'addsub'"'"': NUM_THREADS = 1; NUM_INTEROP_THREADS = 1;'
        'Torch parallelism settings for '"'"'neuralnet'"'"': NUM_THREADS = 4; NUM_INTEROP_THREADS = 2;'
        'Torch parallelism settings for '"'"'resnet50_libtorch'"'"': NUM_THREADS = 1; NUM_INTEROP_THREADS = 1;'
        ''"'"'torch.compile'"'"' optional parameter(s) for '"'"'addsub'"'"': {'"'"'disable'"'"': True}'
        ''"'"'torch.compile'"'"' optional parameter(s) for '"'"'neuralnet'"'"': {}'
        ''"'"'torch.compile'"'"' optional parameter(s) for '"'"'resnet50_libtorch'"'"': {}'
    )
    for EXPECTED_LOG_MSG in "${EXPECTED_LOG_MSGS[@]}"; do
        grep "$EXPECTED_LOG_MSG" $SERVER_LOG
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Cannot find \"$EXPECTED_LOG_MSG\" on server log. \n***"
            cat $SERVER_LOG
            RET=1
        fi
    done

    # Infer TorchScript model
    CLIENT_LOG="./infer.torchscript.log"
    python $IMAGE_CLIENT -m "resnet50_libtorch" -s INCEPTION -c 1 -b 2 "$IMAGE_DIR/vulture.jpeg" > $CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed to inference TorchScript model\n***"
        cat $CLIENT_LOG
        RET=1
    fi

    # Infer PyTorch models
    CLIENT_LOG="./infer.pytorch.log"
    python infer.py > $CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed to inference PyTorch models\n***"
        cat $CLIENT_LOG
        RET=1
    fi

    set -e

    kill $SERVER_PID
    wait $SERVER_PID
fi

#
# Print result and exit
#
if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi
exit $RET
