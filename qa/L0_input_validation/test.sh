#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

source ../common/util.sh

RET=0

DATADIR=/data/inferenceserver/${REPO_VERSION}
SERVER=/opt/tritonserver/bin/tritonserver
CLIENT_LOG="./input_validation_client.log"
TEST_PY=./input_validation_test.py
TEST_RESULT_FILE='./test_results.txt'
SERVER_LOG="./inference_server.log"
TEST_LOG="./input_byte_size_test.log"
TEST_EXEC=./input_byte_size_test

export CUDA_VISIBLE_DEVICES=0

rm -fr *.log

# input_validation_test
SERVER_ARGS="--model-repository=`pwd`/models"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python3 -m pytest --junitxml="input_validation.report.xml" $TEST_PY::InputValTest >> $CLIENT_LOG 2>&1

if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    cat $SERVER_LOG
    echo -e "\n***\n*** input_validation_test.py::InputValTest FAILED. \n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# input_shape_validation_test
pip install torch
pip install pytest-asyncio

mkdir -p models/pt_identity/1
PYTHON_CODE=$(cat <<END
import torch
torch.jit.save(
    torch.jit.script(torch.nn.Identity()),
    "`pwd`/models/pt_identity/1/model.pt",
)
END
)
res="$(python3 -c "$PYTHON_CODE")"

if [ $? -ne 0 ]; then
    echo -e "\n***\n*** model "pt_identity" initialization FAILED. \n***"
    echo $res
    exit 1
fi

# Create the config.pbtxt file with the specified configuration
cat > models/pt_identity/config.pbtxt << EOL
name: "pt_identity"
backend: "pytorch"
max_batch_size: 8
input [
  {
    name: "INPUT0"
    data_type: TYPE_FP32
    dims: [8]
  }
]
output [
  {
    name: "OUTPUT0"
    data_type: TYPE_FP32
    dims: [8]
  }
]
# ensure we batch requests together
dynamic_batching {
    max_queue_delay_microseconds: 1000000
}
EOL

cp -r $DATADIR/qa_model_repository/graphdef_object_int32_int32 models/.
cp -r $DATADIR/qa_shapetensor_model_repository/plan_nobatch_zero_1_float32_int32 models/.
cp -r $DATADIR/qa_shapetensor_model_repository/plan_zero_1_float32_int32 models/.

SERVER_ARGS="--model-repository=`pwd`/models"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python3 -m pytest --junitxml="input_shape_validation.report.xml" $TEST_PY::InputShapeTest >> $CLIENT_LOG 2>&1

if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    cat $SERVER_LOG
    echo -e "\n***\n*** input_validation_test.py::InputShapeTest FAILED. \n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# input_byte_size_test
cp -r /data/inferenceserver/${REPO_VERSION}/qa_identity_model_repository/{savedmodel_zero_1_float32,savedmodel_zero_1_object} ./models

set +e
LD_LIBRARY_PATH=/opt/tritonserver/lib:$LD_LIBRARY_PATH $TEST_EXEC >> $TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $TEST_LOG
    echo -e "\n***\n*** input_byte_size_test FAILED\n***"
    RET=1
fi
set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Input Validation Test Passed\n***"
else
    echo -e "\n***\n*** Input Validation Test FAILED\n***"
fi

exit $RET
