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

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1"
CLIENT_PY=./test_infer_shm_leak.py
CLIENT_LOG="./client.log"
EXPECTED_NUM_TESTS="1"
TEST_RESULT_FILE='test_results.txt'
SERVER_LOG="./inference_server.log"
export CUDA_VISIBLE_DEVICES=0,1,2,3

RET=0
rm -fr *.log ./models

source ../common/util.sh

# Uninstall the non CUDA version of PyTorch
pip3 uninstall -y torch
pip3 install torch==2.3.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install tensorflow

# Install CuPy for testing non_blocking compute streams
pip3 install cupy-cuda12x

rm -fr *.log ./models

mkdir -p models/dlpack_test/1/
cp ../python_models/dlpack_test/model.py models/dlpack_test/1/
cp ../python_models/dlpack_test/config.pbtxt models/dlpack_test
cp ../L0_backend_python/test_infer_shm_leak.py .
sed -i 's#sys.path.append("../../common")#sys.path.append("../common")#g' test_infer_shm_leak.py

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

set +e
export MODEL_NAME="dlpack_test"
python3 -m pytest --junitxml=dlpack_multi_gpu.report.xml $CLIENT_PY > $CLIENT_LOG 2>&1

if [ $? -ne 0 ]; then
    echo -e "\n***\n*** python_unittest.py FAILED. \n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 1 ]; then
    cat $CLIENT_LOG
    cat $SERVER_LOG
    echo -e "\n***\n*** dlpack_multi_gpu test FAILED. \n***"
else
    echo -e "\n***\n*** dlpack_multi_gpu test PASSED. \n***"
fi

exit $RET
