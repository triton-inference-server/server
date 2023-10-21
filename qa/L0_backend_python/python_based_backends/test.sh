#!/bin/bash
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

source ../../common/util.sh

TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
BACKEND_DIR=${TRITON_DIR}/backends
QA_MODELS_PATH="../../python_models"
MODEL_REPOSITORY="`pwd`/models"
SERVER_ARGS="--model-repository=${MODEL_REPOSITORY} --backend-directory=${BACKEND_DIR} --model-control-mode=explicit --log-verbose=1"
SERVER_LOG="./python_based_backends_server.log"
CLIENT_LOG="./python_based_backends_client.log"
TEST_RESULT_FILE="./test_results.txt"
CLIENT_PY="./python_based_backends_test.py"
GEN_PYTORCH_MODEL_PY="../../common/gen_qa_pytorch_model.py"
EXPECTED_NUM_TESTS=4
RET=0

# Setup vllm_backend
pip3 install vllm
mkdir -p ${BACKEND_DIR}/vllm
wget -P ${BACKEND_DIR}/vllm https://raw.githubusercontent.com/triton-inference-server/vllm_backend/main/src/model.py

rm -rf ${MODEL_REPOSITORY}

# Setup vllm backend models
mkdir -p ${MODEL_REPOSITORY}/vllm_opt_1/1/
cp ${QA_MODELS_PATH}/python_based_backends/vllm_opt/model.json ${MODEL_REPOSITORY}/vllm_opt_1/1/
cp ${QA_MODELS_PATH}/python_based_backends/vllm_opt/config.pbtxt ${MODEL_REPOSITORY}/vllm_opt_1

cp -r ${MODEL_REPOSITORY}/vllm_opt_1/ ${MODEL_REPOSITORY}/vllm_opt_2/

# Setup add_sub backend and models
mkdir -p ${BACKEND_DIR}/add_sub
cp ${QA_MODELS_PATH}/python_based_backends/add_sub_backend/model.py ${BACKEND_DIR}/add_sub/model.py

mkdir -p ${MODEL_REPOSITORY}/add/1/
cp ${QA_MODELS_PATH}/python_based_backends/add_sub_backend/add_model.json ${MODEL_REPOSITORY}/add/1/model.json
cp ${QA_MODELS_PATH}/python_based_backends/add_sub_backend/config.pbtxt ${MODEL_REPOSITORY}/add/

mkdir -p ${MODEL_REPOSITORY}/sub/1/
cp ${QA_MODELS_PATH}/python_based_backends/add_sub_backend/sub_model.json ${MODEL_REPOSITORY}/sub/1/model.json
cp ${QA_MODELS_PATH}/python_based_backends/add_sub_backend/config.pbtxt ${MODEL_REPOSITORY}/sub/

# Setup python backend model
mkdir -p ${MODEL_REPOSITORY}/add_sub/1
cp ${QA_MODELS_PATH}/add_sub/model.py ${MODEL_REPOSITORY}/add_sub/1/
cp ${QA_MODELS_PATH}/add_sub/config.pbtxt ${MODEL_REPOSITORY}/add_sub/

# Setup pytorch backend model
cp ${GEN_PYTORCH_MODEL_PY} ./gen_qa_pytorch_model.py
GEN_PYTORCH_MODEL_PY=./gen_qa_pytorch_model.py

set +e
python3 ${GEN_PYTORCH_MODEL_PY} -m ${MODEL_REPOSITORY}

if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Running ${GEN_PYTORCH_MODEL_PY} FAILED. \n***"
    exit 1
fi
set -e

run_server
if [ "$SERVER_PID" == "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    exit 1
fi

set +e
python3 -m unittest -v $CLIENT_PY > $CLIENT_LOG 2>&1

if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Running $CLIENT_PY FAILED. \n***"
    RET=1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Result Verification FAILED.\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVER_PID
rm -rf ${MODEL_REPOSITORY} ${GEN_PYTORCH_MODEL_PY}

if [ $RET -eq 1 ]; then
    cat $CLIENT_LOG
    cat $SERVER_LOG
    echo -e "\n***\n*** Python-based Backends test FAILED. \n***"
else
    echo -e "\n***\n*** Python-based Backends test PASSED. \n***"
fi

collect_artifacts_from_subdir

exit $RET
