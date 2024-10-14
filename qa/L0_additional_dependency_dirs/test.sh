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

if [[ -v WSL_DISTRO_NAME ]] || [[ -v MSYSTEM ]]; then
    DATADIR=${DATADIR:="/mnt/nvdl/datasets/inferenceserver/${REPO_VERSION}"}
    TRITON_DIR=${TRITON_DIR:=c:/tritonserver}
    SERVER=${SERVER:=c:/tritonserver/bin/tritonserver.exe}
    BACKEND_DIR=${BACKEND_DIR:=c:/tritonserver/backends}
    MODELDIR=${MODELDIR:=c:/}
    TRITONSERVER_IPADDR=${TRITONSERVER_IPADDR:="localhost"}
else
    echo -e "Test is not currently supported for Linux"
    exit 1
fi

# FIXME: TPRD-244 - Do not use hard-coded dependency paths when TRT models
# are being regularly generated on Windows.
DEPENDENCY_PATH="C:/ci_test_deps/24.07"
STALE_DEPENDENCY_PATH="C:/ci_test_deps/24.05"
LOCAL_CI_TEST_DEPS_DIR=${MODELDIR}/ci_test_deps
CUSTOM_DEPENDENCY_DIR=${MODELDIR}/custom_dependency_location
STALE_DEPENDENCY_DIR=${MODELDIR}/stale_dependency_location

source ../common/util.sh
rm -rf ${CUSTOM_DEPENDENCY_DIR} ${LOCAL_CI_TEST_DEPS_DIR} ${STALE_DEPENDENCY_DIR} ${MODELDIR}/models
rm -f ./*.log ./*.out
RET=0;

mkdir ${LOCAL_CI_TEST_DEPS_DIR} ${CUSTOM_DEPENDENCY_DIR} ${STALE_DEPENDENCY_DIR}

# Make a copy of the ci_test_deps directory and move out the TRT dependencies
# A pre-test step will add ${LOCAL_CI_TEST_DEPS_DIR} to the PATH
cp -r ${DEPENDENCY_PATH}/* ${LOCAL_CI_TEST_DEPS_DIR} && mv ${LOCAL_CI_TEST_DEPS_DIR}/nvinfer* ${CUSTOM_DEPENDENCY_DIR}/

cp -r ${STALE_DEPENDENCY_PATH}/nvinfer* ${STALE_DEPENDENCY_DIR}/

mkdir ${MODELDIR}/models && \
    cp -r ${DATADIR}/qa_model_repository/plan_int32_int32_int32 ${MODELDIR}/models/plan_int32_int32_int32 

function simple_inference_check()
{
    INFER_SUCCESS=a
    set +e
    code=`curl -s -w %{http_code} -o ./curl.out -d'{"inputs":[{"name":"INPUT0","datatype":"INT32","shape":[1,16],"data":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]},{"name":"INPUT1","datatype":"INT32","shape":[1,16],"data":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}]}' ${TRITONSERVER_IPADDR}:8000/v2/models/plan_int32_int32_int32/infer`
    set -e
    if [ "$code" != "200" ]; then
        cat ./curl.out
        INFER_SUCCESS=0
    fi
}

# Test Case 1: Run server when TRT implicit dependencies are not on path (expect FAIL)
SERVER_LOG="./not_on_path_server.log"
SERVER_ARGS="--model-repository=${MODELDIR}/models --backend-directory=${BACKEND_DIR} --log-verbose=2"
run_server
if [ "$SERVER_PID" != "0" ]; then
    echo -e "\n***\n*** FAILED on line ${LINENO}: unexpected success starting $SERVER\n***"
    kill_server
    RET=1
fi

# Test Case 2: Launch server with additional options to point to custom dependency location (expect SUCCESS)
SERVER_LOG="./custom_dependency_dir_server.log"
SERVER_ARGS="--model-repository=${MODELDIR}/models --backend-directory=${BACKEND_DIR} --log-verbose=2 --backend-config=tensorrt,additional-dependency-dirs=${CUSTOM_DEPENDENCY_DIR};"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

simple_inference_check
if [ "${INFER_SUCCESS}" == "0" ]; then
    echo -e "\n***\n*** FAILED on line ${LINENO}: simple inference check failed\n***"
    RET=1
fi

kill_server

# Test Case 3: Launch server with additional options to point to custom dependency location and load model dynamically (expect SUCCESS)
SERVER_LOG="./dynamic_loading_server.log"
SERVER_ARGS="--model-repository=${MODELDIR}/models --backend-directory=${BACKEND_DIR} --log-verbose=2 --backend-config=tensorrt,additional-dependency-dirs=${CUSTOM_DEPENDENCY_DIR}; --model-control-mode=explicit"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

code=`curl -s -w %{http_code} -o ./curl.out -X POST ${TRITONSERVER_IPADDR}:8000/v2/repository/models/plan_int32_int32_int32/load`
if [ "$code" != "200" ]; then
    echo -e "\n***\n*** FAILED on line ${LINENO}: Unable to load model: plan_int32_int32_int32\n***"
    cat ./curl.out
    RET=1
fi

simple_inference_check
if [ "${INFER_SUCCESS}" == "0" ]; then
    echo -e "\n***\n*** FAILED on line ${LINENO}: simple inference check failed\n***"
    RET=1
fi

kill_server

# Test Case 4: Run server when when pointing to stale TRT dependencies (expect FAIL)
SERVER_LOG="./stale_dependencies_server.log"
SERVER_ARGS="--model-repository=${MODELDIR}/models --backend-directory=${BACKEND_DIR} --log-verbose=2 --backend-config=tensorrt,additional-dependency-dirs=${STALE_DEPENDENCY_DIR};"
run_server
if [ "$SERVER_PID" != "0" ]; then
    echo -e "\n***\n*** FAILED on line ${LINENO}: unexpected success starting $SERVER\n***"
    kill_server
    RET=1
fi

# Test Case 5: [Test ordering] Run server when when pointing to stale and correct TRT dependencies (stale first - expect FAIL).
SERVER_LOG="./stale_first_server.log"
SERVER_ARGS="--model-repository=${MODELDIR}/models --backend-directory=${BACKEND_DIR} --log-verbose=2 --backend-config=tensorrt,additional-dependency-dirs=${STALE_DEPENDENCY_DIR};${CUSTOM_DEPENDENCY_DIR};"
run_server
if [ "$SERVER_PID" != "0" ]; then
    echo -e "\n***\n*** FAILED on line ${LINENO}: unexpected success starting $SERVER\n***"
    kill_server
    RET=1
fi

# Test Case 6:  [Test ordering] Run server when when pointing to stale and correct TRT dependencies (correct first - expect SUCCESS).
SERVER_LOG="./correct_first_server.log"
SERVER_ARGS="--model-repository=${MODELDIR}/models --backend-directory=${BACKEND_DIR} --log-verbose=2 --backend-config=tensorrt,additional-dependency-dirs=${CUSTOM_DEPENDENCY_DIR};${CUSTOM_DEPENDENCY_DIR}; --model-control-mode=explicit"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

code=`curl -s -w %{http_code} -X POST ${TRITONSERVER_IPADDR}:8000/v2/repository/models/plan_int32_int32_int32/load`
if [ "$code" != "200" ]; then
    echo -e "\n***\n*** FAILED on line ${LINENO}: Unable to load model: ${model}\n***"
    RET=1
fi

simple_inference_check
if [ "${INFER_SUCCESS}" == "0" ]; then
    echo -e "\n***\n*** FAILED on line ${LINENO}: simple inference check failed\n***"
    RET=1
fi

kill_server

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi
