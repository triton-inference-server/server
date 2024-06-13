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

source ../common/util.sh

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

RET=0
TRT_TEST="trt_bf16_dtype_test.py"
TEST_RESULT_FILE="./test_results.txt"
SERVER=/opt/tritonserver/bin/tritonserver

rm -rf ./fixed_models/ ./dynamic_models/ *.log* && mkdir ./fixed_models/ ./dynamic_models/
cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/plan_*bf16_bf16_bf16 ./fixed_models/
cp -r /data/inferenceserver/${REPO_VERSION}/qa_variable_model_repository/plan_*bf16_bf16_bf16 ./dynamic_models/

for TEST in "fixed" "dynamic"; do
  MODELDIR="./${TEST}_models"
  CLIENT_LOG="./${TEST}_client.log"
  SERVER_LOG="./${TEST}_inference_server.log"
  SERVER_ARGS="--model-repository=${MODELDIR} --log-verbose=1"

  run_server
  if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
  fi

  set +e
  python3 $TRT_TEST TrtBF16DataTypeTest.test_${TEST} >>$CLIENT_LOG 2>&1
  if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Running $TRT_TEST TrtBF16DataTypeTest.test_${TEST} Failed\n***"
    cat $CLIENT_LOG
    RET=1
  else
    check_test_results $TEST_RESULT_FILE 1
    if [ $? -ne 0 ]; then
      cat $CLIENT_LOG
      echo -e "\n***\n*** Test Result Verification Failed\n***"
      RET=1
    fi
  fi
  set -e

  kill $SERVER_PID
  wait $SERVER_PID
done

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test Failed\n***"
fi

exit $RET
