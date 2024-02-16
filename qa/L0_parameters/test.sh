#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

CLIENT_LOG="./client.log"
TEST_SCRIPT_PY="parameters_test.py"

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_LOG="./inference_server.log"
source ../common/util.sh

MODELDIR="model_repository"
# Use identity model as dummy step to ensure parameters pass through each step
mkdir -p "${MODELDIR}/identity/1"
mkdir -p "${MODELDIR}/ensemble/1"

# TODO: Add support and testing for C++ client parameters:
# https://jirasw.nvidia.com/browse/DLIS-4673

RET=0
for i in {0..2}; do

  # TEST_HEADER is a parameter used by `parameters_test.py` that controls
  # whether the script will test for inclusion of headers in parameters or not.
  if [ $i == 1 ]; then
    SERVER_ARGS="--model-repository=${MODELDIR} --exit-timeout-secs=120 --grpc-header-forward-pattern my_header.* --http-header-forward-pattern my_header.*"
  elif [ $i == 2 ]; then
    SERVER_ARGS="--model-repository=${MODELDIR} --exit-timeout-secs=120 --grpc-header-forward-pattern MY_HEADER.* --http-header-forward-pattern MY_HEADER.*"
  else
    SERVER_ARGS="--model-repository=${MODELDIR} --exit-timeout-secs=120"
  fi
  run_server
  if [ "$SERVER_PID" == "0" ]; then
      echo -e "\n***\n*** Failed to start $SERVER\n***"
      cat $SERVER_LOG
      exit 1
  fi

  set +e
  TEST_HEADER=$i python3 $TEST_SCRIPT_PY >$CLIENT_LOG 2>&1
  if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
      echo -e "\n***\n*** Test Failed\n***"
      RET=1
  fi

  set -e

  kill $SERVER_PID
  wait $SERVER_PID
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET

