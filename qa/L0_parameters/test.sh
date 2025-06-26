#!/bin/bash
# Copyright 2023-2025, NVIDIA CORPORATION. All rights reserved.
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

all_tests=("test_params"
           "test_headers"
           "test_header_forward_pattern_case_insensitive"
           "test_grpc_header_forward_pattern_case_sensitive")

RET=0
for i in "${all_tests[@]}"; do
  # TEST_HEADER is a parameter used by `parameters_test.py` that controls
  # whether the script will test for inclusion of headers in parameters or not.
  SERVER_ARGS="--model-repository=${MODELDIR} --exit-timeout-secs=120"
  if [ "$i" == "test_headers" ]; then
    SERVER_ARGS+=" --grpc-header-forward-pattern my_header.*"
    SERVER_ARGS+=" --http-header-forward-pattern my_header.*"
  elif [ "$i" == "test_header_forward_pattern_case_insensitive" ]; then
    SERVER_ARGS+=" --grpc-header-forward-pattern MY_HEADER.*"
    SERVER_ARGS+=" --http-header-forward-pattern MY_HEADER.*"
  # NOTE: headers sent through the python HTTP client may be automatically
  # lowercased by internal libraries like geventhttpclient, so we only test
  # GRPC client for case-sensitivity here:
  # https://github.com/geventhttpclient/geventhttpclient/blob/d1e14356c3b02099c879cf9b3bdb684a0cbd8bf5/src/geventhttpclient/header.py#L62-L63
  elif [ "$i" == "test_grpc_header_forward_pattern_case_sensitive" ]; then
    SERVER_ARGS+=" --grpc-header-forward-pattern (?-i)MY_HEADER.*"
  fi
  run_server
  if [ "$SERVER_PID" == "0" ]; then
      echo -e "\n***\n*** Failed to start $SERVER\n***"
      cat $SERVER_LOG
      exit 1
  fi

  set +e
  TEST_HEADER="$i" python3 $TEST_SCRIPT_PY >$CLIENT_LOG 2>&1
  if [ $? -ne 0 ]; then
      cat $CLIENT_LOG
      echo -e "\n***\n*** Test Failed\n***"
      RET=1
  fi

  set -e

  kill $SERVER_PID
  wait $SERVER_PID
done


# Test Classification Extension
MODELDIR="models"
TEST_RESULT_FILE="test_results.txt"
TEST_SCRIPT_PY="./class_count_test.py"

rm -rf $MODELDIR
mkdir -p "${MODELDIR}/identity_fp32/1"
cp ../python_models/identity_fp32/config.pbtxt "${MODELDIR}/identity_fp32/"
cp ../python_models/identity_fp32/model.py "${MODELDIR}/identity_fp32/1/"

mkdir -p "${MODELDIR}/identity_bytes/1"
cp ../python_models/identity_fp32/config.pbtxt "${MODELDIR}/identity_bytes/"
cp ../python_models/identity_fp32/model.py "${MODELDIR}/identity_bytes/1/"
(cd "${MODELDIR}/identity_bytes" && \
    sed -i 's/identity_fp32/identity_bytes/' config.pbtxt && \
    sed -i 's/TYPE_FP32/TYPE_STRING/' config.pbtxt )

SERVER_ARGS="--model-repository=`pwd`/${MODELDIR} --log-verbose=1"
for client_type in http grpc; do
  export CLIENT_TYPE=$client_type
  SERVER_LOG="./class_count_test_${client_type}_server.log"
  CLIENT_LOG="./class_count_test_${client_type}_client.log"
  rm -f $SERVER_LOG $CLIENT_LOG
  run_server
  if [ "$SERVER_PID" == "0" ]; then
      echo -e "\n***\n*** Failed to start $SERVER\n***"
      cat $SERVER_LOG
      exit 1
  fi

  set +e
  python3 $TEST_SCRIPT_PY -v >>"$CLIENT_LOG" 2>&1
  if [ $? -ne 0 ]; then
      cat $CLIENT_LOG
      echo -e "\n***\n*** Test Failed - class_count_${client_type}_test_client\n***"
      RET=1
  else
      check_test_results $TEST_RESULT_FILE 2
      if [ $? -ne 0 ]; then
          cat $TEST_RESULT_FILE
          echo -e "\n***\n*** Test Result Verification Failed - class_count_${client_type}_test_client\n***"
          RET=1
      fi
  fi
  kill $SERVER_PID
  wait $SERVER_PID

  if [ $? -ne 0 ]; then
      echo -e "\n***\n*** Test Server shut down non-gracefully\n***"
      RET=1
  fi
  set -e
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET

