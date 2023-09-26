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

export CUDA_VISIBLE_DEVICES=0

RET=0
CLEANUP_TEST=cleanup_test.py

rm -f *.log

CLIENT_LOG=`pwd`/client.log
SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

function check_state_release() {
  local log_file=$1

  num_state_release=`cat $log_file | grep  "StateRelease" | wc -l`
  num_state_new=`cat $log_file | grep  "StateNew" | wc -l`

  if [ $num_state_release -ne $num_state_new ]; then
    cat $log_file
    echo -e "\n***\n*** Test Failed: Mismatch detected, $num_state_new state(s) created, $num_state_release state(s) released. \n***" >> $log_file
    return 1
  fi

  return 0
}


for i in test_decoupled_infer \
            test_decoupled_cancellation \
            test_decoupled_timeout; do
  SERVER_LOG="./inference_server.$i.log"
  SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=2"
  run_server
  if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
  fi

  echo "Test: $i" >>$CLIENT_LOG

  set +e
  python $CLEANUP_TEST CleanUpTest.$i >>$CLIENT_LOG 2>&1
  if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test $i Failed\n***" >>$CLIENT_LOG
    echo -e "\n***\n*** Test $i Failed\n***"
    RET=1
  fi

  kill $SERVER_PID
  wait $SERVER_PID

  check_state_release $SERVER_LOG
  if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** State Verification Failed\n***"
      RET=1
  fi
  set -e
done


for i in test_decoupled_error_status; do
  SERVER_LOG="./inference_server.$i.log"
  SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=2 --grpc-restricted-protocol=inference,health:infer-key=infer-value"
  run_server
  if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
  fi

  echo "Test: $i" >>$CLIENT_LOG

  set +e
  python $CLEANUP_TEST CleanUpTest.$i >>$CLIENT_LOG 2>&1
  if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test $i Failed\n***" >>$CLIENT_LOG
    echo -e "\n***\n*** Test $i Failed\n***"
    RET=1
  fi

  kill $SERVER_PID
  wait $SERVER_PID

  check_state_release $SERVER_LOG
  if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** State Verification Failed\n***"
      RET=1
  fi

  set -e
done

for i in test_decoupled_infer_shutdownserver \
         test_decoupled_infer_with_params_shutdownserver; do
  SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=2"
  SERVER_LOG="./inference_server.$i.log"
  run_server
  if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
  fi

  echo "Test: $i" >>$CLIENT_LOG

  set +e
  SERVER_PID=$SERVER_PID python $CLEANUP_TEST CleanUpTest.$i >>$CLIENT_LOG 2>&1
  if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test $i Failed\n***" >>$CLIENT_LOG
    echo -e "\n***\n*** Test $i Failed\n***"
    RET=1
  fi

  wait $SERVER_PID

  check_state_release $SERVER_LOG
  if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** State Verification Failed\n***"
      RET=1
  fi

  set -e
done


if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test Failed\n***"
fi

exit $RET
