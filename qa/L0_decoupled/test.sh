#!/bin/bash
# Copyright 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
TEST_RESULT_FILE='test_results.txt'
DECOUPLED_TEST=decoupled_test.py

rm -f *.log

CLIENT_LOG=`pwd`/client.log
DATADIR=/data/inferenceserver/${REPO_VERSION}/qa_model_repository
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=`pwd`/models"
SERVER_LOG="./inference_server.log"
source ../common/util.sh


TRIALS="python custom"

for trial in $TRIALS; do
  if [ $trial == "python" ]; then
    MODELDIR=`pwd`/python_models
  else
    MODELDIR=`pwd`/models
  fi

  SERVER_ARGS="--model-repository=$MODELDIR"
  cp -r $DATADIR/libtorch_nobatch_int32_int32_int32 $MODELDIR/.
  (cd $MODELDIR/libtorch_nobatch_int32_int32_int32 && \
   sed -i "s/dims:.*\[.*\]/dims: \[ 1 \]/g" config.pbtxt)

  run_server
  if [ "$SERVER_PID" == "0" ]; then
      echo -e "\n***\n*** Failed to start $SERVER\n***"
      cat $SERVER_LOG
      exit 1
  fi

  for i in \
              test_one_to_none \
              test_one_to_one \
              test_one_to_many \
              test_no_streaming \
              test_response_order \
	      test_wrong_shape; do

      echo "Test: $i" >>$CLIENT_LOG
      set +e
      python $DECOUPLED_TEST DecoupledTest.$i >>$CLIENT_LOG 2>&1
      if [ $? -ne 0 ]; then
              echo -e "\n***\n*** Test $i Failed\n***" >>$CLIENT_LOG
              echo -e "\n***\n*** Test $i Failed\n***"
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
  done

  # Will delay the writing of each response by the specified many milliseconds.
  # This will ensure that there are multiple responses available to be written.
  export TRITONSERVER_DELAY_GRPC_RESPONSE=2000

  echo "Test: test_one_to_multi_many" >>$CLIENT_LOG
  set +e
  python $DECOUPLED_TEST DecoupledTest.test_one_to_multi_many >>$CLIENT_LOG 2>&1
  if [ $? -ne 0 ]; then
      echo -e "\n***\n*** Test test_one_to_multi_many Failed\n***" >>$CLIENT_LOG
          echo -e "\n***\n*** Test test_one_to_multi_many Failed\n***"
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

  unset TRITONSERVER_DELAY_GRPC_RESPONSE

  kill $SERVER_PID
  wait $SERVER_PID

  SERVER_ARGS="--model-repository=$MODELDIR --grpc-max-response-pool-size=1"
  SERVER_LOG="grpc_max_response_pool_size_1_${trial}_server.log"
  CLIENT_LOG="grpc_max_response_pool_size_1_${trial}_client.log"
  run_server
  if [ "$SERVER_PID" == "0" ]; then
      echo -e "\n***\n*** Failed to start $SERVER\n***"
      cat $SERVER_LOG
      exit 1
  fi

  for test in \
              test_one_to_none \
              test_one_to_one \
              test_one_to_many \
              test_no_streaming \
              test_response_order \
        test_wrong_shape; do

      echo "Test: $test" >>$CLIENT_LOG
      set +e
      python $DECOUPLED_TEST DecoupledTest.$test >>$CLIENT_LOG 2>&1
      if [ $? -ne 0 ]; then
              echo -e "\n***\n*** Test grpc-max-response-pool-size=1 ${trial} - $test Failed\n***" >>$CLIENT_LOG
              echo -e "\n***\n*** Test grpc-max-response-pool-size=1 ${trial} - $test Failed\n***"
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
  done

  kill $SERVER_PID
  wait $SERVER_PID
done

# Test the server frontend can merge the responses of non-decoupled model that
# sends inference response and COMPLETE flag separately. In other words, from
# the client's perspective there will still be one response.
NON_DECOUPLED_DIR=`pwd`/non_decoupled_models
rm -rf ${NON_DECOUPLED_DIR} && mkdir -p ${NON_DECOUPLED_DIR}
cp -r `pwd`/models/repeat_int32 ${NON_DECOUPLED_DIR}/. && \
    (cd ${NON_DECOUPLED_DIR}/repeat_int32 && \
        sed -i "s/decoupled: True/decoupled: False/" config.pbtxt)

SERVER_ARGS="--model-repository=${NON_DECOUPLED_DIR}"
SERVER_LOG="./non_decoupled_inference_server.log"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

CLIENT_LOG=`pwd`/non_decoupled_client.log
echo "Test: NonDecoupledTest" >>$CLIENT_LOG
set +e
python $DECOUPLED_TEST NonDecoupledTest >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test NonDecoupledTest Failed\n***" >>$CLIENT_LOG
        echo -e "\n***\n*** Test NonDecoupledTest Failed\n***"
        RET=1
else
    check_test_results $TEST_RESULT_FILE 2
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test Failed\n***"
fi

exit $RET