# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

## Basic framework from other test.sh

## Copy over libtorch non-sequence model
## Make model have multiple instances on one device
## Make non-sequence model have share_weights on
## Use test script to push high concurrency on both and check accuracy
## Check for expected logs
## Make model have share_weights off
## Use test script to push high concurrency on both and check accuracy (repeat)
## Check for expected logs

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

CLIENT_LOG="./client.log"
DATADIR=/data/inferenceserver/${REPO_VERSION}
WEIGHTS_TEST=libtorch_shared_weights_test.py
SERVER=/opt/tritonserver/bin/tritonserver
TEST_RESULT_FILE='test_results.txt'
source ../common/util.sh

RET=0
rm -fr *.log

LOG_IDX=0

# Prepare model repository
rm -fr models
mkdir models
for i in graphdef savedmodel ; do
    cp -r $DATADIR//qa_identity_model_repository/libtorch_nobatch_zero_1_float32 models/.
done

# SharedWeightsTest.test_pytorch_identity_model
# Without shared weights
# TODO: Add grep to make sure weights are not shared
SERVER_ARGS="--model-repository=`pwd`/models --exit-on-error=false \
             --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $WEIGHTS_TEST SharedWeightsTest.test_pytorch_identity_model >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
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

LOG_IDX=$((LOG_IDX+1))

# SharedWeightsTest.test_pytorch_identity_model
# With shared weights
# TODO: Add sed to share weights, grep to make sure weights are shared
SERVER_ARGS="--model-repository=`pwd`/models --exit-on-error=false \
             --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $WEIGHTS_TEST SharedWeightsTest.test_pytorch_identity_model >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
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

# Test Cleanup
rm -f $CLIENT_LOG

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
fi

exit $RET