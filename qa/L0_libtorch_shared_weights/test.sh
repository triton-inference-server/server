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
INSTANCE_CNT=16
REUSE_MSG="Reusing TorchScript model for instance"
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=`pwd`/models --exit-on-error=false \
             --exit-timeout-secs=10"
TEST_RESULT_FILE='test_results.txt'
WEIGHTS_TEST=libtorch_shared_weights_test.py
source ../common/util.sh

RET=0
rm -fr *.log

LOG_IDX=0

# SharedWeightsTest.test_pytorch_identity_model
# Without shared weights, GPU

# Prepare model repository
rm -fr models
mkdir models
for i in models; do
    cp -r $DATADIR/qa_identity_model_repository/libtorch_nobatch_zero_1_float32 models/.
done

for MC in `ls models/libtorch*/config.pbtxt`; do
    echo "instance_group [ { count: ${INSTANCE_CNT} kind: KIND_GPU}]" >> $MC
done

# Start server
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# Run test
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

if [ `grep -c "$REUSE_MSG" $SERVER_LOG` != "0" ]; then
    echo -e "\n***\n*** Failed. Expected 0 "$REUSE_MSG"\n***"
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

# SharedWeightsTest.test_pytorch_identity_model
# With shared weights

for KIND in KIND_CPU KIND_GPU; do

    # Prepare model repository
    rm -fr models
    mkdir models
    for i in models; do
        cp -r $DATADIR/qa_identity_model_repository/libtorch_nobatch_zero_1_float32 models/.
    done

    LOG_IDX=$((LOG_IDX+1))
    for MC in `ls models/libtorch*/config.pbtxt`; do
        echo "instance_group [ { count: ${INSTANCE_CNT} kind: ${KIND}}]" >> $MC
    done

    for MC in `ls models/libtorch*/config.pbtxt`; do
        echo """
        parameters: {
            key: \"ENABLE_WEIGHT_SHARING\"
            value: {
                string_value: \"true\"
            }
        }""" >> $MC
    done

    # Start server
    SERVER_LOG="./inference_server_$LOG_IDX.log"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    # Run test
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

    if [ `grep -c "$REUSE_MSG" $SERVER_LOG` != "15" ]; then
        echo -e "\n***\n*** Failed. Expected 15 "$REUSE_MSG"\n***"
        RET=1
    fi

    set -e

    kill $SERVER_PID
    wait $SERVER_PID
done

# Test Cleanup
rm -f $CLIENT_LOG

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
fi

exit $RET
