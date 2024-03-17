#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
TEST_RESULT_FILE='test_results.txt'
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

unset AWS_ACCESS_KEY_ID
unset AWS_SECRET_ACCESS_KEY
unset AWS_DEFAULT_REGION

pip3 install --no-deps awscli-plugin-endpoint

# cli_legacy_plugin_path = /usr/local/lib/python3.8/site-packages

mkdir -p ~/.aws
# Swiftstack S3 credentials are necessary for this test. Passed via ENV variables
echo "[plugins]
endpoint = awscli_plugin_endpoint

[default]
aws_access_key_id = $SWIFTSTACK_ACCESS_KEY_ID
aws_secret_access_key = $SWIFTSTACK_SECRET_ACCESS_KEY
region = $SWIFTSTACK_DEFAULT_REGION

s3 =
    endpoint_url = https://pbss.s8k.io
    signature_version = s3v4
    payload_signing_enabled = true
" > ~/.aws/config

export AWS_ACCESS_KEY_ID=$SWIFTSTACK_ACCESS_KEY_ID &&
export AWS_SECRET_ACCESS_KEY=$SWIFTSTACK_SECRET_ACCESS_KEY &&
export AWS_DEFAULT_REGION=$SWIFTSTACK_DEFAULT_REGION

# S3 bucket path (Point to bucket when testing cloud storage)
BUCKET_URL="s3://triton-bucket-${CI_JOB_ID}"

# S3 repo path to pass to Triton server
S3_REPO_URL="s3://https://pbss.s8k.io:443/triton-bucket-${CI_JOB_ID}"

# Cleanup S3 test bucket if exists (due to test failure)
aws s3 rm $BUCKET_URL --recursive --include "*" && \
    aws s3 rb $BUCKET_URL || true

# Make S3 test bucket
aws s3 mb $BUCKET_URL

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_TIMEOUT=420

CLIENT_LOG_BASE="./client"
SERVER_LOG_BASE="./inference_server"
INFER_TEST=infer_test.py
EXPECTED_NUM_TESTS="2"
source ../common/util.sh

rm -f $SERVER_LOG_BASE* $CLIENT_LOG_BASE*
RET=0

SERVER_LOG=$SERVER_LOG_BASE.log
CLIENT_LOG=$CLIENT_LOG_BASE.log

# Copy models in model directory
rm -rf models && mkdir -p models

aws s3 rm $BUCKET_URL/ --recursive --include "*"

# Now start model tests

for FW in graphdef savedmodel onnx libtorch plan; do
    cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/${FW}_float32_float32_float32/ models/
done

for FW in graphdef savedmodel onnx libtorch plan; do
    for MC in `ls models/${FW}*/config.pbtxt`; do
        echo "instance_group [ { kind: KIND_GPU }]" >> $MC
    done
done

# copy contents of /models into S3 bucket.
aws s3 cp models/ $BUCKET_URL/ --recursive --include "*"

# Test without polling
SERVER_ARGS="--model-repository=$S3_REPO_URL --exit-timeout-secs=120"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

python $INFER_TEST >$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

# Clean up bucket contents
aws s3 rm $BUCKET_URL/ --recursive --include "*"


# Test with polling enabled
SERVER_ARGS="--model-repository=$S3_REPO_URL --exit-timeout-secs=120 --model-control-mode=poll"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# copy contents of /models into S3 bucket and wait for them to be loaded.
aws s3 cp models/ $BUCKET_URL/ --recursive --include "*"
sleep 420

set +e

python $INFER_TEST >$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

# Clean up bucket contents and delete bucket
aws s3 rm $BUCKET_URL/ --recursive --include "*"
aws s3 rb $BUCKET_URL

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
fi

exit $RET
