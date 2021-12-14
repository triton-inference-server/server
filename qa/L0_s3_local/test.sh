#!/bin/bash
# Copyright 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
PERF_CLIENT=../clients/perf_client

DATADIR="/data/inferenceserver/${REPO_VERSION}/qa_model_repository"
BACKENDS="graphdef libtorch onnx plan savedmodel"

rm -rf models && mkdir models
for BACKEND in $BACKENDS; do
    cp -r $DATADIR/${BACKEND}_float32_float32_float32 models/.
    # Remove version policy from config.pbtxt
    sed -i '/^version_policy/d' models/${BACKEND}_float32_float32_float32/config.pbtxt
done

# Create model with name that has all types of allowed characters
DUMMY_MODEL="Model_repo-1.0"
cp -r models/libtorch_float32_float32_float32 models/$DUMMY_MODEL
sed -i 's/libtorch_float32_float32_float32/Model_repo-1.0/g' models/$DUMMY_MODEL/config.pbtxt

SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

rm -f *.log*

## Setup local MINIO server
(wget https://dl.min.io/server/minio/release/linux-amd64/minio && \
    chmod +x minio && \
    mv minio /usr/local/bin && \
    mkdir /usr/local/share/minio && \
    mkdir /etc/minio)

export MINIO_ACCESS_KEY="minio"
MINIO_VOLUMES="/usr/local/share/minio/"
MINIO_OPTS="-C /etc/minio --address localhost:4572"
export MINIO_SECRET_KEY="miniostorage"

(curl -O https://raw.githubusercontent.com/minio/minio-service/master/linux-systemd/minio.service && \
    mv minio.service /etc/systemd/system)

# Start minio server
/usr/local/bin/minio server $MINIO_OPTS $MINIO_VOLUMES &
MINIO_PID=$!

export AWS_ACCESS_KEY_ID=minio && \
    export AWS_SECRET_ACCESS_KEY=miniostorage

# Force version to 0.7 to prevent failures due to version changes
python -m pip install awscli-local==0.07

# Needed to set correct port for awscli-local
ENDPOINT_FLAG="--endpoint-url=http://localhost:4572"

# Cleanup bucket if exists
awslocal $ENDPOINT_FLAG s3 rm s3://demo-bucket1.0 --recursive --include "*" && \
    awslocal $ENDPOINT_FLAG s3 rb s3://demo-bucket1.0 || true

# Create and add data to bucket
awslocal $ENDPOINT_FLAG s3 mb s3://demo-bucket1.0 && \
    awslocal $ENDPOINT_FLAG s3 sync models s3://demo-bucket1.0

RET=0

# Test with hostname and IP address
for HOST in "127.0.0.1" "localhost"; do
    SERVER_ARGS="--model-repository=s3://$HOST:4572/demo-bucket1.0 --model-control-mode=explicit"
    if [ "$HOST" = "127.0.0.1" ]; then
        SERVER_LOG="./inference_server_hostname.log"
    else
        SERVER_LOG="./inference_server_ip.log"
    fi

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        # Kill minio server
        kill $MINIO_PID
        wait $MINIO_PID
        exit 1
    fi

    set +e
    for BACKEND in $BACKENDS; do
        code=`curl -s -w %{http_code} -X POST localhost:8000/v2/repository/models/${BACKEND}_float32_float32_float32/load`
        if [ "$code" != "200" ]; then
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi

        $PERF_CLIENT -m ${BACKEND}_float32_float32_float32 -p 3000 -t 1 >$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Failed\n***"
            cat $CLIENT_LOG
            RET=1
        fi
    done

    # Try to load model with name that checks for all types of allowed characters
    code=`curl -s -w %{http_code} -X POST localhost:8000/v2/repository/models/${DUMMY_MODEL}/load`
    if [ "$code" != "200" ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    set -e

    kill $SERVER_PID
    wait $SERVER_PID
done

# Test with Polling
SERVER_ARGS="--model-repository=s3://localhost:4572/demo-bucket1.0 --model-control-mode=poll"
SERVER_LOG="./inference_server_poll.log"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    # Kill minio server
    kill $MINIO_PID
    wait $MINIO_PID
    exit 1
fi

cp -r models/libtorch_float32_float32_float32/1 models/libtorch_float32_float32_float32/4
awslocal $ENDPOINT_FLAG s3 sync models s3://demo-bucket1.0

sleep 20

set + e
CURL_LOG=$(curl -X POST localhost:8000/v2/repository/index)
if [[ "$CURL_LOG" != *"{\"name\":\"libtorch_float32_float32_float32\",\"version\":\"3\",\"state\":\"UNAVAILABLE\",\"reason\":\"unloaded\"}"* ]]; then
    echo -e "\n***\n*** Failed. Server did not unload libtorch_float32_float32_float32 version 3\n***"
    RET=1
fi

if [[ "$CURL_LOG" != *"{\"name\":\"libtorch_float32_float32_float32\",\"version\":\"4\",\"state\":\"READY\"}"* ]]; then
    echo -e "\n***\n*** Failed. Server did not load libtorch_float32_float32_float32 version 4\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# Destroy bucket
awslocal $ENDPOINT_FLAG s3 rm s3://demo-bucket1.0 --recursive --include "*" && \
    awslocal $ENDPOINT_FLAG s3 rb s3://demo-bucket1.0

# Test with Polling, no model configuration file - with strict model config disabled
rm -rf models && mkdir models
cp -r $DATADIR/savedmodel_float32_float32_float32 models/.
rm models/savedmodel_float32_float32_float32/config.pbtxt

awslocal $ENDPOINT_FLAG s3 mb s3://demo-bucket1.0 && \
    awslocal $ENDPOINT_FLAG s3 sync models s3://demo-bucket1.0

SERVER_ARGS="--model-repository=s3://localhost:4572/demo-bucket1.0 --model-control-mode=poll --strict-model-config=false"
SERVER_LOG="./inference_server_noconfig.log"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    # Kill minio server
    kill $MINIO_PID
    wait $MINIO_PID
    exit 1
fi

$PERF_CLIENT -m savedmodel_float32_float32_float32 -p 3000 -t 1 > $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    cat $CLIENT_LOG
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

# Destroy bucket
awslocal $ENDPOINT_FLAG s3 rm s3://demo-bucket1.0 --recursive --include "*" && \
    awslocal $ENDPOINT_FLAG s3 rb s3://demo-bucket1.0

# Kill minio server
kill $MINIO_PID
wait $MINIO_PID

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test Failed\n***"
fi

exit $RET
