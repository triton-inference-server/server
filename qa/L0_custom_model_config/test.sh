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
if [ ! -z "$TEST_REPO_ARCH" ]; then
    REPO_VERSION=${REPO_VERSION}_${TEST_REPO_ARCH}
fi

export CUDA_VISIBLE_DEVICES=0

DATADIR="/data/inferenceserver/${REPO_VERSION}"
CLIENT_LOG="./client.log"
SERVER_LOG="./inference_server.log"

SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

RET=0
rm -fr *.log

rm -fr models && mkdir models
cp -r $DATADIR/qa_model_repository/savedmodel_nobatch_float32_float32_float32 models/.
mkdir models/savedmodel_nobatch_float32_float32_float32/configs

test_custom_config()
{
    VERSION=$@

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e
    code=`curl -s -w %{http_code} -o ./curl.out localhost:8000/v2/models/savedmodel_nobatch_float32_float32_float32/config`
    set -e
    if [ "$code" != "200" ]; then
        cat $out.out
        echo -e "\n***\n*** Test Failed to GET model configuration\n***"
        RET=1
    fi

    matches=`grep -o "\"version_policy\":{\"specific\":{\"versions\":\[$VERSION\]}}" curl.out | wc -l`
    if [ $matches -ne 1 ]; then
        cat curl.out
        echo -e "\n***\n*** Expected 1 version_policy:specific:versions, got $matches\n***"
        RET=1
    fi

    kill $SERVER_PID
    wait $SERVER_PID
}

# Prepare the file structure
VERSION_DEFAULT="1,3"
VERSION_H100="1"
VERSION_V100="2"
VERSION_CUSTOM="3"

# Distinguish configs with different model versions
(cd models/savedmodel_nobatch_float32_float32_float32 && \
     sed -i "s/^version_policy:.*/version_policy: { specific: { versions: [$VERSION_DEFAULT] }}/" config.pbtxt)
(cd models/savedmodel_nobatch_float32_float32_float32 && \
     cp config.pbtxt configs/h100.pbtxt && \
     sed -i "s/^version_policy:.*/version_policy: { specific: { versions: [$VERSION_H100] }}/" configs/h100.pbtxt)
(cd models/savedmodel_nobatch_float32_float32_float32 && \
     cp config.pbtxt configs/v100.pbtxt && \
     sed -i "s/^version_policy:.*/version_policy: { specific: { versions: [$VERSION_V100] }}/" configs/v100.pbtxt)
(cd models/savedmodel_nobatch_float32_float32_float32 && \
     cp config.pbtxt configs/config.pbtxt && \
     sed -i "s/^version_policy:.*/version_policy: { specific: { versions: [$VERSION_CUSTOM] }}/" configs/config.pbtxt)

# Test default model config
SERVER_ARGS="--model-repository=`pwd`/models"
test_custom_config $VERSION_DEFAULT

# Test model-config-name=h100
SERVER_ARGS="--model-repository=`pwd`/models --model-config-name=h100"
test_custom_config $VERSION_H100

# Test model-config-name=v100
SERVER_ARGS="--model-repository=`pwd`/models --model-config-name=v100"
test_custom_config $VERSION_V100

# Test model-config-name=config
SERVER_ARGS="--model-repository=`pwd`/models --model-config-name=config"
test_custom_config $VERSION_CUSTOM

# Test model-config-name=h200. Expect fall back to default config since h200 config does not exist.
SERVER_ARGS="--model-repository=`pwd`/models --model-config-name=h200"
test_custom_config $VERSION_DEFAULT

# Test model-config-name=
SERVER_ARGS="--model-repository=`pwd`/models --model-config-name="
run_server
if [ "$SERVER_PID" != "0" ]; then
    echo -e "\n***\n*** Failed: $SERVER started successfully when it was expected to fail\n***"
    cat $SERVER_LOG
    RET=1

    kill $SERVER_PID
    wait $SERVER_PID
fi

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test Failed\n***"
fi

exit $RET
