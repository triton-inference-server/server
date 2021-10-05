#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TRITON_COMMON_REPO_TAG=${TRITON_COMMON_REPO_TAG:="main"}

RET=0

rm -f *.log.*

# Get the proto files from the common repo
rm -fr common
git clone --single-branch --depth=1 -b $TRITON_COMMON_REPO_TAG \
    https://github.com/triton-inference-server/common.git
cp common/protobuf/*.proto java/library/src/main/proto/.

# Compile library
(cd java/library && \
    mvn compile && \
    cp -R target/generated-sources/protobuf/java/inference ../examples/src/main/java/inference && \
    cp -r target/generated-sources/protobuf/grpc-java/inference/*.java ../examples/src/main/java/inference/)

# Build simple java and scala client example
(cd java/examples && mvn clean install)

CLIENT_LOG=`pwd`/client.log
DATADIR=`pwd`/models
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=$DATADIR"
source ../common/util.sh

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
pushd java/examples

# Test grpc_generated simple java client example
mvn exec:java -Dexec.mainClass=clients.SimpleJavaClient -Dexec.args="localhost 8001" >> ${CLIENT_LOG}.java 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.java
    RET=1
fi

# Test grpc_generated simple scala client example
mvn exec:java -Dexec.mainClass=clients.SimpleClient -Dexec.args="localhost 8001" >> ${CLIENT_LOG}.scala 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.scala
    RET=1
fi

popd

# Test simple infer java client
SIMPLE_INFER_JAVA_CLIENT=../clients/SimpleInferClient.jar

pushd ../clients

java -jar ${SIMPLE_INFER_JAVA_CLIENT} >> ${CLIENT_LOG}.simple_infer_java 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.simple_infer_java
    RET=1
fi

popd
set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
