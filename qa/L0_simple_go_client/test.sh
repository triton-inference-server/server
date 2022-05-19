#!/bin/bash
# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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

export CUDA_VISIBLE_DEVICES=0

TRITON_COMMON_REPO_TAG=${TRITON_COMMON_REPO_TAG:="main"}

SIMPLE_GO_CLIENT=grpc_simple_client.go

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS=--model-repository=`pwd`/models
SERVER_LOG="./inference_server.log"
source ../common/util.sh

rm -f *.log

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

RET=0

# Fix to allow global stubs import
sed -i 's/.\/nvidia_inferenceserver/nvidia_inferenceserver/g' $SIMPLE_GO_CLIENT

PACKAGE_PATH="${GOPATH}/src"
mkdir -p ${PACKAGE_PATH}

# Get the proto files from the common repo
rm -fr common
git clone --single-branch --depth=1 -b $TRITON_COMMON_REPO_TAG \
    https://github.com/triton-inference-server/common.git
mkdir core && cp common/protobuf/*.proto core/.

# Requires protoc and protoc-gen-go plugin: https://github.com/golang/protobuf#installation
# Use "M" arguments since go_package is not specified in .proto files.
# As mentioned here: https://developers.google.com/protocol-buffers/docs/reference/go-generated#package
GO_PACKAGE="nvidia_inferenceserver"
protoc -I core --go_out=plugins=grpc:${PACKAGE_PATH} --go_opt=Mgrpc_service.proto=./${GO_PACKAGE} \
    --go_opt=Mmodel_config.proto=./${GO_PACKAGE} core/*.proto

set +e

# Runs test for GRPC variant of go client
GO111MODULE=off go run $SIMPLE_GO_CLIENT >>client.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

if [ `grep -c "Checking Inference Outputs" client.log` != "1" ]; then
    echo -e "\n***\n*** Failed. Unable to run inference.\n***"
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
