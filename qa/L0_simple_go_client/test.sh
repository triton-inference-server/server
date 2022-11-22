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

GO_CLIENT_DIR=client/src/grpc_generated/go
SIMPLE_GO_CLIENT=${GO_CLIENT_DIR}/grpc_simple_client.go

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

# Generate Go stubs.
rm -fr client common
git clone -b dyas-go-fix https://github.com/triton-inference-server/client.git
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

pushd ${GO_CLIENT_DIR}
git clone --single-branch --depth=1 -b $TRITON_COMMON_REPO_TAG \
    https://github.com/triton-inference-server/common.git
bash gen_go_stubs.sh
popd

PACKAGE_PATH="${GOPATH}/src/github.com/triton-inference-server"
rm -r ${PACKAGE_PATH}/client
mkdir -p ${PACKAGE_PATH}
cp -r client $PACKAGE_PATH

set +e

# Run test for GRPC variant of go client
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
