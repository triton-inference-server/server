#!/bin/bash
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

source ../common/util.sh

RET=0

TEST_LOG="./register_api_test.log"
TEST_EXEC=./register_api_test

export CUDA_VISIBLE_DEVICES=0

rm -fr *.log

# Setup repositories for testing, note that we use
# model version as hint for which directory is used for model loading
mkdir empty_models models_0 models_1
mkdir -p models_0/model_0/1 && \
    cp config.pbtxt models_0/model_0/. && \
    (cd models_0/model_0 && \
        sed -i "s/^name:.*/name: \"model_0\"/" config.pbtxt)
mkdir -p models_1/model_0/2 && \
    cp config.pbtxt models_1/model_0/. && \
    (cd models_1/model_0 && \
        sed -i "s/^name:.*/name: \"model_0\"/" config.pbtxt)
mkdir -p models_1/model_1/3 && \
    cp config.pbtxt models_1/model_1/. && \
    (cd models_1/model_1 && \
        sed -i "s/^name:.*/name: \"model_1\"/" config.pbtxt)

set +e
LD_LIBRARY_PATH=/opt/tritonserver/lib:$LD_LIBRARY_PATH $TEST_EXEC >>$TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Register API Unit Test Failed\n***"
    RET=1
fi
set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $TEST_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
