#!/bin/bash
# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TEST_LOG="./triton_repo_agent_test.log"
TRITON_REPO_AGENT_TEST=./repo_agent_test


export CUDA_VISIBLE_DEVICES=0

rm -fr *.log

set +e
$TRITON_REPO_AGENT_TEST >>$TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Repo Agent Unit Test Failed\n***"
    RET=1
fi
set -e

rm -rf /opt/tritonserver/repoagents/relocation
mkdir -p /opt/tritonserver/repoagents/relocation &&
    cp libtritonrepoagent_relocation.so /opt/tritonserver/repoagents/relocation/.

SERVER=/opt/tritonserver/bin/tritonserver

SERVER_ARGS="--model-repository=`pwd`/models"
SERVER_LOG="./inference_server.log"
run_server
if [ "$SERVER_PID" != "0" ]; then
    kill $SERVER_PID
    wait $SERVER_PID

    echo -e "\n***\n*** Expect fail to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
grep "Poll failed for model directory 'relocation_sanity_check': Relocation repoagent expects config does not contain 'model_repository_agents' field when 'empty_config' has value 'true' for relocation agent" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected repo agent of 'relocation_sanity_check' returns error on load\n***"
    RET=1
fi
grep "Poll failed for model directory 'chain_relocation': Relocation repoagent" $SERVER_LOG
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** Failed. Expected repo agent of 'chain_relocation' returns success on load\n***"
    RET=1
fi
set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $TEST_LOG
    cat $SERVER_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
