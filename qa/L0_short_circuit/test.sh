#!/bin/bash
# Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
MODELDIR=${MODELDIR:=`pwd`/models}

CLIENT_LOG="./client.log"
CLIENT=short_circuit_test.py

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_TIMEOUT=20
SERVER_LOG_BASE="./inference_server"
SERVER_ARGS="--model-repository=./models --model-control-mode=explicit --log-verbose=2"
source ../common/util.sh

RET=0

export CUDA_VISIBLE_DEVICES=0

rm -rf models
mkdir models

rm $CLIENT_LOG

############################################################
# Test cases which need to be addressed:
#     - backends
#         - tensorflow
#             - savedmodel
#             - graphdef
#         - onnxruntime
#         - tensorrt
#         - python 
#         - pytorch
#     - types of actions
#         - polling a new config
#         - explicit loading with new config
#         - sending data and seeing it execute on the new instance
#         - metrics being reported
#         - OOM error? This can be exra
#     - types of config changes
#         - increase count (single group)
#         - increase count when no instance group provided in config.pbtxt (single)
#         - increase count all groups (multiple)
#         - increase count some groups (multiple)
#         - increase count some groups but re-arrange groups (multiple)
#         - decrease count
#         - decrease count all groups (multiple)
#         - decrease count some groups (multiple)
#         - change other fields within instance group
############################################################

#
# Single Instance Group Test
#

# Don't need to test different models so we can copy the same one 
# multiple times.
mkdir -p models/increase_count && cp -r $DATADIR/qa_identity_model_repository/savedmodel_nobatch_zero_1_float32/* models/increase_count
mkdir -p models/decrease_count && cp -r $DATADIR/qa_identity_model_repository/savedmodel_nobatch_zero_1_float32/* models/decrease_count
mkdir -p models/decrease_count_past_zero && cp -r $DATADIR/qa_identity_model_repository/savedmodel_nobatch_zero_1_float32/* models/decrease_count_past_zero

mkdir -p models/increase_count_no_config && cp -r $DATADIR/qa_identity_model_repository/savedmodel_nobatch_zero_1_float32/* models/increase_count_no_config
rm models/increase_count_no_config/config.pbtxt

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
echo "Starting python test..."
python3 $CLIENT >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***" >>$CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
echo "Python test complete"

set -e
echo "Killing server..."
kill_server
echo "Killing server complete"


if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
