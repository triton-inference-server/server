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

export CUDA_VISIBLE_DEVICES=0

IO_NAMES_CLIENT=./io_names_client.py
DATADIR=/data/inferenceserver/${REPO_VERSION}/qa_model_repository

rm -rf models && mkdir -p models

# Prepare models
cp -r $DATADIR/libtorch_float32_float32_float32 models/libtorch_output_index && \
    sed -i 's/libtorch_float32_float32_float32/libtorch_output_index/' models/libtorch_output_index/config.pbtxt

cp -r $DATADIR/libtorch_float32_float32_float32 models/libtorch_io_index && \
    sed -i 's/libtorch_float32_float32_float32/libtorch_io_index/' models/libtorch_io_index/config.pbtxt && \
    sed -i 's/INPUT0/INPUT__0/' models/libtorch_io_index/config.pbtxt && \
    sed -i 's/INPUT1/INPUT__1/' models/libtorch_io_index/config.pbtxt

cp -r $DATADIR/libtorch_float32_float32_float32 models/libtorch_no_output_index && \
    sed -i 's/libtorch_float32_float32_float32/libtorch_no_output_index/' models/libtorch_no_output_index/config.pbtxt && \
    sed -i 's/OUTPUT__0/OUTPUT0/' models/libtorch_no_output_index/config.pbtxt && \
    sed -i 's/OUTPUT__1/OUTPUT1/' models/libtorch_no_output_index/config.pbtxt

cp -r $DATADIR/libtorch_float32_float32_float32 models/libtorch_no_arguments_output_index && \
    sed -i 's/libtorch_float32_float32_float32/libtorch_no_arguments_output_index/' models/libtorch_no_arguments_output_index/config.pbtxt && \
    sed -i 's/INPUT0/INPUTA/' models/libtorch_no_arguments_output_index/config.pbtxt && \
    sed -i 's/INPUT1/INPUTB/' models/libtorch_no_arguments_output_index/config.pbtxt && \
    sed -i 's/OUTPUT__0/OUTPUTA/' models/libtorch_no_arguments_output_index/config.pbtxt && \
    sed -i 's/OUTPUT__1/OUTPUTB/' models/libtorch_no_arguments_output_index/config.pbtxt

cp -r $DATADIR/libtorch_float32_float32_float32 models/libtorch_mix_index && \
    sed -i 's/libtorch_float32_float32_float32/libtorch_mix_index/' models/libtorch_mix_index/config.pbtxt && \
    sed -i 's/INPUT0/INPUTA/' models/libtorch_mix_index/config.pbtxt && \
    sed -i 's/INPUT1/INPUT__1/' models/libtorch_mix_index/config.pbtxt && \
    sed -i 's/OUTPUT__0/OUTPUTA/' models/libtorch_mix_index/config.pbtxt

cp -r $DATADIR/libtorch_float32_float32_float32 models/libtorch_mix_arguments && \
    sed -i 's/libtorch_float32_float32_float32/libtorch_mix_arguments/' models/libtorch_mix_arguments/config.pbtxt && \
    sed -i 's/INPUT1/INPUTB/' models/libtorch_mix_arguments/config.pbtxt && \
    sed -i 's/OUTPUT__0/OUTPUTA/' models/libtorch_mix_arguments/config.pbtxt

cp -r $DATADIR/libtorch_float32_float32_float32 models/libtorch_mix_arguments_index && \
    sed -i 's/libtorch_float32_float32_float32/libtorch_mix_arguments_index/' models/libtorch_mix_arguments_index/config.pbtxt && \
    sed -i 's/INPUT1/INPUT__1/' models/libtorch_mix_arguments_index/config.pbtxt && \
    sed -i 's/OUTPUT__0/OUTPUT0/' models/libtorch_mix_arguments_index/config.pbtxt

cp -r $DATADIR/libtorch_float32_float32_float32 models/libtorch_unordered_index && \
    sed -i 's/libtorch_float32_float32_float32/libtorch_unordered_index/' models/libtorch_unordered_index/config.pbtxt && \
    sed -i 's/INPUT0/INPUT_TMP1/' models/libtorch_unordered_index/config.pbtxt && \
    sed -i 's/INPUT1/INPUT0/' models/libtorch_unordered_index/config.pbtxt && \
    sed -i 's/INPUT_TMP1/INPUT1/' models/libtorch_unordered_index/config.pbtxt && \
    sed -i 's/OUTPUT__0/OUT__1/' models/libtorch_unordered_index/config.pbtxt && \
    sed -i 's/OUTPUT__1/OUT__0/' models/libtorch_unordered_index/config.pbtxt


SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=models"
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

set +e

CLIENT_LOG=client.log
python $IO_NAMES_CLIENT >> $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
