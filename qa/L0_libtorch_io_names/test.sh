#!/bin/bash
# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=models"
SERVER_LOG="./inference_server.log"
DATADIR=/data/inferenceserver/${REPO_VERSION}/qa_model_repository
source ../common/util.sh

# Test io names
rm -rf models && mkdir -p models
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

rm -f *.log

run_server
if [ "$SERVER_PID" == "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    exit 1
fi

set +e
IO_NAMES_CLIENT=./io_names_client.py
CLIENT_LOG=client.log
python $IO_NAMES_CLIENT >> $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
    exit 1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# Test unsupported INPUT data type
rm -rf models && mkdir -p models
cp -r $DATADIR/libtorch_int32_int8_int8 models/libtorch_invalid_input_type && \
    sed -i 's/libtorch_int32_int8_int8/libtorch_invalid_input_type/' models/libtorch_invalid_input_type/config.pbtxt && \
    sed -i 's/TYPE_INT32/TYPE_UINT32/' models/libtorch_invalid_input_type/config.pbtxt

rm -f *.log

run_server
if [ "$SERVER_PID" != "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Unexpected server start $SERVER\n***"
    exit 1
fi

set +e
grep "unsupported datatype TYPE_UINT32 for input 'INPUT0' for model 'libtorch_invalid_input_type'" $SERVER_LOG
if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Unsupported INPUT datatype not found in server log\n***"
    exit 1
fi
set -e

# Test unsupported OUTPUT data type
rm -rf models && mkdir -p models
cp -r $DATADIR/libtorch_int32_int8_int8 models/libtorch_invalid_output_type && \
    sed -i 's/libtorch_int32_int8_int8/libtorch_invalid_output_type/' models/libtorch_invalid_output_type/config.pbtxt && \
    sed -i 's/TYPE_INT8/TYPE_UINT64/' models/libtorch_invalid_output_type/config.pbtxt

rm -f *.log

run_server
if [ "$SERVER_PID" != "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Unexpected server start $SERVER\n***"
    exit 1
fi

set +e
grep "unsupported datatype TYPE_UINT64 for output 'OUTPUT__0' for model 'libtorch_invalid_output_type'" $SERVER_LOG
if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Unsupported OUTPUT datatype not found in server log\n***"
    exit 1
fi
set -e

# Test unsupported sequence_batching data type
rm -rf models && mkdir -p models
cp -r $DATADIR/../qa_variable_sequence_model_repository/libtorch_sequence_int32 models/libtorch_invalid_sequence_int32 && \
    sed -i 's/libtorch_sequence_int32/libtorch_invalid_sequence_int32/' models/libtorch_invalid_sequence_int32/config.pbtxt && \
    sed -i 's/READY__2/CORRID__2/' models/libtorch_invalid_sequence_int32/config.pbtxt && \
    sed -i 's/CONTROL_SEQUENCE_READY/CONTROL_SEQUENCE_CORRID/' models/libtorch_invalid_sequence_int32/config.pbtxt && \
    sed -i ':begin;$!N;s/CORRID\n\(.*\)int32_false_true: \[ 0, 1 \]/CORRID\ndata_type: TYPE_UINT32/' models/libtorch_invalid_sequence_int32/config.pbtxt

rm -f *.log

run_server
if [ "$SERVER_PID" != "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Unexpected server start $SERVER\n***"
    exit 1
fi

set +e
grep "input 'CORRID__2' type 'TYPE_UINT32' is not supported by PyTorch." $SERVER_LOG
if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Unsupported sequence_batching datatype not found in server log\n***"
    exit 1
fi
set -e

# Test passed
echo -e "\n***\n*** Test Passed\n***"
exit 0
