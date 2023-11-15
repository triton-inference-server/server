#!/bin/bash
# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
TEST_RESULT_FILE='test_results.txt'

export ENSEMBLES=0
BACKENDS=${BACKENDS:="libtorch onnx plan"}
export BACKENDS
export IMPLICIT_STATE=1
INITIAL_STATE_ZERO=${INITIAL_STATE_ZERO:="0"}
INITIAL_STATE_FILE=${INITIAL_STATE_FILE:="0"}
SINGLE_STATE_BUFFER=${SINGLE_STATE_BUFFER:="0"}

export INITIAL_STATE_ZERO
export INITIAL_STATE_FILE
export SINGLE_STATE_BUFFER

MODELDIR=${MODELDIR:=`pwd`/models}
TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
BACKEND_DIR=${TRITON_DIR}/backends
source ../common/util.sh

# Setup the custom models shared library
cp ./libtriton_implicit_state.so models/no_implicit_state/
cp ./libtriton_implicit_state.so models/no_state_update/
cp ./libtriton_implicit_state.so models/wrong_internal_state/
cp ./libtriton_implicit_state.so models/single_state_buffer/
cp ./libtriton_implicit_state.so models/growable_memory/

mkdir -p models/no_implicit_state/1/
mkdir -p models/no_state_update/1/
mkdir -p models/wrong_internal_state/1/
mkdir -p models/single_state_buffer/1/
mkdir -p models/growable_memory/1/

for BACKEND in $BACKENDS; do
    dtype="int32"
    model_name=${BACKEND}_nobatch_sequence_${dtype}
    rm -rf models/$model_name
    cp -r $DATADIR/qa_sequence_implicit_model_repository/$model_name models
    output_dtype=

    # In order to allow the state to be returned, the model must describe
    # state as one of the outputs of the model.
    model_name_allow_output=${BACKEND}_nobatch_sequence_${dtype}_output
    rm -rf models/$model_name_allow_output
    cp -r $DATADIR/qa_sequence_implicit_model_repository/$model_name models/$model_name_allow_output

    if [ $BACKEND == "libtorch" ]; then
    	(cd models/$model_name_allow_output && \
    	    sed -i "s/^name:.*/name: \"$model_name_allow_output\"/" config.pbtxt && \
    	    echo -e "output [{ name: \"OUTPUT_STATE__1\" \n data_type: TYPE_INT32 \n dims: [ 1 ] }]" >> config.pbtxt)
    else
    	(cd models/$model_name_allow_output && \
    	    sed -i "s/^name:.*/name: \"$model_name_allow_output\"/" config.pbtxt && \
    	    echo -e "output [{ name: \"OUTPUT_STATE\" \n data_type: TYPE_INT32 \n dims: [ 1 ] }]" >> config.pbtxt)
    fi
done

CLIENT_LOG=`pwd`/client.log
SERVER_ARGS="--backend-directory=${BACKEND_DIR} --model-repository=${MODELDIR} --cuda-virtual-address-size=0:$((1024*1024*4))"
IMPLICIT_STATE_CLIENT='implicit_state.py'
EXPECTED_TEST_NUM=7
rm -rf $CLIENT_LOG

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

python3 $IMPLICIT_STATE_CLIENT > $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Implicit State FAILED\n***"
    cat ${CLIENT_LOG}
    exit 1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_TEST_NUM
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

(cd ../L0_sequence_batcher/ && bash -ex test.sh)
RET=$?

if [ $RET == 0 ]; then
    echo -e "\n***\n*** Implicit State Passed\n***"
else
    echo -e "\n***\n*** Implicit State FAILED\n***"
    exit 1
fi

exit $RET

