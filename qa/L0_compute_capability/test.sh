#!/bin/bash
# Copyright 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
source ../common/util.sh

rm -f *.log

RET=0

# If BACKENDS not specified, set to all
BACKENDS=${BACKENDS:="graphdef savedmodel onnx libtorch plan"}

for BACKEND in $BACKENDS; do
    # Need just one model for the backend...
    rm -fr models && mkdir models
    cp -r ${DATADIR}/qa_model_repository/${BACKEND}_float32_float32_float32 \
        models/.

    if [ "$BACKEND" != "plan" ]; then
        for MC in `ls models/*/config.pbtxt`; do
            echo "instance_group [ { kind: KIND_GPU }]" >> $MC
        done
    fi

    # Run with a high minimum capability so that no GPUs are
    # recognized. This should cause the server to fail to start since
    # we explicitly asked for a GPU in the instance_group.
    SERVER_ARGS="--min-supported-compute-capability=100.0 --model-repository=`pwd`/models"
    SERVER_LOG="./inference_server_${BACKEND}_cc100.log"
    run_server
    if [ "$SERVER_PID" != "0" ]; then
        echo -e "\n***\n*** Unexpected success with min compute 100.0 for ${BACKEND}\n***"
        RET=1

        kill $SERVER_PID
        wait $SERVER_PID
    fi

    # Run with a low minimum capability and make sure GPUs are
    # recognized.
    SERVER_ARGS="--min-supported-compute-capability=1.0 --model-repository=`pwd`/models"
    SERVER_LOG="./inference_server_${BACKEND}_cc1.log"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Unexpected failure with min compute 1.0 for ${BACKEND}\n***"
        RET=1
    else
        kill $SERVER_PID
        wait $SERVER_PID
    fi
done

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
fi

exit $RET
