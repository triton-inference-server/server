#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

MODELSDIR=`pwd`/models
DATADIR=/data/inferenceserver/20.09dev/qa_model_repository

export CUDA_VISIBLE_DEVICES=0

# Must explicitly set LD_LIBRARY_PATH so that clients can find
# libtritonserver.so.
LD_LIBRARY_PATH=/opt/tritonserver/lib:$LD_LIBRARY_PATH

rm -f *.log

RET=0

MULTI_SERVER=multi_server
CLIENT_LOG=$MULTI_SERVER
MULTI_SERVER=./$MULTI_SERVER

I=1
for trial in graphdef onnx libtorch ; do
    full=${trial}_float32_float32_float32
    rm -rf ${MODELSDIR}${I}
    mkdir -p ${MODELSDIR}${I}/simple${I}/1 && \
        cp -r $DATADIR/${full}/1/* ${MODELSDIR}${I}/simple${I}/1/. && \
        cp $DATADIR/${full}/config.pbtxt ${MODELSDIR}${I}/simple${I}/. && \
        (cd ${MODELSDIR}${I}/simple${I} && \
                sed -i "s/^name:.*/name: \"simple${I}\"/" config.pbtxt && \
                sed -i "s/label_filename:.*//" config.pbtxt)
    I=$(($I+1))
done

set +e

# No memory type enforcement
$MULTI_SERVER -r ${MODELSDIR}1 -r ${MODELSDIR}2 -r ${MODELSDIR}3 >>$CLIENT_LOG.$full.log 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG.$full.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

# Enforce I/O to be in specific memory type
for MEM_TYPE in system pinned gpu ; do
    $MULTI_SERVER -r ${MODELSDIR}1 -r ${MODELSDIR}2 -r ${MODELSDIR}3 -m $MEM_TYPE >>$CLIENT_LOG.$full.$MEM_TYPE.log 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG.$full.$MEM_TYPE.log
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
done

set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
