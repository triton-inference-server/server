#!/bin/bash
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

REPO_VERSION=${NVIDIA_TENSORRT_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0

set +e

RET=0

SIMPLE_HEALTH_CLIENT=../clients/simple_http_v2_health_metadata.py
SIMPLE_INFER_CLIENT=../clients/simple_http_v2_infer_client.py
SIMPLE_ASYNC_INFER_CLIENT=../clients/simple_http_v2_async_infer_client.py

rm -f *.log
rm -f *.log.*

CLIENT_LOG=`pwd`/client.log
DATADIR=`pwd`/models
SERVER=/opt/tritonserver/bin/trtserver
SERVER_ARGS="--model-repository=$DATADIR --api-version 2"
source ../common/util.sh

run_server_v2
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# Test health
python $SIMPLE_HEALTH_CLIENT -v >> ${CLIENT_LOG}.health 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.health
    RET=1
fi

for i in \
        $SIMPLE_INFER_CLIENT \
        $SIMPLE_ASYNC_INFER_CLIENT \
        ; do
    BASE=$(basename -- $i)
    SUFFIX="${BASE%.*}"
    python $i -v >> "${CLIENT_LOG}.${SUFFIX}" 2>&1
    if [ $? -ne 0 ]; then
        cat "${CLIENT_LOG}.${SUFFIX}"
        RET=1
    fi
done

kill $SERVER_PID
wait $SERVER_PID

set -e


if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
