#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

# Must run on a single device or else the TRITONSERVER_DELAY_SCHEDULER
# can fail when the requests are distributed to multiple devices.
export CUDA_VISIBLE_DEVICES=0

LEAKCHECK=/usr/bin/valgrind
LEAKCHECK_ARGS_BASE="--max-threads=3000 --tool=massif --time-unit=B"
SERVER_TIMEOUT=3600
rm -f *.log *.massif

MEMORY_GROWTH_TEST=../clients/memory_leak_test
MASSIF_TEST=../common/check_massif_log.py

DATADIR=`pwd`/models
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=$DATADIR"
source ../common/util.sh

mkdir -p $DATADIR/custom_identity_int32/1
cp libidentity.so $DATADIR/custom_identity_int32/1/.

RET=0

# Threshold memory growth in MB
MAX_ALLOWED_ALLOC="0"
export MAX_ALLOWED_ALLOC

# Run test for both HTTP and GRPC, not re-using client object. 
# 10000 inferences in each case.
EXTRA_ARGS="-r 10000"
for PROTOCOL in http grpc; do
    LEAKCHECK_LOG="./valgrind.${PROTOCOL}.c++.log"
    CLIENT_LOG="./client.${PROTOCOL}.c++.log"
    MASSIF_LOG="./${PROTOCOL}.c++.massif"
    LEAKCHECK_ARGS="$LEAKCHECK_ARGS_BASE --log-file=$LEAKCHECK_LOG --massif-out-file=$MASSIF_LOG"
    EXTRA_CLIENT_ARGS="${EXTRA_ARGS} -i ${PROTOCOL}"

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    $LEAKCHECK $LEAKCHECK_ARGS $MEMORY_GROWTH_TEST $EXTRA_CLIENT_ARGS >> ${CLIENT_LOG} 2>&1
    if [ $? -ne 0 ]; then
        cat ${CLIENT_LOG}
        RET=1
        echo -e "\n***\n*** Test FAILED\n***"
    else
        check_valgrind_log $LEAKCHECK_LOG
        if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Memory leak detected\n***"
        RET=1
        fi

        set +e
        # Check for memory growth
        python $MASSIF_TEST $MASSIF_LOG $MAX_ALLOWED_ALLOC >> ${CLIENT_LOG}.massif 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Massif Test for ${PROTOCOL} Failed\n***"
            RET=1
        fi
        cat ${CLIENT_LOG}.massif
        set -e
    fi
    # Stop Server
    kill $SERVER_PID
    wait $SERVER_PID
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
