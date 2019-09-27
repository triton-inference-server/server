#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

SIMPLE_CLIENT=../clients/simple_client
TRACE_SUMMARY=../common/trace_summary.py

SERVER=/opt/tensorrtserver/bin/trtserver
source ../common/util.sh

rm -f *.log
RET=0

# trace-level=OFF make sure no tracing
SERVER_ARGS="--trace-file=trace_off.log --trace-level=OFF --trace-rate=1 --model-repository=`pwd`/models"
SERVER_LOG="./inference_server_off.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

for p in {1..10}; do
    $SIMPLE_CLIENT >> client_off.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    $SIMPLE_CLIENT -i grpc -u localhost:8001 >> client_off.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi
done

set -e

kill $SERVER_PID
wait $SERVER_PID

set +e

if [ -f ./trace_off.log ]; then
    echo -e "\n***\n*** Test Failed, unexpected generation of trace_off.log\n***"
    RET=1
fi

set -e

# trace-rate == 1, trace-level=MIN make sure every request is traced
SERVER_ARGS="--trace-file=trace_1.log --trace-level=MIN --trace-rate=1 --model-repository=`pwd`/models"
SERVER_LOG="./inference_server_1.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

for p in {1..10}; do
    $SIMPLE_CLIENT >> client_1.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    $SIMPLE_CLIENT -i grpc -u localhost:8001 >> client_1.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi
done

set -e

kill $SERVER_PID
wait $SERVER_PID

set +e

$TRACE_SUMMARY -t trace_1.log > summary_1.log

if [ `grep -c "compute input end" summary_1.log` != "0" ]; then
    cat summary_1.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

if [ `grep -c ^simple summary_1.log` != "20" ]; then
    cat summary_1.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

# trace-rate == 6, trace-level=MIN
SERVER_ARGS="--grpc-infer-thread-count=1 --grpc-stream-infer-thread-count=1 --http-thread-count=1 --trace-file=trace_6.log --trace-level=MIN --trace-rate=6 --model-repository=`pwd`/models"
SERVER_LOG="./inference_server_6.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

for p in {1..10}; do
    $SIMPLE_CLIENT >> client_6.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    $SIMPLE_CLIENT -i grpc -u localhost:8001 >> client_6.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi
done

set -e

kill $SERVER_PID
wait $SERVER_PID

set +e

$TRACE_SUMMARY -t trace_6.log > summary_6.log

if [ `grep -c "compute input end" summary_6.log` != "0" ]; then
    cat summary_6.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

if [ `grep -c ^simple summary_6.log` != "3" ]; then
    cat summary_6.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

# trace-rate == 9, trace-level=MAX
SERVER_ARGS="--grpc-infer-thread-count=1 --grpc-stream-infer-thread-count=1 --http-thread-count=1 --trace-file=trace_9.log --trace-level=MAX --trace-rate=9 --model-repository=`pwd`/models"
SERVER_LOG="./inference_server_9.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

for p in {1..10}; do
    $SIMPLE_CLIENT >> client_9.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    $SIMPLE_CLIENT -i grpc -u localhost:8001 >> client_9.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi
done

set -e

kill $SERVER_PID
wait $SERVER_PID

set +e

$TRACE_SUMMARY -t trace_9.log > summary_9.log

if [ `grep -c "compute input end" summary_9.log` != "2" ]; then
    cat summary_9.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

if [ `grep -c ^simple summary_9.log` != "2" ]; then
    cat summary_9.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
