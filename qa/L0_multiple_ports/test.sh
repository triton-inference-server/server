#!/bin/bash
# Copyright 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

export CUDA_VISIBLE_DEVICES=0

MULTI_PORT_TESTS_PY=multi_port_tests.py

CLIENT_LOG="./client.log"
SERVER_LOG="./inference_server.log"

DATADIR=`pwd`/models
SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

rm -f $CLIENT_LOG $SERVER_LOG

RET=0

# CUSTOM CASES

# allow overrules - grpc still works
SERVER_ARGS="--model-repository=$DATADIR --http-port -1 --allow-http 0"
run_server_nowait
sleep 5
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
kill $SERVER_PID || true
wait $SERVER_PID || true

# overlap with grpc default
SERVER_ARGS="--model-repository=$DATADIR --http-port 8001"
set +e
run_server_nowait
sleep 5
if kill $SERVER_PID && wait $SERVER_PID ; then
    echo -e "\n***\n*** Should not have started $SERVER\n***"
    RET=1
    cat $SERVER_LOG
fi
set -e

# overlap with http default
SERVER_ARGS="--model-repository=$DATADIR --grpc-port 8000"
set +e
run_server_nowait
sleep 5
if kill $SERVER_PID && wait $SERVER_PID ; then
    echo -e "\n***\n*** Should not have started $SERVER\n***"
    RET=1
    cat $SERVER_LOG
fi
set -e

# overlap with metrics default
SERVER_ARGS="--model-repository=$DATADIR --http-port 8002"
set +e
run_server_nowait
sleep 5
if kill $SERVER_PID && wait $SERVER_PID ; then
    echo -e "\n***\n*** Should not have started $SERVER\n***"
    RET=1
    cat $SERVER_LOG
fi
set -e

# overlap with metrics default
SERVER_ARGS="--model-repository=$DATADIR --grpc-port 8002"
set +e
run_server_nowait
sleep 5
if kill $SERVER_PID && wait $SERVER_PID ; then
    echo -e "\n***\n*** Should not have started $SERVER\n***"
    RET=1
    cat $SERVER_LOG
fi
set -e

# disable metrics - no overlap with metrics default
SERVER_ARGS="--model-repository=$DATADIR --http-port 8002 --allow-metrics 0"
run_server_nowait
sleep 5
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
set +e
code=`curl -s -w %{http_code} -o ./curl.out localhost:8002/v2/health/ready`
set -e
if [ "$code" != "200" ]; then
    RET=1
fi
kill $SERVER_PID || true
wait $SERVER_PID || true

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test Failed\n***"
fi
exit $RET
