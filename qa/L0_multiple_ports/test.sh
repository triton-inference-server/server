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
SERVER_TIMEOUT=15
source ../common/util.sh

rm -f $CLIENT_LOG $SERVER_LOG

RET=0

# CUSTOM CASES

for p in http grpc; do
    # allow illegal http/grpc port if disabled
    SERVER_ARGS="--model-repository=$DATADIR --${p}-port -47 --allow-${p} 0"
    run_server_nowait
    sleep 15
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi
    kill $SERVER_PID
    wait $SERVER_PID

    # allow http/grpc port overlap with grpc/http default if disabled
    if [ "$p" == "http" ]; then
        SERVER_ARGS="--model-repository=$DATADIR --http-port 8001 --allow-http 0"
        run_server_nowait
        sleep 15
    else
        SERVER_ARGS="--model-repository=$DATADIR --grpc-port 8000 --allow-grpc 0"
        run_server
    fi
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi
    kill $SERVER_PID
    wait $SERVER_PID

    # error if http/grpc port overlaps with grpc/http default port
    if [ "$p" == "http" ]; then
        SERVER_ARGS="--model-repository=$DATADIR --http-port 8001"
    else
        SERVER_ARGS="--model-repository=$DATADIR --grpc-port 8000"
    fi
    run_server
    if [ "$SERVER_PID" != "0" ]; then
        set +e
        kill $SERVER_PID
        wait $SERVER_PID
        if [ "$?" == "0" ]; then
            echo -e "\n***\n*** unexpected start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi
        set -e
    fi

    # allow http/grpc port overlap with grpc/http explicit if disabled
    if [ "$p" == "http" ]; then
        SERVER_ARGS="--model-repository=$DATADIR --http-port 8007 --grpc-port 8007 --allow-http 0"
    else
        SERVER_ARGS="--model-repository=$DATADIR --grpc-port 8007 --http-port 8007 --allow-grpc 0"
    fi
    run_server_nowait
    sleep 15
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi
    kill $SERVER_PID
    wait $SERVER_PID

    # error if http/grpc port overlaps with grpc/http explicit port
    if [ "$p" == "http" ]; then
        SERVER_ARGS="--model-repository=$DATADIR --http-port 8003 --grpc-port 8003"
        run_server_nowait
        sleep 15
        if [ "$SERVER_PID" != "0" ]; then
            set +e
            kill $SERVER_PID
            wait $SERVER_PID
            if [ "$?" == "0" ]; then
                echo -e "\n***\n*** unexpected start $SERVER\n***"
                cat $SERVER_LOG
                exit 1
            fi
            set -e
        fi
    else
        # skip, same as http case
        true
    fi

    # allow http/grpc port overlap with metrics default port if disabled
    if [ "$p" == "http" ]; then
        SERVER_ARGS="--model-repository=$DATADIR --http-port 8002 --allow-http 0"
        run_server_nowait
        sleep 15
    else
        SERVER_ARGS="--model-repository=$DATADIR --grpc-port 8002 --allow-grpc 0"
        run_server
    fi
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi
    kill $SERVER_PID
    wait $SERVER_PID

    # error if http/grpc port overlaps with metrics default port
    if [ "$p" == "http" ]; then
        SERVER_ARGS="--model-repository=$DATADIR --http-port 8002"
    else
        SERVER_ARGS="--model-repository=$DATADIR --grpc-port 8002"
    fi
    run_server
    if [ "$SERVER_PID" != "0" ]; then
        set +e
        kill $SERVER_PID
        wait $SERVER_PID
        if [ "$?" == "0" ]; then
            echo -e "\n***\n*** unexpected start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi
        set -e
    fi

    # allow metrics port overlap with http/grpc default port if disabled
    if [ "$p" == "http" ]; then
        SERVER_ARGS="--model-repository=$DATADIR --metrics-port 8000 --allow-metrics 0"
    else
        SERVER_ARGS="--model-repository=$DATADIR --metrics-port 8001 --allow-metrics 0"
    fi
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi
    kill $SERVER_PID
    wait $SERVER_PID

    # error if metrics port overlaps with http/grpc default port
    if [ "$p" == "http" ]; then
        SERVER_ARGS="--model-repository=$DATADIR --metrics-port 8000"
    else
        SERVER_ARGS="--model-repository=$DATADIR --metrics-port 8001"
    fi
    run_server
    if [ "$SERVER_PID" != "0" ]; then
        set +e
        kill $SERVER_PID
        wait $SERVER_PID
        if [ "$?" == "0" ]; then
            echo -e "\n***\n*** unexpected start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi
        set -e
    fi
done

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test Failed\n***"
fi
exit $RET
