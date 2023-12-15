#!/bin/bash
# Copyright 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

CLIENT_LOG="./client.log"
SERVER_LOG="./inference_server.log"

DATADIR=`pwd`/models
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_TIMEOUT=15
source ../common/util.sh

rm -f *.log

RET=0

# CUSTOM CASES
for address in default explicit; do
    if [ "$address" == "default" ]; then
        # without specifying address, will use "0.0.0.0" as default
        SAME_EXPLICIT_ADDRESS=""
        DIFF_EXPLICIT_ADDRESS_ARGS=""
    else
        SAME_EXPLICIT_ADDRESS="--http-address 127.0.0.1 --grpc-address 127.0.0.1 --metrics-address 127.0.0.1"
        DIFF_EXPLICIT_ADDRESS="--http-address 127.0.0.1 --grpc-address 127.0.0.2 --metrics-address 127.0.0.3"
    fi

    for p in http grpc; do
        if [ "$address" == "default" ]; then
            # allow illegal http/grpc port if disabled
            SERVER_ARGS="--model-repository=$DATADIR --${p}-port -47 --allow-${p} 0"
        else
            # allow illegal http/grpc address if disabled
            SERVER_ARGS="--model-repository=$DATADIR --${p}-address -47 --allow-${p} 0"
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

        # allow http/grpc port overlap with grpc/http default if disabled
        if [ "$p" == "http" ]; then
            SERVER_ARGS="--model-repository=$DATADIR $SAME_EXPLICIT_ADDRESS --http-port 8001 --allow-http 0"
            run_server_nowait
            sleep 15
        else
            SERVER_ARGS="--model-repository=$DATADIR $SAME_EXPLICIT_ADDRESS --grpc-port 8000 --allow-grpc 0"
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
            SERVER_ARGS="--model-repository=$DATADIR $SAME_EXPLICIT_ADDRESS --http-port 8001"
        else
            SERVER_ARGS="--model-repository=$DATADIR $SAME_EXPLICIT_ADDRESS --grpc-port 8000"
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

        # when using different addresses, allow http/grpc port overlap with grpc/http default port
        if [ "$address" == "explicit" ]; then
            if [ "$p" == "http" ]; then
                SERVER_ARGS="--model-repository=$DATADIR $DIFF_EXPLICIT_ADDRESS --http-port 8001"
            else
                SERVER_ARGS="--model-repository=$DATADIR $DIFF_EXPLICIT_ADDRESS --grpc-port 8000"
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
        fi

        # allow http/grpc port overlap with grpc/http explicit if disabled
        if [ "$p" == "http" ]; then
            SERVER_ARGS="--model-repository=$DATADIR $SAME_EXPLICIT_ADDRESS --http-port 8007 --grpc-port 8007 --allow-http 0"
        else
            SERVER_ARGS="--model-repository=$DATADIR $SAME_EXPLICIT_ADDRESS --grpc-port 8007 --http-port 8007 --allow-grpc 0"
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
            SERVER_ARGS="--model-repository=$DATADIR $SAME_EXPLICIT_ADDRESS --http-port 8003 --grpc-port 8003"
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

        # when using different addresses, allow http/grpc port overlap with grpc/http explicit
        if [ "$address" == "explicit" ]; then
            if [ "$p" == "http" ]; then
                SERVER_ARGS="--model-repository=$DATADIR $DIFF_EXPLICIT_ADDRESS --http-port 8007 --grpc-port 8007"
            else
                SERVER_ARGS="--model-repository=$DATADIR $DIFF_EXPLICIT_ADDRESS --grpc-port 8007 --http-port 8007"
            fi
            run_server_nowait
            sleep 15
            if [ "$SERVER_PID" == "0" ]; then
                echo -e "\n***\n*** Failed to start $SERVER\n***"
                cat $SERVER_LOG
                exit 1
            fi
            code=`curl -s -w %{http_code} 127.0.0.1:8007/v2/health/ready`
            if [ "$code" != "200" ]; then
                echo -e "\n***\n*** Server is not ready\n***"
                RET=1
            fi
            kill $SERVER_PID
            wait $SERVER_PID
        fi

        # allow http/grpc port overlap with metrics default port if disabled
        if [ "$p" == "http" ]; then
            SERVER_ARGS="--model-repository=$DATADIR $SAME_EXPLICIT_ADDRESS --http-port 8002 --allow-http 0"
            run_server_nowait
            sleep 15
        else
            SERVER_ARGS="--model-repository=$DATADIR $SAME_EXPLICIT_ADDRESS --grpc-port 8002 --allow-grpc 0"
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
            SERVER_ARGS="--model-repository=$DATADIR $SAME_EXPLICIT_ADDRESS --http-port 8002"
        else
            SERVER_ARGS="--model-repository=$DATADIR $SAME_EXPLICIT_ADDRESS --grpc-port 8002"
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

        # when using different addresses, allow grpc port overlap with metrics default port
        if [ "$address" == "explicit" ]; then
            if [ "$p" == "grpc" ]; then
                SERVER_ARGS="--model-repository=$DATADIR $DIFF_EXPLICIT_ADDRESS --grpc-port 8002"
                run_server_nowait
                sleep 15
                if [ "$SERVER_PID" == "0" ]; then
                    echo -e "\n***\n*** Failed to start $SERVER\n***"
                    cat $SERVER_LOG
                    exit 1
                fi
                code=`curl -s -w %{http_code} 127.0.0.1:8000/v2/health/ready`
                if [ "$code" != "200" ]; then
                    echo -e "\n***\n*** Server is not ready\n***"
                    RET=1
                fi
                kill $SERVER_PID
                wait $SERVER_PID
            else
                # http and metrics server bind to the same address, should skip this test case.
                true
            fi
        fi

        # allow metrics port overlap with http/grpc default port if disabled
        if [ "$p" == "http" ]; then
            SERVER_ARGS="--model-repository=$DATADIR $SAME_EXPLICIT_ADDRESS --metrics-port 8000 --allow-metrics 0"
        else
            SERVER_ARGS="--model-repository=$DATADIR $SAME_EXPLICIT_ADDRESS --metrics-port 8001 --allow-metrics 0"
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
            SERVER_ARGS="--model-repository=$DATADIR $SAME_EXPLICIT_ADDRESS --metrics-port 8000"
        else
            SERVER_ARGS="--model-repository=$DATADIR $SAME_EXPLICIT_ADDRESS --metrics-port 8001"
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

        # when using different addresses, allow metrics port overlap with grpc default port
        if [ "$address" == "explicit" ]; then
            if [ "$p" == "grpc" ]; then
                SERVER_ARGS="--model-repository=$DATADIR $DIFF_EXPLICIT_ADDRESS --metrics-port 8001"
                run_server_nowait
                sleep 15
                if [ "$SERVER_PID" == "0" ]; then
                    echo -e "\n***\n*** Failed to start $SERVER\n***"
                    cat $SERVER_LOG
                    exit 1
                fi
                code=`curl -s -w %{http_code} 127.0.0.1:8000/v2/health/ready`
                if [ "$code" != "200" ]; then
                    echo -e "\n***\n*** Server is not ready\n***"
                    RET=1
                fi
                kill $SERVER_PID
                wait $SERVER_PID
            else
                # http and metrics server bind to the same address, should skip this test case.
                true
            fi
        fi
    done
done

# Test multiple servers binding to the same http/grpc port
SERVER0_LOG="./inference_server0.log"
SERVER1_LOG="./inference_server1.log"
SERVER2_LOG="./inference_server2.log"

for p in http grpc; do
    # error if servers bind to the same http/grpc port without setting the reuse flag
    if [ "$p" == "http" ]; then
        SERVER_ARGS="--model-repository=$DATADIR --metrics-port 8002 --reuse-grpc-port=true"
        SERVER0_ARGS="--model-repository=$DATADIR --metrics-port 8003 --reuse-grpc-port=true"
        SERVER1_ARGS="--model-repository=$DATADIR --metrics-port 8004 --reuse-grpc-port=true"
    else
        SERVER_ARGS="--model-repository=$DATADIR --metrics-port 8002 --reuse-http-port=true"
        SERVER0_ARGS="--model-repository=$DATADIR --metrics-port 8003 --reuse-http-port=true"
        SERVER1_ARGS="--model-repository=$DATADIR --metrics-port 8004 --reuse-http-port=true"
    fi
    # make sure the first server is launched successfully, then run the other
    # two servers and expect them to fail
    run_server
    run_multiple_servers_nowait 2
    sleep 15
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start SERVER $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi
    if [ "$SERVER1_PID" != "0" ]; then
        set +e
        kill $SERVER0_PID
        wait $SERVER0_PID
        if [ "$?" == "0" ]; then
            echo -e "\n***\n*** unexpected start SERVER0 $SERVER\n***"
            cat $SERVER0_LOG
            exit 1
        fi
        set -e
    fi
    if [ "$SERVER1_PID" != "0" ]; then
        set +e
        kill $SERVER1_PID
        wait $SERVER1_PID
        if [ "$?" == "0" ]; then
            echo -e "\n***\n*** unexpected start SERVER1 $SERVER\n***"
            cat $SERVER1_LOG
            exit 1
        fi
        set -e
    fi
    kill_server

    # 1. Allow multiple servers bind to the same http/grpc port with setting the reuse flag
    # 2. Test different forms of setting --metrics-address and verify metrics are queryable
    #   (a) Test default metrics-address being same as http-address
    #   (b) Test setting metrics-address explicitly to 0.0.0.0
    #   (c) Test setting metrics-address explicitly to 127.0.0.1
    SERVER0_ARGS="--model-repository=$DATADIR --metrics-port 8002 --reuse-http-port=true --reuse-grpc-port=true"
    SERVER1_ARGS="--model-repository=$DATADIR --metrics-address 0.0.0.0 --metrics-port 8003 --reuse-http-port=true --reuse-grpc-port=true"
    SERVER2_ARGS="--model-repository=$DATADIR --metrics-address 127.0.0.2 --metrics-port 8004 --reuse-http-port=true --reuse-grpc-port=true"
    run_multiple_servers_nowait 3
    sleep 15
    if [ "$SERVER0_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start SERVER0 $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi
    if [ "$SERVER1_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start SERVER1 $SERVER\n***"
        cat $SERVER1_LOG
        exit 1
    fi
    if [ "$SERVER2_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start SERVER2 $SERVER\n***"
        cat $SERVER2_LOG
        exit 1
    fi

    set +e

    # test if requests are being distributed among three servers
    if [ "$p" == "http" ]; then
        CLIENT_PY=../clients/simple_http_infer_client.py
    else
        CLIENT_PY=../clients/simple_grpc_infer_client.py
    fi

    pids=()
    for i in {0..10}; do
        python3 $CLIENT_PY >> $CLIENT_LOG 2>&1 &
        pids+=" $!"
    done
    wait $pids || { echo -e "\n***\n*** Python ${p} Async Infer Test Failed\n***"; cat $CLIENT_LOG; RET=1; }

    set -e

    server0_request_count=`curl -s localhost:8002/metrics | awk '/nv_inference_request_success{/ {print $2}'`
    server1_request_count=`curl -s localhost:8003/metrics | awk '/nv_inference_request_success{/ {print $2}'`
    server2_request_count=`curl -s 127.0.0.2:8004/metrics | awk '/nv_inference_request_success{/ {print $2}'`
    if [ ${server0_request_count%.*} -eq 0 ] || \
       [ ${server1_request_count%.*} -eq 0 ] || \
       [ ${server2_request_count%.*} -eq 0 ]; then
        echo -e "\n***\n*** Failed: ${p} requests are not distributed among all servers.\n***"
        RET=1
    fi
    kill_servers
done

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test Failed\n***"
fi
exit $RET
