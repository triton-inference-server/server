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

MULTI_PORT_TESTS_PY=multi_port_tests.py

CLIENT_LOG="./client.log"
SERVER_LOG="./inference_server.log"

DATADIR=`pwd`/models
SERVER=/opt/tensorrtserver/bin/trtserver
source ../common/util.sh

HP_ARR=(8000 8005 -1)
len=${#HP_ARR[@]}
rm -f $CLIENT_LOG $SERVER_LOG

RET=0
# HTTP Health w/o Main Port
for (( n=0; n<$len; n++ )) ; do
  SERVER_ARGS_ADD_HTTP="--http-health-port ${HP_ARR[n]} --allow-http 1"
  SERVER_ARGS="--model-repository=$DATADIR $SERVER_ARGS_ADD_HTTP"

  run_server_nowait
  sleep 5
  if [ "$SERVER_PID" == "0" ]; then
      echo -e "\n***\n*** Failed to start $SERVER\n***"
      cat $SERVER_LOG
      exit 1
  fi

  set +e
  python $MULTI_PORT_TESTS_PY -v >>$CLIENT_LOG 2>&1 -hp ${HP_ARR[n]} -i http
  if [ $? -ne 0 ]; then
      RET=1
  fi
  set -e

  kill $SERVER_PID
  wait $SERVER_PID
done

# HTTP main port + Health
P=(8004 -1)
for (( i=0; i<2; i++ )) ; do
  for (( n=0; n<$len; n++ )) ; do
    SERVER_ARGS_ADD_HTTP="--http-port ${P[i]} --http-health-port ${HP_ARR[n]} \
      --allow-http 1"
    SERVER_ARGS="--model-repository=$DATADIR $SERVER_ARGS_ADD_HTTP"

    set +e
    run_server_nowait
    sleep 5
    set -e
    if ! [[ "${P[i]}" == "-1" && "${HP_ARR[n]}" == "-1" ]] ; then
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        else
          set +e
          python $MULTI_PORT_TESTS_PY -v >>$CLIENT_LOG 2>&1 -p ${P[i]} -hp ${HP_ARR[n]} -i http
          if [ $? -ne 0 ]; then
              RET=1
          fi
          set -e

          kill $SERVER_PID
          wait $SERVER_PID
        fi
    fi
  done
done

# CUSTOM CASES

# set http port to -1 after setting health to 8007
SERVER_ARGS_ADD_HTTP="--http-health-port 8007 --http-port -1 --allow-http 1"
SERVER_ARGS="--model-repository=$DATADIR $SERVER_ARGS_ADD_HTTP"
set +e
run_server_nowait
sleep 5
if kill $SERVER_PID && wait $SERVER_PID ; then
    echo -e "\n***\n*** Should not have started $SERVER\n***"
    RET=1
    cat $SERVER_LOG
fi
set -e
# allow overrules - grpc still works
SERVER_ARGS_ADD_HTTP="--http-port -1 --http-health-port 8007 --allow-http 0"
SERVER_ARGS="--model-repository=$DATADIR $SERVER_ARGS_ADD_HTTP"
run_server_nowait
sleep 5
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
kill $SERVER_PID
wait $SERVER_PID
# overlap with grpc default
SERVER_ARGS_ADD_HTTP="--http-health-port 8001"
SERVER_ARGS="--model-repository=$DATADIR $SERVER_ARGS_ADD_HTTP"
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
SERVER_ARGS_ADD_HTTP="--http-status-port 8002 --http-health-port 8007 \
  --http-profile-port 8007 --http-infer-port 8007"
SERVER_ARGS="--model-repository=$DATADIR $SERVER_ARGS_ADD_HTTP"
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
SERVER_ARGS_ADD_HTTP="--http-health-port 8002 --allow-metrics 0"
SERVER_ARGS="--model-repository=$DATADIR $SERVER_ARGS_ADD_HTTP"
run_server_nowait
sleep 5
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
set +e
python $MULTI_PORT_TESTS_PY -v >>$CLIENT_LOG 2>&1 -hp 8002
if [ $? -ne 0 ]; then
    RET=1
fi
set -e
kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test Failed\n***"
fi
exit $RET
