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

# MULTI_PORT_CLIENT=../clients/multi_port_client
MULTI_PORT_CLIENT_PY=../clients/multi_port_client.py

CLIENT_LOG="./client.log"

DATADIR=`pwd`/models
HSP_ARR=(8008 8009 8010 8011)
HHP_ARR=(8008 8010 8010 8009)
HPP_ARR=(8008 8011 8011 8010)
HIP_ARR=(8008 8011 8008 8008)
SERVER=/opt/tensorrtserver/bin/trtserver
for (( n=0; n<4; n++ ))
do
:
# do whatever on $i
  SERVER_ARGS_ADD_HTTP="--http-status-port ${HSP_ARR[n]} --http-health-port ${HHP_ARR[n]} --http-profile-port ${HPP_ARR[n]}\
    --http-infer-port ${HIP_ARR[n]} --http-port -1"
  SERVER_ARGS_ADD_GRPC="--grpc-status-port $((${HSP_ARR[n]}+10)) --grpc-health-port $((${HHP_ARR[n]}+10)) \
    --grpc-profile-port $((${HPP_ARR[n]}+10)) --grpc-infer-port $((${HIP_ARR[n]}+10)) --grpc-port -1"
  SERVER_ARGS="--model-store=$DATADIR $SERVER_ARGS_ADD_GRPC $SERVER_ARGS_ADD_HTTP"
  SERVER_LOG="./inference_server.log"
  SERVER_TIMEOUT=20
  source ../common/util.sh

  rm -f $CLIENT_LOG $SERVER_LOG

  run_server
  if [ "$SERVER_PID" == "0" ]; then
      echo -e "\n***\n*** Failed to start $SERVER\n***"
      cat $SERVER_LOG
      exit 1
  fi

  RET=0

  # $MULTI_PORT_CLIENT -v >>$CLIENT_LOG 2>&1 -hsp $ ${HSP_ARR[n]} -hhp ${HHP_ARR[n]} -hpp ${HPP_ARR[n]} -hip ${HIP_ARR[n]}
  # if [ $? -ne 0 ]; then
  #     RET=1
  # fi

  python $MULTI_PORT_CLIENT_PY -v >>$CLIENT_LOG 2>&1 -hsp $ ${HSP_ARR[n]} -hhp ${HHP_ARR[n]} -hpp ${HPP_ARR[n]} -hip ${HIP_ARR[n]}
  if [ $? -ne 0 ]; then
      RET=1
  fi

  python $MULTI_PORT_CLIENT_PY -v >>$CLIENT_LOG 2>&1 -hsp $ ${HSP_ARR[n]} -hhp ${HHP_ARR[n]} -hpp ${HPP_ARR[n]} -hip ${HIP_ARR[n]} -i grpc
  if [ $? -ne 0 ]; then
      RET=1
  fi

  kill $SERVER_PID
  wait $SERVER_PID

  if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test $n PASSED\n***"
  else
      cat $CLIENT_LOG
      echo -e "\n***\n*** Test $n FAILED\n***"
  fi
done

exit $RET
