#!/bin/bash
# Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
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

source ../common/util.sh

TEST_LOG="./memory_test.log"
MEMORY_TEST=./memory_test
PINNED_MEMORY_MANAGER_TEST=./pinned_memory_manager_test

RET=0

# Must run on multiple devices
export CUDA_VISIBLE_DEVICES=0,1

rm -f TEST_LOG

set +e
$MEMORY_TEST >>$TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $TEST_LOG
    echo -e "\n***\n*** Memory Test Failed\n***"
    RET=1
fi
set -e

set +e
$PINNED_MEMORY_MANAGER_TEST >>$TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $TEST_LOG
    echo -e "\n***\n*** Pinned Memory Manager Test Failed\n***"
    RET=1
fi
set -e


###### Test --grpc-max-response-pool-size server option #######

monitor_memory() {
  local SERVER_PID=$1
  local MAX_MEM_FILE=$(mktemp)
  echo "0" > "$MAX_MEM_FILE"
  (
    local MAX_MEM=0
    while ps -p "$SERVER_PID" >/dev/null 2>&1; do
      CURRENT_MEM=$(awk '/Rss:/ {print $2}' /proc/$SERVER_PID/smaps_rollup)
      CURRENT_MEM=${CURRENT_MEM:-0}
      if [ "$CURRENT_MEM" -gt "$MAX_MEM" ]; then
        MAX_MEM=$CURRENT_MEM
        echo "$MAX_MEM" > "$MAX_MEM_FILE"
      fi
      sleep 0.1
    done
    echo "$MAX_MEM" > "$MAX_MEM_FILE"
    exit 0
  ) &

  MONITOR_PID=$!
  echo "$MONITOR_PID $MAX_MEM_FILE"
}

stop_server_and_monitoring_memory() {
  local MONITOR_PID=$1
  local SERVER_PID=$2
  kill "$MONITOR_PID" 2>/dev/null && wait "$MONITOR_PID" 2>/dev/null || true
  kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null || true
}

MODELDIR="./python_models"
export OUTPUT_NUM_ELEMENTS=49807360
sed -i '$a\parameters: [{ key: "output_num_elements" value: { string_value: "'"$OUTPUT_NUM_ELEMENTS"'" }}]' $MODELDIR/repeat_int32/config.pbtxt

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_BASE_ARGS="--model-repository=${MODELDIR} --log-verbose=2 --allow-metrics=0"

declare -A MEMORY_USAGE=()

for POOL_SIZE in 1 25 50 default; do
  if [[ "$POOL_SIZE" = "default" ]]; then
    SERVER_ARGS="${SERVER_BASE_ARGS}"
  else
    SERVER_ARGS="${SERVER_BASE_ARGS} --grpc-max-response-pool-size=${POOL_SIZE}"
  fi

  CLIENT_LOG="./client_pool_size_${POOL_SIZE}.log"
  SERVER_LOG="./server_pool_size_${POOL_SIZE}.log"

  run_server
  if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    stop_server_and_monitoring_memory $MONITOR_PID $SERVER_PID
    exit 1
  fi
  sleep 2

  # Capture initial memory usage
  INIT_MEM=$(awk '/Rss:/ {print $2}' /proc/$SERVER_PID/smaps_rollup)
  read -r MONITOR_PID MAX_MEM_FILE < <(monitor_memory "$SERVER_PID")

  # Run client script
  set +e
  python3 client.py >> $CLIENT_LOG 2>&1
  if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Running client for grpc-max-response-pool-size=${POOL_SIZE} FAILED\n***" >> $CLIENT_LOG 2>&1
    echo -e "\n***\n*** Running client for grpc-max-response-pool-size=${POOL_SIZE} FAILED\n***"
    stop_server_and_monitoring_memory $MONITOR_PID $SERVER_PID
    exit 1
  fi
  set -e
  sleep 2

  stop_server_and_monitoring_memory $MONITOR_PID $SERVER_PID

  if [[ -s "$MAX_MEM_FILE" ]]; then
    MAX_MEM=$(tail -n 1 "$MAX_MEM_FILE" 2>/dev/null || echo 0)
    MEMORY_USAGE["$POOL_SIZE"]=$((MAX_MEM - INIT_MEM))
    echo "Pool size: $POOL_SIZE | Initial Memory: ${INIT_MEM} KB | Peak Memory: ${MEMORY_USAGE[$POOL_SIZE]} KB" >> "memory.log"
    rm -f "$MAX_MEM_FILE"
  else
    echo "FAILED to collect memory usage for grpc-max-response-pool-size=${POOL_SIZE}"
    exit 1
  fi
done

prev_mem=0
prev_size=""
for size in default 50 25 1; do
  current_mem=${MEMORY_USAGE[$size]}
  if [[ -n "$prev_size" && "$prev_mem" -ne 0 && "$current_mem" -ge "$prev_mem" ]]; then
    echo -e "\n***\n*** FAILED - Memory $current_mem KB with pool=$size >= $prev_mem KB (with pool=$prev_size)\n***"
    RET=1
  fi
  prev_mem=$current_mem
  prev_size=$size
done


if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
