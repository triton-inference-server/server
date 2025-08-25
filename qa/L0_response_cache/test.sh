#!/bin/bash
# Copyright 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh
RET=0


############### Response Cache Memory Growth Test ###############

# Set server, client and valgrind arguments
LEAKCHECK=/usr/bin/valgrind
MASSIF_TEST=../common/check_massif_log.py
MODEL="identity_cache"
LEAKCHECK_LOG="${MODEL}.valgrind.log"
MASSIF_LOG="${MODEL}.valgrind.massif"
GRAPH_LOG="memory_growth_${MODEL}.log"
SERVER_LOG="${MODEL}.server.log"
CLIENT_LOG="${MODEL}_PA.client.log"
RANDOM_DATA_CLIENT_LOG="${MODEL}_random_data_script.log"
RANDOM_DATA_JSON="`pwd`/random_inputs.json"
RANDOM_DATA_GENERATOR="generate_random_data.py"

LEAKCHECK_ARGS="--tool=massif --time-unit=B --massif-out-file=$MASSIF_LOG --max-threads=3000 --log-file=$LEAKCHECK_LOG"
SERVER_ARGS="--model-repository=`pwd`/models --model-control-mode=explicit --load-model=${MODEL} --cache-config=local,size=10485760" # 10MB cache

set +e
# Generate random data for perf_analyzer requests to fill the cache and maximize cache misses
python "$RANDOM_DATA_GENERATOR" --num-inputs=10000 --batch-size=1 --output-file="${RANDOM_DATA_JSON}" >> "$RANDOM_DATA_CLIENT_LOG" 2>&1
if [ $? -ne 0 ]; then
    cat "$RANDOM_DATA_CLIENT_LOG"
    echo -e "\n***\n*** Failed to run ${RANDOM_DATA_GENERATOR}.\n***"
    RET=1
    exit 1
else
    # Check if the JSON data file was generated
    if [ ! -f "${RANDOM_DATA_JSON}" ]; then
        echo -e "\n***\n*** FAILED - JSON data file was not found at the expected path: ${RANDOM_DATA_JSON}\n***"
        RET=1
        exit 1
    fi
fi
set -e

# Run the server
run_server_leakcheck
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi


TEMP_RET=0
REPETITION=10
CONCURRENCY=20
CLIENT_BS=1
PERF_ANALYZER=../clients/perf_analyzer
TEMP_CLIENT_LOG=temp_client.log

set +e
SECONDS=0
# Run the perf analyzer 'REPETITION' times
for ((i=1; i<=$REPETITION; i++)); do
    # Use random data to ensure cache misses
    $PERF_ANALYZER -v -m $MODEL --shape=INPUT0:1024 -i grpc --concurrency-range $CONCURRENCY -b $CLIENT_BS -p 20000 --input-data="${RANDOM_DATA_JSON}" > $TEMP_CLIENT_LOG 2>&1
    PA_RET=$?
    cat $TEMP_CLIENT_LOG >> $CLIENT_LOG
    # Success
    if [ ${PA_RET} -eq 0 ]; then
      continue
    # Unstable measurement: OK for this test
    elif [ ${PA_RET} -eq 2 ]; then
      continue
    # Other failures unexpected, report error
    else
        echo -e "\n***\n*** perf_analyzer for $MODEL failed on iteration $i\n***" >> $CLIENT_LOG
        RET=1
    fi
done
TEST_DURATION=$SECONDS
set -e

# Stop Server
kill $SERVER_PID
wait $SERVER_PID

set +e

# Log test duration and the graph for memory growth
MAX_ALLOWED_ALLOC=2 # MB
hrs=$(printf "%02d" $((TEST_DURATION / 3600)))
mins=$(printf "%02d" $(((TEST_DURATION / 60) % 60)))
secs=$(printf "%02d" $((TEST_DURATION % 60)))
echo -e "Test Duration: $hrs:$mins:$secs (HH:MM:SS)" >> ${GRAPH_LOG}
ms_print ${MASSIF_LOG} | head -n35 >> ${GRAPH_LOG}
cat ${GRAPH_LOG}
# Check the massif output
python $MASSIF_TEST $MASSIF_LOG $MAX_ALLOWED_ALLOC --start-from-middle >> $GRAPH_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Memory growth test for $MODEL Failed.\n***"
    RET=1
fi
# Always output memory usage for easier triage of MAX_ALLOWED_ALLOC settings in the future
grep -i "Change in memory allocation" "${GRAPH_LOG}" || true
set -e

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
