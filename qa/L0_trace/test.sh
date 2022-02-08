#!/bin/bash
# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

SIMPLE_HTTP_CLIENT=../clients/simple_http_infer_client
SIMPLE_GRPC_CLIENT=../clients/simple_grpc_infer_client
TRACE_SUMMARY=../common/trace_summary.py

CLIENT_TEST=trace_endpoint_test.py
CLIENT_LOG="client.log"
TEST_RESULT_FILE="test_results.txt"
EXPECTED_NUM_TESTS="6"

REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi
if [ ! -z "$TEST_REPO_ARCH" ]; then
    REPO_VERSION=${REPO_VERSION}_${TEST_REPO_ARCH}
fi

export CUDA_VISIBLE_DEVICES=0

DATADIR=/data/inferenceserver/${REPO_VERSION}/qa_model_repository
ENSEMBLEDIR=$DATADIR/../qa_ensemble_model_repository/qa_model_repository/
MODELBASE=onnx_int32_int32_int32

MODELSDIR=`pwd`/trace_models

SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

rm -f *.log
rm -fr $MODELSDIR && mkdir -p $MODELSDIR

# set up simple and global_simple model using MODELBASE
rm -fr $MODELSDIR && mkdir -p $MODELSDIR && \
    cp -r $DATADIR/$MODELBASE $MODELSDIR/simple && \
    rm -r $MODELSDIR/simple/2 && rm -r $MODELSDIR/simple/3 && \
    (cd $MODELSDIR/simple && \
            sed -i "s/^name:.*/name: \"simple\"/" config.pbtxt) && \
    cp -r $MODELSDIR/simple $MODELSDIR/global_simple && \
    (cd $MODELSDIR/global_simple && \
            sed -i "s/^name:.*/name: \"global_simple\"/" config.pbtxt) && \

RET=0

# start with trace-level=OFF
SERVER_ARGS="--trace-file=trace_off_to_min.log --trace-level=OFF --trace-rate=1 --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_off.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

for p in {1..10}; do
    $SIMPLE_HTTP_CLIENT >> client_off.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    $SIMPLE_GRPC_CLIENT >> client_off.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi
done

# Enable via trace API and send again
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -d'{"trace_level":["TIMESTAMPS"]}' localhost:8000/v2/trace/setting`
set -e
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
# Check if the current setting is returned
if [ `grep -c "\"trace_level\":\[\"TIMESTAMPS\"\]" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_rate\":\"1\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"log_frequency\":\"0\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_file\":\"trace_off_to_min.log\"" ./curl.out` != "1" ]; then
    RET=1
fi

for p in {1..10}; do
    $SIMPLE_HTTP_CLIENT >> client_min.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    $SIMPLE_GRPC_CLIENT >> client_min.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi
done

set -e

kill $SERVER_PID
wait $SERVER_PID

set +e

# Expect only the requests after calling trace API are traced 
$TRACE_SUMMARY -t trace_off_to_min.log > summary_off_to_min.log

if [ `grep -c "COMPUTE_INPUT_END" summary_off_to_min.log` != "20" ]; then
    cat summary_off_to_min.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

if [ `grep -c ^simple summary_off_to_min.log` != "20" ]; then
    cat summary_off_to_min.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

# Add model specific setting
SERVER_ARGS="--trace-file=global_trace.log --trace-level=TIMESTAMPS --trace-rate=6 --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_off.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

# Add trace setting for 'simple' via trace API, first use the same trace file
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -d'{"trace_file":"global_trace.log"}' localhost:8000/v2/models/simple/trace/setting`
set -e
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
# Check if the current setting is returned (not specified setting from global) 
if [ `grep -c "\"trace_level\":\[\"TIMESTAMPS\"\]" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_rate\":\"6\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"log_frequency\":\"0\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_file\":\"global_trace.log\"" ./curl.out` != "1" ]; then
    RET=1
fi

# Use a different name
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -d'{"trace_file":"simple_trace.log","log_frequency":"2"}' localhost:8000/v2/models/simple/trace/setting`
set -e
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

# Check if the current setting is returned (not specified setting from global) 
if [ `grep -c "\"trace_level\":\[\"TIMESTAMPS\"\]" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_rate\":\"6\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_count\":\"-1\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"log_frequency\":\"2\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_file\":\"simple_trace.log\"" ./curl.out` != "1" ]; then
    RET=1
fi

for p in {1..10}; do
    $SIMPLE_HTTP_CLIENT >> client_simple.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    $SIMPLE_GRPC_CLIENT >> client_simple.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi
done

set -e

kill $SERVER_PID
wait $SERVER_PID

set +e

if [ -f ./global_trace.log ]; then
    echo -e "\n***\n*** Test Failed, unexpected generation of global_trace.log\n***"
    RET=1
fi

$TRACE_SUMMARY -t simple_trace.log.0 > summary_simple_trace.log.0

if [ `grep -c "COMPUTE_INPUT_END" summary_simple_trace.log.0` != "2" ]; then
    cat summary_simple_trace.log.0
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

if [ `grep -c ^simple summary_simple_trace.log.0` != "2" ]; then
    cat summary_simple_trace.log.0
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

$TRACE_SUMMARY -t simple_trace.log.1 > summary_simple_trace.log.1

if [ `grep -c "COMPUTE_INPUT_END" summary_simple_trace.log.1` != "1" ]; then
    cat summary_simple_trace.log.1
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

if [ `grep -c ^simple summary_simple_trace.log.1` != "1" ]; then
    cat summary_simple_trace.log.1
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

# Update and clear model specific setting
SERVER_ARGS="--trace-file=global_trace.log --trace-level=TIMESTAMPS --trace-rate=6 --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_off.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

# Add model setting and update it
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -d'{"trace_file":"update_trace.log", "trace_rate":"1"}' localhost:8000/v2/models/simple/trace/setting`
set -e
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -d'{"trace_file":"update_trace.log", "trace_level":["OFF"]}' localhost:8000/v2/models/simple/trace/setting`
set -e
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

# Check if the current setting is returned
if [ `grep -c "\"trace_level\":\[\"OFF\"\]" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_rate\":\"1\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_count\":\"-1\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"log_frequency\":\"0\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_file\":\"update_trace.log\"" ./curl.out` != "1" ]; then
    RET=1
fi

# Send requests to simple where trace is explicitly disabled
for p in {1..10}; do
    $SIMPLE_HTTP_CLIENT >> client_update.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    $SIMPLE_GRPC_CLIENT >> client_update.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi
done

rm -f ./curl.out
set +e

# Clear trace setting by explicitly asking removal for every feild except 'trace_rate'
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -d'{"trace_file":null, "trace_level":null}' localhost:8000/v2/models/simple/trace/setting`
set -e
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

# Check if the current setting (global) is returned
if [ `grep -c "\"trace_level\":\[\"TIMESTAMPS\"\]" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_rate\":\"1\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_count\":\"-1\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"log_frequency\":\"0\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_file\":\"global_trace.log\"" ./curl.out` != "1" ]; then
    RET=1
fi

# Send requests to simple where now uses global setting
for p in {1..5}; do
    $SIMPLE_HTTP_CLIENT >> client_clear.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    $SIMPLE_GRPC_CLIENT >> client_clear.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi
done

set -e

kill $SERVER_PID
wait $SERVER_PID

set +e

if [ -f ./update_trace.log ]; then
    echo -e "\n***\n*** Test Failed, unexpected generation of update_trace.log\n***"
    RET=1
fi

$TRACE_SUMMARY -t global_trace.log > summary_global_trace.log

if [ `grep -c "COMPUTE_INPUT_END" summary_global_trace.log` != "10" ]; then
    cat summary_global_trace.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

if [ `grep -c ^simple summary_global_trace.log` != "10" ]; then
    cat summary_global_trace.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

# Update trace count
SERVER_ARGS="--trace-file=global_count.log --trace-level=TIMESTAMPS --trace-rate=1 --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_off.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

# Send requests without trace count
for p in {1..10}; do
    $SIMPLE_HTTP_CLIENT >> client_update.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    $SIMPLE_GRPC_CLIENT >> client_update.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi
done

set -e

# Check the current setting
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out localhost:8000/v2/models/simple/trace/setting`
set -e
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ `grep -c "\"trace_level\":\[\"TIMESTAMPS\"\]" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_rate\":\"1\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_count\":\"-1\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"log_frequency\":\"0\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_file\":\"global_count.log\"" ./curl.out` != "1" ]; then
    RET=1
fi

# Set trace count
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -d'{"trace_count":"5"}' localhost:8000/v2/trace/setting`
set -e
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

# Check if the current setting is returned
if [ `grep -c "\"trace_level\":\[\"TIMESTAMPS\"\]" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_rate\":\"1\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_count\":\"5\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"log_frequency\":\"0\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_file\":\"global_count.log\"" ./curl.out` != "1" ]; then
    RET=1
fi

# Send requests to simple where trace is explicitly disabled
for p in {1..10}; do
    $SIMPLE_HTTP_CLIENT >> client_update.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    $SIMPLE_GRPC_CLIENT >> client_update.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi
done

# Check the current setting agian and expect 'trace_count' becomes 0
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out localhost:8000/v2/models/simple/trace/setting`
set -e
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ `grep -c "\"trace_level\":\[\"TIMESTAMPS\"\]" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_rate\":\"1\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_count\":\"0\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"log_frequency\":\"0\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_file\":\"global_count.log\"" ./curl.out` != "1" ]; then
    RET=1
fi

# Check if the indexed file has been generated when trace count reaches 0
if [ -f ./global_trace.log.0 ]; then
    echo -e "\n***\n*** Test Failed, expect generation of global_trace.log.0 before stopping server\n***"
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

set +e

# There should be two trace files for trace counted requests and before trace
# counted requests
$TRACE_SUMMARY -t global_count.log > summary_global_count.log

if [ `grep -c "COMPUTE_INPUT_END" summary_global_count.log` != "20" ]; then
    cat summary_global_count.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

if [ `grep -c ^simple summary_global_count.log` != "20" ]; then
    cat summary_global_count.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

$TRACE_SUMMARY -t global_count.log.0 > summary_global_count.log.0

if [ `grep -c "COMPUTE_INPUT_END" summary_global_count.log.0` != "5" ]; then
    cat summary_global_count.log.0
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

if [ `grep -c ^simple summary_global_count.log.0` != "5" ]; then
    cat summary_global_count.log.0
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

# Test Python client library
SERVER_ARGS="--trace-file=global_unittest.log --trace-level=TIMESTAMPS --trace-rate=1 --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_unittest.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

RET=0

set +e

python $CLIENT_TEST >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    RET=1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi


exit $RET
