#!/bin/bash
# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
BLSDIR=../python_models/bls_simple
MODELBASE=onnx_int32_int32_int32

MODELSDIR=`pwd`/trace_models

SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

rm -f *.log
rm -fr $MODELSDIR && mkdir -p $MODELSDIR

# set up simple and global_simple model using MODELBASE
cp -r $DATADIR/$MODELBASE $MODELSDIR/simple && \
    rm -r $MODELSDIR/simple/2 && rm -r $MODELSDIR/simple/3 && \
    (cd $MODELSDIR/simple && \
            sed -i "s/^name:.*/name: \"simple\"/" config.pbtxt) && \
    cp -r $MODELSDIR/simple $MODELSDIR/global_simple && \
    (cd $MODELSDIR/global_simple && \
            sed -i "s/^name:.*/name: \"global_simple\"/" config.pbtxt) && \
    cp -r $ENSEMBLEDIR/simple_onnx_int32_int32_int32 $MODELSDIR/ensemble_add_sub_int32_int32_int32 && \
    rm -r $MODELSDIR/ensemble_add_sub_int32_int32_int32/2 && \
    rm -r $MODELSDIR/ensemble_add_sub_int32_int32_int32/3 && \
    (cd $MODELSDIR/ensemble_add_sub_int32_int32_int32 && \
            sed -i "s/^name:.*/name: \"ensemble_add_sub_int32_int32_int32\"/" config.pbtxt && \
            sed -i "s/model_name:.*/model_name: \"simple\"/" config.pbtxt) && \
    mkdir -p $MODELSDIR/bls_simple/1 && cp $BLSDIR/bls_simple.py $MODELSDIR/bls_simple/1/model.py

RET=0

# Helpers =======================================
function assert_curl_success {
  message="${1}"
  if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** ${message} : line ${BASH_LINENO}\n***"
    RET=1
  fi
}

function assert_curl_failure {
  message="${1}"
  if [ "$code" == "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** ${message} : line ${BASH_LINENO}\n***"
    RET=1
  fi
}

function get_global_trace_setting {
  rm -f ./curl.out
  set +e
  code=`curl -s -w %{http_code} -o ./curl.out localhost:8000/v2/trace/setting`
  set -e
}

function get_trace_setting {
  model_name="${1}"
  rm -f ./curl.out
  set +e
  code=`curl -s -w %{http_code} -o ./curl.out localhost:8000/v2/models/${model_name}/trace/setting`
  set -e
}

function update_global_trace_setting {
  settings="${1}"
  rm -f ./curl.out
  set +e
  code=`curl -s -w %{http_code} -o ./curl.out -X POST localhost:8000/v2/trace/setting -d ${settings}`
  set -e
}

function update_trace_setting {
  model_name="${1}"
  settings="${2}"
  rm -f ./curl.out
  set +e
  code=`curl -s -w %{http_code} -o ./curl.out -X POST localhost:8000/v2/models/${model_name}/trace/setting -d ${settings}`
  set -e
}

function send_inference_requests {
    log_file="${1}"
    upper_bound="${2}"
    for (( p = 1; p <= $upper_bound; p++ )) do
        $SIMPLE_HTTP_CLIENT >> ${log_file} 2>&1
        if [ $? -ne 0 ]; then
            RET=1
        fi

        $SIMPLE_GRPC_CLIENT >> ${log_file} 2>&1
        if [ $? -ne 0 ]; then
            RET=1
        fi
    done
}

#=======================================

# start with trace-level=OFF
SERVER_ARGS="--trace-config triton,file=trace_off_to_min.log --trace-config level=OFF --trace-config rate=1 --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_off.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

# Enable via trace API and send again
update_global_trace_setting '{"trace_level":["TIMESTAMPS"]}'
assert_curl_success "Failed to modify global trace settings"

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

send_inference_requests "client_min.log" 10

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
SERVER_ARGS="--trace-config triton,file=global_trace.log --trace-config level=TIMESTAMPS --trace-config rate=6 --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_off.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

# Add trace setting for 'simple' via trace API, first use the same trace file
update_trace_setting "simple" '{"trace_file":"global_trace.log"}'
assert_curl_success "Failed to modify trace settings for 'simple' model"

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
update_trace_setting "simple" '{"trace_file":"simple_trace.log","log_frequency":"2"}'
assert_curl_success "Failed to modify trace settings for 'simple' model"

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

send_inference_requests "client_simple.log" 10

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
SERVER_ARGS="--trace-config triton,file=global_trace.log --trace-config level=TIMESTAMPS --trace-config rate=6 --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_off.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

# Add model setting and update it
update_trace_setting "simple" '{"trace_file":"update_trace.log","trace_rate":"1"}'
assert_curl_success "Failed to modify trace settings for 'simple' model"

update_trace_setting "simple" '{"trace_file":"update_trace.log","trace_level":["OFF"]}'
assert_curl_success "Failed to modify trace settings for 'simple' model"

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
send_inference_requests "client_update.log" 10

rm -f ./curl.out
set +e

# Clear trace setting by explicitly asking removal for every field except 'trace_rate'
update_trace_setting "simple" '{"trace_file":null,"trace_level":null}'
assert_curl_success "Failed to modify trace settings for 'simple' model"

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
send_inference_requests "client_clear.log" 5

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
SERVER_ARGS="--trace-config triton,file=global_count.log --trace-config level=TIMESTAMPS --trace-config rate=1 --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_off.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

# Send requests without trace count
send_inference_requests "client_update.log" 10

set -e

# Check the current setting
get_trace_setting "simple"
assert_curl_success "Failed to obtain trace settings for 'simple' model"

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
update_global_trace_setting '{"trace_count":"5"}'
assert_curl_success "Failed to modify global trace settings"

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
send_inference_requests "client_update.log" 10

# Check the current setting again and expect 'trace_count' becomes 0
get_trace_setting "simple"
assert_curl_success "Failed to obtain trace settings for 'simple' model"

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

SETTINGS="trace_count trace_rate log_frequency"

for SETTING in $SETTINGS; do
    # Check `out of range` errors
    update_trace_setting "simple" '{"'${SETTING}'":"10000000000"}'
    assert_curl_failure "Server modified '${SETTING}' with an out of range value."
done

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
SERVER_ARGS="--trace-config triton,file=global_unittest.log --trace-config level=TIMESTAMPS --trace-config rate=1 --model-repository=$MODELSDIR"
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


# Check `--trace-config` sets arguments properly
SERVER_ARGS="--trace-config=triton,file=bls_trace.log --trace-config=level=TIMESTAMPS \
            --trace-config=rate=4 --trace-config=count=6 --trace-config=mode=triton --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_trace_config.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

get_trace_setting "simple"
assert_curl_success "Failed to obtain trace settings for 'simple' model"

if [ `grep -c "\"trace_level\":\[\"TIMESTAMPS\"\]" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_rate\":\"4\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_count\":\"6\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"log_frequency\":\"0\"" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\"trace_file\":\"bls_trace.log\"" ./curl.out` != "1" ]; then
    RET=1
fi

set +e
# Send bls requests to make sure simple model is traced
for p in {1..4}; do
    python -c 'import opentelemetry_unittest; \
        opentelemetry_unittest.send_bls_request(model_name="ensemble_add_sub_int32_int32_int32")'  >> client_update.log 2>&1
done

set -e

kill $SERVER_PID
wait $SERVER_PID

set +e

$TRACE_SUMMARY -t bls_trace.log > summary_bls.log

if [ `grep -c "COMPUTE_INPUT_END" summary_bls.log` != "2" ]; then
    cat summary_bls.log
    echo -e "\n***\n*** Test Failed: Unexpected number of traced "COMPUTE_INPUT_END" events.\n***"
    RET=1
fi

if [ `grep -c ^ensemble_add_sub_int32_int32_int32 summary_bls.log` != "1" ]; then
    cat summary_bls.log
    echo -e "\n***\n*** Test Failed: BLS child ensemble model wasn't traced. \n***"
    RET=1
fi

if [ `grep -c ^simple summary_bls.log` != "1" ]; then
    cat summary_bls.log
    echo -e "\n***\n*** Test Failed: ensemble's model 'simple' wasn't traced. \n***"
    RET=1
fi

if [ `grep -o 'parent_id' bls_trace.log | wc -l` != "2" ]; then
    cat bls_trace.log
    echo -e "\n***\n*** Test Failed: Unexpected number of 'parent id' fields. \n***"
    RET=1
fi

# Attempt to trace non-existent model
SERVER_ARGS="--model-control-mode=explicit --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_nonexistent_model.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# Explicitly load model
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -X POST localhost:8000/v2/repository/models/simple/load`
set -e
assert_curl_success "Failed to load 'simple' model"

# Non-existent model (get)
get_trace_setting "does-not-exist"
assert_curl_failure "Server returned trace settings for a non-existent model"

# Non-existent model (post)
update_trace_setting "does-not-exist" '{"log_frequency":"1"}'
assert_curl_failure "Server modified trace settings for a non-existent model"

# Local model (get)
get_trace_setting "simple"
assert_curl_success "Failed to obtain trace settings for 'simple' model"

# Local model (post)
update_trace_setting "simple" '{"log_frequency":"1"}'
assert_curl_success "Failed to modify trace settings for 'simple' model"

# Local model (unload)
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -X POST localhost:8000/v2/repository/models/simple/unload`
set -e
assert_curl_success "Failed to unload 'simple' model"

get_trace_setting "simple"
assert_curl_failure "Server returned trace settings for an unloaded model"

update_trace_setting "simple" '{"log_frequency":"1"}'
assert_curl_failure "Server modified trace settings for an unloaded model"

# Local model (reload)
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -X POST localhost:8000/v2/repository/models/simple/load`
set -e
assert_curl_success "Failed to load 'simple' model"

get_trace_setting "simple"
assert_curl_success "Failed to obtain trace settings for 'simple' model"

update_trace_setting "simple" '{"log_frequency":"1"}'
assert_curl_success "Failed to modify trace settings for 'simple' model"

kill $SERVER_PID
wait $SERVER_PID

set +e

# Check opentelemetry trace exporter sends proper info.
# A helper python script starts listening on $OTLP_PORT, where
# OTLP exporter sends traces.
export TRITON_OPENTELEMETRY_TEST='false'
OTLP_PORT=10000
OTEL_COLLECTOR_DIR=./opentelemetry-collector
OTEL_COLLECTOR=./opentelemetry-collector/bin/otelcorecol_*
OTEL_COLLECTOR_LOG="./trace_collector_http_exporter.log"

# Building the latest version of the OpenTelemetry collector.
# Ref: https://opentelemetry.io/docs/collector/getting-started/#local
if [ -d "$OTEL_COLLECTOR_DIR" ]; then rm -Rf $OTEL_COLLECTOR_DIR; fi
git clone --depth 1 --branch v0.82.0 https://github.com/open-telemetry/opentelemetry-collector.git
cd $OTEL_COLLECTOR_DIR
make install-tools
make otelcorecol
cd ..
$OTEL_COLLECTOR --config ./trace-config.yaml >> $OTEL_COLLECTOR_LOG 2>&1 & COLLECTOR_PID=$!


SERVER_ARGS="--trace-config=level=TIMESTAMPS --trace-config=rate=1 \
                --trace-config=count=100 --trace-config=mode=opentelemetry \
                --trace-config=opentelemetry,url=localhost:$OTLP_PORT/v1/traces \
                --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_otel_http_exporter.log"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

$SIMPLE_HTTP_CLIENT >>$CLIENT_LOG 2>&1

set -e

kill $SERVER_PID
wait $SERVER_PID

kill $COLLECTOR_PID
wait $COLLECTOR_PID

set +e

if ! [[ -s $OTEL_COLLECTOR_LOG && `grep -c 'InstrumentationScope triton-server' $OTEL_COLLECTOR_LOG` == 3 ]] ; then
    echo -e "\n***\n*** HTTP exporter test failed.\n***"
    cat $OTEL_COLLECTOR_LOG
    exit 1
fi


# Unittests then check that produced spans have expected format and events
OPENTELEMETRY_TEST=opentelemetry_unittest.py
OPENTELEMETRY_LOG="opentelemetry_unittest.log"
EXPECTED_NUM_TESTS="3"

export TRITON_OPENTELEMETRY_TEST='true'

SERVER_ARGS="--trace-config=level=TIMESTAMPS --trace-config=rate=1 \
                --trace-config=count=100 --trace-config=mode=opentelemetry \
                --trace-config=opentelemetry,resource=test.key=test.value \
                --trace-config=opentelemetry,resource=service.name=test_triton \
                --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_otel_ostream_exporter.log"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
# Preparing traces for unittest.
# Note: running this separately, so that I could extract spans with `grep`
# from server log later.
python -c 'import opentelemetry_unittest; \
        opentelemetry_unittest.prepare_traces()' >>$CLIENT_LOG 2>&1

sleep 5

set -e

kill $SERVER_PID
wait $SERVER_PID

set +e

grep -z -o -P '({\n(?s).*}\n)' $SERVER_LOG >> trace_collector.log

if ! [ -s trace_collector.log ] ; then
    echo -e "\n***\n*** $SERVER_LOG did not contain any OpenTelemetry spans.\n***"
    exit 1
fi

# Unittest will not start until expected number of spans is collected.
python $OPENTELEMETRY_TEST >>$OPENTELEMETRY_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $OPENTELEMETRY_LOG
    RET=1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $OPENTELEMETRY_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi

exit $RET
