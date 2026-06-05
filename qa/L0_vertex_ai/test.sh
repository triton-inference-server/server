#!/bin/bash
# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TEST_RESULT_FILE='test_results.txt'

export CUDA_VISIBLE_DEVICES=0

RET=0

rm -rf multi_models single_model
rm -f *.log
rm -f *.out

CLIENT_TEST_SCRIPT=vertex_ai_test.py
UNIT_TEST_COUNT=8
CLIENT_LOG="./client.log"

DATADIR=/data/inferenceserver/${REPO_VERSION}
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_LOG="./server.log"
source ../common/util.sh

# Set up the multi model repository with the swap and non-swap versions
mkdir multi_models && \
    cp -r $DATADIR/qa_model_repository/onnx_int32_int32_int32 multi_models/addsub && \
    rm -r multi_models/addsub/2 && rm -r multi_models/addsub/3 && \
    sed -i "s/onnx_int32_int32_int32/addsub/" multi_models/addsub/config.pbtxt && \
    cp -r $DATADIR/qa_model_repository/onnx_int32_int32_int32 multi_models/subadd && \
    rm -r multi_models/subadd/1 && rm -r multi_models/subadd/2 && \
    sed -i "s/onnx_int32_int32_int32/subadd/" multi_models/subadd/config.pbtxt
mkdir single_model && \
    cp -r multi_models/addsub single_model/.

# Use Vertex AI's health endpoint to check server status
# Wait until server health endpoint shows ready. Sets WAIT_RET to 0 on
# success, 1 on failure
function vertex_ai_wait_for_server_ready() {
    local spid="$1"; shift
    local wait_time_secs="${1:-30}"; shift

    WAIT_RET=0

    ping_address="localhost:8080${AIP_HEALTH_ROUTE}"
    if [ -n "$AIP_HTTP_PORT" ]; then
        ping_address="localhost:${AIP_HTTP_PORT}${AIP_HEALTH_ROUTE}"
    fi

    local wait_secs=$wait_time_secs
    until test $wait_secs -eq 0 ; do
        if ! kill -0 $spid; then
            echo "=== Server not running."
            WAIT_RET=1
            return
        fi

        sleep 1;

        set +e
        code=`curl -s -w %{http_code} $ping_address`
        set -e
        if [ "$code" == "200" ]; then
            return
        fi

        ((wait_secs--));
    done

    echo "=== Timeout $wait_time_secs secs. Server not ready."
    WAIT_RET=1
}

# Helper function to unset all AIP variables before test
function unset_vertex_variables() {
    unset AIP_MODE
    unset AIP_HTTP_PORT
    unset AIP_HEALTH_ROUTE
    unset AIP_PREDICT_ROUTE
    unset AIP_STORAGE_URI
}

#
# Test default allow-vertex-ai
#
unset_vertex_variables

# Enable HTTP endpoint to check server readiness in the case of disabling Vertex AI
BASE_SERVER_ARGS="--allow-http true --model-repository=single_model"
export AIP_HEALTH_ROUTE="/health"
export AIP_PREDICT_ROUTE="/predict"

# Default false
SERVER_ARGS=${BASE_SERVER_ARGS}
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
kill $SERVER_PID
wait $SERVE_PID
set +e
# Expect no message regarding Vertex AI as it is disabled
grep "failed to start Vertex AI service" $SERVER_LOG
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** Failed. Expected Vertex AI service is disabled\n***"
    RET=1
fi
grep "Started Vertex AI HTTPService at" $SERVER_LOG
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** Failed. Expected Vertex AI service is disabled\n***"
    RET=1
fi
set -e
# Enable
SERVER_ARGS="${BASE_SERVER_ARGS} --allow-vertex-ai=true"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
kill $SERVER_PID
wait $SERVE_PID
set +e
grep "Started Vertex AI HTTPService at" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected Vertex AI service is enabled\n***"
    RET=1
fi
set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

# Default true
# Note that when default true, HTTP / GRPC endpoints will be disabled,
# check those endpoints by enabling one of them at a time and greping keywords
export AIP_MODE=PREDICTION
SERVER_ARGS="--model-repository=single_model --allow-grpc=true"
# Using nowait as 'run_server' requires HTTP endpoint enabled
run_server_nowait
sleep 10
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
kill $SERVER_PID
wait $SERVE_PID
set +e
grep "Started Vertex AI HTTPService at" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected Vertex AI service is enabled\n***"
    RET=1
fi
grep "Started GRPCInferenceService at" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected GRPC service is enabled\n***"
    RET=1
fi
# Expect no message regarding HTTP as it is disabled
grep "failed to start HTTP service" $SERVER_LOG
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** Failed. Expected HTTP service is disabled\n***"
    RET=1
fi
grep "Started HTTPService at" $SERVER_LOG
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** Failed. Expected HTTP service is disabled\n***"
    RET=1
fi
set -e

# Disable
SERVER_ARGS="${BASE_SERVER_ARGS} --allow-vertex-ai=false --allow-http=true"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
kill $SERVER_PID
wait $SERVE_PID
set +e
# Expect no message regarding Vertex AI as it is disabled
grep "failed to start Vertex AI service" $SERVER_LOG
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** Failed. Expected Vertex AI service is disabled\n***"
    RET=1
fi
grep "Started Vertex AI HTTPService at" $SERVER_LOG
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** Failed. Expected Vertex AI service is disabled\n***"
    RET=1
fi
grep "Started HTTPService at" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected HTTP service is enabled\n***"
    RET=1
fi
# Expect no message regarding GRPC as it is disabled
grep "failed to start GRPC service" $SERVER_LOG
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** Failed. Expected GRPC service is disabled\n***"
    RET=1
fi
grep "Started GRPCInferenceService at" $SERVER_LOG
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** Failed. Expected GRPC service is disabled\n***"
    RET=1
fi
set -e

#
# Test missing route
#
unset_vertex_variables
export AIP_HEALTH_ROUTE="/health"

SERVER_ARGS="--allow-vertex-ai=true --model-repository=single_model"
run_server_nowait
vertex_ai_wait_for_server_ready $SERVER_PID 10
if [ "$WAIT_RET" == "0" ]; then
    echo -e "\n***\n*** Expect failed to start $SERVER\n***"
    kill $SERVER_PID || true
    cat $SERVER_LOG
    RET=1
else
  set +e
  grep "API_PREDICT_ROUTE is not defined for Vertex AI endpoint" $SERVER_LOG
  set -e
  if [ $? -ne 0 ]; then
      echo -e "\n***\n*** Failed. Expected error on using undefined route\n***"
      RET=1
  fi
fi

unset_vertex_variables
export AIP_PREDICT_ROUTE="/predict"
run_server_nowait
vertex_ai_wait_for_server_ready $SERVER_PID 10
if [ "$WAIT_RET" == "0" ]; then
    echo -e "\n***\n*** Expect failed to start $SERVER\n***"
    kill $SERVER_PID || true
    cat $SERVER_LOG
    RET=1
else
  set +e
  grep "AIP_HEALTH_ROUTE is not defined for Vertex AI endpoint" $SERVER_LOG
  set -e
  if [ $? -ne 0 ]; then
      echo -e "\n***\n*** Failed. Expected error on using undefined route\n***"
      RET=1
  fi
fi

#
# Test endpoints
#
unset_vertex_variables
export AIP_PREDICT_ROUTE="/predict"
export AIP_HEALTH_ROUTE="/health"

SERVER_ARGS="--allow-vertex-ai=true --model-repository=single_model"
run_server_nowait
# health
vertex_ai_wait_for_server_ready $SERVER_PID 10
if [ "$WAIT_RET" != "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    kill $SERVER_PID
    wait $SERVER_PID
    cat $SERVER_LOG
    exit 1
fi

# predict (single model)
set +e
python $CLIENT_TEST_SCRIPT >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    cat $CLIENT_LOG
    RET=1
else
    check_test_results $TEST_RESULT_FILE $UNIT_TEST_COUNT
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVE_PID

#
# AIP_STORAGE_URI / AIP_HTTP_PORT
#
unset_vertex_variables
export AIP_PREDICT_ROUTE="/predict"
export AIP_HEALTH_ROUTE="/health"
export AIP_STORAGE_URI=single_model
export AIP_HTTP_PORT=5234

SERVER_ARGS="--allow-vertex-ai=true"
run_server_nowait
vertex_ai_wait_for_server_ready $SERVER_PID 10
if [ "$WAIT_RET" != "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    kill $SERVER_PID
    wait $SERVER_PID
    cat $SERVER_LOG
    exit 1
fi

set +e
python $CLIENT_TEST_SCRIPT >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    cat $CLIENT_LOG
    RET=1
else
    check_test_results $TEST_RESULT_FILE $UNIT_TEST_COUNT
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVE_PID

#
# default model
#
unset_vertex_variables
export AIP_MODE=PREDICTION
export AIP_PREDICT_ROUTE="/predict"
export AIP_HEALTH_ROUTE="/health"

export AIP_STORAGE_URI=single_model
SERVER_ARGS="--vertex-ai-default-model=subadd"
run_server_nowait
vertex_ai_wait_for_server_ready $SERVER_PID 10
if [ "$WAIT_RET" == "0" ]; then
    echo -e "\n***\n*** Expect failed to start $SERVER\n***"
    kill $SERVER_PID || true
    cat $SERVER_LOG
    RET=1
else
  set +e
  grep "Expect the default model 'subadd' is loaded" $SERVER_LOG
  set -e
  if [ $? -ne 0 ]; then
      echo -e "\n***\n*** Failed. Expected error on nonexistent default model\n***"
      RET=1
  fi
fi

export AIP_STORAGE_URI=multi_models
SERVER_ARGS=""
run_server_nowait
vertex_ai_wait_for_server_ready $SERVER_PID 10
if [ "$WAIT_RET" == "0" ]; then
    echo -e "\n***\n*** Expect failed to start $SERVER\n***"
    kill $SERVER_PID || true
    cat $SERVER_LOG
    RET=1
else
  set +e
  grep "Expect the model repository contains only a single model if default model is not specified" $SERVER_LOG
  set -e
  if [ $? -ne 0 ]; then
      echo -e "\n***\n*** Failed. Expected error on unspecified default model\n***"
      RET=1
  fi
fi

# Test AIP_STORAGE_URI won't be used if model repository is specified
SERVER_ARGS="--model-repository=single_model"
run_server_nowait
vertex_ai_wait_for_server_ready $SERVER_PID 10
if [ "$WAIT_RET" != "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    kill $SERVER_PID
    wait $SERVER_PID
    cat $SERVER_LOG
    exit 1
fi

set +e
# subadd should not be loaded
code=`curl -s -w %{http_code} -o ./curl.out -X POST -H "X-Vertex-Ai-Triton-Redirect: v2/models/subadd/ready" localhost:8080/predict`
if [ "$code" == "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Expect 'subadd' is not loaded\n***"
    RET=1
fi
python $CLIENT_TEST_SCRIPT >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    cat $CLIENT_LOG
    RET=1
else
    check_test_results $TEST_RESULT_FILE $UNIT_TEST_COUNT
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e
kill $SERVER_PID
wait $SERVE_PID

# Test default model as well as multi model
SERVER_ARGS="--vertex-ai-default-model=addsub"
run_server_nowait
vertex_ai_wait_for_server_ready $SERVER_PID 10
if [ "$WAIT_RET" != "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    kill $SERVER_PID
    wait $SERVER_PID
    cat $SERVER_LOG
    exit 1
fi

set +e
python $CLIENT_TEST_SCRIPT >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    cat $CLIENT_LOG
    RET=1
else
    check_test_results $TEST_RESULT_FILE $UNIT_TEST_COUNT
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

# Defer the server exit to test redirection as the same time

#
# Redirect
#

# Metrics
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -X POST -H "X-Vertex-Ai-Triton-Redirect: metrics" localhost:8080/predict`
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    grep "nv_inference_request_success" ./curl.out
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected metrics are returned\n***"
        RET=1
    fi
fi
set -e

# All Model stats
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -X POST -H "X-Vertex-Ai-Triton-Redirect: v2/models/stats" localhost:8080/predict`
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    grep "model_stats" ./curl.out
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected model stats are returned\n***"
        RET=1
    fi
    grep "addsub" ./curl.out
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected 'addsub' model stats are returned\n***"
        RET=1
    fi
    grep "subadd" ./curl.out
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected 'subadd' model stats are returned\n***"
        RET=1
    fi
fi
set -e

# Single model stats
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -X POST -H "X-Vertex-Ai-Triton-Redirect: v2/models/subadd/stats" localhost:8080/predict`
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    grep "model_stats" ./curl.out
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected model stats are returned\n***"
        RET=1
    fi
    grep "addsub" ./curl.out
    if [ $? -eq 0 ]; then
        echo -e "\n***\n*** Failed. Unexpected 'addsub' model stats are returned\n***"
        RET=1
    fi
    grep "subadd" ./curl.out
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected 'subadd' model stats are returned\n***"
        RET=1
    fi
fi
set -e

# Server health
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -X POST -H "X-Vertex-Ai-Triton-Redirect: v2/health/live" localhost:8080/predict`
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

# Model ready
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -X POST -H "X-Vertex-Ai-Triton-Redirect: v2/models/addsub/ready" localhost:8080/predict`
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

# Server metadata
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -X POST -H "X-Vertex-Ai-Triton-Redirect: v2" localhost:8080/predict`
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    grep "extensions" ./curl.out
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected server metadata are returned\n***"
        RET=1
    fi
fi
set -e

# Model metadata
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -X POST -H "X-Vertex-Ai-Triton-Redirect: v2/models/addsub" localhost:8080/predict`
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    grep "platform" ./curl.out
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected model metadata are returned\n***"
        RET=1
    fi
fi
set -e

# Model config
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -X POST -H "X-Vertex-Ai-Triton-Redirect: v2/models/addsub/config" localhost:8080/predict`
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    grep "version_policy" ./curl.out
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected model configuration are returned\n***"
        RET=1
    fi
fi
set -e

# shared memory (only test "status" as register requires shared memory allocation)
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -X POST -H "X-Vertex-Ai-Triton-Redirect: v2/systemsharedmemory/status" localhost:8080/predict`
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    grep "name" ./curl.out
    if [ $? -eq 0 ]; then
        echo -e "\n***\n*** Failed. Expected no region is registered\n***"
        RET=1
    fi
fi
set -e

# cuda shared memory (only test "status" as register requires shared memory allocation)
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -X POST -H "X-Vertex-Ai-Triton-Redirect: v2/cudasharedmemory/status" localhost:8080/predict`
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    grep "name" ./curl.out
    if [ $? -eq 0 ]; then
        echo -e "\n***\n*** Failed. Expected no region is registered\n***"
        RET=1
    fi
fi
set -e

# repository index
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -X POST -H "X-Vertex-Ai-Triton-Redirect: v2/repository/index" localhost:8080/predict`
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    grep "state" ./curl.out
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected model index are returned\n***"
        RET=1
    fi
    grep "addsub" ./curl.out
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected 'addsub' in the index\n***"
        RET=1
    fi
    grep "subadd" ./curl.out
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected 'subadd' in the index\n***"
        RET=1
    fi
fi
set -e

# repository control (expect error)
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -X POST -H "X-Vertex-Ai-Triton-Redirect: v2/repository/models/subadd/unload" localhost:8080/predict`
if [ "$code" == "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    grep "explicit model load / unload is not allowed" ./curl.out
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected error on model control\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVE_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi
exit $RET
