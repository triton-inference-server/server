#!/bin/bash
# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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
# Make sure we can safety use symbolic link for SageMaker serve script
if [ -d "/opt/ml/model" ] || [ -L "/opt/ml/model" ]; then
    echo -e "Default SageMaker model path must not be used for testing"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0

RET=0

rm -rf models
rm -f *.log
rm -f *.out

SAGEMAKER_TEST=sagemaker_test.py
SAGEMAKER_MULTI_MODEL_TEST=sagemaker_multi_model_test.py
SAGEMAKER_GENERATE_TEST=sagemaker_generate_test.py
SAGEMAKER_GENERATE_STREAM_TEST=sagemaker_generate_stream_test.py
MULTI_MODEL_UNIT_TEST_COUNT=7
UNIT_TEST_COUNT=9
GENERATE_UNIT_TEST_COUNT=1
GENERATE_STREAM_UNIT_TEST_COUNT=1
CLIENT_LOG="./client.log"

DATADIR=/data/inferenceserver/${REPO_VERSION}
ENSEMBLEDIR=/data/inferenceserver/${REPO_VERSION}/qa_ensemble_model_repository/qa_model_repository
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_LOG="./server.log"
# Link model repository to "/opt/ml/model"
mkdir /opt/ml/
ln -s `pwd`/models /opt/ml/model
source ../common/util.sh

mkdir models && \
    cp -r $DATADIR/qa_model_repository/onnx_int32_int32_int32 models/sm_model && \
    rm -r models/sm_model/2 && rm -r models/sm_model/3 && \
    sed -i "s/onnx_int32_int32_int32/sm_model/" models/sm_model/config.pbtxt

mkdir -p models/mock_llm/1 && \
    cp ../python_models/generate_models/mock_llm/1/model.py models/mock_llm/1 && \
    cp ../python_models/generate_models/mock_llm/config.pbtxt models/mock_llm

# Use SageMaker's ping endpoint to check server status
# Wait until server health endpoint shows ready. Sets WAIT_RET to 0 on
# success, 1 on failure
function sagemaker_wait_for_server_ready() {
    local spid="$1"; shift
    local wait_time_secs="${1:-30}"; shift

    WAIT_RET=0

    ping_address="localhost:8080/ping"
    if [ -n "$SAGEMAKER_BIND_TO_PORT" ]; then
        ping_address="localhost:${SAGEMAKER_BIND_TO_PORT}/ping"
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

# Start server with 'serve' script
export SAGEMAKER_TRITON_DEFAULT_MODEL_NAME=sm_model
serve > $SERVER_LOG 2>&1 &
SERVE_PID=$!
# Obtain Triton PID in such way as $! will return the script PID
sleep 1
SERVER_PID=`ps | grep tritonserver | awk '{ printf $1 }'`
sagemaker_wait_for_server_ready $SERVER_PID 10
if [ "$WAIT_RET" != "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    kill $SERVER_PID || true
    cat $SERVER_LOG
    exit 1
fi

# Ping
set +e
code=`curl -s -w %{http_code} -o ./ping.out localhost:8080/ping`
set -e
if [ "$code" != "200" ]; then
    cat ./ping.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

# Inference in default setting
set +e
python $SAGEMAKER_TEST SageMakerTest >>$CLIENT_LOG 2>&1
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

# Change SageMaker port
export SAGEMAKER_BIND_TO_PORT=8000
serve > $SERVER_LOG 2>&1 &
SERVE_PID=$!
# Obtain Triton PID in such way as $! will return the script PID
sleep 1
SERVER_PID=`ps | grep tritonserver | awk '{ printf $1 }'`
sagemaker_wait_for_server_ready $SERVER_PID 10
if [ "$WAIT_RET" != "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    kill $SERVER_PID || true
    cat $SERVER_LOG
    exit 1
fi

# Inference with the new port
set +e
python $SAGEMAKER_TEST SageMakerTest >>$CLIENT_LOG 2>&1
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

unset SAGEMAKER_BIND_TO_PORT

kill $SERVER_PID
wait $SERVE_PID

# Set SageMaker safe port range
export SAGEMAKER_SAFE_PORT_RANGE="8081-9000"

# Start Triton in a similar way to 'serve' script, as 'serve' script can't
# be used to satisfy the setting under test
SAGEMAKER_ARGS="--model-repository=/opt/ml/model"
if [ -n "$SAGEMAKER_BIND_TO_PORT" ]; then
    SAGEMAKER_ARGS="${SAGEMAKER_ARGS} --sagemaker-port=${SAGEMAKER_BIND_TO_PORT}"
fi
if [ -n "$SAGEMAKER_SAFE_PORT_RANGE" ]; then
    SAGEMAKER_ARGS="${SAGEMAKER_ARGS} --sagemaker-safe-port-range=${SAGEMAKER_SAFE_PORT_RANGE}"
fi

# Enable HTTP endpoint and expect server fail to start (default port 8000 < 8081)
SERVER_ARGS="--allow-sagemaker=true --allow-grpc false --allow-http true --allow-metrics false \
             --model-control-mode=explicit --load-model=${SAGEMAKER_TRITON_DEFAULT_MODEL_NAME} \
             $SAGEMAKER_ARGS"
run_server_nowait
sagemaker_wait_for_server_ready $SERVER_PID 10
if [ "$WAIT_RET" == "0" ]; then
    echo -e "\n***\n*** Expect failed to start $SERVER\n***"
    kill $SERVER_PID || true
    cat $SERVER_LOG
    RET=1
else
    grep "The server cannot listen to HTTP requests at port" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected error on using disallowed port\n***"
        RET=1
    fi
fi

# Run 'serve' script and expect SageMaker endpoint on default port 8080 (< 8081)
# is working
serve > $SERVER_LOG 2>&1 &
SERVE_PID=$!
# Obtain Triton PID in such way as $! will return the script PID
sleep 1
SERVER_PID=`ps | grep tritonserver | awk '{ printf $1 }'`

sagemaker_wait_for_server_ready $SERVER_PID 10
if [ "$WAIT_RET" != "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    kill $SERVER_PID || true
    cat $SERVER_LOG
    exit 1
fi

# Inference with the new port
set +e
python $SAGEMAKER_TEST SageMakerTest >>$CLIENT_LOG 2>&1
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

# Start server with LLM and set inference type to generate
export SAGEMAKER_TRITON_DEFAULT_MODEL_NAME=mock_llm
export SAGEMAKER_TRITON_INFERENCE_TYPE=generate
serve > $SERVER_LOG 2>&1 &
SERVE_PID=$!
# Obtain Triton PID in such way as $! will return the script PID
sleep 1
SERVER_PID=`ps | grep tritonserver | awk '{ printf $1 }'`
sagemaker_wait_for_server_ready $SERVER_PID 10
if [ "$WAIT_RET" != "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    kill $SERVER_PID || true
    cat $SERVER_LOG
    exit 1
fi

# Inference with generate inference type
set +e
python $SAGEMAKER_GENERATE_TEST SageMakerGenerateTest >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    cat $CLIENT_LOG
    RET=1
else
    check_test_results $TEST_RESULT_FILE $GENERATE_UNIT_TEST_COUNT
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

unset SAGEMAKER_TRITON_DEFAULT_MODEL_NAME
unset SAGEMAKER_TRITON_INFERENCE_TYPE

kill $SERVER_PID
wait $SERVE_PID

# Start server with LLM and set inference type to generate_stream
export SAGEMAKER_TRITON_DEFAULT_MODEL_NAME=mock_llm
export SAGEMAKER_TRITON_INFERENCE_TYPE=generate_stream
serve > $SERVER_LOG 2>&1 &
SERVE_PID=$!
# Obtain Triton PID in such way as $! will return the script PID
sleep 1
SERVER_PID=`ps | grep tritonserver | awk '{ printf $1 }'`
sagemaker_wait_for_server_ready $SERVER_PID 10
if [ "$WAIT_RET" != "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    kill $SERVER_PID || true
    cat $SERVER_LOG
    exit 1
fi

# Helper library to parse SSE events
# https://github.com/mpetazzoni/sseclient
pip install sseclient-py

# Inference with generate_stream inference type
set +e
python $SAGEMAKER_GENERATE_STREAM_TEST SageMakerGenerateStreamTest >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    cat $CLIENT_LOG
    RET=1
else
    check_test_results $TEST_RESULT_FILE $GENERATE_STREAM_UNIT_TEST_COUNT
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

unset SAGEMAKER_TRITON_DEFAULT_MODEL_NAME
unset SAGEMAKER_TRITON_INFERENCE_TYPE

kill $SERVER_PID
wait $SERVE_PID

# Test serve with incorrect inference type
export SAGEMAKER_TRITON_INFERENCE_TYPE=incorrect_inference_type
serve > $SERVER_LOG 2>&1 &
SERVE_PID=$!
# Obtain Triton PID in such way as $! will return the script PID
sleep 1
SERVER_PID=`ps | grep tritonserver | awk '{ printf $1 }'`
if [ -n "$SERVER_PID" ]; then
    echo -e "\n***\n*** Expect failed to start $SERVER\n***"
    kill $SERVER_PID || true
    cat $SERVER_LOG
    RET=1
else
    grep "ERROR: Invalid SAGEMAKER_TRITON_INFERENCE_TYPE" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected error on incorrect inference type\n***"
        RET=1
    fi
fi
unset SAGEMAKER_TRITON_INFERENCE_TYPE

unset SAGEMAKER_SAFE_PORT_RANGE
unset SAGEMAKER_TRITON_DEFAULT_MODEL_NAME

# Test serve with incorrect model name
export SAGEMAKER_TRITON_DEFAULT_MODEL_NAME=incorrect_model_name
serve > $SERVER_LOG 2>&1 &
SERVE_PID=$!
# Obtain Triton PID in such way as $! will return the script PID
sleep 1
SERVER_PID=`ps | grep tritonserver | awk '{ printf $1 }'`
if [ -n "$SERVER_PID" ]; then
    echo -e "\n***\n*** Expect failed to start $SERVER\n***"
    kill $SERVER_PID || true
    cat $SERVER_LOG
    RET=1
else
    grep "ERROR: Directory with provided SAGEMAKER_TRITON_DEFAULT_MODEL_NAME ${SAGEMAKER_TRITON_DEFAULT_MODEL_NAME} does not exist" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected error on model name and dir name mismatch\n***"
        RET=1
    fi
fi

unset SAGEMAKER_TRITON_DEFAULT_MODEL_NAME

# Test serve with SAGEMAKER_TRITON_DEFAULT_MODEL_NAME unset, but containing single model directory
rm -rf models/mock_llm
serve > $SERVER_LOG 2>&1 &
SERVE_PID=$!
# Obtain Triton PID in such way as $! will return the script PID
sleep 1
SERVER_PID=`ps | grep tritonserver | awk '{ printf $1 }'`
sagemaker_wait_for_server_ready $SERVER_PID 10
if [ "$WAIT_RET" != "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    kill $SERVER_PID || true
    cat $SERVER_LOG
    exit 1
else
    grep "WARNING: No SAGEMAKER_TRITON_DEFAULT_MODEL_NAME provided" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected server to start with only existing directory as model.\n***"
	RET=1
    fi
fi

kill $SERVER_PID
wait $SERVE_PID

# Test unspecified SAGEMAKER_TRITON_DEFAULT_MODEL_NAME for ecs/eks case
SERVER_ARGS="--allow-sagemaker=true --allow-grpc false --allow-http false --allow-metrics false \
             --model-repository `pwd`/models --model-control-mode=explicit --exit-on-error=false"
run_server_nowait
sleep 5
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
code=`curl -X POST -s -w %{http_code} -o ./invoke.out localhost:8080/invocations --data-raw 'dummy'`
set -e
if [ "$code" == "200" ]; then
    cat ./invoke.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    grep "Request for unknown model: 'unspecified_SAGEMAKER_TRITON_DEFAULT_MODEL_NAME' is not found" ./invoke.out
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected inference to fail with unspecified model error.\n***"
    fi
fi

kill $SERVER_PID
wait $SERVER_PID

# TODO: Test ensemble backend

# Run server with invalid model and exit-on-error=false
rm models/sm_model/1/*
SERVER_ARGS="--allow-sagemaker=true --allow-grpc false --allow-http false --allow-metrics false \
             --model-repository `pwd`/models --model-control-mode=explicit --load-model=sm_model \
             --exit-on-error=false"
run_server_nowait
sleep 5
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# Ping and expect error code in SME mode.
set +e
code=`curl -s -w %{http_code} -o ./ping.out localhost:8080/ping`
set -e
if [ "$code" == "200" ]; then
    cat ./ping.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

# MME begin
# Prepare model repository

ln -s `pwd`/models /opt/ml/models
# Model path will be of the form /opt/ml/models/<hash>/model
MODEL1_PATH="models/123456789abcdefghi/model"
MODEL2_PATH="models/987654321ihgfedcba/model"
mkdir -p "${MODEL1_PATH}"
mkdir -p "${MODEL2_PATH}"

cp -r $DATADIR/qa_model_repository/onnx_int32_int32_int32/* ${MODEL1_PATH} && \
    rm -r ${MODEL1_PATH}/2 && rm -r ${MODEL1_PATH}/3 && \
    sed -i "s/onnx_int32_int32_int32/sm_mme_model_1/" ${MODEL1_PATH}/config.pbtxt

cp -r $DATADIR/qa_identity_model_repository/onnx_zero_1_float32/* ${MODEL2_PATH} && \
    sed -i "s/onnx_zero_1_float32/sm_mme_model_2/" ${MODEL2_PATH}/config.pbtxt

# Ensemble model
ENSEMBLE_MODEL_PATH="models/123456789ensemble/model"
mkdir -p "${ENSEMBLE_MODEL_PATH}"

model_name=python_float32_float32_float32

mkdir -p ${ENSEMBLE_MODEL_PATH}/${model_name}/1 && \
cp ../python_models/add_sub/model.py ${ENSEMBLE_MODEL_PATH}/${model_name}/1/. && \
cp ../python_models/add_sub/config.pbtxt ${ENSEMBLE_MODEL_PATH}/${model_name}/.
(cd ${ENSEMBLE_MODEL_PATH}/${model_name} && \
                    sed -i "s/label_filename:.*//" config.pbtxt && \
                    echo "max_batch_size: 64" >> config.pbtxt)

# Ensemble part
mkdir -p ${ENSEMBLE_MODEL_PATH}/fan_${model_name}/1 && \
            cp ../python_models/add_sub/model.py ${ENSEMBLE_MODEL_PATH}/fan_${model_name}/1/. && \
            cp ../python_models/fan_add_sub/config.pbtxt ${ENSEMBLE_MODEL_PATH}/fan_${model_name}/. && \
            (cd ${ENSEMBLE_MODEL_PATH}/fan_${model_name} && \
                    sed -i "s/label_filename:.*//" config.pbtxt && \
                    sed -i "s/model_name: \"ENSEMBLE_MODEL_NAME\"/model_name: \"${model_name}\"/" config.pbtxt && \
                    sed -i "0,/name:.*/{s/name:.*/name: \"fan_${model_name}\"/}" config.pbtxt && \
                    echo "max_batch_size: 64" >> config.pbtxt)

# # custom float32 component of ensemble
cp -r $ENSEMBLEDIR/nop_TYPE_FP32_-1 ${ENSEMBLE_MODEL_PATH}/. && \
    mkdir -p ${ENSEMBLE_MODEL_PATH}/nop_TYPE_FP32_-1/1

# Start server with 'serve' script
export SAGEMAKER_MULTI_MODEL=true
export SAGEMAKER_TRITON_LOG_VERBOSE=true

serve > $SERVER_LOG 2>&1 &
SERVE_PID=$!
# Obtain Triton PID in such way as $! will return the script PID
sleep 1
SERVER_PID=`ps | grep tritonserver | awk '{ printf $1 }'`
sagemaker_wait_for_server_ready $SERVER_PID 10
if [ "$WAIT_RET" != "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    kill $SERVER_PID || true
    cat $SERVER_LOG
    exit 1
fi

# API tests in default setting
set +e
python $SAGEMAKER_MULTI_MODEL_TEST SageMakerMultiModelTest >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    cat $CLIENT_LOG
    RET=1
else
    check_test_results $TEST_RESULT_FILE $MULTI_MODEL_UNIT_TEST_COUNT
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

unset SAGEMAKER_MULTI_MODEL

unlink /opt/ml/models
rm -rf /opt/ml/models

kill $SERVER_PID
wait $SERVE_PID
# MME end

unlink /opt/ml/model
rm -rf /opt/ml/model

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
