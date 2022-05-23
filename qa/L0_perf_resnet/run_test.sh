#!/bin/bash
# Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

STATIC_BATCH=${STATIC_BATCH:=1}
INSTANCE_CNT=${INSTANCE_CNT:=1}
BACKEND_CONFIG=${BACKEND_CONFIG:=""}

REPORTER=../common/reporter.py

TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
BACKEND_DIR=${TRITON_DIR}/backends
MODEL_REPO="${PWD}/models"
SERVER_ARGS="--model-repository=${MODEL_REPO} --backend-directory=${BACKEND_DIR} ${BACKEND_CONFIG}"
source ../common/util.sh

# Select the single GPU that will be available to the inference
# server. Or use "export CUDA_VISIBLE_DEVICE=" to run on CPU.
export CUDA_VISIBLE_DEVICES=0

RET=0

MAX_BATCH=${STATIC_BATCH}
NAME=${MODEL_NAME}_sbatch${STATIC_BATCH}_instance${INSTANCE_CNT}_${PERF_CLIENT_PROTOCOL}

rm -fr models && mkdir -p models && \
    cp -r $MODEL_PATH models/. && \
    (cd models/$MODEL_NAME && \
            sed -i "s/^max_batch_size:.*/max_batch_size: ${MAX_BATCH}/" config.pbtxt && \
            echo "instance_group [ { count: ${INSTANCE_CNT} }]" >> config.pbtxt)

# Don't start separate server if testing perf with C API
if [[ "${PERF_CLIENT_PROTOCOL}" != "triton_c_api" ]]; then
    SERVER_LOG="${NAME}.serverlog"
    run_server
    if (( $SERVER_PID == 0 )); then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi
fi

# Onnx and onnx-trt models are very slow on Jetson.
MEASUREMENT_WINDOW=5000
if [ "$ARCH" == "aarch64" ]; then
    PERF_CLIENT=${TRITON_DIR}/clients/bin/perf_client
    if [ "$MODEL_FRAMEWORK" == "onnx" ] || [ "$MODEL_FRAMEWORK" == "onnx_trt" ]; then
        MEASUREMENT_WINDOW=20000
    fi
else
    PERF_CLIENT=../clients/perf_client
fi

set +e

# Overload use of PERF_CLIENT_PROTOCOL for convenience with existing test and 
# reporting structure, though "triton_c_api" is not strictly a "protocol".
if [[ "${PERF_CLIENT_PROTOCOL}" == "triton_c_api" ]]; then
    # Using C API requires extra info to start in-process server
    # NOTE: PA C API doesn't expose --backend-directory, but defaults to
    #       ${TRITON_DIR}/backends
    # NOTE: PA C API doesn't expose --backend-config, but this is only used
    #       for tfserving grpc test currently.
    SERVICE_ARGS="--service-kind triton_c_api \
                  --triton-server-directory ${TRITON_DIR} \
                  --model-repository ${MODEL_REPO}"
# Otherwise run as normal via HTTP/GRPC
else
    SERVICE_ARGS="-i ${PERF_CLIENT_PROTOCOL}"
fi

# Run the model once to warm up. Some frameworks do optimization on the first requests.
# Must warmup similar to actual run so that all instances are ready
$PERF_CLIENT -v -m $MODEL_NAME -p${MEASUREMENT_WINDOW} \
                -b${STATIC_BATCH} --concurrency-range ${CONCURRENCY} \
                ${SERVICE_ARGS}

$PERF_CLIENT -v -m $MODEL_NAME -p${MEASUREMENT_WINDOW} \
                -b${STATIC_BATCH} --concurrency-range ${CONCURRENCY} \
                ${SERVICE_ARGS} \
                -f ${NAME}.csv 2>&1 | tee ${NAME}.log

if (( $? != 0 )); then
    RET=1
fi

# No metrics endpoint available when running with C API
if [[ "${PERF_CLIENT_PROTOCOL}" != "triton_c_api" ]]; then
    # NOTE: The metrics API output is not currently used
    curl localhost:8002/metrics -o ${NAME}.metrics >> ${NAME}.log 2>&1
    if (( $? != 0 )); then
        RET=1
    fi
fi

set -e

echo -e "[{\"s_benchmark_kind\":\"benchmark_perf\"," >> ${NAME}.tjson
echo -e "\"s_benchmark_name\":\"resnet50\"," >> ${NAME}.tjson
echo -e "\"s_server\":\"triton\"," >> ${NAME}.tjson
echo -e "\"s_protocol\":\"${PERF_CLIENT_PROTOCOL}\"," >> ${NAME}.tjson
echo -e "\"s_framework\":\"${MODEL_FRAMEWORK}\"," >> ${NAME}.tjson
echo -e "\"s_model\":\"${MODEL_NAME}\"," >> ${NAME}.tjson
echo -e "\"l_concurrency\":${CONCURRENCY}," >> ${NAME}.tjson
echo -e "\"l_batch_size\":${STATIC_BATCH}," >> ${NAME}.tjson
echo -e "\"l_instance_count\":${INSTANCE_CNT}," >> ${NAME}.tjson
echo -e "\"s_architecture\":\"${ARCH}\"}]" >> ${NAME}.tjson

# SERVER_PID may not be set if using "triton_c_api" for example
if [[ "${SERVER_PID}" -ne 0 ]]; then
  kill $SERVER_PID
  wait $SERVER_PID
fi

if [ -f $REPORTER ]; then
    set +e

    URL_FLAG=
    if [ ! -z ${BENCHMARK_REPORTER_URL} ]; then
        URL_FLAG="-u ${BENCHMARK_REPORTER_URL}"
    fi

    $REPORTER -v -o ${NAME}.json --csv ${NAME}.csv ${URL_FLAG} ${NAME}.tjson
    if (( $? != 0 )); then
        RET=1
    fi

    set -e
fi

if (( $RET == 0 )); then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
