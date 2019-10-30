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

REPO_VERSION=${NVIDIA_TENSORRT_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi

CLIENT=../clients/simple_perf_client
REPORTER=../common/reporter.py

BACKENDS=${BACKENDS:="custom plan graphdef savedmodel onnx libtorch netdef"}

if [[ -z "${SIMPLE_PERF_LARGE_INPUT}" ]] || [[ "${SIMPLE_PERF_LARGE_INPUT}" == "0" ]]; then
  TENSOR_SIZE=16384 # 16k fp32 elements
  WARMUP_ITERS=1000
  MEASURE_ITERS=20000
else
  TENSOR_SIZE=16777216 # 16m fp32 elements
  WARMUP_ITERS=10
  MEASURE_ITERS=50
fi

CONCURRENCIES="1 8"

DATADIR=/data/inferenceserver/${REPO_VERSION}

SERVER=/opt/tensorrtserver/bin/trtserver
source ../common/util.sh

# Select the single GPU that will be available to the inference server
export CUDA_VISIBLE_DEVICES=0

rm -f *.json *.log *.serverlog
RET=0

rm -fr ./custom_models && mkdir ./custom_models && \
    cp -r ../custom_models/custom_zero_1_float32 ./custom_models/. && \
    mkdir -p ./custom_models/custom_zero_1_float32/1 && \
    cp ./libidentity.so ./custom_models/custom_zero_1_float32/1/libcustom.so

#
# Use "identity" model for all model types.
#
for BACKEND in $BACKENDS; do
    MODEL_NAME=${BACKEND}_zero_1_float32
    REPO_DIR=./custom_models && \
        [ $BACKEND != "custom" ] && REPO_DIR=$DATADIR/qa_identity_big_model_repository && \
        [ $BACKEND != "plan" ] && REPO_DIR=$DATADIR/qa_identity_model_repository
    KIND="KIND_GPU" && [ $BACKEND == "custom" ] && KIND="KIND_CPU"

    rm -fr models && mkdir -p models && \
        cp -r $REPO_DIR/$MODEL_NAME models/. && \
        (cd models/$MODEL_NAME && \
                sed -i "s/dims:.*\[.*\]/dims: \[ -1 \]/g" config.pbtxt && \
                echo "instance_group [ { kind: ${KIND} }]" >> config.pbtxt)

    SERVER_ARGS=--model-repository=`pwd`/models
    SERVER_LOG="${BACKEND}.serverlog"
    run_server
    if (( $SERVER_PID == 0 )); then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    echo -e "[" > ${BACKEND}.log

    FIRST=1
    set +e

    for CONCURRENCY in $CONCURRENCIES; do

        if (( $FIRST != 1 )); then
            echo -e "," >> ${BACKEND}.log
        fi
        FIRST=0

        # sync HTTP API
        $CLIENT -l"sync" \
                -f${BACKEND} -m${MODEL_NAME} -c${CONCURRENCY} -s${TENSOR_SIZE} \
                -w${WARMUP_ITERS} -n${MEASURE_ITERS} >> ${BACKEND}.log 2>&1
        if (( $? != 0 )); then
            RET=1
        fi

        echo -e "," >> ${BACKEND}.log

        # sync GRPC API
        $CLIENT -l"sync" -i grpc -u localhost:8001 \
                -f${BACKEND} -m${MODEL_NAME} -c${CONCURRENCY} -s${TENSOR_SIZE} \
                -w${WARMUP_ITERS} -n${MEASURE_ITERS} >> ${BACKEND}.log 2>&1
        if (( $? != 0 )); then
            RET=1
        fi


        echo -e "," >> ${BACKEND}.log

        # async HTTP API
        $CLIENT -a -l"async" \
                -f${BACKEND} -m${MODEL_NAME} -c${CONCURRENCY} -s${TENSOR_SIZE} \
                -w${WARMUP_ITERS} -n${MEASURE_ITERS} >> ${BACKEND}.log 2>&1
        if (( $? != 0 )); then
            RET=1
        fi

        echo -e "," >> ${BACKEND}.log

        # async GRPC API
        $CLIENT -a -l"async" -i grpc -u localhost:8001 \
                -f${BACKEND} -m${MODEL_NAME} -c${CONCURRENCY} -s${TENSOR_SIZE} \
                -w${WARMUP_ITERS} -n${MEASURE_ITERS} >> ${BACKEND}.log 2>&1
        if (( $? != 0 )); then
            RET=1
        fi
    done
    set -e

    echo -e "]" >> ${BACKEND}.log

    kill $SERVER_PID
    wait $SERVER_PID

    if [ -f $REPORTER ]; then
        set +e

        URL_FLAG=
        if [ ! -z ${BENCHMARK_REPORTER_URL} ]; then
            URL_FLAG="-u ${BENCHMARK_REPORTER_URL}"
        fi

        $REPORTER -v -o ${BACKEND}.json ${URL_FLAG} ${BACKEND}.log
        if (( $? != 0 )); then
            RET=1
        fi

        set -e
    fi
done

if (( $RET == 0 )); then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
