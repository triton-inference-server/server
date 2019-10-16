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

# L0_simple_perf tests across different backends. Here just use custom
# backend as a proxy for all backends since the dynamic batcher
# behaves the same for all.
BACKENDS=${BACKENDS:="custom"}
TENSOR_SIZES=16384   # 16k fp32 elements
CONCURRENCIES=16
DBATCH_PREFS="1 4 8"
MODEL_LATENCY_MS=5

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
        [ $BACKEND != "custom" ] && REPO_DIR=$DATADIR/qa_identity_model_repository
    KIND="KIND_GPU" && [ $BACKEND == "custom" ] && KIND="KIND_CPU"

    echo -e "[" > ${BACKEND}.log

    FIRST=1
    for DBATCH_PREF in $DBATCH_PREFS; do
        rm -fr models && mkdir -p models && \
            cp -r $REPO_DIR/$MODEL_NAME models/. && \
            (cd models/$MODEL_NAME && \
                    sed -i "s/dims:.*\[.*\]/dims: \[ -1 \]/g" config.pbtxt && \
                    sed -i "s/max_batch_size:.*/max_batch_size: ${DBATCH_PREF}/g" config.pbtxt && \
                    echo "dynamic_batching { preferred_batch_size: [${DBATCH_PREF}] }" >> config.pbtxt && \
                    echo "instance_group [ { kind: ${KIND} }]" >> config.pbtxt && \
                    echo "parameters [ { key: \"execute_delay_ms\"; value: { string_value: \"${MODEL_LATENCY_MS}\" }}]" >> config.pbtxt)
        SERVER_ARGS=--model-repository=`pwd`/models
        SERVER_LOG="${BACKEND}.serverlog"
        run_server
        if (( $SERVER_PID == 0 )); then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        for TENSOR_SIZE in $TENSOR_SIZES; do
            set +e

            for CONCURRENCY in $CONCURRENCIES; do
                WARMUP_ITERS=10
                MEASURE_ITERS=200 && [[ "$TENSOR_SIZE" != "16384" ]] && \
                    MEASURE_ITERS=50

                if (( $FIRST != 1 )); then
                    echo -e "," >> ${BACKEND}.log
                fi
                FIRST=0

                # sync GRPC API
                $CLIENT -i grpc -u localhost:8001 \
                        -l"sync" -d${DBATCH_PREF} \
                        -f${BACKEND} -m${MODEL_NAME} -c${CONCURRENCY} -s${TENSOR_SIZE} \
                        -w${WARMUP_ITERS} -n${MEASURE_ITERS} >> ${BACKEND}.log 2>&1
                if (( $? != 0 )); then
                    RET=1
                fi
            done
            set -e
        done

        kill $SERVER_PID
        wait $SERVER_PID
    done

    echo -e "]" >> ${BACKEND}.log

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
