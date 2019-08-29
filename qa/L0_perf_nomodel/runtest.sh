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

BACKENDS=${BACKENDS:="plan custom graphdef savedmodel onnx libtorch netdef"}
STATIC_BATCH_SIZES=${STATIC_BATCH_SIZES:=1}
DYNAMIC_BATCH_SIZES=${DYNAMIC_BATCH_SIZES:=1}
INSTANCE_COUNTS=${INSTANCE_COUNTS:=1}
REQUIRED_MAX_CONCURRENCY=${REQUIRED_MAX_CONCURRENCY:=4}

PERF_CLIENT=../clients/perf_client
PERF_CLIENT_PROTOCOL=${PERF_CLIENT_PROTOCOL:=grpc}
PERF_CLIENT_PERCENTILE=${PERF_CLIENT_PERCENTILE:=95}
PERF_CLIENT_STABILIZE_WINDOW=${PERF_CLIENT_STABILIZE_WINDOW:=5000}
PERF_CLIENT_STABILIZE_THRESHOLD=${PERF_CLIENT_STABILIZE_THRESHOLD:=5}
TENSOR_SIZE=${TENSOR_SIZE:=1}

DATADIR=/data/inferenceserver/$1
RESULTDIR=${RESULTDIR:=.}

SERVER=/opt/tensorrtserver/bin/trtserver
SERVER_ARGS=--model-repository=`pwd`/models
source ../common/util.sh

# Select the single GPU that will be available to the inference server
export CUDA_VISIBLE_DEVICES=0

mkdir -p ${RESULTDIR}
rm -f *.log *.serverlog *.csv *.metrics
RET=0

rm -fr ./custom_models && mkdir ./custom_models && \
    cp -r ../custom_models/custom_zero_1_float32 ./custom_models/. && \
    mkdir -p ./custom_models/custom_zero_1_float32/1 && \
    cp ./libidentity.so ./custom_models/custom_zero_1_float32/1/libcustom.so

PERF_CLIENT_PROTOCOL_ARGS="-i grpc -u localhost:8001" &&
    [ $PERF_CLIENT_PROTOCOL != "grpc" ] && PERF_CLIENT_PROTOCOL_ARGS=""
PERF_CLIENT_PERCENTILE_ARGS="" &&
    (( ${PERF_CLIENT_PERCENTILE} != 0 )) &&
    PERF_CLIENT_PERCENTILE_ARGS="--percentile=${PERF_CLIENT_PERCENTILE}"

#
# Use "identity" model for all model types.
#
for BACKEND in $BACKENDS; do
 for STATIC_BATCH in $STATIC_BATCH_SIZES; do
  for DYNAMIC_BATCH in $DYNAMIC_BATCH_SIZES; do
   for INSTANCE_CNT in $INSTANCE_COUNTS; do
    if (( ($DYNAMIC_BATCH > 1) && ($STATIC_BATCH >= $DYNAMIC_BATCH) )); then
        continue
    fi

    MAX_LATENCY=300
    MAX_BATCH=${STATIC_BATCH} && (( $DYNAMIC_BATCH > $STATIC_BATCH )) && MAX_BATCH=${DYNAMIC_BATCH}
    MAX_CONCURRENCY=$(( $INSTANCE_CNT * $DYNAMIC_BATCH * 2 ))
    if (( $MAX_CONCURRENCY < $REQUIRED_MAX_CONCURRENCY )); then
        MAX_CONCURRENCY=$REQUIRED_MAX_CONCURRENCY
    fi

    if (( $DYNAMIC_BATCH > 1 )); then
        NAME=${BACKEND}_sbatch${STATIC_BATCH}_dbatch${DYNAMIC_BATCH}_instance${INSTANCE_CNT}
    else
        NAME=${BACKEND}_sbatch${STATIC_BATCH}_instance${INSTANCE_CNT}
    fi

    MODEL_NAME=${BACKEND}_zero_1_float32
    REPO_DIR=./custom_models && \
        [ $BACKEND != "custom" ] && REPO_DIR=$DATADIR/qa_reshape_model_repository && \
        [ $BACKEND != "plan" ] && [ $BACKEND != "libtorch" ] && \
        REPO_DIR=$DATADIR/qa_identity_model_repository
    SHAPE=${TENSOR_SIZE} && [ $BACKEND == "plan" ] && SHAPE="1,1,${TENSOR_SIZE}"
    KIND="KIND_GPU" && [ $BACKEND == "custom" ] && KIND="KIND_CPU"

    rm -fr models && mkdir -p models && \
        cp -r $REPO_DIR/$MODEL_NAME models/.
        (cd models/$MODEL_NAME && \
                sed -i "s/^max_batch_size:.*/max_batch_size: ${MAX_BATCH}/" config.pbtxt && \
                sed -i "s/dims:.*\[.*\]/dims: \[ ${SHAPE} \]/g" config.pbtxt && \
                echo "instance_group [ { kind: ${KIND}, count: ${INSTANCE_CNT} }]" >> config.pbtxt)
    if (( $DYNAMIC_BATCH > 1 )); then
        (cd models/$MODEL_NAME && \
                echo "dynamic_batching { preferred_batch_size: [ ${DYNAMIC_BATCH} ] }" >> config.pbtxt)
    fi

    SERVER_LOG="${RESULTDIR}/${NAME}.serverlog"
    run_server
    if (( $SERVER_PID == 0 )); then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e
    $PERF_CLIENT -v -d \
                 -p${PERF_CLIENT_STABILIZE_WINDOW} \
                 -s${PERF_CLIENT_STABILIZE_THRESHOLD} \
                 ${PERF_CLIENT_PERCENTILE_ARGS} \
                 ${PERF_CLIENT_PROTOCOL_ARGS} -m ${MODEL_NAME} \
                 -b${STATIC_BATCH} -l${MAX_LATENCY} -c${MAX_CONCURRENCY} \
                 -f ${RESULTDIR}/${NAME}.csv >> ${RESULTDIR}/${NAME}.log 2>&1
    if (( $? != 0 )); then
        RET=1
    fi
    curl localhost:8002/metrics -o ${RESULTDIR}/${NAME}.metrics >> ${RESULTDIR}/${NAME}.log 2>&1
    if (( $? != 0 )); then
        RET=1
    fi
    set -e

    kill $SERVER_PID
    wait $SERVER_PID
   done
  done
 done
done

if (( $RET == 0 )); then
    echo -e "\n***\n*** Test ${RESULTNAME} Passed\n***"
else
    echo -e "\n***\n*** Test ${RESULTNAME} FAILED\n***"
fi

exit $RET
