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

REPO_VERSION=$1

BACKENDS=${BACKENDS:="plan custom graphdef savedmodel onnx libtorch python"}
STATIC_BATCH_SIZES=${STATIC_BATCH_SIZES:=1}
DYNAMIC_BATCH_SIZES=${DYNAMIC_BATCH_SIZES:=1}
INSTANCE_COUNTS=${INSTANCE_COUNTS:=1}
CONCURRENCY=${CONCURRENCY:=1}

PERF_CLIENT_PROTOCOL=${PERF_CLIENT_PROTOCOL:=grpc}
PERF_CLIENT_PERCENTILE=${PERF_CLIENT_PERCENTILE:=95}
PERF_CLIENT_STABILIZE_WINDOW=${PERF_CLIENT_STABILIZE_WINDOW:=5000}
PERF_CLIENT_STABILIZE_THRESHOLD=${PERF_CLIENT_STABILIZE_THRESHOLD:=5}
TENSOR_SIZE=${TENSOR_SIZE:=1}
SHARED_MEMORY=${SHARED_MEMORY:="none"}
REPORTER=../common/reporter.py

RESULTDIR=${RESULTDIR:=.}

TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
ARCH=${ARCH:="x86_64"}
SERVER=${TRITON_DIR}/bin/tritonserver
BACKEND_DIR=${TRITON_DIR}/backends
MODEL_REPO="${PWD}/models"
TF_VERSION=${TF_VERSION:=2}
SERVER_ARGS="--model-repository=${MODEL_REPO} --backend-directory=${BACKEND_DIR} --backend-config=tensorflow,version=${TF_VERSION}"
source ../common/util.sh

# DATADIR is already set in environment variable for aarch64
if [ "$ARCH" == "aarch64" ]; then
    PERF_CLIENT=${TRITON_DIR}/clients/bin/perf_client
else
    PERF_CLIENT=../clients/perf_client
    DATADIR=/data/inferenceserver/${REPO_VERSION}
fi

# Select the single GPU that will be available to the inference server
export CUDA_VISIBLE_DEVICES=0

mkdir -p ${RESULTDIR}
RET=0

if [[ $BACKENDS == *"python"* ]]; then
    cp /opt/tritonserver/backends/python/triton_python_backend_utils.py .

    mkdir -p python_models/python_zero_1_float32/1 && \
        cp ../python_models/identity_fp32/model.py ./python_models/python_zero_1_float32/1/model.py && \
        cp ../python_models/identity_fp32/config.pbtxt ./python_models/python_zero_1_float32/config.pbtxt
    (cd python_models/python_zero_1_float32 && \
        sed -i "s/^name:.*/name: \"python_zero_1_float32\"/" config.pbtxt)
fi

PERF_CLIENT_PERCENTILE_ARGS="" &&
    (( ${PERF_CLIENT_PERCENTILE} != 0 )) &&
    PERF_CLIENT_PERCENTILE_ARGS="--percentile=${PERF_CLIENT_PERCENTILE}"
PERF_CLIENT_EXTRA_ARGS="$PERF_CLIENT_PERCENTILE_ARGS --shared-memory \"${SHARED_MEMORY}\""

# Overload use of PERF_CLIENT_PROTOCOL for convenience with existing test and 
# reporting structure, though "triton_c_api" is not strictly a "protocol".
if [[ "${PERF_CLIENT_PROTOCOL}" == "triton_c_api" ]]; then
    # Server will be run in-process with C API
    SERVICE_ARGS="--service-kind triton_c_api \
                  --triton-server-directory ${TRITON_DIR} \
                  --model-repository ${MODEL_REPO}"
else
    SERVICE_ARGS="-i ${PERF_CLIENT_PROTOCOL}"
fi

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

    # plan and openvino models do not support 16MB I/O tests
    if ([ $BACKEND == "plan" ] || [ $BACKEND == "openvino" ]) && [ $TENSOR_SIZE != 1 ]; then
        continue
    fi

    # set input name (special case for libtorch model)
    INPUT_NAME="INPUT0" && [ $BACKEND == "libtorch" ] && INPUT_NAME="INPUT__0"

    MAX_LATENCY=300
    MAX_BATCH=${STATIC_BATCH} && [ $DYNAMIC_BATCH > $STATIC_BATCH ] && MAX_BATCH=${DYNAMIC_BATCH}

    # TODO Add openvino identity model that supports batching/dynamic batching
    # The current openvino identity model does also not support batching
    if [ $BACKEND == "openvino" ]; then
        if [ $MAX_BATCH != 1 ]; then
            continue
        else
            MAX_BATCH=0
        fi
    fi

    if [ $DYNAMIC_BATCH > 1 ]; then
        NAME=${BACKEND}_sbatch${STATIC_BATCH}_dbatch${DYNAMIC_BATCH}_instance${INSTANCE_CNT}
    else
        NAME=${BACKEND}_sbatch${STATIC_BATCH}_instance${INSTANCE_CNT}
    fi

    # set model name (special case for openvino i.e. nobatch)
    MODEL_NAME=${BACKEND}_zero_1_float32 && [ $BACKEND == "openvino" ] && MODEL_NAME=${BACKEND}_nobatch_zero_1_float32

    if [ $BACKEND == "custom" ]; then
        REPO_DIR=./custom_models
    elif [ $BACKEND == "python" ]; then
        REPO_DIR=./python_models
    else
        REPO_DIR=$DATADIR/qa_identity_model_repository
    fi

    SHAPE=${TENSOR_SIZE}
    KIND="KIND_GPU" && [ $BACKEND == "custom" ] || [ $BACKEND == "python" ] || [ $BACKEND == "openvino" ] && KIND="KIND_CPU"

    rm -fr models && mkdir -p models && \
        cp -r $REPO_DIR/$MODEL_NAME models/. && \
        (cd models/$MODEL_NAME && \
                sed -i "s/^max_batch_size:.*/max_batch_size: ${MAX_BATCH}/" config.pbtxt)

    # python model already has instance count and kind
    if [ $BACKEND == "python" ]; then
        (cd models/$MODEL_NAME && \
                sed -i "s/count:.*/count: ${INSTANCE_CNT}/" config.pbtxt)
    else
        (cd models/$MODEL_NAME && \
                echo "instance_group [ { kind: ${KIND}, count: ${INSTANCE_CNT} }]" >> config.pbtxt)
    fi

    if [ $BACKEND == "custom" ]; then
        (cd models/$MODEL_NAME && \
                sed -i "s/dims:.*\[.*\]/dims: \[ ${SHAPE} \]/g" config.pbtxt)
    fi
    if [ $DYNAMIC_BATCH > 1 ] && [ $BACKEND != "openvino" ]; then
        (cd models/$MODEL_NAME && \
                echo "dynamic_batching { preferred_batch_size: [ ${DYNAMIC_BATCH} ] }" >> config.pbtxt)
    fi

    # Only start separate server if not using C API, since C API runs server in-process
    if [[ "${PERF_CLIENT_PROTOCOL}" != "triton_c_api" ]]; then
        SERVER_LOG="${RESULTDIR}/${NAME}.serverlog"
        run_server
        if [ $SERVER_PID == 0 ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi
    fi

    set +e
    $PERF_CLIENT -v \
                 -p${PERF_CLIENT_STABILIZE_WINDOW} \
                 -s${PERF_CLIENT_STABILIZE_THRESHOLD} \
                 ${PERF_CLIENT_EXTRA_ARGS} \
                 -m ${MODEL_NAME} \
                 -b${STATIC_BATCH} -t${CONCURRENCY} \
                 --shape ${INPUT_NAME}:${SHAPE} \
                 ${SERVICE_ARGS} \
                 -f ${RESULTDIR}/${NAME}.csv 2>&1 | tee ${RESULTDIR}/${NAME}.log
    if [ $? -ne 0 ]; then
        RET=1
    fi
    set -e

    echo -e "[{\"s_benchmark_kind\":\"benchmark_perf\"," >> ${RESULTDIR}/${NAME}.tjson
    echo -e "\"s_benchmark_name\":\"nomodel\"," >> ${RESULTDIR}/${NAME}.tjson
    echo -e "\"s_server\":\"triton\"," >> ${RESULTDIR}/${NAME}.tjson
    echo -e "\"s_protocol\":\"${PERF_CLIENT_PROTOCOL}\"," >> ${RESULTDIR}/${NAME}.tjson
    echo -e "\"s_framework\":\"${BACKEND}\"," >> ${RESULTDIR}/${NAME}.tjson
    echo -e "\"s_model\":\"${MODEL_NAME}\"," >> ${RESULTDIR}/${NAME}.tjson
    echo -e "\"l_concurrency\":${CONCURRENCY}," >> ${RESULTDIR}/${NAME}.tjson
    echo -e "\"l_dynamic_batch_size\":${DYNAMIC_BATCH}," >> ${RESULTDIR}/${NAME}.tjson
    echo -e "\"l_batch_size\":${STATIC_BATCH}," >> ${RESULTDIR}/${NAME}.tjson
    echo -e "\"l_size\":${TENSOR_SIZE}," >> ${RESULTDIR}/${NAME}.tjson
    echo -e "\"s_shared_memory\":\"${SHARED_MEMORY}\"," >> ${RESULTDIR}/${NAME}.tjson
    echo -e "\"l_instance_count\":${INSTANCE_CNT}," >> ${RESULTDIR}/${NAME}.tjson
    echo -e "\"s_architecture\":\"${ARCH}\"}]" >> ${RESULTDIR}/${NAME}.tjson

    # SERVER_PID may not be set if using "triton_c_api" for example
    if [[ -n "${SERVER_PID}" ]]; then
        kill $SERVER_PID
        wait $SERVER_PID
    fi

    if [ -f $REPORTER ]; then
        set +e

        URL_FLAG=
        if [ ! -z ${BENCHMARK_REPORTER_URL} ]; then
            URL_FLAG="-u ${BENCHMARK_REPORTER_URL}"
        fi

        $REPORTER -v -o ${RESULTDIR}/${NAME}.json --csv ${RESULTDIR}/${NAME}.csv ${URL_FLAG} ${RESULTDIR}/${NAME}.tjson
        if [ $? -ne 0 ]; then
            RET=1
        fi

        set -e
    fi
   done
  done
 done
done

if [ $RET == 0 ]; then
    echo -e "\n***\n*** Test ${RESULTNAME} Passed\n***"
else
    echo -e "\n***\n*** Test ${RESULTNAME} FAILED\n***"
fi

exit $RET
