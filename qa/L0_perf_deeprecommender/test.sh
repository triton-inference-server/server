#!/bin/bash
# Copyright 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

REPODIR=/data/inferenceserver/${REPO_VERSION}
TRTEXEC=/usr/src/tensorrt/bin/trtexec
MODEL="deeprecommender"
PROTOCOLS="grpc http"

rm -f *.log *.serverlog *.csv *.metrics *.tjson *.json

#
# Test minimum latency
#
STATIC_BATCH=1
INSTANCE_CNT=1
CONCURRENCY=1

# Create the TensorRT plan from TF
rm -fr tensorrt_models && mkdir tensorrt_models
    cp -r $REPODIR/perf_model_store/deeprecommender_graphdef tensorrt_models/deeprecommender_plan && \
    (cd tensorrt_models/deeprecommender_plan && \
        sed -i "s/^name:.*/name: \"deeprecommender_plan\"/" config.pbtxt && \
        sed -i "s/tensorflow_graphdef/tensorrt_plan/" config.pbtxt && \
        sed -i "s/max_batch_size:.*/max_batch_size: ${STATIC_BATCH}/" config.pbtxt && \
        sed -i "s/\[17736\]/\[17736,1,1\]/" config.pbtxt)

$TRTEXEC --uff=$REPODIR/perf_model_store/deeprecommender_graphdef/deeprecommender_graphdef.uff \
         --uffInput=Placeholder,1,1,17736\
         --batch=${STATIC_BATCH} --output=fc5/Relu --verbose \
         --saveEngine=tensorrt_models/deeprecommender_plan/0/model.plan

OPTIMIZED_MODEL_NAMES="deeprecommender_graphdef_trt"

# Create optimized models (TF-TRT and ONNX-TRT)
rm -fr optimized_model_store && mkdir optimized_model_store
for MODEL_NAME in $OPTIMIZED_MODEL_NAMES; do
    BASE_MODEL=$(echo ${MODEL_NAME} | cut -d '_' -f 1,2)
    cp -r $REPODIR/perf_model_store/${BASE_MODEL} optimized_model_store/${MODEL_NAME}
    CONFIG_PATH="optimized_model_store/${MODEL_NAME}/config.pbtxt"
    sed -i "s/^name: \"${BASE_MODEL}\"/name: \"${MODEL_NAME}\"/" ${CONFIG_PATH}
    echo "optimization { execution_accelerators {" >> ${CONFIG_PATH}
    echo "gpu_execution_accelerator : [ {" >> ${CONFIG_PATH}
    echo "name : \"tensorrt\" " >> ${CONFIG_PATH}
    echo "} ]" >> ${CONFIG_PATH}
    echo "}}" >> ${CONFIG_PATH}
done

# Tests with each model
for FRAMEWORK in graphdef plan graphdef_trt onnx libtorch; do
    MODEL_NAME=${MODEL}_${FRAMEWORK}
    if [ "$FRAMEWORK" == "plan" ]; then
        REPO=`pwd`/tensorrt_models
    elif [[ "$FRAMEWORK" == *"_trt" ]]; then
        REPO=`pwd`/optimized_model_store
    else
        REPO=$REPODIR/perf_model_store
    fi
    for PROTOCOL in $PROTOCOLS; do
        MODEL_NAME=${MODEL_NAME} \
                MODEL_FRAMEWORK=${FRAMEWORK} \
                MODEL_PATH="$REPO/${MODEL_NAME}" \
                STATIC_BATCH_SIZES=${STATIC_BATCH} \
                DYNAMIC_BATCH_SIZES=1 \
                PERF_CLIENT_PROTOCOL=${PROTOCOL} \
                INSTANCE_COUNTS=${INSTANCE_CNT} \
                CONCURRENCY=${CONCURRENCY} \
                bash -x run_test.sh
    done
done

#
# Test large static batch = 256 w/ 2 instances
#
STATIC_BATCH=256
INSTANCE_CNT=2
CONCURRENCY=4

# Create the TensorRT plan from TF
rm -fr tensorrt_models && mkdir tensorrt_models
    cp -r $REPODIR/perf_model_store/deeprecommender_graphdef tensorrt_models/deeprecommender_plan && \
    (cd tensorrt_models/deeprecommender_plan && \
        sed -i "s/^name:.*/name: \"deeprecommender_plan\"/" config.pbtxt && \
        sed -i "s/tensorflow_graphdef/tensorrt_plan/" config.pbtxt && \
        sed -i "s/max_batch_size:.*/max_batch_size: ${STATIC_BATCH}/" config.pbtxt && \
        sed -i "s/\[17736\]/\[17736,1,1\]/" config.pbtxt)

$TRTEXEC --uff=$REPODIR/perf_model_store/deeprecommender_graphdef/deeprecommender_graphdef.uff \
         --uffInput=Placeholder,1,1,17736\
         --batch=${STATIC_BATCH} --output=fc5/Relu --verbose \
         --saveEngine=tensorrt_models/deeprecommender_plan/0/model.plan

# Tests with each model
for FRAMEWORK in graphdef plan graphdef_trt onnx libtorch; do
    MODEL_NAME=${MODEL}_${FRAMEWORK}
    if [ "$FRAMEWORK" == "plan" ]; then
        REPO=`pwd`/tensorrt_models
    elif [[ "$FRAMEWORK" == *"_trt" ]]; then
        REPO=`pwd`/optimized_model_store
    else
        REPO=$REPODIR/perf_model_store
    fi
    for PROTOCOL in $PROTOCOLS; do
        MODEL_NAME=${MODEL_NAME} \
                MODEL_FRAMEWORK=${FRAMEWORK} \
                MODEL_PATH="$REPO/${MODEL_NAME}" \
                STATIC_BATCH_SIZES=${STATIC_BATCH} \
                DYNAMIC_BATCH_SIZES=1 \
                PERF_CLIENT_PROTOCOL=${PROTOCOL} \
                INSTANCE_COUNTS=${INSTANCE_CNT} \
                CONCURRENCY=${CONCURRENCY} \
                bash -x run_test.sh
    done
done
