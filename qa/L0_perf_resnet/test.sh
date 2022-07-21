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

rm -f *.log *.serverlog *.csv *.tjson *.json

PROTOCOLS="grpc http triton_c_api"

TRT_MODEL_NAME="resnet50_fp16_plan"
TF_MODEL_NAME="resnet50v1.5_fp16_savedmodel"
PYT_MODEL_NAME="resnet50_fp32_libtorch"
ONNX_MODEL_NAME="resnet50_fp32_onnx"

# The base model name should be the prefix to the
# respective optimized model name.
TFTRT_MODEL_NAME="resnet50v1.5_fp16_savedmodel_trt"
ONNXTRT_MODEL_NAME="resnet50_fp32_onnx_trt"
TFAMP_MODEL_NAME="resnet50v1.5_fp16_savedmodel_amp"

ARCH=${ARCH:="x86_64"}
REPODIR=${REPODIR:="/data/inferenceserver/${REPO_VERSION}"}
TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
CACHE_PATH=`pwd`/trt_cache


#
# Test minimum latency
#
STATIC_BATCH=1
INSTANCE_CNT=1
CONCURRENCY=1

# Disable TF-TRT test on Jetson due to Segfault
# Disable ORT-TRT test on Jetson due to support being disabled
if [ "$ARCH" == "aarch64" ]; then
    MODEL_NAMES="${TRT_MODEL_NAME} ${TF_MODEL_NAME} ${ONNX_MODEL_NAME} ${PYT_MODEL_NAME}"
    OPTIMIZED_MODEL_NAMES="${TFAMP_MODEL_NAME}"
    CAFFE2PLAN=${TRITON_DIR}/bin/caffe2plan
else
    MODEL_NAMES="${TRT_MODEL_NAME} ${TF_MODEL_NAME} ${ONNX_MODEL_NAME} ${PYT_MODEL_NAME}"
    OPTIMIZED_MODEL_NAMES="${TFTRT_MODEL_NAME} ${TFAMP_MODEL_NAME} ${ONNXTRT_MODEL_NAME}"
    CAFFE2PLAN=../common/caffe2plan
fi

# Create optimized models
rm -fr optimized_model_store && mkdir optimized_model_store
for MODEL_NAME in $OPTIMIZED_MODEL_NAMES; do
    BASE_MODEL=$(echo ${MODEL_NAME} | cut -d '_' -f 1,2,3)
    cp -r $REPODIR/perf_model_store/${BASE_MODEL} optimized_model_store/${MODEL_NAME}
    CONFIG_PATH="optimized_model_store/${MODEL_NAME}/config.pbtxt"
    sed -i "s/^name: \"${BASE_MODEL}\"/name: \"${MODEL_NAME}\"/" ${CONFIG_PATH}
    echo "optimization { execution_accelerators {" >> ${CONFIG_PATH}
    echo "gpu_execution_accelerator : [ {" >> ${CONFIG_PATH}
    if [ "${MODEL_NAME}" = "${TFAMP_MODEL_NAME}" ] ; then
        echo "name : \"auto_mixed_precision\" " >> ${CONFIG_PATH}
    else
        echo "name : \"tensorrt\" " >> ${CONFIG_PATH}
        if [ "${MODEL_NAME}" = "${TFTRT_MODEL_NAME}" ] ; then
            echo "parameters { key: \"precision_mode\" value: \"FP16\" }" >> ${CONFIG_PATH}
        fi

        if [ "${MODEL_NAME}" = "${ONNXTRT_MODEL_NAME}" ] ; then
            echo "parameters { key: \"precision_mode\" value: \"FP16\" }" >> ${CONFIG_PATH}
            echo "parameters { key: \"max_workspace_size_bytes\" value: \"1073741824\" }" >> ${CONFIG_PATH}
            echo "parameters { key: \"trt_engine_cache_enable\" value: \"1\" }" >> ${CONFIG_PATH}
            echo "parameters { key: \"trt_engine_cache_path\" value: \"${CACHE_PATH}\" } " >> ${CONFIG_PATH}
        fi
    fi
    echo "} ]" >> ${CONFIG_PATH}
    echo "}}" >> ${CONFIG_PATH}
done

# Create the TensorRT plan from Caffe model
rm -fr tensorrt_models && mkdir tensorrt_models
cp -r $REPODIR/caffe_models/trt_model_store/resnet50_plan tensorrt_models/${TRT_MODEL_NAME} && \
    (cd tensorrt_models/${TRT_MODEL_NAME} && \
            sed -i "s/^name:.*/name: \"${TRT_MODEL_NAME}\"/" config.pbtxt && \
            sed -i "s/max_batch_size:.*/max_batch_size: ${STATIC_BATCH}/" config.pbtxt) && \
    mkdir -p tensorrt_models/${TRT_MODEL_NAME}/1
$CAFFE2PLAN -h -b ${STATIC_BATCH} \
            -n prob -o tensorrt_models/${TRT_MODEL_NAME}/1/model.plan \
            $REPODIR/caffe_models/resnet50.prototxt $REPODIR/caffe_models/resnet50.caffemodel

# Tests with each "non-optimized" model
for MODEL_NAME in $MODEL_NAMES; do
    for PROTOCOL in $PROTOCOLS; do
        REPO=`pwd`/tensorrt_models && [ "$MODEL_NAME" != "$TRT_MODEL_NAME" ] && \
            REPO=$REPODIR/perf_model_store
        FRAMEWORK=$(echo ${MODEL_NAME} | cut -d '_' -f 3)
        MODEL_NAME=${MODEL_NAME} \
                MODEL_FRAMEWORK=${FRAMEWORK} \
                MODEL_PATH="$REPO/${MODEL_NAME}" \
                STATIC_BATCH=${STATIC_BATCH} \
                PERF_CLIENT_PROTOCOL=${PROTOCOL} \
                INSTANCE_CNT=${INSTANCE_CNT} \
                CONCURRENCY=${CONCURRENCY} \
                ARCH=${ARCH} \
                bash -x run_test.sh
    done
done

# Tests with optimization enabled models
for MODEL_NAME in $OPTIMIZED_MODEL_NAMES; do
    for PROTOCOL in $PROTOCOLS; do
        REPO=`pwd`/optimized_model_store
        FRAMEWORK=$(echo ${MODEL_NAME} | cut -d '_' -f 3,4)
        MODEL_NAME=${MODEL_NAME} \
                MODEL_FRAMEWORK=${FRAMEWORK} \
                MODEL_PATH="$REPO/${MODEL_NAME}" \
                STATIC_BATCH=${STATIC_BATCH} \
                PERF_CLIENT_PROTOCOL=${PROTOCOL} \
                INSTANCE_CNT=${INSTANCE_CNT} \
                CONCURRENCY=${CONCURRENCY} \
                ARCH=${ARCH} \
                bash -x run_test.sh
    done
done

#
# Test large static batch = 128 w/ 2 instances (Use batch size 64 on Jetson Xavier)
#
if [ "$ARCH" == "aarch64" ]; then
    STATIC_BATCH=64
else
    STATIC_BATCH=128
fi

INSTANCE_CNT=2
CONCURRENCY=4

# Create the TensorRT plan from Caffe model
rm -fr tensorrt_models && mkdir tensorrt_models
cp -r $REPODIR/caffe_models/trt_model_store/resnet50_plan tensorrt_models/${TRT_MODEL_NAME} && \
    (cd tensorrt_models/${TRT_MODEL_NAME} && \
            sed -i "s/^name:.*/name: \"${TRT_MODEL_NAME}\"/" config.pbtxt && \
            sed -i "s/max_batch_size:.*/max_batch_size: ${STATIC_BATCH}/" config.pbtxt) && \
    mkdir -p tensorrt_models/${TRT_MODEL_NAME}/1
$CAFFE2PLAN -h -b ${STATIC_BATCH} \
            -n prob -o tensorrt_models/${TRT_MODEL_NAME}/1/model.plan \
            $REPODIR/caffe_models/resnet50.prototxt $REPODIR/caffe_models/resnet50.caffemodel

for MODEL_NAME in $MODEL_NAMES; do
    for PROTOCOL in $PROTOCOLS; do
        REPO=`pwd`/tensorrt_models && [ "$MODEL_NAME" != "$TRT_MODEL_NAME" ] && \
            REPO=$REPODIR/perf_model_store
        FRAMEWORK=$(echo ${MODEL_NAME} | cut -d '_' -f 3)
        MODEL_NAME=${MODEL_NAME} \
                MODEL_FRAMEWORK=${FRAMEWORK} \
                MODEL_PATH="$REPO/${MODEL_NAME}" \
                STATIC_BATCH=${STATIC_BATCH} \
                PERF_CLIENT_PROTOCOL=${PROTOCOL} \
                INSTANCE_CNT=${INSTANCE_CNT} \
                CONCURRENCY=${CONCURRENCY} \
                ARCH=${ARCH} \
                bash -x run_test.sh
    done
done

for MODEL_NAME in $OPTIMIZED_MODEL_NAMES; do
    for PROTOCOL in $PROTOCOLS; do
        REPO=`pwd`/optimized_model_store
        FRAMEWORK=$(echo ${MODEL_NAME} | cut -d '_' -f 3,4)
        MODEL_NAME=${MODEL_NAME} \
                MODEL_FRAMEWORK=${FRAMEWORK} \
                MODEL_PATH="$REPO/${MODEL_NAME}" \
                STATIC_BATCH=${STATIC_BATCH} \
                PERF_CLIENT_PROTOCOL=${PROTOCOL} \
                INSTANCE_CNT=${INSTANCE_CNT} \
                CONCURRENCY=${CONCURRENCY} \
                ARCH=${ARCH} \
                bash -x run_test.sh
    done
done

# FIXME Disable the following due to
# https://jirasw.nvidia.com/browse/DLIS-2933.
#
# Needs this additional test configuration for comparing against TFS.
if [ "$ARCH" == "x86_64" ]; then
   MODEL_NAME=${TF_MODEL_NAME}
   REPO=$REPODIR/perf_model_store
   STATIC_BATCH=128
   INSTANCE_CNT=1
   CONCURRENCY=1
   FRAMEWORK=$(echo ${MODEL_NAME} | cut -d '_' -f 3)
   MODEL_NAME=${MODEL_NAME} \
       MODEL_FRAMEWORK=${FRAMEWORK} \
       MODEL_PATH="$REPO/${MODEL_NAME}" \
       STATIC_BATCH=${STATIC_BATCH} \
       PERF_CLIENT_PROTOCOL="grpc" \
       INSTANCE_CNT=${INSTANCE_CNT} \
       CONCURRENCY=${CONCURRENCY} \
       ARCH=${ARCH} \
       BACKEND_CONFIG=" --backend-config=tensorflow,version=1" \
       bash -x run_test.sh
fi
