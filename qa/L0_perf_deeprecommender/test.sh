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

REPODIR=/data/inferenceserver/${REPO_VERSION}
TRTEXEC=/usr/src/tensorrt/bin/trtexec
MODEL="deeprecommender"

rm -f *.log *.serverlog *.csv *.metrics *.tjson *.json

#
# Test minimum latency
#
STATIC_BATCH=1
DYNAMIC_BATCH=1
INSTANCE_CNT=1

# Copy TF Model
rm -fr tfmodels && mkdir -p tfmodels && \
    cp -r $REPODIR/perf_model_store/deeprecommender_graphdef tfmodels/.

# Create the TensorRT plan from TF
rm -fr tensorrt_models && mkdir tensorrt_models
    cp -r $REPODIR/perf_model_store/deeprecommender_graphdef tensorrt_models/deeprecommender_plan && \
    (cd tensorrt_models/deeprecommender_plan && \
        sed -i "s/^name:.*/name: \"deeprecommender_plan\"/" config.pbtxt && \
        sed -i "s/tensorflow_graphdef/tensorrt_plan/" config.pbtxt && \
        sed -i "s/\[17736\]/\[17736,1,1\]/" config.pbtxt)

$TRTEXEC --uff=tfmodels/deeprecommender_graphdef/deeprecommender_graphdef.uff \
         --uffInput=Placeholder,1,1,17736\
         --batch=${STATIC_BATCH} --output=fc5/Relu --verbose \
         --saveEngine=tensorrt_models/deeprecommender_plan/0/model.plan

# Create the TFTRT plan from TF
rm -fr tftrt_models && mkdir tftrt_models
    cp -r $REPODIR/perf_model_store/deeprecommender_graphdef tftrt_models/deeprecommender_graphdef_trt && \
    (cd tftrt_models/deeprecommender_graphdef_trt && \
        sed -i "s/^name:.*/name: \"deeprecommender_graphdef_trt\"/" config.pbtxt && \
        echo "optimization { execution_accelerators {" >> config.pbtxt
        echo "gpu_execution_accelerator : [ {" >> config.pbtxt
        echo "name : \"tensorrt\" " >> config.pbtxt
        echo "} ]" >> config.pbtxt
        echo "}}" >> config.pbtxt)

# Tests with each model
for FRAMEWORK in graphdef plan graphdef_trt; do
    MODEL_NAME=${MODEL}_${FRAMEWORK}
    if [ "$FRAMEWORK" == "graphdef" ]; then
        REPO=`pwd`/tfmodels
    elif [ "$FRAMEWORK" == "plan" ]; then
        REPO=`pwd`/tensorrt_models
    else
        REPO=`pwd`/tftrt_models
    fi
    MODEL_NAME=${MODEL_NAME} \
            MODEL_FRAMEWORK=${FRAMEWORK} \
            MODEL_PATH="$REPO/${MODEL_NAME}" \
            STATIC_BATCH_SIZES=${STATIC_BATCH} \
            DYNAMIC_BATCH_SIZES=${DYNAMIC_BATCH} \
            INSTANCE_COUNTS=${INSTANCE_CNT} \
            bash -x run_test.sh
done

#
# Test large static batch = 1024 w/ 2 instances
#
STATIC_BATCH=1024
DYNAMIC_BATCH=1
INSTANCE_CNT=2

# Copy TF Model
rm -fr models && mkdir -p models && \
    cp -r $REPODIR/perf_model_store/deeprecommender_graphdef tfmodels/.

# Create the TensorRT plan from TF
rm -fr tensorrt_models && mkdir tensorrt_models
    cp -r $REPODIR/perf_model_store/deeprecommender_graphdef tensorrt_models/deeprecommender_plan && \
    (cd tensorrt_models/deeprecommender_plan && \
        sed -i "s/^name:.*/name: \"deeprecommender_plan\"/" config.pbtxt && \
        sed -i "s/tensorflow_graphdef/tensorrt_plan/" config.pbtxt && \
        sed -i "s/\[17736\]/\[17736,1,1\]/" config.pbtxt)

$TRTEXEC --uff=tfmodels/deeprecommender_graphdef/deeprecommender_graphdef.uff \
         --uffInput=Placeholder,1,1,17736\
         --batch=${STATIC_BATCH} --output=fc5/Relu --verbose \
         --saveEngine=tensorrt_models/deeprecommender_plan/0/model.plan

# Create the TFTRT plan from TF
rm -fr tftrt_models && mkdir tftrt_models
    cp -r $REPODIR/perf_model_store/deeprecommender_graphdef tftrt_models/deeprecommender_graphdef_trt && \
    (cd tftrt_models/deeprecommender_graphdef_trt && \
        sed -i "s/^name:.*/name: \"deeprecommender_graphdef_trt\"/" config.pbtxt && \
        echo "optimization { execution_accelerators {" >> config.pbtxt
        echo "gpu_execution_accelerator : [ {" >> config.pbtxt
        echo "name : \"tensorrt\" " >> config.pbtxt
        echo "} ]" >> config.pbtxt
        echo "}}" >> config.pbtxt)

# Tests with each model
for FRAMEWORK in graphdef plan graphdef_trt; do
    MODEL_NAME=${MODEL}_${FRAMEWORK}
    if [ "$FRAMEWORK" == "graphdef" ]; then
        REPO=`pwd`/tfmodels
    elif [ "$FRAMEWORK" == "plan" ]; then
        REPO=`pwd`/tensorrt_models
    else
        REPO=`pwd`/tftrt_models
    fi
    MODEL_NAME=${MODEL_NAME} \
            MODEL_FRAMEWORK=${FRAMEWORK} \
            MODEL_PATH="$REPO/${MODEL_NAME}" \
            STATIC_BATCH_SIZES=${STATIC_BATCH} \
            DYNAMIC_BATCH_SIZES=${DYNAMIC_BATCH} \
            INSTANCE_COUNTS=${INSTANCE_CNT} \
            bash -x run_test.sh
done

#
# Test dynamic batcher 64 w/ 2 instances
#
STATIC_BATCH=1
DYNAMIC_BATCH=64
INSTANCE_CNT=2

# Copy TF Model
rm -fr models && mkdir -p models && \
    cp -r $REPODIR/perf_model_store/deeprecommender_graphdef tfmodels/.

# Create the TensorRT plan from TF
rm -fr tensorrt_models && mkdir tensorrt_models
    cp -r $REPODIR/perf_model_store/deeprecommender_graphdef tensorrt_models/deeprecommender_plan && \
    (cd tensorrt_models/deeprecommender_plan && \
        sed -i "s/^name:.*/name: \"deeprecommender_plan\"/" config.pbtxt && \
        sed -i "s/tensorflow_graphdef/tensorrt_plan/" config.pbtxt && \
        sed -i "s/\[17736\]/\[17736,1,1\]/" config.pbtxt)

$TRTEXEC --uff=tfmodels/deeprecommender_graphdef/deeprecommender_graphdef.uff \
         --uffInput=Placeholder,1,1,17736\
         --batch=${DYNAMIC_BATCH} --output=fc5/Relu --verbose \
         --saveEngine=tensorrt_models/deeprecommender_plan/0/model.plan

# Create the TFTRT plan from TF
rm -fr tftrt_models && mkdir tftrt_models
    cp -r $REPODIR/perf_model_store/deeprecommender_graphdef tftrt_models/deeprecommender_graphdef_trt && \
    (cd tftrt_models/deeprecommender_graphdef_trt && \
        sed -i "s/^name:.*/name: \"deeprecommender_graphdef_trt\"/" config.pbtxt && \
        echo "optimization { execution_accelerators {" >> config.pbtxt
        echo "gpu_execution_accelerator : [ {" >> config.pbtxt
        echo "name : \"tensorrt\" " >> config.pbtxt
        echo "} ]" >> config.pbtxt
        echo "}}" >> config.pbtxt)

# Tests with each model
for FRAMEWORK in graphdef plan graphdef_trt; do
    MODEL_NAME=${MODEL}_${FRAMEWORK}
    if [ "$FRAMEWORK" == "graphdef" ]; then
        REPO=`pwd`/tfmodels
    elif [ "$FRAMEWORK" == "plan" ]; then
        REPO=`pwd`/tensorrt_models
    else
        REPO=`pwd`/tftrt_models
    fi
    MODEL_NAME=${MODEL_NAME} \
            MODEL_FRAMEWORK=${FRAMEWORK} \
            MODEL_PATH="$REPO/${MODEL_NAME}" \
            STATIC_BATCH_SIZES=${STATIC_BATCH} \
            DYNAMIC_BATCH_SIZES=${DYNAMIC_BATCH} \
            INSTANCE_COUNTS=${INSTANCE_CNT} \
            bash -x run_test.sh
done
