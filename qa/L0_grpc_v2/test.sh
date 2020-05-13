#!/bin/bash
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

export CUDA_VISIBLE_DEVICES=0

RET=0

SIMPLE_HEALTH_CLIENT_PY=../clients/simple_grpc_v2_health_metadata.py
SIMPLE_INFER_CLIENT_PY=../clients/simple_grpc_v2_infer_client.py
SIMPLE_ASYNC_INFER_CLIENT_PY=../clients/simple_grpc_v2_async_infer_client.py
SIMPLE_STRING_INFER_CLIENT_PY=../clients/simple_grpc_v2_string_infer_client.py
SIMPLE_STREAM_INFER_CLIENT_PY=../clients/simple_grpc_v2_sequence_stream_infer_client.py
SIMPLE_SEQUENCE_INFER_CLIENT_PY=../clients/simple_grpc_v2_sequence_sync_infer_client.py
SIMPLE_IMAGE_CLIENT_PY=../clients/v2_image_client.py
SIMPLE_ENSEMBLE_IMAGE_CLIENT_PY=../clients/v2_ensemble_image_client.py
SIMPLE_SHM_STRING_CLIENT_PY=../clients/simple_grpc_v2_shm_string_client.py
SIMPLE_SHM_CLIENT_PY=../clients/simple_grpc_v2_shm_client.py
SIMPLE_CUDASHM_CLIENT_PY=../clients/simple_grpc_v2_cudashm_client.py
SIMPLE_MODEL_CONTROL_PY=../clients/simple_grpc_v2_model_control.py
EXPLICIT_BYTE_CONTENT_CLIENT_PY=../clients/grpc_v2_explicit_byte_content_client.py
EXPLICIT_INT_CONTENT_CLIENT_PY=../clients/grpc_v2_explicit_int_content_client.py
EXPLICIT_INT8_CONTENT_CLIENT_PY=../clients/grpc_v2_explicit_int8_content_client.py
GRPC_V2_CLIENT_PY=../clients/grpc_v2_client.py
GRPC_IMAGE_CLIENT_PY=../clients/grpc_v2_image_client.py

SIMPLE_HEALTH_CLIENT=../clients/simple_grpc_v2_health_metadata
SIMPLE_INFER_CLIENT=../clients/simple_grpc_v2_infer_client
SIMPLE_STRING_INFER_CLIENT=../clients/simple_grpc_v2_string_infer_client
SIMPLE_ASYNC_INFER_CLIENT=../clients/simple_grpc_v2_async_infer_client
SIMPLE_MODEL_CONTROL=../clients/simple_grpc_v2_model_control
SIMPLE_STREAM_INFER_CLIENT=../clients/simple_grpc_v2_sequence_stream_infer_client
SIMPLE_SEQUENCE_INFER_CLIENT=../clients/simple_grpc_v2_sequence_sync_infer_client
SIMPLE_SHM_CLIENT=../clients/simple_grpc_v2_shm_client
SIMPLE_CUDASHM_CLIENT=../clients/simple_grpc_v2_cudashm_client
SIMPLE_IMAGE_CLIENT=../clients/v2_image_client
SIMPLE_ENSEMBLE_IMAGE_CLIENT=../clients/v2_ensemble_image_client

rm -f *.log
rm -f *.log.*

# Get the TensorFlow inception model
mkdir -p models/inception_graphdef/1
wget -O /tmp/inception_v3_2016_08_28_frozen.pb.tar.gz \
     https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz
(cd /tmp && tar xzf inception_v3_2016_08_28_frozen.pb.tar.gz)
mv /tmp/inception_v3_2016_08_28_frozen.pb models/inception_graphdef/1/model.graphdef
cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/graphdef_int8_int32_int32 models/
cp -r /data/inferenceserver/${REPO_VERSION}/tf_model_store/resnet_v1_50_graphdef models/

# Create model repository layout for ensemble image classification
cp -r ../L0_docs/docs/examples/ensemble_model_repository/image_preprocess_nchw_3x224x224_inception models/ && \
cp -r ../L0_docs/docs/examples/ensemble_model_repository/preprocess_resnet50_ensemble models/ && \
mkdir -p models/image_preprocess_nchw_3x224x224_inception/1 && \
mkdir -p models/preprocess_resnet50_ensemble/1 && \

# Obtain actual models
cp -r /data/inferenceserver/${REPO_VERSION}/c2_model_store/resnet50_netdef models/ && \
    cp ../L0_custom_image_preprocess/models/image_preprocess_nhwc_224x224x3/1/libimagepreprocess.so \
        models/image_preprocess_nchw_3x224x224_inception/1/.

CLIENT_LOG=`pwd`/client.log
DATADIR=`pwd`/models
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=$DATADIR --api-version 2"
source ../common/util.sh

run_server_v2
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

python $SIMPLE_HEALTH_CLIENT_PY -v >> ${CLIENT_LOG}.health 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.health
    RET=1
fi

IMAGE=../images/vulture.jpeg
for i in \
        $SIMPLE_INFER_CLIENT_PY \
        $SIMPLE_ASYNC_INFER_CLIENT_PY \
        $SIMPLE_STRING_INFER_CLIENT_PY \
        $SIMPLE_IMAGE_CLIENT_PY \
        $SIMPLE_ENSEMBLE_IMAGE_CLIENT_PY \
        $SIMPLE_STREAM_INFER_CLIENT_PY \
        $SIMPLE_SEQUENCE_INFER_CLIENT_PY \
        $SIMPLE_SHM_STRING_CLIENT_PY \
        $SIMPLE_SHM_CLIENT_PY \
        $SIMPLE_CUDASHM_CLIENT_PY \
        $EXPLICIT_BYTE_CONTENT_CLIENT_PY \
        $EXPLICIT_INT_CONTENT_CLIENT_PY \
        $EXPLICIT_INT8_CONTENT_CLIENT_PY \
        $GRPC_V2_CLIENT_PY \
        $GRPC_IMAGE_CLIENT_PY \
        ; do
    BASE=$(basename -- $i)
    SUFFIX="${BASE%.*}"
    if [ $SUFFIX == "grpc_v2_image_client" ]; then
        python $i -m inception_graphdef -s INCEPTION -c 1 -b 1 $IMAGE >> "${CLIENT_LOG}.${SUFFIX}" 2>&1
        if [ `grep -c VULTURE ${CLIENT_LOG}.${SUFFIX}` != "1" ]; then
            echo -e "\n***\n*** Failed. Expected 1 VULTURE results\n***"
            cat $CLIENT_LOG.${SUFFIX}
            RET=1
        fi
    elif [ $SUFFIX == "v2_image_client" ]; then
        python $i -m inception_graphdef -s INCEPTION -c 1 -b 1 -i grpc -u localhost:8001 $IMAGE >> "${CLIENT_LOG}.${SUFFIX}" 2>&1
        if [ `grep -c VULTURE ${CLIENT_LOG}.${SUFFIX}` != "1" ]; then
            echo -e "\n***\n*** Failed. Expected 1 VULTURE results\n***"
            cat $CLIENT_LOG.${SUFFIX}
            RET=1
        fi
    elif [ $SUFFIX == "v2_ensemble_image_client" ]; then
        python $i -c 1 -i grpc -u localhost:8001 ../images >> "${CLIENT_LOG}.${SUFFIX}" 2>&1
        for result in "SPORTS CAR" "COFFEE MUG" "VULTURE"; do
            if [ `grep -c "$result" ${CLIENT_LOG}.${SUFFIX}` != "1" ]; then
                echo -e "\n***\n*** Failed. Expected 1 $result result\n***"
                RET=1
            fi
        done
    else
        python $i -v >> "${CLIENT_LOG}.${SUFFIX}" 2>&1
    fi

    if [ $? -ne 0 ]; then
        cat "${CLIENT_LOG}.${SUFFIX}"
        RET=1
    fi

    if [ $(cat "${CLIENT_LOG}.${SUFFIX}" | grep "PASS" | wc -l) -ne 1 ]; then
        cat "${CLIENT_LOG}.${SUFFIX}"
        RET=1
    fi
done

for i in \
   $SIMPLE_INFER_CLIENT \
   $SIMPLE_STRING_INFER_CLIENT \
   $SIMPLE_ASYNC_INFER_CLIENT \
   $SIMPLE_HEALTH_CLIENT \
   $SIMPLE_STREAM_INFER_CLIENT \
   $SIMPLE_SEQUENCE_INFER_CLIENT \
   $SIMPLE_SHM_CLIENT \
   $SIMPLE_CUDASHM_CLIENT \
   $SIMPLE_IMAGE_CLIENT \
   $SIMPLE_ENSEMBLE_IMAGE_CLIENT \
   ; do
   BASE=$(basename -- $i)
   SUFFIX="${BASE%.*}"
    if [ $SUFFIX == "v2_image_client" ]; then
        $i -m inception_graphdef -s INCEPTION -a -c 1 -b 1 -i grpc -u localhost:8001 $IMAGE >> "${CLIENT_LOG}.c++.async.${SUFFIX}" 2>&1
        if [ `grep -c VULTURE ${CLIENT_LOG}.c++.async.${SUFFIX}` != "1" ]; then
            echo -e "\n***\n*** Failed. Expected 1 VULTURE results\n***"
            cat $CLIENT_LOG.c++.${SUFFIX}
            RET=1
        fi
        $i -m inception_graphdef -s INCEPTION -a --streaming -c 1 -b 1 -i grpc -u localhost:8001 $IMAGE >> "${CLIENT_LOG}.c++.streaming.${SUFFIX}" 2>&1
        if [ `grep -c VULTURE ${CLIENT_LOG}.c++.streaming.${SUFFIX}` != "1" ]; then
            echo -e "\n***\n*** Failed. Expected 1 VULTURE results\n***"
            cat $CLIENT_LOG.c++.${SUFFIX}
            RET=1
        fi
        $i -m inception_graphdef -s INCEPTION -c 1 -b 1 -i grpc -u localhost:8001 $IMAGE >> "${CLIENT_LOG}.c++.${SUFFIX}" 2>&1
        if [ `grep -c VULTURE ${CLIENT_LOG}.c++.${SUFFIX}` != "1" ]; then
            echo -e "\n***\n*** Failed. Expected 1 VULTURE results\n***"
            cat $CLIENT_LOG.c++.${SUFFIX}
            RET=1
        fi
    elif [ $SUFFIX == "v2_ensemble_image_client" ]; then
        $i -c 1 -i grpc -u localhost:8001 ../images >> "${CLIENT_LOG}.c++.${SUFFIX}" 2>&1
        for result in "SPORTS CAR" "COFFEE MUG" "VULTURE"; do
            if [ `grep -c "$result" ${CLIENT_LOG}.c++.${SUFFIX}` != "1" ]; then
                echo -e "\n***\n*** Failed. Expected 1 $result result\n***"
                RET=1
            fi
        done
    else
        $i -v -H test:1 >> ${CLIENT_LOG}.c++.${SUFFIX} 2>&1
        if [ $? -ne 0 ]; then
            cat ${CLIENT_LOG}.c++.${SUFFIX}
            RET=1
        fi
    fi
done

set -e
kill $SERVER_PID
wait $SERVER_PID

SERVER_ARGS="--model-repository=$DATADIR --model-control-mode=explicit --api-version 2"
run_server_v2
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
# Test Model Control API
python $SIMPLE_MODEL_CONTROL_PY -v >> ${CLIENT_LOG}.model_control 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.model_control
    RET=1
fi

if [ $(cat ${CLIENT_LOG}.model_control | grep "PASS" | wc -l) -ne 1 ]; then
    cat ${CLIENT_LOG}.model_control
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

SERVER_ARGS="--model-repository=$DATADIR --model-control-mode=explicit --api-version 2"
run_server_v2
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
# Test Model Control API
$SIMPLE_MODEL_CONTROL -v >> ${CLIENT_LOG}.c++.model_control 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.c++.model_control
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
