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

SIMPLE_HEALTH_CLIENT_PY=../clients/simple_grpc_health_metadata.py
SIMPLE_INFER_CLIENT_PY=../clients/simple_grpc_infer_client.py
SIMPLE_ASYNC_INFER_CLIENT_PY=../clients/simple_grpc_async_infer_client.py
SIMPLE_STRING_INFER_CLIENT_PY=../clients/simple_grpc_string_infer_client.py
SIMPLE_STREAM_INFER_CLIENT_PY=../clients/simple_grpc_sequence_stream_infer_client.py
SIMPLE_SEQUENCE_INFER_CLIENT_PY=../clients/simple_grpc_sequence_sync_infer_client.py
SIMPLE_IMAGE_CLIENT_PY=../clients/image_client.py
SIMPLE_SHM_STRING_CLIENT_PY=../clients/simple_grpc_shm_string_client.py
SIMPLE_SHM_CLIENT_PY=../clients/simple_grpc_shm_client.py
SIMPLE_CUDASHM_CLIENT_PY=../clients/simple_grpc_cudashm_client.py
SIMPLE_MODEL_CONTROL_PY=../clients/simple_grpc_model_control.py
SIMPLE_REUSE_INFER_OBJECTS_CLIENT_PY=../clients/reuse_infer_objects_client.py
EXPLICIT_BYTE_CONTENT_CLIENT_PY=../clients/grpc_explicit_byte_content_client.py
EXPLICIT_INT_CONTENT_CLIENT_PY=../clients/grpc_explicit_int_content_client.py
EXPLICIT_INT8_CONTENT_CLIENT_PY=../clients/grpc_explicit_int8_content_client.py
GRPC_CLIENT_PY=../clients/grpc_client.py
GRPC_IMAGE_CLIENT_PY=../clients/grpc_image_client.py

SIMPLE_HEALTH_CLIENT=../clients/simple_grpc_health_metadata
SIMPLE_INFER_CLIENT=../clients/simple_grpc_infer_client
SIMPLE_STRING_INFER_CLIENT=../clients/simple_grpc_string_infer_client
SIMPLE_ASYNC_INFER_CLIENT=../clients/simple_grpc_async_infer_client
SIMPLE_MODEL_CONTROL=../clients/simple_grpc_model_control
SIMPLE_STREAM_INFER_CLIENT=../clients/simple_grpc_sequence_stream_infer_client
SIMPLE_SEQUENCE_INFER_CLIENT=../clients/simple_grpc_sequence_sync_infer_client
SIMPLE_SHM_CLIENT=../clients/simple_grpc_shm_client
SIMPLE_CUDASHM_CLIENT=../clients/simple_grpc_cudashm_client
SIMPLE_IMAGE_CLIENT=../clients/image_client
SIMPLE_REUSE_INFER_OBJECTS_CLIENT=../clients/reuse_infer_objects_client

rm -f *.log
rm -f *.log.*

set -e

CLIENT_LOG=`pwd`/client.log
DATADIR=`pwd`/models
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=$DATADIR"
source ../common/util.sh

run_server
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
        $GRPC_CLIENT_PY \
        $GRPC_IMAGE_CLIENT_PY \
        ; do
    BASE=$(basename -- $i)
    SUFFIX="${BASE%.*}"
    EXTRA_ARGS=""
    if [ $SUFFIX == "image_client" ]; then
        EXTRA_ARGS="-i grpc -u localhost:8001"
    fi
    if [[ ($SUFFIX == "image_client") || ($SUFFIX == "grpc_image_client") ]]; then
        python $i -m inception_graphdef -s INCEPTION -a -c 1 -b 1 $EXTRA_ARGS $IMAGE >> "${CLIENT_LOG}.async.${SUFFIX}" 2>&1
        if [ `grep -c VULTURE ${CLIENT_LOG}.async.${SUFFIX}` != "1" ]; then
            echo -e "\n***\n*** Failed. Expected 1 VULTURE results\n***"
            cat $CLIENT_LOG.async.${SUFFIX}
            RET=1
        fi
        python $i -m inception_graphdef -s INCEPTION -a --streaming -c 1 -b 1 $EXTRA_ARGS $IMAGE >> "${CLIENT_LOG}.streaming.${SUFFIX}" 2>&1
        if [ `grep -c VULTURE ${CLIENT_LOG}.streaming.${SUFFIX}` != "1" ]; then
            echo -e "\n***\n*** Failed. Expected 1 VULTURE results\n***"
            cat $CLIENT_LOG.streaming.${SUFFIX}
            RET=1
        fi
        python $i -m inception_graphdef -s INCEPTION -c 1 -b 1 $EXTRA_ARGS $IMAGE >> "${CLIENT_LOG}.${SUFFIX}" 2>&1
        if [ `grep -c VULTURE ${CLIENT_LOG}.${SUFFIX}` != "1" ]; then
            echo -e "\n***\n*** Failed. Expected 1 VULTURE results\n***"
            cat $CLIENT_LOG.${SUFFIX}
            RET=1
        fi
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

# Test while reusing the InferInput and InferRequestedOutput objects
$SIMPLE_REUSE_INFER_OBJECTS_CLIENT_PY -v -i grpc -u localhost:8001 >> ${CLIENT_LOG}.reuse 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.reuse
    RET=1
fi

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
    if [ $SUFFIX == "image_client" ]; then
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
    else
        $i -v -H test:1 >> ${CLIENT_LOG}.c++.${SUFFIX} 2>&1
        if [ $? -ne 0 ]; then
            cat ${CLIENT_LOG}.c++.${SUFFIX}
            RET=1
        fi
    fi
done

# Test while reusing the InferInput and InferRequestedOutput objects
$SIMPLE_REUSE_INFER_OBJECTS_CLIENT -v -i grpc -u localhost:8001 >> ${CLIENT_LOG}.c++.reuse 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.c++.reuse
    RET=1
fi

set -e
kill $SERVER_PID
wait $SERVER_PID

export GRPC_TRACE=compression, channel
export GRPC_VERBOSITY=DEBUG
SERVER_ARGS="--model-repository=$DATADIR --grpc-infer-response-compression-level=high"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

$SIMPLE_INFER_CLIENT -v -C deflate>> ${CLIENT_LOG}.c++.compress 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.c++.compress
    RET=1
fi
if [ $(cat ${CLIENT_LOG}.c++.compress | grep "Compressed\[deflate\]" | wc -l) -eq 0 ]; then
    cat ${CLIENT_LOG}.c++.compress
    RET=1
fi

python $SIMPLE_INFER_CLIENT_PY -v -C deflate>> ${CLIENT_LOG}.compress 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.compress
    RET=1
fi
if [ $(cat ${CLIENT_LOG}.compress | grep "Compressed\[deflate\]" | wc -l) -eq 0 ]; then
    cat ${CLIENT_LOG}.compress
    RET=1
fi

set -e
kill $SERVER_PID
wait $SERVER_PID

unset GRPC_TRACE
unset GRPC_VERBOSITY

SERVER_ARGS="--model-repository=$DATADIR --model-control-mode=explicit"
run_server
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

SERVER_ARGS="--model-repository=$DATADIR --model-control-mode=explicit"
run_server
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

# Test with dynamic sequence models
SERVER_ARGS="--model-repository=`pwd`/models"
SERVER_LOG="./inference_server_dyna.log"
CLIENT_LOG="./client_dyna.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
set +e

for i in \
    $SIMPLE_STREAM_INFER_CLIENT_PY \
    $SIMPLE_SEQUENCE_INFER_CLIENT_PY \
    $SIMPLE_STREAM_INFER_CLIENT \
    $SIMPLE_SEQUENCE_INFER_CLIENT; do

    $i -v -d >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi
done

set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
