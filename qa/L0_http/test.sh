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

SIMPLE_HEALTH_CLIENT_PY=../clients/simple_http_health_metadata.py
SIMPLE_INFER_CLIENT_PY=../clients/simple_http_infer_client.py
SIMPLE_ASYNC_INFER_CLIENT_PY=../clients/simple_http_async_infer_client.py
SIMPLE_STRING_INFER_CLIENT_PY=../clients/simple_http_string_infer_client.py
SIMPLE_IMAGE_CLIENT_PY=../clients/image_client.py
SIMPLE_SHM_STRING_CLIENT_PY=../clients/simple_http_shm_string_client.py
SIMPLE_SHM_CLIENT_PY=../clients/simple_http_shm_client.py
SIMPLE_CUDASHM_CLIENT_PY=../clients/simple_http_cudashm_client.py
SIMPLE_MODEL_CONTROL_PY=../clients/simple_http_model_control.py
SIMPLE_SEQUENCE_INFER_CLIENT_PY=../clients/simple_http_sequence_sync_infer_client.py
SIMPLE_REUSE_INFER_OBJECTS_CLIENT_PY=../clients/reuse_infer_objects_client.py

SIMPLE_HEALTH_CLIENT=../clients/simple_http_health_metadata
SIMPLE_INFER_CLIENT=../clients/simple_http_infer_client
SIMPLE_STRING_INFER_CLIENT=../clients/simple_http_string_infer_client
SIMPLE_ASYNC_INFER_CLIENT=../clients/simple_http_async_infer_client
SIMPLE_MODEL_CONTROL=../clients/simple_http_model_control
SIMPLE_SEQUENCE_INFER_CLIENT=../clients/simple_http_sequence_sync_infer_client
SIMPLE_SHM_CLIENT=../clients/simple_http_shm_client
SIMPLE_CUDASHM_CLIENT=../clients/simple_http_cudashm_client
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

# Test health
python $SIMPLE_HEALTH_CLIENT_PY -v >> ${CLIENT_LOG}.health 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.health
    RET=1
fi

IMAGE=../images/vulture.jpeg
for i in \
        $SIMPLE_INFER_CLIENT_PY \
        $SIMPLE_ASYNC_INFER_CLIENT_PY \
        $SIMPLE_IMAGE_CLIENT_PY \
        $SIMPLE_ENSEMBLE_IMAGE_CLIENT_PY \
        $SIMPLE_SHM_STRING_CLIENT_PY \
        $SIMPLE_SHM_CLIENT_PY \
        $SIMPLE_CUDASHM_CLIENT_PY \
        $SIMPLE_STRING_INFER_CLIENT_PY \
        $SIMPLE_SEQUENCE_INFER_CLIENT_PY \
        ; do
    BASE=$(basename -- $i)
    SUFFIX="${BASE%.*}"
    if [ $SUFFIX == "image_client" ]; then
        python $i -m inception_graphdef -s INCEPTION -a -c 1 -b 1 $IMAGE >> "${CLIENT_LOG}.async.${SUFFIX}" 2>&1
        if [ `grep -c VULTURE ${CLIENT_LOG}.async.${SUFFIX}` != "1" ]; then
            echo -e "\n***\n*** Failed. Expected 1 VULTURE results\n***"
            cat $CLIENT_LOG.async.${SUFFIX}
            RET=1
        fi
        python $i -m inception_graphdef -s INCEPTION -c 1 -b 1 $IMAGE >> "${CLIENT_LOG}.${SUFFIX}" 2>&1
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
done

# Test while reusing the InferInput and InferRequestedOutput objects
$SIMPLE_REUSE_INFER_OBJECTS_CLIENT_PY -v >> ${CLIENT_LOG}.reuse 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.reuse
    RET=1
fi

# Test with the base path in url.
$SIMPLE_INFER_CLIENT_PY -u localhost:8000/base_path -v >> ${CLIENT_LOG}.base_path_url 2>&1
if [ $? -eq 0 ]; then
    cat ${CLIENT_LOG}.base_path_url
    RET=1
fi
if [ $(cat ${CLIENT_LOG}.base_path_url | grep "POST /base_path/v2/models/simple/infer" | wc -l) -eq 0 ]; then
    cat ${CLIENT_LOG}.base_path_url
    RET=1
fi

for i in \
   $SIMPLE_INFER_CLIENT \
   $SIMPLE_STRING_INFER_CLIENT \
   $SIMPLE_ASYNC_INFER_CLIENT \
   $SIMPLE_HEALTH_CLIENT \
   $SIMPLE_SHM_CLIENT \
   $SIMPLE_CUDASHM_CLIENT \
   $SIMPLE_SEQUENCE_INFER_CLIENT \
   ; do
   BASE=$(basename -- $i)
   SUFFIX="${BASE%.*}"

    $i -v -H test:1 >> ${CLIENT_LOG}.c++.${SUFFIX} 2>&1
    if [ $? -ne 0 ]; then
        cat ${CLIENT_LOG}.c++.${SUFFIX}
        RET=1
    fi
done

# Test while reusing the InferInput and InferRequestedOutput objects
$SIMPLE_REUSE_INFER_OBJECTS_CLIENT -v >> ${CLIENT_LOG}.c++.reuse 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.c++.reuse
    RET=1
fi

# Test with the base path in url.
$SIMPLE_INFER_CLIENT -u localhost:8000/base_path -v >> ${CLIENT_LOG}.c++.base_path_url 2>&1
if [ $? -eq 0 ]; then
    cat ${CLIENT_LOG}.c++.base_path_url
    RET=1
fi
if [ $(cat ${CLIENT_LOG}.c++.base_path_url | grep "POST /base_path/v2/models/simple/infer" | wc -l) -eq 0 ]; then
    cat ${CLIENT_LOG}.c++.base_path_url
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
    $SIMPLE_SEQUENCE_INFER_CLIENT \
    $SIMPLE_SEQUENCE_INFER_CLIENT_PY; do

    $i -v -d >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi
done

set -e

kill $SERVER_PID
wait $SERVER_PID

# Test combinations of binary and JSON data
SERVER_ARGS="--model-repository=`pwd`/models"
SERVER_LOG="./inference_server_binaryjson.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# no parameters, no outputs == json output
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -d'{"inputs":[{"name":"INPUT0","datatype":"INT32","shape":[1,16],"data":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]},{"name":"INPUT1","datatype":"INT32","shape":[1,16],"data":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}]}' localhost:8000/v2/models/simple/infer`
set -e
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ `grep -c "\[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32\]" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\]" ./curl.out` != "1" ]; then
    RET=1
fi

# binary_data=true on INPUT0, binary_data=false on INPUT1
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -d'{"inputs":[{"name":"INPUT0","datatype":"INT32","shape":[1,16],"data":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]},{"name":"INPUT1","datatype":"INT32","shape":[1,16],"data":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}],"outputs":[{"name":"OUTPUT0","parameters":{"binary_data":true}},{"name":"OUTPUT1","parameters":{"binary_data":false}}]}' localhost:8000/v2/models/simple/infer`
set -e
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ `grep -c "\[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32\]" ./curl.out` != "0" ]; then
    RET=1
fi
if [ `grep -c "\[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\]" ./curl.out` != "1" ]; then
    RET=1
fi

# binary_data=true on INPUT0, binary_data not given on INPUT1
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -d'{"inputs":[{"name":"INPUT0","datatype":"INT32","shape":[1,16],"data":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]},{"name":"INPUT1","datatype":"INT32","shape":[1,16],"data":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}],"outputs":[{"name":"OUTPUT0","parameters":{"binary_data":true}},{"name":"OUTPUT1"}]}' localhost:8000/v2/models/simple/infer`
set -e
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ `grep -c "\[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32\]" ./curl.out` != "0" ]; then
    RET=1
fi
if [ `grep -c "\[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\]" ./curl.out` != "1" ]; then
    RET=1
fi

# binary_data_output=true, no outputs requested
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -d'{"parameters":{"binary_data_output":true},"inputs":[{"name":"INPUT0","datatype":"INT32","shape":[1,16],"data":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]},{"name":"INPUT1","datatype":"INT32","shape":[1,16],"data":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}]}' localhost:8000/v2/models/simple/infer`
set -e
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ `grep -c "\[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32\]" ./curl.out` != "0" ]; then
    RET=1
fi
if [ `grep -c "\[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\]" ./curl.out` != "0" ]; then
    RET=1
fi

# binary_data_output=true
# binary_data=false on INPUT0, binary_data not given on INPUT1
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -d'{"parameters":{"binary_data_output":true},"inputs":[{"name":"INPUT0","datatype":"INT32","shape":[1,16],"data":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]},{"name":"INPUT1","datatype":"INT32","shape":[1,16],"data":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}],"outputs":[{"name":"OUTPUT0","parameters":{"binary_data":false}},{"name":"OUTPUT1"}]}' localhost:8000/v2/models/simple/infer`
set -e
if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ `grep -c "\[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32\]" ./curl.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\]" ./curl.out` != "1" ]; then
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID


if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
