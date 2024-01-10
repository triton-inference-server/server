#!/bin/bash
# Copyright 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

export CUDA_VISIBLE_DEVICES=0

RET=0

CLIENT_PLUGIN_TEST="./http_client_plugin_test.py"
BASIC_AUTH_TEST="./http_basic_auth_test.py"
RESTRICTED_API_TEST="./http_restricted_api_test.py"
NGINX_CONF="./nginx.conf"
# On windows the paths invoked by the script (running in WSL) must use
# /mnt/c when needed but the paths on the tritonserver command-line
# must be C:/ style.
if [[ "$(< /proc/sys/kernel/osrelease)" == *microsoft* ]]; then
    SDKDIR=${SDKDIR:=C:/sdk}
    MODELDIR=${MODELDIR:=C:/models}
    DATADIR=${DATADIR:="/mnt/c/data/inferenceserver/${REPO_VERSION}"}
    BACKEND_DIR=${BACKEND_DIR:=C:/tritonserver/backends}
    SERVER=${SERVER:=/mnt/c/tritonserver/bin/tritonserver.exe}

    SIMPLE_AIO_INFER_CLIENT_PY=${SDKDIR}/python/simple_http_aio_infer_client.py
    SIMPLE_HEALTH_CLIENT_PY=${SDKDIR}/python/simple_http_health_metadata.py
    SIMPLE_INFER_CLIENT_PY=${SDKDIR}/python/simple_http_infer_client.py
    SIMPLE_ASYNC_INFER_CLIENT_PY=${SDKDIR}/python/simple_http_async_infer_client.py
    SIMPLE_STRING_INFER_CLIENT_PY=${SDKDIR}/python/simple_http_string_infer_client.py
    SIMPLE_IMAGE_CLIENT_PY=${SDKDIR}/python/image_client.py
    # SIMPLE_ENSEMBLE_IMAGE_CLIENT_PY=${SDKDIR}/python/ensemble_image_client.py
    SIMPLE_SHM_STRING_CLIENT_PY=${SDKDIR}/python/simple_http_shm_string_client.py
    SIMPLE_SHM_CLIENT_PY=${SDKDIR}/python/simple_http_shm_client.py
    SIMPLE_CUDASHM_CLIENT_PY=${SDKDIR}/python/simple_http_cudashm_client.py
    SIMPLE_MODEL_CONTROL_PY=${SDKDIR}/python/simple_http_model_control.py
    SIMPLE_SEQUENCE_INFER_CLIENT_PY=${SDKDIR}/python/simple_http_sequence_sync_infer_client.py
    SIMPLE_REUSE_INFER_OBJECTS_CLIENT_PY=${SDKDIR}/python/reuse_infer_objects_client.py

    SIMPLE_HEALTH_CLIENT=${SDKDIR}/python/simple_http_health_metadata
    SIMPLE_INFER_CLIENT=${SDKDIR}/python/simple_http_infer_client
    SIMPLE_STRING_INFER_CLIENT=${SDKDIR}/python/simple_http_string_infer_client
    SIMPLE_ASYNC_INFER_CLIENT=${SDKDIR}/python/simple_http_async_infer_client
    SIMPLE_MODEL_CONTROL=${SDKDIR}/python/simple_http_model_control
    SIMPLE_SEQUENCE_INFER_CLIENT=${SDKDIR}/python/simple_http_sequence_sync_infer_client
    SIMPLE_SHM_CLIENT=${SDKDIR}/python/simple_http_shm_client
    SIMPLE_CUDASHM_CLIENT=${SDKDIR}/python/simple_http_cudashm_client
    SIMPLE_REUSE_INFER_OBJECTS_CLIENT=${SDKDIR}/python/reuse_infer_objects_client
    # [FIXME] point to proper client
    CC_UNIT_TEST=${SDKDIR}/python/cc_client_test
else
    MODELDIR=${MODELDIR:=`pwd`/models}
    DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
    TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
    SERVER=${TRITON_DIR}/bin/tritonserver
    BACKEND_DIR=${TRITON_DIR}/backends

    SIMPLE_AIO_INFER_CLIENT_PY=../clients/simple_http_aio_infer_client.py
    SIMPLE_HEALTH_CLIENT_PY=../clients/simple_http_health_metadata.py
    SIMPLE_INFER_CLIENT_PY=../clients/simple_http_infer_client.py
    SIMPLE_ASYNC_INFER_CLIENT_PY=../clients/simple_http_async_infer_client.py
    SIMPLE_STRING_INFER_CLIENT_PY=../clients/simple_http_string_infer_client.py
    SIMPLE_IMAGE_CLIENT_PY=../clients/image_client.py
    # SIMPLE_ENSEMBLE_IMAGE_CLIENT_PY=../clients/ensemble_image_client.py
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
    CC_UNIT_TEST=../clients/cc_client_test
fi

# Add string_dyna_sequence model to repo
cp -r ${MODELDIR}/simple_dyna_sequence ${MODELDIR}/simple_string_dyna_sequence
sed -i "s/simple_dyna_sequence/simple_string_dyna_sequence/g" ${MODELDIR}/simple_string_dyna_sequence/config.pbtxt
sed -i "s/^platform: .*/backend: \"dyna_sequence\"/g" ${MODELDIR}/simple_string_dyna_sequence/config.pbtxt
sed -i "/CONTROL_SEQUENCE_CORRID/{n;s/data_type:.*/data_type: TYPE_STRING/}" ${MODELDIR}/simple_string_dyna_sequence/config.pbtxt
rm -f ${MODELDIR}/simple_string_dyna_sequence/1/model.graphdef
cp ../custom_models/custom_dyna_sequence_int32/1/libtriton_dyna_sequence.so ${MODELDIR}/simple_string_dyna_sequence/1/

rm -f *.log
rm -f *.log.*

set -e

CLIENT_LOG=`pwd`/client.log
SERVER_ARGS="--backend-directory=${BACKEND_DIR} --model-repository=${MODELDIR}"
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
        $SIMPLE_AIO_INFER_CLIENT_PY \
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
    # elif [ $SUFFIX == "ensemble_image_client" ]; then
    #     python $i -c 1 ../images >> "${CLIENT_LOG}.${SUFFIX}" 2>&1
    #     for result in "SPORTS CAR" "COFFEE MUG" "VULTURE"; do
    #         if [ `grep -c "$result" ${CLIENT_LOG}.${SUFFIX}` != "1" ]; then
    #             echo -e "\n***\n*** Failed. Expected 1 $result result\n***"
    #             RET=1
    #         fi
    #     done
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

# Test with json input and output data
$SIMPLE_STRING_INFER_CLIENT --json-input-data --json-output-data >> ${CLIENT_LOG}.c++.json 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.c++.json
    RET=1
fi

# Test while reusing the InferInput and InferRequestedOutput objects
$SIMPLE_REUSE_INFER_OBJECTS_CLIENT -v >> ${CLIENT_LOG}.c++.reuse 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.c++.reuse
    RET=1
fi

python $CLIENT_PLUGIN_TEST >> ${CLIENT_LOG}.python.plugin 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.python.plugin
    RET=1
fi

# Create a password file with username:password
echo -n 'username:' > pswd
echo "password" | openssl passwd -stdin -apr1 >> pswd
nginx -c `pwd`/$NGINX_CONF

python $BASIC_AUTH_TEST
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.python.plugin.auth
    RET=1
fi
service nginx stop

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

SERVER_ARGS="--backend-directory=${BACKEND_DIR} --model-repository=${MODELDIR} --model-control-mode=explicit"
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
if [ $(cat ${SERVER_LOG} | grep "Invalid config override" | wc -l) -eq 0 ]; then
    cat ${SERVER_LOG}
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

SERVER_ARGS="--backend-directory=${BACKEND_DIR} --model-repository=${MODELDIR} --model-control-mode=explicit"
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

# Send bad request where the 'data' field misaligns with the 'shape' field of the input
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -d'{"inputs":[{"name":"INPUT0","datatype":"INT32","shape":[1,16],"data":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}]}' localhost:8000/v2/models/simple/infer`
set -e
if [ "$code" == "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ `grep -c "\{\"error\":\"Unable to parse 'data': Shape does not match true shape of 'data' field\"\}" ./curl.out` != "1" ]; then
    RET=1
fi

rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -d'{"inputs":[{"name":"INPUT0","datatype":"INT32","shape":[1,16],"data":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]}]}' localhost:8000/v2/models/simple/infer`
set -e
if [ "$code" == "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ `grep -c "\{\"error\":\"Unable to parse 'data': Shape does not match true shape of 'data' field\"\}" ./curl.out` != "1" ]; then
    RET=1
fi

# Check if the server is still working after the above bad requests
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

kill $SERVER_PID
wait $SERVER_PID

# Run cpp client unit test
rm -rf unit_test_models && mkdir unit_test_models
cp -r $DATADIR/qa_model_repository/onnx_int32_int32_int32 unit_test_models/.
cp -r ${MODELDIR}/simple unit_test_models/.

SERVER_ARGS="--backend-directory=${BACKEND_DIR} --model-repository=unit_test_models
            --trace-file=global_unittest.log --trace-level=TIMESTAMPS --trace-rate=1"
SERVER_LOG="./inference_server_cc_unit_test.log"
CLIENT_LOG="./cc_unit_test.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
# Run all unit tests except load
$CC_UNIT_TEST --gtest_filter=HTTP*:-*Load* >> ${CLIENT_LOG} 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# Run cpp client load API unit test
rm -rf unit_test_models && mkdir unit_test_models
cp -r $DATADIR/qa_model_repository/onnx_int32_int32_int32 unit_test_models/.
# Make only version 2, 3 is valid version directory while config requests 1, 3
rm -rf unit_test_models/onnx_int32_int32_int32/1

# Start with EXPLICIT mode and load onnx_float32_float32_float32
SERVER_ARGS="--model-repository=`pwd`/unit_test_models \
             --model-control-mode=explicit \
             --load-model=onnx_int32_int32_int32 \
             --strict-model-config=false"
SERVER_LOG="./inference_server_cc_unit_test.load.log"
CLIENT_LOG="./cc_unit_test.load.log"

for i in \
   "LoadWithFileOverride" \
   "LoadWithConfigOverride" \
   ; do
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e
    $CC_UNIT_TEST --gtest_filter=HTTP*$i >> ${CLIENT_LOG}.$i 2>&1
    if [ $? -ne 0 ]; then
        cat ${CLIENT_LOG}.$i
        RET=1
    fi
    set -e

    kill $SERVER_PID
    wait $SERVER_PID
done

# Run python http aio unit test
PYTHON_HTTP_AIO_TEST=python_http_aio_test.py
CLIENT_LOG=`pwd`/python_http_aio_test.log
SERVER_ARGS="--backend-directory=${BACKEND_DIR} --model-repository=${MODELDIR}"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
set +e
python $PYTHON_HTTP_AIO_TEST > $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Python HTTP AsyncIO Test Failed\n***"
    RET=1
fi
set -e
kill $SERVER_PID
wait $SERVER_PID

# Run python unit test
MODELDIR=python_unit_test_models
mkdir -p $MODELDIR
rm -rf ${MODELDIR}/*
cp -r $DATADIR/qa_identity_model_repository/onnx_zero_1_float32 ${MODELDIR}/.
cp -r $DATADIR/qa_identity_model_repository/onnx_zero_1_object ${MODELDIR}/.
cp -r $DATADIR/qa_identity_model_repository/onnx_zero_1_float16 ${MODELDIR}/.
cp -r $DATADIR/qa_identity_model_repository/onnx_zero_3_float32 ${MODELDIR}/.
cp -r ${MODELDIR}/onnx_zero_1_object ${MODELDIR}/onnx_zero_1_object_1_element && \
    (cd $MODELDIR/onnx_zero_1_object_1_element && \
        sed -i "s/onnx_zero_1_object/onnx_zero_1_object_1_element/" config.pbtxt && \
        sed -i "0,/-1/{s/-1/1/}" config.pbtxt)
# Model for error code test
cp -r ${MODELDIR}/onnx_zero_1_float32 ${MODELDIR}/onnx_zero_1_float32_queue && \
    (cd $MODELDIR/onnx_zero_1_float32_queue && \
        sed -i "s/onnx_zero_1_float32/onnx_zero_1_float32_queue/" config.pbtxt && \
        echo "dynamic_batching { " >> config.pbtxt && \
        echo "    max_queue_delay_microseconds: 1000000" >> config.pbtxt && \
        echo "    preferred_batch_size: [ 8 ]" >> config.pbtxt && \
        echo "    default_queue_policy {" >> config.pbtxt && \
        echo "        max_queue_size: 1" >> config.pbtxt && \
        echo "    }" >> config.pbtxt && \
        echo "}" >> config.pbtxt)

SERVER_ARGS="--backend-directory=${BACKEND_DIR} --model-repository=${MODELDIR}"
SERVER_LOG="./inference_server_http_test.log"
CLIENT_LOG="./http_test.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

TEST_RESULT_FILE='test_results.txt'
PYTHON_TEST=http_test.py
EXPECTED_NUM_TESTS=9
set +e
python $PYTHON_TEST >$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    RET=1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

### LLM / Generate REST API Endpoint Tests ###

# Helper library to parse SSE events
# https://github.com/mpetazzoni/sseclient
pip install sseclient-py

SERVER_ARGS="--model-repository=`pwd`/generate_models"
SERVER_LOG="./inference_server_generate_endpoint_test.log"
CLIENT_LOG="./generate_endpoint_test.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

## Python Unit Tests
TEST_RESULT_FILE='test_results.txt'
PYTHON_TEST=generate_endpoint_test.py
EXPECTED_NUM_TESTS=14
set +e
python $PYTHON_TEST >$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    RET=1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

### Test Restricted  APIs ###
### Repeated API not allowed

MODELDIR="`pwd`/models"
SERVER_ARGS="--model-repository=${MODELDIR}
             --http-restricted-api=model-repository,health:k1=v1 \
             --http-restricted-api=metadata,health:k2=v2"
SERVER_LOG="./http_restricted_endpoint_test.log"
CLIENT_LOG="./http_restricted_endpoint_test.log"
run_server
EXPECTED_MSG="api 'health' can not be specified in multiple config groups"
if [ "$SERVER_PID" != "0" ]; then
    echo -e "\n***\n*** Expect fail to start $SERVER\n***"
    kill $SERVER_PID
    wait $SERVER_PID
    RET=1
elif [ `grep -c "${EXPECTED_MSG}" ${SERVER_LOG}` != "1" ]; then
    echo -e "\n***\n*** Failed. Expected ${EXPECTED_MSG} to be found in log\n***"
    cat $SERVER_LOG
    RET=1
fi

### Test Unknown Restricted  API###
### Unknown API not allowed

MODELDIR="`pwd`/models"
SERVER_ARGS="--model-repository=${MODELDIR}
             --http-restricted-api=model-reposit,health:k1=v1 \
             --http-restricted-api=metadata,health:k2=v2"
run_server
EXPECTED_MSG="unknown restricted api 'model-reposit'"
if [ "$SERVER_PID" != "0" ]; then
    echo -e "\n***\n*** Expect fail to start $SERVER\n***"
    kill $SERVER_PID
    wait $SERVER_PID
    RET=1
elif [ `grep -c "${EXPECTED_MSG}" ${SERVER_LOG}` != "1" ]; then
    echo -e "\n***\n*** Failed. Expected ${EXPECTED_MSG} to be found in log\n***"
    cat $SERVER_LOG
    RET=1
fi

### Test Restricted  APIs ###
### Restricted model-repository, metadata, and inference

SERVER_ARGS="--model-repository=${MODELDIR} \
             --http-restricted-api=model-repository:admin-key=admin-value \
             --http-restricted-api=inference,metadata:infer-key=infer-value"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
set +e

python $RESTRICTED_API_TEST RestrictedAPITest > $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Python HTTP Restricted Protocol Test Failed\n***"
    RET=1
fi
set -e
kill $SERVER_PID
wait $SERVER_PID

###

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
