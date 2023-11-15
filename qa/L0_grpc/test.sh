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

CLIENT_PLUGIN_TEST="./grpc_client_plugin_test.py"
BASIC_AUTH_TEST="./grpc_basic_auth_test.py"
NGINX_CONF="./nginx.conf"
# On windows the paths invoked by the script (running in WSL) must use
# /mnt/c when needed but the paths on the tritonserver command-line
# must be C:/ style.
if [[ "$(< /proc/sys/kernel/osrelease)" == *microsoft* ]]; then
    SDKDIR=${SDKDIR:=C:/sdk}
    MODELDIR=${MODELDIR:=C:/models}
    CLIENT_PLUGIN_MODELDIR=${MODELDIR:=C:/client_plugin_models}
    DATADIR=${DATADIR:="/mnt/c/data/inferenceserver/${REPO_VERSION}"}
    BACKEND_DIR=${BACKEND_DIR:=C:/tritonserver/backends}
    SERVER=${SERVER:=/mnt/c/tritonserver/bin/tritonserver.exe}

    SIMPLE_AIO_INFER_CLIENT_PY=${SDKDIR}/python/simple_grpc_aio_infer_client.py
    SIMPLE_AIO_STREAM_INFER_CLIENT_PY=${SDKDIR}/python/simple_grpc_aio_sequence_stream_infer_client.py
    SIMPLE_HEALTH_CLIENT_PY=${SDKDIR}/python/simple_grpc_health_metadata.py
    SIMPLE_INFER_CLIENT_PY=${SDKDIR}/python/simple_grpc_infer_client.py
    SIMPLE_ASYNC_INFER_CLIENT_PY=${SDKDIR}/python/simple_grpc_async_infer_client.py
    SIMPLE_STRING_INFER_CLIENT_PY=${SDKDIR}/python/simple_grpc_string_infer_client.py
    SIMPLE_STREAM_INFER_CLIENT_PY=${SDKDIR}/python/simple_grpc_sequence_stream_infer_client.py
    SIMPLE_SEQUENCE_INFER_CLIENT_PY=${SDKDIR}/python/simple_grpc_sequence_sync_infer_client.py
    SIMPLE_IMAGE_CLIENT_PY=${SDKDIR}/python/image_client.py
    # SIMPLE_ENSEMBLE_IMAGE_CLIENT_PY=${SDKDIR}/python/ensemble_image_client.py
    SIMPLE_SHM_STRING_CLIENT_PY=${SDKDIR}/python/simple_grpc_shm_string_client.py
    SIMPLE_SHM_CLIENT_PY=${SDKDIR}/python/simple_grpc_shm_client.py
    SIMPLE_CUDASHM_CLIENT_PY=${SDKDIR}/python/simple_grpc_cudashm_client.py
    SIMPLE_MODEL_CONTROL_PY=${SDKDIR}/python/simple_grpc_model_control.py
    SIMPLE_REUSE_INFER_OBJECTS_CLIENT_PY=${SDKDIR}/python/reuse_infer_objects_client.py
    SIMPLE_KEEPALIVE_CLIENT_PY=${SDKDIR}/python/simple_grpc_keepalive_client.py
    SIMPLE_CUSTOM_ARGS_CLIENT_PY=${SDKDIR}/python/simple_grpc_custom_args_client.py
    EXPLICIT_BYTE_CONTENT_CLIENT_PY=${SDKDIR}/python/grpc_explicit_byte_content_client.py
    EXPLICIT_INT_CONTENT_CLIENT_PY=${SDKDIR}/python/grpc_explicit_int_content_client.py
    EXPLICIT_INT8_CONTENT_CLIENT_PY=${SDKDIR}/python/grpc_explicit_int8_content_client.py
    GRPC_CLIENT_PY=${SDKDIR}/python/grpc_client.py
    GRPC_IMAGE_CLIENT_PY=${SDKDIR}/python/grpc_image_client.py

    SIMPLE_HEALTH_CLIENT=${SDKDIR}/python/simple_grpc_health_metadata
    SIMPLE_INFER_CLIENT=${SDKDIR}/python/simple_grpc_infer_client
    SIMPLE_STRING_INFER_CLIENT=${SDKDIR}/python/simple_grpc_string_infer_client
    SIMPLE_ASYNC_INFER_CLIENT=${SDKDIR}/python/simple_grpc_async_infer_client
    SIMPLE_MODEL_CONTROL=${SDKDIR}/python/simple_grpc_model_control
    SIMPLE_STREAM_INFER_CLIENT=${SDKDIR}/python/simple_grpc_sequence_stream_infer_client
    SIMPLE_SEQUENCE_INFER_CLIENT=${SDKDIR}/python/simple_grpc_sequence_sync_infer_client
    SIMPLE_SHM_CLIENT=${SDKDIR}/python/simple_grpc_shm_client
    SIMPLE_CUDASHM_CLIENT=${SDKDIR}/python/simple_grpc_cudashm_client
    SIMPLE_IMAGE_CLIENT=${SDKDIR}/python/image_client
    # SIMPLE_ENSEMBLE_IMAGE_CLIENT=${SDKDIR}/python/ensemble_image_client
    SIMPLE_REUSE_INFER_OBJECTS_CLIENT=${SDKDIR}/python/reuse_infer_objects_client
    SIMPLE_KEEPALIVE_CLIENT=${SDKDIR}/python/simple_grpc_keepalive_client
    SIMPLE_CUSTOM_ARGS_CLIENT=${SDKDIR}/python/simple_grpc_custom_args_client
    # [FIXME] point to proper client
    CC_UNIT_TEST=${SDKDIR}/python/cc_client_test
else
    MODELDIR=${MODELDIR:=`pwd`/models}
    CLIENT_PLUGIN_MODELDIR=${CLIENTPLUGINMODELDIR:=`pwd`/client_plugin_models}
    DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
    TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
    SERVER=${TRITON_DIR}/bin/tritonserver
    BACKEND_DIR=${TRITON_DIR}/backends

    SIMPLE_AIO_INFER_CLIENT_PY=../clients/simple_grpc_aio_infer_client.py
    SIMPLE_AIO_STREAM_INFER_CLIENT_PY=../clients/simple_grpc_aio_sequence_stream_infer_client.py
    SIMPLE_HEALTH_CLIENT_PY=../clients/simple_grpc_health_metadata.py
    SIMPLE_INFER_CLIENT_PY=../clients/simple_grpc_infer_client.py
    SIMPLE_ASYNC_INFER_CLIENT_PY=../clients/simple_grpc_async_infer_client.py
    SIMPLE_STRING_INFER_CLIENT_PY=../clients/simple_grpc_string_infer_client.py
    SIMPLE_STREAM_INFER_CLIENT_PY=../clients/simple_grpc_sequence_stream_infer_client.py
    SIMPLE_SEQUENCE_INFER_CLIENT_PY=../clients/simple_grpc_sequence_sync_infer_client.py
    SIMPLE_IMAGE_CLIENT_PY=../clients/image_client.py
    # SIMPLE_ENSEMBLE_IMAGE_CLIENT_PY=../clients/ensemble_image_client.py
    SIMPLE_SHM_STRING_CLIENT_PY=../clients/simple_grpc_shm_string_client.py
    SIMPLE_SHM_CLIENT_PY=../clients/simple_grpc_shm_client.py
    SIMPLE_CUDASHM_CLIENT_PY=../clients/simple_grpc_cudashm_client.py
    SIMPLE_MODEL_CONTROL_PY=../clients/simple_grpc_model_control.py
    SIMPLE_REUSE_INFER_OBJECTS_CLIENT_PY=../clients/reuse_infer_objects_client.py
    SIMPLE_KEEPALIVE_CLIENT_PY=../clients/simple_grpc_keepalive_client.py
    SIMPLE_CUSTOM_ARGS_CLIENT_PY=../clients/simple_grpc_custom_args_client.py
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
    # SIMPLE_ENSEMBLE_IMAGE_CLIENT=../clients/ensemble_image_client
    SIMPLE_REUSE_INFER_OBJECTS_CLIENT=../clients/reuse_infer_objects_client
    SIMPLE_KEEPALIVE_CLIENT=../clients/simple_grpc_keepalive_client
    SIMPLE_CUSTOM_ARGS_CLIENT=../clients/simple_grpc_custom_args_client
    CC_UNIT_TEST=../clients/cc_client_test
fi
PYTHON_UNIT_TEST=python_unit_test.py

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

python $SIMPLE_HEALTH_CLIENT_PY -v >> ${CLIENT_LOG}.health 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.health
    RET=1
fi

IMAGE=../images/vulture.jpeg
for i in \
        $SIMPLE_AIO_INFER_CLIENT_PY \
        $SIMPLE_AIO_STREAM_INFER_CLIENT_PY \
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
        $SIMPLE_KEEPALIVE_CLIENT_PY \
        $SIMPLE_CUSTOM_ARGS_CLIENT_PY \
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
    # elif [ $SUFFIX == "ensemble_image_client" ]; then
    #     python $i -c 1 $EXTRA_ARGS ../images >> "${CLIENT_LOG}.${SUFFIX}" 2>&1
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
   $SIMPLE_KEEPALIVE_CLIENT \
   $SIMPLE_CUSTOM_ARGS_CLIENT \
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
    # elif [ $SUFFIX == "ensemble_image_client" ]; then
    #     $i -c 1 -i grpc -u localhost:8001 ../images >> "${CLIENT_LOG}.c++.${SUFFIX}" 2>&1
    #     for result in "SPORTS CAR" "COFFEE MUG" "VULTURE"; do
    #         if [ `grep -c "$result" ${CLIENT_LOG}.c++.${SUFFIX}` != "1" ]; then
    #             echo -e "\n***\n*** Failed. Expected 1 $result result\n***"
    #             RET=1
    #         fi
    #     done
    elif [ $BASE = ${SIMPLE_INFER_CLIENT} ]; then
        # Test forcing new channel creation with simple infer client
        NEW_CHANNEL_STRING = "creating client_channel for channel stack"
        GRPC_TRACE=subchannel GRPC_VERBOSITY=info $i -v -c "true" >> ${CLIENT_LOG}.c++.${SUFFIX} 2>&1
        if [ $? -ne 0 ]; then
            cat ${CLIENT_LOG}.c++.${SUFFIX}
            RET=1
        fi
        if [ `grep -c ${NEW_CHANNEL_STRING} ${CLIENT_LOG}.c++.${SUFFIX}` != "1" ]; then
            echo -e "\n***\n*** Failed. Expected 1 ${NEW_CHANNEL_STRING} calls\n***"
            cat $CLIENT_LOG.c++.${SUFFIX}
            RET=1
        fi
        GRPC_TRACE=subchannel GRPC_VERBOSITY=info $i -v -c "false" >> ${CLIENT_LOG}.c++.${SUFFIX} 2>&1
        if [ $? -ne 0 ]; then
            cat ${CLIENT_LOG}.c++.${SUFFIX}
            RET=1
        fi
        if [ `grep -c ${NEW_CHANNEL_STRING} ${CLIENT_LOG}.c++.${SUFFIX}` != "2" ]; then
            echo -e "\n***\n*** Failed. Expected 2 ${NEW_CHANNEL_STRING} calls\n***"
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

SERVER_ARGS="--backend-directory=${BACKEND_DIR} --model-repository=${CLIENT_PLUGIN_MODELDIR} --http-header-forward-pattern=.* --grpc-header-forward-pattern=.*"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python3 $CLIENT_PLUGIN_TEST >> ${CLIENT_LOG}.python.plugin 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.python.plugin
    RET=1
fi
set -e

# Create a password file with username:password
echo -n 'username:' > pswd
echo "password" | openssl passwd -stdin -apr1 >> pswd
nginx -c `pwd`/$NGINX_CONF

python3 $BASIC_AUTH_TEST
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.python.plugin.auth
    RET=1
fi
service nginx stop

kill $SERVER_PID
wait $SERVER_PID

export GRPC_TRACE=compression, channel
export GRPC_VERBOSITY=DEBUG
SERVER_ARGS="--backend-directory=${BACKEND_DIR} --model-repository=${MODELDIR} --grpc-infer-response-compression-level=high"
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
$CC_UNIT_TEST --gtest_filter=GRPC*:-*Load* >> ${CLIENT_LOG} 2>&1
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
    $CC_UNIT_TEST --gtest_filter=GRPC*$i >> ${CLIENT_LOG}.$i 2>&1
    if [ $? -ne 0 ]; then
        cat ${CLIENT_LOG}.$i
        RET=1
    fi
    set -e

    kill $SERVER_PID
    wait $SERVER_PID
done

# Run python grpc aio unit test
PYTHON_GRPC_AIO_TEST=python_grpc_aio_test.py
CLIENT_LOG=`pwd`/python_grpc_aio_test.log
SERVER_ARGS="--backend-directory=${BACKEND_DIR} --model-repository=${MODELDIR}"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
set +e
python $PYTHON_GRPC_AIO_TEST > $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Python GRPC AsyncIO Test Failed\n***"
    RET=1
fi
set -e
kill $SERVER_PID
wait $SERVER_PID

# Test GRPC health check implemented
go install github.com/grpc-ecosystem/grpc-health-probe@latest
HEALTH_PROBE="${GOPATH}/bin/grpc-health-probe -addr=localhost:8001"

CLIENT_LOG=`pwd`/grpc_health_probe_offline.log
set +e
$HEALTH_PROBE > $CLIENT_LOG 2>&1
set -e
if [ `grep -c "timeout: failed to connect service" ${CLIENT_LOG}` != "1" ]; then
    echo -e "\n***\n*** Failed. Expected health check timeout\n***"
    cat $CLIENT_LOG
    RET=1
fi

SERVER_ARGS="--backend-directory=${BACKEND_DIR} --model-repository=${MODELDIR}"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

CLIENT_LOG=`pwd`/grpc_health_probe_online.log
set +e
$HEALTH_PROBE > $CLIENT_LOG 2>&1
set -e
if [ `grep -c "status: SERVING" ${CLIENT_LOG}` != "1" ]; then
    echo -e "\n***\n*** Failed. Expected health check to return SERVING\n***"
    cat $CLIENT_LOG
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

# Repeated protocol, not allowed
SERVER_ARGS="--model-repository=${MODELDIR} \
             --grpc-restricted-protocol=model-repository,health:k1=v1 \
             --grpc-restricted-protocol=metadata,health:k2=v2"
run_server
EXPECTED_MSG="protocol 'health' can not be specified in multiple config groups"
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

# Unknown protocol, not allowed
SERVER_ARGS="--model-repository=${MODELDIR} \
             --grpc-restricted-protocol=model-reposit,health:k1=v1 \
             --grpc-restricted-protocol=metadata,health:k2=v2"
run_server
EXPECTED_MSG="unknown restricted protocol 'model-reposit'"
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

# Test restricted protocols
SERVER_ARGS="--model-repository=${MODELDIR} \
             --grpc-restricted-protocol=model-repository:admin-key=admin-value \
             --grpc-restricted-protocol=inference,health:infer-key=infer-value"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
set +e
python $PYTHON_UNIT_TEST RestrictedProtocolTest > $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Python GRPC Restricted Protocol Test Failed\n***"
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

