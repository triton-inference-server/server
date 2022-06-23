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

export CUDA_VISIBLE_DEVICES=0

CLIENT=../clients/image_client
CLIENT_LOG="./client.log"

IMAGE="../images/vulture.jpeg"

DATADIR=`pwd`/models

# If BACKENDS not specified, set to all
BACKENDS=${BACKENDS:="graphdef savedmodel onnx libtorch plan"}

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=$DATADIR --log-verbose=1 --exit-timeout-secs=120"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

RET=0

for BACKEND in ${BACKENDS}; do
    rm -f $SERVER_LOG $CLIENT_LOG
    # Test for fixed-size data type
    # Use the addsub models as example.
    rm -fr models && mkdir models
    cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/${BACKEND}_float32_float32_float32 models/. && \
    cp -r /data/inferenceserver/${REPO_VERSION}/qa_sequence_model_repository/${BACKEND}_sequence_int32 models/.

    INPUT_PREFIX="INPUT"
    IDENTITY_INPUT_PREFIX="INPUT" && [ "$BACKEND" == "libtorch" ] && IDENTITY_INPUT_PREFIX="INPUT__"
    SEQ_INPUT="INPUT" && [ "$BACKEND" == "libtorch" ] && SEQ_INPUT="INPUT__0"
    START="START" && [ "$BACKEND" == "libtorch" ] && START="START__1"
    READY="READY" && [ "$BACKEND" == "libtorch" ] && READY="READY__2"

    # 2 instances per device with random / zero data.
    # The zero data sample will run twice
    #
    # Provide warmup instruction (batch size 1) in model config
    (cd models/${BACKEND}_float32_float32_float32 && \
        echo "model_warmup [{" >> config.pbtxt && \
        echo "    name : \"regular sample\"" >> config.pbtxt && \
        echo "    batch_size: 1" >> config.pbtxt && \
        echo "    inputs {" >> config.pbtxt && \
        echo "        key: \"${INPUT_PREFIX}0\"" >> config.pbtxt && \
        echo "        value: {" >> config.pbtxt && \
        echo "            data_type: TYPE_FP32" >> config.pbtxt && \
        echo "            dims: 16" >> config.pbtxt && \
        echo "            zero_data: true" >> config.pbtxt && \
        echo "        }" >> config.pbtxt && \
        echo "    }" >> config.pbtxt && \
        echo "    inputs {" >> config.pbtxt && \
        echo "        key: \"${INPUT_PREFIX}1\"" >> config.pbtxt && \
        echo "        value: {" >> config.pbtxt && \
        echo "            data_type: TYPE_FP32" >> config.pbtxt && \
        echo "            dims: 16" >> config.pbtxt && \
        echo "            random_data: true" >> config.pbtxt && \
        echo "        }" >> config.pbtxt && \
        echo "    }" >> config.pbtxt && \
        echo "}]" >> config.pbtxt )

    # zero data. For realistic sequence model, 'count' may not work
    # well because the model will expect a valid sequence of requests which
    # should be represented by a series of warmup samples. 'count > 1'
    # essentially "resends" one of the sample, which may invalidate the
    # sequence. This is okay for this specific test because the synthetic model
    # is not data sensitive.
    #
    # Instruction for sequence model (batch size 8), need to specify control tensor
    (cd models/${BACKEND}_sequence_int32 && \
        echo "model_warmup [{" >> config.pbtxt && \
        echo "    name : \"sequence sample\"" >> config.pbtxt && \
        echo "    count : 2" >> config.pbtxt && \
        echo "    batch_size: 8" >> config.pbtxt && \
        echo "    inputs {" >> config.pbtxt && \
        echo "        key: \"${SEQ_INPUT}\"" >> config.pbtxt && \
        echo "        value: {" >> config.pbtxt && \
        echo "            data_type: TYPE_INT32" >> config.pbtxt && \
        echo "            dims: 1" >> config.pbtxt && \
        echo "            zero_data: true" >> config.pbtxt && \
        echo "        }" >> config.pbtxt && \
        echo "    }" >> config.pbtxt && \
        echo "    inputs {" >> config.pbtxt && \
        echo "        key: \"${START}\"" >> config.pbtxt && \
        echo "        value: {" >> config.pbtxt && \
        echo "            data_type: TYPE_INT32" >> config.pbtxt && \
        echo "            dims: 1" >> config.pbtxt && \
        echo "            zero_data: true" >> config.pbtxt && \
        echo "        }" >> config.pbtxt && \
        echo "    }" >> config.pbtxt && \
        echo "    inputs {" >> config.pbtxt && \
        echo "        key: \"${READY}\"" >> config.pbtxt && \
        echo "        value: {" >> config.pbtxt && \
        echo "            data_type: TYPE_INT32" >> config.pbtxt && \
        echo "            dims: 1" >> config.pbtxt && \
        echo "            zero_data: true" >> config.pbtxt && \
        echo "        }" >> config.pbtxt && \
        echo "    }" >> config.pbtxt && \
        echo "}]" >> config.pbtxt )

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e

    grep "is running warmup sample 'regular sample'" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected warmup for stateless model\n***"
        RET=1
    fi
    grep "is running warmup sample 'sequence sample' for iteration 1" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected 1st warmup iteration for stateful model\n***"
        RET=1
    fi
    grep "is running warmup sample 'sequence sample' for iteration 2" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected 2nd warmup iteration for stateful model\n***"
        RET=1
    fi
    grep "failed to run warmup" $SERVER_LOG
    if [ $? -eq 0 ]; then
        echo -e "\n***\n*** Failed. Expected no warmup error\n***"
        RET=1
    fi

    set -e

    kill $SERVER_PID
    wait $SERVER_PID

    # Test for variable-size data type (string)
    rm -fr models && mkdir models
    SUPPORT_STRING=0 && ([[ $BACKEND == "savedmodel" ]] || [[ $BACKEND == "onnx" ]] || [[ $BACKEND == "savedmodel" ]]) && SUPPORT_STRING=1
    if [ "$SUPPORT_STRING" == "1" ] ; then
        cp -r /data/inferenceserver/${REPO_VERSION}/qa_sequence_model_repository/${BACKEND}_sequence_object models/.
        cp -r /data/inferenceserver/${REPO_VERSION}/qa_identity_model_repository/${BACKEND}_zero_1_object models/.

        # random and zero data (two samples)
        #
        # Provide warmup instruction (batch size 1) in model config
        (cd models/${BACKEND}_zero_1_object && \
            echo "model_warmup [" >> config.pbtxt && \
            echo "{" >> config.pbtxt && \
            echo "    name : \"zero string stateless\"" >> config.pbtxt && \
            echo "    batch_size: 1" >> config.pbtxt && \
            echo "    inputs {" >> config.pbtxt && \
            echo "        key: \"${IDENTITY_INPUT_PREFIX}0\"" >> config.pbtxt && \
            echo "        value: {" >> config.pbtxt && \
            echo "            data_type: TYPE_STRING" >> config.pbtxt && \
            echo "            dims: 16" >> config.pbtxt && \
            echo "            zero_data: true" >> config.pbtxt && \
            echo "        }" >> config.pbtxt && \
            echo "    }" >> config.pbtxt && \
            echo "}," >> config.pbtxt && \
            echo "{" >> config.pbtxt && \
            echo "    name : \"random string stateless\"" >> config.pbtxt && \
            echo "    batch_size: 1" >> config.pbtxt && \
            echo "    inputs {" >> config.pbtxt && \
            echo "        key: \"${IDENTITY_INPUT_PREFIX}0\"" >> config.pbtxt && \
            echo "        value: {" >> config.pbtxt && \
            echo "            data_type: TYPE_STRING" >> config.pbtxt && \
            echo "            dims: 16" >> config.pbtxt && \
            echo "            random_data: true" >> config.pbtxt && \
            echo "        }" >> config.pbtxt && \
            echo "    }" >> config.pbtxt && \
            echo "}" >> config.pbtxt && \
            echo "]" >> config.pbtxt )

        # user provided data
        #
        # Instruction for sequence model (batch size 8), need to specify control tensor
        (cd models/${BACKEND}_sequence_object && \
            echo "model_warmup [{" >> config.pbtxt && \
            echo "    name : \"string statefull\"" >> config.pbtxt && \
            echo "    batch_size: 8" >> config.pbtxt && \
            echo "    inputs {" >> config.pbtxt && \
            echo "        key: \"${SEQ_INPUT}\"" >> config.pbtxt && \
            echo "        value: {" >> config.pbtxt && \
            echo "            data_type: TYPE_STRING" >> config.pbtxt && \
            echo "            dims: 1" >> config.pbtxt && \
            echo "            input_data_file: \"raw_string_data\"" >> config.pbtxt && \
            echo "        }" >> config.pbtxt && \
            echo "    }" >> config.pbtxt && \
            echo "    inputs {" >> config.pbtxt && \
            echo "        key: \"${START}\"" >> config.pbtxt && \
            echo "        value: {" >> config.pbtxt && \
            echo "            data_type: TYPE_INT32" >> config.pbtxt && \
            echo "            dims: 1" >> config.pbtxt && \
            echo "            zero_data: true" >> config.pbtxt && \
            echo "        }" >> config.pbtxt && \
            echo "    }" >> config.pbtxt && \
            echo "    inputs {" >> config.pbtxt && \
            echo "        key: \"${READY}\"" >> config.pbtxt && \
            echo "        value: {" >> config.pbtxt && \
            echo "            data_type: TYPE_INT32" >> config.pbtxt && \
            echo "            dims: 1" >> config.pbtxt && \
            echo "            zero_data: true" >> config.pbtxt && \
            echo "        }" >> config.pbtxt && \
            echo "    }" >> config.pbtxt && \
            echo "}]" >> config.pbtxt )

        # Prepare string data (one element that is "233")
        mkdir -p models/${BACKEND}_sequence_object/warmup && \
            (cd models/${BACKEND}_sequence_object/warmup && \
                    echo -n -e '\x03\x00\x00\x00\x32\x33\x33' > raw_string_data)

        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        set +e

        grep "is running warmup sample 'zero string stateless'" $SERVER_LOG
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Failed. Expected warmup for zero string stateless model\n***"
            RET=1
        fi
        grep "is running warmup sample 'random string stateless'" $SERVER_LOG
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Failed. Expected warmup for random string stateless model\n***"
            RET=1
        fi
        grep "is running warmup sample 'string statefull'" $SERVER_LOG
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Failed. Expected warmup for string stateful model\n***"
            RET=1
        fi
        grep "failed to run warmup" $SERVER_LOG
        if [ $? -eq 0 ]; then
            echo -e "\n***\n*** Failed. Expected no warmup error\n***"
            RET=1
        fi

        set -e

        kill $SERVER_PID
        wait $SERVER_PID
    fi

    if [ "$BACKEND" == "graphdef" ]; then
        # Show effect of warmup by using a TF model with TF-TRT optimization which is
        # known to be slow on first inference.
        # Note: model can be obatined via the fetching script in docs/example
        rm -fr models && \
            mkdir models && \
            cp -r /data/inferenceserver/${REPO_VERSION}/tf_model_store/inception_v3_graphdef models/.

        # Enable TF-TRT optimization
        (cd models/inception_v3_graphdef && \
            echo "optimization { execution_accelerators { gpu_execution_accelerator : [ { name : \"tensorrt\"} ] } }" >> config.pbtxt)

        # Duplicate the same model with warmup enabled
        cp -r models/inception_v3_graphdef models/inception_v3_warmup &&
            (cd models/inception_v3_warmup && \
                sed -i 's/inception_v3_graphdef/inception_v3_warmup/' config.pbtxt)

        (cd models/inception_v3_warmup && \
            echo 'model_warmup [{' >> config.pbtxt && \
            echo '    name : "image sample"' >> config.pbtxt && \
            echo '    batch_size: 1' >> config.pbtxt && \
            echo '    inputs {' >> config.pbtxt && \
            echo '        key: "input"' >> config.pbtxt && \
            echo '        value: {' >> config.pbtxt && \
            echo '            data_type: TYPE_FP32' >> config.pbtxt && \
            echo '            dims: [ 299, 299, 3 ]' >> config.pbtxt && \
            echo '            input_data_file: "raw_mug_data"' >> config.pbtxt && \
            echo '        }' >> config.pbtxt && \
            echo '    }' >> config.pbtxt && \
            echo '}]' >> config.pbtxt )

        # prepare provided data instead of synthetic one
        mkdir -p models/inception_v3_warmup/warmup && \
            cp raw_mug_data models/inception_v3_warmup/warmup/.

        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        set +e

        grep "is running warmup sample 'image sample'" $SERVER_LOG
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Failed. Expected warmup for image model\n***"
            RET=1
        fi
        grep "failed to run warmup" $SERVER_LOG
        if [ $? -eq 0 ]; then
            echo -e "\n***\n*** Failed. Expected no warmup error\n***"
            RET=1
        fi

        # Time the first inference for both models
        time $CLIENT -m inception_v3_graphdef -s INCEPTION $IMAGE -i grpc -u localhost:8001 >>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Failed\n***"
            cat $CLIENT_LOG
            RET=1
        fi
        time $CLIENT -m inception_v3_warmup -s INCEPTION $IMAGE -i grpc -u localhost:8001 >>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Failed\n***"
            cat $CLIENT_LOG
            RET=1
        fi

        set -e

        kill $SERVER_PID
        wait $SERVER_PID
    fi
done

# Test warmup sample failure
rm -fr models && \
    mkdir models && \
    cp -r failing_infer models/.

run_server
if [ "$SERVER_PID" != "0" ]; then
    echo -e "\n***\n*** Expect fail to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
grep "failed to run warmup sample 'zero sample': An Error Occurred;" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected warmup error\n***"
    cat $SERVER_LOG
    RET=1
fi
set -e

# Test decoupled model
rm -fr models && \
    mkdir models && \
    cp -r decoupled models/.

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
grep "is running warmup sample 'decoupled sample'" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected warmup for decoupled model\n***"
    RET=1
fi
grep "failed to run warmup" $SERVER_LOG
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** Failed. Expected no warmup error\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
fi

exit $RET
