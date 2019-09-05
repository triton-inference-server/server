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

CLIENT_LOG="./client.log"
CLIENT=model_config_test.py

SERVER=/opt/tensorrtserver/bin/trtserver
SERVER_TIMEOUT=20
SERVER_LOG_BASE="./inference_server"
source ../common/util.sh

export CUDA_VISIBLE_DEVICES=0

TRIALS="tensorflow_savedmodel tensorflow_graphdef tensorrt_plan caffe2_netdef onnxruntime_onnx pytorch_libtorch custom"

# Copy TensorRT plans into the test model repositories.
for modelpath in \
        autofill_noplatform/tensorrt/bad_input_dims/1 \
        autofill_noplatform/tensorrt/bad_input_type/1 \
        autofill_noplatform/tensorrt/bad_output_dims/1 \
        autofill_noplatform/tensorrt/bad_output_type/1 \
        autofill_noplatform/tensorrt/too_few_inputs/1 \
        autofill_noplatform/tensorrt/too_many_inputs/1 \
        autofill_noplatform/tensorrt/unknown_input/1 \
        autofill_noplatform/tensorrt/unknown_output/1 \
        autofill_noplatform_success/tensorrt/no_name_platform/1 \
        autofill_noplatform_success/tensorrt/empty_config/1     \
        autofill_noplatform_success/tensorrt/no_config/1 \
        autofill_noplatform_success/tensorrt/incomplete_input/1 \
        autofill_noplatform_success/tensorrt/incomplete_output/1 ; do
    mkdir -p $modelpath
    cp /data/inferenceserver/${REPO_VERSION}/qa_model_repository/plan_float32_float32_float32/1/model.plan \
       $modelpath/.
done

for modelpath in \
   autofill_noplatform/ensemble/invalid_input_map/invalid_input_map/1 \
       autofill_noplatform/ensemble/invalid_input_map/fp32_dim1_batch4/1 \
       autofill_noplatform/ensemble/invalid_input_map/fp32_dim1_batch4_input4/1 \
       autofill_noplatform/ensemble/invalid_input_map/fp32_dim1_batch4_output3/1 \
       autofill_noplatform/ensemble/invalid_output_map/invalid_output_map/1 \
       autofill_noplatform/ensemble/invalid_output_map/fp32_dim1_batch4/1 \
       autofill_noplatform/ensemble/invalid_output_map/fp32_dim1_batch4_input4/1 \
       autofill_noplatform/ensemble/invalid_output_map/fp32_dim1_batch4_output3/1 \
       autofill_noplatform/ensemble/invalid_batch_size/invalid_batch_size/1 \
       autofill_noplatform/ensemble/invalid_batch_size/invalid_batch_size/1 \
       autofill_noplatform/ensemble/invalid_batch_size/fp32_dim1_batch2/1 \
       autofill_noplatform/ensemble/invalid_batch_size/fp32_dim1_batch4/1 \
       autofill_noplatform/ensemble/inconsistent_shape/inconsistent_shape/1 \
       autofill_noplatform/ensemble/inconsistent_shape/fp32_dim1_batch4/1 \
       autofill_noplatform/ensemble/inconsistent_shape/fp32_dim3_batch4/1 \
       autofill_noplatform/ensemble/inconsistent_data_type/inconsistent_data_type/1 \
       autofill_noplatform/ensemble/inconsistent_data_type/fp32_dim1_batch2/1 \
       autofill_noplatform/ensemble/inconsistent_data_type/int32_dim1_batch4/1 \
       autofill_noplatform/ensemble/non_existing_model/non_existing_model/1 \
       autofill_noplatform/ensemble/non_existing_model/fp32_dim1_batch4/1 \
       autofill_noplatform/ensemble/non_existing_model/fp32_dim1_batch4_output3/1 \
       autofill_noplatform/ensemble/self_circular_dependency/self_circular_dependency/1 \
       autofill_noplatform/ensemble/self_circular_dependency/fp32_dim1_batch4/1 \
       autofill_noplatform/ensemble/self_circular_dependency/fp32_dim1_batch4_input4/1 \
       autofill_noplatform/ensemble/self_circular_dependency/fp32_dim1_batch4_output3/1 \
       autofill_noplatform/ensemble/unmapped_input/unmapped_input/1 \
       autofill_noplatform/ensemble/unmapped_input/fp32_dim1_batch4/1 \
       autofill_noplatform/ensemble/unmapped_input/fp32_dim1_batch4_input4/1 \
       autofill_noplatform/ensemble/circular_dependency/circular_dependency/1 \
       autofill_noplatform/ensemble/circular_dependency/circular_dependency_2/1 \
       autofill_noplatform/ensemble/no_required_version/no_required_version/1 \
       autofill_noplatform/ensemble/no_required_version/simple/1 \
       autofill_noplatform/ensemble/no_required_version_2/no_required_version_2/1 \
       autofill_noplatform/ensemble/no_required_version_2/simple/1 \
       autofill_noplatform/ensemble/no_required_version_3/no_required_version_3/1 \
       autofill_noplatform/ensemble/no_required_version_3/simple/1 \
       autofill_noplatform_success/ensemble/embedded_ensemble/embedded_ensemble/1 \
       autofill_noplatform_success/ensemble/embedded_ensemble/fp32_dim1_batch4/1 \
       autofill_noplatform_success/ensemble/embedded_ensemble/inner_ensemble/1 \
       autofill_noplatform_success/ensemble/inconsistent_shape/inconsistent_shape/1 \
       autofill_noplatform_success/ensemble/inconsistent_shape/fp32_dim1_batch4/1 \
       autofill_noplatform_success/ensemble/inconsistent_shape/fp32_dim2_nobatch/1 \
       autofill_noplatform_success/ensemble/inconsistent_shape_2/inconsistent_shape_2/1 \
       autofill_noplatform_success/ensemble/inconsistent_shape_2/fp32_dim1_batch4/1 \
       autofill_noplatform_success/ensemble/inconsistent_shape_2/fp32_dim2_nobatch/1 \
       autofill_noplatform_success/ensemble/unmapped_output/unmapped_output/1 \
       autofill_noplatform_success/ensemble/unmapped_output/fp32_dim1_batch4_output3/1 ; do
   mkdir -p $modelpath
   cp ./libidentity.so $modelpath/libcustom.so
done


rm -f $SERVER_LOG_BASE* $CLIENT_LOG
RET=0

for TRIAL in $TRIALS; do
    # Run all tests that require no autofill but that add the platform to
    # the model config before running the test
    for TARGET in `ls noautofill_platform`; do
        SERVER_ARGS="--model-repository=`pwd`/models --strict-model-config=true"
        SERVER_LOG=$SERVER_LOG_BASE.noautofill_platform_${TARGET}.log

        rm -fr models && mkdir models
        cp -r noautofill_platform/$TARGET models/.

        CONFIG=models/$TARGET/config.pbtxt
        EXPECTEDS=models/$TARGET/expected*

        # If there is a config.pbtxt change/add platform to it
        if [ -f $CONFIG ]; then
            sed -i '/platform:/d' $CONFIG
            echo "platform: \"$TRIAL\"" >> $CONFIG
            cat $CONFIG
        fi

        echo -e "Test platform $TRIAL on noautofill_platform/$TARGET" >> $CLIENT_LOG

        # We expect all the tests to fail with one of the expected
        # error messages
        run_server
        if [ "$SERVER_PID" != "0" ]; then
            echo -e "*** FAILED: unexpected success starting $SERVER" >> $CLIENT_LOG
            RET=1
            kill $SERVER_PID
            wait $SERVER_PID
        else
            EXFOUND=0
            for EXPECTED in `ls $EXPECTEDS`; do
                EX=`cat $EXPECTED`
                if grep ^E[0-9][0-9][0-9][0-9].*"$EX" $SERVER_LOG; then
                    echo -e "Found \"$EX\"" >> $CLIENT_LOG
                    EXFOUND=1
                    break
                else
                    echo -e "Not found \"$EX\"" >> $CLIENT_LOG
                fi
            done

            if [ "$EXFOUND" == "0" ]; then
                echo -e "*** FAILED: platform $TRIAL noautofill_platform/$TARGET" >> $CLIENT_LOG
                RET=1
            fi
        fi
    done
done

# Run all autofill tests that don't add a platform to the model config
# before running the test
for TARGET_DIR in `ls -d autofill_noplatform/*/*`; do
    TARGET_DIR_DOT=`echo $TARGET_DIR | tr / .`
    TARGET=`basename ${TARGET_DIR}`

    SERVER_ARGS="--model-repository=`pwd`/models --strict-model-config=false"
    SERVER_LOG=$SERVER_LOG_BASE.${TARGET_DIR_DOT}.log

    # If there is a config.pbtxt at the top-level of the test then
    # assume that the directory is a single model. Otherwise assume
    # that the directory is an entire model repository.
    rm -fr models && mkdir models
    if [ -f ${TARGET_DIR}/config.pbtxt ]; then
        cp -r ${TARGET_DIR} models/.
        EXPECTEDS=models/$TARGET/expected*
    else
        cp -r ${TARGET_DIR}/* models/.
        EXPECTEDS=models/expected*
    fi

    echo -e "Test ${TARGET_DIR}" >> $CLIENT_LOG

    # We expect all the tests to fail with one of the expected
    # error messages
    run_server
    if [ "$SERVER_PID" != "0" ]; then
        echo -e "*** FAILED: unexpected success starting $SERVER" >> $CLIENT_LOG
        RET=1
        kill $SERVER_PID
        wait $SERVER_PID
    else
        EXFOUND=0
        for EXPECTED in `ls $EXPECTEDS`; do
            EX=`cat $EXPECTED`
            if grep ^E[0-9][0-9][0-9][0-9].*"$EX" $SERVER_LOG; then
                echo -e "Found \"$EX\"" >> $CLIENT_LOG
                EXFOUND=1
                break
            else
                echo -e "Not found \"$EX\"" >> $CLIENT_LOG
            fi
        done

        if [ "$EXFOUND" == "0" ]; then
            echo -e "*** FAILED: ${TARGET_DIR}" >> $CLIENT_LOG
            RET=1
        fi
    fi
done

# Run all autofill tests that are expected to be successful. These
# tests don't add a platform to the model config before running
for TARGET_DIR in `ls -d autofill_noplatform_success/*/*`; do
    TARGET_DIR_DOT=`echo $TARGET_DIR | tr / .`
    TARGET=`basename ${TARGET_DIR}`

    SERVER_ARGS="--model-repository=`pwd`/models --strict-model-config=false"
    SERVER_LOG=$SERVER_LOG_BASE.${TARGET_DIR_DOT}.log

    # If there is a config.pbtxt at the top-level of the test then
    # assume that the directory is a single model. Otherwise assume
    # that the directory is an entire model repository.
    rm -fr models && mkdir models
    if [ -f ${TARGET_DIR}/config.pbtxt ] || [ "$TARGET" = "no_config" ]; then
        cp -r ${TARGET_DIR} models/.
    else
        cp -r ${TARGET_DIR}/* models/.
    fi

    echo -e "Test $TARGET_DIR" >> $CLIENT_LOG

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "*** FAILED: unable to start $SERVER" >> $CLIENT_LOG
        RET=1
    else
        set +e
        python ./compare_status.py --expected_dir models/$TARGET --model $TARGET >>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            echo -e "*** FAILED: unexpected model config" >> $CLIENT_LOG
            RET=1
        fi
        set -e

        kill $SERVER_PID
        wait $SERVER_PID
    fi
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
    cat $CLIENT_LOG
fi

exit $RET
