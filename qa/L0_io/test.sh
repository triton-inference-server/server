#!/bin/bash
# Copyright 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This test requires at least 2 GPUs to test h2d and d2d transfer combinations
export CUDA_VISIBLE_DEVICES=0,1

IO_TEST_UTIL=./memory_alloc
CLIENT_LOG="./client.log"
MODELSDIR=`pwd`/models

DATADIR=/data/inferenceserver/${REPO_VERSION}/qa_model_repository
ENSEMBLEDIR=/data/inferenceserver/${REPO_VERSION}/qa_ensemble_model_repository/qa_model_repository

# Must explicitly set LD_LIBRARY_PATH so that IO_TEST_UTIL can find
# libtritonserver.so.
LD_LIBRARY_PATH=/opt/tritonserver/lib:$LD_LIBRARY_PATH

rm -f $CLIENT_LOG*

# PyTorch is required for the Python backend dlpack add sub models
pip3 install torch==2.3.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
RET=0

# Prepare float32 models with basic config
rm -rf $MODELSDIR

for trial in graphdef savedmodel onnx libtorch plan python python_dlpack; do
    full=${trial}_float32_float32_float32
    if [ "$trial" == "python" ]; then
        mkdir -p $MODELSDIR/${full}/1 && \
            cp ../python_models/add_sub/model.py $MODELSDIR/${full}/1/. && \
            cp ../python_models/add_sub/config.pbtxt $MODELSDIR/${full}/. && \
            (cd $MODELSDIR/${full} && \
                    sed -i "s/label_filename:.*//" config.pbtxt && \
                    echo "max_batch_size: 64" >> config.pbtxt)

        # ensemble version of the model.
        mkdir -p $MODELSDIR/fan_${full}/1 && \
            cp ../python_models/add_sub/model.py $MODELSDIR/fan_${full}/1/. && \
            cp ../python_models/fan_add_sub/config.pbtxt $MODELSDIR/fan_${full}/. && \
            (cd $MODELSDIR/fan_${full} && \
                    sed -i "s/label_filename:.*//" config.pbtxt && \
                    sed -i "s/model_name: \"ENSEMBLE_MODEL_NAME\"/model_name: \"${full}\"/" config.pbtxt && \
                    sed -i "0,/name:.*/{s/name:.*/name: \"fan_${full}\"/}" config.pbtxt && \
                    echo "max_batch_size: 64" >> config.pbtxt)
        continue
    fi

    if [ "$trial" == "python_dlpack" ]; then
        mkdir -p $MODELSDIR/${full}/1 && \
            cp ../python_models/dlpack_add_sub/model.py $MODELSDIR/${full}/1/. && \
            cp ../python_models/dlpack_add_sub/config.pbtxt $MODELSDIR/${full}/. && \
            (cd $MODELSDIR/${full} && \
                    sed -i "s/label_filename:.*//" config.pbtxt && \
                    sed -i "0,/name:.*/{s/name:.*/name: \"${full}\"/}" config.pbtxt && \
                    echo "max_batch_size: 64" >> config.pbtxt)

        # ensemble version of the model.
        mkdir -p $MODELSDIR/fan_${full}/1 && \
            cp ../python_models/dlpack_add_sub/model.py $MODELSDIR/fan_${full}/1/. && \
            cp ../python_models/fan_add_sub/config.pbtxt $MODELSDIR/fan_${full}/. && \
            (cd $MODELSDIR/fan_${full} && \
                    sed -i "s/label_filename:.*//" config.pbtxt && \
                    sed -i "s/model_name: \"ENSEMBLE_MODEL_NAME\"/model_name: \"${full}\"/" config.pbtxt && \
                    sed -i "0,/name:.*/{s/name:.*/name: \"fan_${full}\"/}" config.pbtxt && \
                    echo "max_batch_size: 64" >> config.pbtxt)
        continue
    fi

    mkdir -p $MODELSDIR/${full}/1 && \
        cp -r $DATADIR/${full}/1/* $MODELSDIR/${full}/1/. && \
        cp $DATADIR/${full}/config.pbtxt $MODELSDIR/${full}/. && \
        (cd $MODELSDIR/${full} && \
                sed -i "s/label_filename:.*//" config.pbtxt && \
                echo "instance_group [{ kind: KIND_CPU }]" >> config.pbtxt)

    # ensemble version of the model.
    mkdir -p $MODELSDIR/fan_${full}/1 && \
    cp $ENSEMBLEDIR/fan_${full}/config.pbtxt $MODELSDIR/fan_${full}/. && \
        (cd $MODELSDIR/fan_${full} && \
                sed -i "s/label_filename:.*//" config.pbtxt)

    if [ "$trial" == "libtorch" ]; then
        (cd $MODELSDIR/fan_${full} && \
                sed -i -e '{
                    N
                    s/key: "OUTPUT\([0-9]\)"\n\(.*\)value: "same_output/key: "OUTPUT__\1"\n\2value: "same_output/
                }' config.pbtxt)
    fi
done

# Prepare string models with basic config
for trial in graphdef savedmodel onnx ; do
    full=${trial}_object_object_object
    mkdir -p $MODELSDIR/${full}/1 && \
        cp -r $DATADIR/${full}/1/* $MODELSDIR/${full}/1/. && \
        cp $DATADIR/${full}/config.pbtxt $MODELSDIR/${full}/. && \
                (cd $MODELSDIR/${full} && \
                sed -i "s/label_filename:.*//" config.pbtxt && \
                echo "instance_group [{ kind: KIND_CPU }]" >> config.pbtxt)
done

# set up "addsub" ensemble for custom float32 model
cp -r $MODELSDIR/fan_graphdef_float32_float32_float32 $MODELSDIR/fan_${full} && \
    (cd $MODELSDIR/fan_${full} && \
            sed -i "s/graphdef_float32_float32_float32/${full}/" config.pbtxt)

# custom float32 component of ensemble
cp -r $ENSEMBLEDIR/nop_TYPE_FP32_-1 $MODELSDIR/. && \
    mkdir -p $MODELSDIR/nop_TYPE_FP32_-1/1

# prepare libtorch multi-device and multi-gpu models
cp -r ../L0_libtorch_instance_group_kind_model/models/libtorch_multi_device $MODELSDIR/.
mkdir -p $MODELSDIR/libtorch_multi_device/1
mkdir -p $MODELSDIR/libtorch_multi_gpu/1
cp $MODELSDIR/libtorch_multi_device/config.pbtxt $MODELSDIR/libtorch_multi_gpu/.
(cd $MODELSDIR/libtorch_multi_gpu && \
    sed -i "s/name: \"libtorch_multi_device\"/name: \"libtorch_multi_gpu\"/" config.pbtxt)

set +e
python3 gen_libtorch_model.py >> $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Error when generating libtorch models. \n***"
    cat $CLIENT_LOG
    exit 1
fi
set -e

TRIALS="graphdef savedmodel onnx libtorch plan python python_dlpack libtorch_multi_gpu libtorch_multi_device"
for input_device in -1 0 1; do
    for output_device in -1 0 1; do
        for trial in ${TRIALS}; do
            # TensorRT Plan should only be deployed on GPU device
            model_devices="-1 0 1" && [[ "$trial" == "plan" ]] && model_devices="0 1"
            full=${trial}_float32_float32_float32 && [[ "$trial" == "libtorch_multi"* ]] && full=${trial}

            for model_device in $model_devices; do
                full_log=$CLIENT_LOG.$full.$input_device.$output_device.$model_device

                host_policy=cpu
                if [ "$model_device" == "-1" ]; then
                    if [[ "$trial" != "libtorch_multi"* ]]; then
                        (cd $MODELSDIR/${full} && \
                            sed -i "s/instance_group.*/instance_group [{ kind: KIND_CPU }]/" config.pbtxt)
                    fi
                else
                    host_policy=gpu_${model_device}
                    if [[ "$trial" != "libtorch_multi"* ]]; then
                        (cd $MODELSDIR/${full} && \
                            sed -i "s/instance_group.*/instance_group [{ kind: KIND_GPU, gpus: [${model_device}] }]/" config.pbtxt)
                    fi
                fi

                set +e
                $IO_TEST_UTIL -i $input_device -o $output_device -r $MODELSDIR -m $full >>$full_log 2>&1
                if [ $? -ne 0 ]; then
                    cat $full_log
                    echo -e "\n***\n*** Test Failed\n***"
                    RET=1
                fi
                set -e

                # Test with host policy
                set +e
                $IO_TEST_UTIL -i $input_device -o $output_device -h $host_policy -r $MODELSDIR -m $full >>$full_log 2>&1
                # FIXME currently only apply the new changes to ORT backend, should apply to others
                if [[ "$trial" == "onnx" ]]; then
                  if [ $? -ne 0 ]; then
                      cat $full_log
                      echo -e "\n***\n*** Test Failed. Expect passing \n***"
                      RET=1
                  fi
                else
                  if [ $? -eq 0 ]; then
                      cat $full_log
                      echo -e "\n***\n*** Test Failed. Expect failure \n***"
                      RET=1
                  fi
                fi
                set -e

                # ensemble
                if [[ "$trial" != "libtorch_multi"* ]]; then
                    set +e
                    $IO_TEST_UTIL -i $input_device -o $output_device -r $MODELSDIR -m fan_$full >>$full_log.ensemble 2>&1
                    if [ $? -ne 0 ]; then
                        cat $full_log.ensemble
                        echo -e "\n***\n*** Test Failed\n***"
                        RET=1
                    fi
                    set -e
                fi
            done
        done

        for trial in graphdef savedmodel onnx; do
            model_devices="-1 0 1"
            for model_device in $model_devices; do
                full=${trial}_object_object_object
                full_log=$CLIENT_LOG.$full.$input_device.$output_device.$model_device

                if [ "$model_device" == "-1" ]; then
                    (cd $MODELSDIR/${full} && \
                        sed -i "s/instance_group.*/instance_group [{ kind: KIND_CPU }]/" config.pbtxt)
                else
                    (cd $MODELSDIR/${full} && \
                        sed -i "s/instance_group.*/instance_group [{ kind: KIND_GPU, gpus: [${model_device}] }]/" config.pbtxt)
                fi

                set +e
                $IO_TEST_UTIL -i $input_device -o $output_device -r $MODELSDIR -m $full >>$full_log 2>&1
                if [ $? -ne 0 ]; then
                    cat $full_log
                    echo -e "\n***\n*** Test Failed\n***"
                    RET=1
                fi
                set -e
            done
        done
    done
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
