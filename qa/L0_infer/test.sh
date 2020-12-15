#!/bin/bash
# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

CLIENT_LOG_BASE="./client"
INFER_TEST=infer_test.py
EXPECTED_NUM_TESTS=${EXPECTED_NUM_TESTS:="42"}

if [ -z "$TEST_SYSTEM_SHARED_MEMORY" ]; then
    TEST_SYSTEM_SHARED_MEMORY="0"
fi

if [ -z "$TEST_CUDA_SHARED_MEMORY" ]; then
    TEST_CUDA_SHARED_MEMORY="0"
fi

if [ -z "$TEST_VALGRIND" ]; then
    TEST_VALGRIND="0"
fi

if [ "$TEST_VALGRIND" -eq 1 ]; then
    LEAKCHECK_LOG_BASE="./valgrind_test"
    LEAKCHECK=/usr/bin/valgrind
    LEAKCHECK_ARGS_BASE="--leak-check=full --show-leak-kinds=definite --max-threads=3000"
    SERVER_TIMEOUT=3600
    rm -f $LEAKCHECK_LOG_BASE*
fi

if [ "$TEST_SYSTEM_SHARED_MEMORY" -eq 1 ] || [ "$TEST_CUDA_SHARED_MEMORY" -eq 1 ]; then
    EXPECTED_NUM_TESTS="29"
fi

TF_VERSION=${TF_VERSION:=1}

# On windows the paths invoked by the script (running in WSL) must use
# /mnt/c when needed but the paths on the tritonserver command-line
# must be C:/ style.
if [[ "$(< /proc/sys/kernel/osrelease)" == *Microsoft ]]; then
    MODELDIR=${MODELDIR:=C:/models}
    DATADIR=${DATADIR:="/mnt/c/data/inferenceserver/${REPO_VERSION}"}
    BACKEND_DIR=${BACKEND_DIR:=C:/tritonserver/backends}
    SERVER=${SERVER:=/mnt/c/tritonserver/bin/tritonserver.exe}
    export USE_HTTP=0
else
    MODELDIR=${MODELDIR:=`pwd`/models}
    DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
    OPTDIR=${OPTDIR:="/opt"}
    SERVER=${OPTDIR}/tritonserver/bin/tritonserver
    BACKEND_DIR=${OPTDIR}/tritonserver/backends
fi

# Allow more time to exit. Ensemble brings in too many models
SERVER_ARGS_EXTRA="--exit-timeout-secs=120 --backend-directory=${BACKEND_DIR} --backend-config=tensorflow,version=${TF_VERSION}"
SERVER_ARGS="--model-repository=${MODELDIR} ${SERVER_ARGS_EXTRA}"
SERVER_LOG_BASE="./inference_server"
source ../common/util.sh

rm -f $SERVER_LOG_BASE* $CLIENT_LOG_BASE*

RET=0

# Verify the flag is set only on CPU-only device
if [ "$TRITON_SERVER_CPU_ONLY" == "1" ]; then
    gpu_count=`nvidia-smi -L | grep GPU | wc -l`
    if [ "$gpu_count" -ne 0 ]; then
    echo -e "\n***\n*** Running on a device with GPU\n***"
    echo -e "\n***\n*** Test Failed To Run\n***"
    exit 1
    fi
fi

# If BACKENDS not specified, set to all
BACKENDS=${BACKENDS:="graphdef savedmodel onnx libtorch plan custom python"}
export BACKENDS

# If ENSEMBLES not specified, set to 1
ENSEMBLES=${ENSEMBLES:="1"}
export ENSEMBLES


for TARGET in cpu gpu; do
    if [ "$TRITON_SERVER_CPU_ONLY" == "1" ]; then
        if [ "$TARGET" == "gpu" ]; then
            echo -e "Skip GPU testing on CPU-only device"
            continue
        fi
        # set strict readiness=false on CPU-only device to allow
        # unsuccessful load of TensorRT plans, which require GPU.
        SERVER_ARGS="--model-repository=${MODELDIR} --strict-readiness=false --exit-on-error=false ${SERVER_ARGS_EXTRA}"
    fi

    SERVER_LOG=$SERVER_LOG_BASE.${TARGET}.log
    CLIENT_LOG=$CLIENT_LOG_BASE.${TARGET}.log

    rm -fr models && mkdir models
    for BACKEND in $BACKENDS; do
      if [ "$BACKEND" != "custom" ] && [ "$BACKEND" != "python" ]; then
        cp -r ${DATADIR}/qa_model_repository/${BACKEND}* \
          models/.
      elif [ "$BACKEND" == "custom" ]; then
        cp -r ../custom_models/custom_float32_* models/. && \
        cp -r ../custom_models/custom_int32_* models/. && \
        cp -r ../custom_models/custom_nobatch_* models/.
      elif [ "$BACKEND" == "python" ]; then
        # We will be using ONNX models config.pbtxt and tweak them to make them
        # appropriate for Python backend
        onnx_models=`find ${DATADIR}/qa_model_repository/ -maxdepth 1 -type d -regex '.*onnx_.*'`

        # Types that need to use SubAdd instead of AddSub
        swap_types="float32 int32 int16 int8"
        for onnx_model in $onnx_models; do
          python_model=`echo $onnx_model | sed 's/onnx/python/g' | sed 's,'"$DATADIR/qa_model_repository/"',,g'`
          mkdir -p models/$python_model/1/
          # Remove platform and use Python as the backend
          cat $onnx_model/config.pbtxt | sed 's/platform:.*//g' | sed 's/version_policy.*/backend:\ "python"/g' | sed 's/onnx/python/g' > models/$python_model/config.pbtxt
          cp $onnx_model/output0_labels.txt models/$python_model

          is_swap_type="0"

          # Check whether this model needs to be swapped
          for swap_type in $swap_types; do
            model_type="$swap_type"_"$swap_type"_"$swap_type"
            model_name=python_$model_type
            model_name_nobatch=python_nobatch_$model_type
            if [ $python_model == $model_name ] || [ $python_model == $model_name_nobatch ]; then
                cp ../python_models/sub_add/model.py models/$python_model/1/
                is_swap_type="1"
            fi
          done

          # Use the AddSub model if it doesn't need to be swapped
          if [ $is_swap_type == "0" ]; then
                cp ../python_models/add_sub/model.py models/$python_model/1/
          fi
        done
      fi
    done

    if [ "$ENSEMBLES" == "1" ]; then
      if [[ $BACKENDS == *"custom"* ]]; then
        for BACKEND in $BACKENDS; do
          if [ "$BACKEND" != "custom" ] && [ "$BACKEND" != "python" ]; then
              cp -r ${DATADIR}/qa_ensemble_model_repository/qa_model_repository/*${BACKEND}* \
                models/.
          elif [ "$BACKEND" == "custom" ]; then
            cp -r ${DATADIR}/qa_ensemble_model_repository/qa_model_repository/nop_* \
              models/.
          fi
        done

        create_nop_modelfile `pwd`/libidentity.so `pwd`/models
      fi

      if [[ $BACKENDS == *"graphdef"* ]]; then
        ENSEMBLE_MODELS="wrong_label_int32_float32_float32 label_override_int32_float32_float32 mix_type_int32_float32_float32"

        if [[ $BACKENDS == *"custom"* ]]; then
          ENSEMBLE_MODELS="${ENSEMBLE_MODELS} batch_to_nobatch_float32_float32_float32 batch_to_nobatch_nobatch_float32_float32_float32 nobatch_to_batch_float32_float32_float32 nobatch_to_batch_nobatch_float32_float32_float32 mix_nobatch_batch_float32_float32_float32"
        fi

        if [[ $BACKENDS == *"savedmodel"* ]] ; then
          ENSEMBLE_MODELS="${ENSEMBLE_MODELS} mix_platform_float32_float32_float32 mix_ensemble_int32_float32_float32"
        fi

        for EM in $ENSEMBLE_MODELS; do
          mkdir -p ../ensemble_models/$EM/1 && cp -r ../ensemble_models/$EM models/.
        done
      fi
    fi

    KIND="KIND_GPU" && [[ "$TARGET" == "cpu" ]] && KIND="KIND_CPU"
    for FW in $BACKENDS; do
      if [ "$FW" != "plan" ] && [ "$FW" != "python" ];then
        for MC in `ls models/${FW}*/config.pbtxt`; do
            echo "instance_group [ { kind: ${KIND} }]" >> $MC
        done
      elif [ "$FW" == "python" ]; then
        for MC in `ls models/${FW}*/config.pbtxt`; do
            echo "instance_group [ { kind: KIND_CPU }]" >> $MC
        done
      fi
    done

    # Modify custom_zero_1_float32 and custom_nobatch_zero_1_float32 for relevant ensembles
    # This is done after the instance group change above so that identity custom backends
    # are run on CPU
    if [[ $BACKENDS == *"custom"* ]]; then
      cp -r ../custom_models/custom_zero_1_float32 models/. &&\
          mkdir -p models/custom_zero_1_float32/1 && \
          cp `pwd`/libidentity.so models/custom_zero_1_float32/1/. && \
          (cd models/custom_zero_1_float32 && \
              echo "default_model_filename: \"libidentity.so\"" >> config.pbtxt && \
              echo "instance_group [ { kind: KIND_CPU }]" >> config.pbtxt)
      cp -r models/custom_zero_1_float32 models/custom_nobatch_zero_1_float32 && \
          (cd models/custom_zero_1_float32 && \
              sed -i "s/max_batch_size: 1/max_batch_size: 8/" config.pbtxt && \
              sed -i "s/dims: \[ 1 \]/dims: \[ -1 \]/" config.pbtxt) && \
          (cd models/custom_nobatch_zero_1_float32 && \
              sed -i "s/custom_zero_1_float32/custom_nobatch_zero_1_float32/" config.pbtxt && \
              sed -i "s/max_batch_size: 1/max_batch_size: 0/" config.pbtxt && \
              sed -i "s/dims: \[ 1 \]/dims: \[ -1, -1 \]/" config.pbtxt)
    fi

    # Check if running a memory leak check
    if [ "$TEST_VALGRIND" -eq 1 ]; then
        LEAKCHECK_LOG=$LEAKCHECK_LOG_BASE.${TARGET}.log
        LEAKCHECK_ARGS="$LEAKCHECK_ARGS_BASE --log-file=$LEAKCHECK_LOG"
        run_server_leakcheck
    else
        run_server
    fi

    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e

    python3 $INFER_TEST >$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        RET=1
    else
        check_test_results $CLIENT_LOG $EXPECTED_NUM_TESTS
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Result Verification Failed\n***"
            RET=1
        fi
    fi


    set -e

    kill $SERVER_PID
    wait $SERVER_PID

    set +e
    if [ "$TEST_VALGRIND" -eq 1 ]; then
        check_valgrind_log $LEAKCHECK_LOG
        if [ $? -ne 0 ]; then
            RET=1
        fi
    fi
    set -e
done

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
