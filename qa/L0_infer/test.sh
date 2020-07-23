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
if [ -z "$TEST_SYSTEM_SHARED_MEMORY" ] then
elif [ -z "$TEST_CUDA_SHARED_MEMORY" ] then
elif [ -z "$CPU_ONLY" ] then
elif [ -z "$ENSEMBLES" ] then
elif [ -z "$BACKENDS" ] then
fi

MODELDIR=`pwd`/models
DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
OPTDIR=${OPTDIR:="/opt"}
SERVER=${OPTDIR}/tritonserver/bin/tritonserver

# Allow more time to exit. Ensemble brings in too many models
SERVER_ARGS="--model-repository=${MODELDIR} --exit-timeout-secs=120"
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
BACKENDS=${BACKENDS:="graphdef savedmodel netdef onnx libtorch plan custom"}
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
        SERVER_ARGS="--model-repository=${MODELDIR} --exit-timeout-secs=120 --strict-readiness=false --exit-on-error=false"
    fi

    SERVER_LOG=$SERVER_LOG_BASE.${TARGET}.log
    CLIENT_LOG=$CLIENT_LOG_BASE.${TARGET}.log

    rm -fr models && mkdir models
    for BACKEND in $BACKENDS; do
      if [ "$BACKEND" != "custom" ]; then
        cp -r ${DATADIR}/qa_model_repository/${BACKEND}* \
          models/.
      else
        cp -r ../custom_models/custom_float32_* models/. && \
        cp -r ../custom_models/custom_int32_* models/. && \
        cp -r ../custom_models/custom_nobatch_* models/.
      fi
    done

    if [ "$ENSEMBLES" == "1" ]; then
      if [[ $BACKENDS == *"custom"* ]]; then
        for BACKEND in $BACKENDS; do
          if [ "$BACKEND" != "custom" ]; then
              cp -r ${DATADIR}/qa_ensemble_model_repository/qa_model_repository/*${BACKEND}* \
                models/.
          else
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

        if [[ $BACKENDS == *"savedmodel"* ]] && [[ $BACKENDS == *"netdef"* ]] ; then
          ENSEMBLE_MODELS="${ENSEMBLE_MODELS} mix_platform_float32_float32_float32 mix_ensemble_int32_float32_float32"
        fi

        for EM in $ENSEMBLE_MODELS; do
          mkdir -p ../ensemble_models/$EM/1 && cp -r ../ensemble_models/$EM models/.
        done
      fi
    fi

    KIND="KIND_GPU" && [[ "$TARGET" == "cpu" ]] && KIND="KIND_CPU"
    for FW in $BACKENDS; do
      if [ "$FW" != "plan" ]; then
        for MC in `ls models/${FW}*/config.pbtxt`; do
            echo "instance_group [ { kind: ${KIND} }]" >> $MC
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

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e

    python $INFER_TEST >$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi

    # At the end of $CLIENT_LOG there is a single line JSON containing the
    # result of unittests.
    test_result_json=`tail -n 1 $CLIENT_LOG`
    check_test_results $CLIENT_LOG $EXPECTED_NUM_TESTS

    set -e

    kill $SERVER_PID
    wait $SERVER_PID
done

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
fi

exit $RET
