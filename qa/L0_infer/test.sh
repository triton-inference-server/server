#!/bin/bash
# Copyright 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ldconfig || true

export CUDA_VISIBLE_DEVICES=0

TEST_RESULT_FILE='test_results.txt'
CLIENT_LOG_BASE="./client"
INFER_TEST=infer_test.py
SERVER_TIMEOUT=${SERVER_TIMEOUT:=600}

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
    LEAKCHECK_ARGS_BASE="--leak-check=full --show-leak-kinds=definite --max-threads=3000 --num-callers=20"
    SERVER_TIMEOUT=4000
    rm -f $LEAKCHECK_LOG_BASE*
    # Remove 'python', 'python_dlpack' and 'onnx' from BACKENDS and test them
    # separately below.
    BACKENDS="graphdef savedmodel libtorch plan openvino"
fi

if [ "$TEST_SYSTEM_SHARED_MEMORY" -eq 1 ] || [ "$TEST_CUDA_SHARED_MEMORY" -eq 1 ]; then
    EXPECTED_NUM_TESTS=${EXPECTED_NUM_TESTS:="33"}
else
    EXPECTED_NUM_TESTS=${EXPECTED_NUM_TESTS:="46"}
fi

TF_VERSION=${TF_VERSION:=2}
TEST_JETSON=${TEST_JETSON:=0}

# Default size (in MB) of shared memory to be used by each python model
# instance (Default is 1MB)
DEFAULT_SHM_SIZE_MB=${DEFAULT_SHM_SIZE_MB:=1}
DEFAULT_SHM_SIZE_BYTES=$((1024*1024*$DEFAULT_SHM_SIZE_MB))

# On windows the paths invoked by the script (running in WSL) must use
# /mnt/c when needed but the paths on the tritonserver command-line
# must be C:/ style.
if [[ "$(< /proc/sys/kernel/osrelease)" == *microsoft* ]]; then
    MODELDIR=${MODELDIR:=C:/models}
    DATADIR=${DATADIR:="/mnt/c/data/inferenceserver/${REPO_VERSION}"}
    BACKEND_DIR=${BACKEND_DIR:=C:/tritonserver/backends}
    SERVER=${SERVER:=/mnt/c/tritonserver/bin/tritonserver.exe}
else
    MODELDIR=${MODELDIR:=`pwd`/models}
    DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
    TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
    SERVER=${TRITON_DIR}/bin/tritonserver
    BACKEND_DIR=${TRITON_DIR}/backends

    # PyTorch on SBSA requires libgomp to be loaded first. See the following
    # GitHub issue for more information:
    # https://github.com/pytorch/pytorch/issues/2575
    arch=`uname -m`
    if [ $arch = "aarch64" ]; then
      SERVER_LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libgomp.so.1
    fi
fi

# Allow more time to exit. Ensemble brings in too many models
SERVER_ARGS_EXTRA="--exit-timeout-secs=${SERVER_TIMEOUT} --backend-directory=${BACKEND_DIR} --backend-config=tensorflow,version=${TF_VERSION} --backend-config=python,stub-timeout-seconds=120 --backend-config=python,shm-default-byte-size=${DEFAULT_SHM_SIZE_BYTES}"
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
BACKENDS=${BACKENDS:="graphdef savedmodel onnx libtorch plan python python_dlpack openvino"}
export BACKENDS

# If ENSEMBLES not specified, set to 1
ENSEMBLES=${ENSEMBLES:="1"}
export ENSEMBLES

# Test for both batch and nobatch models
NOBATCH=${NOBATCH:="1"}
export NOBATCH
BATCH=${BATCH:="1"}
export BATCH

if [[ $BACKENDS == *"python_dlpack"* ]]; then
    if [ "$TEST_JETSON" == "0" ]; then
        if [[ "aarch64" != $(uname -m) ]] ; then
            pip3 install torch==1.13.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
        else
            pip3 install torch==1.13.0 -f https://download.pytorch.org/whl/torch_stable.html
        fi
    fi
fi

function generate_model_repository() {
    rm -fr models && mkdir models
    for BACKEND in $BACKENDS; do
      if [ "$BACKEND" == "python" ] || [ "$BACKEND" == "python_dlpack" ]; then
        # We will be using ONNX models config.pbtxt and tweak them to make them
        # appropriate for Python backend
        onnx_models=`find ${DATADIR}/qa_model_repository/ -maxdepth 1 -type d -regex '.*onnx_.*'`

        # Types that need to use SubAdd instead of AddSub
        swap_types="float32 int32 int16 int8"
        for onnx_model in $onnx_models; do
          if [ "$BACKEND" == "python_dlpack" ]; then
            python_model=`echo $onnx_model | sed 's/onnx/python_dlpack/g' | sed 's,'"$DATADIR/qa_model_repository/"',,g'`
          else
            python_model=`echo $onnx_model | sed 's/onnx/python/g' | sed 's,'"$DATADIR/qa_model_repository/"',,g'`
          fi

          mkdir -p models/$python_model/1/
          # Remove platform and use Python as the backend
          if [ "$BACKEND" == "python" ]; then
            cat $onnx_model/config.pbtxt | sed 's/platform:.*//g' | sed 's/version_policy.*/backend:\ "python"/g' | sed 's/onnx/python/g' > models/$python_model/config.pbtxt
          else
            cat $onnx_model/config.pbtxt | sed 's/platform:.*//g' | sed 's/version_policy.*/backend:\ "python"/g' | sed 's/onnx/python_dlpack/g' > models/$python_model/config.pbtxt
          fi
          cp $onnx_model/output0_labels.txt models/$python_model

          is_swap_type="0"

          # Check whether this model needs to be swapped
          for swap_type in $swap_types; do
            model_type="$swap_type"_"$swap_type"_"$swap_type"
            if [ "$BACKEND" == "python_dlpack" ]; then
              model_name=python_dlpack_$model_type
              model_name_nobatch=python_dlpack_nobatch_$model_type
              if [ $python_model == $model_name ] || [ $python_model == $model_name_nobatch ]; then
                  cp ../python_models/dlpack_sub_add/model.py models/$python_model/1/
                  is_swap_type="1"
              fi
            else
              model_name=python_$model_type
              model_name_nobatch=python_nobatch_$model_type
              if [ $python_model == $model_name ] || [ $python_model == $model_name_nobatch ]; then
                  cp ../python_models/sub_add/model.py models/$python_model/1/
                  is_swap_type="1"
              fi
            fi
          done

          # Use the AddSub model if it doesn't need to be swapped
          if [ $is_swap_type == "0" ]; then
            if [ "$BACKEND" == "python_dlpack" ]; then
                    cp ../python_models/dlpack_add_sub/model.py models/$python_model/1/
            else
                    cp ../python_models/add_sub/model.py models/$python_model/1/
            fi
          fi
        done
      elif [ "$BACKEND" == "plan" ] && [ "$TRITON_SERVER_CPU_ONLY" == "1" ]; then
        # skip plan_tensorrt models since they don't run on CPU only containers
        continue
      else
        cp -r ${DATADIR}/qa_model_repository/${BACKEND}* \
          models/.
      fi
    done

    if [ "$ENSEMBLES" == "1" ]; then

      # Copy identity backend models and ensembles
      for BACKEND in $BACKENDS; do
        if [ "$BACKEND" == "plan" ] && [ "$TRITON_SERVER_CPU_ONLY" == "1" ]; then
            # skip plan_tensorrt models since they don't run on CPU only containers
            continue
        elif [ "$BACKEND" != "python" ] && [ "$BACKEND" != "python_dlpack" ] && [ "$BACKEND" != "openvino" ]; then
            cp -r ${DATADIR}/qa_ensemble_model_repository/qa_model_repository/*${BACKEND}* \
              models/.
        fi
      done

      cp -r ${DATADIR}/qa_ensemble_model_repository/qa_model_repository/nop_* \
        models/.

      create_nop_version_dir `pwd`/models

      if [[ $BACKENDS == *"graphdef"* ]]; then
        ENSEMBLE_MODELS="wrong_label_int32_float32_float32 label_override_int32_float32_float32 mix_type_int32_float32_float32"

        ENSEMBLE_MODELS="${ENSEMBLE_MODELS} batch_to_nobatch_float32_float32_float32 batch_to_nobatch_nobatch_float32_float32_float32 nobatch_to_batch_float32_float32_float32 nobatch_to_batch_nobatch_float32_float32_float32 mix_nobatch_batch_float32_float32_float32"

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
      if [ "$FW" == "onnx" ] && [ "$TEST_VALGRIND" -eq 1 ]; then
        # Reduce the instance count to make loading onnx models faster
        for MC in `ls models/${FW}*/config.pbtxt`; do
            echo "instance_group [ { kind: ${KIND} count: 1 }]" >> $MC
        done
      elif [ "$FW" != "plan" ] && [ "$FW" != "python" ] && [ "$FW" != "python_dlpack" ] && [ "$FW" != "openvino" ];then
        for MC in `ls models/${FW}*/config.pbtxt`; do
            echo "instance_group [ { kind: ${KIND} }]" >> $MC
        done
      elif [ "$FW" == "python" ] || [ "$FW" == "python_dlpack" ] || [ "$FW" == "openvino" ]; then
        for MC in `ls models/${FW}*/config.pbtxt`; do
            echo "instance_group [ { kind: KIND_CPU }]" >> $MC
        done
      fi
    done

    # Modify custom_zero_1_float32 and custom_nobatch_zero_1_float32 for relevant ensembles
    # This is done after the instance group change above so that identity backend models
    # are run on CPU. Skip for Windows test.
    cp -r ../custom_models/custom_zero_1_float32 models/. &&\
        mkdir -p models/custom_zero_1_float32/1 && \
        (cd models/custom_zero_1_float32 && \
            echo "instance_group [ { kind: KIND_CPU }]" >> config.pbtxt)
    cp -r models/custom_zero_1_float32 models/custom_nobatch_zero_1_float32 && \
        (cd models/custom_zero_1_float32 && \
            sed -i "s/max_batch_size: 1/max_batch_size: 8/" config.pbtxt && \
            sed -i "s/dims: \[ 1 \]/dims: \[ -1 \]/" config.pbtxt) && \
        (cd models/custom_nobatch_zero_1_float32 && \
            sed -i "s/custom_zero_1_float32/custom_nobatch_zero_1_float32/" config.pbtxt && \
            sed -i "s/max_batch_size: 1/max_batch_size: 0/" config.pbtxt && \
            sed -i "s/dims: \[ 1 \]/dims: \[ -1, -1 \]/" config.pbtxt)

}

for TARGET in cpu gpu; do
    if [ "$TRITON_SERVER_CPU_ONLY" == "1" ]; then
        if [ "$TARGET" == "gpu" ]; then
            echo -e "Skip GPU testing on CPU-only device"
            continue
        fi
    fi

    SERVER_LOG=$SERVER_LOG_BASE.${TARGET}.log
    CLIENT_LOG=$CLIENT_LOG_BASE.${TARGET}.log

    generate_model_repository

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
        check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Result Verification Failed\n***"
            RET=1
        fi
    fi

    set -e

    kill_server

    set +e
    if [ "$TEST_VALGRIND" -eq 1 ]; then
        python3 ../common/check_valgrind_log.py -f $LEAKCHECK_LOG
        if [ $? -ne 0 ]; then
            RET=1
        fi
    fi
    set -e
done

# Run 'python', 'python_dlpack' and 'onnx' models separately in valgrind test.
# Loading python and python_dlpack models has OOM issue when running with
# valgrind, so loading only batch or nobatch models for each time.
# Loading all the onnx models at once requires more than 12 hours. Loading them
# separately to reduce the loading time.
if [ "$TEST_VALGRIND" -eq 1 ]; then
  TESTING_BACKENDS="python python_dlpack onnx"
  EXPECTED_NUM_TESTS=42
  if [ "$TEST_JETSON" == "0" ]; then
    if [[ "aarch64" != $(uname -m) ]] ; then
        pip3 install torch==1.13.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    else
        pip3 install torch==1.13.0 -f https://download.pytorch.org/whl/torch_stable.html
    fi
  fi

  for BACKENDS in $TESTING_BACKENDS; do
    export BACKENDS
    for TARGET in cpu gpu; do
      rm -fr *models
      generate_model_repository
      mkdir nobatch_models
      mv ./models/*nobatch_* ./nobatch_models/.
      cp -fr ./models/nop_* ./nobatch_models/.

      for BATCHING_MODE in batch nobatch; do
        if [ "$TRITON_SERVER_CPU_ONLY" == "1" ]; then
          if [ "$TARGET" == "gpu" ]; then
              echo -e "Skip GPU testing on CPU-only device"
              continue
          fi
        fi

        SERVER_LOG=$SERVER_LOG_BASE.${TARGET}.${BACKENDS}.${BATCHING_MODE}.log
        CLIENT_LOG=$CLIENT_LOG_BASE.${TARGET}.${BACKENDS}.${BATCHING_MODE}.log

        if [ "$BATCHING_MODE" == "batch" ]; then
          NOBATCH="0"
          export NOBATCH
          BATCH="1"
          export BATCH
          MODELDIR=`pwd`/models
        else
          NOBATCH="1"
          export NOBATCH
          BATCH="0"
          export BATCH
          MODELDIR=`pwd`/nobatch_models
        fi

        SERVER_ARGS="--model-repository=${MODELDIR} ${SERVER_ARGS_EXTRA}"
        LEAKCHECK_LOG=$LEAKCHECK_LOG_BASE.${TARGET}.${BACKENDS}.${BATCHING_MODE}.log
        LEAKCHECK_ARGS="$LEAKCHECK_ARGS_BASE --log-file=$LEAKCHECK_LOG"
        run_server_leakcheck

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
            check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG
                cat $TEST_RESULT_FILE
                echo -e "\n***\n*** Test Result Verification Failed\n***"
                RET=1
            fi
        fi

        set -e

        kill_server

        set +e
        python3 ../common/check_valgrind_log.py -f $LEAKCHECK_LOG
        if [ $? -ne 0 ]; then
            RET=1
        fi
        set -e
      done
    done
  done
fi

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
