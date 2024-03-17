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

TEST_RESULT_FILE='test_results.txt'

# Must run on a single device or else the TRITONSERVER_DELAY_SCHEDULER
# can fail when the requests are distributed to multiple devices.
ldconfig || true

export CUDA_VISIBLE_DEVICES=0

CLIENT_LOG="./client.log"
BATCHER_TEST=sequence_batcher_test.py

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
    LEAKCHECK=/usr/bin/valgrind
    LEAKCHECK_ARGS_BASE="--leak-check=full --show-leak-kinds=definite --max-threads=3000"
    SERVER_TIMEOUT=3600
    rm -f *.valgrind.log

    # Shortened tests due valgrind overhead
    MODEL_TRIALS="0 v"
    NO_DELAY_TESTS="test_simple_sequence \
                      test_no_sequence_start \
                      test_batch_size"
    DELAY_TESTS="test_backlog_fill_no_end \
                    test_backlog_sequence_timeout \
                    test_ragged_batch"
    QUEUE_DELAY_TESTS="test_queue_delay_full_min_util"
fi

if [ -z "$TEST_JETSON" ]; then
    TEST_JETSON="0"
fi

# Shortened tests due to jetson slowdown
if [ "$TEST_JETSON" -eq 1 ]; then
    MODEL_TRIALS="0 v"
fi

TF_VERSION=${TF_VERSION:=2}

# On windows the paths invoked by the script (running in WSL) must use
# /mnt/c when needed but the paths on the tritonserver command-line
# must be C:/ style.
WINDOWS=0
if [[ "$(< /proc/sys/kernel/osrelease)" == *microsoft* ]]; then
    MODELDIR=${MODELDIR:=C:/models}
    DATADIR=${DATADIR:="/mnt/c/data/inferenceserver/${REPO_VERSION}"}
    BACKEND_DIR=${BACKEND_DIR:=C:/tritonserver/backends}
    SERVER=${SERVER:=/mnt/c/tritonserver/bin/tritonserver.exe}
    export WSLENV=$WSLENV:TRITONSERVER_DELAY_SCHEDULER:TRITONSERVER_BACKLOG_DELAY_SCHEDULER
    WINDOWS=1
else
    MODELDIR=${MODELDIR:=`pwd`}
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

SERVER_ARGS_EXTRA="--backend-directory=${BACKEND_DIR} --backend-config=tensorflow,version=${TF_VERSION} --log-verbose=1"

source ../common/util.sh

RET=0

# If BACKENDS not specified, set to all
BACKENDS=${BACKENDS:="graphdef savedmodel onnx plan libtorch custom python"}
export BACKENDS

# If MODEL_TRIALS not specified set to 0 1 2 4 v
MODEL_TRIALS=${MODEL_TRIALS:="0 1 2 4 v"}

# Basic sequence batcher tests
NO_DELAY_TESTS=${NO_DELAY_TESTS:="test_simple_sequence \
                                    test_length1_sequence \
                                    test_batch_size \
                                    test_no_sequence_start \
                                    test_no_sequence_start2 \
                                    test_no_sequence_end \
                                    test_no_correlation_id"}

# Tests that use scheduler delay
DELAY_TESTS=${DELAY_TESTS:="test_backlog_fill \
                              test_backlog_fill_no_end \
                              test_backlog_same_correlation_id \
                              test_backlog_same_correlation_id_no_end \
                              test_backlog_sequence_timeout \
                              test_half_batch \
                              test_skip_batch \
                              test_full_batch \
                              test_ragged_batch \
                              test_backlog"}

# Tests on queue delay
QUEUE_DELAY_TESTS=${QUEUE_DELAY_TESTS:="test_queue_delay_no_min_util \
                                    test_queue_delay_half_min_util \
                                    test_queue_delay_full_min_util"}

# If ENSEMBLES not specified, set to 1
ENSEMBLES=${ENSEMBLES:="1"}
export ENSEMBLES

# If IMPLICIT_STATE not specified, set to 0
IMPLICIT_STATE=${IMPLICIT_STATE:="0"}
export IMPLICIT_STATE

# If INITIAL_STATE_FILE is not specified, set to 0
INITIAL_STATE_FILE=${INITIAL_STATE_FILE:="0"}
export INITIAL_STATE_FILE

# If INITIAL_STATE_ZERO is not specified, set to 0
INITIAL_STATE_ZERO=${INITIAL_STATE_ZERO:="0"}
export INITIAL_STATE_ZERO

# If USE_SINGLE_BUFFER is not specified, set to 0
USE_SINGLE_BUFFER=${USE_SINGLE_BUFFER:="0"}
export USE_SINGLE_BUFFER

# Setup non-variable-size model repositories. The same models are in each
# repository but they are configured as:
#   models0 - four instances with non-batching model
#   models1 - one instance with batch-size 4
#   models2 - two instances with batch-size 2
#   models4 - four instances with batch-size 1
rm -fr *.log  models{0,1,2,4} queue_delay_models && mkdir models{0,1,2,4} queue_delay_models

# Get the datatype to use based on the backend
function get_datatype () {
  local dtype="int32 bool"
  if [[ $1 == "plan" ]]; then
    dtype="float32"
  elif [[ $1 == "savedmodel" ]]; then
    dtype="float32 bool"
  elif [[ $1 == "graphdef" ]]; then
    dtype="object bool int32"
  fi

  # Add type string to the onnx model tests only for implicit state.
  if [ "$IMPLICIT_STATE" == "1" ]; then
    if [[ $1 == "onnx" ]]; then
        dtype="object int32 bool"
    fi
    if [[ $1 == "libtorch" ]]; then
        dtype="object int32 bool"
    fi
  fi
  echo $dtype
}

# Modify corresponding onnx config.pbtxt to create python config.pbtxt
function generate_python_models () {
  model_path=$1
  dest_dir=$2
  onnx_model=$(echo ${model_path//python/onnx})
  python_model=$(basename $model_path)
  mkdir -p $dest_dir/$python_model/1/
  # for emsemble models keep "platform: ensemble"
  if [[ "$model_path" == *"ensemble_model"* ]]; then
    cat $onnx_model/config.pbtxt | sed 's/onnx/python/g' > $dest_dir/$python_model/config.pbtxt
  else
    cat $onnx_model/config.pbtxt | sed 's/platform:.*/backend:\ "python"/g' | sed 's/onnx/python/g' > $dest_dir/$python_model/config.pbtxt
    cp ../python_models/sequence_int32/model.py $dest_dir/$python_model/1/
  fi
}

if [[ "$INITIAL_STATE_ZERO" == "1" && "$INITIAL_STATE_FILE" == "1" ]]; then
  echo -e "\n***\n*** 'INITIAL_STATE_ZERO' and 'INITIAL_STATE_FILE' can't be enabled simultaneously. \n***"
  exit 1
fi

FIXED_MODEL_REPOSITORY=''
VAR_MODEL_REPOSITORY=''
if [ "$IMPLICIT_STATE" == "1" ]; then
  if [[ "$INITIAL_STATE_ZERO" == "0" && "$INITIAL_STATE_FILE" == "0" ]]; then
    FIXED_MODEL_REPOSITORY="qa_sequence_implicit_model_repository"
    VAR_MODEL_REPOSITORY="qa_variable_sequence_implicit_model_repository"
  else
    FIXED_MODEL_REPOSITORY="qa_sequence_initial_state_implicit_model_repository"
    VAR_MODEL_REPOSITORY="qa_variable_sequence_initial_state_implicit_model_repository"
  fi
else
  FIXED_MODEL_REPOSITORY="qa_sequence_model_repository"
  VAR_MODEL_REPOSITORY="qa_variable_sequence_model_repository"
fi

MODELS=""
PYTHON_MODELS=""
for BACKEND in $BACKENDS; do
  if [[ $BACKEND == "custom" ]]; then
    MODELS="$MODELS ../custom_models/custom_sequence_int32"
  else
    DTYPES=$(get_datatype $BACKEND)

    for DTYPE in $DTYPES; do
      MODELS="$MODELS $DATADIR/$FIXED_MODEL_REPOSITORY/${BACKEND}_sequence_${DTYPE}"
    done

    if [ "$ENSEMBLES" == "1" ]; then
      for DTYPE in $DTYPES; do
        # We don't generate ensemble models for bool data type.
        if [[ $DTYPE != "bool" ]]; then
          if [ "$BACKEND" == "python" ]; then
            PYTHON_MODELS="$DATADIR/qa_ensemble_model_repository/$FIXED_MODEL_REPOSITORY/*_onnx_sequence_${DTYPE}"
            TMP=$(echo $PYTHON_MODELS)
            MODELS="$MODELS ${TMP//onnx/python}"
          else
            MODELS="$MODELS $DATADIR/qa_ensemble_model_repository/$FIXED_MODEL_REPOSITORY/*_${BACKEND}_sequence_${DTYPE}"
          fi
        fi
      done
    fi
  fi
done

if [ "$INITIAL_STATE_FILE" == "1" ]; then
  # Create the input_state_data file.
  rm -rf input_state_data
  echo -n -e "\\x64\\x00\\x00\\x00" > input_state_data
fi

for MODEL in $MODELS; do
  if [[ ! "$TEST_VALGRIND" -eq 1 ]]; then
    # Skip libtorch string models
    if [[ "$MODEL" =~ .*"libtorch".*"object".* ]]; then
        continue
    fi
    if [[ "$MODEL" =~ .*"python".* ]]; then
      generate_python_models "$MODEL" "models1"
    else
      cp -r $MODEL models1/.
    fi
      (cd models1/$(basename $MODEL) && \
        sed -i "s/^max_batch_size:.*/max_batch_size: 4/" config.pbtxt && \
        sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 1/" config.pbtxt && \
        sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 1/" config.pbtxt)

    # Skip libtorch string models
    if [[ "$MODEL" =~ .*"libtorch".*"object".* ]]; then
        continue
    fi

    if [[ "$MODEL" =~ .*"python".* ]]; then
      generate_python_models "$MODEL" "models2"
    else
      cp -r $MODEL models2/.
    fi
      (cd models2/$(basename $MODEL) && \
        sed -i "s/^max_batch_size:.*/max_batch_size: 2/" config.pbtxt && \
        sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 2/" config.pbtxt && \
        sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 2/" config.pbtxt)

    if [[ "$MODEL" =~ .*"python".* ]]; then
      generate_python_models "$MODEL" "models4"
    else
      cp -r $MODEL models4/.
    fi
      (cd models4/$(basename $MODEL) && \
        sed -i "s/^max_batch_size:.*/max_batch_size: 1/" config.pbtxt && \
        sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 4/" config.pbtxt && \
        sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 4/" config.pbtxt)

    # Duplicate the models for different delay settings
    if [[ "$MODEL" =~ .*"python".* ]]; then
      generate_python_models "$MODEL" "queue_delay_models"
    else
      cp -r $MODEL queue_delay_models/.
    fi
      (cd queue_delay_models/$(basename $MODEL) && \
        sed -i "s/^max_batch_size:.*/max_batch_size: 4/" config.pbtxt && \
        sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 1/" config.pbtxt && \
        sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 1/" config.pbtxt && \
        sed -i "s/sequence_batching {/sequence_batching {\\ndirect {\\nmax_queue_delay_microseconds: 3000000\\nminimum_slot_utilization: 0\\n}/" config.pbtxt)

    cp -r queue_delay_models/$(basename $MODEL) queue_delay_models/$(basename $MODEL)_half && \
      (cd queue_delay_models/$(basename $MODEL)_half && \
        sed -i "s/$(basename $MODEL)/$(basename $MODEL)_half/" config.pbtxt && \
        sed -i "s/minimum_slot_utilization: 0/minimum_slot_utilization: 0.5/" config.pbtxt)
    cp -r queue_delay_models/$(basename $MODEL) queue_delay_models/$(basename $MODEL)_full && \
      (cd queue_delay_models/$(basename $MODEL)_full && \
        sed -i "s/$(basename $MODEL)/$(basename $MODEL)_full/" config.pbtxt && \
        sed -i "s/minimum_slot_utilization: 0/minimum_slot_utilization: 1/" config.pbtxt)

    # TODO: Enable single state buffer testing for sequence batcher
    # if [ "$USE_SINGLE_BUFFER" == "1" && "$IMPLICIT_STATE" == "1" ]; then
    #   SED_REPLACE_PATTERN="N;N;N;N;N;/state.*dims:.*/a use_single_buffer: true"
    #   (cd models0/$(basename $MODEL) && \
    #     sed -i "$SED_REPLACE_PATTERN" config.pbtxt)
    #   (cd models1/$(basename $MODEL) && \
    #     sed -i "$SED_REPLACE_PATTERN" config.pbtxt)
    #   (cd models2/$(basename $MODEL) && \
    #     sed -i "$SED_REPLACE_PATTERN" config.pbtxt)
    #   (cd models4/$(basename $MODEL) && \
    #     sed -i "$SED_REPLACE_PATTERN" config.pbtxt)
    #   (cd queue_delay_models/$(basename $MODEL)_full && \
    #     sed -i "$SED_REPLACE_PATTERN" config.pbtxt)
    #   (cd queue_delay_models/$(basename $MODEL)_half && \
    #     sed -i "$SED_REPLACE_PATTERN" config.pbtxt)
    # fi
  else
    cp -r $MODEL queue_delay_models/$(basename $MODEL)_full && \
      (cd queue_delay_models/$(basename $MODEL)_full && \
        sed -i "s/$(basename $MODEL)/$(basename $MODEL)_full/" config.pbtxt && \
        sed -i "s/^max_batch_size:.*/max_batch_size: 4/" config.pbtxt && \
        sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 1/" config.pbtxt && \
        sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 1/" config.pbtxt && \
        sed -i "s/sequence_batching {/sequence_batching {\\ndirect {\\nmax_queue_delay_microseconds: 3000000\\nminimum_slot_utilization: 0\\n}/" config.pbtxt && \
        sed -i "s/minimum_slot_utilization: 0/minimum_slot_utilization: 1/" config.pbtxt)
  fi
done

# Adjust the model repository for reading initial state for implicit state from file
if [ "$INITIAL_STATE_FILE" == "1" ]; then
  for MODEL in $MODELS; do
    if [[ ! "$TEST_VALGRIND" -eq 1 ]]; then
      mkdir -p models1/$(basename $MODEL)/initial_state/ && cp input_state_data models1/$(basename $MODEL)/initial_state/ && \
      (cd models1/$(basename $MODEL) && \
        sed -i "s/zero_data.*/data_file:\"input_state_data\"/" config.pbtxt)

      mkdir -p models2/$(basename $MODEL)/initial_state/ && cp input_state_data models2/$(basename $MODEL)/initial_state/ && \
      (cd models2/$(basename $MODEL) && \
        sed -i "s/zero_data.*/data_file:\"input_state_data\"/" config.pbtxt)

      mkdir -p models4/$(basename $MODEL)/initial_state/ && cp input_state_data models4/$(basename $MODEL)/initial_state/ && \
      (cd models4/$(basename $MODEL) && \
        sed -i "s/zero_data.*/data_file:\"input_state_data\"/" config.pbtxt)

      mkdir -p queue_delay_models/$(basename $MODEL)/initial_state/ && cp input_state_data queue_delay_models/$(basename $MODEL)/initial_state/ && \
      (cd queue_delay_models/$(basename $MODEL) && \
        sed -i "s/zero_data.*/data_file:\"input_state_data\"/" config.pbtxt)

      mkdir -p queue_delay_models/$(basename $MODEL)_half/initial_state/ && cp input_state_data queue_delay_models/$(basename $MODEL)_half/initial_state/ && \
      (cd queue_delay_models/$(basename $MODEL)_half && \
        sed -i "s/zero_data.*/data_file:\"input_state_data\"/" config.pbtxt)

      mkdir -p queue_delay_models/$(basename $MODEL)_full/initial_state/ && cp input_state_data queue_delay_models/$(basename $MODEL)_full/initial_state/ && \
      (cd queue_delay_models/$(basename $MODEL)_full && \
        sed -i "s/zero_data.*/data_file:\"input_state_data\"/" config.pbtxt)
    else
      mkdir -p queue_delay_models/$(basename $MODEL)_full/initial_state/ && cp input_state_data queue_delay_models/$(basename $MODEL)_full/initial_state/ && \
       (cd queue_delay_models/$(basename $MODEL)_full && \
         sed -i "s/zero_data.*/data_file:\"input_state_data\"/" config.pbtxt)
    fi
  done
fi

MODELS=""
PYTHON_MODELS=""
for BACKEND in $BACKENDS; do
  if [[ $BACKEND == "custom" ]]; then
    MODELS="$MODELS ../custom_models/custom_sequence_int32"
  else
    DTYPES=$(get_datatype $BACKEND)
    for DTYPE in $DTYPES; do
      MODELS="$MODELS $DATADIR/$FIXED_MODEL_REPOSITORY/${BACKEND}_nobatch_sequence_${DTYPE}"
    done

    if [ "$ENSEMBLES" == "1" ]; then
      for DTYPE in $DTYPES; do
        # We don't generate ensemble models for bool data type.
        if [[ $DTYPE != "bool" ]]; then
          if [ "$BACKEND" == "python" ]; then
            PYTHON_MODELS="$DATADIR/qa_ensemble_model_repository/$FIXED_MODEL_REPOSITORY/*_onnx_nobatch_sequence_${DTYPE}"
            TMP=$(echo $PYTHON_MODELS)
            MODELS="$MODELS ${TMP//onnx/python}"
          else
            MODELS="$MODELS $DATADIR/qa_ensemble_model_repository/$FIXED_MODEL_REPOSITORY/*_${BACKEND}_nobatch_sequence_${DTYPE}"
          fi
        fi
      done

    fi
  fi
done

for MODEL in $MODELS; do
  if [[ "$MODEL" =~ .*"python".* ]]; then
      generate_python_models "$MODEL" "models0"
  else
    cp -r $MODEL models0/.
  fi
    (cd models0/$(basename $MODEL) && \
      sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 4/" config.pbtxt && \
      sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 4/" config.pbtxt)

  if [ "$INITIAL_STATE_FILE" == "1" ]; then
      mkdir -p models0/$(basename $MODEL)/initial_state/ && cp input_state_data models0/$(basename $MODEL)/initial_state/ && \
          (cd models0/$(basename $MODEL) && \
          sed -i "s/zero_data.*/data_file:\"input_state_data\"/" config.pbtxt)
  fi
done

# modelsv - one instance with batch-size 4
rm -fr modelsv && mkdir modelsv

MODELS=""
PYTHON_MODELS=""
for BACKEND in $BACKENDS; do
  if [[ $BACKEND == "custom" ]]; then
    MODELS="$MODELS ../custom_models/custom_sequence_int32"
  else
    DTYPES=$(get_datatype $BACKEND)
    for DTYPE in $DTYPES; do
      MODELS="$MODELS $DATADIR/${VAR_MODEL_REPOSITORY}/${BACKEND}_sequence_${DTYPE}"
    done

    if [ "$ENSEMBLES" == "1" ]; then
      for DTYPE in $DTYPES; do
        # We don't generate ensemble models for bool data type.
        if [[ $DTYPE != "bool" ]]; then
          if [ "$BACKEND" == "python" ]; then
            PYTHON_MODELS="$DATADIR/qa_ensemble_model_repository/$FIXED_MODEL_REPOSITORY/*_onnx_sequence_${DTYPE}"
            TMP=$(echo $PYTHON_MODELS)
            MODELS="$MODELS ${TMP//onnx/python}"
          else
            MODELS="$MODELS $DATADIR/qa_ensemble_model_repository/${VAR_MODEL_REPOSITORY}/*_${BACKEND}_sequence_${DTYPE}"
          fi
        fi
      done
    fi
  fi
done

for MODEL in $MODELS; do
  # Skip libtorch string models
  if [[ "$MODEL" =~ .*"libtorch".*"object".* ]]; then
      continue
  fi
  if [[ "$MODEL" =~ .*"python".* ]]; then
      generate_python_models "$MODEL" "modelsv"
  else
    cp -r $MODEL modelsv/.
  fi
    (cd modelsv/$(basename $MODEL) && \
      sed -i "s/^max_batch_size:.*/max_batch_size: 4/" config.pbtxt && \
      sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 1/" config.pbtxt && \
      sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 1/" config.pbtxt)

  if [ "$INITIAL_STATE_FILE" == "1" ]; then
      mkdir -p modelsv/$(basename $MODEL)/initial_state/ && cp input_state_data modelsv/$(basename $MODEL)/initial_state/ && \
          (cd modelsv/$(basename $MODEL) && \
          sed -i "s/zero_data.*/data_file:\"input_state_data\"/" config.pbtxt)
  fi
done

# Same test work on all models since they all have same total number
# of batch slots.
for model_trial in $MODEL_TRIALS; do
    export NO_BATCHING=1 &&
        [[ "$model_trial" != "0" ]] && export NO_BATCHING=0
    export MODEL_INSTANCES=1 &&
        [[ "$model_trial" != "v" ]] && export MODEL_INSTANCES=4 &&
        [[ "$model_trial" != "0" ]] && export MODEL_INSTANCES=$model_trial

    MODEL_PATH=models${model_trial}

    if [ "$ENSEMBLES" == "1" ]; then
      cp -r $DATADIR/qa_ensemble_model_repository/${FIXED_MODEL_REPOSITORY}/nop_* `pwd`/$MODEL_PATH/.
        create_nop_version_dir `pwd`/$MODEL_PATH
      # Must load identity backend on GPU to avoid cuda init delay during 1st run
      for NOP_MODEL in `pwd`/$MODEL_PATH/nop_*; do
        (cd $NOP_MODEL && sed -i "s/kind: KIND_CPU/kind: KIND_GPU/" config.pbtxt)
      done
    fi

    # Need to launch the server for each test so that the model status
    # is reset (which is used to make sure the correct batch size was
    # used for execution). Test everything with fixed-tensor-size
    # models and variable-tensor-size models.
    export BATCHER_TYPE="VARIABLE" &&
        [[ "$model_trial" != "v" ]] && export BATCHER_TYPE="FIXED"

    for i in $NO_DELAY_TESTS; do
        SERVER_ARGS="--model-repository=$MODELDIR/$MODEL_PATH ${SERVER_ARGS_EXTRA}"
        SERVER_LOG="./$i.$MODEL_PATH.server.log"

        if [ "$TEST_VALGRIND" -eq 1 ]; then
            LEAKCHECK_LOG="./$i.$MODEL_PATH.valgrind.log"
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

        echo "Test: $i, repository $MODEL_PATH" >>$CLIENT_LOG

        set +e
        python3 $BATCHER_TEST SequenceBatcherTest.$i >>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test $i Failed\n***" >>$CLIENT_LOG
            echo -e "\n***\n*** Test $i Failed\n***"
            RET=1
        else
            check_test_results $TEST_RESULT_FILE 1
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

    # Tests that require TRITONSERVER_DELAY_SCHEDULER so that the
    # scheduler is delayed and requests can collect in the queue.
    for i in $DELAY_TESTS; do
        export TRITONSERVER_BACKLOG_DELAY_SCHEDULER=3 &&
            [[ "$i" != "test_backlog_fill_no_end" ]] && export TRITONSERVER_BACKLOG_DELAY_SCHEDULER=2 &&
            [[ "$i" != "test_backlog_fill" ]] &&
            [[ "$i" != "test_backlog_same_correlation_id" ]] && export TRITONSERVER_BACKLOG_DELAY_SCHEDULER=0
        export TRITONSERVER_DELAY_SCHEDULER=10 &&
            [[ "$i" != "test_backlog_fill_no_end" ]] &&
            [[ "$i" != "test_backlog_fill" ]] && export TRITONSERVER_DELAY_SCHEDULER=16 &&
            [[ "$i" != "test_backlog_same_correlation_id_no_end" ]] && export TRITONSERVER_DELAY_SCHEDULER=8 &&
            [[ "$i" != "test_half_batch" ]] && export TRITONSERVER_DELAY_SCHEDULER=4 &&
            [[ "$i" != "test_backlog_sequence_timeout" ]] && export TRITONSERVER_DELAY_SCHEDULER=12
        SERVER_ARGS="--model-repository=$MODELDIR/$MODEL_PATH ${SERVER_ARGS_EXTRA}"
        SERVER_LOG="./$i.$MODEL_PATH.server.log"

        if [ "$TEST_VALGRIND" -eq 1 ]; then
            LEAKCHECK_LOG="./$i.$MODEL_PATH.valgrind.log"
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

        echo "Test: $i, repository $MODEL_PATH" >>$CLIENT_LOG

        set +e
        python3 $BATCHER_TEST SequenceBatcherTest.$i >>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test $i Failed\n***" >>$CLIENT_LOG
            echo -e "\n***\n*** Test $i Failed\n***"
            RET=1
        else
            check_test_results $TEST_RESULT_FILE 1
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG
                echo -e "\n***\n*** Test Result Verification Failed\n***"
                RET=1
            fi
        fi
        set -e

        unset TRITONSERVER_DELAY_SCHEDULER
        unset TRITONSERVER_BACKLOG_DELAY_SCHEDULER
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
done

# ragged models
if [[ $BACKENDS == *"custom"* ]]; then
  rm -fr ragged_models && mkdir ragged_models
  cp -r ../custom_models/custom_sequence_int32 ragged_models/.
  (cd ragged_models/custom_sequence_int32 && \
          sed -i "s/name:.*\"INPUT\"/name: \"INPUT\"\\nallow_ragged_batch: true/" config.pbtxt)

  export NO_BATCHING=0
  export MODEL_INSTANCES=1
  export BATCHER_TYPE="FIXED"
  MODEL_PATH=ragged_models

  # Need to launch the server for each test so that the model status
  # is reset (which is used to make sure the correct batch size was
  # used for execution). Test everything with fixed-tensor-size
  # models and variable-tensor-size models.
  for i in test_ragged_batch_allowed ; do
    export TRITONSERVER_BACKLOG_DELAY_SCHEDULER=0
    export TRITONSERVER_DELAY_SCHEDULER=12

    SERVER_ARGS="--model-repository=$MODELDIR/$MODEL_PATH ${SERVER_ARGS_EXTRA}"
    SERVER_LOG="./$i.$MODEL_PATH.server.log"

    if [ "$TEST_VALGRIND" -eq 1 ]; then
      LEAKCHECK_LOG="./$i.$MODEL_PATH.valgrind.log"
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

    echo "Test: $i, repository $MODEL_PATH" >>$CLIENT_LOG

    set +e
    python3 $BATCHER_TEST SequenceBatcherTest.$i >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
      echo -e "\n***\n*** Test $i Failed\n***" >>$CLIENT_LOG
      echo -e "\n***\n*** Test $i Failed\n***"
      RET=1
    else
      check_test_results $TEST_RESULT_FILE 1
      if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
      fi
    fi
    set -e

    unset TRITONSERVER_DELAY_SCHEDULER
    unset TRITONSERVER_BACKLOG_DELAY_SCHEDULER
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
fi

# max queue delay
MODEL_PATH=queue_delay_models
# remove ensemble models from the test model repo
rm -rf queue_delay_models/simple_* queue_delay_models/fan_* queue_delay_models/sequence_*
for i in $QUEUE_DELAY_TESTS ; do
    export NO_BATCHING=0
    export TRITONSERVER_BACKLOG_DELAY_SCHEDULER=0
    export TRITONSERVER_DELAY_SCHEDULER=2
    SERVER_ARGS="--model-repository=$MODELDIR/$MODEL_PATH ${SERVER_ARGS_EXTRA}"
    SERVER_LOG="./$i.$MODEL_PATH.server.log"

    if [ "$TEST_VALGRIND" -eq 1 ]; then
        LEAKCHECK_LOG="./$i.$MODEL_PATH.valgrind.log"
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

    echo "Test: $i, repository $MODEL_PATH" >>$CLIENT_LOG

    set +e
    python3 $BATCHER_TEST SequenceBatcherTest.$i >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test $i Failed\n***" >>$CLIENT_LOG
        echo -e "\n***\n*** Test $i Failed\n***"
        RET=1
    else
        check_test_results $TEST_RESULT_FILE 1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Result Verification Failed\n***"
            RET=1
        fi
    fi
    set -e

    unset TRITONSERVER_DELAY_SCHEDULER
    unset TRITONSERVER_BACKLOG_DELAY_SCHEDULER
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

# Test request timeout with sequence batcher
# only run the test outside shared memory setting as
# shared memory feature is irrelevant
if [ "$TEST_SYSTEM_SHARED_MEMORY" -ne 1 ] && [ "$TEST_CUDA_SHARED_MEMORY" -ne 1 ]; then
    export NO_BATCHING=0
    export MODEL_INSTANCES=1
    export BATCHER_TYPE="FIXED"

    TEST_CASE=SequenceBatcherRequestTimeoutTest
    MODEL_PATH=request_timeout_models
    mkdir -p ${MODEL_PATH}/custom_sequence_int32_timeout/1

    SERVER_ARGS="--model-repository=$MODELDIR/$MODEL_PATH ${SERVER_ARGS_EXTRA}"
    SERVER_LOG="./$TEST_CASE.$MODEL_PATH.server.log"

    if [ "$TEST_VALGRIND" -eq 1 ]; then
        LEAKCHECK_LOG="./$i.$MODEL_PATH.valgrind.log"
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

    echo "Test: $TEST_CASE, repository $MODEL_PATH" >>$CLIENT_LOG

    set +e
    python3 $BATCHER_TEST $TEST_CASE >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test $TEST_CASE Failed\n***" >>$CLIENT_LOG
        echo -e "\n***\n*** Test $TEST_CASE Failed\n***"
        RET=1
    else
        check_test_results $TEST_RESULT_FILE 2
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
fi

### Start Preserve Ordering Tests ###

# Test only supported on windows currently due to use of python backend models
if [ ${WINDOWS} -ne 1 ]; then
    # Test preserve ordering true/false and decoupled/non-decoupled
    TEST_CASE=SequenceBatcherPreserveOrderingTest
    MODEL_PATH=preserve_ordering_models
    BASE_MODEL="../python_models/sequence_py"
    rm -rf ${MODEL_PATH}

    # FIXME [DLIS-5280]: This may fail for decoupled models if writes to GRPC
    # stream are done out of order in server, so decoupled tests are disabled.
    MODES="decoupled nondecoupled"
    for mode in $MODES; do
        NO_PRESERVE="${MODEL_PATH}/seqpy_no_preserve_ordering_${mode}"
        mkdir -p ${NO_PRESERVE}/1
        cp ${BASE_MODEL}/config.pbtxt ${NO_PRESERVE}
        cp ${BASE_MODEL}/model.py ${NO_PRESERVE}/1

        PRESERVE="${MODEL_PATH}/seqpy_preserve_ordering_${mode}"
        cp -r ${NO_PRESERVE} ${PRESERVE}
        sed -i "s/^preserve_ordering: False/preserve_ordering: True/" ${PRESERVE}/config.pbtxt

        if [ ${mode} == "decoupled" ]; then
          echo -e "\nmodel_transaction_policy { decoupled: true }" >> ${NO_PRESERVE}/config.pbtxt
          echo -e "\nmodel_transaction_policy { decoupled: true }" >> ${PRESERVE}/config.pbtxt
        fi
    done

    SERVER_ARGS="--model-repository=$MODELDIR/$MODEL_PATH ${SERVER_ARGS_EXTRA}"
    SERVER_LOG="./$TEST_CASE.$MODEL_PATH.server.log"

    if [ "$TEST_VALGRIND" -eq 1 ]; then
        LEAKCHECK_LOG="./$i.$MODEL_PATH.valgrind.log"
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

    echo "Test: $TEST_CASE, repository $MODEL_PATH" >>$CLIENT_LOG

    set +e
    python3 $BATCHER_TEST $TEST_CASE >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test $TEST_CASE Failed\n***" >>$CLIENT_LOG
        echo -e "\n***\n*** Test $TEST_CASE Failed\n***"
        RET=1
    else
        # 2 for preserve_ordering = True/False
        check_test_results $TEST_RESULT_FILE 2
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
fi

### End Preserve Ordering Tests ###

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
