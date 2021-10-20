#!/bin/bash
# Copyright 2018-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
export CUDA_VISIBLE_DEVICES=0

CLIENT_LOG="./client.log"
BATCHER_TEST=sequence_batcher_test.py

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

TF_VERSION=${TF_VERSION:=1}

# On windows the paths invoked by the script (running in WSL) must use
# /mnt/c when needed but the paths on the tritonserver command-line
# must be C:/ style.
if [[ "$(< /proc/sys/kernel/osrelease)" == *Microsoft ]]; then
    MODELDIR=${MODELDIR:=C:/models}
    DATADIR=${DATADIR:="/mnt/c/data/inferenceserver/${REPO_VERSION}"}
    BACKEND_DIR=${BACKEND_DIR:=C:/tritonserver/backends}
    SERVER=${SERVER:=/mnt/c/tritonserver/bin/tritonserver.exe}
    export WSLENV=$WSLENV:TRITONSERVER_DELAY_SCHEDULER:TRITONSERVER_BACKLOG_DELAY_SCHEDULER
else
    MODELDIR=${MODELDIR:=`pwd`}
    DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
    TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
    SERVER=${TRITON_DIR}/bin/tritonserver
    BACKEND_DIR=${TRITON_DIR}/backends
fi

SERVER_ARGS_EXTRA="--backend-directory=${BACKEND_DIR} --backend-config=tensorflow,version=${TF_VERSION}"

source ../common/util.sh

RET=0

# If BACKENDS not specified, set to all
BACKENDS=${BACKENDS:="graphdef savedmodel onnx plan custom"}
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

# Setup non-variable-size model repositories. The same models are in each
# repository but they are configured as:
#   models0 - four instances with non-batching model
#   models1 - one instance with batch-size 4
#   models2 - two instances with batch-size 2
#   models4 - four instances with batch-size 1
rm -fr *.log *.serverlog models{0,1,2,4} queue_delay_models && mkdir models{0,1,2,4} queue_delay_models

# Get the datatype to use based on the backend
function get_datatype () {
  local dtype="int32 bool"
  if [[ $1 == "plan" ]]; then
    dtype="float32"
  elif [[ $1 == "savedmodel" ]]; then
    dtype="float32 bool"
  elif [[ $1 == "graphdef" ]]; then
    dtype="object bool"
  fi
  echo $dtype
}

FIXED_MODEL_REPOSITORY=''
VAR_MODEL_REPOSITORY=''
if [ "$IMPLICIT_STATE" == "1" ]; then
  FIXED_MODEL_REPOSITORY="qa_sequence_implicit_model_repository"
  VAR_MODEL_REPOSITORY="qa_variable_sequence_implicit_model_repository"
else
  FIXED_MODEL_REPOSITORY="qa_sequence_model_repository"
  VAR_MODEL_REPOSITORY="qa_variable_sequence_model_repository"
fi

MODELS=""
for BACKEND in $BACKENDS; do
  if [[ $BACKEND == "custom" ]]; then
    MODELS="$MODELS ../custom_models/custom_sequence_int32"
  else
    DTYPES=$(get_datatype $BACKEND)

    for DTYPE in $DTYPES; do
      MODELS="$MODELS $DATADIR/$FIXED_MODEL_REPOSITORY/${BACKEND}_sequence_${DTYPE}"
    done

    if [[ $BACKEND == "graphdef" ]]; then
      MODELS="$MODELS $DATADIR/$FIXED_MODEL_REPOSITORY/${BACKEND}_sequence_graphdef_sequence_int32"
    fi

    if [ "$ENSEMBLES" == "1" ]; then
      for DTYPE in $DTYPES; do
        MODELS="$MODELS $DATADIR/qa_ensemble_model_repository/$FIXED_MODEL_REPOSITORY/*_${BACKEND}_sequence_${DTYPE}"
      done
    fi
  fi
done

for MODEL in $MODELS; do
  if [[ ! "$TEST_VALGRIND" -eq 1 ]]; then
    cp -r $MODEL models1/. && \
      (cd models1/$(basename $MODEL) && \
        sed -i "s/^max_batch_size:.*/max_batch_size: 4/" config.pbtxt && \
        sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 1/" config.pbtxt && \
        sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 1/" config.pbtxt)
    cp -r $MODEL models2/. && \
      (cd models2/$(basename $MODEL) && \
        sed -i "s/^max_batch_size:.*/max_batch_size: 2/" config.pbtxt && \
        sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 2/" config.pbtxt && \
        sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 2/" config.pbtxt)
    cp -r $MODEL models4/. && \
      (cd models4/$(basename $MODEL) && \
        sed -i "s/^max_batch_size:.*/max_batch_size: 1/" config.pbtxt && \
        sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 4/" config.pbtxt && \
        sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 4/" config.pbtxt)
    # Duplicate the models for different delay settings
    cp -r $MODEL queue_delay_models/. && \
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

MODELS=""
for BACKEND in $BACKENDS; do
  if [[ $BACKEND == "custom" ]]; then
    MODELS="$MODELS ../custom_models/custom_sequence_int32"
  else
    DTYPES=$(get_datatype $BACKEND)
    for DTYPE in $DTYPES; do
      MODELS="$MODELS $DATADIR/$FXIED_MODEL_REPOSITORY/${BACKEND}_nobatch_sequence_${DTYPE}"
    done

    if [[ $BACKEND == "graphdef" ]]; then
      MODELS="$MODELS $DATADIR/$FIXED_MODEL_REPOSITORY/graphdef_nobatch_sequence_int32"
    fi

    if [ "$ENSEMBLES" == "1" ]; then
      for DTYPE in $DTYPES; do
      MODELS="$MODELS $DATADIR/qa_ensemble_model_repository/$FIXED_MODEL_REPOSITORY/*_${BACKEND}_nobatch_sequence_${DTYPE}"
      done

      if [[ $BACKEND == "graphdef" ]]; then
        MODELS="$MODELS $DATADIR/qa_ensemble_model_repository/$FIXED_MODEL_REPOSITORY/*_graphdef_nobatch_sequence_int32"
      fi
    fi
  fi
done

for MODEL in $MODELS; do
    cp -r $MODEL models0/. && \
        (cd models0/$(basename $MODEL) && \
            sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 4/" config.pbtxt && \
            sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 4/" config.pbtxt)
done

# modelsv - one instance with batch-size 4
rm -fr modelsv && mkdir modelsv

MODELS=""
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
        MODELS="$MODELS $DATADIR/qa_ensemble_model_repository/${VAR_MODEL_REPOSITORY}/*_${BACKEND}_sequence_${DTYPE}"
        done
    fi
  fi
done

for MODEL in $MODELS; do
    cp -r $MODEL modelsv/. && \
        (cd modelsv/$(basename $MODEL) && \
            sed -i "s/^max_batch_size:.*/max_batch_size: 4/" config.pbtxt && \
            sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 1/" config.pbtxt && \
            sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 1/" config.pbtxt)
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
        SERVER_LOG="./$i.$MODEL_PATH.serverlog"

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
        SERVER_LOG="./$i.$MODEL_PATH.serverlog"

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
    SERVER_LOG="./$i.$MODEL_PATH.serverlog"

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
    SERVER_LOG="./$i.$MODEL_PATH.serverlog"

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

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
