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

# Must run on a single device or else the TRITONSERVER_DELAY_SCHEDULER
# can fail when the requests are distributed to multiple devices.
export CUDA_VISIBLE_DEVICES=0

CLIENT_LOG="./client.log"
BATCHER_TEST=sequence_batcher_test.py

DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
OPTDIR=${OPTDIR:="/opt"}
SERVER=${OPTDIR}/tritonserver/bin/tritonserver

source ../common/util.sh

RET=0

# If BACKENDS not specified, set to all
BACKENDS=${BACKENDS:="graphdef savedmodel netdef onnx plan custom"}
export BACKENDS

# If ENSEMBLES not specified, set to 1
ENSEMBLES=${ENSEMBLES:="1"}
export ENSEMBLES

# Setup non-variable-size model repositories. The same models are in each
# repository but they are configured as:
#   models0 - four instance with non-batching model
#   models1 - one instance with batch-size 4
#   models2 - two instances with batch-size 2
#   models4 - four instances with batch-size 1
rm -fr *.log *.serverlog models{0,1,2,4} && mkdir models{0,1,2,4}

# Get the datatype to use based on the backend
function get_datatype () {
  local dtype='int32'
  if [[ $1 == "plan" ]] || [[ $1 == "savedmodel" ]]; then
      dtype='float32'
  elif [[ $1 == "graphdef" ]]; then
      dtype='object'
  fi
  echo $dtype
}

MODELS=""
for BACKEND in $BACKENDS; do
  if [[ $BACKEND == "custom" ]]; then
    MODELS="$MODELS ../custom_models/custom_sequence_int32"
  else
    DTYPE=$(get_datatype $BACKEND)
    MODELS="$MODELS $DATADIR/qa_sequence_model_repository/${BACKEND}_sequence_${DTYPE}"

    if [[ $BACKEND == "graphdef" ]]; then
      MODELS="$MODELS $DATADIR/qa_sequence_model_repository/graphdef_sequence_int32"
    fi

    if [ "$ENSEMBLES" == "1" ]; then
      MODELS="$MODELS $DATADIR/qa_ensemble_model_repository/qa_sequence_model_repository/*_${BACKEND}_sequence_${DTYPE}"
    fi
  fi
done

for MODEL in $MODELS; do
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
done

MODELS=""
for BACKEND in $BACKENDS; do
  if [[ $BACKEND == "custom" ]]; then
    MODELS="$MODELS ../custom_models/custom_sequence_int32"
  else
    DTYPE=$(get_datatype $BACKEND)
    MODELS="$MODELS $DATADIR/qa_sequence_model_repository/${BACKEND}_nobatch_sequence_${DTYPE}"

    if [[ $BACKEND == "graphdef" ]]; then
      MODELS="$MODELS $DATADIR/qa_sequence_model_repository/graphdef_nobatch_sequence_int32"
    fi

    if [ "$ENSEMBLES" == "1" ]; then
      MODELS="$MODELS $DATADIR/qa_ensemble_model_repository/qa_sequence_model_repository/*_${BACKEND}_nobatch_sequence_${DTYPE}"

      if [[ $BACKEND == "graphdef" ]]; then
        MODELS="$MODELS $DATADIR/qa_ensemble_model_repository/qa_sequence_model_repository/*_graphdef_nobatch_sequence_int32"
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

#   modelsv - one instance with batch-size 4
rm -fr modelsv && mkdir modelsv

MODELS=""
for BACKEND in $BACKENDS; do
  if [[ $BACKEND == "custom" ]]; then
    MODELS="$MODELS ../custom_models/custom_sequence_int32"
  else
    DTYPE=$(get_datatype $BACKEND)
    MODELS="$MODELS $DATADIR/qa_variable_sequence_model_repository/${BACKEND}_sequence_${DTYPE}"

    if [ "$ENSEMBLES" == "1" ]; then
      MODELS="$MODELS $DATADIR/qa_ensemble_model_repository/qa_variable_sequence_model_repository/*_${BACKEND}_sequence_${DTYPE}"
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
for model_trial in v 0 1 2 4; do
    export NO_BATCHING=1 &&
        [[ "$model_trial" != "0" ]] && export NO_BATCHING=0
    export MODEL_INSTANCES=1 &&
        [[ "$model_trial" != "v" ]] && export MODEL_INSTANCES=4 &&
        [[ "$model_trial" != "0" ]] && export MODEL_INSTANCES=$model_trial

    MODEL_DIR=models${model_trial}

    if [ "$ENSEMBLES" == "1" ]; then
      cp -r $DATADIR/qa_ensemble_model_repository/qa_sequence_model_repository/nop_* `pwd`/$MODEL_DIR/.
      create_nop_modelfile `pwd`/libidentity.so `pwd`/$MODEL_DIR
    fi

    # Need to launch the server for each test so that the model status
    # is reset (which is used to make sure the correct batch size was
    # used for execution). Test everything with fixed-tensor-size
    # models and variable-tensor-size models.
    export BATCHER_TYPE="VARIABLE" &&
        [[ "$model_trial" != "v" ]] && export BATCHER_TYPE="FIXED"

    for i in \
            test_simple_sequence \
            test_length1_sequence \
            test_batch_size \
            test_no_sequence_start \
            test_no_sequence_start2 \
            test_no_sequence_end \
            test_no_correlation_id ; do
        SERVER_ARGS="--model-repository=`pwd`/$MODEL_DIR"
        SERVER_LOG="./$i.$MODEL_DIR.serverlog"
        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        echo "Test: $i, repository $MODEL_DIR" >>$CLIENT_LOG

        set +e
        python $BATCHER_TEST SequenceBatcherTest.$i >>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test $i Failed\n***" >>$CLIENT_LOG
            echo -e "\n***\n*** Test $i Failed\n***"
            RET=1
        else
            check_test_results $CLIENT_LOG 1
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG
                echo -e "\n***\n*** Test Failed\n***"
                RET=1
            fi
        fi
        set -e

        kill $SERVER_PID
        wait $SERVER_PID
    done

    # Tests that require TRITONSERVER_DELAY_SCHEDULER so that the
    # scheduler is delayed and requests can collect in the queue.
    for i in \
            test_backlog_fill \
            test_backlog_fill_no_end \
            test_backlog_same_correlation_id \
            test_backlog_same_correlation_id_no_end \
            test_backlog_sequence_timeout \
            test_half_batch \
            test_skip_batch \
            test_full_batch \
            test_ragged_batch \
            test_backlog ; do
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
        SERVER_ARGS="--model-repository=`pwd`/$MODEL_DIR"
        SERVER_LOG="./$i.$MODEL_DIR.serverlog"
        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        echo "Test: $i, repository $MODEL_DIR" >>$CLIENT_LOG

        set +e
        python $BATCHER_TEST SequenceBatcherTest.$i >>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test $i Failed\n***" >>$CLIENT_LOG
            echo -e "\n***\n*** Test $i Failed\n***"
            RET=1
        else
            check_test_results $CLIENT_LOG 1
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG
                echo -e "\n***\n*** Test Failed\n***"
                RET=1
            fi
        fi
        set -e

        unset TRITONSERVER_DELAY_SCHEDULER
        unset TRITONSERVER_BACKLOG_DELAY_SCHEDULER
        kill $SERVER_PID
        wait $SERVER_PID
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
  MODEL_DIR=ragged_models

  # Need to launch the server for each test so that the model status
  # is reset (which is used to make sure the correct batch size was
  # used for execution). Test everything with fixed-tensor-size
  # models and variable-tensor-size models.
  for i in test_ragged_batch_allowed ; do
      export TRITONSERVER_BACKLOG_DELAY_SCHEDULER=0
      export TRITONSERVER_DELAY_SCHEDULER=12

      SERVER_ARGS="--model-repository=`pwd`/$MODEL_DIR"
      SERVER_LOG="./$i.$MODEL_DIR.serverlog"
      run_server
      if [ "$SERVER_PID" == "0" ]; then
          echo -e "\n***\n*** Failed to start $SERVER\n***"
          cat $SERVER_LOG
          exit 1
      fi

      echo "Test: $i, repository $MODEL_DIR" >>$CLIENT_LOG

      set +e
      python $BATCHER_TEST SequenceBatcherTest.$i >>$CLIENT_LOG 2>&1
      if [ $? -ne 0 ]; then
          echo -e "\n***\n*** Test $i Failed\n***" >>$CLIENT_LOG
          echo -e "\n***\n*** Test $i Failed\n***"
          RET=1
      else
          check_test_results $CLIENT_LOG 1
          if [ $? -ne 0 ]; then
              cat $CLIENT_LOG
              echo -e "\n***\n*** Test Failed\n***"
              RET=1
          fi
      fi
      set -e

      unset TRITONSERVER_DELAY_SCHEDULER
      unset TRITONSERVER_BACKLOG_DELAY_SCHEDULER
      kill $SERVER_PID
      wait $SERVER_PID
  done
fi

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
