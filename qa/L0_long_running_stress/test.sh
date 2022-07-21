#!/bin/bash
# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

CLIENT_LOG="./client.log"
STRESS_TEST=stress.py

DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

# If the test should be run in long and high load setting
if [ "$TRITON_PERF_LONG" == 1 ]; then
    # ~ 6.5 days
    TEST_DURATION=480000
    LOAD_THREAD_COUNT=2
    EMAIL_SUBJECT="Long"
else
    # ~ 7 hours
    TEST_DURATION=25000
    LOAD_THREAD_COUNT=0
    EMAIL_SUBJECT=""
fi

RET=0

# If BACKENDS not specified, set to all
BACKENDS=${BACKENDS:="graphdef savedmodel onnx libtorch"}
export BACKENDS

export CI_JOB_ID=${CI_JOB_ID}

MODEL_DIR=models

rm -fr *.log *.txt *.serverlog models validation_data csv_dir && mkdir models validation_data csv_dir

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

# Setup model repository - two instances with batch-size 2
MODELS=""
for BACKEND in $BACKENDS; do
  DTYPE=$(get_datatype $BACKEND)
  MODELS="$MODELS $DATADIR/qa_sequence_model_repository/${BACKEND}_sequence_${DTYPE}"
done

for MODEL in $MODELS; do
    cp -r $MODEL $MODEL_DIR/. && \
      (cd $MODEL_DIR/$(basename $MODEL) && \
        sed -i "s/^max_batch_size:.*/max_batch_size: 2/" config.pbtxt && \
        sed -i "s/max_sequence_idle_microseconds:.*/max_sequence_idle_microseconds: 7000000/" config.pbtxt && \
        sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 2/" config.pbtxt && \
        sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 2/" config.pbtxt)
done

MODELS=""
for BACKEND in $BACKENDS; do
    DTYPE=$(get_datatype $BACKEND)
    MODELS="$MODELS $DATADIR/qa_sequence_model_repository/${BACKEND}_nobatch_sequence_${DTYPE}"
done

for MODEL in $MODELS; do
    cp -r $MODEL $MODEL_DIR/. && \
      (cd $MODEL_DIR/$(basename $MODEL) && \
        sed -i "s/max_sequence_idle_microseconds:.*/max_sequence_idle_microseconds: 7000000/" config.pbtxt && \
        sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 2/" config.pbtxt && \
        sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 2/" config.pbtxt)
done

MODELS=""
for BACKEND in $BACKENDS; do
    MODELS="$MODELS $DATADIR/qa_identity_model_repository/${BACKEND}_nobatch_zero_1_float32"
done

for MODEL in $MODELS; do
    cp -r $MODEL $MODEL_DIR/. && \
      (cd $MODEL_DIR/$(basename $MODEL) && \
        echo "parameters [" >> config.pbtxt && \
        echo "{ key: \"execute_delay_ms\"; value: { string_value: \"1000\" }}" >> config.pbtxt && \
        echo "]" >> config.pbtxt)
done
cp -r ../custom_models/custom_zero_1_float32 $MODEL_DIR/custom_zero_1_float32 && \
  mkdir $MODEL_DIR/custom_zero_1_float32/1 && \
  (cd $MODEL_DIR/custom_zero_1_float32 && \
    echo "parameters [" >> config.pbtxt && \
        echo "{ key: \"execute_delay_ms\"; value: { string_value: \"10000\" }}" >> config.pbtxt && \
        echo "]" >> config.pbtxt)

cp -r $DATADIR/tf_model_store/resnet_v1_50_graphdef $MODEL_DIR/resnet_v1_50_graphdef_def && \
  (cd $MODEL_DIR/resnet_v1_50_graphdef_def && \
    sed -i 's/^name: "resnet_v1_50_graphdef"/name: "resnet_v1_50_graphdef_def"/' config.pbtxt && \
    echo "optimization { }" >> config.pbtxt)

SERVER_ARGS="--model-repository=`pwd`/$MODEL_DIR"
SERVER_LOG="./serverlog"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $STRESS_TEST -d ${TEST_DURATION} --load-thread ${LOAD_THREAD_COUNT} >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

# Run only if both TRITON_FROM and TRITON_TO_DL are set
if [[ ! -z "$TRITON_FROM" ]] && [[ ! -z "$TRITON_TO_DL" ]]; then
    python stress_mail.py "$EMAIL_SUBJECT"
fi

exit $RET
