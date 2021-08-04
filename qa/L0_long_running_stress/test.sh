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

export CUDA_VISIBLE_DEVICES=0

CLIENT_LOG="./client.log"
STRESS_TEST=stress.py

DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

RET=0

# If BACKENDS not specified, set to all
BACKENDS=${BACKENDS:="graphdef savedmodel onnx plan custom"}
export BACKENDS

# If MODEL_TRIALS not specified set to 1 2 4
MODEL_TRIALS=${MODEL_TRIALS:="1 2 4"}

# Setup model repository.
#   models1 - one instance with batch-size 4
#   models2 - two instances with batch-size 2
#   models4 - four instances with batch-size 1
rm -fr *.log *.serverlog models{1,2,4} && mkdir models{1,2,4}

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
    DTYPE=$(get_datatype $BACKEND)
    MODELS="$MODELS $DATADIR/qa_sequence_model_repository/${BACKEND}_nobatch_sequence_${DTYPE}"
done

for MODEL in $MODELS; do
    cp -r $MODEL models1/. && \
      (cd models1/$(basename $MODEL) && \
        sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 1/" config.pbtxt && \
        sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 1/" config.pbtxt)
    cp -r $MODEL models2/. && \
      (cd models2/$(basename $MODEL) && \
        sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 2/" config.pbtxt && \
        sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 2/" config.pbtxt)
    cp -r $MODEL models4/. && \
      (cd models4/$(basename $MODEL) && \
        sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 4/" config.pbtxt && \
        sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 4/" config.pbtxt)
done

mkdir -p models/custom_identity_int32/1

for model_trial in $MODEL_TRIALS; do
    MODEL_DIR=models${model_trial}
    cp -r $DATADIR/tf_model_store/resnet_v1_50_graphdef $MODEL_DIR/resnet_v1_50_graphdef_def && \
    (cd $MODEL_DIR/resnet_v1_50_graphdef_def && \
            sed -i 's/^name: "resnet_v1_50_graphdef"/name: "resnet_v1_50_graphdef_def"/' config.pbtxt && \
            echo "optimization { }" >> config.pbtxt)
    cp -r `pwd`/models/. $MODEL_DIR
done

# Stress-test each model repository
for model_trial in $MODEL_TRIALS; do
    MODEL_DIR=models${model_trial}
    SERVER_ARGS="--model-repository=`pwd`/$MODEL_DIR"
    SERVER_LOG="./$MODEL_DIR.serverlog"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e
    python $STRESS_TEST >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    set -e

    kill $SERVER_PID
    wait $SERVER_PID
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
