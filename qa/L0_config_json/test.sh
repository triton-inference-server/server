#!/bin/bash
# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

DATADIR="/data/inferenceserver/${REPO_VERSION}"
CLIENT_LOG="./client.log"
SERVER_LOG="./inference_server.log"

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=`pwd`/models"
source ../common/util.sh

RET=0
rm -fr *.log

rm -fr models && mkdir models
cp -r $DATADIR/qa_model_repository/savedmodel_nobatch_float32_float32_float32 models/.

# Test input and output dims are shown as numbers
TRIAL=ios

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
code=`curl -s -w %{http_code} -o ./$TRIAL.out localhost:8000/v2/models/savedmodel_nobatch_float32_float32_float32/config`
set -e
if [ "$code" != "200" ]; then
    cat $TRIAL.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

matches=`grep -o "\"dims\":\[16\]" $TRIAL.out | wc -l`
if [ $matches -ne 4 ]; then
    cat $TRIAL.out
    echo -e "\n***\n*** Expected 4 dims, got $matches\n***"
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

# Test input and output reshape are shown as numbers
TRIAL=reshape

rm -fr models && mkdir models
cp -r $DATADIR/qa_model_repository/savedmodel_nobatch_float32_float32_float32 models/.
(cd models/savedmodel_nobatch_float32_float32_float32 && \
     sed -i "s/data_type:.*TYPE_FP32/data_type: TYPE_FP32\nreshape: { shape: [ 16 ]}/g" config.pbtxt)

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
code=`curl -s -w %{http_code} -o ./$TRIAL.out localhost:8000/v2/models/savedmodel_nobatch_float32_float32_float32/config`
set -e
if [ "$code" != "200" ]; then
    cat $TRIAL.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

matches=`grep -o "\"reshape\":{\"shape\":\[16\]}" $TRIAL.out | wc -l`
if [ $matches -ne 4 ]; then
    cat $TRIAL.out
    echo -e "\n***\n*** Expected 4 reshape:shape, got $matches\n***"
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

# Test version_policy::specific
TRIAL=specific

rm -fr models && mkdir models
cp -r $DATADIR/qa_model_repository/savedmodel_nobatch_float32_float32_float32 models/.
(cd models/savedmodel_nobatch_float32_float32_float32 && \
    sed -i "s/^version_policy:.*/version_policy: { specific: { versions: [1] }}/" config.pbtxt)

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
code=`curl -s -w %{http_code} -o ./$TRIAL.out localhost:8000/v2/models/savedmodel_nobatch_float32_float32_float32/config`
set -e
if [ "$code" != "200" ]; then
    cat $TRIAL.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

matches=`grep -o "\"version_policy\":{\"specific\":{\"versions\":\[1\]}}" $TRIAL.out | wc -l`
if [ $matches -ne 1 ]; then
    cat $TRIAL.out
    echo -e "\n***\n*** Expected 1 version_policy:specific:versions, got $matches\n***"
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

# Test dynamic_batching::max_queue_delay_microseconds,
# dynamic_batching::default_queue_policy::default_timeout_microseconds,
# dynamic_batching::priority_queue_policy::value::default_timeout_microseconds
TRIAL=dbatch

rm -fr models && mkdir models
cp -r $DATADIR/qa_model_repository/savedmodel_nobatch_float32_float32_float32 models/.
(cd models/savedmodel_nobatch_float32_float32_float32 && \
     echo "dynamic_batching: { max_queue_delay_microseconds: 42 \
          default_queue_policy: { default_timeout_microseconds: 123 } \
          priority_queue_policy: { key: 1  value: { default_timeout_microseconds: 123 }} \
          priority_queue_policy: { key: 2  value: { default_timeout_microseconds: 123 }}}" >> config.pbtxt)

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
code=`curl -s -w %{http_code} -o ./$TRIAL.out localhost:8000/v2/models/savedmodel_nobatch_float32_float32_float32/config`
set -e
if [ "$code" != "200" ]; then
    cat $TRIAL.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

matches=`grep -o "\"dynamic_batching\":{" $TRIAL.out | wc -l`
if [ $matches -ne 1 ]; then
    cat $TRIAL.out
    echo -e "\n***\n*** Expected 1 dynamic_batching, got $matches\n***"
    RET=1
fi

matches=`grep -o "\"max_queue_delay_microseconds\":42" $TRIAL.out | wc -l`
if [ $matches -ne 1 ]; then
    cat $TRIAL.out
    echo -e "\n***\n*** Expected 1 dynamic_batching:max_queue_delay_microseconds, got $matches\n***"
    RET=1
fi

matches=`grep -o "\"default_timeout_microseconds\":123" $TRIAL.out | wc -l`
if [ $matches -ne 3 ]; then
    cat $TRIAL.out
    echo -e "\n***\n*** Expected 3 dynamic_batching:*_queue_policy:default_timeout_microseconds, got $matches\n***"
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

# Test sequence_batching::oldest::max_queue_delay_microseconds,
# sequence_batching::max_sequence_idle_microseconds
TRIAL=sbatch

rm -fr models && mkdir models
cp -r $DATADIR/qa_model_repository/savedmodel_nobatch_float32_float32_float32 models/.
(cd models/savedmodel_nobatch_float32_float32_float32 && \
     echo "sequence_batching: { max_sequence_idle_microseconds: 42 \
          oldest: { max_queue_delay_microseconds: 987 }}" >> config.pbtxt)

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
code=`curl -s -w %{http_code} -o ./$TRIAL.out localhost:8000/v2/models/savedmodel_nobatch_float32_float32_float32/config`
set -e
if [ "$code" != "200" ]; then
    cat $TRIAL.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

matches=`grep -o "\"sequence_batching\":{" $TRIAL.out | wc -l`
if [ $matches -ne 1 ]; then
    cat $TRIAL.out
    echo -e "\n***\n*** Expected 1 sequence_batching, got $matches\n***"
    RET=1
fi

matches=`grep -o "\"max_sequence_idle_microseconds\":42" $TRIAL.out | wc -l`
if [ $matches -ne 1 ]; then
    cat $TRIAL.out
    echo -e "\n***\n*** Expected 1 sequence_batching:max_sequence_idle_microseconds, got $matches\n***"
    RET=1
fi

matches=`grep -o "\"oldest\":{" $TRIAL.out | wc -l`
if [ $matches -ne 1 ]; then
    cat $TRIAL.out
    echo -e "\n***\n*** Expected 1 sequence_batching:oldest, got $matches\n***"
    RET=1
fi

matches=`grep -o "\"max_queue_delay_microseconds\":987" $TRIAL.out | wc -l`
if [ $matches -ne 1 ]; then
    cat $TRIAL.out
    echo -e "\n***\n*** Expected 1 sequence_batching:oldest:max_queue_delay_microseconds, got $matches\n***"
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

# Test ensemble_scheduling::step::model_version
TRIAL=ensemble

rm -fr models && mkdir models
cp -r $DATADIR/qa_model_repository/savedmodel_nobatch_float32_float32_float32 models/.
mkdir -p models/simple_ensemble/1 && cp ensemble_config.pbtxt models/simple_ensemble/config.pbtxt

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
code=`curl -s -w %{http_code} -o ./$TRIAL.out localhost:8000/v2/models/simple_ensemble/config`
set -e
if [ "$code" != "200" ]; then
    cat $TRIAL.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

matches=`grep -o "\"model_version\":1" $TRIAL.out | wc -l`
if [ $matches -ne 1 ]; then
    cat $TRIAL.out
    echo -e "\n***\n*** Expected 1 ensemble_scheduling:step:model_version == 1, got $matches\n***"
    RET=1
fi

matches=`grep -o "\"model_version\":-1" $TRIAL.out | wc -l`
if [ $matches -ne 1 ]; then
    cat $TRIAL.out
    echo -e "\n***\n*** Expected 1 ensemble_scheduling:step:model_version == -1, got $matches\n***"
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

rm -fr models/simple_ensemble

# Test model_warmup::inputs::value::dims
TRIAL=warmup

rm -fr models && mkdir models
cp -r $DATADIR/qa_model_repository/savedmodel_nobatch_float32_float32_float32 models/.
(cd models/savedmodel_nobatch_float32_float32_float32 && \
     echo "model_warmup [{" >> config.pbtxt && \
     echo "    name : \"warmup 1\"" >> config.pbtxt && \
     echo "    batch_size: 1" >> config.pbtxt && \
     echo "    inputs [{" >> config.pbtxt && \
     echo "        key: \"INPUT0\"" >> config.pbtxt && \
     echo "        value: {" >> config.pbtxt && \
     echo "            data_type: TYPE_FP32" >> config.pbtxt && \
     echo "            dims: 16" >> config.pbtxt && \
     echo "            zero_data: true" >> config.pbtxt && \
     echo "        }" >> config.pbtxt && \
     echo "    }, {" >> config.pbtxt && \
     echo "        key: \"INPUT1\"" >> config.pbtxt && \
     echo "        value: {" >> config.pbtxt && \
     echo "            data_type: TYPE_FP32" >> config.pbtxt && \
     echo "            dims: 16" >> config.pbtxt && \
     echo "            random_data: true" >> config.pbtxt && \
     echo "        }" >> config.pbtxt && \
     echo "    }]" >> config.pbtxt && \
     echo "  }, {" >> config.pbtxt && \
     echo "    name : \"warmup 2\"" >> config.pbtxt && \
     echo "    batch_size: 1" >> config.pbtxt && \
     echo "    inputs [{" >> config.pbtxt && \
     echo "        key: \"INPUT0\"" >> config.pbtxt && \
     echo "        value: {" >> config.pbtxt && \
     echo "            data_type: TYPE_FP32" >> config.pbtxt && \
     echo "            dims: 16" >> config.pbtxt && \
     echo "            zero_data: true" >> config.pbtxt && \
     echo "        }" >> config.pbtxt && \
     echo "    }, {" >> config.pbtxt && \
     echo "        key: \"INPUT1\"" >> config.pbtxt && \
     echo "        value: {" >> config.pbtxt && \
     echo "            data_type: TYPE_FP32" >> config.pbtxt && \
     echo "            dims: 16" >> config.pbtxt && \
     echo "            random_data: true" >> config.pbtxt && \
     echo "        }" >> config.pbtxt && \
     echo "    }]" >> config.pbtxt && \
     echo "  }]" >> config.pbtxt)

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
code=`curl -s -w %{http_code} -o ./$TRIAL.out localhost:8000/v2/models/savedmodel_nobatch_float32_float32_float32/config`
set -e
if [ "$code" != "200" ]; then
    cat $TRIAL.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

matches=`grep -o "\"dims\":\[16\]" $TRIAL.out | wc -l`
if [ $matches -ne 8 ]; then
    cat $TRIAL.out
    echo -e "\n***\n*** Expected 8 model_warmup:inputs:dims, got $matches\n***"
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test Failed\n***"
fi

exit $RET
